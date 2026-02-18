"""
Preprocessing utilities for Level 0 visual pretraining.

Converts a raw replay frame tree (``maps/img/<track>/<replay>/*.jpg``) into a
cached NumPy format optimised for fast training I/O.  The cache lives in a
single directory and consists of three files:

  train.npy         — (N_train, n_stack, 1, H, W) float32, C-contiguous.
                      Readable via ``np.load(..., mmap_mode='r')`` for low-RAM
                      memory-mapped access, or fully into RAM with mmap_mode=None.
  val.npy           — same shape, for validation samples.
                      Absent when val_fraction == 0.
  cache_meta.json   — cache parameters and source fingerprint used for
                      validation (see ``is_cache_valid``).

Cache validation logic
----------------------
A cache is considered valid for the current training run when **all** of the
following conditions hold:

1. ``train.npy`` and ``cache_meta.json`` are present (and ``val.npy`` when
   ``val_fraction > 0``).
2. ``meta.source_data_dir`` equals the resolved absolute path of the current
   ``data_dir``.
3. ``meta.image_size``, ``meta.n_stack``, ``meta.val_fraction``, and
   ``meta.seed`` all match the current config values.
4. ``meta.source_signature`` matches the result of scanning ``data_dir`` right
   now (n_tracks, n_replays, n_frame_files).  This detects new/removed replays.

Usage
-----
Standalone preprocessing (manual invocation)::

    from pathlib import Path
    from trackmania_rl.pretrain_visual.preprocess import build_cache

    build_cache(
        data_dir=Path("maps/img"),
        cache_dir=Path("cache/pretrain_64"),
        image_size=64,
        n_stack=1,
        val_fraction=0.1,
        seed=42,
    )

Integrated into the training pipeline — see ``PretrainConfig.preprocess_cache_dir``
and ``train_pretrain`` in ``train.py``.  The pipeline checks the cache
automatically and rebuilds it when stale.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from trackmania_rl.pretrain_visual.datasets import (
    IMAGE_EXTS,
    _load_one_frame,
    split_track_ids,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-name constants (shared with CachedPretrainDataset / DataModule)
# ---------------------------------------------------------------------------

CACHE_TRAIN_FILE = "train.npy"
CACHE_VAL_FILE = "val.npy"
CACHE_META_FILE = "cache_meta.json"


# ---------------------------------------------------------------------------
# Source signature
# ---------------------------------------------------------------------------

def compute_source_signature(data_dir: Path) -> dict:
    """Scan *data_dir* and return a lightweight fingerprint of its contents.

    Only traverses directory metadata (no file reads), so this is fast even
    for large datasets.

    Returns
    -------
    dict
        ``{"n_tracks": int, "n_replays": int, "n_frame_files": int}``
    """
    data_dir = Path(data_dir)
    n_tracks = 0
    n_replays = 0
    n_frame_files = 0

    for track_dir in sorted(data_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        n_tracks += 1
        for replay_dir in sorted(track_dir.iterdir()):
            if not replay_dir.is_dir():
                continue
            n_replays += 1
            n_frame_files += sum(
                1 for p in replay_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )

    return {
        "n_tracks": n_tracks,
        "n_replays": n_replays,
        "n_frame_files": n_frame_files,
    }


# ---------------------------------------------------------------------------
# Cache validation
# ---------------------------------------------------------------------------

def is_cache_valid(
    cache_dir: Path,
    data_dir: Path,
    image_size: int,
    n_stack: int,
    val_fraction: float,
    seed: int,
) -> bool:
    """Return ``True`` iff the cache in *cache_dir* is compatible with the given params.

    All four conditions described in the module docstring must be satisfied.
    Reasons for invalidity are logged at INFO level.
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / CACHE_META_FILE
    train_path = cache_dir / CACHE_TRAIN_FILE

    # --- Condition 1: required files present ---
    if not meta_path.exists() or not train_path.exists():
        log.info(
            "Cache invalid: required files missing in %s  (need %s and %s)",
            cache_dir, CACHE_META_FILE, CACHE_TRAIN_FILE,
        )
        return False

    if val_fraction > 0 and not (cache_dir / CACHE_VAL_FILE).exists():
        log.info(
            "Cache invalid: val.npy missing in %s but val_fraction=%g",
            cache_dir, val_fraction,
        )
        return False

    # --- Load meta ---
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.info("Cache invalid: cannot read %s: %s", meta_path, exc)
        return False

    # --- Condition 2: source_data_dir ---
    stored_dir = Path(meta.get("source_data_dir", "")).resolve()
    current_dir = Path(data_dir).resolve()
    if stored_dir != current_dir:
        log.info(
            "Cache invalid: source_data_dir mismatch — stored=%s  current=%s",
            stored_dir, current_dir,
        )
        return False

    # --- Condition 3: numeric params ---
    for key, current_val in [
        ("image_size", image_size),
        ("n_stack", n_stack),
        ("val_fraction", val_fraction),
        ("seed", seed),
    ]:
        if meta.get(key) != current_val:
            log.info(
                "Cache invalid: %s mismatch — stored=%s  current=%s",
                key, meta.get(key), current_val,
            )
            return False

    # --- Condition 4: source_signature ---
    stored_sig = meta.get("source_signature")
    if stored_sig is None:
        log.info("Cache invalid: source_signature missing in %s", meta_path)
        return False

    current_sig = compute_source_signature(data_dir)
    if stored_sig != current_sig:
        log.info(
            "Cache invalid: source_signature changed — stored=%s  current=%s",
            stored_sig, current_sig,
        )
        return False

    log.info(
        "Cache valid: %d train + %d val samples  "
        "(image_size=%d  n_stack=%d  val_fraction=%g  seed=%d)",
        meta.get("n_train", "?"), meta.get("n_val", "?"),
        image_size, n_stack, val_fraction, seed,
    )
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_replay_index(
    data_dir: Path,
    track_ids: list[str],
    n_stack: int,
) -> list[tuple[list[Path], int]]:
    """Build a flat list of ``(replay_frames, start_idx)`` items for *track_ids*.

    Each item corresponds to one training sample (a temporal window of length
    *n_stack* within a single replay).  The replay boundary is never crossed.
    """
    items: list[tuple[list[Path], int]] = []
    for tid in sorted(track_ids):
        track_dir = data_dir / tid
        if not track_dir.is_dir():
            continue
        for replay_dir in sorted(track_dir.iterdir()):
            if not replay_dir.is_dir():
                continue
            frames = sorted(
                p for p in replay_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            if len(frames) < max(1, n_stack):
                continue
            n_items = len(frames) if n_stack <= 1 else len(frames) - n_stack + 1
            for start in range(n_items):
                items.append((frames, start))
    return items


def _load_sample(
    frames: list[Path],
    start: int,
    n_stack: int,
    image_size: int,
) -> np.ndarray:
    """Load one sample as a float32 ndarray of shape ``(n_stack, 1, H, W)``."""
    stack = [
        _load_one_frame(frames[start + i], image_size).numpy()
        for i in range(max(1, n_stack))
    ]
    return np.stack(stack, axis=0).astype(np.float32)


def _write_split(
    items: list[tuple[list[Path], int]],
    out_path: Path,
    n_stack: int,
    image_size: int,
    workers: int,
) -> None:
    """Write all *items* to *out_path* as a memory-mapped NumPy ``.npy`` file.

    Shape: ``(N, n_stack, 1, image_size, image_size)``, dtype float32.

    Parameters
    ----------
    items:
        Flat list of ``(replay_frames, start_idx)`` produced by
        ``_collect_replay_index``.
    out_path:
        Destination ``.npy`` file path.
    n_stack:
        Number of consecutive frames per sample.
    image_size:
        Target square resolution in pixels.
    workers:
        Number of threads for parallel frame loading.  ``0`` = single-threaded
        (no GIL contention; better for CPU-bound workloads on Windows).
    """
    n = len(items)
    shape = (n, max(1, n_stack), 1, image_size, image_size)

    # Open (create) a writable memory-mapped .npy file — no full allocation yet
    arr = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float32, shape=shape
    )

    try:
        if workers > 0:
            def _load_indexed(idx_item: tuple) -> tuple[int, np.ndarray]:
                idx, (frames, start) = idx_item
                return idx, _load_sample(frames, start, n_stack, image_size)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futs = {
                    executor.submit(_load_indexed, (i, item)): i
                    for i, item in enumerate(items)
                }
                done = 0
                for fut in as_completed(futs):
                    idx, sample = fut.result()
                    arr[idx] = sample
                    done += 1
                    if done % 5000 == 0 or done == n:
                        log.info("  Written %d / %d samples", done, n)
        else:
            for i, (frames, start) in enumerate(items):
                arr[i] = _load_sample(frames, start, n_stack, image_size)
                if (i + 1) % 5000 == 0 or (i + 1) == n:
                    log.info("  Written %d / %d samples", i + 1, n)
    finally:
        arr.flush()
        del arr  # release mmap handle


# ---------------------------------------------------------------------------
# Main entry point: build_cache
# ---------------------------------------------------------------------------

def build_cache(
    data_dir: Path,
    cache_dir: Path,
    image_size: int = 64,
    n_stack: int = 1,
    val_fraction: float = 0.1,
    seed: int = 42,
    workers: int = 0,
) -> None:
    """Build a preprocessed cache directory from a raw replay frame tree.

    Performs a track-level train/val split (same logic as ``split_track_ids``),
    preprocesses all frames (grayscale decode + bilinear resize), and writes
    ``train.npy``, ``val.npy`` (when ``val_fraction > 0``), and
    ``cache_meta.json`` to *cache_dir*.

    On subsequent calls with the same parameters, check ``is_cache_valid``
    first to avoid redundant work.

    Parameters
    ----------
    data_dir:
        Root of the frame tree (``maps/img/<track_id>/<replay_name>/``).
    cache_dir:
        Directory to write cache files into (created if absent).
    image_size:
        Target square resolution in pixels.  Must match the ``image_size``
        in ``PretrainConfig`` and the IQN ``w_downsized`` / ``h_downsized``.
    n_stack:
        Number of consecutive frames per sample (temporal stack depth).
    val_fraction:
        Fraction of track IDs reserved for validation.  ``0`` = no val split
        (``val.npy`` is not written).
    seed:
        RNG seed for the deterministic track-level split.
    workers:
        Number of threads for parallel frame loading.  ``0`` = single-threaded
        (safest on Windows; no GIL pressure from I/O-only threads).
        Setting ``workers > 0`` can speed up preprocessing when disk I/O is
        the bottleneck (e.g. SSD with many small JPEG files).
    """
    data_dir = Path(data_dir).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Building cache: data_dir=%s  →  cache_dir=%s", data_dir, cache_dir
    )
    log.info(
        "Params: image_size=%d  n_stack=%d  val_fraction=%g  seed=%d  workers=%d",
        image_size, n_stack, val_fraction, seed, workers,
    )

    # --- Source fingerprint (fast directory scan) ---
    log.info("Computing source signature...")
    signature = compute_source_signature(data_dir)
    log.info("Source: %s", signature)

    # --- Track-level split ---
    if val_fraction > 0:
        train_ids, val_ids = split_track_ids(data_dir, val_fraction, seed)
    else:
        train_ids = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
        val_ids = []
    log.info(
        "Split: %d train tracks, %d val tracks", len(train_ids), len(val_ids)
    )

    # --- Build flat sample indices (no pixel I/O yet) ---
    log.info("Indexing train samples...")
    train_items = _collect_replay_index(data_dir, train_ids, n_stack)
    log.info("Train samples: %d", len(train_items))

    val_items: list = []
    if val_ids:
        log.info("Indexing val samples...")
        val_items = _collect_replay_index(data_dir, val_ids, n_stack)
        log.info("Val samples: %d", len(val_items))

    if len(train_items) == 0:
        raise RuntimeError(
            f"No training samples found in {data_dir}.  "
            "Check that data_dir contains <track_id>/<replay_name>/<frames> "
            "with extensions in {'.jpg', '.jpeg', '.png', '.bmp', '.npy'}."
        )

    # --- Write train.npy ---
    train_path = cache_dir / CACHE_TRAIN_FILE
    log.info("Writing train.npy (%d samples)...", len(train_items))
    _write_split(train_items, train_path, n_stack, image_size, workers)
    log.info("train.npy done: %s  (%.1f MB)", train_path,
             train_path.stat().st_size / 1024 ** 2)

    # --- Write val.npy ---
    val_path = cache_dir / CACHE_VAL_FILE
    if val_items:
        log.info("Writing val.npy (%d samples)...", len(val_items))
        _write_split(val_items, val_path, n_stack, image_size, workers)
        log.info("val.npy done: %s  (%.1f MB)", val_path,
                 val_path.stat().st_size / 1024 ** 2)
    elif val_path.exists():
        val_path.unlink()
        log.info("Removed stale val.npy (val_fraction=0)")

    # --- Write cache_meta.json ---
    meta: dict = {
        "source_data_dir": str(data_dir),
        "image_size": image_size,
        "n_stack": n_stack,
        "val_fraction": val_fraction,
        "seed": seed,
        "source_signature": signature,
        "n_train": len(train_items),
        "n_val": len(val_items),
    }
    meta_path = cache_dir / CACHE_META_FILE
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info("cache_meta.json written: %s", meta_path)

    log.info(
        "Cache build complete.  Total: %d train + %d val samples.",
        len(train_items), len(val_items),
    )
