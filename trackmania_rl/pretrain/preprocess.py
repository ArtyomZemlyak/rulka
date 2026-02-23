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
    from trackmania_rl.pretrain.preprocess import build_cache

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
from tqdm import tqdm

from trackmania_rl.pretrain.contract import (
    CACHE_TRAIN_ACTIONS_FILE,
    CACHE_VAL_ACTIONS_FILE,
)
from trackmania_rl.pretrain.datasets import (
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


def _manifest_entries(data: list | dict) -> list:
    """Normalize manifest.json payload to a list of entries.

    Supports both formats: a list of entries (legacy) or
    ``{"entries": [...], "actions": [...]}`` (with full action list).
    """
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, list):
        return data
    return []


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


# ---------------------------------------------------------------------------
# BC cache: manifest-based index and build
# ---------------------------------------------------------------------------


def _collect_bc_index(
    data_dir: Path,
    track_ids: list[str],
    n_stack: int,
) -> list[tuple[list[Path], int]]:
    """Build a flat list of (frame_paths, action_idx) for BC training.

    For each replay in *track_ids*, reads manifest.json. Entries with valid
    action_idx are used; temporal windows of *n_stack* consecutive frames
    get the action of the last frame in the window.
    """
    items: list[tuple[list[Path], int]] = []
    for tid in sorted(track_ids):
        track_dir = data_dir / tid
        if not track_dir.is_dir():
            continue
        for replay_dir in sorted(track_dir.iterdir()):
            if not replay_dir.is_dir():
                continue
            manifest_path = replay_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path, encoding="utf-8") as fh:
                    entries = _manifest_entries(json.load(fh))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Skip %s: %s", manifest_path, e)
                continue
            if len(entries) < max(1, n_stack):
                continue
            # Sort by step for deterministic order
            entries = sorted(entries, key=lambda e: (e.get("step", 0), e.get("time_ms", 0)))
            n_win = len(entries) - n_stack + 1 if n_stack > 1 else len(entries)
            for start in range(n_win):
                if n_stack > 1:
                    end = start + n_stack
                    window = entries[start:end]
                else:
                    window = [entries[start]]
                last = window[-1]
                action_idx = last.get("action_idx")
                if action_idx is None:
                    continue
                try:
                    action_idx = int(action_idx)
                except (TypeError, ValueError):
                    continue
                paths = []
                for ent in window:
                    fname = ent.get("file")
                    if not fname:
                        break
                    p = replay_dir / fname
                    if not p.exists():
                        break
                    paths.append(p)
                if len(paths) == len(window):
                    items.append((paths, action_idx))
    return items


def _get_action_for_level0_sample(
    frames: list[Path],
    start: int,
    n_stack: int,
    manifest_cache: dict[Path, list],
) -> int | None:
    """Return action_idx for the sample (frames, start) in Level 0 order.

    The last frame in the window (frames[start + n_stack - 1]) must have an
    entry in the replay's manifest.json with ``action_idx``. Uses
    *manifest_cache* (replay_dir -> list of entries sorted by step/time_ms).
    """
    last_path = frames[start + n_stack - 1]
    replay_dir = last_path.parent
    if replay_dir not in manifest_cache:
        try:
            manifest_path = replay_dir / "manifest.json"
            with open(manifest_path, encoding="utf-8") as fh:
                entries = _manifest_entries(json.load(fh))
            if not entries:
                return None
            entries = sorted(entries, key=lambda e: (e.get("step", 0), e.get("time_ms", 0)))
            manifest_cache[replay_dir] = entries
        except (OSError, json.JSONDecodeError):
            return None
    entries = manifest_cache[replay_dir]
    name = last_path.name
    for ent in entries:
        f = ent.get("file")
        if f is None:
            continue
        # Match by filename only (manifest may store "file" as name or relative path)
        if Path(f).name != name:
            continue
        a = ent.get("action_idx")
        if a is None:
            return None
        try:
            return int(a)
        except (TypeError, ValueError):
            return None
    return None


def _add_bc_actions_to_level0_cache(
    data_dir: Path,
    cache_dir: Path,
    image_size: int,
    n_stack: int,
    val_fraction: float,
    seed: int,
    n_actions: int,
) -> bool:
    """If *cache_dir* has a valid Level 0 cache (train.npy, val.npy), add only
    train_actions.npy and val_actions.npy in the same row order and update
    cache_meta.json with n_actions. Returns True if successful; False if cache
    is not reusable (then caller should run full build_bc_cache).
    """
    cache_dir = Path(cache_dir)
    if (cache_dir / CACHE_TRAIN_ACTIONS_FILE).exists():
        return False  # Already BC cache
    if not (cache_dir / CACHE_TRAIN_FILE).exists() or not (cache_dir / CACHE_META_FILE).exists():
        return False
    if not is_cache_valid(cache_dir, data_dir, image_size, n_stack, val_fraction, seed):
        return False

    with open(cache_dir / CACHE_META_FILE, encoding="utf-8") as fh:
        meta = json.load(fh)
    n_train = meta.get("n_train", 0)
    n_val = meta.get("n_val", 0)

    if val_fraction > 0 and not (cache_dir / CACHE_VAL_FILE).exists():
        return False

    train_ids, val_ids = split_track_ids(data_dir, val_fraction, seed) if val_fraction > 0 else (
        sorted(d.name for d in data_dir.iterdir() if d.is_dir()), [],
    )
    train_items = _collect_replay_index(data_dir, train_ids, n_stack)
    if len(train_items) != n_train:
        log.info(
            "Cannot reuse Level 0 cache: train sample count mismatch (cache=%d, index=%d)",
            n_train, len(train_items),
        )
        return False

    manifest_cache: dict[Path, list] = {}
    train_actions: list[int] = []
    for frames, start in tqdm(train_items, desc="BC actions from Level 0 (train)", unit="sample"):
        a = _get_action_for_level0_sample(frames, start, max(1, n_stack), manifest_cache)
        if a is None:
            log.info(
                "Cannot reuse Level 0 cache: sample has no action_idx in manifest (replay=%s, start=%d). "
                "Level 0 cache includes every frame; BC requires action_idx for each. Building full BC cache.",
                frames[0].parent, start,
            )
            return False
        train_actions.append(a)

    np.save(cache_dir / CACHE_TRAIN_ACTIONS_FILE, np.array(train_actions, dtype=np.int64))
    log.info("Added %s from Level 0 cache (reused train.npy)", CACHE_TRAIN_ACTIONS_FILE)

    if n_val > 0:
        val_items = _collect_replay_index(data_dir, val_ids, n_stack)
        if len(val_items) != n_val:
            log.info(
                "Cannot reuse Level 0 cache: val sample count mismatch (cache=%d, index=%d)",
                n_val, len(val_items),
            )
            (cache_dir / CACHE_TRAIN_ACTIONS_FILE).unlink(missing_ok=True)
            return False
        val_actions = []
        for frames, start in tqdm(val_items, desc="BC actions from Level 0 (val)", unit="sample"):
            a = _get_action_for_level0_sample(frames, start, max(1, n_stack), manifest_cache)
            if a is None:
                log.info(
                    "Cannot reuse Level 0 cache: missing action_idx for val sample (replay=%s)",
                    frames[0].parent,
                )
                (cache_dir / CACHE_TRAIN_ACTIONS_FILE).unlink(missing_ok=True)
                return False
            val_actions.append(a)
        np.save(cache_dir / CACHE_VAL_ACTIONS_FILE, np.array(val_actions, dtype=np.int64))
        log.info("Added %s from Level 0 cache (reused val.npy)", CACHE_VAL_ACTIONS_FILE)

    meta["n_actions"] = n_actions
    with open(cache_dir / CACHE_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info("BC actions added to Level 0 cache: %d train + %d val (reused frames)", len(train_actions), len(val_actions) if n_val else 0)
    return True


def is_bc_cache_valid(
    cache_dir: Path,
    data_dir: Path,
    image_size: int,
    n_stack: int,
    val_fraction: float,
    seed: int,
    n_actions: int,
) -> bool:
    """Return True iff the BC cache in *cache_dir* is valid for the given params."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / CACHE_META_FILE
    train_path = cache_dir / CACHE_TRAIN_FILE
    train_actions_path = cache_dir / CACHE_TRAIN_ACTIONS_FILE

    if not meta_path.exists() or not train_path.exists() or not train_actions_path.exists():
        log.info(
            "BC cache not found (required files missing in %s: need %s, %s, %s)",
            cache_dir, CACHE_TRAIN_FILE, CACHE_TRAIN_ACTIONS_FILE, CACHE_META_FILE,
        )
        return False

    if val_fraction > 0:
        if not (cache_dir / CACHE_VAL_FILE).exists() or not (cache_dir / CACHE_VAL_ACTIONS_FILE).exists():
            log.info("BC cache invalid: val_fraction=%g but val.npy or val_actions.npy missing", val_fraction)
            return False

    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.info("BC cache invalid: cannot read %s: %s", meta_path, exc)
        return False

    if Path(meta.get("source_data_dir", "")).resolve() != Path(data_dir).resolve():
        log.info("BC cache invalid: source_data_dir mismatch")
        return False

    for key, current_val in [
        ("image_size", image_size),
        ("n_stack", n_stack),
        ("val_fraction", val_fraction),
        ("seed", seed),
        ("n_actions", n_actions),
    ]:
        if meta.get(key) != current_val:
            log.info("BC cache invalid: %s mismatch (stored=%s, current=%s)", key, meta.get(key), current_val)
            return False

    stored_sig = meta.get("source_signature")
    if stored_sig is None:
        log.info("BC cache invalid: source_signature missing")
        return False
    if stored_sig != compute_source_signature(data_dir):
        log.info("BC cache invalid: source_signature changed")
        return False

    log.info(
        "BC cache valid: %d train + %d val (image_size=%d n_stack=%d n_actions=%d)",
        meta.get("n_train", 0), meta.get("n_val", 0), image_size, n_stack, n_actions,
    )
    return True


def build_bc_cache(
    data_dir: Path,
    cache_dir: Path,
    image_size: int = 64,
    n_stack: int = 1,
    val_fraction: float = 0.1,
    seed: int = 42,
    n_actions: int = 12,
    workers: int = 0,
) -> None:
    """Build BC preprocessed cache: train.npy, train_actions.npy, val.npy, val_actions.npy, cache_meta.json.

    If *cache_dir* already contains a valid Level 0 cache (train.npy, val.npy from
    ``build_cache``), only train_actions.npy and val_actions.npy are added (same
    row order as the existing frames), so you can reuse one cache dir for both
    Level 0 and BC. If reuse is not possible (e.g. missing action_idx in manifest),
    a full BC cache is built from scratch.
    """
    data_dir = Path(data_dir).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if (cache_dir / CACHE_TRAIN_FILE).exists() and not (cache_dir / CACHE_TRAIN_ACTIONS_FILE).exists():
        log.info("Found cache files from another task (Level 0) — reusing and adding BC actions")
    if _add_bc_actions_to_level0_cache(
        data_dir, cache_dir, image_size, n_stack, val_fraction, seed, n_actions,
    ):
        return

    log.info("Building BC cache: data_dir=%s -> cache_dir=%s", data_dir, cache_dir)
    signature = compute_source_signature(data_dir)

    if val_fraction > 0:
        train_ids, val_ids = split_track_ids(data_dir, val_fraction, seed)
    else:
        train_ids = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
        val_ids = []

    train_items = _collect_bc_index(data_dir, train_ids, n_stack)
    val_items = _collect_bc_index(data_dir, val_ids, n_stack) if val_ids else []

    if len(train_items) == 0:
        raise RuntimeError(
            f"No BC samples (frame + action_idx) in {data_dir}. "
            "Possible causes: (1) manifest.json has no entries with non-null 'action_idx' — replays must be "
            "captured with the capture path that records actions (policy rollout), not the TMI-script-only path "
            "which does not write action_idx; (2) manifest 'file' does not match actual frame filenames; "
            "(3) no manifest.json or invalid JSON. Check one replay: open <data_dir>/<track_id>/<replay_name>/manifest.json "
            "and ensure entries have 'file' and 'action_idx'."
        )

    # Write train.npy (frames)
    train_path = cache_dir / CACHE_TRAIN_FILE
    log.info("Writing %s (%d samples)...", train_path, len(train_items))
    n_stack_act = max(1, n_stack)
    shape = (len(train_items), n_stack_act, 1, image_size, image_size)
    arr = np.lib.format.open_memmap(str(train_path), mode="w+", dtype=np.float32, shape=shape)
    try:
        for i, (paths, _) in enumerate(tqdm(train_items, desc="BC cache train", unit="sample")):
            stack = [
                _load_one_frame(paths[j], image_size).numpy()
                for j in range(n_stack_act)
            ]
            arr[i] = np.stack(stack, axis=0).astype(np.float32)
    finally:
        arr.flush()
        del arr

    # Write train_actions.npy
    actions_path = cache_dir / CACHE_TRAIN_ACTIONS_FILE
    actions_arr = np.array([a for _, a in train_items], dtype=np.int64)
    np.save(actions_path, actions_arr)

    if val_items:
        val_path = cache_dir / CACHE_VAL_FILE
        log.info("Writing %s (%d samples)...", val_path, len(val_items))
        arr = np.lib.format.open_memmap(
            str(val_path), mode="w+", dtype=np.float32,
            shape=(len(val_items), n_stack_act, 1, image_size, image_size),
        )
        try:
            for i, (paths, _) in enumerate(tqdm(val_items, desc="BC cache val", unit="sample")):
                stack = [_load_one_frame(paths[j], image_size).numpy() for j in range(n_stack_act)]
                arr[i] = np.stack(stack, axis=0).astype(np.float32)
        finally:
            arr.flush()
            del arr
        np.save(cache_dir / CACHE_VAL_ACTIONS_FILE, np.array([a for _, a in val_items], dtype=np.int64))
    elif (cache_dir / CACHE_VAL_FILE).exists():
        (cache_dir / CACHE_VAL_FILE).unlink(missing_ok=True)
        (cache_dir / CACHE_VAL_ACTIONS_FILE).unlink(missing_ok=True)

    meta = {
        "source_data_dir": str(data_dir),
        "image_size": image_size,
        "n_stack": n_stack,
        "val_fraction": val_fraction,
        "seed": seed,
        "source_signature": signature,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_actions": n_actions,
    }
    with open(cache_dir / CACHE_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info("BC cache build complete: %d train + %d val", len(train_items), len(val_items))
