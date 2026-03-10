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

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from trackmania_rl.pretrain.contract import (
    CACHE_TRAIN_ACTIONS_FILE,
    CACHE_TRAIN_FLOATS_FILE,
    CACHE_VAL_ACTIONS_FILE,
    CACHE_VAL_FLOATS_FILE,
)
from trackmania_rl.pretrain.datasets import (
    IMAGE_EXTS,
    _load_one_frame,
    split_track_ids,
    split_track_ids_from_list,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-name constants (shared with CachedPretrainDataset / DataModule)
# ---------------------------------------------------------------------------

CACHE_TRAIN_FILE = "train.npy"
CACHE_VAL_FILE = "val.npy"
CACHE_META_FILE = "cache_meta.json"
CACHE_SKIP_INDICES_FILE = "skip_indices.json"


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

def compute_source_signature(data_dir: Path, track_ids: list[str] | None = None) -> dict:
    """Scan *data_dir* and return a lightweight fingerprint of its contents.

    Only traverses directory metadata (no file reads), so this is fast even
    for large datasets. When *track_ids* is given, only those subdirs are scanned.

    Returns
    -------
    dict
        ``{"n_tracks": int, "n_replays": int, "n_frame_files": int}``
    """
    data_dir = Path(data_dir)
    n_tracks = 0
    n_replays = 0
    n_frame_files = 0

    if track_ids is not None:
        track_dirs = [data_dir / tid for tid in sorted(track_ids) if (data_dir / tid).is_dir()]
    else:
        track_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

    for track_dir in track_dirs:
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

# metadata.json in each replay dir (from capture_replays_tmnf) has "step_ms" (game step in ms).
# manifest.json can be {"entries": [...], "actions": [...]}; "actions" is the full action timeline
# (one action per game step: actions[i] = action at time i*step_ms). Use that for multi-offset
# so that offset 0 vs +10 ms use actual action timings, not "closest frame".


def _action_at_time_from_timeline(
    actions: list[int],
    step_ms: int,
    target_time_ms: float | int,
) -> int | None:
    """Return action at *target_time_ms* from the action timeline (game steps).

    actions[k] is the action at game time k * step_ms. So action at target_time_ms
    is actions[round(target_time_ms / step_ms)], clamped to valid index.
    Returns None if actions is empty or step_ms <= 0.
    """
    if not actions or step_ms <= 0:
        return None
    try:
        t = float(target_time_ms)
    except (TypeError, ValueError):
        return None
    step_idx = int(round(t / step_ms))
    step_idx = max(0, min(step_idx, len(actions) - 1))
    try:
        return int(actions[step_idx])
    except (TypeError, ValueError, IndexError):
        return None


def _load_replay_manifest_and_timeline(
    replay_dir: Path,
) -> tuple[list[dict], list[int] | None, int | None, list[dict] | None]:
    """Load manifest.json and metadata.json for a replay.

    Returns (entries, actions_or_none, step_ms_or_none, meta_or_none). entries are sorted by (step, time_ms).
    actions is the full action timeline from manifest["actions"] when present;
    step_ms from metadata.json. meta is manifest["meta"] when present (race state snapshots).
    """
    manifest_path = replay_dir / "manifest.json"
    meta_path = replay_dir / "metadata.json"
    if not manifest_path.exists():
        return [], None, None, None
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return [], None, None, None
    entries = _manifest_entries(data)
    if not entries:
        return [], None, None, None
    entries = sorted(entries, key=lambda e: (e.get("step", 0), e.get("time_ms", 0)))
    actions: list[int] | None = None
    if isinstance(data, dict) and "actions" in data:
        raw = data["actions"]
        if isinstance(raw, list) and len(raw) > 0:
            try:
                actions = [int(x) for x in raw]
            except (TypeError, ValueError):
                pass
    step_ms: int | None = None
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            step_ms = meta.get("step_ms")
            if step_ms is not None:
                step_ms = int(step_ms)
                if step_ms <= 0:
                    step_ms = None
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    meta: list[dict] | None = None
    if isinstance(data, dict) and "meta" in data and isinstance(data["meta"], list):
        meta = data["meta"]
    return entries, actions, step_ms, meta


def _action_at_time(entries: list[dict], target_time_ms: float | int) -> int | None:
    """Return action_idx from the entry whose time_ms is closest to *target_time_ms*.

    Fallback when the manifest has no "actions" timeline or metadata has no step_ms.
    *entries* must be sorted by (step, time_ms). Prefer using the action timeline
    (manifest["actions"] + metadata["step_ms"]) via _action_at_time_from_timeline
    so that offsets use actual action timings, not closest frame.
    """
    if not entries:
        return None
    best_entry: dict | None = None
    best_diff: float = float("inf")
    for e in entries:
        t = e.get("time_ms")
        a = e.get("action_idx")
        if t is None or a is None:
            continue
        try:
            t_float = float(t)
            int(a)
        except (TypeError, ValueError):
            continue
        diff = abs(t_float - target_time_ms)
        if diff < best_diff:
            best_diff = diff
            best_entry = e
    if best_entry is None:
        return None
    try:
        return int(best_entry["action_idx"])
    except (TypeError, ValueError):
        return None


def _collect_bc_index(
    data_dir: Path,
    track_ids: list[str],
    n_stack: int,
    bc_target: str = "current_tick",
    bc_time_offsets_ms: list[int] | None = None,
) -> list[tuple[list[Path], list[int]]]:
    """Build a flat list of (frame_paths, action_indices) for BC training.

    When *bc_time_offsets_ms* is None or [0], *bc_target* applies: current_tick =
    action at last frame (MDP π(a_t|s_t)), next_tick = action at next timestep.
    When *bc_time_offsets_ms* has multiple offsets (e.g. [-10, 0, 10, 100]), for
    each window the last frame time T is used and for each offset d the action at
    T+d ms is taken from the **action timeline** (manifest["actions"] and
    metadata["step_ms"]) when available, so labels use actual action timings, not
    closest frame. Fallback: closest entry by time_ms. Returns
    (paths, list of n_offsets action indices); n_offsets=1 when single head.
    """
    if bc_time_offsets_ms is None:
        bc_time_offsets_ms = [0]
    use_multi_offset = len(bc_time_offsets_ms) > 1 or (
        len(bc_time_offsets_ms) == 1 and bc_time_offsets_ms[0] != 0
    )
    use_next_tick = not use_multi_offset and bc_target == "next_tick"

    replay_dirs = []
    for tid in sorted(track_ids):
        track_dir = data_dir / tid
        if not track_dir.is_dir():
            continue
        for replay_dir in sorted(track_dir.iterdir()):
            if replay_dir.is_dir():
                replay_dirs.append(replay_dir)

    items: list[tuple[list[Path], list[int]]] = []
    for replay_dir in tqdm(replay_dirs, desc="BC index", unit="replay"):
        entries, actions_timeline, step_ms, _ = _load_replay_manifest_and_timeline(replay_dir)
        if not entries:
            continue
        if len(entries) < max(1, n_stack):
            continue
        if use_next_tick:
            n_win = max(0, len(entries) - n_stack)
        else:
            n_win = len(entries) - n_stack + 1 if n_stack > 1 else len(entries)
        for start in range(n_win):
            if n_stack > 1:
                end = start + n_stack
                window = entries[start:end]
            else:
                window = [entries[start]]
            if use_multi_offset:
                last_time = window[-1].get("time_ms")
                if last_time is None:
                    continue
                try:
                    T = float(last_time)
                except (TypeError, ValueError):
                    continue
                action_indices: list[int] = []
                use_timeline = actions_timeline is not None and step_ms is not None
                for d in bc_time_offsets_ms:
                    if use_timeline:
                        a = _action_at_time_from_timeline(actions_timeline, step_ms, T + d)
                    else:
                        a = _action_at_time(entries, T + d)
                    if a is None:
                        break
                    action_indices.append(a)
                if len(action_indices) != len(bc_time_offsets_ms):
                    continue
            else:
                if use_next_tick:
                    next_entry = entries[start + n_stack]
                    action_idx = next_entry.get("action_idx")
                else:
                    action_idx = window[-1].get("action_idx")
                if action_idx is None:
                    continue
                try:
                    action_indices = [int(action_idx)]
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
                items.append((paths, action_indices))
    return items


def _get_action_for_level0_sample(
    frames: list[Path],
    start: int,
    n_stack: int,
    manifest_cache: dict[Path, tuple[list[dict], list[int] | None, int | None, list[dict] | None]],
) -> int | None:
    """Return action_idx for the sample (frames, start) in Level 0 order.

    The last frame in the window (frames[start + n_stack - 1]) must have an
    entry in the replay's manifest.json with ``action_idx``. Uses
    *manifest_cache* (replay_dir -> (entries, actions_timeline, step_ms)).
    """
    last_path = frames[start + n_stack - 1]
    replay_dir = last_path.parent
    if replay_dir not in manifest_cache:
        manifest_cache[replay_dir] = _load_replay_manifest_and_timeline(replay_dir)  # (entries, actions, step_ms, meta)
    entries = manifest_cache[replay_dir][0]
    if not entries:
        return None
    name = last_path.name
    for ent in entries:
        f = ent.get("file")
        if f is None:
            continue
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


def _get_actions_for_level0_sample(
    frames: list[Path],
    start: int,
    n_stack: int,
    bc_time_offsets_ms: list[int],
    manifest_cache: dict[Path, tuple[list[dict], list[int] | None, int | None, list[dict] | None]],
) -> list[int] | None:
    """Return list of action_idx per offset for the sample (frames, start).

    Last frame of window is used to get time T from manifest; then for each
    offset d in bc_time_offsets_ms returns action at T+d ms from the **action
    timeline** (manifest["actions"] + metadata["step_ms"]) when available,
    else from closest entry by time_ms. Returns None if any offset has no valid action.
    """
    last_path = frames[start + n_stack - 1]
    replay_dir = last_path.parent
    if replay_dir not in manifest_cache:
        manifest_cache[replay_dir] = _load_replay_manifest_and_timeline(replay_dir)  # (entries, actions, step_ms, meta)
    entries, actions_timeline, step_ms, _ = manifest_cache[replay_dir]
    if not entries:
        return None
    name = last_path.name
    T_ent = None
    for ent in entries:
        f = ent.get("file")
        if f is None:
            continue
        if Path(f).name != name:
            continue
        t = ent.get("time_ms")
        if t is None:
            return None
        try:
            T_ent = float(t)
        except (TypeError, ValueError):
            return None
        break
    if T_ent is None:
        return None
    use_timeline = actions_timeline is not None and step_ms is not None
    out: list[int] = []
    for d in bc_time_offsets_ms:
        if use_timeline:
            a = _action_at_time_from_timeline(actions_timeline, step_ms, T_ent + d)
        else:
            a = _action_at_time(entries, T_ent + d)
        if a is None:
            return None
        out.append(a)
    return out


def _nearest_meta_by_time(meta_list: list[dict], target_time_ms: float) -> dict | None:
    """Binary search for meta entry with time_ms closest to target_time_ms."""
    if not meta_list:
        return None
    times = [m.get("time_ms") for m in meta_list]
    valid = [(i, t) for i, t in enumerate(times) if t is not None]
    if not valid:
        return None
    best_i, best_t = min(valid, key=lambda x: abs(float(x[1]) - target_time_ms))
    return meta_list[best_i]


def _get_float_config_signature(floats_config) -> dict:
    """Build a deterministic signature of float-related config for cache invalidation."""
    inputs_hash = hashlib.sha256(
        json.dumps(floats_config.inputs, sort_keys=True).encode()
    ).hexdigest()
    return {
        "n_zone_centers_in_inputs": floats_config.n_zone_centers_in_inputs,
        "one_every_n_zone_centers_in_inputs": floats_config.one_every_n_zone_centers_in_inputs,
        "n_prev_actions_in_inputs": floats_config.n_prev_actions_in_inputs,
        "n_contact_material_physics_behavior_types": floats_config.n_contact_material_physics_behavior_types,
        "action_forward_idx": floats_config.action_forward_idx,
        "float_input_dim": int(floats_config.float_input_dim),
        "inputs_hash": inputs_hash,
        # v2: prev_actions exclude current step (RL-aligned; was v1 leaky)
        "bc_prev_actions_version": 2,
    }


def _add_bc_actions_to_level0_cache(
    data_dir: Path,
    cache_dir: Path,
    image_size: int,
    n_stack: int,
    val_fraction: float,
    seed: int,
    n_actions: int,
    bc_time_offsets_ms: list[int],
) -> bool:
    """If *cache_dir* has a valid Level 0 cache (train.npy, val.npy), add
    train_actions.npy and val_actions.npy in the same row order and update
    cache_meta.json. When bc_time_offsets_ms is [0], saves shape (N,); else (N, n_offsets).
    Returns True if successful; False if cache is not reusable.
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

    manifest_cache: dict[Path, tuple[list[dict], list[int] | None, int | None]] = {}
    single_head = len(bc_time_offsets_ms) == 1 and bc_time_offsets_ms[0] == 0
    train_actions: list[int] | list[list[int]] = []
    for frames, start in tqdm(train_items, desc="BC actions from Level 0 (train)", unit="sample"):
        if single_head:
            a = _get_action_for_level0_sample(frames, start, max(1, n_stack), manifest_cache)
            if a is None:
                log.info(
                    "Cannot reuse Level 0 cache: sample has no action_idx in manifest (replay=%s, start=%d). "
                    "Building full BC cache.",
                    frames[0].parent, start,
                )
                return False
            train_actions.append(a)
        else:
            acts = _get_actions_for_level0_sample(
                frames, start, max(1, n_stack), bc_time_offsets_ms, manifest_cache
            )
            if acts is None:
                log.info(
                    "Cannot reuse Level 0 cache: missing action for some offset (replay=%s, start=%d). Building full BC cache.",
                    frames[0].parent, start,
                )
                return False
            train_actions.append(acts)

    if single_head:
        np.save(cache_dir / CACHE_TRAIN_ACTIONS_FILE, np.array(train_actions, dtype=np.int64))
    else:
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
        val_actions_list: list[int] | list[list[int]] = []
        for frames, start in tqdm(val_items, desc="BC actions from Level 0 (val)", unit="sample"):
            if single_head:
                a = _get_action_for_level0_sample(frames, start, max(1, n_stack), manifest_cache)
                if a is None:
                    log.info(
                        "Cannot reuse Level 0 cache: missing action_idx for val sample (replay=%s)",
                        frames[0].parent,
                    )
                    (cache_dir / CACHE_TRAIN_ACTIONS_FILE).unlink(missing_ok=True)
                    return False
                val_actions_list.append(a)
            else:
                acts = _get_actions_for_level0_sample(
                    frames, start, max(1, n_stack), bc_time_offsets_ms, manifest_cache
                )
                if acts is None:
                    log.info(
                        "Cannot reuse Level 0 cache: missing action for val sample (replay=%s)",
                        frames[0].parent,
                    )
                    (cache_dir / CACHE_TRAIN_ACTIONS_FILE).unlink(missing_ok=True)
                    return False
                val_actions_list.append(acts)
        if single_head:
            np.save(cache_dir / CACHE_VAL_ACTIONS_FILE, np.array(val_actions_list, dtype=np.int64))
        else:
            np.save(cache_dir / CACHE_VAL_ACTIONS_FILE, np.array(val_actions_list, dtype=np.int64))
        log.info("Added %s from Level 0 cache (reused val.npy)", CACHE_VAL_ACTIONS_FILE)

    meta["n_actions"] = n_actions
    meta["bc_time_offsets_ms"] = bc_time_offsets_ms
    with open(cache_dir / CACHE_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info("BC actions added to Level 0 cache: %d train + %d val (reused frames)", len(train_actions), len(val_actions_list) if n_val else 0)
    return True


def _bc_cache_data_dir_matches(meta: dict, data_dir: Path) -> bool:
    """True iff cache_meta.source_data_dir matches current data_dir (resolved paths)."""
    stored = meta.get("source_data_dir") or ""
    if not stored:
        return False
    try:
        stored_path = Path(stored).resolve()
        current_path = Path(data_dir).resolve()
        # Same path: use samefile if both exist (handles symlinks/case), else strict equality
        if stored_path.exists() and current_path.exists():
            return stored_path.samefile(current_path)
        return stored_path == current_path
    except (OSError, TypeError):
        return False


def is_bc_cache_valid(
    cache_dir: Path,
    data_dir: Path,
    image_size: int,
    n_stack: int,
    val_fraction: float,
    seed: int,
    n_actions: int,
    bc_target: str = "current_tick",
    bc_time_offsets_ms: list[int] | None = None,
    use_floats: bool = False,
    floats_config: Any = None,
    track_ids: list[str] | None = None,
) -> bool:
    """Return True iff the BC cache in *cache_dir* is valid for the given params.

    All of the following from cache_meta.json must match; otherwise cache is invalid and
    train_bc will call build_bc_cache() to rebuild:
      - source_data_dir (must match current data_dir)
      - track_ids (if present in meta or in args, must match)
      - image_size, n_stack, val_fraction, seed, n_actions
      - bc_target, bc_time_offsets_ms
      - source_signature (fingerprint of data_dir contents, or subset when track_ids set)
      - if use_floats and cache has_floats: float_config_signature
    """
    if bc_time_offsets_ms is None:
        bc_time_offsets_ms = [0]
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

    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.info("BC cache invalid: cannot read %s: %s", meta_path, exc)
        return False

    # Require val files only when cache was built with a val split (n_val > 0), e.g. multiple tracks.
    # Single-track runs have n_val=0 and no val.npy/val_actions.npy by design.
    n_val = meta.get("n_val", 0)
    if val_fraction > 0 and n_val > 0:
        if not (cache_dir / CACHE_VAL_FILE).exists() or not (cache_dir / CACHE_VAL_ACTIONS_FILE).exists():
            log.info("BC cache invalid: val_fraction=%g and n_val=%d but val.npy or val_actions.npy missing", val_fraction, n_val)
            return False

    if not _bc_cache_data_dir_matches(meta, data_dir):
        log.info(
            "BC cache invalid: source_data_dir mismatch (stored=%s, current=%s)",
            meta.get("source_data_dir"), Path(data_dir).resolve(),
        )
        return False

    stored_track_ids = meta.get("track_ids")
    if (stored_track_ids is None) != (track_ids is None):
        log.info(
            "BC cache invalid: track_ids mismatch (cache built for %s, current track_ids=%s)",
            "all tracks" if stored_track_ids is None else stored_track_ids,
            "all tracks" if track_ids is None else track_ids,
        )
        return False
    if track_ids is not None and sorted(stored_track_ids) != sorted(track_ids):
        log.info(
            "BC cache invalid: track_ids mismatch (stored=%s, current=%s)",
            stored_track_ids, track_ids,
        )
        return False

    for key, current_val in [
        ("image_size", image_size),
        ("n_stack", n_stack),
        ("val_fraction", val_fraction),
        ("seed", seed),
        ("n_actions", n_actions),
        ("bc_target", bc_target),
        ("bc_time_offsets_ms", bc_time_offsets_ms),
    ]:
        if key == "bc_target":
            stored = meta.get("bc_target", "current_tick")
        elif key == "bc_time_offsets_ms":
            stored = meta.get("bc_time_offsets_ms", [0])
        else:
            stored = meta.get(key)
        if stored != current_val:
            log.info("BC cache invalid: %s mismatch (stored=%s, current=%s)", key, stored, current_val)
            return False

    stored_sig = meta.get("source_signature")
    if stored_sig is None:
        log.info("BC cache invalid: source_signature missing")
        return False
    if stored_sig != compute_source_signature(data_dir, track_ids=track_ids):
        log.info("BC cache invalid: source_signature changed")
        return False

    if use_floats and meta.get("has_floats"):
        if floats_config is None:
            log.info("BC cache invalid: use_floats but floats_config is None")
            return False
        current_sig = _get_float_config_signature(floats_config)
        stored_float_sig = meta.get("float_config_signature")
        if stored_float_sig is None:
            log.info("BC cache invalid: has_floats but float_config_signature missing")
            return False
        if stored_float_sig != current_sig:
            log.info(
                "BC cache invalid: float_config_signature changed (stored=%s, current=%s)",
                stored_float_sig, current_sig,
            )
            return False

    log.info(
        "BC cache valid: %d train + %d val (image_size=%d n_stack=%d n_actions=%d)",
        meta.get("n_train", 0), meta.get("n_val", 0), image_size, n_stack, n_actions,
    )
    return True


def is_bc_cache_valid_except_float_signature(
    cache_dir: Path,
    data_dir: Path,
    image_size: int,
    n_stack: int,
    val_fraction: float,
    seed: int,
    n_actions: int,
    bc_target: str = "current_tick",
    bc_time_offsets_ms: list[int] | None = None,
    track_ids: list[str] | None = None,
) -> bool:
    """Like is_bc_cache_valid but skips float_config_signature check.

    Use to detect when train/val/actions are valid and only floats need rebuilding
    (e.g. after prev_actions formula change).
    """
    if bc_time_offsets_ms is None:
        bc_time_offsets_ms = [0]
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / CACHE_META_FILE
    train_path = cache_dir / CACHE_TRAIN_FILE
    train_actions_path = cache_dir / CACHE_TRAIN_ACTIONS_FILE

    if not meta_path.exists() or not train_path.exists() or not train_actions_path.exists():
        return False
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False
    n_val = meta.get("n_val", 0)
    if val_fraction > 0 and n_val > 0:
        if not (cache_dir / CACHE_VAL_FILE).exists() or not (cache_dir / CACHE_VAL_ACTIONS_FILE).exists():
            return False
    if not _bc_cache_data_dir_matches(meta, data_dir):
        return False
    stored_track_ids = meta.get("track_ids")
    if (stored_track_ids is None) != (track_ids is None):
        return False
    if track_ids is not None and sorted(stored_track_ids) != sorted(track_ids):
        return False
    for key, current_val in [
        ("image_size", image_size),
        ("n_stack", n_stack),
        ("val_fraction", val_fraction),
        ("seed", seed),
        ("n_actions", n_actions),
        ("bc_target", bc_target),
        ("bc_time_offsets_ms", bc_time_offsets_ms),
    ]:
        stored = meta.get("bc_target", "current_tick") if key == "bc_target" else (
            meta.get("bc_time_offsets_ms", [0]) if key == "bc_time_offsets_ms" else meta.get(key)
        )
        if stored != current_val:
            return False
    stored_sig = meta.get("source_signature")
    if stored_sig is None or stored_sig != compute_source_signature(data_dir, track_ids=track_ids):
        return False
    return True


def build_bc_cache_floats_only(
    cache_dir: Path,
    data_dir: Path,
    floats_config: Any,
) -> None:
    """Build only train_floats.npy / val_floats.npy for an existing BC cache.

    Use when the cache is valid (substantive params match) but has_floats=False
    (e.g. float build was skipped due to bad samples). Re-indexes, builds floats
    with skip logic, and removes bad samples from all cache files.
    """
    cache_dir = Path(cache_dir)
    data_dir = Path(data_dir).resolve()
    meta_path = cache_dir / CACHE_META_FILE
    if not meta_path.exists() or not (cache_dir / CACHE_TRAIN_FILE).exists() or not (cache_dir / CACHE_TRAIN_ACTIONS_FILE).exists():
        raise RuntimeError(
            f"Cannot build floats only: cache incomplete (need {CACHE_META_FILE}, {CACHE_TRAIN_FILE}, {CACHE_TRAIN_ACTIONS_FILE})"
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    val_fraction = meta.get("val_fraction", 0.1)
    seed = meta.get("seed", 42)
    n_stack = meta.get("n_stack", 1)
    bc_target = meta.get("bc_target", "current_tick")
    bc_time_offsets_ms = meta.get("bc_time_offsets_ms") or [0]
    actions_path = cache_dir / CACHE_TRAIN_ACTIONS_FILE

    if val_fraction > 0:
        train_ids, val_ids = split_track_ids(data_dir, val_fraction=val_fraction, seed=seed)
    else:
        train_ids = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
        val_ids = []
    train_items = _collect_bc_index(data_dir, train_ids, n_stack, bc_target=bc_target, bc_time_offsets_ms=bc_time_offsets_ms)
    val_items = _collect_bc_index(data_dir, val_ids, n_stack, bc_target=bc_target, bc_time_offsets_ms=bc_time_offsets_ms) if val_ids else []

    log.info("Building BC float inputs (floats only): %d train + %d val samples", len(train_items), len(val_items))
    _build_bc_floats_and_update_meta(
        train_items, val_items, cache_dir, meta, floats_config, actions_path, bc_time_offsets_ms,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _verify_bc_cache_consistency(actions_path, meta, bc_time_offsets_ms)


def _build_bc_floats_and_update_meta(
    train_items: list,
    val_items: list,
    cache_dir: Path,
    meta: dict,
    floats_config: Any,
    actions_path: Path,
    bc_time_offsets_ms: list[int],
) -> None:
    """Build train_floats.npy / val_floats.npy, update meta, optionally remove skip indices from cache."""
    manifest_cache: dict = {}
    has_meta = False
    float_dim = 0
    if train_items and floats_config is not None:
        try:
            entries, _, _, meta_list = _load_replay_manifest_and_timeline(train_items[0][0][-1].parent)
            has_meta = meta_list is not None and len(meta_list) > 0
            if has_meta:
                float_dim = int(floats_config.float_input_dim)
        except Exception as e:
            log.debug("Skipping float build: %s", e)

    if not (has_meta and float_dim > 0 and floats_config is not None):
        meta["has_floats"] = False
        return

    def _get_float_for_sample(paths: list[Path], _mc: dict) -> np.ndarray | None:
        replay_dir = paths[-1].parent
        if replay_dir not in _mc:
            _mc[replay_dir] = _load_replay_manifest_and_timeline(replay_dir)
        entries, actions_tl, step_ms_v, meta_list = _mc[replay_dir]
        if not meta_list or not entries:
            return None
        last_name = paths[-1].name
        t_ms = None
        for e in entries:
            if Path(e.get("file", "")).name == last_name:
                t_ms = e.get("time_ms")
                break
        if t_ms is None:
            return None
        try:
            t = float(t_ms)
        except (TypeError, ValueError):
            return None
        nearest = _nearest_meta_by_time(meta_list, t)
        if nearest is None:
            return None
        # Match RL: prev_actions are the n_prev actions *before* the current step (step_idx).
        # In RL: indices [len(actions)-n_prev, len(actions)-1] = steps t-n_prev..t-1 (exclude current).
        # Here: step_idx is "current" (frame time); we want steps step_idx-n_prev..step_idx-1.
        n_prev = floats_config.n_prev_actions_in_inputs
        prev_indices: list[int] = []
        if actions_tl is not None and step_ms_v is not None and step_ms_v > 0:
            step_idx = int(round(t / step_ms_v))
            for k in range(n_prev):
                idx = step_idx - n_prev + k
                prev_indices.append(floats_config.action_forward_idx if idx < 0 or idx >= len(actions_tl) else actions_tl[idx])
        else:
            prev_indices = [floats_config.action_forward_idx] * n_prev
        from trackmania_rl.float_inputs import build_float_vector, prev_actions_flat_from_indices, state_dict_from_meta
        state_dict = state_dict_from_meta(nearest, floats_config)
        prev_flat = prev_actions_flat_from_indices(prev_indices, floats_config.inputs, floats_config.action_forward_idx)
        return build_float_vector(state_dict, prev_flat, 0.0, floats_config)

    train_skip_indices: list[int] = []
    val_skip_indices: list[int] = []
    train_floats_list: list[np.ndarray] = []
    for i, (paths, _) in enumerate(tqdm(train_items, desc="BC floats train", unit="sample")):
        fv = _get_float_for_sample(paths, manifest_cache)
        if fv is None:
            log.info("Sample %d missing meta/float, adding to skip list", i)
            train_skip_indices.append(i)
            continue
        train_floats_list.append(fv)
    if not train_floats_list:
        meta["has_floats"] = False
        return
    np.save(cache_dir / CACHE_TRAIN_FLOATS_FILE, np.stack(train_floats_list, axis=0).astype(np.float32))
    val_floats_list: list[np.ndarray] = []
    if val_items:
        for j, (paths, _) in enumerate(tqdm(val_items, desc="BC floats val", unit="sample")):
            fv = _get_float_for_sample(paths, manifest_cache)
            if fv is None:
                log.info("Val sample %d missing meta/float, adding to skip list", j)
                val_skip_indices.append(j)
                continue
            val_floats_list.append(fv)
        if val_floats_list:
            np.save(cache_dir / CACHE_VAL_FLOATS_FILE, np.stack(val_floats_list, axis=0).astype(np.float32))
    meta["has_floats"] = True
    meta["float_input_dim"] = float_dim
    meta["float_config_signature"] = _get_float_config_signature(floats_config)

    if train_skip_indices or val_skip_indices:
        with open(cache_dir / CACHE_SKIP_INDICES_FILE, "w", encoding="utf-8") as fh:
            json.dump({"train": train_skip_indices, "val": val_skip_indices}, fh, indent=2)
        log.info("Skipped %d train + %d val samples (saved to %s)", len(train_skip_indices), len(val_skip_indices), CACHE_SKIP_INDICES_FILE)

        def _remove_rows(path: Path, skip_indices: list[int]) -> None:
            if not skip_indices or not path.exists():
                return
            data = np.load(str(path), mmap_mode="r")
            keep = np.ones(data.shape[0], dtype=bool)
            keep[np.asarray(skip_indices)] = False
            filtered = np.asarray(data[keep], dtype=data.dtype)
            del data  # release mmap so we can overwrite on Windows
            tmp_path = path.with_suffix(".npy.tmp")
            np.save(tmp_path, filtered)
            path.unlink()
            tmp_path.rename(path)
            log.info("Removed %d rows from %s", len(skip_indices), path.name)

        _remove_rows(cache_dir / CACHE_TRAIN_FILE, train_skip_indices)
        _remove_rows(cache_dir / CACHE_TRAIN_ACTIONS_FILE, train_skip_indices)
        _remove_rows(cache_dir / CACHE_VAL_FILE, val_skip_indices)
        _remove_rows(cache_dir / CACHE_VAL_ACTIONS_FILE, val_skip_indices)
        meta["n_train"] = len(train_items) - len(train_skip_indices)
        meta["n_val"] = len(val_items) - len(val_skip_indices)


def _verify_bc_cache_consistency(actions_path: Path, meta: dict, bc_time_offsets_ms: list[int]) -> None:
    train_actions_arr = np.load(str(actions_path), mmap_mode="r")
    n_train_final = meta["n_train"]
    n_offsets = len(bc_time_offsets_ms)
    if train_actions_arr.shape[0] != n_train_final:
        raise RuntimeError(f"BC cache inconsistent: train_actions.npy shape[0]={train_actions_arr.shape[0]} != n_train={n_train_final}")
    if n_offsets == 1:
        if train_actions_arr.ndim != 1:
            raise RuntimeError(f"BC cache inconsistent: single offset but train_actions.npy ndim={train_actions_arr.ndim}")
    else:
        if train_actions_arr.ndim != 2 or train_actions_arr.shape[1] != n_offsets:
            raise RuntimeError(f"BC cache inconsistent: multi-offset expected shape (N, {n_offsets}), got {train_actions_arr.shape}")
    log.info("BC cache verification passed: train_actions shape %s, n_offsets=%d", train_actions_arr.shape, n_offsets)


def build_bc_cache(
    data_dir: Path,
    cache_dir: Path,
    image_size: int = 64,
    n_stack: int = 1,
    val_fraction: float = 0.1,
    seed: int = 42,
    n_actions: int = 12,
    workers: int = 0,
    bc_target: str = "current_tick",
    bc_time_offsets_ms: list[int] | None = None,
    bc_offset_weights: list[float] | None = None,
    floats_config: Any = None,
    track_ids: list[str] | None = None,
) -> None:
    """Build BC preprocessed cache: train.npy, train_actions.npy, val.npy, val_actions.npy, cache_meta.json.

    If *cache_dir* already contains a valid Level 0 cache and bc_time_offsets_ms is [0],
    only train_actions.npy and val_actions.npy are added. With multiple offsets a full
    BC cache is built. Actions shape: (N,) when single offset, (N, n_offsets) otherwise.
    When *track_ids* is set, only those tracks are indexed and signature is computed for them.
    """
    if bc_time_offsets_ms is None:
        bc_time_offsets_ms = [0]
    data_dir = Path(data_dir).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if bc_time_offsets_ms == [0] and (cache_dir / CACHE_TRAIN_FILE).exists() and not (cache_dir / CACHE_TRAIN_ACTIONS_FILE).exists():
        log.info("Found cache files from another task (Level 0) — reusing and adding BC actions")
        if _add_bc_actions_to_level0_cache(
            data_dir, cache_dir, image_size, n_stack, val_fraction, seed, n_actions, bc_time_offsets_ms,
        ):
            return

    log.info("Building BC cache: data_dir=%s -> cache_dir=%s (bc_time_offsets_ms=%s)", data_dir, cache_dir, bc_time_offsets_ms)
    signature = compute_source_signature(data_dir, track_ids=track_ids)

    if track_ids is not None:
        if val_fraction > 0:
            train_ids, val_ids = split_track_ids_from_list(track_ids, val_fraction, seed)
        else:
            train_ids = list(track_ids)
            val_ids = []
    elif val_fraction > 0:
        train_ids, val_ids = split_track_ids(data_dir, val_fraction, seed)
    else:
        train_ids = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
        val_ids = []

    log.info("Indexing BC train samples (scanning manifests)...")
    train_items = _collect_bc_index(data_dir, train_ids, n_stack, bc_target, bc_time_offsets_ms)
    log.info("Indexed %d train samples", len(train_items))
    if val_ids:
        log.info("Indexing BC val samples...")
        val_items = _collect_bc_index(data_dir, val_ids, n_stack, bc_target, bc_time_offsets_ms)
        log.info("Indexed %d val samples", len(val_items))
    else:
        val_items = []

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
    if len(bc_time_offsets_ms) == 1:
        actions_arr = np.array([acts[0] for _, acts in train_items], dtype=np.int64)
    else:
        actions_arr = np.array([acts for _, acts in train_items], dtype=np.int64)
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
        np.save(cache_dir / CACHE_VAL_ACTIONS_FILE, np.array([acts[0] for _, acts in val_items], dtype=np.int64) if len(bc_time_offsets_ms) == 1 else np.array([acts for _, acts in val_items], dtype=np.int64))
    elif (cache_dir / CACHE_VAL_FILE).exists():
        (cache_dir / CACHE_VAL_FILE).unlink(missing_ok=True)
        (cache_dir / CACHE_VAL_ACTIONS_FILE).unlink(missing_ok=True)

    meta: dict = {
        "source_data_dir": str(data_dir),
        "image_size": image_size,
        "n_stack": n_stack,
        "val_fraction": val_fraction,
        "seed": seed,
        "bc_target": bc_target,
        "bc_time_offsets_ms": bc_time_offsets_ms,
        "source_signature": signature,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_actions": n_actions,
    }
    if track_ids is not None:
        meta["track_ids"] = track_ids
    if bc_offset_weights is not None:
        meta["bc_offset_weights"] = bc_offset_weights

    _build_bc_floats_and_update_meta(
        train_items, val_items, cache_dir, meta, floats_config, actions_path, bc_time_offsets_ms,
    )
    if not meta.get("has_floats"):
        meta["has_floats"] = False

    with open(cache_dir / CACHE_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info("BC cache build complete: %d train + %d val", meta["n_train"], meta["n_val"])
    _verify_bc_cache_consistency(actions_path, meta, bc_time_offsets_ms)