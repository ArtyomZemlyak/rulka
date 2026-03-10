#!/usr/bin/env python
"""Fix BC cache and optionally clean source data for skip_indices.

Reads skip_indices.json from a BC cache dir. Can:

1. Fix .npy cache files — remove rows at skip indices from train.npy,
   train_actions.npy, val.npy, val_actions.npy, train_floats.npy, val_floats.npy.
   Updates cache_meta.json (n_train, n_val).

2. Clean source data — delete frame files for skipped samples, remove their
   manifest entries, delete empty replay dirs, delete empty track dirs.

Usage:
  python scripts/cleanup_skip_indices.py --cache-dir cache/v1
  python scripts/cleanup_skip_indices.py --cache-dir cache/v1 --fix-npy --yes
  python scripts/cleanup_skip_indices.py --cache-dir cache/v1 --fix-npy --clean-source --yes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trackmania_rl.pretrain.datasets import split_track_ids
from trackmania_rl.pretrain.preprocess import (
    CACHE_META_FILE,
    CACHE_SKIP_INDICES_FILE,
    CACHE_TRAIN_ACTIONS_FILE,
    CACHE_TRAIN_FILE,
    CACHE_TRAIN_FLOATS_FILE,
    CACHE_VAL_ACTIONS_FILE,
    CACHE_VAL_FILE,
    CACHE_VAL_FLOATS_FILE,
    _collect_bc_index,
    _manifest_entries,
    compute_source_signature,
)


def _remove_rows_from_npy(path: Path, skip_indices: list[int]) -> bool:
    """Remove rows at skip_indices. Uses temp file for Windows (mmap). Returns True if done."""
    if not skip_indices or not path.exists():
        return False
    path = Path(path).resolve()
    data = np.load(str(path), mmap_mode="r")
    keep = np.ones(data.shape[0], dtype=bool)
    keep[np.asarray(skip_indices)] = False
    filtered = np.asarray(data[keep], dtype=data.dtype)
    del data
    import tempfile
    import os
    fd, tmp_path_str = tempfile.mkstemp(suffix=".npy", dir=str(path.parent), prefix="cleanup_")
    try:
        os.close(fd)
        np.save(tmp_path_str, filtered)
        os.replace(tmp_path_str, str(path))  # atomic overwrite on Windows
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix BC cache and clean source data for skip_indices."
    )
    parser.add_argument("--cache-dir", type=Path, required=True, help="BC cache directory")
    parser.add_argument(
        "--fix-npy",
        action="store_true",
        help="Remove skip-index rows from .npy cache files",
    )
    parser.add_argument(
        "--clean-source",
        action="store_true",
        help="Delete bad frames, update manifests, remove empty replay/track dirs",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    skip_path = cache_dir / CACHE_SKIP_INDICES_FILE
    meta_path = cache_dir / CACHE_META_FILE

    if not skip_path.exists():
        print(f"ERROR: {skip_path} not found")
        return 1
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found")
        return 1

    with open(skip_path, encoding="utf-8") as f:
        skip_data = json.load(f)
    train_skip = sorted(skip_data.get("train", []))
    val_skip = sorted(skip_data.get("val", []))

    if not train_skip and not val_skip:
        print("No skip indices. Nothing to do.")
        return 0

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    data_dir = Path(meta["source_data_dir"])
    val_fraction = meta.get("val_fraction", 0.1)
    seed = meta.get("seed", 42)
    n_stack = meta.get("n_stack", 1)
    bc_target = meta.get("bc_target", "current_tick")
    bc_time_offsets_ms = meta.get("bc_time_offsets_ms") or [0]

    train_ids, val_ids = split_track_ids(data_dir, val_fraction=val_fraction, seed=seed)
    train_items = _collect_bc_index(
        data_dir, train_ids, n_stack, bc_target=bc_target, bc_time_offsets_ms=bc_time_offsets_ms
    )
    val_items = (
        _collect_bc_index(
            data_dir, val_ids, n_stack, bc_target=bc_target, bc_time_offsets_ms=bc_time_offsets_ms
        )
        if val_ids
        else []
    )

    # Map: replay_dir -> set of frame filenames to remove
    files_to_remove: dict[Path, set[str]] = {}
    for i in train_skip:
        if 0 <= i < len(train_items):
            paths, _ = train_items[i]
            replay_dir = paths[-1].parent
            for p in paths:
                files_to_remove.setdefault(replay_dir, set()).add(p.name)
    for j in val_skip:
        if 0 <= j < len(val_items):
            paths, _ = val_items[j]
            replay_dir = paths[-1].parent
            for p in paths:
                files_to_remove.setdefault(replay_dir, set()).add(p.name)

    n_replays = len(files_to_remove)
    n_samples = len(train_skip) + len(val_skip)
    print(f"Skip indices: {len(train_skip)} train + {len(val_skip)} val ({n_samples} total)")
    print(f"Affected replays: {n_replays}")

    if not args.fix_npy and not args.clean_source:
        print("\nUse --fix-npy to fix cache .npy files, --clean-source to clean maps/img")
        return 0

    if args.fix_npy:
        if not args.yes:
            confirm = input("\nFix .npy cache files (remove skip rows)? [y/N] ")
            if confirm.lower() != "y":
                print("Aborted.")
                return 0
        print("\nFixing .npy cache files...")
        # train_floats/val_floats are built without skip rows, so only fix train/val/actions
        if train_skip:
            for name in (CACHE_TRAIN_FILE, CACHE_TRAIN_ACTIONS_FILE):
                p = cache_dir / name
                if p.exists() and _remove_rows_from_npy(p, train_skip):
                    print(f"  Removed {len(train_skip)} rows from {name}")
            if (cache_dir / CACHE_TRAIN_FLOATS_FILE).exists():
                print(f"  {CACHE_TRAIN_FLOATS_FILE} already has correct rows (built without skips)")
        if val_skip:
            for name in (CACHE_VAL_FILE, CACHE_VAL_ACTIONS_FILE):
                p = cache_dir / name
                if p.exists() and _remove_rows_from_npy(p, val_skip):
                    print(f"  Removed {len(val_skip)} rows from {name}")
            if (cache_dir / CACHE_VAL_FLOATS_FILE).exists():
                print(f"  {CACHE_VAL_FLOATS_FILE} already has correct rows (built without skips)")
        meta["n_train"] = meta.get("n_train", 0) - len(train_skip)
        meta["n_val"] = meta.get("n_val", 0) - len(val_skip)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("  Updated cache_meta.json")

    if args.clean_source:
        if not data_dir.exists():
            print(f"WARN: data_dir {data_dir} not found, skipping source cleanup")
        else:
            if not args.yes:
                confirm = input(
                    f"\nClean source: delete frames, update manifests, remove empty dirs? [y/N] "
                )
                if confirm.lower() != "y":
                    print("Aborted.")
                    return 0
            print("\nCleaning source data...")
            empty_replays: list[Path] = []
            for replay_dir, to_remove in files_to_remove.items():
                if not replay_dir.exists():
                    continue
                manifest_path = replay_dir / "manifest.json"
                if not manifest_path.exists():
                    continue
                with open(manifest_path, encoding="utf-8") as f:
                    data = json.load(f)
                entries = _manifest_entries(data)
                if not entries:
                    continue
                actions = None
                if isinstance(data, dict) and "actions" in data and isinstance(data["actions"], list):
                    actions = data["actions"]
                indices_to_drop = {i for i, e in enumerate(entries) if e.get("file") in to_remove}
                if not indices_to_drop:
                    continue
                new_entries = [e for i, e in enumerate(entries) if i not in indices_to_drop]
                for fname in to_remove:
                    fp = replay_dir / fname
                    if fp.exists():
                        fp.unlink()
                if isinstance(data, dict) and "entries" in data:
                    data["entries"] = new_entries
                    if actions is not None and len(actions) == len(entries):
                        data["actions"] = [a for i, a in enumerate(actions) if i not in indices_to_drop]
                else:
                    data = new_entries
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                if not new_entries:
                    empty_replays.append(replay_dir)
            for rp in empty_replays:
                shutil.rmtree(rp)
                try:
                    rel = rp.relative_to(data_dir)
                except ValueError:
                    rel = rp
                print(f"  Removed empty replay {rel}")
            for track_dir in sorted(data_dir.iterdir()):
                if not track_dir.is_dir():
                    continue
                replays = [d for d in track_dir.iterdir() if d.is_dir()]
                if not replays:
                    shutil.rmtree(track_dir)
                    try:
                        rel = track_dir.relative_to(data_dir)
                    except ValueError:
                        rel = track_dir
                    print(f"  Removed empty track {rel}")
            # Update source_signature so is_bc_cache_valid won't trigger rebuild
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            meta["source_signature"] = compute_source_signature(data_dir)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print("  Updated cache_meta.json source_signature")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
