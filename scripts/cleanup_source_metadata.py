#!/usr/bin/env python
"""Validate and clean source data before BC cache build.

Checks each replay for metadata validity (manifest entries, meta snapshots for floats).
Removes bad samples (frame files + manifest entries), then removes empty replay dirs
and empty track dirs.

Run this BEFORE building BC cache to avoid skip_indices during cache build.

Usage:
  python scripts/cleanup_source_metadata.py --data-dir maps/img
  python scripts/cleanup_source_metadata.py --data-dir maps/img --apply --yes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from config_files.config_loader import load_config, set_config, get_config
from trackmania_rl.pretrain.preprocess import (
    _load_replay_manifest_and_timeline,
    _manifest_entries,
    _nearest_meta_by_time,
)
from trackmania_rl.float_inputs import state_dict_from_meta


def _is_entry_valid(
    entry: dict,
    replay_dir: Path,
    meta_list: list[dict] | None,
    actions_tl: list[int] | None,
    step_ms: int | None,
    floats_config,
) -> bool:
    """Return True if this entry has valid metadata for BC with floats."""
    fname = entry.get("file")
    if not fname:
        return False
    fp = replay_dir / fname
    if not fp.exists():
        return False
    t_ms = entry.get("time_ms")
    if t_ms is None:
        return False
    try:
        t = float(t_ms)
    except (TypeError, ValueError):
        return False
    if not meta_list:
        return False
    nearest = _nearest_meta_by_time(meta_list, t)
    if nearest is None:
        return False
    try:
        state_dict_from_meta(nearest, floats_config)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and clean source data metadata before BC cache build."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("maps/img"),
        help="Root of frame tree (default: maps/img)",
    )
    parser.add_argument(
        "--rl-config",
        type=Path,
        default=None,
        help="RL config for float validation (default: config_files/rl/config_default.yaml)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove bad samples and empty dirs",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    rl_config_path = args.rl_config or root / "config_files" / "rl" / "config_default.yaml"
    if not rl_config_path.exists():
        print(f"ERROR: RL config not found: {rl_config_path}")
        return 1
    set_config(load_config(rl_config_path))
    floats_config = get_config()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"ERROR: data_dir {data_dir} not found")
        return 1

    stats = {"replays": 0, "entries_checked": 0, "entries_bad": 0, "files_deleted": 0, "replays_removed": 0, "tracks_removed": 0}
    replays_with_bad: list[tuple[Path, list[int], list[str]]] = []  # (replay_dir, indices, filenames)

    replay_dirs = [
        replay_dir
        for track_dir in sorted(data_dir.iterdir())
        if track_dir.is_dir()
        for replay_dir in sorted(track_dir.iterdir())
        if replay_dir.is_dir()
    ]
    for replay_dir in tqdm(replay_dirs, desc="Scanning replays", unit="replay"):
        manifest_path = replay_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        stats["replays"] += 1
        entries, actions_tl, step_ms, meta_list = _load_replay_manifest_and_timeline(replay_dir)
        if not entries:
            continue
        bad_indices: list[int] = []
        bad_files: list[str] = []
        for i, e in enumerate(entries):
            stats["entries_checked"] += 1
            if not _is_entry_valid(e, replay_dir, meta_list, actions_tl, step_ms, floats_config):
                stats["entries_bad"] += 1
                bad_indices.append(i)
                bad_files.append(e.get("file", ""))
        if bad_indices:
            replays_with_bad.append((replay_dir, bad_indices, bad_files))

    print(f"Scanned {stats['replays']} replays, {stats['entries_checked']} entries")
    print(f"Bad entries: {stats['entries_bad']} (in {len(replays_with_bad)} replays)")

    if not replays_with_bad:
        print("No bad samples. Nothing to do.")
        return 0

    if not args.apply:
        print("\nRun with --apply to remove bad samples and empty dirs.")
        return 0

    if not args.yes:
        confirm = input(
            f"\nRemove {stats['entries_bad']} bad entries from {len(replays_with_bad)} replays? [y/N] "
        )
        if confirm.lower() != "y":
            print("Aborted.")
            return 0

    for replay_dir, bad_indices, bad_files in replays_with_bad:
        manifest_path = replay_dir / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        entries = _manifest_entries(data)
        actions = None
        if isinstance(data, dict) and "actions" in data and isinstance(data["actions"], list):
            actions = data["actions"]
        indices_to_drop = set(bad_indices)
        new_entries = [e for i, e in enumerate(entries) if i not in indices_to_drop]
        for fname in bad_files:
            if fname:
                fp = replay_dir / fname
                if fp.exists():
                    fp.unlink()
                    stats["files_deleted"] += 1
        if isinstance(data, dict) and "entries" in data:
            data["entries"] = new_entries
            if actions is not None and len(actions) == len(entries):
                data["actions"] = [a for i, a in enumerate(actions) if i not in indices_to_drop]
        else:
            data = new_entries
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if not new_entries:
            shutil.rmtree(replay_dir)
            stats["replays_removed"] += 1
            try:
                rel = replay_dir.relative_to(data_dir)
            except ValueError:
                rel = replay_dir
            print(f"  Removed empty replay {rel}")

    for track_dir in sorted(data_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        replays = [d for d in track_dir.iterdir() if d.is_dir()]
        if not replays:
            shutil.rmtree(track_dir)
            stats["tracks_removed"] += 1
            try:
                rel = track_dir.relative_to(data_dir)
            except ValueError:
                rel = track_dir
            print(f"  Removed empty track {rel}")

    print(f"\nDone: {stats['files_deleted']} files deleted, {stats['replays_removed']} replays removed, {stats['tracks_removed']} tracks removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
