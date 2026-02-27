"""
Backfill manifest.json with action_idx and inputs per frame, and full "actions" list.

Scans data_dir (e.g. maps/img) for replay dirs that have manifest.json, finds the
corresponding .replay.gbx in replays_dir, extracts inputs via pygbx, and updates
each manifest entry with action_idx and inputs at that frame's time_ms; also
writes the full "actions" list (one action index per physics step) in the manifest.

Usage:
  python scripts/backfill_manifest_actions.py --data-dir maps/img --replays-dir maps/replays
  python scripts/backfill_manifest_actions.py --data-dir maps/img --replays-dir maps/replays --config config_files/rl/config_default.yaml --step-ms 10
  python scripts/backfill_manifest_actions.py --data-dir maps/img --replays-dir maps/replays --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

_script_root = Path(__file__).resolve().parents[1]
_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_root))
sys.path.insert(0, str(_scripts_dir))

from config_files.config_loader import get_config, load_config, set_config

log = logging.getLogger(__name__)


def _manifest_entries(data: list | dict) -> list:
    """Normalize manifest payload to list of entries (supports list or {entries, actions})."""
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, list):
        return data
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill manifest.json with action_idx and actions from .replay.gbx")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root of capture tree (e.g. maps/img)")
    parser.add_argument("--replays-dir", type=Path, required=True, help="Root of replays (e.g. maps/replays)")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML for inputs / action_dict_to_index")
    parser.add_argument("--step-ms", type=int, default=10, help="Physics step in ms (must match capture)")
    parser.add_argument("--dry-run", action="store_true", help="Only log what would be updated")
    parser.add_argument("--input-time-offset-ms", type=int, default=0, help="Input time offset used during capture")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

    config_path = args.config or (_script_root / "config_files" / "rl" / "config_default.yaml")
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)
    set_config(load_config(config_path))

    try:
        from capture_frames_from_replays import action_dict_to_index
    except ModuleNotFoundError:
        from importlib.util import spec_from_file_location, module_from_spec
        _spec = spec_from_file_location("capture_frames_from_replays", _scripts_dir / "capture_frames_from_replays.py")
        _mod = module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        action_dict_to_index = _mod.action_dict_to_index

    from capture_replays_tmnf import extract_inputs_from_replay_gbx

    data_dir = args.data_dir.resolve()
    replays_dir = args.replays_dir.resolve()
    step_ms = args.step_ms
    input_time_offset_ms = args.input_time_offset_ms

    if not data_dir.is_dir():
        log.error("Data dir is not a directory: %s", data_dir)
        sys.exit(1)
    if not replays_dir.is_dir():
        log.error("Replays dir is not a directory: %s", replays_dir)
        sys.exit(1)

    # Collect all (track_id, replay_dir) that have manifest.json
    tasks = []
    for track_dir in sorted(data_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        track_id = track_dir.name
        for replay_dir in sorted(track_dir.iterdir()):
            if not replay_dir.is_dir():
                continue
            if (replay_dir / "manifest.json").exists():
                tasks.append((track_id, replay_dir))

    updated = 0
    skipped_no_gbx = 0
    skipped_no_inputs = 0
    skipped_error = 0

    for track_id, replay_dir in tqdm(tasks, unit="replay", desc="Backfill manifests"):
        manifest_path = replay_dir / "manifest.json"

        # Replay dir name = stem of .replay.gbx (e.g. pos10_pacov_50020ms.replay)
        gbx_path = replays_dir / track_id / (replay_dir.name + ".gbx")
        if not gbx_path.exists():
            gbx_path = replays_dir / track_id / (replay_dir.name + ".replay.gbx")
        if not gbx_path.exists():
            skipped_no_gbx += 1
            continue

        result = extract_inputs_from_replay_gbx(
            gbx_path,
            step_ms,
            input_time_offset_ms=input_time_offset_ms,
        )
        if result is None:
            skipped_no_inputs += 1
            continue

        inputs_per_step, _, _ = result
        action_indices = [action_dict_to_index(inp) for inp in inputs_per_step]
        if not action_indices:
            skipped_no_inputs += 1
            continue

        try:
            with open(manifest_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError):
            skipped_error += 1
            continue

        entries = _manifest_entries(raw)
        if not entries:
            skipped_error += 1
            continue

        for ent in entries:
            t_ms = ent.get("time_ms")
            if t_ms is None:
                continue
            step_idx = min(max(0, int(round(t_ms / step_ms))), len(action_indices) - 1)
            ent["action_idx"] = action_indices[step_idx]
            ent["inputs"] = dict(inputs_per_step[step_idx])

        out = {"entries": entries, "actions": action_indices}
        if not args.dry_run:
            manifest_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        updated += 1

    tqdm.write(
        f"Done: {updated} updated, {skipped_no_gbx} no GBX, {skipped_no_inputs} no inputs, {skipped_error} errors"
    )


if __name__ == "__main__":
    main()
