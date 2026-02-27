"""
Capture frames (screenshots) from replays for visual pretraining.

Supports two input types:

1) **.inputs files** (TMI format, e.g. from our best runs or converted replays)
   - Parses .inputs to a list of action indices per step.
   - To actually capture frames you need the game running with TMInterface:
     either run the game, load the map, and play the .inputs replay while
     recording the screen (OBS, etc.), or use the training pipeline with a
     "replay" policy that feeds these actions and save rollout_results["frames"].

2) **.replay.gbx files** (e.g. from Nadeo / download_top_replays.py)
   - The game can play these via the built-in replay viewer.
   - We do not parse .gbx here. To get frames: copy .replay.gbx into the game's
     Replays folder, load the map in game, watch the replay, and record the
     screen (OBS, etc.). Then extract frames:
       ffmpeg -i video.mp4 -vf fps=10 -q:v 2 frames/%05d.png

This script can:
  - Parse .inputs files to action-index lists and save as .joblib (for use
    with a replay policy or to inspect timing).
  - Optionally run the game + TMI and capture frames (if --capture-with-tmi
    and map/config are provided); requires one map and one .inputs file.

Usage:
  python scripts/capture_frames_from_replays.py --inputs-dir ./best_runs --output-dir ./parsed_actions
  python scripts/capture_frames_from_replays.py --replays-dir ./replays_downloaded
  python scripts/capture_frames_from_replays.py --inputs-file run.inputs --map-path "A01-Race.Challenge.Gbx" --output-dir ./frames --capture-with-tmi
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Load config for tm_engine_step_per_action and inputs
import sys

_script_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_script_root))

from config_files.config_loader import get_config, load_config, set_config

_default_yaml = _script_root / "config_files" / "rl" / "config_default.yaml"
if _default_yaml.exists():
    set_config(load_config(_default_yaml))

KEY_TO_NAME = {"up": "accelerate", "down": "brake", "left": "left", "right": "right"}


def parse_inputs_file_tmi(path: Path, dt_s: float | None = None) -> list[dict]:
    """
    Parse a .inputs file in TMI format ("t1-t2 press key" lines).
    Returns a list of {accelerate, brake, left, right} per step at times 0, dt, 2*dt, ...
    """
    cfg = get_config()
    if dt_s is None:
        dt_s = cfg.tm_engine_step_per_action * 0.01

    events: list[tuple[float, float, str]] = []  # (t_start, t_end, key)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # "0.0-0.05 press up" or "1.2-2.3 press left"
        m = re.match(r"([\d.]+)-([\d.]+)\s+press\s+(\w+)", line)
        if m:
            t_start = float(m.group(1))
            t_end = float(m.group(2))
            key = m.group(3).lower()
            if key in KEY_TO_NAME:
                events.append((t_start, t_end, KEY_TO_NAME[key]))

    if not events:
        return []

    t_max = max(e[1] for e in events)
    n_steps = int(t_max / dt_s) + 1
    out: list[dict] = []
    for i in range(n_steps):
        t = i * dt_s
        state = {"accelerate": False, "brake": False, "left": False, "right": False}
        for t_start, t_end, key in events:
            if t_start <= t < t_end:
                state[key] = True
        out.append(state)
    return out


def action_dict_to_index(state: dict) -> int:
    """Map {accelerate, brake, left, right} to config.inputs index.

    Falls back to ``action_forward_idx`` if the combination is not found in
    ``cfg.inputs`` (e.g. left+right pressed simultaneously).  A warning is
    logged so silent mismatches are visible in diagnostics.
    """
    cfg = get_config()
    for idx, inp in enumerate(cfg.inputs):
        if all(inp.get(k) == state.get(k, False) for k in ("accelerate", "brake", "left", "right")):
            return idx
    import logging as _log
    _log.getLogger(__name__).warning(
        "Unmapped input combination %s -> falling back to action_forward_idx=%d (%s)",
        {k: state.get(k, False) for k in ("accelerate", "brake", "left", "right")},
        cfg.action_forward_idx,
        cfg.inputs[cfg.action_forward_idx] if cfg.action_forward_idx < len(cfg.inputs) else "?",
    )
    return cfg.action_forward_idx


def parse_inputs_to_actions(path: Path, dt_s: float | None = None) -> list[int]:
    """Parse .inputs file and return list of action indices per step."""
    states = parse_inputs_file_tmi(path, dt_s)
    return [action_dict_to_index(s) for s in states]


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse replays and optionally capture frames.")
    ap.add_argument("--inputs-dir", type=Path, help="Directory containing .inputs files")
    ap.add_argument("--inputs-file", type=Path, help="Single .inputs file")
    ap.add_argument("--replays-dir", type=Path, help="Directory with .replay.gbx (prints instructions only)")
    ap.add_argument("--output-dir", type=Path, default=Path("capture_frames_out"), help="Output for parsed actions or frames")
    ap.add_argument("--map-path", type=str, help="Map path for TMI capture (e.g. A01-Race.Challenge.Gbx)")
    ap.add_argument("--capture-with-tmi", action="store_true", help="Run game+TMI and capture frames (requires --inputs-file and --map-path)")
    ap.add_argument("--save-parsed", action="store_true", default=True, help="Save parsed action lists as .joblib")
    args = ap.parse_args()

    if args.replays_dir and args.replays_dir.exists():
        gbx = list(args.replays_dir.rglob("*.replay.gbx")) + list(args.replays_dir.rglob("*.Replay.gbx"))
        if gbx:
            print("For .replay.gbx files (e.g. from Nadeo):")
            print("  1) Copy .replay.gbx into the game's Replays folder.")
            print("  2) In game, load the map and watch the replay.")
            print("  3) Record the screen (OBS, etc.), then extract frames:")
            print('     ffmpeg -i video.mp4 -vf fps=10 -q:v 2 frames/%05d.png')
            print()

    if args.inputs_dir and args.inputs_dir.exists():
        inputs_files = list(args.inputs_dir.rglob("*.inputs"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for p in inputs_files:
            actions = parse_inputs_to_actions(p)
            rel = p.relative_to(args.inputs_dir)
            out_name = rel.with_suffix(".joblib")
            out_path = args.output_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if args.save_parsed:
                import joblib
                joblib.dump(actions, out_path)
            print(f"Parsed {p.name} -> {len(actions)} steps -> {out_path}")
    elif args.inputs_file and args.inputs_file.exists():
        actions = parse_inputs_to_actions(args.inputs_file)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_parsed:
            import joblib
            joblib.dump(actions, args.output_dir / (args.inputs_file.stem + "_actions.joblib"))
        print(f"Parsed {args.inputs_file.name} -> {len(actions)} steps")

        if args.capture_with_tmi and args.map_path:
            print("Capture with TMI: not implemented in this script.")
            print("To capture frames from .inputs: run the game with TMInterface, load the map,")
            print("then either record the screen while playing the .inputs replay in game,")
            print("or use the training pipeline with a replay policy that feeds the parsed actions")
            print("and save rollout_results['frames'] when processing a run.")
    elif not (args.replays_dir and args.replays_dir.exists()):
        print("Provide --inputs-dir, --inputs-file, or --replays-dir.")
        ap.print_help()


if __name__ == "__main__":
    main()
