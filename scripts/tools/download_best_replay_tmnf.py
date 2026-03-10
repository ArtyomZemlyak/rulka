"""
Download the best-time replay for a track from TMNF Exchange.

Usage (from project root):
  set PYTHONPATH=scripts
  python scripts/tools/download_best_replay_tmnf.py 2233 -o maps/replays/2233/best_A01.Replay.gbx
  python scripts/tools/gbx_to_vcp.py maps/replays/2233/best_A01.Replay.gbx -o maps/A01_0.5m_cl_best.npy

A01-Race track ID on TMNF-X is 2233.
"""

import argparse
import sys
from pathlib import Path

# Allow importing replays_tmnf when run as script (replays_tmnf is in scripts/)
_scripts_dir = Path(__file__).resolve().parents[1]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from replays_tmnf.api import get_track_replays, download_replay


def main():
    ap = argparse.ArgumentParser(description="Download best replay for a track from TMNF Exchange")
    ap.add_argument("track_id", type=int, help="TMNF-X track ID (e.g. 2233 = A01-Race)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output path for .Replay.gbx file")
    args = ap.parse_args()

    replays = get_track_replays(args.track_id, limit=1)
    if not replays:
        print("No replays found for this track.", file=sys.stderr)
        sys.exit(1)
    r = replays[0]
    replay_id = r.get("ReplayId")
    if not replay_id:
        print("Replay entry has no ReplayId:", r, file=sys.stderr)
        sys.exit(1)
    print("Best replay: ReplayId =", replay_id, "| Time =", r.get("ReplayTime"), "ms |", r.get("User", {}).get("Name"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not download_replay(replay_id, args.output):
        print("Download failed.", file=sys.stderr)
        sys.exit(1)
    print("Saved to", args.output.resolve())


if __name__ == "__main__":
    main()
