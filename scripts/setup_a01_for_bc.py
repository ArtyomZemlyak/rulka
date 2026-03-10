"""
Setup A01-Race data for BC fine-tuning: download best replay, capture frames, verify.

Pipeline:
  1. Download best replay from TMNF Exchange (track 2233 = A01-Race) to maps/replays-A01/A01-Race/
  2. Run capture_replays_tmnf.py with --replays-dir maps/replays-A01 --output-dir maps/img-A01
  3. (Optional) Verify output: maps/img-A01/A01-Race/<replay>/ has manifest.json, metadata.json, frames

Usage (from project root):
  set PYTHONPATH=scripts
  python scripts/setup_a01_for_bc.py download
  python scripts/setup_a01_for_bc.py capture
  python scripts/setup_a01_for_bc.py verify
  python scripts/setup_a01_for_bc.py all   # download + capture (verify after manually or run verify)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLAYS_A01 = PROJECT_ROOT / "maps" / "replays-A01"
IMG_A01 = PROJECT_ROOT / "maps" / "img-A01"
TRACK_ID_TMNFX = 2233  # A01-Race on TMNF Exchange
TRACK_ID_FOLDER = "A01-Race"


def cmd_download() -> int:
    """Download best A01-Race replay to maps/replays-A01/A01-Race/."""
    out_file = REPLAYS_A01 / TRACK_ID_FOLDER / "best.Replay.gbx"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ret = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "tools" / "download_best_replay_tmnf.py"),
            str(TRACK_ID_TMNFX),
            "-o",
            str(out_file),
        ],
        cwd=PROJECT_ROOT,
    )
    if ret.returncode != 0:
        return ret.returncode
    print("Downloaded to", out_file)
    # Write track_ids.txt so capture can use --track-ids maps/replays-A01/track_ids.txt
    track_list = REPLAYS_A01 / "track_ids.txt"
    track_list.write_text(TRACK_ID_FOLDER + "\n", encoding="utf-8")
    print("Wrote", track_list)
    return 0


def cmd_capture() -> int:
    """Run capture_replays_tmnf.py for maps/replays-A01 -> maps/img-A01."""
    if not (REPLAYS_A01 / TRACK_ID_FOLDER).exists():
        print("Run 'download' first. No", REPLAYS_A01 / TRACK_ID_FOLDER, file=sys.stderr)
        return 1

    track_ids_file = REPLAYS_A01 / "track_ids.txt"
    if not track_ids_file.exists():
        track_ids_file.write_text(TRACK_ID_FOLDER + "\n", encoding="utf-8")

    args = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "capture_replays_tmnf.py"),
        "--replays-dir",
        str(REPLAYS_A01),
        "--output-dir",
        str(IMG_A01),
        "--track-ids",
        str(track_ids_file),
        "--workers",
        "1",
        "--width",
        "256",
        "--height",
        "256",
        "--running-speed",
        "16",
        "--fps",
        "100",
        "--max-replays-per-track",
        "1",
        "--vcp-dir",
        "maps/vcp",
    ]
    print("Running:", " ".join(args))
    return subprocess.run(args, cwd=PROJECT_ROOT).returncode


def cmd_verify() -> int:
    """Check that maps/img-A01/A01-Race/ has at least one replay dir with manifest + frames."""
    track_dir = IMG_A01 / TRACK_ID_FOLDER
    if not track_dir.is_dir():
        print("Missing", track_dir, "- run capture first.", file=sys.stderr)
        return 1

    replay_dirs = [d for d in track_dir.iterdir() if d.is_dir()]
    if not replay_dirs:
        print("No replay dirs under", track_dir, file=sys.stderr)
        return 1

    ok = 0
    for rdir in replay_dirs:
        manifest = rdir / "manifest.json"
        meta = rdir / "metadata.json"
        frames = list(rdir.glob("frame_*.jpeg")) or list(rdir.glob("frame_*.jpg"))
        if manifest.exists() and meta.exists() and len(frames) > 0:
            print("OK", rdir.name, "manifest, metadata,", len(frames), "frames")
            ok += 1
        else:
            print("FAIL", rdir.name, "manifest=%s metadata=%s frames=%d" % (manifest.exists(), meta.exists(), len(frames)))

    if ok == 0:
        print("No valid replay dirs.", file=sys.stderr)
        return 1
    print("Verified:", ok, "replay(s). data_dir for BC:", IMG_A01)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Setup A01-Race for BC fine-tuning")
    ap.add_argument("step", choices=["download", "capture", "verify", "all"], help="Pipeline step or 'all' (download+capture)")
    args = ap.parse_args()

    if args.step == "download":
        return cmd_download()
    if args.step == "capture":
        return cmd_capture()
    if args.step == "verify":
        return cmd_verify()
    if args.step == "all":
        if cmd_download() != 0:
            return 1
        return cmd_capture()
    return 0


if __name__ == "__main__":
    sys.exit(main())
