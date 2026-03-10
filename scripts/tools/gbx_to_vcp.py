"""
This script reads a .gbx file, and creates a list of Virtual CheckPoints (VCP) based on the best ghost found in that file.

The .gbx file may either be a .challenge.gbx, or a .replay.gbx.

The VCP file is saved in base_dir/maps/map.npy by default, or to --output if given.
The distance between virtual checkpoints is 0.5 m by default (--distance).
"""

import argparse
from pathlib import Path

from trackmania_rl.geometry import extract_cp_distance_interval
from trackmania_rl.map_loader import gbx_to_raw_pos_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gbx_path", type=Path, help="Path to .Challenge.gbx or .Replay.gbx")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output .npy path (default: base_dir/maps/map.npy)")
    parser.add_argument("--distance", type=float, default=0.5, help="Distance between checkpoints in meters (default: 0.5)")
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parents[2]

    raw_positions_list = gbx_to_raw_pos_list(args.gbx_path)
    out_path = (base_dir / args.output) if args.output is not None and not args.output.is_absolute() else args.output
    _ = extract_cp_distance_interval(raw_positions_list, args.distance, base_dir, out_path=out_path)


if __name__ == "__main__":
    main()
