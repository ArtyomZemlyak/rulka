"""
Build comparison plots from TensorBoard data (by relative time; optionally by step).
One graph per metric; multiple runs as lines. Saves compressed JPG files.

Usage:
  python scripts/plot_experiment_comparison.py uni_12 uni_15 --prefix exp_exploration --output-dir docs/source/_static
  python scripts/plot_experiment_comparison.py uni_12 uni_13 uni_14 --prefix exp_temporal --by-step
"""

import argparse
import sys
from pathlib import Path

# Allow importing from same directory when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from generate_experiment_plots import _load_comparison_data
from experiment_plot_utils import plot_comparison


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot comparison graphs from TensorBoard data (one metric per graph, multiple runs as lines). Saves JPG."
    )
    p.add_argument("--logdir", type=Path, default=Path("tensorboard"))
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--step_interval", type=int, default=50000)
    p.add_argument(
        "--time-axis",
        choices=("auto", "wall_minutes", "cumul_training_hours"),
        default="auto",
        help="auto: try cumul_training_hours, fall back to wall_minutes if scalar missing",
    )
    p.add_argument("--interval-training-hours", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--prefix", type=str, default="")
    p.add_argument("--by-step", action="store_true", help="Also generate plots by training step")
    p.add_argument("--quality", type=int, default=85, help="JPG quality 1-95")
    p.add_argument("runs", nargs="+")
    args = p.parse_args()

    base_dir = args.logdir
    if not base_dir.is_absolute():
        base_dir = (Path.cwd() / base_dir).resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    data, axis_used = _load_comparison_data(
        args.runs,
        base_dir,
        interval_min=args.interval,
        step_interval=args.step_interval,
        time_axis_mode=args.time_axis,
        interval_training_hours=args.interval_training_hours,
    )
    print(f"X axis: {axis_used}")
    saved = plot_comparison(
        data,
        output_dir,
        prefix=args.prefix,
        by_step=args.by_step,
        jpg_quality=args.quality,
    )
    print(f"Saved {len(saved)} plot(s) to {output_dir}")
    for path in saved:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
