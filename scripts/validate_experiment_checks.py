"""
Validate experiment analysis completeness.

What this checks:
1) Run suffix chunks are discovered and merged (run, run_2, run_3, ...).
2) Required scalar/race metrics exist in merged data.
3) By-step checkpoints are built with the requested interval (default: 1,000,000).
4) Comparison plots are actually generated.

Usage:
  .venv\\Scripts\\python.exe scripts\\validate_experiment_checks.py --logdir tensorboard --output-dir docs/source/_static
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from analyze_experiment_by_relative_time import compute_comparison_data, discover_run_paths
from experiment_plot_utils import plot_comparison


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def validate_group(
    group_name: str,
    runs: List[str],
    base_dir: Path,
    output_dir: Path,
    step_interval: int,
    interval_min: int,
    prefix: str,
) -> Tuple[int, int]:
    print(f"[CHECK] {group_name}: {', '.join(runs)}")

    # 1) Suffix discovery sanity
    multi_chunk_runs = 0
    for run in runs:
        paths = discover_run_paths(base_dir, run)
        _assert(paths, f"{run}: no tensorboard path found")
        if len(paths) > 1:
            multi_chunk_runs += 1
        print(f"  - {run}: {len(paths)} chunk(s) -> {', '.join(p.name for p in paths)}")
    _assert(multi_chunk_runs >= 1, f"{group_name}: expected at least one run with suffix chunks")

    # 2) Compute merged tables/series
    data = compute_comparison_data(
        runs,
        base_dir=base_dir,
        interval_min=interval_min,
        step_interval=step_interval,
    )
    _assert(data["step_checkpoints"], f"{group_name}: no by-step checkpoints")
    _assert(data["checkpoints"], f"{group_name}: no relative-time checkpoints")

    # 3) Required metrics in merged scalar cache
    required_scalar = {"alltime_min_ms_A01", "Training/loss", "RL/avg_Q_trained_A01", "Performance/learner_percentage_training"}
    for run in runs:
        tags = set(data["cache"].get(run, {}).keys())
        missing = required_scalar - tags
        _assert(not missing, f"{group_name}/{run}: missing scalar tags: {sorted(missing)}")

    # 4) Plot generation sanity
    saved = plot_comparison(
        data,
        output_dir=output_dir,
        prefix=prefix,
        by_step=True,
        jpg_quality=85,
    )
    _assert(saved, f"{group_name}: no plots saved")
    _assert(any("A01" in p.name for p in saved), f"{group_name}: expected A01-related plot names")
    print(f"  - plots saved: {len(saved)}")

    return len(saved), len(data["step_checkpoints"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate analysis completeness for experiment docs/checks")
    parser.add_argument("--logdir", type=Path, default=Path("tensorboard"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/source/_static"))
    parser.add_argument("--step-interval", type=int, default=1_000_000)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    base_dir = args.logdir if args.logdir.is_absolute() else (Path.cwd() / args.logdir).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _assert(base_dir.exists(), f"logdir does not exist: {base_dir}")

    total_plots = 0
    total_step_points = 0

    g1_plots, g1_steps = validate_group(
        group_name="global_schedule_speed_v2",
        runs=["A01_as20_long_v2", "A01_as20_long_v2.1", "A01_as20_long_v2.4"],
        base_dir=base_dir,
        output_dir=output_dir,
        step_interval=args.step_interval,
        interval_min=max(10, args.interval),
        prefix="exp_global_schedule_speed_v2",
    )
    total_plots += g1_plots
    total_step_points += g1_steps

    g2_plots, g2_steps = validate_group(
        group_name="multi_action_v3_series",
        runs=["A01_as20_long_v3", "A01_as20_long_v3.1", "A01_as20_long_v3.1_pretrained_bc"],
        base_dir=base_dir,
        output_dir=output_dir,
        step_interval=args.step_interval,
        interval_min=args.interval,
        prefix="exp_multi_action_v3_series",
    )
    total_plots += g2_plots
    total_step_points += g2_steps

    print("[OK] Validation complete")
    print(f"  total plots: {total_plots}")
    print(f"  total by-step checkpoints: {total_step_points}")


if __name__ == "__main__":
    main()

