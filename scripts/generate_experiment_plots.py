"""
Generate comparison plots for all experiments documented in docs/source/experiments/.
Saves JPG files to docs/source/_static with predictable names for embedding in .rst.

By default each experiment tries **cumul_training_hours** on the X axis (matches learner
console "Training hours"; excludes downtime between restarts). If any run lacks that TensorBoard
scalar, falls back to **wall_minutes** (minutes from first TB event; includes calendar gaps).

Usage:
  python scripts/generate_experiment_plots.py [--logdir tensorboard] [--output-dir docs/source/_static]
  python scripts/generate_experiment_plots.py --experiments exploration temporal_mini_race_duration
  python scripts/generate_experiment_plots.py --time-axis wall_minutes   # force wall clock only
"""

import argparse
import sys
from pathlib import Path

# Resolve project root (parent of scripts/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_experiment_by_relative_time import CUMUL_TRAINING_HOURS_TAG, compute_comparison_data
from experiment_plot_utils import plot_comparison


def _load_comparison_data(
    runs: list,
    base_dir: Path,
    *,
    interval_min: int,
    step_interval: int,
    time_axis_mode: str,
    interval_training_hours: float,
):
    """Returns (data, axis_used) where axis_used is 'cumul_training_hours' or 'wall_minutes'."""
    if time_axis_mode == "wall_minutes":
        data = compute_comparison_data(
            runs,
            base_dir=base_dir,
            interval_min=interval_min,
            step_interval=step_interval,
            time_axis="wall_minutes",
        )
        return data, "wall_minutes"
    if time_axis_mode == "cumul_training_hours":
        data = compute_comparison_data(
            runs,
            base_dir=base_dir,
            interval_min=interval_min,
            step_interval=step_interval,
            time_axis="cumul_training_hours",
            interval_training_hours=interval_training_hours,
        )
        return data, "cumul_training_hours"
    # auto: prefer cumulative training hours, fall back to wall clock
    try:
        data = compute_comparison_data(
            runs,
            base_dir=base_dir,
            interval_min=interval_min,
            step_interval=step_interval,
            time_axis="cumul_training_hours",
            interval_training_hours=interval_training_hours,
        )
        return data, "cumul_training_hours"
    except ValueError as err:
        if CUMUL_TRAINING_HOURS_TAG in str(err):
            data = compute_comparison_data(
                runs,
                base_dir=base_dir,
                interval_min=interval_min,
                step_interval=step_interval,
                time_axis="wall_minutes",
            )
            return data, "wall_minutes"
        raise


# Experiment name -> {runs, prefix} for documented experiments
EXPERIMENTS = [
    {"name": "exploration", "runs": ["uni_12", "uni_15"], "prefix": "exp_exploration_uni12_uni15"},
    {"name": "temporal_mini_race_duration", "runs": ["uni_12", "uni_13", "uni_14"], "prefix": "exp_temporal_uni12_uni13_uni14"},
    {"name": "training_speed_uni5_uni7", "runs": ["uni_5", "uni_7"], "prefix": "exp_training_speed_uni5_uni7"},
    {"name": "training_speed_uni7_uni12", "runs": ["uni_7", "uni_12"], "prefix": "exp_training_speed_uni7_uni12"},
    {"name": "training_speed_uni5_uni6", "runs": ["uni_5", "uni_6"], "prefix": "exp_training_speed_uni5_uni6"},
    {"name": "training_speed_uni7_uni8_uni9", "runs": ["uni_7", "uni_8", "uni_9"], "prefix": "exp_training_speed_uni7_uni8_uni9"},
    {"name": "training_speed_uni7_uni10", "runs": ["uni_7", "uni_10"], "prefix": "exp_training_speed_uni7_uni10"},
    {"name": "training_speed_uni7_uni11", "runs": ["uni_7", "uni_11"], "prefix": "exp_training_speed_uni7_uni11"},
    {"name": "iqn_uni12_uni16", "runs": ["uni_12", "uni_16"], "prefix": "exp_iqn_uni12_uni16"},
    {"name": "iqn_uni16_uni17", "runs": ["uni_16", "uni_17"], "prefix": "exp_iqn_uni16_uni17"},
    {"name": "iqn_uni17_uni18", "runs": ["uni_17", "uni_18"], "prefix": "exp_iqn_uni17_uni18"},
    {"name": "iqn_uni17_uni19", "runs": ["uni_17", "uni_19"], "prefix": "exp_iqn_uni17_uni19"},
    {"name": "iqn_uni17_uni20", "runs": ["uni_17", "uni_20"], "prefix": "exp_iqn_uni17_uni20"},
    {"name": "extended_training", "runs": ["uni_20", "uni_20_long"], "prefix": "exp_extended_training_uni20_uni20long"},
    {"name": "extended_training_triple", "runs": ["A01_as20_long", "uni_20", "uni_20_long"], "prefix": "exp_extended_training_A01_uni20_uni20long"},
    {"name": "network_size_big_long", "runs": ["A01_as20_big_long", "A01_as20_long", "uni_20", "uni_20_long"], "prefix": "exp_network_size_big_long"},
    {"name": "pretrain_visual_backbone", "runs": ["A01_as20_long", "A01_as20_long_vis_pretrained"], "prefix": "exp_pretrain_visual_backbone"},
    {"name": "pretrain_bc", "runs": ["A01_as20_long", "A01_as20_long_vis_pretrained", "A01_as20_long_vis_bc_pretrained", "A01_as20_long_vis_bc_ah_pretrained"], "prefix": "exp_pretrain_bc"},
    {"name": "pretrain_bc_full_iqn", "runs": ["A01_as20_long", "A01_as20_long_vis_pretrained", "A01_as20_long_vis_bc_pretrained", "A01_as20_long_vis_bc_ah_pretrained", "A01_as20_long_full_iqn_bc"], "prefix": "exp_pretrain_bc_full_iqn"},
    {"name": "pretrain_bc_enc_freeze", "runs": ["A01_as20_long_vis_bc_ah_pretrained", "A01_as20_long_vis_bc_ah_pretrained_enc_freeze"], "prefix": "exp_pretrain_bc_enc_freeze"},
    {"name": "pretrain_bc_enc_ah_freeze", "runs": ["A01_as20_long_vis_bc_ah_pretrained", "A01_as20_long_vis_bc_ah_pretrained_enc_freeze", "A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze"], "prefix": "exp_pretrain_bc_enc_ah_freeze"},
    {"name": "pretrain_bc_enc_ah_freeze_resume", "runs": ["A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze", "A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze_resume"], "prefix": "exp_pretrain_bc_enc_ah_freeze_resume"},
    {"name": "reward_shaping", "runs": ["A01_as20_long", "A01_as20_long_engineer_rewards"], "prefix": "exp_reward_shaping"},
    {"name": "reward_shaping_bc_resume_v2_v3", "runs": ["A01_as20_long_full_iqn_bc_3_resume_engineer_rewards_v2", "A01_as20_long_full_iqn_bc_3_resume_engineer_rewards_v3"], "prefix": "exp_reward_shaping_bc_resume_v2_v3"},
    {"name": "reward_shaping_bc_resume_triple", "runs": ["A01_as20_long_full_iqn_bc_3_resume_engineer_rewards", "A01_as20_long_full_iqn_bc_3_resume_engineer_rewards_v2", "A01_as20_long_full_iqn_bc_3_resume_engineer_rewards_v3"], "prefix": "exp_reward_shaping_bc_resume_triple"},
    {"name": "iqn_no_image_head", "runs": ["A01_as20_long", "A01_as20_long_no_image"], "prefix": "exp_iqn_no_image_head"},
    {"name": "global_schedule_speed_v2", "runs": ["A01_as20_long_v2", "A01_as20_long_v2.1", "A01_as20_long_v2.4"], "prefix": "exp_global_schedule_speed_v2"},
    {
        "name": "linesight_A1_vs_A01_as20_long_v2",
        "runs": ["linesight_A1", "A01_as20_long_v2"],
        "prefix": "exp_linesight_A1_vs_A01_as20_long_v2",
    },
    {
        "name": "multi_offset_v2_vs_v31bc_pretrained",
        "runs": ["A01_as20_long_v2", "A01_as20_long_v3.1_pretrained_bc"],
        "prefix": "exp_multi_offset_v2_vs_v31bc_pretrained",
    },
]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate experiment comparison plots for documentation")
    p.add_argument("--logdir", type=Path, default=Path("tensorboard"))
    p.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "docs" / "source" / "_static")
    p.add_argument("--experiments", nargs="*", default=None, help="Experiment names to run (default: all)")
    p.add_argument("--interval", type=int, default=5, help="Wall-clock checkpoint spacing (minutes) when using wall_minutes axis")
    p.add_argument("--step_interval", type=int, default=50000)
    p.add_argument(
        "--time-axis",
        choices=("auto", "wall_minutes", "cumul_training_hours"),
        default="auto",
        help="auto: try cumul_training_hours, fall back to wall_minutes if scalar missing",
    )
    p.add_argument(
        "--interval-training-hours",
        type=float,
        default=0.5,
        help="X-axis step when using cumul_training_hours",
    )
    p.add_argument("--by-step", action="store_true", help="Also generate by-step plots")
    p.add_argument("--quality", type=int, default=85)
    args = p.parse_args()

    base_dir = args.logdir
    if not base_dir.is_absolute():
        base_dir = (_PROJECT_ROOT / base_dir).resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (_PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    names_filter = set(args.experiments) if args.experiments else None
    total_saved = 0
    for exp in EXPERIMENTS:
        if names_filter is not None and exp["name"] not in names_filter:
            continue
        runs = exp["runs"]
        prefix = exp["prefix"]
        print(f"Generating plots for {exp['name']} ({' '.join(runs)})...")
        try:
            data, axis_used = _load_comparison_data(
                runs,
                base_dir,
                interval_min=args.interval,
                step_interval=args.step_interval,
                time_axis_mode=args.time_axis,
                interval_training_hours=args.interval_training_hours,
            )
            saved = plot_comparison(
                data,
                output_dir,
                prefix=prefix,
                by_step=args.by_step,
                jpg_quality=args.quality,
            )
            total_saved += len(saved)
            print(f"  Saved {len(saved)} file(s) (X axis: {axis_used})")
        except Exception as e:
            print(f"  Skipped: {e}")
    print(f"Total: {total_saved} plot(s) in {output_dir}")


if __name__ == "__main__":
    main()
