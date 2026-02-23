"""
Generate comparison plots for all experiments documented in docs/source/experiments/.
Saves JPG files to docs/source/_static with predictable names for embedding in .rst.

Usage:
  python scripts/generate_experiment_plots.py [--logdir tensorboard] [--output-dir docs/source/_static]
  python scripts/generate_experiment_plots.py --experiments exploration temporal_mini_race_duration
"""

import argparse
import sys
from pathlib import Path

# Resolve project root (parent of scripts/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_experiment_by_relative_time import compute_comparison_data
from experiment_plot_utils import plot_comparison


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
    {"name": "pretrain_bc", "runs": ["A01_as20_long", "A01_as20_long_vis_pretrained", "A01_as20_long_vis_bc_pretrained"], "prefix": "exp_pretrain_bc"},
]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate experiment comparison plots for documentation")
    p.add_argument("--logdir", type=Path, default=Path("tensorboard"))
    p.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "docs" / "source" / "_static")
    p.add_argument("--experiments", nargs="*", default=None, help="Experiment names to run (default: all)")
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--step_interval", type=int, default=50000)
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
            data = compute_comparison_data(
                runs,
                base_dir=base_dir,
                interval_min=args.interval,
                step_interval=args.step_interval,
            )
            saved = plot_comparison(
                data,
                output_dir,
                prefix=prefix,
                by_step=args.by_step,
                jpg_quality=args.quality,
            )
            total_saved += len(saved)
            print(f"  Saved {len(saved)} file(s)")
        except Exception as e:
            print(f"  Skipped: {e}")
    print(f"Total: {total_saved} plot(s) in {output_dir}")


if __name__ == "__main__":
    main()
