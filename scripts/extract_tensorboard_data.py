"""
Script to extract scalar metrics from TensorBoard event files for analysis.

Usage:
    python scripts/extract_tensorboard_data.py --runs uni_5 uni_6 uni_6_2 --metrics "Race/eval_race_time_robust_hock" "Training/loss"
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: tensorboard package is required. Install with: pip install tensorboard")
    sys.exit(1)


# tensorboard_suffix_schedule creates run_2, run_3, ... at step thresholds.
# For "one experiment run", treat them as continuation chunks of the same run.
SUFFIXES = ["", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10"]


def discover_run_paths(base_dir: Path, run_name: str) -> List[Path]:
    """Find all TensorBoard log dirs for a run (run_name, run_name_2, run_name_3, ...)."""
    paths: List[Path] = []
    for suf in SUFFIXES:
        p = base_dir / (run_name + suf)
        if p.exists():
            paths.append(p)
        elif suf == "":
            break  # base run missing
        else:
            break
    return paths


def merge_scalar_points(run_paths: List[Path], metric_tag: str) -> List[Tuple[int, float]]:
    """Load scalar points from multiple run chunks and merge by step."""
    all_points: List[Tuple[int, float]] = []
    for run_path in run_paths:
        if not run_path.exists():
            continue
        ea = EventAccumulator(str(run_path))
        ea.Reload()
        available = ea.Tags().get("scalars", [])
        if metric_tag not in available:
            continue
        scalar_events = ea.Scalars(metric_tag)
        all_points.extend([(event.step, event.value) for event in scalar_events])
    all_points.sort(key=lambda x: x[0])
    return all_points


def extract_scalars(
    log_dir: Path, 
    run_name: str, 
    metric_tags: List[str]
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract scalar metrics from TensorBoard event files.
    
    Args:
        log_dir: Base directory containing TensorBoard logs
        run_name: Name of the run (e.g., 'uni_5')
        metric_tags: List of metric tags to extract (e.g., ['Race/eval_race_time_robust_hock'])
    
    Returns:
        Dictionary mapping metric tags to lists of (step, value) tuples
    """
    run_paths = discover_run_paths(log_dir, run_name)
    if not run_paths:
        run_paths = [log_dir / run_name]
    if not any(p.exists() for p in run_paths):
        print(f"Warning: Run directory {log_dir / run_name} not found. Skipping...")
        return {}
    
    results = {}
    # Discover scalar tags from all existing chunks (tags can differ across suffix dirs).
    available_tag_set = set()
    for p in run_paths:
        if not p.exists():
            continue
        ea0 = EventAccumulator(str(p))
        ea0.Reload()
        available_tag_set.update(ea0.Tags().get('scalars', []))
    available_tags = sorted(available_tag_set)
    
    for tag in metric_tags:
        if tag in available_tags:
            results[tag] = merge_scalar_points(run_paths, tag)
        else:
            # Try to find partial matches (e.g., for map-specific metrics)
            matching_tags = [t for t in available_tags if tag in t or tag.replace('_hock', '') in t]
            if matching_tags:
                print(f"Found matching tags for {tag}: {matching_tags}")
                # Use the first matching tag and merge across suffix chunks.
                results[tag] = merge_scalar_points(run_paths, matching_tags[0])
            else:
                print(f"Warning: Metric {tag} not found in {run_name} (available: {len(available_tags)} tags)")
                results[tag] = []
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract TensorBoard metrics for analysis")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of run names to analyze (e.g., uni_5 uni_6 uni_6_2)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "Race/eval_race_time_robust",
            "Race/eval_race_time_robust_hock",
            "Training/loss",
            "RL/avg_Q",
            "Performance/learner_percentage_training"
        ],
        help="List of metric tags to extract"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="tensorboard",
        help="Base directory containing TensorBoard logs"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional, prints to stdout if not specified)"
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List all available scalar tags in the first run and exit (useful for pretrain or other log schemas)"
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.logdir)
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        sys.exit(1)
    
    # List tags only (e.g. for pretrain TensorBoard with different tag names)
    if args.list_tags:
        run_name = args.runs[0]
        run_paths = discover_run_paths(log_dir, run_name)
        if not run_paths:
            print(f"Error: Run directory {log_dir / run_name} not found")
            sys.exit(1)
        # Use the first chunk only for listing tags (usually identical across chunks).
        run_path = run_paths[0]
        ea = EventAccumulator(str(run_path))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        print(f"Scalar tags in {run_name} ({len(tags)}):")
        for t in sorted(tags):
            print(f"  {t}")
        sys.exit(0)

    # Extract data for all runs
    all_data = {}
    for run_name in args.runs:
        print(f"\nExtracting data for {run_name}...")
        run_data = extract_scalars(log_dir, run_name, args.metrics)
        if run_data:
            all_data[run_name] = run_data
    
    # Print or save results
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TensorBoard Metrics Extraction Results")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    for run_name, run_data in all_data.items():
        output_lines.append(f"Run: {run_name}")
        output_lines.append("-" * 80)
        for metric, values in run_data.items():
            if values:
                output_lines.append(f"  {metric}:")
                output_lines.append(f"    Total data points: {len(values)}")
                if values:
                    output_lines.append(f"    Step range: {values[0][0]} - {values[-1][0]}")
                    output_lines.append(f"    Value range: {min(v[1] for v in values):.6f} - {max(v[1] for v in values):.6f}")
                    output_lines.append(f"    Latest value (step {values[-1][0]}): {values[-1][1]:.6f}")
            output_lines.append("")
    
    output_text = "\n".join(output_lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"\nResults saved to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
