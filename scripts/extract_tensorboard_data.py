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
    run_path = log_dir / run_name
    if not run_path.exists():
        print(f"Warning: Run directory {run_path} not found. Skipping...")
        return {}
    
    # Find all event files in the run directory
    event_files = list(run_path.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"Warning: No event files found in {run_path}. Skipping...")
        return {}
    
    # Load event accumulator
    ea = EventAccumulator(str(run_path))
    ea.Reload()
    
    # Extract scalars
    results = {}
    available_tags = ea.Tags().get('scalars', [])
    
    for tag in metric_tags:
        if tag in available_tags:
            scalar_events = ea.Scalars(tag)
            results[tag] = [(event.step, event.value) for event in scalar_events]
        else:
            # Try to find partial matches (e.g., for map-specific metrics)
            matching_tags = [t for t in available_tags if tag in t or tag.replace('_hock', '') in t]
            if matching_tags:
                print(f"Found matching tags for {tag}: {matching_tags}")
                # Use the first matching tag
                scalar_events = ea.Scalars(matching_tags[0])
                results[tag] = [(event.step, event.value) for event in scalar_events]
            else:
                print(f"Warning: Metric {tag} not found in {run_name}")
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
        run_path = log_dir / run_name
        if not run_path.exists():
            print(f"Error: Run directory {run_path} not found")
            sys.exit(1)
        event_files = list(run_path.glob("events.out.tfevents.*"))
        if not event_files:
            print(f"No event files in {run_path}")
            sys.exit(1)
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
