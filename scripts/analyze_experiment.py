"""
Generic experiment analysis script.
Extracts and compares key metrics for any experiment run.
"""

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Dict, List, Tuple, Optional


def get_metric_data(run_path: Path, metric_tag: str) -> List[Tuple[int, float]]:
    """Extract metric data from TensorBoard event files."""
    if not run_path.exists():
        return []
    
    try:
        ea = EventAccumulator(str(run_path))
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        
        if metric_tag in tags:
            scalar_events = ea.Scalars(metric_tag)
            return [(event.step, event.value) for event in scalar_events]
        else:
            return []
    except Exception as e:
        print(f"Error loading {run_path}: {e}")
        return []


def analyze_run(run_name: str, base_dir: Path = Path("tensorboard")) -> Dict:
    """Analyze a single run and return metrics."""
    run_path = base_dir / run_name
    
    metrics = {
        'hock_best_time': 'alltime_min_ms_hock',
        'a01_best_time': 'alltime_min_ms_A01',
        'loss': 'Training/loss',
        'avg_q': 'RL/avg_Q_trained_A01',
        'training_pct': 'Performance/learner_percentage_training',
        'race_time_robust': 'Race/eval_race_time_robust_trained_A01',
        'race_time_robust_hock': 'Race/eval_race_time_robust_trained_hock',
    }
    
    results = {}
    for metric_name, metric_tag in metrics.items():
        data = get_metric_data(run_path, metric_tag)
        if data:
            results[metric_name] = {
                'data': data,
                'count': len(data),
                'step_range': (data[0][0], data[-1][0]),
                'value_range': (min(v for _, v in data), max(v for _, v in data)),
                'latest': data[-1],
                'first': data[0],
                'best': min(data, key=lambda x: x[1]) if 'time' in metric_name else max(data, key=lambda x: x[1])
            }
    
    return results


def compare_runs(run_names: List[str], base_dir: Path = Path("tensorboard")):
    """Compare multiple runs."""
    print("=" * 80)
    print("Experiment Comparison")
    print("=" * 80)
    print()
    
    all_results = {}
    for run_name in run_names:
        print(f"Analyzing {run_name}...")
        results = analyze_run(run_name, base_dir)
        all_results[run_name] = results
        print(f"  Extracted {len(results)} metrics")
    
    print()
    print("=" * 80)
    print("Key Metrics Comparison")
    print("=" * 80)
    print()
    
    # Compare best times
    for map_name, metric_key in [('Hock', 'hock_best_time'), ('A01', 'a01_best_time')]:
        print(f"{map_name} Map Best Times:")
        for run_name in run_names:
            if metric_key in all_results.get(run_name, {}):
                metric = all_results[run_name][metric_key]
                best_time = metric['best'][1] / 1000.0  # Convert ms to seconds
                best_step = metric['best'][0]
                print(f"  {run_name}: {best_time:.3f}s at step {best_step:,}")
        print()
    
    # Compare training loss
    print("Training Loss:")
    for run_name in run_names:
        if 'loss' in all_results.get(run_name, {}):
            metric = all_results[run_name]['loss']
            print(f"  {run_name}: {metric['latest'][1]:.2f} at step {metric['latest'][0]:,}")
    print()
    
    # Compare Q-values
    print("Average Q-values:")
    for run_name in run_names:
        if 'avg_q' in all_results.get(run_name, {}):
            metric = all_results[run_name]['avg_q']
            print(f"  {run_name}: {metric['latest'][1]:.4f} at step {metric['latest'][0]:,}")
    print()
    
    # Compare GPU utilization
    print("GPU Training Time Percentage:")
    for run_name in run_names:
        if 'training_pct' in all_results.get(run_name, {}):
            metric = all_results[run_name]['training_pct']
            print(f"  {run_name}: {metric['latest'][1]*100:.1f}%")
    print()
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare TensorBoard metrics across experiment runs."
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("tensorboard"),
        help="Base directory with run subdirs (default: tensorboard). "
             "E.g. C:\\path\\to\\rulka\\tensorboard",
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run names, e.g. uni_5 uni_7 uni_8 uni_9",
    )
    args = parser.parse_args()
    compare_runs(args.runs, base_dir=args.logdir)
