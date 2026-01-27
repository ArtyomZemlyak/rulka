"""
Detailed analysis script for batch size experiment.
Extracts and compares key metrics between uni_5 (batch=2048) and uni_6/uni_6_2 (batch=8192).
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


def find_convergence_point(data: List[Tuple[int, float]], threshold: float, direction: str = 'below') -> Optional[int]:
    """Find step where metric converges (stays below/above threshold for significant period)."""
    if not data:
        return None
    
    if direction == 'below':
        # For race times - find when it stays below threshold
        consecutive_below = 0
        for step, value in data:
            if value < threshold:
                consecutive_below += 1
                if consecutive_below >= 10:  # 10 consecutive points
                    return step
            else:
                consecutive_below = 0
    else:
        # For Q-values - find when it stays above threshold
        consecutive_above = 0
        for step, value in data:
            if value > threshold:
                consecutive_above += 1
                if consecutive_above >= 10:
                    return step
            else:
                consecutive_above = 0
    
    return None


def analyze_runs():
    """Analyze and compare runs."""
    base_dir = Path("tensorboard")
    
    runs = {
        'uni_5': {'batch_size': 2048, 'path': base_dir / 'uni_5'},
        'uni_6': {'batch_size': 8192, 'path': base_dir / 'uni_6'},
        'uni_6_2': {'batch_size': 8192, 'path': base_dir / 'uni_6_2', 'is_continuation': True}
    }
    
    metrics = {
        'hock_best_time': 'alltime_min_ms_hock',
        'loss': 'Training/loss',
        'avg_q': 'RL/avg_Q_trained_A01',
        'training_pct': 'Performance/learner_percentage_training',
        'race_time_robust': 'Race/eval_race_time_robust_trained_A01'
    }
    
    results = {}
    
    print("=" * 80)
    print("Batch Size Experiment Analysis")
    print("=" * 80)
    print()
    
    for run_name, run_info in runs.items():
        print(f"Analyzing {run_name} (batch_size={run_info['batch_size']})...")
        run_path = run_info['path']
        run_results = {}
        
        for metric_name, metric_tag in metrics.items():
            data = get_metric_data(run_path, metric_tag)
            if data:
                run_results[metric_name] = {
                    'data': data,
                    'count': len(data),
                    'step_range': (data[0][0], data[-1][0]),
                    'value_range': (min(v for _, v in data), max(v for _, v in data)),
                    'latest': data[-1],
                    'first': data[0]
                }
        
        results[run_name] = run_results
        print(f"  Extracted {sum(1 for v in run_results.values() if v)} metrics")
    
    print()
    print("=" * 80)
    print("Key Comparisons")
    print("=" * 80)
    print()
    
    # Compare hock best times
    if 'hock_best_time' in results.get('uni_5', {}) and 'hock_best_time' in results.get('uni_6', {}):
        uni_5_hock = results['uni_5']['hock_best_time']
        uni_6_hock = results['uni_6']['hock_best_time']
        
        print("Hock Map Best Times:")
        print(f"  uni_5 (batch=2048): {uni_5_hock['latest'][1]:.3f}s at step {uni_5_hock['latest'][0]}")
        print(f"  uni_6 (batch=8192): {uni_6_hock['latest'][1]:.3f}s at step {uni_6_hock['latest'][0]}")
        
        if 'hock_best_time' in results.get('uni_6_2', {}):
            uni_6_2_hock = results['uni_6_2']['hock_best_time']
            print(f"  uni_6_2 (batch=8192, continuation): {uni_6_2_hock['latest'][1]:.3f}s at step {uni_6_2_hock['latest'][0]}")
        
        print()
    
    # Compare training loss
    if 'loss' in results.get('uni_5', {}) and 'loss' in results.get('uni_6', {}):
        uni_5_loss = results['uni_5']['loss']
        uni_6_loss = results['uni_6']['loss']
        
        print("Training Loss:")
        print(f"  uni_5 (batch=2048): {uni_5_loss['latest'][1]:.2f} at step {uni_5_loss['latest'][0]}")
        print(f"  uni_6 (batch=8192): {uni_6_loss['latest'][1]:.2f} at step {uni_6_loss['latest'][0]}")
        print(f"  Loss ratio (uni_6/uni_5): {uni_6_loss['latest'][1] / uni_5_loss['latest'][1]:.2f}x")
        print()
    
    # Compare Q-values
    if 'avg_q' in results.get('uni_5', {}) and 'avg_q' in results.get('uni_6', {}):
        uni_5_q = results['uni_5']['avg_q']
        uni_6_q = results['uni_6']['avg_q']
        
        print("Average Q-values:")
        print(f"  uni_5 (batch=2048): {uni_5_q['latest'][1]:.4f} at step {uni_5_q['latest'][0]}")
        print(f"  uni_6 (batch=8192): {uni_6_q['latest'][1]:.4f} at step {uni_6_q['latest'][0]}")
        print()
    
    # Compare training percentage (GPU utilization)
    if 'training_pct' in results.get('uni_5', {}) and 'training_pct' in results.get('uni_6', {}):
        uni_5_pct = results['uni_5']['training_pct']
        uni_6_pct = results['uni_6']['training_pct']
        
        print("GPU Training Time Percentage:")
        print(f"  uni_5 (batch=2048): {uni_5_pct['latest'][1]*100:.1f}%")
        print(f"  uni_6 (batch=8192): {uni_6_pct['latest'][1]*100:.1f}%")
        print(f"  Difference: {(uni_6_pct['latest'][1] - uni_5_pct['latest'][1])*100:+.1f}%")
        print()
    
    # Convergence analysis
    print("=" * 80)
    print("Convergence Analysis")
    print("=" * 80)
    print()
    
    if 'hock_best_time' in results.get('uni_5', {}) and 'hock_best_time' in results.get('uni_6', {}):
        uni_5_data = results['uni_5']['hock_best_time']['data']
        uni_6_data = results['uni_6']['hock_best_time']['data']
        
        # Find when each run reached 26 seconds (example threshold)
        threshold = 26.0
        uni_5_conv = find_convergence_point(uni_5_data, threshold, 'below')
        uni_6_conv = find_convergence_point(uni_6_data, threshold, 'below')
        
        if uni_5_conv and uni_6_conv:
            print(f"Steps to reach {threshold}s on hock map:")
            print(f"  uni_5 (batch=2048): {uni_5_conv:,} steps")
            print(f"  uni_6 (batch=8192): {uni_6_conv:,} steps")
            if uni_6_conv > uni_5_conv:
                print(f"  uni_6 took {((uni_6_conv / uni_5_conv) - 1) * 100:.1f}% more steps to converge")
        print()
    
    return results


if __name__ == "__main__":
    analyze_runs()
