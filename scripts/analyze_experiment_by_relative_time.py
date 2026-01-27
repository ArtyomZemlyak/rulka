"""
Compare TensorBoard metrics by RELATIVE TIME (minutes from run start).
Use this when runs had different durations — comparing "last value" is meaningless.
Samples at 5, 10, 15, 20, ... min up to the shortest run in each comparison.

Usage:
  python scripts/analyze_experiment_by_relative_time.py uni_5 uni_7 [--logdir tensorboard] [--interval 5]
"""

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Dict, List, Tuple, Optional


METRICS = {
    'hock_best_time_ms': 'alltime_min_ms_hock',   # lower is better
    'a01_best_time_ms': 'alltime_min_ms_A01',     # lower is better
    'loss': 'Training/loss',                       # lower is better
    'avg_q': 'RL/avg_Q_trained_A01',              # higher is better
    'training_pct': 'Performance/learner_percentage_training',  # 0..1
}


def load_run_metrics(
    run_path: Path,
) -> Dict[str, List[Tuple[float, int, float]]]:
    """Load run once, return {tag: [(relative_minutes, step, value), ...]} for all METRICS."""
    if not run_path.exists():
        return {}
    try:
        ea = EventAccumulator(str(run_path))
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        out = {}
        for tag in METRICS.values():
            if tag not in tags:
                continue
            events = ea.Scalars(tag)
            if not events:
                continue
            t0 = events[0].wall_time
            out[tag] = [
                ((e.wall_time - t0) / 60.0, e.step, e.value)
                for e in events
            ]
        return out
    except Exception as e:
        print(f"Error loading {run_path}: {e}")
        return {}


def value_at_minutes(
    data: List[Tuple[float, int, float]],
    target_min: float,
    kind: str,
) -> Optional[Tuple[float, int, float]]:
    """Value at checkpoint target_min (relative minutes).
    - 'time': best (min) value by that time — for race times.
    - 'last': last value at or before target_min — for loss, avg_q, training_pct.
    """
    if not data:
        return None
    candidates = [(r, s, v) for r, s, v in data if r <= target_min]
    if not candidates:
        return None
    if kind == 'time':
        return min(candidates, key=lambda x: x[2])  # best race time by T min
    # last value at or before target_min (sort by relative_min, take last)
    return max(candidates, key=lambda x: x[0])


def compare_by_relative_time(
    run_names: List[str],
    base_dir: Path = Path("tensorboard"),
    interval_min: int = 5,
) -> None:
    base_dir = Path(base_dir)
    # Load each run once
    cache: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    for name in run_names:
        p = base_dir / name
        cache[name] = load_run_metrics(p)
        print(f"Loaded {name} ({len(cache[name])} metrics)")

    durations = {}
    for name in run_names:
        d = 0.0
        for data in cache.get(name, {}).values():
            if data:
                d = max(d, max(r for r, _, _ in data))
        durations[name] = d
        print(f"{name}: duration ~{d:.0f} min (relative time)")

    common_max_min = min(durations.values()) if durations else 0
    checkpoints = list(range(interval_min, int(common_max_min) + 1, interval_min))
    if not checkpoints and common_max_min > 0:
        checkpoints = [int(common_max_min)]

    print(f"\nCheckpoints (min from run start): {checkpoints}")
    print("=" * 80)

    # For each metric, print table: rows = checkpoint, cols = runs
    for key, tag in METRICS.items():
        kind = 'time' if ('time' in key and 'ms' in key) else 'last'

        rows = []
        for t in checkpoints:
            row = {"min": t}
            for name in run_names:
                data = cache.get(name, {}).get(tag, [])
                v = value_at_minutes(data, float(t), kind)
                if v is not None:
                    rel, step, val = v
                    row[name] = (rel, step, val / (1000.0 if 'time' in key and 'ms' in key else 1) if 'time' in key else (val * 100 if key == 'training_pct' else val))
                    row[f"{name}_step"] = step
                else:
                    row[name] = None
            rows.append(row)

        # Print table
        print(f"\n{tag}")
        print("-" * 60)
        head = "min\t" + "\t".join(run_names)
        print(head)
        for r in rows:
            cells = [str(r["min"])]
            for name in run_names:
                x = r.get(name)
                if x is None:
                    cells.append("-")
                else:
                    if isinstance(x, tuple):
                        val = x[2]  # value
                    else:
                        val = x
                    if key == 'training_pct':
                        cells.append(f"{val:.1f}%")
                    elif 'time' in key:
                        cells.append(f"{val:.3f}s")
                    elif key == 'avg_q':
                        cells.append(f"{val:.4f}")
                    else:
                        cells.append(f"{val:.2f}")
            print("\t".join(cells))
    print()
    return durations, checkpoints


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compare metrics by relative time (min from run start)")
    p.add_argument("--logdir", type=Path, default=Path("tensorboard"))
    p.add_argument("--interval", type=int, default=5, help="Checkpoint interval in minutes")
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    compare_by_relative_time(args.runs, base_dir=args.logdir, interval_min=args.interval)
