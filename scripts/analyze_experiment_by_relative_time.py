"""
Compare TensorBoard metrics by RELATIVE TIME (minutes from run start).
Use this when runs had different durations — comparing "last value" is meaningless.
Samples at 5, 10, 15, 20, ... min up to the shortest run in each comparison.

Race times (preferred, more info and dynamics):
  - Uses Race/eval_race_time_* and Race/explo_race_time_* (per-race events).
  - All race events use ONE run-wide t0 (min wall_time across race tags) so rel_min is comparable.
  - At each checkpoint T min: best = min of all race times with rel_min <= T, mean, std, finish rate, first finish.
  - Time<->finished matching still uses step (same rollout = same step) for which races finished.

Scalar metrics (alltime_min_ms_*, loss, Q, etc.): last or best value at each checkpoint.

Usage:
  python scripts/analyze_experiment_by_relative_time.py uni_5 uni_7 [--logdir tensorboard] [--interval 5]
  python scripts/analyze_experiment_by_relative_time.py uni_12 uni_13 uni_14 --interval 5   # 3+ runs supported
"""

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Dict, List, Optional, Set, Tuple
import math

METRICS = {
    'hock_best_time_ms': 'alltime_min_ms_hock',   # best time so far (5-min scalar; use race events for richer view)
    'a01_best_time_ms': 'alltime_min_ms_A01',
    'loss': 'Training/loss',
    'avg_q': 'RL/avg_Q_trained_A01',
    'training_pct': 'Performance/learner_percentage_training',  # 0..1
}

# Per-race event tags: best/mean/std at checkpoint = min/mean/std of events up to that time
RACE_TIME_PREFIXES = ("Race/eval_race_time_", "Race/explo_race_time_")
# Exclude "race_time_finished" (subset) so we get every race
RACE_TIME_EXCLUDE = ("race_time_finished",)
RACE_FINISHED_PREFIXES = ("Race/eval_race_finished_", "Race/explo_race_finished_")


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


def _is_race_time_tag(tag: str) -> bool:
    if not any(tag.startswith(p) for p in RACE_TIME_PREFIXES):
        return False
    if any(ex in tag for ex in RACE_TIME_EXCLUDE):
        return False
    return True


def _race_time_to_finished_tag(tag: str) -> Optional[str]:
    """Race/eval_race_time_trained_hock -> Race/eval_race_finished_trained_hock."""
    for p in RACE_TIME_PREFIXES:
        if tag.startswith(p):
            suffix = tag[len(p):]  # e.g. "trained_hock"
            return ("Race/eval_race_finished_" if "eval" in p else "Race/explo_race_finished_") + suffix
    return None


def load_race_events(
    run_path: Path,
) -> Tuple[
    Dict[str, List[Tuple[float, int, float]]],
    Dict[str, List[Tuple[float, int, float]]],
]:
    """Load per-race time and finished events. All rel_min use ONE run-wide t0 so comparison is by relative time."""
    if not run_path.exists():
        return {}, {}
    try:
        ea = EventAccumulator(str(run_path))
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        time_tags = [t for t in tags if _is_race_time_tag(t)]
        finished_tags = [t for t in tags if any(t.startswith(p) for p in RACE_FINISHED_PREFIXES)]
        # One t0 for the whole run = min wall_time across all race-related events
        run_t0: Optional[float] = None
        for tag in time_tags + finished_tags:
            for e in ea.Scalars(tag):
                if run_t0 is None or e.wall_time < run_t0:
                    run_t0 = e.wall_time
        if run_t0 is None:
            return {}, {}
        time_events: Dict[str, List[Tuple[float, int, float]]] = {}
        for tag in time_tags:
            events = ea.Scalars(tag)
            if not events:
                continue
            time_events[tag] = [
                ((e.wall_time - run_t0) / 60.0, e.step, e.value)
                for e in events
            ]
        finished_events: Dict[str, List[Tuple[float, int, float]]] = {}
        for tag in finished_tags:
            events = ea.Scalars(tag)
            if not events:
                continue
            finished_events[tag] = [
                ((e.wall_time - run_t0) / 60.0, e.step, e.value)
                for e in events
            ]
        return time_events, finished_events
    except Exception as e:
        print(f"Error loading race events from {run_path}: {e}")
        return {}, {}


def race_stats_at_checkpoint(
    events: List[Tuple[float, int, float]],
    target_min: float,
    *,
    only_finished_steps: Optional[Set[int]] = None,
) -> Optional[Tuple[float, float, float, int]]:
    """(best_s, mean_s, std_s, n) for events with rel_min <= target_min. only_finished_steps: set of steps that finished (to filter)."""
    candidates = [(r, s, v) for r, s, v in events if r <= target_min]
    if not candidates:
        return None
    if only_finished_steps is not None:
        candidates = [(r, s, v) for r, s, v in candidates if s in only_finished_steps]
    if not candidates:
        return None
    vals = [v for (_, _, v) in candidates]
    n = len(vals)
    best = min(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n if n else 0
    std = math.sqrt(var) if n > 1 else 0.0
    return (best, mean, std, n)


def finish_stats_at_checkpoint(
    time_events: List[Tuple[float, int, float]],
    finished_events: List[Tuple[float, int, float]],
    target_min: float,
) -> Optional[Tuple[float, float, Optional[float]]]:
    """(finish_rate_0_1, n_finished, first_finish_min). Match by step (same rollout = same step)."""
    finished_steps = set(s for r, s, v in finished_events if r <= target_min and v >= 0.5)
    all_up_to = [(r, s, v) for r, s, v in time_events if r <= target_min]
    if not all_up_to:
        return None
    n_total = len(all_up_to)
    n_finished = sum(1 for (_, s, _) in all_up_to if s in finished_steps)
    rate = n_finished / n_total if n_total else 0.0
    first = min((r for (r, s, _) in all_up_to if s in finished_steps), default=None)
    return (rate, n_finished, first)


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
    # Load scalar metrics
    cache: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    race_time: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    race_finished: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    for name in run_names:
        p = base_dir / name
        cache[name] = load_run_metrics(p)
        race_time[name], race_finished[name] = load_race_events(p)
        print(f"Loaded {name} ({len(cache[name])} scalar metrics, {len(race_time[name])} race-time tags)")

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

    # ---- Race-time tables (best/mean/std from per-race events, finish rate, first finish) ----
    all_race_tags: Set[str] = set()
    for name in run_names:
        all_race_tags |= set(race_time.get(name, {}).keys())
    for tag in sorted(all_race_tags):
        runs_with_tag = [n for n in run_names if tag in race_time.get(n, {})]
        if not runs_with_tag:
            continue
        finished_tag = _race_time_to_finished_tag(tag)
        print(f"\n{tag} (from per-race events: best / mean / std / best_fin; finish rate; first finish min)")
        print("-" * 80)
        parts = ["min"]
        for n in runs_with_tag:
            parts.append(f"{n}_best")
            parts.append(f"{n}_mean")
            parts.append(f"{n}_std")
            parts.append(f"{n}_best_fin")
            parts.append(f"{n}_rate")
            parts.append(f"{n}_first")
        print("\t".join(parts))
        for t in checkpoints:
            row = [str(t)]
            for name in runs_with_tag:
                events = race_time[name][tag]
                fin_events = race_finished[name].get(finished_tag, []) if finished_tag else []
                finished_steps: Optional[Set[int]] = None
                if fin_events:
                    finished_steps = set(s for r, s, v in fin_events if r <= float(t) and v >= 0.5)
                st_all = race_stats_at_checkpoint(events, float(t), only_finished_steps=None)
                st_fin = race_stats_at_checkpoint(events, float(t), only_finished_steps=finished_steps) if finished_steps else None
                fin_stat = finish_stats_at_checkpoint(events, fin_events, float(t)) if fin_events else None
                if st_all is not None:
                    best_all, mean_all, std_all, _ = st_all
                    row.append(f"{best_all:.3f}s")
                    row.append(f"{mean_all:.2f}s")
                    row.append(f"{std_all:.2f}s")
                else:
                    row.extend(["-", "-", "-"])
                if st_fin is not None:
                    best_fin, _, _, _ = st_fin
                    row.append(f"{best_fin:.3f}s")
                else:
                    row.append("-")
                if fin_stat is not None:
                    rate, n_fin, first = fin_stat
                    row.append(f"{rate*100:.0f}%")
                    row.append(f"{first:.1f}" if first is not None else "-")
                else:
                    row.append("-")
                    row.append("-")
            print("\t".join(row))

    # ---- Scalar metrics (alltime_min_ms, loss, Q, etc.) ----
    print("\n" + "=" * 80)
    print("Scalar metrics (alltime_min_ms, loss, Q, training %)")
    print("=" * 80)
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
                        val = x[2]
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
