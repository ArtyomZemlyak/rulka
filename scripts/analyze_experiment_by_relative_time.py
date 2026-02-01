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

Also prints BY STEP: same tables keyed by training step (e.g. 50k, 100k, 150k) so you can compare
runs at equal numbers of gradient updates / transitions, not only at equal wall-clock time.

Usage:
  python scripts/analyze_experiment_by_relative_time.py uni_5 uni_7 [--logdir tensorboard] [--interval 5] [--step_interval 50000]
  python scripts/analyze_experiment_by_relative_time.py uni_12 uni_13 uni_14 --interval 5   # 3+ runs supported
"""

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, List, Optional, Set, Tuple
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
    tags_to_load: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[float, int, float]]]:
    """Load run once, return {tag: [(relative_minutes, step, value), ...]} for requested tags.
    If tags_to_load is None, loads METRICS.values(). Otherwise loads only tags_to_load (any TensorBoard scalar tag).
    """
    if not run_path.exists():
        return {}
    requested = tags_to_load if tags_to_load is not None else list(METRICS.values())
    try:
        ea = EventAccumulator(str(run_path))
        ea.Reload()
        available = ea.Tags().get('scalars', [])
        out = {}
        for tag in requested:
            if tag not in available:
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


def get_available_scalar_tags(run_path: Path) -> List[str]:
    """Return all scalar tag names in the run, excluding race-time and race-finished (handled separately)."""
    if not run_path.exists():
        return []
    try:
        ea = EventAccumulator(str(run_path))
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        out = []
        for tag in tags:
            if _is_race_time_tag(tag):
                continue
            if any(tag.startswith(p) for p in RACE_FINISHED_PREFIXES):
                continue
            out.append(tag)
        return out
    except Exception:
        return []


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


def race_stats_at_step(
    events: List[Tuple[float, int, float]],
    target_step: int,
    *,
    only_finished_steps: Optional[Set[int]] = None,
) -> Optional[Tuple[float, float, float, int]]:
    """(best_s, mean_s, std_s, n) for events with step <= target_step."""
    candidates = [(r, s, v) for r, s, v in events if s <= target_step]
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


def finish_stats_at_step(
    time_events: List[Tuple[float, int, float]],
    finished_events: List[Tuple[float, int, float]],
    target_step: int,
) -> Optional[Tuple[float, int, Optional[int]]]:
    """(finish_rate_0_1, n_finished, first_finish_step). Match by step."""
    finished_steps = set(s for r, s, v in finished_events if s <= target_step and v >= 0.5)
    all_up_to = [(r, s, v) for r, s, v in time_events if s <= target_step]
    if not all_up_to:
        return None
    n_total = len(all_up_to)
    n_finished = sum(1 for (_, s, _) in all_up_to if s in finished_steps)
    rate = n_finished / n_total if n_total else 0.0
    first_step = min((s for (_, s, _) in all_up_to if s in finished_steps), default=None)
    return (rate, n_finished, first_step)


def value_at_step(
    data: List[Tuple[float, int, float]],
    target_step: int,
    kind: str,
) -> Optional[Tuple[float, int, float]]:
    """Value at checkpoint target_step (training step).
    - 'time': best (min) value by that step — for race times.
    - 'last': last value at or before target_step — for loss, avg_q, training_pct.
    """
    if not data:
        return None
    candidates = [(r, s, v) for r, s, v in data if s <= target_step]
    if not candidates:
        return None
    if kind == 'time':
        return min(candidates, key=lambda x: x[2])
    return max(candidates, key=lambda x: x[1])  # last by step


def _scalar_tag_kind(tag: str) -> str:
    """Infer value kind for a scalar tag: 'time' (best so far) or 'last'."""
    if 'time' in tag and 'ms' in tag:
        return 'time'
    return 'last'


def _scalar_value_to_plot(tag: str, val: float) -> float:
    """Convert raw TensorBoard value to plot y (e.g. ms -> s, 0..1 -> 0..100)."""
    if 'time' in tag and 'ms' in tag:
        return val / 1000.0
    if 'percentage' in tag or 'learner_percentage' in tag:
        return val * 100.0
    return val


def _tag_to_scalar_key(tag: str) -> str:
    """Metric key for by_time/by_step scalar dict: METRICS short name or tag."""
    for key, t in METRICS.items():
        if t == tag:
            return key
    return tag


def compute_comparison_data(
    run_names: List[str],
    base_dir: Path,
    interval_min: int = 5,
    step_interval: int = 50000,
    use_all_scalars: bool = False,
    extra_scalar_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load runs and compute structured data for tables and plots.
    Returns dict with: run_names, durations, checkpoints, step_checkpoints,
    by_time (race_best, race_rate, scalar), by_step (same keys).
    Series format: metric_id -> run_name -> [(x, y), ...] where x is min or step.

    If use_all_scalars is True, loads every scalar tag present in any run (except race tags).
    extra_scalar_tags: additional TensorBoard tag names to load and plot.
    """
    base_dir = Path(base_dir)
    tags_to_load: List[str] = list(METRICS.values())
    if use_all_scalars:
        for name in run_names:
            tags_to_load = list(set(tags_to_load) | set(get_available_scalar_tags(base_dir / name)))
    if extra_scalar_tags:
        tags_to_load = list(set(tags_to_load) | set(extra_scalar_tags))

    cache: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    race_time: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    race_finished: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    for name in run_names:
        p = base_dir / name
        cache[name] = load_run_metrics(p, tags_to_load=tags_to_load)
        race_time[name], race_finished[name] = load_race_events(p)

    durations: Dict[str, float] = {}
    for name in run_names:
        d = 0.0
        for data in cache.get(name, {}).values():
            if data:
                d = max(d, max(r for r, _, _ in data))
        durations[name] = d

    common_max_min = min(durations.values()) if durations else 0
    checkpoints = list(range(interval_min, int(common_max_min) + 1, interval_min))
    if not checkpoints and common_max_min > 0:
        checkpoints = [int(common_max_min)]

    all_race_tags: Set[str] = set()
    for name in run_names:
        all_race_tags |= set(race_time.get(name, {}).keys())

    # by_time: series for plotting (one metric per key, runs as lines)
    by_time_race_best: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    by_time_race_rate: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    by_time_scalar: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    for tag in sorted(all_race_tags):
        runs_with_tag = [n for n in run_names if tag in race_time.get(n, {})]
        if not runs_with_tag:
            continue
        finished_tag = _race_time_to_finished_tag(tag)
        by_time_race_best[tag] = {}
        by_time_race_rate[tag] = {}
        for run in runs_with_tag:
            events = race_time[run][tag]
            fin_events = race_finished[run].get(finished_tag, []) if finished_tag else []
            best_series = []
            rate_series = []
            for t in checkpoints:
                finished_steps: Optional[Set[int]] = None
                if fin_events:
                    finished_steps = set(s for r, s, v in fin_events if r <= float(t) and v >= 0.5)
                st = race_stats_at_checkpoint(events, float(t), only_finished_steps=None)
                fin_stat = finish_stats_at_checkpoint(events, fin_events, float(t)) if fin_events else None
                if st is not None:
                    best_series.append((float(t), st[0]))
                if fin_stat is not None:
                    rate_series.append((float(t), fin_stat[0] * 100.0))
            if best_series:
                by_time_race_best[tag][run] = best_series
            if rate_series:
                by_time_race_rate[tag][run] = rate_series
    all_scalar_tags: Set[str] = set()
    for name in run_names:
        all_scalar_tags |= set(cache.get(name, {}).keys())
    for tag in sorted(all_scalar_tags):
        key = _tag_to_scalar_key(tag)
        kind = _scalar_tag_kind(tag)
        by_time_scalar[key] = {}
        for run in run_names:
            data = cache.get(run, {}).get(tag, [])
            series = []
            for t in checkpoints:
                v = value_at_minutes(data, float(t), kind)
                if v is not None:
                    rel, step, val = v
                    y = _scalar_value_to_plot(tag, val)
                    series.append((float(t), y))
            if series:
                by_time_scalar[key][run] = series

    # by_step
    max_step_per_run: Dict[str, int] = {}
    for name in run_names:
        m = 0
        for data in cache.get(name, {}).values():
            if data:
                m = max(m, max(s for _, s, _ in data))
        for tag_data in race_time.get(name, {}).values():
            if tag_data:
                m = max(m, max(s for _, s, _ in tag_data))
        for tag_data in race_finished.get(name, {}).values():
            if tag_data:
                m = max(m, max(s for _, s, _ in tag_data))
        max_step_per_run[name] = m
    common_max_step = min(max_step_per_run.values()) if max_step_per_run else 0
    step_checkpoints = list(range(step_interval, int(common_max_step) + 1, step_interval))
    if not step_checkpoints and common_max_step > 0:
        step_checkpoints = [int(common_max_step)]

    by_step_race_best: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    by_step_race_rate: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    by_step_scalar: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    for tag in sorted(all_race_tags):
        runs_with_tag = [n for n in run_names if tag in race_time.get(n, {})]
        if not runs_with_tag:
            continue
        finished_tag = _race_time_to_finished_tag(tag)
        by_step_race_best[tag] = {}
        by_step_race_rate[tag] = {}
        for run in runs_with_tag:
            events = race_time[run][tag]
            fin_events = race_finished[run].get(finished_tag, []) if finished_tag else []
            best_series = []
            rate_series = []
            for S in step_checkpoints:
                finished_steps = set(s for r, s, v in fin_events if s <= S and v >= 0.5) if fin_events else None
                st = race_stats_at_step(events, S, only_finished_steps=None)
                fin_stat = finish_stats_at_step(events, fin_events, S) if fin_events else None
                if st is not None:
                    best_series.append((S, st[0]))
                if fin_stat is not None:
                    rate_series.append((S, fin_stat[0] * 100.0))
            if best_series:
                by_step_race_best[tag][run] = best_series
            if rate_series:
                by_step_race_rate[tag][run] = rate_series
    for tag in sorted(all_scalar_tags):
        key = _tag_to_scalar_key(tag)
        kind = _scalar_tag_kind(tag)
        by_step_scalar[key] = {}
        for run in run_names:
            data = cache.get(run, {}).get(tag, [])
            series = []
            for S in step_checkpoints:
                v = value_at_step(data, S, kind)
                if v is not None:
                    rel, step, val = v
                    y = _scalar_value_to_plot(tag, val)
                    series.append((S, y))
            if series:
                by_step_scalar[key][run] = series

    return {
        "run_names": run_names,
        "durations": durations,
        "checkpoints": checkpoints,
        "step_checkpoints": step_checkpoints,
        "all_race_tags": sorted(all_race_tags),
        "all_scalar_tags": sorted(all_scalar_tags),
        "cache": cache,
        "race_time": race_time,
        "race_finished": race_finished,
        "by_time": {
            "race_best": by_time_race_best,
            "race_rate": by_time_race_rate,
            "scalar": by_time_scalar,
        },
        "by_step": {
            "race_best": by_step_race_best,
            "race_rate": by_step_race_rate,
            "scalar": by_step_scalar,
        },
    }


def _print_tables(data: Dict[str, Any]) -> None:
    """Print comparison tables from compute_comparison_data result."""
    run_names = data["run_names"]
    durations = data["durations"]
    checkpoints = data["checkpoints"]
    step_checkpoints = data["step_checkpoints"]
    all_race_tags = data["all_race_tags"]
    cache = data["cache"]
    race_time = data["race_time"]
    race_finished = data["race_finished"]
    all_scalar_tags = data.get("all_scalar_tags") or sorted(
        set().union(*(set(cache.get(n, {}).keys()) for n in run_names))
    )

    for name in run_names:
        print(f"Loaded {name} ({len(cache[name])} scalar metrics, {len(race_time[name])} race-time tags)")
    for name in run_names:
        print(f"{name}: duration ~{durations[name]:.0f} min (relative time)")
    print(f"\nCheckpoints (min from run start): {checkpoints}")
    print("=" * 80)

    for tag in all_race_tags:
        runs_with_tag = [n for n in run_names if tag in race_time.get(n, {})]
        if not runs_with_tag:
            continue
        finished_tag = _race_time_to_finished_tag(tag)
        print(f"\n{tag} (from per-race events: best / mean / std / best_fin; finish rate; first finish min)")
        print("-" * 80)
        parts = ["min"]
        for n in runs_with_tag:
            parts.extend([f"{n}_best", f"{n}_mean", f"{n}_std", f"{n}_best_fin", f"{n}_rate", f"{n}_first"])
        print("\t".join(parts))
        for t in checkpoints:
            row = [str(t)]
            for name in runs_with_tag:
                events = race_time[name][tag]
                fin_events = race_finished[name].get(finished_tag, []) if finished_tag else []
                finished_steps = set(s for r, s, v in fin_events if r <= float(t) and v >= 0.5) if fin_events else None
                st_all = race_stats_at_checkpoint(events, float(t), only_finished_steps=None)
                st_fin = race_stats_at_checkpoint(events, float(t), only_finished_steps=finished_steps) if finished_steps else None
                fin_stat = finish_stats_at_checkpoint(events, fin_events, float(t)) if fin_events else None
                if st_all is not None:
                    row.extend([f"{st_all[0]:.3f}s", f"{st_all[1]:.2f}s", f"{st_all[2]:.2f}s"])
                else:
                    row.extend(["-", "-", "-"])
                if st_fin is not None:
                    row.append(f"{st_fin[0]:.3f}s")
                else:
                    row.append("-")
                if fin_stat is not None:
                    row.append(f"{fin_stat[0]*100:.0f}%")
                    row.append(f"{fin_stat[2]:.1f}" if fin_stat[2] is not None else "-")
                else:
                    row.extend(["-", "-"])
            print("\t".join(row))

    print("\n" + "=" * 80)
    print("Scalar metrics (alltime_min_ms, loss, Q, training %, and any loaded tags)")
    print("=" * 80)
    for tag in all_scalar_tags:
        key = _tag_to_scalar_key(tag)
        kind = _scalar_tag_kind(tag)
        print(f"\n{tag}")
        print("-" * 60)
        print("min\t" + "\t".join(run_names))
        for t in checkpoints:
            cells = [str(t)]
            for name in run_names:
                d = cache.get(name, {}).get(tag, [])
                v = value_at_minutes(d, float(t), kind)
                if v is None:
                    cells.append("-")
                else:
                    rel, step, val = v
                    y = _scalar_value_to_plot(tag, val)
                    if 'percentage' in tag or 'learner_percentage' in tag:
                        cells.append(f"{y:.1f}%")
                    elif 'time' in tag and 'ms' in tag:
                        cells.append(f"{y:.3f}s")
                    elif 'avg_q' in key or 'Q' in tag:
                        cells.append(f"{y:.4f}")
                    else:
                        cells.append(f"{y:.2f}")
            print("\t".join(cells))

    print("\n" + "=" * 80)
    print("BY STEP (training step checkpoints; compare at equal gradient updates)")
    print("=" * 80)
    print(f"\nStep checkpoints: {step_checkpoints}")
    print("=" * 80)
    for tag in all_race_tags:
        runs_with_tag = [n for n in run_names if tag in race_time.get(n, {})]
        if not runs_with_tag:
            continue
        finished_tag = _race_time_to_finished_tag(tag)
        print(f"\n[BY STEP] {tag} (best / mean / std / best_fin; finish rate; first finish step)")
        print("-" * 80)
        parts = ["step"]
        for n in runs_with_tag:
            parts.extend([f"{n}_best", f"{n}_mean", f"{n}_std", f"{n}_best_fin", f"{n}_rate", f"{n}_first_step"])
        print("\t".join(parts))
        for S in step_checkpoints:
            row = [str(S)]
            for name in runs_with_tag:
                events = race_time[name][tag]
                fin_events = race_finished[name].get(finished_tag, []) if finished_tag else []
                finished_steps = set(s for r, s, v in fin_events if s <= S and v >= 0.5) if fin_events else None
                st_all = race_stats_at_step(events, S, only_finished_steps=None)
                st_fin = race_stats_at_step(events, S, only_finished_steps=finished_steps) if finished_steps else None
                fin_stat = finish_stats_at_step(events, fin_events, S) if fin_events else None
                if st_all is not None:
                    row.extend([f"{st_all[0]:.3f}s", f"{st_all[1]:.2f}s", f"{st_all[2]:.2f}s"])
                else:
                    row.extend(["-", "-", "-"])
                if st_fin is not None:
                    row.append(f"{st_fin[0]:.3f}s")
                else:
                    row.append("-")
                if fin_stat is not None:
                    row.append(f"{fin_stat[0]*100:.0f}%")
                    row.append(str(fin_stat[2]) if fin_stat[2] is not None else "-")
                else:
                    row.extend(["-", "-"])
            print("\t".join(row))

    print("\n" + "=" * 80)
    print("[BY STEP] Scalar metrics (alltime_min_ms, loss, Q, training %, and any loaded tags)")
    print("=" * 80)
    for tag in all_scalar_tags:
        key = _tag_to_scalar_key(tag)
        kind = _scalar_tag_kind(tag)
        print(f"\n{tag}")
        print("-" * 60)
        print("step\t" + "\t".join(run_names))
        for S in step_checkpoints:
            cells = [str(S)]
            for name in run_names:
                d = cache.get(name, {}).get(tag, [])
                v = value_at_step(d, S, kind)
                if v is None:
                    cells.append("-")
                else:
                    rel, step, val = v
                    y = _scalar_value_to_plot(tag, val)
                    if 'percentage' in tag or 'learner_percentage' in tag:
                        cells.append(f"{y:.1f}%")
                    elif 'time' in tag and 'ms' in tag:
                        cells.append(f"{y:.3f}s")
                    elif 'avg_q' in key or 'Q' in tag:
                        cells.append(f"{y:.4f}")
                    else:
                        cells.append(f"{y:.2f}")
            print("\t".join(cells))
    print()


def compare_by_relative_time(
    run_names: List[str],
    base_dir: Path = Path("tensorboard"),
    interval_min: int = 5,
    step_interval: int = 50000,
    plot_output_dir: Optional[Path] = None,
    plot_prefix: str = "",
    use_all_scalars: bool = False,
    extra_scalar_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare runs by relative time; print tables; optionally save comparison plots as JPG."""
    data = compute_comparison_data(
        run_names,
        base_dir,
        interval_min=interval_min,
        step_interval=step_interval,
        use_all_scalars=use_all_scalars,
        extra_scalar_tags=extra_scalar_tags,
    )
    _print_tables(data)
    if plot_output_dir is not None and plot_output_dir != Path(""):
        import sys
        _scripts_dir = Path(__file__).resolve().parent
        if str(_scripts_dir) not in sys.path:
            sys.path.insert(0, str(_scripts_dir))
        from experiment_plot_utils import plot_comparison
        plot_comparison(data, Path(plot_output_dir), prefix=plot_prefix)
    return data


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compare metrics by relative time and by training step")
    p.add_argument("--logdir", type=Path, default=Path("tensorboard"))
    p.add_argument("--interval", type=int, default=5, help="Checkpoint interval in minutes (relative time)")
    p.add_argument("--step_interval", type=int, default=50000, help="Checkpoint interval in training steps (by-step tables)")
    p.add_argument("--plot", action="store_true", help="Save comparison plots as JPG to --output-dir")
    p.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for plot JPGs (used with --plot)")
    p.add_argument("--prefix", type=str, default="", help="Filename prefix for plot JPGs (e.g. exp_exploration)")
    p.add_argument("--all-scalars", action="store_true", help="Load and plot every scalar tag present in runs (not only METRICS)")
    p.add_argument("--metrics", type=str, nargs="*", default=None, help="Extra TensorBoard scalar tag names to load (e.g. Training/loss)")
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    compare_by_relative_time(
        args.runs,
        base_dir=args.logdir,
        interval_min=args.interval,
        step_interval=args.step_interval,
        plot_output_dir=args.output_dir if args.plot else None,
        plot_prefix=args.prefix or "",
        use_all_scalars=args.all_scalars,
        extra_scalar_tags=args.metrics,
    )
