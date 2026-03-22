"""
Compare TensorBoard metrics along a time axis, plus BY STEP tables.

**Default (``--time-axis auto``):** uses **cumulative training hours** when every run logs the scalar;
otherwise **wall-clock minutes** from the first TensorBoard event (calendar time in the log, including gaps).

**Wall-clock mode:** ``--time-axis wall_minutes`` — X = minutes from the first merged TB event.

**Cumulative training hours mode:** ``--time-axis cumul_training_hours`` uses the scalar
``cumul_training_hours`` (logged every 5 min while the learner runs). This matches the
console "Training hours" counter: it does **not** grow while the process is down, so restarts
do not inject fake "elapsed" time. Requires that scalar in TensorBoard (Rulka learner logs it).

Race times (preferred):
  - Per-race Race/eval_race_time_* and Race/explo_race_time_*.
  - Each event is mapped to X via wall minutes OR via cumul_training_hours at that event's step.
  - At each checkpoint: best / mean / std, finish rate, first finish (same X semantics).

Scalar metrics: last or best value at each checkpoint (same X axis).

BY STEP: unchanged (training step checkpoints) — best for equal compute regardless of calendar time.

Usage:
  python scripts/analyze_experiment_by_relative_time.py uni_5 uni_7 [--interval 5] [--step_interval 50000]
  python scripts/analyze_experiment_by_relative_time.py RUN1 RUN2 --time-axis cumul_training_hours --interval-training-hours 0.5

Default ``--time-axis`` is **auto**: use cumulative training hours when all runs log ``cumul_training_hours``,
otherwise wall minutes (see module docstring).
"""

from pathlib import Path
import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, List, Optional, Set, Tuple

METRICS = {
    'hock_best_time_ms': 'alltime_min_ms_hock',   # best time so far (5-min scalar; use race events for richer view)
    'a01_best_time_ms': 'alltime_min_ms_A01',
    'loss': 'Training/loss',
    'avg_q': 'RL/avg_Q_trained_A01',
    'training_pct': 'Performance/learner_percentage_training',  # 0..1
}

# Logged from learner step_stats (accumulated_stats); same as console "Training hours"
CUMUL_TRAINING_HOURS_TAG = "cumul_training_hours"

# tensorboard_suffix_schedule creates run_2, run_3, ... at step thresholds; we merge all
SUFFIXES = ["", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10"]

# Per-race event tags: best/mean/std at checkpoint = min/mean/std of events up to that time
RACE_TIME_PREFIXES = ("Race/eval_race_time_", "Race/explo_race_time_")
# Exclude "race_time_finished" (subset) so we get every race
RACE_TIME_EXCLUDE = ("race_time_finished",)
RACE_FINISHED_PREFIXES = ("Race/eval_race_finished_", "Race/explo_race_finished_")


def discover_run_paths(base_dir: Path, run_name: str) -> List[Path]:
    """Find all TensorBoard log dirs for a run (run_name, run_name_2, run_name_3, ...).
    Matches tensorboard_suffix_schedule: logs split at 6M, 15M, 30M, ... steps."""
    paths: List[Path] = []
    for suf in SUFFIXES:
        p = base_dir / (run_name + suf)
        if p.exists():
            paths.append(p)
        elif suf == "":
            break  # base run missing
    return paths


def load_run_metrics(
    run_path: Path,
    tags_to_load: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[float, int, float]]]:
    """Load run from one path. Prefer load_run_metrics_from_paths for runs with suffix schedule."""
    return load_run_metrics_from_paths([run_path], tags_to_load)


def load_run_metrics_from_paths(
    run_paths: List[Path],
    tags_to_load: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[float, int, float]]]:
    """Load run from one or more paths (e.g. uni_20, uni_20_2, uni_20_3) and merge.
    Uses global t0 = min wall_time across all chunks for consistent relative_min."""
    if not run_paths:
        return {}
    requested = tags_to_load if tags_to_load is not None else list(METRICS.values())
    global_t0: Optional[float] = None
    all_events: Dict[str, List[Tuple[float, int, float]]] = {}

    for run_path in run_paths:
        if not run_path.exists():
            continue
        try:
            ea = EventAccumulator(str(run_path))
            ea.Reload()
            available = ea.Tags().get('scalars', [])
            for tag in requested:
                if tag not in available:
                    continue
                events = ea.Scalars(tag)
                if not events:
                    continue
                for e in events:
                    if global_t0 is None or e.wall_time < global_t0:
                        global_t0 = e.wall_time
                lst = all_events.setdefault(tag, [])
                for e in events:
                    lst.append((e.wall_time, e.step, e.value))
        except Exception as e:
            print(f"Error loading {run_path}: {e}")

    if global_t0 is None:
        return {}
    out: Dict[str, List[Tuple[float, int, float]]] = {}
    for tag, raw in all_events.items():
        merged = [((w - global_t0) / 60.0, s, v) for (w, s, v) in raw]
        merged.sort(key=lambda x: x[1])
        out[tag] = merged
    return out


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
    return get_available_scalar_tags_from_paths([run_path])


def get_available_scalar_tags_from_paths(run_paths: List[Path]) -> List[str]:
    """Union of scalar tags from all run paths."""
    seen: Set[str] = set()
    out: List[str] = []
    for run_path in run_paths:
        if not run_path.exists():
            continue
        try:
            ea = EventAccumulator(str(run_path))
            ea.Reload()
            for tag in ea.Tags().get('scalars', []):
                if _is_race_time_tag(tag) or any(tag.startswith(p) for p in RACE_FINISHED_PREFIXES):
                    continue
                if tag not in seen:
                    seen.add(tag)
                    out.append(tag)
        except Exception:
            pass
    return out


def load_race_events(
    run_path: Path,
) -> Tuple[
    Dict[str, List[Tuple[float, int, float]]],
    Dict[str, List[Tuple[float, int, float]]],
]:
    """Load per-race time and finished events from one path."""
    return load_race_events_from_paths([run_path])


def load_race_events_from_paths(
    run_paths: List[Path],
) -> Tuple[
    Dict[str, List[Tuple[float, int, float]]],
    Dict[str, List[Tuple[float, int, float]]],
]:
    """Load per-race time and finished events from all paths, merge, use global t0."""
    if not run_paths:
        return {}, {}
    run_t0: Optional[float] = None
    raw_time: Dict[str, List[Tuple[float, int, float]]] = {}
    raw_finished: Dict[str, List[Tuple[float, int, float]]] = {}

    for run_path in run_paths:
        if not run_path.exists():
            continue
        try:
            ea = EventAccumulator(str(run_path))
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            time_tags = [t for t in tags if _is_race_time_tag(t)]
            finished_tags = [t for t in tags if any(t.startswith(p) for p in RACE_FINISHED_PREFIXES)]
            for tag in time_tags + finished_tags:
                for e in ea.Scalars(tag):
                    if run_t0 is None or e.wall_time < run_t0:
                        run_t0 = e.wall_time
            for tag in time_tags:
                events = ea.Scalars(tag)
                if events:
                    lst = raw_time.setdefault(tag, [])
                    lst.extend([(e.wall_time, e.step, e.value) for e in events])
            for tag in finished_tags:
                events = ea.Scalars(tag)
                if events:
                    lst = raw_finished.setdefault(tag, [])
                    lst.extend([(e.wall_time, e.step, e.value) for e in events])
        except Exception as e:
            print(f"Error loading race events from {run_path}: {e}")

    if run_t0 is None:
        return {}, {}
    time_events: Dict[str, List[Tuple[float, int, float]]] = {}
    for tag, raw in raw_time.items():
        merged = [((w - run_t0) / 60.0, s, v) for (w, s, v) in raw]
        merged.sort(key=lambda x: x[1])
        time_events[tag] = merged
    finished_events: Dict[str, List[Tuple[float, int, float]]] = {}
    for tag, raw in raw_finished.items():
        merged = [((w - run_t0) / 60.0, s, v) for (w, s, v) in raw]
        merged.sort(key=lambda x: x[1])
        finished_events[tag] = merged
    return time_events, finished_events


def load_cumul_training_hours_step_series(run_paths: List[Path]) -> List[Tuple[int, float]]:
    """Merged (step, cumul_training_hours) from all TB chunks, sorted by step; last wins on duplicate step."""
    rows: List[Tuple[int, float]] = []
    for run_path in run_paths:
        if not run_path.exists():
            continue
        try:
            ea = EventAccumulator(str(run_path))
            ea.Reload()
            if CUMUL_TRAINING_HOURS_TAG not in ea.Tags().get("scalars", []):
                continue
            for e in ea.Scalars(CUMUL_TRAINING_HOURS_TAG):
                rows.append((int(e.step), float(e.value)))
        except Exception as ex:
            print(f"Error loading {CUMUL_TRAINING_HOURS_TAG} from {run_path}: {ex}")
    rows.sort(key=lambda x: x[0])
    out: List[Tuple[int, float]] = []
    for step, val in rows:
        if out and out[-1][0] == step:
            out[-1] = (step, val)
        else:
            out.append((step, val))
    return out


def hours_at_step(step_hours: List[Tuple[int, float]], step: int) -> float:
    """Cumulative training hours at or before ``step`` (last logged value); 0 if before first sample."""
    if not step_hours:
        return 0.0
    if step < step_hours[0][0]:
        return 0.0
    lo, hi = 0, len(step_hours) - 1
    best = step_hours[0][1]
    while lo <= hi:
        mid = (lo + hi) // 2
        sm, hm = step_hours[mid]
        if sm <= step:
            best = hm
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _remap_first_coord_to_training_hours(
    events: List[Tuple[float, int, float]],
    step_hours: List[Tuple[int, float]],
) -> List[Tuple[float, int, float]]:
    return [(hours_at_step(step_hours, s), s, v) for (_, s, v) in events]


def _remap_run_dict_to_training_hours(
    d: Dict[str, List[Tuple[float, int, float]]],
    step_hours: List[Tuple[int, float]],
) -> None:
    for k in list(d.keys()):
        d[k] = _remap_first_coord_to_training_hours(d[k], step_hours)


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
    time_axis: str = "wall_minutes",
    interval_training_hours: float = 0.5,
) -> Dict[str, Any]:
    """Load runs and compute structured data for tables and plots.
    Returns dict with: run_names, durations, checkpoints, step_checkpoints,
    by_time (race_best, race_mean, race_rate, scalar), by_step (same keys).
    Series format: metric_id -> run_name -> [(x, y), ...] where x is minutes, training hours, or step.

    time_axis:
      - ``wall_minutes``: first tensor in tuple is minutes from first TB event (includes downtime between sessions).
      - ``cumul_training_hours``: first coord is ``cumul_training_hours`` at that point's training step
        (matches console "Training hours"; pauses between restarts are excluded).

    If use_all_scalars is True, loads every scalar tag present in any run (except race tags).
    extra_scalar_tags: additional TensorBoard tag names to load and plot.
    """
    base_dir = Path(base_dir)
    tags_to_load: List[str] = list(METRICS.values())
    if use_all_scalars:
        for name in run_names:
            paths = discover_run_paths(base_dir, name)
            if not paths:
                paths = [base_dir / name]
            tags_to_load = list(set(tags_to_load) | set(get_available_scalar_tags_from_paths(paths)))
    if extra_scalar_tags:
        tags_to_load = list(set(tags_to_load) | set(extra_scalar_tags))

    if time_axis not in ("wall_minutes", "cumul_training_hours"):
        raise ValueError(f"time_axis must be 'wall_minutes' or 'cumul_training_hours', got {time_axis!r}")
    if time_axis == "cumul_training_hours":
        tags_to_load = list(set(tags_to_load) | {CUMUL_TRAINING_HOURS_TAG})

    cache: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    race_time: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    race_finished: Dict[str, Dict[str, List[Tuple[float, int, float]]]] = {}
    multi_chunk_runs: List[str] = []
    for name in run_names:
        paths = discover_run_paths(base_dir, name)
        if not paths:
            paths = [base_dir / name]
        if len(paths) > 1:
            print(f"[INFO] {name}: merging {len(paths)} log dirs ({', '.join(p.name for p in paths)})")
            multi_chunk_runs.append(name)
        cache[name] = load_run_metrics_from_paths(paths, tags_to_load=tags_to_load)
        race_time[name], race_finished[name] = load_race_events_from_paths(paths)

    if time_axis == "wall_minutes" and multi_chunk_runs:
        print(
            "[WARN] Wall-clock axis: X is minutes from the earliest TensorBoard wall_time in the "
            "merged run(s). Calendar gaps between log folders/sessions are included and can dwarf "
            "actual training time. Affected runs: "
            + ", ".join(multi_chunk_runs)
            + ". Prefer --time-axis cumul_training_hours (or BY STEP tables) for learning curves.",
            file=sys.stderr,
        )

    hours_series_by_run: Dict[str, List[Tuple[int, float]]] = {}
    if time_axis == "cumul_training_hours":
        for name in run_names:
            paths = discover_run_paths(base_dir, name)
            if not paths:
                paths = [base_dir / name]
            sh = load_cumul_training_hours_step_series(paths)
            if not sh:
                raise ValueError(
                    f"{name!r}: no TensorBoard scalar {CUMUL_TRAINING_HOURS_TAG}; "
                    "use time_axis wall_minutes or upgrade learner logging."
                )
            hours_series_by_run[name] = sh
            _remap_run_dict_to_training_hours(race_time[name], sh)
            _remap_run_dict_to_training_hours(race_finished[name], sh)
            for tag in list(cache[name].keys()):
                cache[name][tag] = _remap_first_coord_to_training_hours(cache[name][tag], sh)

    durations: Dict[str, float] = {}
    for name in run_names:
        if time_axis == "cumul_training_hours":
            sh = hours_series_by_run[name]
            durations[name] = sh[-1][1] if sh else 0.0
        else:
            d = 0.0
            for data in cache.get(name, {}).values():
                if data:
                    d = max(d, max(r for r, _, _ in data))
            durations[name] = d

    common_max_min = min(durations.values()) if durations else 0.0
    if time_axis == "cumul_training_hours":
        ih = float(interval_training_hours)
        if ih <= 0:
            raise ValueError("interval_training_hours must be positive")
        checkpoints: List[float] = []
        t = ih
        while t <= common_max_min + 1e-9:
            checkpoints.append(round(t, 4))
            t += ih
        if not checkpoints and common_max_min > 0:
            checkpoints = [round(float(common_max_min), 4)]
    else:
        checkpoints = list(range(interval_min, int(common_max_min) + 1, interval_min))
        if not checkpoints and common_max_min > 0:
            checkpoints = [int(common_max_min)]

    all_race_tags: Set[str] = set()
    for name in run_names:
        all_race_tags |= set(race_time.get(name, {}).keys())

    # by_time: series for plotting (one metric per key, runs as lines)
    by_time_race_best: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    by_time_race_mean: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    by_time_race_rate: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    by_time_scalar: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    for tag in sorted(all_race_tags):
        runs_with_tag = [n for n in run_names if tag in race_time.get(n, {})]
        if not runs_with_tag:
            continue
        finished_tag = _race_time_to_finished_tag(tag)
        by_time_race_best[tag] = {}
        by_time_race_mean[tag] = {}
        by_time_race_rate[tag] = {}
        for run in runs_with_tag:
            events = race_time[run][tag]
            fin_events = race_finished[run].get(finished_tag, []) if finished_tag else []
            best_series = []
            mean_series = []
            rate_series = []
            for t in checkpoints:
                finished_steps: Optional[Set[int]] = None
                if fin_events:
                    finished_steps = set(s for r, s, v in fin_events if r <= float(t) and v >= 0.5)
                st = race_stats_at_checkpoint(events, float(t), only_finished_steps=None)
                fin_stat = finish_stats_at_checkpoint(events, fin_events, float(t)) if fin_events else None
                if st is not None:
                    best_series.append((float(t), st[0]))
                    mean_series.append((float(t), st[1]))
                if fin_stat is not None:
                    rate_series.append((float(t), fin_stat[0] * 100.0))
            if best_series:
                by_time_race_best[tag][run] = best_series
            if mean_series:
                by_time_race_mean[tag][run] = mean_series
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
    by_step_race_mean: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    by_step_race_rate: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    by_step_scalar: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    for tag in sorted(all_race_tags):
        runs_with_tag = [n for n in run_names if tag in race_time.get(n, {})]
        if not runs_with_tag:
            continue
        finished_tag = _race_time_to_finished_tag(tag)
        by_step_race_best[tag] = {}
        by_step_race_mean[tag] = {}
        by_step_race_rate[tag] = {}
        for run in runs_with_tag:
            events = race_time[run][tag]
            fin_events = race_finished[run].get(finished_tag, []) if finished_tag else []
            best_series = []
            mean_series = []
            rate_series = []
            for S in step_checkpoints:
                finished_steps = set(s for r, s, v in fin_events if s <= S and v >= 0.5) if fin_events else None
                st = race_stats_at_step(events, S, only_finished_steps=None)
                fin_stat = finish_stats_at_step(events, fin_events, S) if fin_events else None
                if st is not None:
                    best_series.append((S, st[0]))
                    mean_series.append((S, st[1]))
                if fin_stat is not None:
                    rate_series.append((S, fin_stat[0] * 100.0))
            if best_series:
                by_step_race_best[tag][run] = best_series
            if mean_series:
                by_step_race_mean[tag][run] = mean_series
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
        "time_axis": time_axis,
        "interval_training_hours": interval_training_hours,
        "all_race_tags": sorted(all_race_tags),
        "all_scalar_tags": sorted(all_scalar_tags),
        "cache": cache,
        "race_time": race_time,
        "race_finished": race_finished,
        "by_time": {
            "race_best": by_time_race_best,
            "race_mean": by_time_race_mean,
            "race_rate": by_time_race_rate,
            "scalar": by_time_scalar,
        },
        "by_step": {
            "race_best": by_step_race_best,
            "race_mean": by_step_race_mean,
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
    time_axis = data.get("time_axis", "wall_minutes")
    all_race_tags = data["all_race_tags"]
    cache = data["cache"]
    race_time = data["race_time"]
    race_finished = data["race_finished"]
    all_scalar_tags = data.get("all_scalar_tags") or sorted(
        set().union(*(set(cache.get(n, {}).keys()) for n in run_names))
    )
    x_label = "cumul_training_h" if time_axis == "cumul_training_hours" else "min"
    first_finish_note = (
        "first finish (cumul training h)"
        if time_axis == "cumul_training_hours"
        else "first finish min"
    )

    for name in run_names:
        print(f"Loaded {name} ({len(cache[name])} scalar metrics, {len(race_time[name])} race-time tags)")
    for name in run_names:
        if time_axis == "cumul_training_hours":
            print(f"{name}: duration ~{durations[name]:.2f} h (cumulative training hours, excl. process downtime)")
        else:
            print(f"{name}: duration ~{durations[name]:.0f} min (wall clock from first TensorBoard event)")
    if time_axis == "cumul_training_hours":
        ih = data.get("interval_training_hours", 0.5)
        print(f"\nCheckpoints (cumulative training hours, step every {ih} h): {checkpoints}")
    else:
        print(f"\nCheckpoints (min from first TensorBoard event): {checkpoints}")
    print("=" * 80)

    for tag in all_race_tags:
        runs_with_tag = [n for n in run_names if tag in race_time.get(n, {})]
        if not runs_with_tag:
            continue
        finished_tag = _race_time_to_finished_tag(tag)
        print(
            f"\n{tag} (from per-race events: best / mean / std / best_fin; finish rate; {first_finish_note})"
        )
        print("-" * 80)
        parts = [x_label]
        for n in runs_with_tag:
            parts.extend([f"{n}_best", f"{n}_mean", f"{n}_std", f"{n}_best_fin", f"{n}_rate", f"{n}_first"])
        print("\t".join(parts))
        for t in checkpoints:
            row = [f"{t:g}" if isinstance(t, float) else str(t)]
            for name in runs_with_tag:
                events = race_time[name][tag]
                fin_events = race_finished[name].get(finished_tag, []) if finished_tag else []
                tc = float(t)
                finished_steps = set(s for r, s, v in fin_events if r <= tc and v >= 0.5) if fin_events else None
                st_all = race_stats_at_checkpoint(events, tc, only_finished_steps=None)
                st_fin = race_stats_at_checkpoint(events, tc, only_finished_steps=finished_steps) if finished_steps else None
                fin_stat = finish_stats_at_checkpoint(events, fin_events, tc) if fin_events else None
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
                    row.append(f"{fin_stat[2]:.2f}" if fin_stat[2] is not None else "-")
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
        print(f"{x_label}\t" + "\t".join(run_names))
        for t in checkpoints:
            cells = [f"{t:g}" if isinstance(t, float) else str(t)]
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
    time_axis: str = "auto",
    interval_training_hours: float = 0.5,
) -> Dict[str, Any]:
    """Compare runs by wall minutes, cumulative training hours, or auto; print tables; optional plots."""
    resolved_axis = time_axis
    if time_axis == "auto":
        try:
            data = compute_comparison_data(
                run_names,
                base_dir,
                interval_min=interval_min,
                step_interval=step_interval,
                use_all_scalars=use_all_scalars,
                extra_scalar_tags=extra_scalar_tags,
                time_axis="cumul_training_hours",
                interval_training_hours=interval_training_hours,
            )
            resolved_axis = "cumul_training_hours"
        except ValueError as err:
            if CUMUL_TRAINING_HOURS_TAG not in str(err):
                raise
            data = compute_comparison_data(
                run_names,
                base_dir,
                interval_min=interval_min,
                step_interval=step_interval,
                use_all_scalars=use_all_scalars,
                extra_scalar_tags=extra_scalar_tags,
                time_axis="wall_minutes",
                interval_training_hours=interval_training_hours,
            )
            resolved_axis = "wall_minutes"
        print(
            f"[INFO] --time-axis auto: using {resolved_axis!r} for this comparison.",
            file=sys.stderr,
        )
    else:
        data = compute_comparison_data(
            run_names,
            base_dir,
            interval_min=interval_min,
            step_interval=step_interval,
            use_all_scalars=use_all_scalars,
            extra_scalar_tags=extra_scalar_tags,
            time_axis=time_axis,
            interval_training_hours=interval_training_hours,
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
    import sys

    p = argparse.ArgumentParser(description="Compare metrics by wall time, training hours, or by training step")
    p.add_argument("--logdir", type=Path, default=Path("tensorboard"))
    p.add_argument(
        "--time-axis",
        choices=("auto", "wall_minutes", "cumul_training_hours"),
        default="auto",
        help="auto: cumul_training_hours if all runs log it, else wall_minutes; or force one axis",
    )
    p.add_argument(
        "--interval-training-hours",
        type=float,
        default=0.5,
        help="Checkpoint step on X when --time-axis cumul_training_hours (ignored for wall_minutes)",
    )
    p.add_argument("--interval", type=int, default=5, help="Checkpoint interval in minutes (only for wall_minutes axis)")
    p.add_argument("--step_interval", type=int, default=50000, help="Checkpoint interval in training steps (by-step tables)")
    p.add_argument("--plot", action="store_true", help="Save comparison plots as JPG to --output-dir")
    p.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for plot JPGs (used with --plot)")
    p.add_argument("--prefix", type=str, default="", help="Filename prefix for plot JPGs (e.g. exp_exploration)")
    p.add_argument("--all-scalars", action="store_true", help="Load and plot every scalar tag present in runs (not only METRICS)")
    p.add_argument("--metrics", type=str, nargs="*", default=None, help="Extra TensorBoard scalar tag names to load (e.g. Training/loss)")
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    try:
        compare_by_relative_time(
            args.runs,
            base_dir=args.logdir,
            interval_min=args.interval,
            step_interval=args.step_interval,
            plot_output_dir=args.output_dir if args.plot else None,
            plot_prefix=args.prefix or "",
            use_all_scalars=args.all_scalars,
            extra_scalar_tags=args.metrics,
            time_axis=args.time_axis,
            interval_training_hours=args.interval_training_hours,
        )
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        raise SystemExit(1) from err
