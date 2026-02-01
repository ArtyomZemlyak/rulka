"""
Plot comparison data from compute_comparison_data (analyze_experiment_by_relative_time).
One graph per metric; multiple runs as lines. Saves compressed JPG files.

Y-axis:
- **Best-time metrics** (race best, scalar best time): scale from **min(time)** to **mean(time) + 1s**
  so the initial 300s spike is off-scale and the improvement (e.g. down to 30s) is readable.
- **Other metrics** (loss, Q, finish rate): robust scaling (percentiles).
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _all_y_from_run_series(
    run_series: Dict[str, List[Tuple[float, float]]],
) -> List[float]:
    """Collect all y values from run_series (metric_id -> run -> [(x,y), ...])."""
    ys: List[float] = []
    for points in run_series.values():
        ys.extend(p[1] for p in points)
    return ys


def _robust_ylim(
    run_series: Dict[str, List[Tuple[float, float]]],
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    margin: float = 0.05,
    min_span_ratio: float = 0.02,
) -> Tuple[float, float]:
    """Compute y-axis limits from percentiles so outliers don't dominate.
    Used for non-time metrics (loss, Q, rate).
    """
    ys = _all_y_from_run_series(run_series)
    if not ys:
        return (0.0, 1.0)
    ys_sorted = sorted(ys)
    n = len(ys_sorted)
    lo_idx = max(0, int(n * low_pct / 100.0))
    hi_idx = min(n - 1, int(n * high_pct / 100.0))
    y_lo = ys_sorted[lo_idx]
    y_hi = ys_sorted[hi_idx]
    span = y_hi - y_lo
    if span <= 0 or (abs(y_hi) > 1e-9 and span / abs(y_hi) < min_span_ratio):
        pad = abs(y_lo) * 0.05 + 0.01 if abs(y_lo) > 1e-9 else 0.01
        return (y_lo - pad, y_lo + pad)
    pad = span * margin
    return (y_lo - pad, y_hi + pad)


def _time_axis_ylim(
    run_series: Dict[str, List[Tuple[float, float]]],
    margin_above_mean: float = 1.0,
    min_span: float = 2.0,
) -> Tuple[float, float]:
    """Y-axis for best-time metrics: from minimum time to mean + margin.
    Hides the initial 300s spike so the improvement (e.g. 300s -> 30s) is readable:
    scale is [best_time, mean_time + 1s], so the curve lives in the visible range.
    """
    ys = _all_y_from_run_series(run_series)
    if not ys:
        return (0.0, 1.0)
    y_min = min(ys)
    y_mean = sum(ys) / len(ys)
    y_hi = y_mean + margin_above_mean
    if y_hi <= y_min:
        y_hi = y_min + min_span
    span = y_hi - y_min
    if span < min_span:
        y_hi = y_min + min_span
    return (y_min, y_hi)


def _tag_to_slug(tag: str) -> str:
    """Convert race tag to short filename slug. E.g. Race/eval_race_time_trained_A01 -> eval_A01."""
    for prefix in ("Race/eval_race_time_trained_", "Race/explo_race_time_trained_"):
        if tag.startswith(prefix):
            return tag[len(prefix) :].replace(" ", "_")
    return tag.replace("/", "_").replace(" ", "_")


def _safe_filename(s: str) -> str:
    """Replace characters unsafe for filenames."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def plot_comparison(
    data: Dict[str, Any],
    output_dir: Path,
    prefix: str = "",
    by_step: bool = False,
    jpg_quality: int = 85,
) -> List[Path]:
    """Build one figure per metric (multiple runs as lines); save as compressed JPG.
    Returns list of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_names = data["run_names"]
    by_time = data["by_time"]
    by_step_data = data.get("by_step")
    saved: List[Path] = []

    def save_fig(name: str) -> None:
        fname = _safe_filename(f"{prefix}_{name}.jpg" if prefix else f"{name}.jpg")
        path = output_dir / fname
        plt.savefig(path, format="jpg", pil_kwargs={"quality": jpg_quality})
        plt.close()
        saved.append(path)

    # ---- By time: race best ----
    for tag, run_series in by_time["race_best"].items():
        if not run_series:
            continue
        has_data = any(points for points in run_series.values())
        if not has_data:
            continue
        plt.figure()
        for run, points in run_series.items():
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.plot(xs, ys, label=run, marker="o", markersize=3)
        y_lo, y_hi = _time_axis_ylim(run_series)
        plt.ylim(y_lo, y_hi)
        plt.xlabel("Time (min from run start)")
        plt.ylabel("Best race time (s)")
        plt.title(f"Best race time by relative time — {_tag_to_slug(tag)}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_fig(f"{_tag_to_slug(tag)}_best")

    # ---- By time: race finish rate ----
    for tag, run_series in by_time["race_rate"].items():
        if not run_series:
            continue
        if not any(points for points in run_series.values()):
            continue
        plt.figure()
        for run, points in run_series.items():
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.plot(xs, ys, label=run, marker="o", markersize=3)
        y_lo, y_hi = _robust_ylim(run_series)
        plt.ylim(max(0, y_lo), min(100, y_hi))  # clamp to 0–100%
        plt.xlabel("Time (min from run start)")
        plt.ylabel("Finish rate (%)")
        plt.title(f"Finish rate by relative time — {_tag_to_slug(tag)}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_fig(f"{_tag_to_slug(tag)}_rate")

    # ---- By time: scalar metrics ----
    for key, run_series in by_time["scalar"].items():
        if not run_series:
            continue
        plt.figure()
        for run, points in run_series.items():
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.plot(xs, ys, label=run, marker="o", markersize=3)
        is_time_metric = "time" in key and ("ms" in key or "best" in key)
        y_lo, y_hi = _time_axis_ylim(run_series) if is_time_metric else _robust_ylim(run_series)
        plt.ylim(y_lo, y_hi)
        plt.xlabel("Time (min from run start)")
        if key == "training_pct":
            plt.ylabel("Learner % training")
        elif key == "avg_q":
            plt.ylabel("Avg Q (trained A01)")
        elif "time" in key or "time_ms" in key:
            plt.ylabel("Best time (s)")
        else:
            plt.ylabel(key)
        plt.title(f"{key} by relative time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_fig(key)

    # ---- By step (optional) ----
    if by_step and by_step_data:
        step_prefix = f"{prefix}_by_step" if prefix else "by_step"
        for tag, run_series in by_step_data["race_best"].items():
            if not run_series:
                continue
            plt.figure()
            for run, points in run_series.items():
                if not points:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                plt.plot(xs, ys, label=run, marker="o", markersize=3)
            y_lo, y_hi = _time_axis_ylim(run_series)
            plt.ylim(y_lo, y_hi)
            plt.xlabel("Training step")
            plt.ylabel("Best race time (s)")
            plt.title(f"Best race time by step — {_tag_to_slug(tag)}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            fname = _safe_filename(f"{step_prefix}_{_tag_to_slug(tag)}_best") + ".jpg"
            path = output_dir / fname
            plt.savefig(path, format="jpg", pil_kwargs={"quality": jpg_quality})
            plt.close()
            saved.append(path)

        for key, run_series in by_step_data["scalar"].items():
            if not run_series:
                continue
            plt.figure()
            for run, points in run_series.items():
                if not points:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                plt.plot(xs, ys, label=run, marker="o", markersize=3)
            is_time_metric = "time" in key and ("ms" in key or "best" in key)
            y_lo, y_hi = _time_axis_ylim(run_series) if is_time_metric else _robust_ylim(run_series)
            plt.ylim(y_lo, y_hi)
            plt.xlabel("Training step")
            if key == "training_pct":
                plt.ylabel("Learner % training")
            elif key == "avg_q":
                plt.ylabel("Avg Q (trained A01)")
            elif "time" in key or "time_ms" in key:
                plt.ylabel("Best time (s)")
            else:
                plt.ylabel(key)
            plt.title(f"{key} by step")
            plt.legend()
            plt.grid(True, alpha=0.3)
            fname = _safe_filename(f"{step_prefix}_{key}") + ".jpg"
            path = output_dir / fname
            plt.savefig(path, format="jpg", pil_kwargs={"quality": jpg_quality})
            plt.close()
            saved.append(path)

    return saved
