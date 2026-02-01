"""
Plot comparison data from compute_comparison_data (analyze_experiment_by_relative_time).
One graph per metric; multiple runs as lines. Saves compressed JPG files.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


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
        plt.xlabel("Time (min from run start)")
        if key == "training_pct":
            plt.ylabel("Learner % training")
        elif key == "avg_q":
            plt.ylabel("Avg Q (trained A01)")
        elif "time" in key:
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
            plt.xlabel("Training step")
            if key == "training_pct":
                plt.ylabel("Learner % training")
            elif key == "avg_q":
                plt.ylabel("Avg Q (trained A01)")
            elif "time" in key:
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
