"""
For each TensorBoard run (including merged suffix dirs), compare:

- **Wall span**: max relative minute from merged scalars (minutes since earliest ``wall_time`` in any chunk).
- **Cumulative training**: last ``cumul_training_hours`` scalar (learner active time only).

If wall span is much larger than training hours converted to minutes, by-time analysis using
**wall_minutes** misrepresents "how long the agent trained". Several TB folders are a hint but not
required (a single folder still jumps in ``wall_time`` across process restarts).

Usage:
  python scripts/audit_tensorboard_training_timeline.py [--logdir tensorboard]
  python scripts/audit_tensorboard_training_timeline.py --runs A01_as20_long_v2 uni_7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_experiment_by_relative_time import (  # noqa: E402
    CUMUL_TRAINING_HOURS_TAG,
    METRICS,
    SUFFIXES,
    discover_run_paths,
    load_cumul_training_hours_step_series,
    load_run_metrics_from_paths,
)


def _infer_run_groups(logdir: Path) -> Dict[str, List[Path]]:
    """Group first-level TB dirs by run base name (strip ``_2``, ``_3``, … suffixes)."""
    groups: Dict[str, List[Path]] = {}
    if not logdir.is_dir():
        return groups
    for p in sorted(logdir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        base = name
        for suf in SUFFIXES[1:]:
            if name.endswith(suf):
                base = name[: -len(suf)]
                break
        groups.setdefault(base, []).append(p)
    return groups


def _max_wall_rel_min(paths: List[Path]) -> float:
    """Use a single dense scalar (loss) for wall span; avoids loading every METRICS tag per run."""
    merged = load_run_metrics_from_paths(paths, ["Training/loss"])
    series = merged.get("Training/loss") or []
    if series:
        return max(r for r, _, _ in series)
    merged = load_run_metrics_from_paths(paths, list(METRICS.values()))
    m = 0.0
    for s in merged.values():
        if s:
            m = max(m, max(r for r, _, _ in s))
    return m


def _audit_one(base: str, paths: List[Path]) -> Tuple[str, int, float, Optional[float], str]:
    wall_min = _max_wall_rel_min(paths)
    ch_series = load_cumul_training_hours_step_series(paths)
    train_h: Optional[float] = ch_series[-1][1] if ch_series else None
    train_min = train_h * 60.0 if train_h is not None else None

    flag = "ok"
    if train_min is None or train_min <= 0:
        flag = "no_cumul_scalar"
    elif wall_min > train_min * 1.10 + 30.0:
        flag = "wall>>training"
    elif len(paths) > 1:
        flag = "merged_dirs"

    return base, len(paths), wall_min, train_h, flag


def main() -> None:
    p = argparse.ArgumentParser(description="Audit TB wall span vs cumul_training_hours per run")
    p.add_argument("--logdir", type=Path, default=Path("tensorboard"))
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run base names (default: infer from --logdir subdirectories)",
    )
    args = p.parse_args()
    logdir = args.logdir
    if not logdir.is_absolute():
        logdir = (_PROJECT_ROOT / logdir).resolve()

    rows: List[Tuple[str, int, float, Optional[float], str]] = []
    if args.runs:
        for name in args.runs:
            paths = discover_run_paths(logdir, name)
            if not paths:
                paths = [logdir / name] if (logdir / name).is_dir() else []
            if not paths:
                print(f"{name}: no TensorBoard directories under {logdir}", file=sys.stderr)
                continue
            rows.append(_audit_one(name, paths))
    else:
        for base, path_list in sorted(_infer_run_groups(logdir).items()):
            rows.append(_audit_one(base, path_list))

    print(
        "run\tchunks\twall_span_min\tcumul_train_h\ttrain_min_equiv\tratio_wall/train\tflag"
    )
    for base, nchunk, wall_min, train_h, flag in rows:
        if train_h is not None and train_h > 0:
            teq = train_h * 60.0
            ratio = wall_min / teq if teq > 0 else 0.0
            ratio_s = f"{ratio:.2f}"
            th_s = f"{train_h:.3f}"
        else:
            teq = float("nan")
            ratio_s = "n/a"
            th_s = "n/a"
        print(
            f"{base}\t{nchunk}\t{wall_min:.1f}\t{th_s}\t{train_h * 60.0 if train_h else 'n/a'}\t{ratio_s}\t{flag}"
        )

    print(
        "\n# flag: wall>>training = wall timeline much longer than cumul_training_hours "
        f"(prefer ``--time-axis cumul_training_hours`` or {CUMUL_TRAINING_HOURS_TAG} in docs).",
        file=sys.stderr,
    )
    print(
        "# merged_dirs = multiple TB roots only; still check ratio - single chunk can have restarts.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
