"""
Analyze BC pretrain runs: compare metrics from Lightning CSV (and optionally TensorBoard).

BC pretrain logs differ from RL: metrics are per-epoch (train_loss, val_loss, train_acc,
val_acc, train_acc_class_0..N, val_acc_class_0..N). This script reads CSV from run
directories (e.g. output/ptretrain/bc/v1/csv/metrics.csv or output/ptretrain/bc/v1/csv/
with version_0/metrics.csv) and prints comparison with action names (not just 0,1,2).

Action names match config_files/rl/config_default.yaml inputs (12 actions):
  0=accel, 1=left+accel, 2=right+accel, 3=coast, 4=left, 5=right,
  6=brake, 7=left+brake, 8=right+brake, 9=accel+brake, 10=left+accel+brake, 11=right+accel+brake

Combined analysis: recommends best epoch by val_loss and reports at that epoch both
loss/overall-acc and "main-actions" validation accuracy (mean over classes 0,1,2,3,10,11).

Usage:
  python scripts/analyze_pretrain_bc.py output/ptretrain/bc/v1 output/ptretrain/bc/v1.1
  python scripts/analyze_pretrain_bc.py --base-dir output/ptretrain/bc v1 v1.1 --plot --output-dir docs/source/_static
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Classes that usually have enough val samples to be meaningful (accel, left+accel, right+accel, coast, left+accel+brake, right+accel+brake)
MAIN_ACTION_INDICES = (0, 1, 2, 3, 10, 11)

# Action index -> short label (matches RL config inputs order)
BC_ACTION_NAMES = [
    "accel",           # 0: left=false, right=false, accelerate=true, brake=false
    "left+accel",      # 1
    "right+accel",     # 2
    "coast",           # 3: no keys
    "left",            # 4
    "right",           # 5
    "brake",           # 6
    "left+brake",     # 7
    "right+brake",     # 8
    "accel+brake",     # 9
    "left+accel+brake",   # 10
    "right+accel+brake",  # 11
]


def _csv_paths_from_run_dirs(run_dirs: list[Path]) -> list[tuple[str, Path]]:
    """Resolve (run_name, csv_path) for each run dir. Prefer run_dir/csv/version_0/metrics.csv then run_dir/csv/metrics.csv."""
    out: list[tuple[str, Path]] = []
    for run_dir in run_dirs:
        name = run_dir.name
        # Lightning CSVLogger can write to csv_dir/version_0/metrics.csv
        v0 = run_dir / "csv" / "version_0" / "metrics.csv"
        plain = run_dir / "csv" / "metrics.csv"
        if v0.exists():
            out.append((name, v0))
        elif plain.exists():
            out.append((name, plain))
        else:
            print(f"Warning: No CSV found for {name} in {run_dir} (tried csv/version_0/metrics.csv, csv/metrics.csv)")
    return out


def _load_meta(run_dir: Path) -> dict[str, Any] | None:
    meta_path = run_dir / "pretrain_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse numbers
            out: dict[str, Any] = {}
            for k, v in row.items():
                if v == "" or v is None:
                    out[k] = None
                    continue
                if k in ("epoch", "step"):
                    try:
                        out[k] = int(float(v))
                    except ValueError:
                        out[k] = v
                else:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            rows.append(out)
    return rows


def _merge_epoch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lightning CSV has alternating rows: one with val_* filled, one with train_* filled. Merge by epoch/step."""
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for r in rows:
        ep = r.get("epoch")
        step = r.get("step")
        if ep is None or step is None:
            continue
        key = (int(ep) if isinstance(ep, (int, float)) else int(ep), int(step) if isinstance(step, (int, float)) else int(step))
        if key not in by_key:
            by_key[key] = {"epoch": key[0], "step": key[1]}
        for k, v in r.items():
            if k in ("epoch", "step"):
                continue
            if v is not None and v != "":
                by_key[key][k] = v
    return [by_key[k] for k in sorted(by_key.keys())]


def _per_epoch_series(merged: list[dict[str, Any]], metric: str) -> list[tuple[int, float | None]]:
    """(epoch, value) for given metric; value None if missing."""
    result: list[tuple[int, float | None]] = []
    for row in merged:
        ep = row.get("epoch", 0)
        if isinstance(ep, float):
            ep = int(ep)
        val = row.get(metric)
        if val is not None and isinstance(val, (int, float)):
            result.append((ep, float(val)))
        else:
            result.append((ep, None))
    return result


def _class_columns(merged: list[dict[str, Any]], stage: str) -> list[str]:
    """Column names like train_acc_class_0, val_acc_class_0, ... that exist in merged."""
    prefix = f"{stage}_acc_class_"
    cols = set()
    for row in merged:
        for k in row:
            if k.startswith(prefix) and k[len(prefix) :].isdigit():
                cols.add(k)
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def _main_actions_val_acc(row: dict[str, Any]) -> float | None:
    """Mean of val_acc_class_* for MAIN_ACTION_INDICES; None if any missing."""
    vals: list[float] = []
    for c in MAIN_ACTION_INDICES:
        v = row.get(f"val_acc_class_{c}")
        if v is None:
            return None
        vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def run_analysis(
    run_dirs: list[Path],
    csv_paths: list[Path] | None,
    base_dir: Path | None,
    run_names: list[str] | None,
    interval_epochs: int,
    plot: bool = False,
    output_dir: Path | None = None,
) -> None:
    if csv_paths:
        # Explicit CSV paths: names from parent dir or file stem
        pairs: list[tuple[str, Path]] = []
        for p in csv_paths:
            name = p.parent.parent.name if "version" in p.parts else p.parent.name
            if "version" in p.parts:
                name = p.parent.parent.parent.name
            pairs.append((name, p))
    elif base_dir is not None and run_names:
        pairs = []
        for name in run_names:
            rdir = base_dir / name
            for sub in [rdir / "csv" / "version_0" / "metrics.csv", rdir / "csv" / "metrics.csv"]:
                if sub.exists():
                    pairs.append((name, sub))
                    break
        if len(pairs) != len(run_names):
            print("Warning: some run names had no CSV under base_dir")
    else:
        pairs = _csv_paths_from_run_dirs(run_dirs)

    if not pairs:
        print("No CSV files found. Use --csv <path1> <path2> or --base-dir <dir> run1 run2 or run dirs.")
        return

    n_actions = len(BC_ACTION_NAMES)

    print("=" * 80)
    print("BC pretrain run comparison (by epoch)")
    print("Action names: 0=accel, 1=left+accel, 2=right+accel, 3=coast, 4=left, 5=right,")
    print("  6=brake, 7=left+brake, 8=right+brake, 9=accel+brake, 10=left+accel+brake, 11=right+accel+brake")
    print("=" * 80)

    runs: list[tuple[str, Path, list[dict[str, Any]], dict[str, Any] | None]] = []
    for name, csv_path in pairs:
        rows = _load_csv(csv_path)
        merged = _merge_epoch_rows(rows)
        run_dir = csv_path.parent.parent
        if "version" in csv_path.parts:
            run_dir = csv_path.parent.parent.parent
        meta = _load_meta(run_dir)
        runs.append((name, csv_path, merged, meta))

    # Epoch range common to all
    all_epochs = [set(row["epoch"] for row in m) for _, _, m, _ in runs]
    common_epochs = sorted(set.intersection(*all_epochs)) if all_epochs else []
    if interval_epochs > 1:
        check_epochs = [e for e in common_epochs if e % interval_epochs == 0 or e == common_epochs[-1]]
    else:
        check_epochs = common_epochs

    # ---- Summary from meta ----
    print("\n--- Run summary (from pretrain_meta.json when present) ---")
    for name, _, _, meta in runs:
        if meta:
            epochs_trained = meta.get("epochs_trained", "?")
            train_acc = meta.get("train_acc_final")
            val_acc = meta.get("val_acc_final")
            train_loss = meta.get("train_loss_final")
            val_loss = meta.get("val_loss_final")
            offsets = meta.get("bc_time_offsets_ms")
            off_s = f"  bc_time_offsets_ms={offsets}" if offsets else ""
            print(f"  {name}: epochs_trained={epochs_trained}  train_acc={train_acc}  val_acc={val_acc}  train_loss={train_loss}  val_loss={val_loss}{off_s}")
        else:
            print(f"  {name}: (no pretrain_meta.json)")

    # ---- Per-offset accuracy (multi-offset BC runs) ----
    def _offset_cols(merged: list[dict[str, Any]]) -> list[str]:
        cols = set()
        for row in merged:
            for k in row:
                if k.startswith("val_acc_offset") and row.get(k) is not None:
                    cols.add(k)
        return sorted(cols)

    has_offset = any(_offset_cols(merged) for _, _, merged, _ in runs)
    if has_offset:
        print("\n--- Per-offset validation accuracy (multi-offset BC) ---")
        for name, _, merged, _ in runs:
            cols = _offset_cols(merged)
            if not cols:
                continue
            print(f"  {name}:")
            last = merged[-1]
            for c in cols:
                v = last.get(c)
                if v is not None:
                    print(f"    {c} = {float(v):.4f}")
            best_row = min((r for r in merged if r.get("val_loss") is not None), key=lambda r: float(r["val_loss"]), default=None)
            if best_row and best_row != last:
                print(f"    (best epoch by val_loss={best_row.get('epoch')}):")
                for c in cols:
                    v = best_row.get(c)
                    if v is not None:
                        print(f"      {c} = {float(v):.4f}")
            print()

    # ---- Loss / overall acc at epoch checkpoints ----
    print("\n--- train_loss / val_loss / train_acc / val_acc at epoch checkpoints ---")
    print(f"  Checkpoint epochs: {check_epochs[:15]}{'...' if len(check_epochs) > 15 else ''}")
    for name, _, merged, _ in runs:
        print(f"\n  {name}:")
        for ep in check_epochs[:20]:
            row = next((r for r in merged if r.get("epoch") == ep), None)
            if row is None:
                continue
            tl = row.get("train_loss")
            vl = row.get("val_loss")
            ta = row.get("train_acc")
            va = row.get("val_acc")
            tl_s = f"{tl:.4f}" if tl is not None else "-"
            vl_s = f"{vl:.4f}" if vl is not None else "-"
            ta_s = f"{ta:.4f}" if ta is not None else "-"
            va_s = f"{va:.4f}" if va is not None else "-"
            print(f"    epoch {ep:2d}: train_loss={tl_s}  val_loss={vl_s}  train_acc={ta_s}  val_acc={va_s}")
        if len(check_epochs) > 20:
            print(f"    ... (and {len(check_epochs) - 20} more)")

    # ---- Combined analysis: best epoch by val_loss + loss and per-action acc at that epoch ----
    print("\n--- Combined analysis (loss + per-action accuracy) ---")
    print("  Best epoch = argmin val_loss. At that epoch: val_loss, val_acc, main_actions_val_acc (mean over accel, left+accel, right+accel, coast, left+accel+brake, right+accel+brake).\n")
    for name, _, merged, _ in runs:
        if not merged:
            continue
        best_row = min((r for r in merged if r.get("val_loss") is not None), key=lambda r: float(r["val_loss"]), default=None)
        if best_row is None:
            print(f"  {name}: no val_loss in CSV")
            continue
        ep = best_row.get("epoch", "?")
        vl = best_row.get("val_loss")
        va = best_row.get("val_acc")
        main_acc = _main_actions_val_acc(best_row)
        vl_s = f"{float(vl):.4f}" if vl is not None else "-"
        va_s = f"{float(va):.4f}" if va is not None else "-"
        main_s = f"{main_acc:.4f}" if main_acc is not None else "-"
        print(f"  {name}: best epoch (min val_loss) = {ep}  val_loss = {vl_s}  val_acc = {va_s}  main_actions_val_acc = {main_s}")
    print()

    # ---- Per-action (per-class) accuracy with names ----
    print("\n--- Per-action validation accuracy (val_acc_class_* with action names) ---")
    print("  Last epoch per run:\n")
    for name, _, merged, _ in runs:
        if not merged:
            continue
        last = merged[-1]
        print(f"  {name} (epoch {last.get('epoch', '?')}):")
        for c in range(n_actions):
            col = f"val_acc_class_{c}"
            val = last.get(col)
            label = BC_ACTION_NAMES[c] if c < len(BC_ACTION_NAMES) else str(c)
            if val is not None:
                print(f"    {c:2d} {label:20s} {val:.4f}")
            else:
                print(f"    {c:2d} {label:20s} -")
        print()

    # Same at a few epoch checkpoints (e.g. 0, 10, 24, 49)
    print("--- Per-action val_acc at selected epochs (first run only, same structure for others) ---")
    ref_name, _, ref_merged, _ = runs[0]
    for ep in [0, 5, 10, 15, 20, 24, 30, 40, 49]:
        row = next((r for r in ref_merged if r.get("epoch") == ep), None)
        if row is None:
            continue
        print(f"\n  {ref_name} at epoch {ep}:")
        for c in range(n_actions):
            col = f"val_acc_class_{c}"
            val = row.get(col)
            label = BC_ACTION_NAMES[c] if c < len(BC_ACTION_NAMES) else str(c)
            if val is not None:
                print(f"    {label:20s} {val:.4f}")
    print("\n" + "=" * 80)

    if plot and output_dir is not None:
        _plot_per_action_accuracy(runs, output_dir)


def _plot_per_action_accuracy(
    runs: list[tuple[str, Path, list[dict[str, Any]], dict[str, Any] | None]],
    output_dir: Path,
) -> None:
    """Plot val_acc_class_* vs epoch for each run; one subplot per run, 12 lines (actions) per subplot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed; skipping --plot")
        return
    n_actions = len(BC_ACTION_NAMES)
    n_runs = len(runs)
    if n_runs == 0:
        return
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5), squeeze=False)
    axes = axes[0]
    for idx, (name, _, merged, _) in enumerate(runs):
        ax = axes[idx] if n_runs > 1 else axes[0]
        epochs: list[int] = []
        for row in merged:
            ep = row.get("epoch")
            if ep is None:
                continue
            if row.get("val_loss") is None:
                continue
            epochs.append(int(ep) if isinstance(ep, (int, float)) else int(ep))
        # Sort by epoch and dedupe
        by_ep: dict[int, dict[str, Any]] = {}
        for row in merged:
            ep = row.get("epoch")
            if ep is None or row.get("val_loss") is None:
                continue
            ep = int(ep) if isinstance(ep, (int, float)) else int(ep)
            by_ep[ep] = row
        epochs = sorted(by_ep.keys())
        for c in range(n_actions):
            col = f"val_acc_class_{c}"
            ys = [by_ep[ep].get(col) for ep in epochs]
            ys = [float(y) if y is not None else float("nan") for y in ys]
            label = BC_ACTION_NAMES[c] if c < len(BC_ACTION_NAMES) else str(c)
            ax.plot(epochs, ys, label=label, alpha=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
        ax.set_title(name)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower left", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "exp_pretrain_bc_per_action_accuracy.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare BC pretrain runs from Lightning CSV (and optional TB); report per-action accuracy with names."
    )
    ap.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Run directories (e.g. output/ptretrain/bc/v1 output/ptretrain/bc/v1.1); CSV resolved inside each.",
    )
    ap.add_argument(
        "--csv",
        dest="csv_paths",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit CSV file paths (run name inferred from path).",
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base dir for run names; use with run names: --base-dir output/ptretrain/bc v1 v1.1",
    )
    ap.add_argument(
        "run_names",
        nargs="*",
        type=str,
        help="Run names when using --base-dir (e.g. v1 v1.1).",
    )
    ap.add_argument(
        "--interval",
        type=int,
        default=5,
        dest="interval_epochs",
        help="Epoch interval for loss/acc checkpoints (default 5).",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Plot per-action validation accuracy vs epoch; save JPG to --output-dir.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plot (e.g. docs/source/_static). Required if --plot.",
    )
    args = ap.parse_args()

    if args.plot and args.output_dir is None:
        ap.error("--output-dir is required when using --plot")

    run_analysis(
        run_dirs=args.run_dirs,
        csv_paths=args.csv_paths,
        base_dir=args.base_dir,
        run_names=args.run_names if args.base_dir else None,
        interval_epochs=args.interval_epochs,
        plot=args.plot,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
