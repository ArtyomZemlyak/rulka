"""
Unsupervised pretraining of the visual backbone (IQN img_head–compatible CNN) on frames.

Configuration is loaded from ``config_files/pretrain/vis/pretrain_config.yaml`` via pydantic-settings
(``PretrainConfig``).  CLI arguments override YAML values; env vars (``PRETRAIN_*``) sit
between the YAML and CLI in priority.

Priority (highest first):
  1. CLI arguments   (this script)
  2. PRETRAIN_*      env vars  (e.g.  set PRETRAIN_TASK=simclr)
  3. config_files/pretrain/vis/pretrain_config.yaml (project-level defaults)
  4. Field defaults  (in PretrainConfig schema)

Tasks: ae (autoencoder), vae (variational AE), simclr (contrastive).
Frameworks:
  native    — pure PyTorch (default, no extra deps).
  lightly   — Lightly SimCLR transforms/loss; task=simclr, n_stack=1 only (pip install lightly).
  lightning — PyTorch Lightning loop: AMP, grad clipping, early stopping, TensorBoard
              (pip install lightning).

Stacked frames (--n-stack N):
  --stack-mode channel  stack as N input channels; init_iqn_from_encoder.py auto-averages to 1-ch.
  --stack-mode concat   run 1-ch encoder per frame + fusion — saves IQN-compatible 1-ch encoder.

Output (in --output-dir):
  encoder.pt          encoder weights loadable into IQN_Network.img_head.
  pretrain_meta.json  full metadata and reproducibility record.
  metrics.csv         per-epoch loss history.
  tb/                 TensorBoard logs (Lightning path only).

Usage:
  # Use YAML defaults (ae, native, 50 epochs):
  python scripts/pretrain_visual_backbone.py --data-dir maps/img

  # Override specific fields:
  python scripts/pretrain_visual_backbone.py --data-dir maps/img --task simclr --epochs 100

  # Lightning + val split:
  python scripts/pretrain_visual_backbone.py --data-dir maps/img --task simclr \\
      --framework lightning --val-fraction 0.1

  # Use a custom config file:
  python scripts/pretrain_visual_backbone.py --config my_experiment.yaml --data-dir maps/img

  # Override via env var (PowerShell):
  $env:PRETRAIN_TASK="simclr"; $env:PRETRAIN_EPOCHS="100"
  python scripts/pretrain_visual_backbone.py --data-dir maps/img
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config_files.pretrain_schema import LightningConfig, PretrainConfig, load_pretrain_config
from trackmania_rl.pretrain.train import train_pretrain

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Unsupervised pretrain visual backbone (IQN img_head-compatible). "
            "Defaults come from config_files/pretrain/vis/pretrain_config.yaml."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Config file ---
    ap.add_argument(
        "--config", type=Path, default=None, metavar="YAML",
        help="Path to a YAML config file.  If given, it fully replaces config_files/pretrain/vis/pretrain_config.yaml "
             "(all CLI overrides still apply on top).",
    )

    # --- Data ---
    ap.add_argument("--data-dir", type=Path, default=None, dest="data_dir",
                    metavar="PATH",
                    help="Root of the frame tree (maps/img/<track_id>/<replay_name>/).")
    ap.add_argument("--output-dir", type=Path, default=None, dest="output_dir",
                    metavar="PATH",
                    help="Base output directory.  Each run creates output_dir/run_NNN/ inside.")
    ap.add_argument("--run-name", type=str, default=None, dest="run_name",
                    metavar="NAME",
                    help="Fixed run subdirectory name (e.g. ae_baseline).  "
                         "Default (null): auto-increment run_001, run_002, ...")

    # --- Task / framework ---
    ap.add_argument("--task", type=str, default=None, choices=["ae", "vae", "simclr"],
                    help="Pretraining objective.")
    ap.add_argument("--framework", type=str, default=None,
                    choices=["native", "lightly", "lightning"],
                    help="Training backend.")

    # --- Image ---
    ap.add_argument("--image-size", type=int, default=None, dest="image_size",
                    metavar="PX",
                    help="Square input resolution (must match IQN config).")
    ap.add_argument("--image-normalization", type=str, default=None, dest="image_normalization",
                    choices=["01", "iqn"],
                    help="01 = [0,1]; iqn = (x-0.5)/0.5 for IQN/BC transfer (default from config).")

    # --- Training ---
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--grad-clip", type=float, default=None, dest="grad_clip",
                    help="Gradient clipping for Lightning Trainer (0 = disabled).")

    # --- VAE ---
    ap.add_argument("--vae-latent", type=int, default=None, dest="vae_latent",
                    help="VAE latent dimension.")

    # --- SimCLR ---
    ap.add_argument("--proj-dim", type=int, default=None, dest="proj_dim",
                    help="SimCLR projection head output dimension.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="SimCLR NT-Xent temperature.")

    # --- Stacking ---
    ap.add_argument("--n-stack", type=int, default=None, dest="n_stack",
                    metavar="N",
                    help="Number of consecutive frames per sample.")
    ap.add_argument("--stack-mode", type=str, default=None, dest="stack_mode",
                    choices=["channel", "concat"],
                    help="'concat' saves IQN-compatible 1-ch encoder (recommended for n_stack>1).")

    # --- Split ---
    ap.add_argument("--val-fraction", type=float, default=None, dest="val_fraction",
                    metavar="F",
                    help="Track-level val split fraction (0 = no split).")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for the train/val split.")

    # --- Misc ---
    ap.add_argument("--kl-weight", type=float, default=None, dest="kl_weight",
                    help="VAE KL divergence coefficient β.")
    ap.add_argument("--no-tqdm", action="store_true",
                    help="Disable tqdm progress bars.")

    # --- Preprocessed data cache ---
    ap.add_argument(
        "--preprocess-cache-dir", type=Path, default=None, dest="preprocess_cache_dir",
        metavar="PATH",
        help=(
            "Directory for preprocessed frame cache (train.npy, val.npy, cache_meta.json). "
            "When set, the pipeline validates the cache and rebuilds it from --data-dir "
            "if stale.  null (default) = load raw images on-the-fly (original behaviour)."
        ),
    )
    ap.add_argument(
        "--cache-load-in-ram", action="store_true", dest="cache_load_in_ram",
        help=(
            "Load preprocessed cache arrays fully into RAM before training.  "
            "Speeds up random I/O on small datasets; avoid on datasets larger than RAM."
        ),
    )
    ap.add_argument(
        "--cache-build-workers", type=int, default=None, dest="cache_build_workers",
        metavar="N",
        help=(
            "Threads for parallel frame loading during cache construction "
            "(0 = single-threaded, safe default)."
        ),
    )

    # --- Lightning settings (override config_files/pretrain/vis/pretrain_config.yaml lightning: section) ---
    lg = ap.add_argument_group(
        "Lightning settings",
        "Applied only when --framework lightning.  "
        "Override YAML 'lightning:' section.  "
        "Env var equivalent: PRETRAIN_LIGHTNING__<FIELD> (e.g. PRETRAIN_LIGHTNING__PATIENCE=5).",
    )
    lg.add_argument("--no-tensorboard", action="store_true",
                    help="Disable TensorBoard logger.")
    lg.add_argument("--no-csv-logger", action="store_true",
                    help="Disable CSV metrics logger.")
    lg.add_argument("--tensorboard-dir", type=str, default=None, dest="lg_tensorboard_dir",
                    metavar="DIR",
                    help="TensorBoard subdirectory inside output_dir (default: 'tb').")
    lg.add_argument("--csv-dir", type=str, default=None, dest="lg_csv_dir",
                    metavar="DIR",
                    help="CSV log subdirectory inside output_dir (default: 'csv').")
    lg.add_argument("--checkpoint-dir", type=str, default=None, dest="lg_checkpoint_dir",
                    metavar="DIR",
                    help="Checkpoint subdirectory inside output_dir (default: 'checkpoints').")
    lg.add_argument("--save-top-k", type=int, default=None, dest="lg_save_top_k",
                    metavar="K",
                    help="Checkpoints to keep: -1=all, 0=none, 1=best (default: 1).")
    lg.add_argument("--no-early-stopping", action="store_true",
                    help="Disable EarlyStopping callback.")
    lg.add_argument("--patience", type=int, default=None, dest="lg_patience",
                    help="EarlyStopping patience in epochs (default: 10).")
    lg.add_argument("--accelerator", type=str, default=None, dest="lg_accelerator",
                    choices=["auto", "cpu", "gpu", "tpu", "mps"],
                    help="Lightning accelerator (default: auto).")
    lg.add_argument("--precision", type=str, default=None, dest="lg_precision",
                    choices=["auto", "32-true", "16-mixed", "bf16-mixed"],
                    help="Training precision (default: auto → 16-mixed if CUDA else 32-true).")
    lg.add_argument("--log-every-n-steps", type=int, default=None, dest="lg_log_every_n_steps",
                    metavar="N",
                    help="Log metrics every N steps (default: 50).")
    lg.add_argument("--deterministic", action="store_true",
                    help="Use deterministic algorithms (reproducible but slower).")
    lg.add_argument("--profiler", type=str, default=None, dest="lg_profiler",
                    choices=["simple", "advanced", "pytorch"],
                    help="Enable Lightning profiler.")

    return ap


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load base config (YAML → env vars → field defaults)
    # -----------------------------------------------------------------------
    if args.config:
        # Custom YAML replaces config_files/pretrain/vis/pretrain_config.yaml entirely
        base_cfg = load_pretrain_config(args.config)
        log.info("Loaded config from %s", args.config)
    else:
        # Default: reads config_files/pretrain/vis/pretrain_config.yaml + PRETRAIN_* env vars
        base_cfg = PretrainConfig()
        log.info("Loaded config from config_files/pretrain/vis/pretrain_config.yaml")

    # -----------------------------------------------------------------------
    # 2. Collect explicit CLI overrides (None = "not set on CLI")
    # -----------------------------------------------------------------------
    overrides: dict = {}
    for field in (
        "data_dir", "output_dir", "run_name", "task", "framework", "image_size", "image_normalization",
        "epochs", "batch_size", "lr", "workers", "grad_clip",
        "vae_latent", "proj_dim", "temperature", "n_stack", "stack_mode",
        "val_fraction", "seed", "kl_weight",
        "preprocess_cache_dir", "cache_build_workers",
    ):
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val

    if args.no_tqdm:
        overrides["use_tqdm"] = False
    if args.cache_load_in_ram:
        overrides["cache_load_in_ram"] = True

    # --- Lightning sub-config overrides ---
    lightning_overrides: dict = {}
    if args.no_tensorboard:
        lightning_overrides["tensorboard"] = False
    if args.no_csv_logger:
        lightning_overrides["csv_logger"] = False
    if args.no_early_stopping:
        lightning_overrides["early_stopping"] = False
    if args.deterministic:
        lightning_overrides["deterministic"] = True
    # Renamed args (prefixed with lg_ to avoid collision with top-level fields)
    for src, dst in [
        ("lg_tensorboard_dir", "tensorboard_dir"),
        ("lg_csv_dir", "csv_dir"),
        ("lg_checkpoint_dir", "checkpoint_dir"),
        ("lg_save_top_k", "save_top_k"),
        ("lg_patience", "patience"),
        ("lg_accelerator", "accelerator"),
        ("lg_precision", "precision"),
        ("lg_log_every_n_steps", "log_every_n_steps"),
        ("lg_profiler", "profiler"),
    ]:
        val = getattr(args, src, None)
        if val is not None:
            lightning_overrides[dst] = val

    if lightning_overrides:
        # Merge overrides into existing lightning config and produce a validated LightningConfig
        base_lightning = base_cfg.lightning.model_dump()
        base_lightning.update(lightning_overrides)
        overrides["lightning"] = LightningConfig.model_validate(base_lightning)

    # -----------------------------------------------------------------------
    # 3. Apply overrides on top of base config (re-validates)
    # -----------------------------------------------------------------------
    if overrides:
        cfg = base_cfg.model_copy(update=overrides)
        # Re-run cross-field validators
        cfg = PretrainConfig.model_validate(cfg.model_dump())
    else:
        cfg = base_cfg

    # -----------------------------------------------------------------------
    # 4. Diagnostics
    # -----------------------------------------------------------------------
    log.info(
        "Pretrain config: task=%s  framework=%s  image_size=%d  epochs=%d  "
        "batch_size=%d  lr=%g  n_stack=%d  stack_mode=%s  val_fraction=%g",
        cfg.task, cfg.framework, cfg.image_size, cfg.epochs,
        cfg.batch_size, cfg.lr, cfg.n_stack, cfg.stack_mode, cfg.val_fraction,
    )
    log.info("data_dir=%s  →  output_dir=%s", cfg.data_dir, cfg.output_dir)

    # -----------------------------------------------------------------------
    # 5. Run
    # -----------------------------------------------------------------------
    run_dir = train_pretrain(cfg)

    # -----------------------------------------------------------------------
    # 6. Post-run hints
    # -----------------------------------------------------------------------
    encoder_pt = run_dir / "encoder.pt"
    log.info("Run directory : %s", run_dir)
    log.info("Saved encoder → %s", encoder_pt)
    if cfg.n_stack > 1 and cfg.stack_mode == "channel":
        log.info(
            "Encoder has %d input channels.  "
            "init_iqn_from_encoder.py will auto-average first layer to 1-ch.",
            cfg.n_stack,
        )
    log.info(
        "Next step: python scripts/init_iqn_from_encoder.py "
        "--encoder-pt %s --save-dir save/",
        encoder_pt,
    )


if __name__ == "__main__":
    main()
