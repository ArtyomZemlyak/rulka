"""
Level 1 BC (behavioral cloning) pretraining: frames + manifest action_idx → encoder.pt.

Configuration from config_files/pretrain/bc/pretrain_config_bc.yaml and PRETRAIN_BC_* env vars.
CLI overrides take priority.

Usage:
  python scripts/pretrain_bc.py --data-dir maps/img
  python scripts/pretrain_bc.py --config my_bc.yaml --epochs 20
  python scripts/pretrain_bc.py --data-dir maps/img --encoder-init-path output/ptretrain/vis/run_001/encoder.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config_files.pretrain_bc_schema import BCPretrainConfig, load_pretrain_bc_config
from trackmania_rl.pretrain.train_bc import train_bc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="BC pretrain: frames + manifest action_idx → encoder.pt (IQN img_head compatible).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=None, help="YAML config (replaces pretrain_config_bc.yaml).")
    ap.add_argument("--data-dir", type=Path, default=None, dest="data_dir", help="Root of track_id/replay_name/ with manifest.json.")
    ap.add_argument("--output-dir", type=Path, default=None, dest="output_dir", help="Base output directory.")
    ap.add_argument("--run-name", type=str, default=None, dest="run_name", help="Run subdirectory name (null = run_001, run_002, ...).")
    ap.add_argument("--bc-mode", type=str, default=None, dest="bc_mode", choices=["backbone", "full_policy", "auxiliary_head"],
                    help="backbone | full_policy | auxiliary_head.")
    ap.add_argument("--encoder-init-path", type=Path, default=None, dest="encoder_init_path",
                    help="Path to Level 0 encoder.pt to init BC CNN.")
    ap.add_argument("--image-size", type=int, default=None, dest="image_size")
    ap.add_argument("--n-stack", type=int, default=None, dest="n_stack")
    ap.add_argument("--n-actions", type=int, default=None, dest="n_actions", help="Number of actions (match RL config.inputs).")
    ap.add_argument("--image-normalization", type=str, default=None, dest="image_normalization", choices=["01", "iqn"],
                    help="01 = [0,1]; iqn = (x-0.5)/0.5 for IQN transfer.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None, dest="val_fraction")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")
    ap.add_argument("--pin-memory", action="store_true", default=None, dest="pin_memory", help="Pin memory for DataLoader (default from config).")
    ap.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    ap.add_argument("--prefetch-factor", type=int, default=None, dest="prefetch_factor", help="DataLoader prefetch factor.")
    ap.add_argument("--use-tqdm", action="store_true", default=None, dest="use_tqdm", help="Show progress bar (default from config).")
    ap.add_argument("--no-tqdm", action="store_false", dest="use_tqdm")
    ap.add_argument("--preprocess-cache-dir", type=Path, default=None, dest="preprocess_cache_dir",
                    help="BC cache directory (train.npy + train_actions.npy). null = on-the-fly from data_dir.")
    ap.add_argument("--cache-load-in-ram", action="store_true", dest="cache_load_in_ram")
    ap.add_argument("--cache-build-workers", type=int, default=None, dest="cache_build_workers")
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    if args.config:
        cfg = load_pretrain_bc_config(args.config)
        log.info("Loaded config from %s", args.config)
    else:
        cfg = BCPretrainConfig()
        log.info("Loaded config from config_files/pretrain/bc/pretrain_config_bc.yaml")

    overrides = {}
    for field in (
        "data_dir", "output_dir", "run_name", "bc_mode", "encoder_init_path",
        "image_size", "n_stack", "n_actions", "image_normalization", "epochs", "batch_size", "lr",
        "workers", "val_fraction", "seed", "grad_clip", "prefetch_factor",
        "preprocess_cache_dir", "cache_build_workers",
    ):
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    if hasattr(args, "pin_memory") and args.pin_memory is not None:
        overrides["pin_memory"] = args.pin_memory
    if hasattr(args, "use_tqdm") and args.use_tqdm is not None:
        overrides["use_tqdm"] = args.use_tqdm
    if args.cache_load_in_ram:
        overrides["cache_load_in_ram"] = True

    if overrides:
        cfg = cfg.model_copy(update=overrides)

    log.info(
        "BC config: bc_mode=%s  image_size=%d  n_stack=%d  n_actions=%d  epochs=%d  batch_size=%d",
        cfg.bc_mode, cfg.image_size, cfg.n_stack, cfg.n_actions, cfg.epochs, cfg.batch_size,
    )
    log.info("data_dir=%s  output_dir=%s", cfg.data_dir, cfg.output_dir)

    run_dir = train_bc(cfg)

    encoder_pt = run_dir / "encoder.pt"
    log.info("Run directory : %s", run_dir)
    log.info("Saved encoder → %s", encoder_pt)
    log.info(
        "Next: set pretrain_encoder_path in RL config to %s (or use init_iqn_from_encoder.py)",
        encoder_pt,
    )


if __name__ == "__main__":
    main()
