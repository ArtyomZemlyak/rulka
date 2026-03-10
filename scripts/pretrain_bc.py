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
    ap.add_argument("--rl-config-path", type=Path, default=None, dest="rl_config_path",
                    help="Path to RL config; image_size, n_actions, etc. loaded from it.")
    ap.add_argument("--n-stack", type=int, default=None, dest="n_stack")
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
    ap.add_argument("--track-ids", type=str, nargs="+", default=None, dest="track_ids",
                    help="Restrict to these track IDs only (e.g. --track-ids A01-Race). Uses on-the-fly loading.")
    ap.add_argument("--bc-resume-run-dir", type=Path, default=None, dest="bc_resume_run_dir",
                    help="Path to previous BC run dir (e.g. output/ptretrain/bc/v3_multi_offset) to load iqn_bc.pt for fine-tuning.")
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
        "data_dir", "output_dir", "run_name", "bc_mode", "encoder_init_path", "rl_config_path",
        "n_stack", "image_normalization", "epochs", "batch_size", "lr",
        "workers", "val_fraction", "seed", "grad_clip", "prefetch_factor",
        "preprocess_cache_dir", "cache_build_workers", "track_ids", "bc_resume_run_dir",
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

    # Initialize RL config before importing train_bc (iqn.py calls get_config() at import time)
    from config_files.config_loader import load_config, set_config
    rl_path = Path(cfg.rl_config_path).resolve()
    if not rl_path.exists():
        raise FileNotFoundError(f"rl_config_path {rl_path} not found")
    set_config(load_config(rl_path))

    from trackmania_rl.pretrain.train_bc import train_bc

    log.info(
        "BC config: bc_mode=%s  n_stack=%d  epochs=%d  batch_size=%d  rl_config=%s",
        cfg.bc_mode, cfg.n_stack, cfg.epochs, cfg.batch_size, cfg.rl_config_path,
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
