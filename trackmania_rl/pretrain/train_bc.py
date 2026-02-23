"""
Level 1 BC (behavioral cloning) training.

Single entry: train_bc(cfg). Uses PyTorch Lightning only. BCPretrainConfig,
CachedBCDataModule or BCReplayDataModule, BCLightningModule, create_lightning_trainer,
save_encoder_artifact. Reuses Lightning setup from Level 0 (train.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from trackmania_rl.pretrain.export import load_encoder_into_bc, save_encoder_artifact
from trackmania_rl.pretrain.models import build_bc_network, get_enc_dim
from trackmania_rl.pretrain.preprocess import build_bc_cache, is_bc_cache_valid
from trackmania_rl.pretrain.train import create_lightning_trainer

log = logging.getLogger(__name__)


def _resolve_run_dir(base_dir: Path, run_name: str | None) -> Path:
    """Return versioned run directory: base_dir/run_name or base_dir/run_001, run_002, ..."""
    if run_name:
        return base_dir / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        candidate = base_dir / f"run_{i:03d}"
        if not candidate.exists():
            return candidate
        i += 1


def train_bc(cfg) -> Path:
    """Run BC pretraining with PyTorch Lightning.

    Cache check/build, then Lightning DataModule + BCLightningModule,
    create_lightning_trainer (shared with Level 0), fit, save encoder.pt.

    cfg: BCPretrainConfig (from config_files.pretrain_bc_schema).
    Returns run_dir (where encoder.pt and pretrain_meta.json are written).
    """
    run_dir = _resolve_run_dir(Path(cfg.output_dir), cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("BC run directory: %s", run_dir)

    log.info("Training on device: %s", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Unlock Tensor Cores on Ampere / Ada / Blackwell GPUs (same as Level 0)
    prec = cfg.lightning.float32_matmul_precision
    if prec is not None:
        torch.set_float32_matmul_precision(prec)
        log.info("torch.set_float32_matmul_precision('%s')", prec)

    data_dir = Path(cfg.data_dir).resolve()
    cache_dir = Path(cfg.preprocess_cache_dir) if cfg.preprocess_cache_dir else None

    if cache_dir is not None:
        if not is_bc_cache_valid(
            cache_dir,
            data_dir,
            cfg.image_size,
            cfg.n_stack,
            cfg.val_fraction,
            cfg.seed,
            cfg.n_actions,
        ):
            log.info("BC cache not found.")
            log.info("Building BC cache at %s ...", cache_dir)
            build_bc_cache(
                data_dir=data_dir,
                cache_dir=cache_dir,
                image_size=cfg.image_size,
                n_stack=cfg.n_stack,
                val_fraction=cfg.val_fraction,
                seed=cfg.seed,
                n_actions=cfg.n_actions,
                workers=cfg.cache_build_workers,
            )

    enc_dim = get_enc_dim(1, cfg.image_size)
    model = build_bc_network(
        enc_dim=enc_dim,
        n_actions=cfg.n_actions,
        use_floats=cfg.use_floats,
        float_dim=0,
        in_channels=1,
        image_size=cfg.image_size,
    )

    if cfg.encoder_init_path and Path(cfg.encoder_init_path).exists():
        load_encoder_into_bc(Path(cfg.encoder_init_path), model)

    from trackmania_rl.pretrain.tasks import BCLightningModule
    from trackmania_rl.pretrain.datasets import CachedBCDataModule, BCReplayDataModule

    bc_module = BCLightningModule(model, lr=cfg.lr)

    if cache_dir is not None:
        log.info("BC DataModule: using preprocessed cache at %s", cache_dir)
        data_module = CachedBCDataModule(
            cache_dir=cache_dir,
            batch_size=cfg.batch_size,
            workers=cfg.workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            load_in_ram=cfg.cache_load_in_ram,
            expected_image_size=cfg.image_size,
            expected_n_stack=cfg.n_stack,
        )
    else:
        data_module = BCReplayDataModule(
            data_dir=data_dir,
            image_size=cfg.image_size,
            n_stack=cfg.n_stack,
            batch_size=cfg.batch_size,
            workers=cfg.workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )

    data_module.setup()
    n_train = data_module.n_train_samples
    n_val = data_module.n_val_samples

    n_train_batches = len(data_module.train_dataloader())
    trainer, metrics_cb, _ = create_lightning_trainer(
        cfg.lightning,
        run_dir,
        max_epochs=cfg.epochs,
        grad_clip=cfg.grad_clip,
        use_tqdm=cfg.use_tqdm,
        has_val=data_module.has_val,
        n_train_batches=n_train_batches,
    )

    try:
        trainer.fit(bc_module, datamodule=data_module)
    except KeyboardInterrupt:
        log.warning(
            "Training interrupted by user (Ctrl+C). "
            "%d epochs completed — saving partial encoder artifact.",
            len(metrics_cb.rows),
        )

    # Build metrics_rows: all callback metrics (loss, overall acc, per-class acc)
    metrics_rows = []
    for row in metrics_cb.rows:
        out = {k: v for k, v in row.items()}
        out["epoch"] = row.get("epoch", 0) + 1
        metrics_rows.append(out)

    backbone = bc_module.encoder
    meta = {
        "task": "bc",
        "framework": "lightning",
        "bc_mode": cfg.bc_mode,
        "image_size": cfg.image_size,
        "n_stack": cfg.n_stack,
        "in_channels": 1,
        "enc_dim": enc_dim,
        "n_actions": cfg.n_actions,
        "encoder_init_path": str(cfg.encoder_init_path) if cfg.encoder_init_path else None,
        "epochs_trained": cfg.epochs,
        "train_loss_final": metrics_rows[-1].get("train_loss") if metrics_rows else None,
        "val_loss_final": metrics_rows[-1].get("val_loss") if metrics_rows else None,
        "train_acc_final": metrics_rows[-1].get("train_acc") if metrics_rows else None,
        "val_acc_final": metrics_rows[-1].get("val_acc") if metrics_rows else None,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "seed": cfg.seed,
    }
    save_encoder_artifact(backbone, meta, run_dir, metrics_rows=metrics_rows)
    log.info("BC pretrain complete. Artifact: %s", run_dir)
    return run_dir
