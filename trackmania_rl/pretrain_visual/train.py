"""
Level 0 visual pretraining — unified entry point.

Configuration is loaded from ``config_files/pretrain_config.yaml`` via
``PretrainConfig`` (pydantic-settings).  CLI overrides, env vars (``PRETRAIN_*``),
and programmatic construction all have higher priority than the YAML file.

Usage
-----
    from config_files.pretrain_schema import PretrainConfig
    from trackmania_rl.pretrain_visual import train_pretrain
    from pathlib import Path

    # Defaults from pretrain_config.yaml:
    cfg = PretrainConfig()

    # Override specific fields:
    cfg = PretrainConfig(
        data_dir=Path("maps/img"),
        output_dir=Path("pretrain_visual_out"),
        task="ae",
        framework="native",
        epochs=50,
    )
    train_pretrain(cfg)

After training, ``cfg.output_dir`` contains:
    encoder.pt          — IQN-compatible backbone weights
    pretrain_meta.json  — full metadata and reproducibility record
    metrics.csv         — per-epoch loss history
    tb/                 — TensorBoard logs (Lightning path only)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# PretrainConfig is defined in config_files (pydantic-settings + YAML)
from config_files.pretrain_schema import PretrainConfig  # re-exported from __init__ too

from trackmania_rl.pretrain_visual.models import (
    StackedEncoderConcat,
    build_ae_decoder,
    build_iqn_encoder,
    build_simclr_projection,
    build_vae_decoder,
    build_vae_head,
    get_enc_dim,
)
from trackmania_rl.pretrain_visual.export import save_encoder_artifact

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run-directory versioning
# ---------------------------------------------------------------------------

def _resolve_run_dir(base_dir: Path, run_name: str | None) -> Path:
    """Return the versioned run directory inside *base_dir*.

    If *run_name* is given, returns ``base_dir / run_name`` (caller chooses the name).
    Otherwise scans for the first non-existing ``run_001``, ``run_002``, ... slot.
    """
    if run_name:
        return base_dir / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        candidate = base_dir / f"run_{i:03d}"
        if not candidate.exists():
            return candidate
        i += 1


# ---------------------------------------------------------------------------
# Native training helpers
# ---------------------------------------------------------------------------

def _make_encoder_and_loader(
    cfg: PretrainConfig,
    device: torch.device,
    cache_dir: Optional[Path] = None,
) -> tuple[nn.Module, DataLoader, int, int, bool, bool]:
    """Build encoder + DataLoader for the native training path.

    When *cache_dir* is given the loader is backed by a pre-processed
    ``CachedPretrainDataset`` (fast mmap / RAM reads) instead of loading raw
    images.  The Lightly SimCLR path is skipped when using a cache because the
    data is already decoded and stored as grayscale tensors; native augmentation
    (``_augment``) is used instead.

    Returns
    -------
    encoder, loader, enc_dim, in_channels, stacked_concat, lightly_available
    """
    from trackmania_rl.pretrain_visual.datasets import (
        FlatFrameDataset,
        ReplayFrameDataset,
        CachedPretrainDataset,
        split_track_ids,
    )
    from trackmania_rl.pretrain_visual.preprocess import CACHE_TRAIN_FILE

    single_enc_dim = get_enc_dim(1, cfg.image_size)

    if cfg.n_stack > 1:
        if cfg.stack_mode == "channel":
            in_channels = cfg.n_stack
            stacked_concat = False
            encoder = build_iqn_encoder(in_channels, cfg.image_size).to(device)
        else:  # concat
            in_channels = 1
            stacked_concat = True
            enc1 = build_iqn_encoder(1, cfg.image_size)
            encoder = StackedEncoderConcat(enc1, cfg.n_stack, single_enc_dim).to(device)
    else:
        in_channels = 1
        stacked_concat = False
        encoder = build_iqn_encoder(1, cfg.image_size).to(device)

    # --- Dataset selection ---
    if cache_dir is not None:
        # Fast path: pre-processed cache.
        # Lightly is not applicable here (tensors, not PIL images), so lightly_ok = False.
        ds = CachedPretrainDataset(
            cache_dir / CACHE_TRAIN_FILE,
            load_in_ram=cfg.cache_load_in_ram,
            expected_image_size=cfg.image_size,
            expected_n_stack=cfg.n_stack,
        )
        if len(ds) == 0:
            raise RuntimeError(f"Cached train.npy is empty: {cache_dir / CACHE_TRAIN_FILE}")
        log.info(
            "Dataset (cached): %d samples  cache_dir=%s  load_in_ram=%s",
            len(ds), cache_dir, cfg.cache_load_in_ram,
        )
        lightly_ok = False
    else:
        # Raw image tree path (original behaviour).
        if cfg.val_fraction > 0:
            train_ids, _ = split_track_ids(cfg.data_dir, cfg.val_fraction, cfg.seed)
            ds = ReplayFrameDataset(cfg.data_dir, track_ids=train_ids, size=cfg.image_size, n_stack=cfg.n_stack)
        else:
            ds = FlatFrameDataset(cfg.data_dir, size=cfg.image_size, n_stack=cfg.n_stack)

        if len(ds) == 0:
            raise RuntimeError(f"No images found in {cfg.data_dir}")
        log.info("Dataset: %d samples (n_stack=%d, framework=%s)", len(ds), cfg.n_stack, cfg.framework)

        lightly_ok = False
        if cfg.framework == "lightly" and cfg.task == "simclr" and cfg.n_stack == 1:
            try:
                from lightly.transforms.simclr_transform import SimCLRTransform  # noqa: F401
                lightly_ok = True
            except ImportError:
                log.info("lightly not installed; falling back to native SimCLR augmentation.")

        if lightly_ok:
            from lightly.transforms.simclr_transform import SimCLRTransform
            from torch.utils.data import Dataset as TorchDS

            transform = SimCLRTransform(input_size=cfg.image_size, gaussian_blur=0.0)

            class _LightlyDS(TorchDS):
                def __init__(self, root: Path, transform_: object) -> None:
                    self.paths = sorted(
                        p for p in Path(root).rglob("*")
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    )
                    self.transform_ = transform_

                def __len__(self) -> int:
                    return len(self.paths)

                def __getitem__(self, idx: int):
                    from PIL import Image
                    img = Image.open(self.paths[idx]).convert("RGB")
                    return self.transform_(img), self.transform_(img)

            ds = _LightlyDS(cfg.data_dir, transform)

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        drop_last=(cfg.task == "simclr"),
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.workers > 0),
        prefetch_factor=cfg.prefetch_factor if cfg.workers > 0 else None,
    )

    enc_dim = single_enc_dim
    return encoder, loader, enc_dim, in_channels, stacked_concat, lightly_ok


def _train_ae_native(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    stacked_concat: bool,
    use_tqdm: bool,
) -> list[dict]:
    """Train autoencoder natively.  Returns per-epoch metric rows."""
    try:
        from tqdm import tqdm as _tqdm
        _tqdm_ok = True
    except ImportError:
        _tqdm_ok = False

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    rows: list[dict] = []

    try:
        for epoch in range(epochs):
            encoder.train(); decoder.train()
            total, n = 0.0, 0
            it = _tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") if (use_tqdm and _tqdm_ok) else loader
            for batch in it:
                x = batch.to(device)
                if x.dim() == 5 and not stacked_concat:
                    x = x.squeeze(2)
                opt.zero_grad()
                z = encoder(x)
                x_recon = decoder(z)
                target = x if x.dim() == 4 else x.squeeze(2)
                loss = nn.functional.mse_loss(x_recon, target)
                loss.backward()
                opt.step()
                total += loss.item() * x.size(0)
                n += x.size(0)
            avg = total / max(n, 1)
            log.info("AE epoch %d/%d  loss=%.6f", epoch + 1, epochs, avg)
            rows.append({"epoch": epoch + 1, "train_loss": avg})
    except KeyboardInterrupt:
        log.warning("Training interrupted at epoch %d/%d — saving partial artifact.", len(rows), epochs)

    return rows


def _train_vae_native(
    encoder: nn.Module,
    vae_head: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    kl_weight: float,
    stacked_concat: bool,
    use_tqdm: bool,
) -> list[dict]:
    """Train VAE natively.  Returns per-epoch metric rows."""
    try:
        from tqdm import tqdm as _tqdm
        _tqdm_ok = True
    except ImportError:
        _tqdm_ok = False

    params = list(encoder.parameters()) + list(vae_head.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    rows: list[dict] = []

    try:
        for epoch in range(epochs):
            encoder.train(); vae_head.train(); decoder.train()
            total, n = 0.0, 0
            it = _tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") if (use_tqdm and _tqdm_ok) else loader
            for batch in it:
                x = batch.to(device)
                if x.dim() == 5 and not stacked_concat:
                    x = x.squeeze(2)
                opt.zero_grad()
                feat = encoder(x)
                out = vae_head(feat)
                mu, logvar = out.chunk(2, dim=1)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                x_recon = decoder(z)
                target = x.squeeze(2) if x.dim() == 5 else x
                recon = nn.functional.mse_loss(x_recon, target)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
                loss = recon + kl_weight * kl
                loss.backward()
                opt.step()
                total += loss.item() * x.size(0)
                n += x.size(0)
            avg = total / max(n, 1)
            log.info("VAE epoch %d/%d  loss=%.6f", epoch + 1, epochs, avg)
            rows.append({"epoch": epoch + 1, "train_loss": avg})
    except KeyboardInterrupt:
        log.warning("Training interrupted at epoch %d/%d — saving partial artifact.", len(rows), epochs)

    return rows


def _train_simclr_native(
    encoder: nn.Module,
    proj_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    temperature: float,
    stacked_concat: bool,
    use_tqdm: bool,
    lightly_available: bool,
) -> list[dict]:
    """Train SimCLR natively.  Returns per-epoch metric rows."""
    try:
        from tqdm import tqdm as _tqdm
        _tqdm_ok = True
    except ImportError:
        _tqdm_ok = False

    from trackmania_rl.pretrain_visual.tasks import _augment, _nt_xent

    criterion = None
    if lightly_available:
        try:
            from lightly.loss import NTXentLoss
            criterion = NTXentLoss(temperature=temperature)
        except ImportError:
            pass

    opt = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=lr)
    rows: list[dict] = []

    try:
        for epoch in range(epochs):
            encoder.train(); proj_head.train()
            total, n = 0.0, 0
            it = _tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") if (use_tqdm and _tqdm_ok) else loader
            for batch in it:
                if isinstance(batch, (list, tuple)):
                    x1_raw, x2_raw = batch
                    x1 = x1_raw.to(device); x2 = x2_raw.to(device)
                    if x1.shape[1] == 3:
                        x1 = x1[:, :1]; x2 = x2[:, :1]
                else:
                    x = batch.to(device)
                    if x.dim() == 5 and not stacked_concat:
                        x = x.squeeze(2)
                    x1, x2 = _augment(x), _augment(x)

                opt.zero_grad()
                z1 = proj_head(encoder(x1))
                z2 = proj_head(encoder(x2))

                if criterion is not None:
                    loss = criterion(torch.cat([z1, z2], dim=0))
                else:
                    loss = _nt_xent(z1, z2, temperature)

                loss.backward()
                opt.step()
                total += loss.item() * x1.size(0)
                n += x1.size(0)
            avg = total / max(n, 1)
            log.info("SimCLR epoch %d/%d  loss=%.6f", epoch + 1, epochs, avg)
            rows.append({"epoch": epoch + 1, "train_loss": avg})
    except KeyboardInterrupt:
        log.warning("Training interrupted at epoch %d/%d — saving partial artifact.", len(rows), epochs)

    return rows


# ---------------------------------------------------------------------------
# Lightning training path
# ---------------------------------------------------------------------------

def _train_lightning(
    cfg: PretrainConfig,
    encoder: nn.Module,
    in_channels: int,
    stacked_concat: bool,
    run_dir: Path,
    cache_dir: Optional[Path] = None,
) -> tuple[list[dict], int, int]:
    """Train using PyTorch Lightning.

    When *cache_dir* is provided, ``CachedPretrainDataModule`` is used instead
    of ``ReplayFrameDataModule``.  All trainer/logger/callback options come
    from ``cfg.lightning`` (LightningConfig).

    Returns
    -------
    (metrics_rows, n_train_samples, n_val_samples)
    """
    try:
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
    except ImportError as exc:
        raise ImportError(
            "PyTorch Lightning is required for framework='lightning'. "
            "Install it with: pip install lightning"
        ) from exc

    from trackmania_rl.pretrain_visual.tasks import (
        AELightningModule,
        VAELightningModule,
        SimCLRLightningModule,
        MetricsCollector,
    )
    from trackmania_rl.pretrain_visual.datasets import (
        ReplayFrameDataModule,
        CachedPretrainDataModule,
    )

    lc = cfg.lightning  # shorthand

    enc_dim = get_enc_dim(1, cfg.image_size)

    # --- Task module ---
    if cfg.task == "ae":
        decoder = build_ae_decoder(enc_dim, out_channels=cfg.n_stack if cfg.n_stack > 1 else 1, image_size=cfg.image_size)
        module = AELightningModule(encoder, decoder, cfg.lr, stacked_concat)
    elif cfg.task == "vae":
        vae_head = build_vae_head(enc_dim, cfg.vae_latent)
        decoder = build_vae_decoder(cfg.vae_latent, out_channels=cfg.n_stack if cfg.n_stack > 1 else 1, image_size=cfg.image_size)
        module = VAELightningModule(encoder, vae_head, decoder, cfg.lr, cfg.kl_weight, stacked_concat)
    elif cfg.task == "simclr":
        proj = build_simclr_projection(enc_dim, cfg.proj_dim)
        module = SimCLRLightningModule(encoder, proj, cfg.lr, cfg.temperature, stacked_concat)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    # --- DataModule ---
    if cache_dir is not None:
        log.info("Lightning DataModule: using preprocessed cache at %s", cache_dir)
        data_module = CachedPretrainDataModule(
            cache_dir=cache_dir,
            batch_size=cfg.batch_size,
            workers=cfg.workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            task=cfg.task,
            load_in_ram=cfg.cache_load_in_ram,
            expected_image_size=cfg.image_size,
            expected_n_stack=cfg.n_stack,
        )
    else:
        data_module = ReplayFrameDataModule(
            data_dir=cfg.data_dir,
            image_size=cfg.image_size,
            n_stack=cfg.n_stack,
            batch_size=cfg.batch_size,
            workers=cfg.workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
            task=cfg.task,
        )
    data_module.setup()

    # --- Resolve monitored metric ---
    if lc.checkpoint_monitor == "auto":
        monitor_key = "val_loss" if data_module.has_val else "train_loss"
    else:
        monitor_key = lc.checkpoint_monitor

    # --- Callbacks ---
    metrics_cb = MetricsCollector()
    callbacks = [metrics_cb]

    if lc.save_top_k != 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(run_dir / lc.checkpoint_dir),
                save_top_k=lc.save_top_k,
                monitor=monitor_key,
                mode="min",
                filename="best-{epoch:03d}-{" + monitor_key + ":.4f}",
            )
        )

    if lc.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor_key,
                patience=lc.patience,
                min_delta=lc.min_delta,
                mode="min",
                verbose=True,
            )
        )

    # --- Loggers ---
    # version="" disables Lightning's own version_N subdirectory so that logs land
    # directly in run_dir/tensorboard/ and run_dir/csv/ (no extra nesting).
    loggers = []
    if lc.tensorboard:
        loggers.append(
            TensorBoardLogger(
                save_dir=str(run_dir),
                name=lc.tensorboard_dir,
                version="",
            )
        )
    if lc.csv_logger:
        loggers.append(
            CSVLogger(
                save_dir=str(run_dir),
                name=lc.csv_dir,
                version="",
            )
        )
    if not loggers:
        loggers = False  # type: ignore[assignment]  # Lightning accepts False = no logger

    # --- Precision ---
    precision = lc.precision
    if precision == "auto":
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    # --- log_every_n_steps: cap to batch count to avoid sparse-step warning ---
    n_train_batches = len(data_module.train_dataloader())
    log_every = max(1, min(lc.log_every_n_steps, n_train_batches))

    # --- Trainer ---
    trainer_kwargs: dict = dict(
        max_epochs=cfg.epochs,
        accelerator=lc.accelerator,
        precision=precision,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=cfg.grad_clip if cfg.grad_clip > 0 else None,
        enable_progress_bar=cfg.use_tqdm,
        enable_model_summary=lc.enable_model_summary,
        deterministic=lc.deterministic,
        limit_val_batches=1.0 if data_module.has_val else 0.0,
        log_every_n_steps=log_every,
    )
    if lc.devices is not None:
        trainer_kwargs["devices"] = lc.devices
    if lc.profiler is not None:
        trainer_kwargs["profiler"] = lc.profiler

    trainer = L.Trainer(**trainer_kwargs)

    log.info(
        "Lightning Trainer: accelerator=%s  precision=%s  max_epochs=%d  "
        "monitor=%s  tensorboard=%s  csv=%s  checkpoints→%s  "
        "early_stopping=%s(patience=%d)",
        lc.accelerator, precision, cfg.epochs,
        monitor_key, lc.tensorboard, lc.csv_logger,
        run_dir / lc.checkpoint_dir,
        lc.early_stopping, lc.patience,
    )

    try:
        trainer.fit(module, datamodule=data_module)
    except KeyboardInterrupt:
        log.warning(
            "Training interrupted by user (Ctrl+C).  "
            "%d epochs completed — saving partial encoder artifact.",
            len(metrics_cb.rows),
        )

    return metrics_cb.rows, data_module.n_train_samples, data_module.n_val_samples


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def train_pretrain(cfg: PretrainConfig) -> Path:
    """Run a full Level 0 pretraining run and save the encoder artifact.

    Creates a versioned run directory ``<output_dir>/run_NNN/`` (or
    ``<output_dir>/<cfg.run_name>/`` when *run_name* is set) and writes all
    outputs there::

        output_dir/run_001/
            encoder.pt
            pretrain_meta.json
            metrics.csv
            tensorboard/        ← Lightning only
            csv/                ← Lightning only
            checkpoints/        ← Lightning only

    When ``cfg.preprocess_cache_dir`` is set, the function first validates the
    cache (or builds it from ``cfg.data_dir`` when stale/absent) and then uses
    the fast ``CachedPretrainDataset`` / ``CachedPretrainDataModule`` for
    training I/O instead of reading raw image files.

    Returns
    -------
    run_dir : Path
        The resolved run directory (useful for post-run logging in the caller).
    """
    run_dir = _resolve_run_dir(cfg.output_dir, cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)

    # Unlock Tensor Cores on Ampere / Ada / Blackwell GPUs (RTX 30xx / 40xx / 50xx).
    # Suppresses the "You are using a CUDA device that has Tensor Cores" warning.
    prec = cfg.lightning.float32_matmul_precision
    if prec is not None:
        torch.set_float32_matmul_precision(prec)
        log.info("torch.set_float32_matmul_precision('%s')", prec)

    # -----------------------------------------------------------------------
    # Preprocessed data cache: check / build
    # -----------------------------------------------------------------------
    cache_dir: Optional[Path] = None
    if cfg.preprocess_cache_dir is not None:
        from trackmania_rl.pretrain_visual.preprocess import (
            is_cache_valid,
            build_cache,
        )
        cache_dir = Path(cfg.preprocess_cache_dir)
        if is_cache_valid(
            cache_dir,
            cfg.data_dir,
            cfg.image_size,
            cfg.n_stack,
            cfg.val_fraction,
            cfg.seed,
        ):
            log.info("Using preprocessed cache: %s", cache_dir)
        else:
            log.info(
                "Preprocessed cache is absent or stale — building from %s → %s ...",
                cfg.data_dir, cache_dir,
            )
            build_cache(
                data_dir=cfg.data_dir,
                cache_dir=cache_dir,
                image_size=cfg.image_size,
                n_stack=cfg.n_stack,
                val_fraction=cfg.val_fraction,
                seed=cfg.seed,
                workers=cfg.cache_build_workers,
            )
            log.info("Cache build complete.  Starting training from cache...")

    enc_dim = get_enc_dim(1, cfg.image_size)

    # --- Lightning path ---
    if cfg.framework == "lightning":
        if cfg.n_stack > 1 and cfg.stack_mode == "channel":
            in_channels = cfg.n_stack
            stacked_concat = False
            encoder = build_iqn_encoder(in_channels, cfg.image_size)
        elif cfg.n_stack > 1 and cfg.stack_mode == "concat":
            in_channels = 1
            stacked_concat = True
            enc1 = build_iqn_encoder(1, cfg.image_size)
            encoder = StackedEncoderConcat(enc1, cfg.n_stack, enc_dim)
        else:
            in_channels = 1
            stacked_concat = False
            encoder = build_iqn_encoder(1, cfg.image_size)

        metrics_rows, n_train, n_val = _train_lightning(
            cfg, encoder, in_channels, stacked_concat, run_dir, cache_dir=cache_dir,
        )
        backbone = encoder.encoder_1ch if stacked_concat else encoder
        meta = _build_meta(cfg, in_channels, enc_dim, metrics_rows, n_train, n_val)
        save_encoder_artifact(backbone, meta, run_dir, metrics_rows)
        log.info("Level 0 pretrain complete.  Artifact: %s", run_dir)
        return run_dir

    # --- Native path ---
    encoder, loader, enc_dim, in_channels, stacked_concat, lightly_ok = _make_encoder_and_loader(
        cfg, device, cache_dir=cache_dir,
    )

    if cfg.task == "ae":
        out_channels = cfg.n_stack if (cfg.n_stack > 1 and not stacked_concat) else 1
        decoder = build_ae_decoder(enc_dim, out_channels, cfg.image_size).to(device)
        rows = _train_ae_native(encoder, decoder, loader, device, cfg.epochs, cfg.lr, stacked_concat, cfg.use_tqdm)
    elif cfg.task == "vae":
        out_channels = cfg.n_stack if (cfg.n_stack > 1 and not stacked_concat) else 1
        vae_head = build_vae_head(enc_dim, cfg.vae_latent).to(device)
        decoder = build_vae_decoder(cfg.vae_latent, out_channels, cfg.image_size).to(device)
        rows = _train_vae_native(encoder, vae_head, decoder, loader, device, cfg.epochs, cfg.lr, cfg.kl_weight, stacked_concat, cfg.use_tqdm)
    elif cfg.task == "simclr":
        proj = build_simclr_projection(enc_dim, cfg.proj_dim).to(device)
        rows = _train_simclr_native(encoder, proj, loader, device, cfg.epochs, cfg.lr, cfg.temperature, stacked_concat, cfg.use_tqdm, lightly_ok)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    backbone = encoder.encoder_1ch if stacked_concat else encoder
    n_train = len(loader.dataset)
    meta = _build_meta(cfg, in_channels, enc_dim, rows, n_train, 0)
    save_encoder_artifact(backbone, meta, run_dir, rows)
    log.info("Level 0 pretrain complete.  Artifact: %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Meta builder
# ---------------------------------------------------------------------------

def _build_meta(
    cfg: PretrainConfig,
    in_channels: int,
    enc_dim: int,
    rows: list[dict],
    n_train: int,
    n_val: int,
) -> dict:
    train_loss_final = rows[-1].get("train_loss") if rows else None
    val_loss_final = rows[-1].get("val_loss") if rows else None
    return {
        "task": cfg.task,
        "framework": cfg.framework,
        "image_size": cfg.image_size,
        "n_stack": cfg.n_stack,
        "stack_mode": cfg.stack_mode,
        "in_channels": in_channels,
        "enc_dim": enc_dim,
        "normalization": "none",
        "epochs_trained": len(rows),
        "train_loss_final": train_loss_final,
        "val_loss_final": val_loss_final,
        "dataset_root": str(cfg.data_dir.resolve()),
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "seed": cfg.seed,
    }
