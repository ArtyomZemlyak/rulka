"""
PyTorch Lightning task modules for Level 0 visual pretraining.

Three tasks are implemented:

  AELightningModule     — Autoencoder (MSE reconstruction loss).
  VAELightningModule    — Variational Autoencoder (ELBO = MSE + β·KL).
  SimCLRLightningModule — Contrastive (NT-Xent loss, crop+brightness augmentation).

All modules expose the encoder as ``self.encoder`` so the caller can extract
it after training without unpacking a checkpoint.

Requires:  pip install lightning
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

try:
    import lightning as L
    from lightning.pytorch.callbacks import Callback
    _LIGHTNING_AVAILABLE = True
    _Base = L.LightningModule
except ImportError:
    _LIGHTNING_AVAILABLE = False
    L = None  # type: ignore[assignment]
    _Base = object
    Callback = object  # type: ignore[assignment,misc]

LIGHTNING_AVAILABLE = _LIGHTNING_AVAILABLE


# ---------------------------------------------------------------------------
# Metrics collection callback
# ---------------------------------------------------------------------------

if _LIGHTNING_AVAILABLE:
    class MetricsCollector(Callback):  # type: ignore[misc]
        """Accumulate per-epoch metrics from ``trainer.callback_metrics``.

        Runs in on_train_epoch_end. By that time Lightning has run validation for
        the current epoch, so callback_metrics typically contains both train_loss
        and val_loss for the same epoch.
        """

        def __init__(self) -> None:
            self.rows: list[dict] = []

        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            row: dict = {"epoch": trainer.current_epoch, "stage": "train"}
            for k, v in trainer.callback_metrics.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
            self.rows.append(row)

else:
    class MetricsCollector:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.rows: list[dict] = []


# ---------------------------------------------------------------------------
# Shared augmentation (domain-specific: grayscale crop + brightness)
# ---------------------------------------------------------------------------

def _augment(x: torch.Tensor) -> torch.Tensor:
    """Random crop + brightness jitter for grayscale TrackMania frames.

    Supports both single-frame ``(B, C, H, W)`` and stacked-frame ``(B, N, 1, H, W)``.
    The same augmentation parameters are applied to all frames in a stack so
    that temporal consistency is preserved.
    """
    if x.dim() == 5:
        b, n, _, h, w = x.shape
        crop = min(8, max(1, h // 8))
        top = int(torch.randint(0, crop + 1, (1,)).item())
        left = int(torch.randint(0, crop + 1, (1,)).item())
        x = x[:, :, :, top: top + h - crop, left: left + w - crop]
        x = x.reshape(b * n, 1, h - crop, w - crop)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = x.reshape(b, n, 1, h, w)
    else:
        b, c, h, w = x.shape
        crop = min(8, max(1, h // 8))
        top = int(torch.randint(0, crop + 1, (1,)).item())
        left = int(torch.randint(0, crop + 1, (1,)).item())
        x = x[:, :, top: top + h - crop, left: left + w - crop]
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    bright = 0.2 * (2.0 * torch.rand(1, device=x.device) - 1.0) + 1.0
    return (x * bright).clamp(0.0, 1.0)


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent contrastive loss (SimCLR)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)                                      # (2B, D)
    sim = torch.mm(z, z.t()) / temperature                               # (2B, 2B)
    mask = torch.eye(z.size(0), device=z.device, dtype=torch.bool)       # (2B, 2B)
    # Large negative representable in sim.dtype (safe for float16/bf16/float32 under AMP)
    fill_val = float(torch.finfo(sim.dtype).min)
    sim.masked_fill_(mask, fill_val)
    B = z1.size(0)
    labels = torch.cat([
        torch.arange(B, device=z.device) + B,
        torch.arange(B, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Helper: flatten a stacked-frame batch if needed
# ---------------------------------------------------------------------------

def _to_2d(x: torch.Tensor, stacked_concat: bool) -> torch.Tensor:
    """Convert (B, N, 1, H, W) → (B, N, H, W) for channel-mode encoders."""
    if x.dim() == 5 and not stacked_concat:
        return x.squeeze(2)   # channel mode: (B, N, H, W)
    return x                  # single-frame or stacked-concat: (B, 1, H, W)


# ---------------------------------------------------------------------------
# Autoencoder Lightning Module
# ---------------------------------------------------------------------------

class AELightningModule(_Base):  # type: ignore[misc]
    """Autoencoder: encode → decode → MSE reconstruction loss.

    Parameters
    ----------
    encoder:
        IQN-compatible CNN encoder.
    decoder:
        Transposed-conv decoder.
    lr:
        Learning rate for Adam optimizer.
    stacked_concat:
        ``True`` when encoder is a ``StackedEncoderConcat`` (input shape (B, N, 1, H, W)).
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-3,
        stacked_concat: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.stacked_concat = stacked_concat

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        x = _to_2d(batch, self.stacked_concat)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        target = x if x.dim() == 4 else x.squeeze(2)
        return F.mse_loss(x_recon, target)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self._loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ---------------------------------------------------------------------------
# VAE Lightning Module
# ---------------------------------------------------------------------------

class VAELightningModule(_Base):  # type: ignore[misc]
    """Variational Autoencoder: ELBO = MSE reconstruction + β·KL divergence.

    Parameters
    ----------
    encoder:
        Shared CNN backbone (IQN-compatible).
    vae_head:
        MLP that maps encoder features → 2·latent_dim (mu ++ log-var).
    decoder:
        Decoder from latent vector back to image.
    lr:
        Learning rate.
    kl_weight:
        β coefficient for KL term.
    stacked_concat:
        Same meaning as in AELightningModule.
    """

    def __init__(
        self,
        encoder: nn.Module,
        vae_head: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-3,
        kl_weight: float = 1e-3,
        stacked_concat: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.vae_head = vae_head
        self.decoder = decoder
        self.lr = lr
        self.kl_weight = kl_weight
        self.stacked_concat = stacked_concat

    def _loss(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = _to_2d(batch, self.stacked_concat)
        feat = self.encoder(x)
        out = self.vae_head(feat)
        mu, logvar = out.chunk(2, dim=1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        x_recon = self.decoder(z)
        target = x.squeeze(2) if x.dim() == 5 else x
        recon = F.mse_loss(x_recon, target)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        total = recon + self.kl_weight * kl
        return total, recon, kl

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, recon, kl = self._loss(batch)
        self.log_dict(
            {"train_loss": loss, "train_recon": recon, "train_kl": kl},
            on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss, recon, kl = self._loss(batch)
        self.log_dict(
            {"val_loss": loss, "val_recon": recon, "val_kl": kl},
            on_step=False, on_epoch=True, prog_bar=True,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ---------------------------------------------------------------------------
# SimCLR Lightning Module
# ---------------------------------------------------------------------------

class SimCLRLightningModule(_Base):  # type: ignore[misc]
    """SimCLR contrastive pretraining with domain-specific augmentation.

    Two random augmentations of the same image are encoded and projected;
    NT-Xent loss maximises agreement between the two views.

    Parameters
    ----------
    encoder:
        IQN-compatible CNN backbone.
    projection_head:
        Two-layer MLP projector (outputs are *not* saved, only encoder is kept).
    lr:
        Learning rate.
    temperature:
        NT-Xent softmax temperature.
    stacked_concat:
        Same meaning as in AELightningModule.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: nn.Module,
        lr: float = 1e-3,
        temperature: float = 0.5,
        stacked_concat: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.lr = lr
        self.temperature = temperature
        self.stacked_concat = stacked_concat

    def _encode_and_project(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_2d(x, self.stacked_concat)
        return self.projection_head(self.encoder(x))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        x1, x2 = _augment(x), _augment(x)
        z1 = self._encode_and_project(x1)
        z2 = self._encode_and_project(x2)
        loss = _nt_xent(z1, z2, self.temperature)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = batch
        x1, x2 = _augment(x), _augment(x)
        z1 = self._encode_and_project(x1)
        z2 = self._encode_and_project(x2)
        loss = _nt_xent(z1, z2, self.temperature)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ---------------------------------------------------------------------------
# BC (behavioral cloning) Lightning Module — Level 1
# ---------------------------------------------------------------------------

if _LIGHTNING_AVAILABLE:
    class BCLightningModule(_Base):  # type: ignore[misc]
        """Behavioral cloning: (image, action_idx) → CrossEntropy loss.

        Wraps a full BC model (encoder + action head). Exposes ``self.encoder``
        for saving the backbone artifact after training.

        Logs train/val loss, overall train/val accuracy, and per-class validation
        accuracy (val_acc_class_0, ...) for interpretation.
        """

        def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
            super().__init__()
            self.model = model
            self.lr = lr
            self.encoder = getattr(model, "encoder", model)
            n_actions = getattr(getattr(model, "action_head", None), "out_features", None)
            self._n_actions = n_actions if n_actions is not None else 12
            self.register_buffer("_train_correct", torch.zeros(self._n_actions, dtype=torch.float))
            self.register_buffer("_train_total", torch.zeros(self._n_actions, dtype=torch.float))
            self.register_buffer("_val_correct", torch.zeros(self._n_actions, dtype=torch.float))
            self.register_buffer("_val_total", torch.zeros(self._n_actions, dtype=torch.float))

        def _forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            img, action_idx = batch
            if img.dim() == 3:
                img = img.unsqueeze(1)
            elif img.dim() == 5 and img.shape[1] > 1:
                img = img[:, -1]
            logits = self.model(img)
            loss = F.cross_entropy(logits, action_idx)
            return loss, logits, action_idx

        def _update_acc_buffers(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            correct_buf: torch.Tensor,
            total_buf: torch.Tensor,
        ) -> None:
            n = self._n_actions
            one_hot = F.one_hot(target.clamp(0, n - 1), n).float()
            correct = (pred == target).float()
            correct_buf.add_((one_hot * correct.unsqueeze(1)).sum(0))
            total_buf.add_(one_hot.sum(0))

        def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            loss, logits, action_idx = self._forward(batch)
            pred = logits.argmax(dim=1)
            acc = (pred == action_idx).float().mean()
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            self._update_acc_buffers(pred, action_idx, self._train_correct, self._train_total)
            return loss

        def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
            loss, logits, action_idx = self._forward(batch)
            pred = logits.argmax(dim=1)
            acc = (pred == action_idx).float().mean()
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            self._update_acc_buffers(pred, action_idx, self._val_correct, self._val_total)

        def on_train_epoch_end(self) -> None:
            self._log_per_class_acc("train", self._train_correct, self._train_total)
            self._train_correct.zero_()
            self._train_total.zero_()

        def on_validation_epoch_end(self) -> None:
            self._log_per_class_acc("val", self._val_correct, self._val_total)
            self._val_correct.zero_()
            self._val_total.zero_()

        def _log_per_class_acc(self, stage: str, correct: torch.Tensor, total: torch.Tensor) -> None:
            with torch.no_grad():
                total_clamped = total.clamp(min=1e-8)
                per_class = (correct / total_clamped).cpu()
                for c in range(self._n_actions):
                    self.log(f"{stage}_acc_class_{c}", per_class[c].item(), on_epoch=True)

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
else:
    class BCLightningModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "BCLightningModule requires lightning. Install it with: pip install lightning"
            )
