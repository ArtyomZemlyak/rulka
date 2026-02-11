"""
Unsupervised pretraining of the visual backbone (IQN img_head–compatible CNN) on frames.

Tasks: ae (autoencoder), vae (variational AE), simclr (contrastive).
Dataset: folder of images (any format; loaded as grayscale, resized to --image-size).
Optional: use Lightly framework for more methods (pip install lightly) via --framework lightly.

Stacked frames: --n-stack N uses N consecutive images (temporal order = file sort order).
  --stack-mode channel: stack as N channels, encoder Conv2d(N, ...); saves encoder.pt (N-ch).
  --stack-mode concat: run 1-ch encoder on each frame, concat features + linear; saves 1-ch encoder (IQN-compatible).

Output: encoder weights saved as encoder.pt, loadable into IQN_Network.img_head (when 1-ch or after averaging N-ch).

Usage:
  python scripts/pretrain_visual_backbone.py --data-dir ./frames --task ae --epochs 50 --batch-size 128
  python scripts/pretrain_visual_backbone.py --data-dir ./frames --task simclr --framework lightly
  python scripts/pretrain_visual_backbone.py --data-dir ./frames --n-stack 4 --stack-mode concat --task ae
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Reuse IQN backbone architecture and conv output dim
from trackmania_rl.agents.iqn import calculate_conv_output_dim

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

try:
    import lightly
    from lightly.loss import NTXentLoss
    from lightly.models.modules.heads import SimCLRProjectionHead
    from lightly.transforms.simclr_transform import SimCLRTransform

    LIGHTLY_AVAILABLE = True
except ImportError:
    LIGHTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def build_encoder(in_channels: int = 1, h: int = 64, w: int = 64) -> nn.Module:
    """Build CNN encoder identical to IQN_Network.img_head (for weight transfer)."""
    channels = [in_channels, 16, 32, 64, 32]
    act = nn.LeakyReLU(inplace=True)
    return nn.Sequential(
        nn.Conv2d(channels[0], channels[1], kernel_size=(4, 4), stride=2),
        act,
        nn.Conv2d(channels[1], channels[2], kernel_size=(4, 4), stride=2),
        act,
        nn.Conv2d(channels[2], channels[3], kernel_size=(3, 3), stride=2),
        act,
        nn.Conv2d(channels[3], channels[4], kernel_size=(3, 3), stride=1),
        act,
        nn.Flatten(),
    )


class StackedEncoderConcat(nn.Module):
    """Run 1-ch encoder on each of N frames, concat features, linear to single vector. Saves 1-ch encoder for IQN."""

    def __init__(self, encoder_1ch: nn.Module, n_stack: int, enc_dim: int):
        super().__init__()
        self.encoder_1ch = encoder_1ch
        self.n_stack = n_stack
        self.fusion = nn.Linear(n_stack * enc_dim, enc_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 1, H, W)
        B, N, _, H, W = x.shape
        feats = []
        for i in range(N):
            feats.append(self.encoder_1ch(x[:, i]))
        z = torch.cat(feats, dim=1)  # (B, N*enc_dim)
        return self.fusion(z)


def build_decoder(encoder_output_dim: int, out_channels: int = 1, h: int = 64, w: int = 64) -> nn.Module:
    """Simple decoder: linear -> reshape -> conv transposes to get back to h x w."""
    # Approximate spatial size after encoder: 64->32->16->8->6->4 (stride 2,2,2,1). So ~4x4.
    base = 4
    return nn.Sequential(
        nn.Linear(encoder_output_dim, 32 * base * base),
        nn.LeakyReLU(inplace=True),
        nn.Unflatten(1, (32, base, base)),
        nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose2d(16, out_channels, 4, stride=2, padding=1),
        nn.Sigmoid(),
    )


def _load_one_frame(p: Path, size: int) -> torch.Tensor:
    import numpy as np
    import cv2

    if p.suffix.lower() == ".npy":
        img = np.load(p).squeeze()
        if img.ndim == 3:
            img = img.mean(axis=-1)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((size, size), dtype=np.float32)
        else:
            img = img.astype(np.float32) / 255.0
    if img.ndim != 2:
        img = img.mean(axis=-1)
    img = torch.from_numpy(img).float().unsqueeze(0)  # 1 x H x W
    if img.shape[1] != size or img.shape[2] != size:
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return img


class ImageFolderFrames(Dataset):
    """Load images from a directory (flat or recursive). Grayscale, resize to (H, W).
    If n_stack > 1, returns N consecutive frames (temporal order = sorted paths): shape (N, 1, H, W).
    """

    def __init__(
        self,
        root: Path,
        size: int = 64,
        n_stack: int = 1,
        extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".npy"),
    ):
        self.root = Path(root)
        self.size = size
        self.n_stack = n_stack
        self.paths = []
        for ext in extensions:
            self.paths.extend(self.root.rglob(f"*{ext}"))
        self.paths = sorted(self.paths)

    def __len__(self) -> int:
        if self.n_stack <= 1:
            return len(self.paths)
        return max(0, len(self.paths) - self.n_stack + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.n_stack <= 1:
            return _load_one_frame(self.paths[idx], self.size)
        frames = []
        for i in range(idx, idx + self.n_stack):
            frames.append(_load_one_frame(self.paths[i], self.size))
        return torch.stack(frames, dim=0)  # (N, 1, H, W)


def train_ae(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: Path,
    stacked_concat: bool = False,
    use_tqdm: bool = True,
) -> None:
    """AE: encode -> decode -> MSE."""
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        n = 0
        batch_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") if (use_tqdm and TQDM_AVAILABLE) else loader
        for batch in batch_iter:
            x = batch.to(device)
            if x.dim() == 5 and not stacked_concat:
                x = x.squeeze(2)  # (B, N, H, W) for channel-mode encoder
            opt.zero_grad()
            z = encoder(x)
            x_recon = decoder(z)
            target = x if x.dim() == 4 else x.squeeze(2)
            loss = nn.functional.mse_loss(x_recon, target)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        avg_loss = total_loss / max(n, 1)
        log.info("epoch %d/%d loss %.6f", epoch + 1, epochs, avg_loss)
    to_save = encoder.encoder_1ch if stacked_concat else encoder
    torch.save(to_save.state_dict(), out_dir / "encoder.pt")


def build_vae_decoder(latent_dim: int, out_channels: int = 1, h: int = 64, w: int = 64) -> nn.Module:
    """Decoder from latent vector to image (for VAE)."""
    base = 4
    return nn.Sequential(
        nn.Linear(latent_dim, 32 * base * base),
        nn.LeakyReLU(inplace=True),
        nn.Unflatten(1, (32, base, base)),
        nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose2d(16, out_channels, 4, stride=2, padding=1),
        nn.Sigmoid(),
    )


def train_vae(
    encoder: nn.Module,
    enc_dim: int,
    latent_dim: int,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: Path,
    stacked_concat: bool = False,
    out_channels: int = 1,
    use_tqdm: bool = True,
) -> None:
    """VAE: encoder -> linear -> mu/logvar; sample z; decode z -> image."""
    encoder_vae = nn.Sequential(
        encoder,
        nn.Linear(enc_dim, 256),
        nn.LeakyReLU(inplace=True),
        nn.Linear(256, 2 * latent_dim),
    ).to(device)
    decoder_vae = build_vae_decoder(latent_dim, out_channels, loader.dataset.size, loader.dataset.size).to(device)
    opt = torch.optim.Adam(list(encoder_vae.parameters()) + list(decoder_vae.parameters()), lr=lr)
    for epoch in range(epochs):
        encoder_vae.train()
        decoder_vae.train()
        total_loss = 0.0
        n = 0
        batch_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") if (use_tqdm and TQDM_AVAILABLE) else loader
        for batch in batch_iter:
            x = batch.to(device)
            if x.dim() == 5 and not stacked_concat:
                x = x.squeeze(2)
            opt.zero_grad()
            out = encoder_vae(x)
            mu, logvar = out.chunk(2, dim=1)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            x_recon = decoder_vae(z)
            target = x.squeeze(2) if x.dim() == 5 else x
            recon = nn.functional.mse_loss(x_recon, target)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            loss = recon + 0.001 * kl
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        avg_loss = total_loss / max(n, 1)
        log.info("epoch %d/%d loss %.6f", epoch + 1, epochs, avg_loss)
    backbone = encoder_vae[0]
    to_save = backbone.encoder_1ch if stacked_concat else backbone
    torch.save(to_save.state_dict(), out_dir / "encoder.pt")


def train_simclr_native(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: Path,
    proj_dim: int = 128,
    temp: float = 0.5,
    image_size: int = 64,
    stacked_concat: bool = False,
    in_channels: int = 1,
    use_tqdm: bool = True,
) -> None:
    """SimCLR without Lightly: random crop + color jitter (grayscale = brightness), projection head, NT-Xent."""
    if stacked_concat:
        enc_dim = encoder(torch.zeros(1, encoder.n_stack, 1, image_size, image_size)).shape[1]
    else:
        enc_dim = encoder(torch.zeros(1, in_channels, image_size, image_size)).shape[1]
    projection = nn.Sequential(
        nn.Linear(enc_dim, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, proj_dim),
    )
    opt = torch.optim.Adam(list(encoder.parameters()) + list(projection.parameters()), lr=lr)

    def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = temp) -> torch.Tensor:
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # 2B x D
        sim = torch.mm(z, z.t()) / temperature
        mask = torch.eye(2 * z.size(0), device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([torch.arange(z.size(0), device=z.device) + z.size(0), torch.arange(z.size(0), device=z.device)])
        return nn.functional.cross_entropy(sim, labels)

    def aug(x: torch.Tensor) -> torch.Tensor:
        """Simple aug: crop + brightness. Supports (B, C, H, W) and (B, N, 1, H, W)."""
        if x.dim() == 5:
            b, n, _, h, w = x.shape
            crop = min(8, max(1, h // 8))
            top = torch.randint(0, crop + 1, (1,)).item()
            left = torch.randint(0, crop + 1, (1,)).item()
            x = x[:, :, :, top : top + h - crop, left : left + w - crop]
            x = x.reshape(b * n, 1, h - crop, w - crop)
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            x = x.reshape(b, n, 1, h, w)
        else:
            b, c, h, w = x.shape
            crop = min(8, max(1, h // 8))
            top = torch.randint(0, crop + 1, (1,)).item()
            left = torch.randint(0, crop + 1, (1,)).item()
            x = x[:, :, top : top + h - crop, left : left + w - crop]
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        bright = 0.2 * (2 * torch.rand(1, device=x.device) - 1) + 1
        x = (x * bright).clamp(0, 1)
        return x

    for epoch in range(epochs):
        encoder.train()
        projection.train()
        total_loss = 0.0
        n = 0
        batch_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") if (use_tqdm and TQDM_AVAILABLE) else loader
        for batch in batch_iter:
            x = batch.to(device)
            if x.dim() == 5 and not stacked_concat:
                x = x.squeeze(2)
            x1, x2 = aug(x), aug(x)
            opt.zero_grad()
            z1 = projection(encoder(x1))
            z2 = projection(encoder(x2))
            loss = nt_xent(z1, z2)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        avg_loss = total_loss / max(n, 1)
        log.info("epoch %d/%d loss %.6f", epoch + 1, epochs, avg_loss)
    to_save = encoder.encoder_1ch if stacked_concat else encoder
    torch.save(to_save.state_dict(), out_dir / "encoder.pt")


def train_simclr_lightly(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: Path,
    image_size: int = 64,
    use_tqdm: bool = True,
) -> None:
    """SimCLR using Lightly (if installed)."""
    enc_dim = encoder(torch.zeros(1, 1, image_size, image_size)).shape[1]
    projection = SimCLRProjectionHead(enc_dim, enc_dim, 128)
    model = nn.Sequential(encoder, nn.Flatten(start_dim=1), projection)
    criterion = NTXentLoss(temperature=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        batch_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") if (use_tqdm and TQDM_AVAILABLE) else loader
        for batch in batch_iter:
            if isinstance(batch, (list, tuple)):
                x = torch.cat(batch, dim=0)
            else:
                x = batch
            x = x.to(device)
            if x.shape[1] == 3:
                x = x[:, :1]
            opt.zero_grad()
            z = model(x)
            loss = criterion(z)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        avg_loss = total_loss / max(n, 1)
        log.info("epoch %d/%d loss %.6f", epoch + 1, epochs, avg_loss)
    torch.save(encoder.state_dict(), out_dir / "encoder.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unsupervised pretrain visual backbone (IQN img_head–compatible).")
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory with images (frames)")
    ap.add_argument("--output-dir", type=Path, default=Path("pretrain_visual_out"), help="Where to save encoder.pt")
    ap.add_argument("--task", type=str, default="ae", choices=["ae", "vae", "simclr"], help="Pretraining task")
    ap.add_argument("--framework", type=str, default="native", choices=["native", "lightly"], help="Use native PyTorch or lightly (if installed)")
    ap.add_argument("--image-size", type=int, default=64, help="Resize images to this (same as IQN)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--vae-latent", type=int, default=64, help="VAE latent dimension")
    ap.add_argument("--n-stack", type=int, default=1, help="Number of consecutive frames (temporal order = file sort)")
    ap.add_argument("--stack-mode", type=str, default="channel", choices=["channel", "concat"], help="channel: N ch encoder; concat: 1-ch per frame + fusion (saves IQN-compatible 1-ch)")
    ap.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars (use only logging)")
    args = ap.parse_args()

    use_tqdm = TQDM_AVAILABLE and not args.no_tqdm
    if not TQDM_AVAILABLE and not args.no_tqdm:
        log.info("tqdm not installed; progress bars disabled. Install with: pip install tqdm")

    if args.framework == "lightly" and not LIGHTLY_AVAILABLE:
        log.info("lightly not installed. Install with: pip install lightly. Using native.")
        args.framework = "native"

    if args.n_stack > 1 and args.framework == "lightly":
        log.info("Stacked frames with --framework lightly is not supported; use native.")
        args.framework = "native"

    dataset = ImageFolderFrames(args.data_dir, size=args.image_size, n_stack=args.n_stack)
    if len(dataset) == 0:
        log.error("No images found in %s", args.data_dir)
        return
    log.info("Dataset: %d samples (n_stack=%d)", len(dataset), args.n_stack)

    loader: DataLoader
    if args.task == "simclr" and args.framework == "lightly" and args.n_stack == 1:
        transform = SimCLRTransform(input_size=args.image_size, gaussian_blur=0.0)

        class LightlyDataset(Dataset):
            def __init__(self, root: Path, size: int, transform_: object):
                self.paths = sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.jpeg"))
                self.size = size
                self.transform_ = transform_

            def __len__(self) -> int:
                return len(self.paths)

            def __getitem__(self, idx: int):
                from PIL import Image

                p = self.paths[idx]
                img = Image.open(p).convert("RGB")
                return self.transform_(img), self.transform_(img)

        dataset_lightly = LightlyDataset(args.data_dir, args.image_size, transform_=transform)
        loader = DataLoader(dataset_lightly, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=(args.task == "simclr"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_enc_dim = calculate_conv_output_dim(args.image_size, args.image_size)
    out_channels = 1
    stacked_concat = False
    in_channels = 1

    if args.n_stack > 1:
        out_channels = args.n_stack
        if args.stack_mode == "channel":
            in_channels = args.n_stack
            encoder = build_encoder(in_channels, args.image_size, args.image_size).to(device)
        else:
            encoder_1ch = build_encoder(1, args.image_size, args.image_size)
            encoder = StackedEncoderConcat(encoder_1ch, args.n_stack, single_enc_dim).to(device)
            stacked_concat = True
            in_channels = 1
    else:
        encoder = build_encoder(1, args.image_size, args.image_size).to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "ae":
        decoder = build_decoder(single_enc_dim, out_channels, args.image_size, args.image_size).to(device)
        train_ae(encoder, decoder, loader, device, args.epochs, args.lr, args.output_dir, stacked_concat=stacked_concat, use_tqdm=use_tqdm)
    elif args.task == "vae":
        enc_dim = single_enc_dim
        train_vae(encoder, enc_dim, args.vae_latent, loader, device, args.epochs, args.lr, args.output_dir, stacked_concat=stacked_concat, out_channels=out_channels, use_tqdm=use_tqdm)
    elif args.task == "simclr":
        if args.framework == "lightly":
            train_simclr_lightly(encoder, loader, device, args.epochs, args.lr, args.output_dir, args.image_size, use_tqdm=use_tqdm)
        else:
            train_simclr_native(
                encoder, loader, device, args.epochs, args.lr, args.output_dir,
                128, 0.5, args.image_size, stacked_concat=stacked_concat, in_channels=in_channels, use_tqdm=use_tqdm,
            )
    else:
        log.error("Unknown task %s", args.task)
        return

    log.info("Saved encoder to %s", args.output_dir / "encoder.pt")
    if args.n_stack > 1 and args.stack_mode == "channel":
        log.info("Encoder has N input channels. For single-frame IQN: use first channel or average first-layer kernels to 1 channel.")
    else:
        log.info("Load into IQN: network.img_head.load_state_dict(torch.load('.../encoder.pt', weights_only=True))")


if __name__ == "__main__":
    main()
