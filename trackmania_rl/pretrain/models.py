"""
Encoder and decoder model factories for Level 0 visual pretraining.

All encoders are architecturally identical to IQN_Network.img_head so that
state dicts can be loaded directly into the RL agent after pretraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from trackmania_rl.agents.iqn import calculate_conv_output_dim


def build_iqn_encoder(in_channels: int = 1, image_size: int = 64) -> nn.Sequential:
    """Build CNN encoder matching IQN_Network.img_head exactly.

    Parameters
    ----------
    in_channels:
        1 for a standard single-frame or ``stack_mode=concat`` encoder (IQN-compatible).
        N for ``stack_mode=channel`` with N stacked frames (requires kernel-averaging before
        loading into IQN, see ``export.average_first_layer_to_1ch``).
    image_size:
        Square input resolution (e.g. 64 for the default IQN setup).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=(4, 4), stride=2),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=(4, 4), stride=2),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1),
        nn.LeakyReLU(inplace=True),
        nn.Flatten(),
    )


class StackedEncoderConcat(nn.Module):
    """Run a 1-ch encoder on each of N stacked frames, concatenate features, project back.

    Saves only the 1-ch encoder for IQN (``stack_mode=concat``).

    Input shape: (B, N, 1, H, W)
    Output shape: (B, enc_dim)
    """

    def __init__(self, encoder_1ch: nn.Module, n_stack: int, enc_dim: int) -> None:
        super().__init__()
        self.encoder_1ch = encoder_1ch
        self.n_stack = n_stack
        self.fusion = nn.Linear(n_stack * enc_dim, enc_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _, H, W = x.shape
        feats = [self.encoder_1ch(x[:, i]) for i in range(N)]
        return self.fusion(torch.cat(feats, dim=1))


def build_ae_decoder(enc_dim: int, out_channels: int = 1, image_size: int = 64) -> nn.Sequential:
    """Transposed-conv decoder for the autoencoder task.

    Reconstructs an image of shape ``(B, out_channels, image_size, image_size)``.
    Works for ``image_size ∈ {64, 128}``.  For other sizes the spatial dimensions
    are adjusted to match via ``image_size``-dependent base spatial size.
    """
    # Approximate spatial size at encoder output: 64→4x4, 128→8x8.
    base = max(1, image_size // 16)
    return nn.Sequential(
        nn.Linear(enc_dim, 32 * base * base),
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


def build_vae_head(enc_dim: int, latent_dim: int) -> nn.Sequential:
    """MLP that maps encoder features to VAE parameters (mu ++ log-var concatenated)."""
    return nn.Sequential(
        nn.Linear(enc_dim, 256),
        nn.LeakyReLU(inplace=True),
        nn.Linear(256, 2 * latent_dim),
    )


def build_vae_decoder(latent_dim: int, out_channels: int = 1, image_size: int = 64) -> nn.Sequential:
    """Transposed-conv decoder from VAE latent vector back to image."""
    base = max(1, image_size // 16)
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


def build_simclr_projection(enc_dim: int, proj_dim: int = 128) -> nn.Sequential:
    """Two-layer MLP projection head for SimCLR."""
    return nn.Sequential(
        nn.Linear(enc_dim, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, proj_dim),
    )


def build_encoder_from_meta(meta: dict) -> nn.Sequential:
    """Reconstruct encoder matching the architecture described in ``pretrain_meta.json``."""
    return build_iqn_encoder(
        in_channels=meta["in_channels"],
        image_size=meta["image_size"],
    )


def get_enc_dim(in_channels: int, image_size: int) -> int:
    """Return the output dimension of an IQN-compatible encoder."""
    return calculate_conv_output_dim(image_size, image_size)


# ---------------------------------------------------------------------------
# BC (behavioral cloning) model
# ---------------------------------------------------------------------------


class BCNetwork(nn.Module):
    """Encoder + action head for BC. encoder is IQN-compatible for transfer."""

    def __init__(
        self,
        encoder: nn.Module,
        enc_dim: int,
        n_actions: int,
        use_floats: bool = False,
        float_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_floats = use_floats
        if use_floats and float_dim > 0:
            self.float_head = nn.Sequential(
                nn.Linear(float_dim, 256),
                nn.LeakyReLU(inplace=True),
                nn.Linear(256, 256),
                nn.LeakyReLU(inplace=True),
            )
            action_input_dim = enc_dim + 256
        else:
            self.float_head = None
            action_input_dim = enc_dim
        self.action_head = nn.Linear(action_input_dim, n_actions)

    def forward(
        self,
        img: torch.Tensor,
        float_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.encoder(img)
        if self.float_head is not None and float_inputs is not None:
            float_feats = self.float_head(float_inputs)
            feats = torch.cat((feats, float_feats), dim=1)
        return self.action_head(feats)


def build_bc_network(
    enc_dim: int,
    n_actions: int,
    use_floats: bool = False,
    float_dim: int = 0,
    in_channels: int = 1,
    image_size: int = 64,
) -> BCNetwork:
    """Build BC model: IQN-compatible encoder + optional float head + action head."""
    encoder = build_iqn_encoder(in_channels=in_channels, image_size=image_size)
    return BCNetwork(
        encoder=encoder,
        enc_dim=enc_dim,
        n_actions=n_actions,
        use_floats=use_floats,
        float_dim=float_dim,
    )
