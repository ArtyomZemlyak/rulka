"""
Encoder and decoder model factories for Level 0 visual pretraining.

All encoders are architecturally identical to IQN_Network.img_head so that
state dicts can be loaded directly into the RL agent after pretraining.
"""

from __future__ import annotations

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from trackmania_rl.agents.iqn import IQN_Network, calculate_conv_output_dim


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
    """Encoder + optional float head + action head(s) for BC. encoder is IQN-compatible for transfer.

    When n_offsets > 1, one Linear head per time offset (no use_actions_head MLP).
    When n_offsets == 1, single head as before (Linear or two-layer MLP when use_actions_head).
    """

    def __init__(
        self,
        encoder: nn.Module,
        enc_dim: int,
        n_actions: int,
        n_offsets: int = 1,
        use_floats: bool = False,
        float_dim: int = 0,
        float_hidden_dim: int = 256,
        float_inputs_mean: Optional[torch.Tensor] = None,
        float_inputs_std: Optional[torch.Tensor] = None,
        use_actions_head: bool = False,
        dense_hidden_dimension: int = 1024,
        dropout: float = 0.0,
        action_head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_offsets = n_offsets
        self.use_floats = use_floats
        self.dropout = dropout
        if use_floats and float_dim > 0:
            self.float_head = nn.Sequential(
                nn.Linear(float_dim, float_hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(float_hidden_dim, float_hidden_dim),
                nn.LeakyReLU(inplace=True),
            )
            action_input_dim = enc_dim + float_hidden_dim
            if float_inputs_mean is not None and float_inputs_std is not None:
                self.register_buffer("_float_mean", float_inputs_mean)
                self.register_buffer("_float_std", float_inputs_std)
            else:
                self._float_mean = None
                self._float_std = None
        else:
            self.float_head = None
            self._float_mean = None
            self._float_std = None
            action_input_dim = enc_dim
        if n_offsets > 1:
            if use_actions_head:
                # Same MLP layout as IQN A_head per offset so head 0 can be merged into RL.
                # Optional dropout between the two Linear layers (saved state_dict maps to 0,2 for IQN).
                def _one_head():
                    layers = [
                        nn.Linear(action_input_dim, dense_hidden_dimension // 2),
                        nn.LeakyReLU(inplace=True),
                    ]
                    if action_head_dropout > 0:
                        layers.append(nn.Dropout(action_head_dropout))
                    layers.append(nn.Linear(dense_hidden_dimension // 2, n_actions))
                    return nn.Sequential(*layers)
                self.action_head = nn.ModuleList([_one_head() for _ in range(n_offsets)])
            else:
                self.action_head = nn.ModuleList([
                    nn.Linear(action_input_dim, n_actions) for _ in range(n_offsets)
                ])
        elif use_actions_head:
            layers = [
                nn.Linear(action_input_dim, dense_hidden_dimension // 2),
                nn.LeakyReLU(inplace=True),
            ]
            if action_head_dropout > 0:
                layers.append(nn.Dropout(action_head_dropout))
            layers.append(nn.Linear(dense_hidden_dimension // 2, n_actions))
            self.action_head = nn.Sequential(*layers)
        else:
            self.action_head = nn.Linear(action_input_dim, n_actions)

    def forward(
        self,
        img: torch.Tensor,
        float_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.encoder(img)
        if self.float_head is not None and float_inputs is not None:
            if getattr(self, "_float_mean", None) is not None and getattr(self, "_float_std", None) is not None:
                float_inputs = (float_inputs - self._float_mean) / self._float_std.clamp(min=1e-6)
            float_feats = self.float_head(float_inputs)
            feats = torch.cat((feats, float_feats), dim=1)
        if self.dropout > 0:
            feats = torch.nn.functional.dropout(feats, p=self.dropout, training=self.training)
        if self.n_offsets > 1:
            return torch.stack([h(feats) for h in self.action_head], dim=1)  # (B, n_offsets, n_actions)
        return self.action_head(feats)


def build_bc_network(
    enc_dim: int,
    n_actions: int,
    n_offsets: int = 1,
    use_floats: bool = False,
    float_dim: int = 0,
    float_hidden_dim: int = 256,
    float_inputs_mean: Optional[list[float]] = None,
    float_inputs_std: Optional[list[float]] = None,
    use_actions_head: bool = False,
    dense_hidden_dimension: int = 1024,
    dropout: float = 0.0,
    action_head_dropout: float = 0.0,
    in_channels: int = 1,
    image_size: int = 64,
) -> BCNetwork:
    """Build BC model: IQN-compatible encoder + optional float head + action head(s).

    When n_offsets > 1, one Linear head per offset. When n_offsets == 1 and use_actions_head
    is True, action head has the same layout as IQN A_head for RL transfer.
    """
    encoder = build_iqn_encoder(in_channels=in_channels, image_size=image_size)
    mean_ten = torch.tensor(float_inputs_mean, dtype=torch.float32) if float_inputs_mean else None
    std_ten = torch.tensor(float_inputs_std, dtype=torch.float32) if float_inputs_std else None
    return BCNetwork(
        encoder=encoder,
        enc_dim=enc_dim,
        n_actions=n_actions,
        n_offsets=n_offsets,
        use_floats=use_floats,
        float_dim=float_dim,
        float_hidden_dim=float_hidden_dim,
        float_inputs_mean=mean_ten,
        float_inputs_std=std_ten,
        use_actions_head=use_actions_head,
        dense_hidden_dimension=dense_hidden_dimension,
        dropout=dropout,
        action_head_dropout=action_head_dropout,
    )


# ---------------------------------------------------------------------------
# Full IQN for BC (use_full_iqn: train full IQN in BC for 1:1 transfer)
# ---------------------------------------------------------------------------


def build_iqn_for_bc(
    image_size: int,
    float_inputs_dim: int,
    float_hidden_dim: int,
    n_actions: int,
    dense_hidden_dimension: int = 1024,
    iqn_embedding_dimension: int = 128,
    float_inputs_mean: Optional[list[float]] = None,
    float_inputs_std: Optional[list[float]] = None,
) -> IQN_Network:
    """Build full IQN_Network for BC training (use_full_iqn).

    Architecture matches RL IQN so that the full state_dict can be loaded into
    the RL agent. Forward in BC uses num_quantiles=1 and tau either random or 0.5.

    Parameters
    ----------
    image_size : int
        Square input resolution (e.g. 64).
    float_inputs_dim, float_hidden_dim, n_actions : int
        Must match RL config.
    dense_hidden_dimension, iqn_embedding_dimension : int
        Must match RL neural_network config.
    float_inputs_mean, float_inputs_std : list[float] or None
        Same as RL state_normalization; length must equal float_inputs_dim.
        If None, use zeros and ones (no normalization).
    """
    conv_head_output_dim = calculate_conv_output_dim(image_size, image_size)
    mean_arr = np.array(float_inputs_mean, dtype=np.float32) if float_inputs_mean else np.zeros(float_inputs_dim, dtype=np.float32)
    std_arr = np.array(float_inputs_std, dtype=np.float32) if float_inputs_std else np.ones(float_inputs_dim, dtype=np.float32)
    if len(mean_arr) != float_inputs_dim or len(std_arr) != float_inputs_dim:
        raise ValueError(
            f"float_inputs_mean/std length must be float_inputs_dim={float_inputs_dim}, "
            f"got {len(mean_arr)} and {len(std_arr)}"
        )
    return IQN_Network(
        float_inputs_dim=float_inputs_dim,
        float_hidden_dim=float_hidden_dim,
        conv_head_output_dim=conv_head_output_dim,
        dense_hidden_dimension=dense_hidden_dimension,
        iqn_embedding_dimension=iqn_embedding_dimension,
        n_actions=n_actions,
        float_inputs_mean=mean_arr,
        float_inputs_std=std_arr,
    )


class IQN_BC_MultiOffset(nn.Module):
    """Full IQN structure with multiple A_heads for multi-offset BC (use_full_iqn + n_offsets > 1).

    Same img_head, float_feature_extractor, iqn_fc as IQN; A_heads is ModuleList of n_offsets
    clones of A_head. Forward returns (B, n_offsets, n_actions). For RL transfer, save
    state_dict with A_head.* = A_heads[0].* so it loads into IQN_Network.
    """

    def __init__(self, iqn: IQN_Network, n_offsets: int) -> None:
        super().__init__()
        self.img_head = iqn.img_head
        self.float_feature_extractor = iqn.float_feature_extractor
        self.iqn_fc = iqn.iqn_fc
        self.V_head = iqn.V_head
        self.register_buffer("float_inputs_mean", iqn.float_inputs_mean.to("cpu"))
        self.register_buffer("float_inputs_std", iqn.float_inputs_std.to("cpu"))
        self.iqn_embedding_dimension = iqn.iqn_embedding_dimension
        self.n_actions = iqn.n_actions
        self.A_heads = nn.ModuleList([copy.deepcopy(iqn.A_head) for _ in range(n_offsets)])

    def forward(
        self,
        img: torch.Tensor,
        float_inputs: torch.Tensor,
        num_quantiles: int = 1,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (B, n_offsets, n_actions) for BC multi-offset; tau is ignored for shape compatibility."""
        batch_size = img.shape[0]
        device = img.device
        dtype = img.dtype
        img_out = self.img_head(img)
        float_norm = (float_inputs - self.float_inputs_mean.to(device)) / self.float_inputs_std.to(device).clamp(min=1e-6)
        float_out = self.float_feature_extractor(float_norm)
        concat = torch.cat((img_out, float_out), dim=1)
        if tau is None:
            tau = torch.full((batch_size * num_quantiles, 1), 0.5, device=device, dtype=torch.float32)
        quantile_net = torch.cos(
            torch.arange(
                1, self.iqn_embedding_dimension + 1, 1, device=device, dtype=torch.float32
            ) * math.pi * tau
        )
        quantile_net = quantile_net.expand(-1, self.iqn_embedding_dimension)
        quantile_net = self.iqn_fc(quantile_net)
        concat = concat.repeat(num_quantiles, 1) * quantile_net
        out = torch.stack([self.A_heads[i](concat) for i in range(len(self.A_heads))], dim=1)
        return out, tau

    def state_dict_for_iqn_transfer(self, merge_all_heads: bool = False) -> dict:
        """State dict loadable into IQN_Network (A_head = first offset head).

        If merge_all_heads True, also add A_head_offset_1.*, A_head_offset_2.*, ...
        for all other offset heads (same layout as A_head).
        """
        sd = {}
        for name, buf in self.named_buffers():
            sd[name] = buf
        for name, param in self.named_parameters():
            if name.startswith("img_head."):
                sd[name] = param
            elif name.startswith("float_feature_extractor."):
                sd[name] = param
            elif name.startswith("iqn_fc."):
                sd[name] = param
            elif name.startswith("V_head."):
                sd[name] = param
            elif name.startswith("A_heads.0."):
                sd["A_head." + name[len("A_heads.0.") :]] = param
            elif merge_all_heads and name.startswith("A_heads."):
                # A_heads.1.0.weight -> A_head_offset_1.0.weight
                rest = name[len("A_heads.") :]
                if "." in rest:
                    idx, key = rest.split(".", 1)
                    if idx != "0":
                        sd["A_head_offset_" + idx + "." + key] = param
        return sd
