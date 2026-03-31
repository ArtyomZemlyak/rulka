"""IQN float extractor and A/V heads from ``NnConfig.decoder`` (``mlp`` or ``transformer`` slot)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from config_files.nn_schema import IqnHeadSlotConfig, MLPConfig


def _linear(in_f: int, out_f: int, use_noisy: bool, noisy_sigma0: float) -> nn.Module:
    if use_noisy:
        from trackmania_rl.agents.iqn import FactorizedNoisyLinear

        return FactorizedNoisyLinear(in_f, out_f, sigma_0=noisy_sigma0)
    return nn.Linear(in_f, out_f)


def _effective_mlp_hidden(mlp: MLPConfig, dense_hidden_dimension: int) -> int:
    if mlp.hidden_dim is None:
        return max(1, dense_hidden_dimension // 2)
    return mlp.hidden_dim


def _append_mlp_hidden_blocks(
    modules: list[nn.Module],
    *,
    in_dim: int,
    hidden_dim: int,
    n_hidden_layers: int,
    use_layer_norm: bool,
    use_noisy_linear: bool,
    noisy_sigma0: float,
    activation_function: type[nn.Module],
) -> int:
    prev = in_dim
    for _ in range(n_hidden_layers):
        modules.append(_linear(prev, hidden_dim, use_noisy_linear, noisy_sigma0))
        if use_layer_norm:
            modules.append(nn.LayerNorm(hidden_dim))
        modules.append(activation_function(inplace=True))
        prev = hidden_dim
    return prev


class IQNTransformerTrunk(nn.Module):
    """Reshape flat (N, D) to tokens (N, D//d_model, d_model), TransformerEncoder, mean-pool -> (N, d_model)."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        n_layers: int,
        nhead: int,
        dim_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if in_dim % d_model != 0:
            raise ValueError(f"IQN transformer trunk: in_dim {in_dim} not divisible by d_model {d_model}")
        self.in_dim = in_dim
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        b, d = x.shape
        n_tok = d // self.d_model
        t = x.view(b, n_tok, self.d_model)
        y = self.enc(t).mean(dim=1)
        return y


def build_iqn_float_extractor(
    float_inputs_dim: int,
    hidden_dim: int,
    use_layer_norm: bool,
) -> nn.Sequential:
    activation_function = nn.LeakyReLU
    if use_layer_norm:
        return nn.Sequential(
            nn.Linear(float_inputs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_function(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_function(inplace=True),
        )
    return nn.Sequential(
        nn.Linear(float_inputs_dim, hidden_dim),
        activation_function(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        activation_function(inplace=True),
    )


def build_iqn_advantage_head(
    *,
    dense_input_dimension: int,
    dense_hidden_dimension: int,
    n_actions: int,
    n_actions_per_block: int,
    head_cfg: IqnHeadSlotConfig,
    use_layer_norm: bool,
    use_noisy_linear: bool,
    noisy_sigma0: float,
) -> Tuple[nn.Module, Optional[nn.Module]]:
    activation_function = nn.LeakyReLU

    if head_cfg.transformer is None:
        assert head_cfg.mlp is not None
        mlp = head_cfg.mlp
        h = _effective_mlp_hidden(mlp, dense_hidden_dimension)
        nl = mlp.n_hidden_layers

        if n_actions_per_block <= 1:
            mods: list[nn.Module] = []
            _append_mlp_hidden_blocks(
                mods,
                in_dim=dense_input_dimension,
                hidden_dim=h,
                n_hidden_layers=nl,
                use_layer_norm=use_layer_norm,
                use_noisy_linear=use_noisy_linear,
                noisy_sigma0=noisy_sigma0,
                activation_function=activation_function,
            )
            mods.append(_linear(h, n_actions, use_noisy_linear, noisy_sigma0))
            return nn.Sequential(*mods), None

        mods_first: list[nn.Module] = []
        _append_mlp_hidden_blocks(
            mods_first,
            in_dim=dense_input_dimension,
            hidden_dim=h,
            n_hidden_layers=nl,
            use_layer_norm=use_layer_norm,
            use_noisy_linear=use_noisy_linear,
            noisy_sigma0=noisy_sigma0,
            activation_function=activation_function,
        )
        first = nn.Sequential(*mods_first)
        last = _linear(h, n_actions_per_block * n_actions, use_noisy_linear, noisy_sigma0)
        return first, last

    tc = head_cfg.transformer
    dim_ff = max(64, tc.d_model * tc.ff_mult)
    a_head_hidden = dense_hidden_dimension // 2
    trunk = IQNTransformerTrunk(
        dense_input_dimension,
        tc.d_model,
        tc.n_layers,
        tc.n_heads,
        dim_ff,
        tc.dropout,
    )
    if n_actions_per_block <= 1:
        if use_layer_norm:
            seq = nn.Sequential(
                trunk,
                _linear(tc.d_model, a_head_hidden, use_noisy_linear, noisy_sigma0),
                nn.LayerNorm(a_head_hidden),
                activation_function(inplace=True),
                _linear(a_head_hidden, n_actions, use_noisy_linear, noisy_sigma0),
            )
        else:
            seq = nn.Sequential(
                trunk,
                _linear(tc.d_model, a_head_hidden, use_noisy_linear, noisy_sigma0),
                activation_function(inplace=True),
                _linear(a_head_hidden, n_actions, use_noisy_linear, noisy_sigma0),
            )
        return seq, None
    if use_layer_norm:
        first = nn.Sequential(
            trunk,
            _linear(tc.d_model, a_head_hidden, use_noisy_linear, noisy_sigma0),
            nn.LayerNorm(a_head_hidden),
            activation_function(inplace=True),
        )
    else:
        first = nn.Sequential(
            trunk,
            _linear(tc.d_model, a_head_hidden, use_noisy_linear, noisy_sigma0),
            activation_function(inplace=True),
        )
    last = _linear(a_head_hidden, n_actions_per_block * n_actions, use_noisy_linear, noisy_sigma0)
    return first, last


def build_iqn_value_head(
    *,
    dense_input_dimension: int,
    dense_hidden_dimension: int,
    head_cfg: IqnHeadSlotConfig,
    use_layer_norm: bool,
    use_noisy_linear: bool,
    noisy_sigma0: float,
) -> nn.Module:
    from trackmania_rl.agents.iqn import MatmulLinear

    activation_function = nn.LeakyReLU

    if head_cfg.transformer is None:
        assert head_cfg.mlp is not None
        mlp = head_cfg.mlp
        h = _effective_mlp_hidden(mlp, dense_hidden_dimension)
        nl = mlp.n_hidden_layers
        mods: list[nn.Module] = []
        _append_mlp_hidden_blocks(
            mods,
            in_dim=dense_input_dimension,
            hidden_dim=h,
            n_hidden_layers=nl,
            use_layer_norm=use_layer_norm,
            use_noisy_linear=use_noisy_linear,
            noisy_sigma0=noisy_sigma0,
            activation_function=activation_function,
        )
        mods.append(MatmulLinear(h, 1))
        return nn.Sequential(*mods)

    tc = head_cfg.transformer
    dim_ff = max(64, tc.d_model * tc.ff_mult)
    v_h = dense_hidden_dimension // 2
    trunk = IQNTransformerTrunk(
        dense_input_dimension,
        tc.d_model,
        tc.n_layers,
        tc.n_heads,
        dim_ff,
        tc.dropout,
    )
    if use_layer_norm:
        return nn.Sequential(
            trunk,
            _linear(tc.d_model, v_h, use_noisy_linear, noisy_sigma0),
            nn.LayerNorm(v_h),
            activation_function(inplace=True),
            MatmulLinear(v_h, 1),
        )
    return nn.Sequential(
        trunk,
        _linear(tc.d_model, v_h, use_noisy_linear, noisy_sigma0),
        activation_function(inplace=True),
        MatmulLinear(v_h, 1),
    )
