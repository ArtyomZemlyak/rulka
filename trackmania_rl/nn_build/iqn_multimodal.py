"""IQN on the shared multimodal vision+fusion stack (HF ViT, post_concat + BERT, etc.).

The submodule ``fusion`` is a headless body exposing ``forward_fusion_hidden`` (same graph as
on-policy policy without trunk/heads). IQN adds ``iqn_fc`` and dueling heads; the state vector
width is ``nn.decoder.dense_hidden_dimension`` after fusion ``bridge``, or ``2 * vis_d_model``
on the HF CLS+float path before the policy MLP trunk.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn

from trackmania_rl import utilities
from trackmania_rl.agents.iqn import FactorizedNoisyLinear
from trackmania_rl.nn_build.iqn_btr_from_config import iqn_btr_mlp_head_kw_from_config
from trackmania_rl.nn_build.iqn_heads import (
    build_iqn_advantage_head,
    build_iqn_value_head,
)
from trackmania_rl.nn_build.iqn_quantile_forward import forward_iqn_q_values_from_state_features


def iqn_uses_torch_fusion_backbone(cfg: Any) -> bool:
    return getattr(getattr(cfg, "transformers", None), "fusion_mode", "none") != "none"


def iqn_uses_hf_vision_only_backbone(cfg: Any) -> bool:
    if iqn_uses_torch_fusion_backbone(cfg):
        return False
    vis = getattr(cfg, "vis", None)
    if vis is None or getattr(vis, "no_image", False):
        return False
    vt = getattr(vis, "transformer", None)
    return vt is not None and bool(getattr(vt, "use_hf_backbone", False))


def iqn_uses_shared_multimodal_backbone(cfg: Any) -> bool:
    return iqn_uses_torch_fusion_backbone(cfg) or iqn_uses_hf_vision_only_backbone(cfg)


class IQNSharedBackboneNetwork(nn.Module):
    """IQN quantile network wrapping a headless multimodal (or HF vision) backbone as ``fusion``."""

    fusion: nn.Module

    def __init__(
        self,
        fusion: nn.Module,
        *,
        state_feature_dim: int,
        decoder: Any,
        iqn_embedding_dimension: int,
        n_actions: int,
        n_actions_per_block: int,
        use_layer_norm: bool,
        use_noisy_linear: bool,
        noisy_sigma0: float,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.n_actions = n_actions
        self.n_actions_per_block = n_actions_per_block
        self.use_noisy_linear = use_noisy_linear

        self.iqn_fc = nn.Sequential(
            nn.Linear(iqn_embedding_dimension, state_feature_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.A_head, self.A_head_multi = build_iqn_advantage_head(
            dense_input_dimension=state_feature_dim,
            dense_hidden_dimension=decoder.dense_hidden_dimension,
            n_actions=n_actions,
            n_actions_per_block=n_actions_per_block,
            head_cfg=decoder.advantage,
            use_layer_norm=use_layer_norm,
            use_noisy_linear=use_noisy_linear,
            noisy_sigma0=noisy_sigma0,
        )
        self.V_head = build_iqn_value_head(
            dense_input_dimension=state_feature_dim,
            dense_hidden_dimension=decoder.dense_hidden_dimension,
            head_cfg=decoder.value,
            use_layer_norm=use_layer_norm,
            use_noisy_linear=use_noisy_linear,
            noisy_sigma0=noisy_sigma0,
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        lrelu_neg_slope = 1e-2
        activation_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)

        def _should_init(m: nn.Module) -> bool:
            if isinstance(m, FactorizedNoisyLinear):
                return False
            return isinstance(m, (nn.Conv2d, nn.Linear))

        def _orthogonal_init(m: nn.Module, gain: float) -> None:
            w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
            torch.nn.init.orthogonal_(w, gain=gain)
            torch.nn.init.zeros_(m.bias)

        a_head_first = self.A_head[:-1] if self.A_head_multi is None else self.A_head
        modules_to_init = [a_head_first, self.V_head[:-1]]
        for module in modules_to_init:
            for m in module.modules():
                if _should_init(m):
                    _orthogonal_init(m, activation_gain)

        utilities.init_orthogonal(self.iqn_fc[0], np.sqrt(2) * activation_gain)

        def _init_last(layer: nn.Module) -> None:
            if isinstance(layer, FactorizedNoisyLinear):
                return
            utilities.init_orthogonal(layer)

        if self.A_head_multi is None:
            _init_last(self.A_head[-1])
        else:
            _init_last(self.A_head_multi)
        _init_last(self.V_head[-1])

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()

    def disable_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.disable_noise()

    def forward(
        self, img: torch.Tensor, float_inputs: torch.Tensor, num_quantiles: int, tau: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fusion.forward_fusion_hidden(img, float_inputs)
        return forward_iqn_q_values_from_state_features(
            h,
            num_quantiles,
            tau,
            iqn_embedding_dimension=self.iqn_embedding_dimension,
            iqn_fc=self.iqn_fc,
            V_head=self.V_head,
            A_head=self.A_head,
            A_head_multi=self.A_head_multi,
            n_actions=self.n_actions,
            n_actions_per_block=self.n_actions_per_block,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


def build_iqn_torch_fusion_network_uncompiled(cfg: Any = None) -> IQNSharedBackboneNetwork:
    from config_files.config_loader import get_config
    from trackmania_rl.agents.policy_models.multimodal_torch_fusion import build_multimodal_fusion_uncompiled

    c = cfg or get_config()
    fusion = build_multimodal_fusion_uncompiled(c, include_policy_heads=False)

    return IQNSharedBackboneNetwork(
        fusion,
        state_feature_dim=int(c.dense_hidden_dimension),
        decoder=c.decoder,
        iqn_embedding_dimension=int(c.iqn_embedding_dimension),
        n_actions=len(c.inputs),
        n_actions_per_block=int(c.n_actions_per_block),
        **iqn_btr_mlp_head_kw_from_config(c),
    )


def build_iqn_hf_vision_network_uncompiled(cfg: Any = None) -> IQNSharedBackboneNetwork:
    from config_files.config_loader import get_config
    from trackmania_rl.agents.policy_models.hf_actor_critic import build_hf_actor_critic

    c = cfg or get_config()
    fusion = build_hf_actor_critic(c, include_policy_heads=False)
    return IQNSharedBackboneNetwork(
        fusion,
        state_feature_dim=int(fusion.pre_trunk_feature_dim),
        decoder=c.decoder,
        iqn_embedding_dimension=int(c.iqn_embedding_dimension),
        n_actions=len(c.inputs),
        n_actions_per_block=int(c.n_actions_per_block),
        **iqn_btr_mlp_head_kw_from_config(c),
    )
