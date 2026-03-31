"""Factories for ``torch.nn`` modules and config-derived kwargs.

Shared kwargs (avoid duplicating dicts across IQN / PPO / BC / multimodal):

- :mod:`trackmania_rl.nn_build.vis_cnn_head` — ``_build_img_head`` flags from ``nn.vis.cnn``.
- :mod:`trackmania_rl.nn_build.iqn_btr_from_config` — BTR MLP-head flags from flat ``cfg``.

Uncompiled PPO backbone routing: ``trackmania_rl.agents.algorithms.ppo_wiring.build_ppo_policy_uncompiled``.
"""

from trackmania_rl.nn_build.iqn_heads import (
    IQNTransformerTrunk,
    build_iqn_advantage_head,
    build_iqn_float_extractor,
    build_iqn_value_head,
)
from trackmania_rl.nn_build.iqn_btr_from_config import iqn_btr_mlp_head_kw_from_config
from trackmania_rl.nn_build.vis_cnn_head import (
    DEFAULT_VIS_CNN_HEAD_KW,
    default_vis_cnn_head_kw,
    merge_vis_cnn_head_kw,
    vis_cnn_head_kw_from_body,
    vis_cnn_head_kw_from_nn_vis,
    vis_cnn_head_kw_from_vis_cnn,
)

__all__ = [
    "DEFAULT_VIS_CNN_HEAD_KW",
    "iqn_btr_mlp_head_kw_from_config",
    "IQNTransformerTrunk",
    "build_iqn_advantage_head",
    "build_iqn_float_extractor",
    "build_iqn_value_head",
    "default_vis_cnn_head_kw",
    "merge_vis_cnn_head_kw",
    "vis_cnn_head_kw_from_body",
    "vis_cnn_head_kw_from_nn_vis",
    "vis_cnn_head_kw_from_vis_cnn",
]
