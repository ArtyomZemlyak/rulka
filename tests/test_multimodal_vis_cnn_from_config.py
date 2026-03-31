"""Multimodal fusion must honor ``nn.vis.cnn`` for the CNN vision stem (same as IQN / PPO CNN)."""

from __future__ import annotations

import numpy as np
import torch

from config_files.config_loader import load_config, set_config
from config_files.nn_schema import MultimodalTransformersConfig, TransformersConfig
from trackmania_rl.agents.iqn import ImpalaCNNBlock
from trackmania_rl.agents.policy_models.multimodal_torch_fusion import (
    build_multimodal_fusion_from_transformers,
    build_multimodal_fusion_uncompiled,
)


def test_post_concat_fusion_uses_impala_when_yaml_says_so() -> None:
    cfg = load_config("config_files/rl/config_btr_post_concat_cnn_transformer.yaml")
    set_config(cfg)
    m = build_multimodal_fusion_uncompiled(cfg)
    assert m.img_head is not None
    assert any(isinstance(x, ImpalaCNNBlock) for x in m.img_head.modules())


def test_explicit_baseline_cnn_kw_has_no_impala_blocks() -> None:
    d_in = 4
    vis_enc = TransformersConfig.model_validate({"d_model": 32, "n_heads": 4, "n_layers": 0, "patch_size": 8})
    enc_tr = TransformersConfig.model_validate(
        {"d_model": 32, "n_heads": 4, "n_layers": 1, "ff_mult": 2, "unified_float_tokens": 1, "post_concat_seq_len": 4}
    )
    t = MultimodalTransformersConfig.model_validate(
        {
            "fusion_mode": "post_concat",
            "transformer": enc_tr.model_dump(),
            "post_concat_layout": "token_sequence",
            "float_token_input": "raw",
            "float_token_layout": "per_feature",
        }
    )
    mean = np.zeros(d_in, dtype=np.float32)
    std = np.ones(d_in, dtype=np.float32)
    pol = build_multimodal_fusion_from_transformers(
        t,
        vis_enc,
        float_inputs_dim=d_in,
        float_hidden_dim=8,
        dense_hidden_dim=32,
        image_h=64,
        image_w=64,
        use_image_head=True,
        vis_branch="cnn",
        float_inputs_mean=mean,
        float_inputs_std=std,
        n_actions=2,
        n_actions_per_block=1,
        vis_cnn_head_kw={
            "use_impala_cnn": False,
            "impala_model_size": 2,
            "use_spectral_norm": False,
            "use_adaptive_maxpool": False,
            "adaptive_maxpool_size": 6,
        },
    )
    assert pol.img_head is not None
    assert not any(isinstance(x, ImpalaCNNBlock) for x in pol.img_head.modules())
    pol.eval()
    with torch.no_grad():
        out = pol(torch.randn(1, 1, 64, 64), torch.randn(1, d_in))
    assert tuple(out.logits.shape) == (1, 2)
