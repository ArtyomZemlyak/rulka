"""Hugging Face ``transformers`` layout for fusion policies (save_pretrained / from_pretrained).

``rulka_transformers`` JSON holds ``MultimodalTransformersConfig`` fields plus a ``vis`` copy of ``vision.transformer`` for roundtrip (same layout as before this refactor).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch import nn

from trackmania_rl import utilities as tr_utilities

_cfg_cls: type | None = None
_wrap_cls: type | None = None

# One shell per process + config fingerprint: avoids rebuilding fusion (HF/timm loads) on every periodic save.
_hf_save_shell: nn.Module | None = None
_hf_save_shell_cfg_key: str | None = None


def _hub_cfg_fingerprint(hub_cfg: Any) -> str:
    d = hub_cfg.to_dict()
    return json.dumps(d, sort_keys=True, default=str)


def _lazy_transformers():
    tr_utilities.quiet_transformers_weight_load_reports()
    try:
        from transformers import PreTrainedModel, PretrainedConfig
    except ImportError as e:
        raise ImportError(
            'Rulka fusion hub requires transformers. Install: pip install -e ".[policy]"'
        ) from e
    return PreTrainedModel, PretrainedConfig


def get_rulka_fusion_config_class() -> type:
    global _cfg_cls
    if _cfg_cls is not None:
        return _cfg_cls
    _, PretrainedConfig = _lazy_transformers()

    class RulkaMultimodalFusionConfig(PretrainedConfig):
        # Serialized hub id — keep string for existing save_pretrained checkpoints.
        model_type = "rulka_ppo_fusion"

        def __init__(
            self,
            float_inputs_dim: int = 128,
            float_hidden_dim: int = 256,
            dense_hidden_dim: int = 1024,
            image_h: int = 256,
            image_w: int = 256,
            use_image_head: bool = True,
            n_actions: int = 1,
            n_actions_per_block: int = 1,
            float_inputs_mean: list[float] | None = None,
            float_inputs_std: list[float] | None = None,
            rulka_transformers: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.float_inputs_dim = int(float_inputs_dim)
            self.float_hidden_dim = int(float_hidden_dim)
            self.dense_hidden_dim = int(dense_hidden_dim)
            self.image_h = int(image_h)
            self.image_w = int(image_w)
            self.use_image_head = bool(use_image_head)
            self.n_actions = int(n_actions)
            self.n_actions_per_block = int(n_actions_per_block)
            self.float_inputs_mean = list(float_inputs_mean or [])
            self.float_inputs_std = list(float_inputs_std or [])
            self.rulka_transformers = dict(rulka_transformers or {})

    try:
        from transformers import AutoConfig

        AutoConfig.register("rulka_ppo_fusion", RulkaMultimodalFusionConfig)
    except Exception:
        pass

    _cfg_cls = RulkaMultimodalFusionConfig
    return _cfg_cls


def _rulka_transformers_bundle_for_hub(cfg) -> dict[str, Any]:
    """Self-contained JSON: multimodal bundle + vision slots for hub roundtrip."""
    from config_files.nn_schema import TransformersConfig, infer_vis_branch

    d = cfg.to_multimodal().model_dump()
    d["vis"] = (cfg.vis.transformer or TransformersConfig()).model_dump()
    d["vis_branch"] = infer_vis_branch(cfg.vis)
    vis_cnn = getattr(cfg.vis, "cnn", None)
    if vis_cnn is not None and hasattr(vis_cnn, "model_dump"):
        d["vis_cnn"] = vis_cnn.model_dump()
    return d


def rulka_fusion_hub_config_from_run_config(cfg) -> Any:
    """Build ``RulkaMultimodalFusionConfig`` from a loaded Rulka config (for save / hub)."""
    RulkaMultimodalFusionConfig = get_rulka_fusion_config_class()
    mean = np.asarray(cfg.float_inputs_mean, dtype=np.float64).reshape(-1).tolist()
    std = np.asarray(cfg.float_inputs_std, dtype=np.float64).reshape(-1).tolist()
    return RulkaMultimodalFusionConfig(
        float_inputs_dim=int(cfg.float_input_dim),
        float_hidden_dim=int(cfg.float_hidden_dim_effective()),
        dense_hidden_dim=int(cfg.dense_hidden_dimension),
        image_h=int(cfg.H_downsized),
        image_w=int(cfg.W_downsized),
        use_image_head=bool(cfg.use_iqn_image_head),
        n_actions=len(cfg.inputs),
        n_actions_per_block=int(cfg.n_actions_per_block),
        float_inputs_mean=mean,
        float_inputs_std=std,
        rulka_transformers=_rulka_transformers_bundle_for_hub(cfg),
    )


def get_rulka_fusion_pretrained_class() -> type:
    global _wrap_cls
    if _wrap_cls is not None:
        return _wrap_cls
    PreTrainedModel, _ = _lazy_transformers()
    from config_files.nn_schema import MultimodalTransformersConfig, TransformersConfig, VisCnnBodyConfig
    from trackmania_rl.agents.policy_models.multimodal_torch_fusion import build_multimodal_fusion_from_transformers
    from trackmania_rl.nn_build.vis_cnn_head import vis_cnn_head_kw_from_body

    RulkaMultimodalFusionConfig = get_rulka_fusion_config_class()

    class RulkaMultimodalFusionPreTrainedModel(PreTrainedModel):
        config_class = RulkaMultimodalFusionConfig
        base_model_prefix = "policy"
        supports_gradient_checkpointing = False

        def __init__(self, config: RulkaMultimodalFusionConfig) -> None:
            super().__init__(config)

            raw = dict(config.rulka_transformers)
            vb = raw.pop("vis_branch", None)
            vis_raw = raw.pop("vis", None)
            vis_cnn_raw = raw.pop("vis_cnn", None)
            t = MultimodalTransformersConfig.model_validate(raw)
            vis_enc = TransformersConfig.model_validate(vis_raw) if vis_raw is not None else TransformersConfig()
            if vb is None:
                raise ValueError(
                    "Fusion hub config is missing rulka_transformers.vis_branch. "
                    "Re-save the checkpoint with a current Rulka version, or add vis_branch "
                    "(same values as config_files.nn_schema.infer_vis_branch for your nn.vis) to config.json."
                )
            mean = np.asarray(config.float_inputs_mean, dtype=np.float32)
            std = np.asarray(config.float_inputs_std, dtype=np.float32)
            vis_cnn_kw = None
            if vb == "cnn" and config.use_image_head and vis_cnn_raw is not None:
                vis_cnn_kw = vis_cnn_head_kw_from_body(VisCnnBodyConfig.model_validate(vis_cnn_raw))
            self.policy = build_multimodal_fusion_from_transformers(
                t,
                vis_enc,
                float_inputs_dim=config.float_inputs_dim,
                float_hidden_dim=config.float_hidden_dim,
                dense_hidden_dim=config.dense_hidden_dim,
                image_h=config.image_h,
                image_w=config.image_w,
                use_image_head=config.use_image_head,
                vis_branch=vb,
                float_inputs_mean=mean,
                float_inputs_std=std,
                n_actions=config.n_actions,
                n_actions_per_block=config.n_actions_per_block,
                vis_cnn_head_kw=vis_cnn_kw,
            )
            self.post_init()

        def forward(self, img: torch.Tensor, float_inputs: torch.Tensor):
            return self.policy(img, float_inputs)

        def evaluate_actions(
            self, img: torch.Tensor, float_inputs: torch.Tensor, actions: torch.Tensor
        ):
            return self.policy.evaluate_actions(img, float_inputs, actions)

        def _init_weights(self, module: nn.Module) -> None:
            return

    _wrap_cls = RulkaMultimodalFusionPreTrainedModel
    return _wrap_cls


def wrap_fusion_policy_for_hf_save(policy: nn.Module, cfg) -> nn.Module:
    """Wrap a live :class:`TorchMultimodalActorCritic` for ``save_pretrained``.

    Copies the **full** ``state_dict`` (HF vision backbone, fusion HF/native trunk, adapters,
    trunk MLP, policy/value heads) into the hub model so RL checkpoints round-trip.

    Reuses a single in-process ``RulkaMultimodalFusionPreTrainedModel`` shell when the hub config
    matches, so periodic checkpoints do not re-run ``from_pretrained`` on BERT/timm every time.
    """
    global _hf_save_shell, _hf_save_shell_cfg_key
    RulkaMultimodalFusionPreTrainedModel = get_rulka_fusion_pretrained_class()
    hub_cfg = rulka_fusion_hub_config_from_run_config(cfg)
    key = _hub_cfg_fingerprint(hub_cfg)
    if _hf_save_shell is None or _hf_save_shell_cfg_key != key:
        _hf_save_shell = RulkaMultimodalFusionPreTrainedModel(hub_cfg)
        _hf_save_shell_cfg_key = key
    _hf_save_shell.policy.load_state_dict(policy.state_dict(), strict=True)
    return _hf_save_shell


def load_fusion_policy_weights_from_hub(policy: nn.Module, pretrained_dir: str, *, trust_remote_code: bool) -> None:
    """Load ``policy`` weights from a directory written by ``save_pretrained``."""
    RulkaMultimodalFusionPreTrainedModel = get_rulka_fusion_pretrained_class()
    m = RulkaMultimodalFusionPreTrainedModel.from_pretrained(
        pretrained_dir,
        trust_remote_code=trust_remote_code,
    )
    policy.load_state_dict(m.policy.state_dict(), strict=True)
