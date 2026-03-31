"""Discrete actor-critic with Hugging Face vision backbone (``nn.fusion_mode == none`` + ``vis.transformer.use_hf_backbone``).

Uses ``nn.vis.transformer`` for backbone id and width/depth/dropout on top of HF.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trackmania_rl.agents.policy_models.hf_vision_utils import hf_vision_backbone_hidden_size
from trackmania_rl.agents.policy_optimization.ppo import discrete_action_logprob_and_entropy
from trackmania_rl.agents.policy_optimization.types import PolicyOutput
from trackmania_rl import utilities as tr_utilities


def _lazy_import_transformers():
    tr_utilities.quiet_transformers_weight_load_reports()
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as e:
        raise ImportError(
            'HfActorCritic requires transformers. Install: pip install -e ".[policy]"'
        ) from e
    return AutoImageProcessor, AutoModel


def _make_encoder(d_model: int, nhead: int, nlayers: int, dim_ff: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        dropout=dropout,
        batch_first=True,
        norm_first=False,
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=nlayers)


class HfActorCritic(nn.Module):
    """ViT (or compatible) CLS embedding fused with float features; policy + value heads."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        image_processor: Any,
        float_inputs_dim: int,
        float_hidden_dim: int,
        dense_hidden_dim: int,
        backbone_hidden_size: int,
        vis_d_model: int,
        vis_nhead: int,
        vis_nlayers: int,
        vis_ff_mult: int,
        vis_dropout: float,
        hidden_dropout_prob: float,
        float_inputs_mean: Tensor,
        float_inputs_std: Tensor,
        n_actions: int,
        n_actions_per_block: int,
        include_policy_heads: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_processor = image_processor
        self.n_actions = n_actions
        self.n_actions_per_block = n_actions_per_block
        self.include_policy_heads = bool(include_policy_heads)
        self.register_buffer("float_inputs_mean", float_inputs_mean)
        self.register_buffer("float_inputs_std", float_inputs_std)
        self._vis_d_model = int(vis_d_model)

        self.float_feature_extractor = nn.Sequential(
            nn.Linear(float_inputs_dim, float_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(float_hidden_dim, float_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.float_to_hidden = nn.Linear(float_hidden_dim, vis_d_model)
        self.cls_dropout = nn.Dropout(float(hidden_dropout_prob))
        self.vis_proj = (
            nn.Linear(backbone_hidden_size, vis_d_model)
            if backbone_hidden_size != vis_d_model
            else nn.Identity()
        )
        dim_ff = max(64, vis_d_model * vis_ff_mult)
        self.vis_refine: Optional[nn.TransformerEncoder] = (
            _make_encoder(vis_d_model, vis_nhead, vis_nlayers, dim_ff, vis_dropout)
            if vis_nlayers > 0
            else None
        )
        self._in_trunk = int(vis_d_model * 2)
        out_pi = n_actions * n_actions_per_block
        if self.include_policy_heads:
            self.trunk = nn.Sequential(
                nn.Linear(self._in_trunk, dense_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(dense_hidden_dim, dense_hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.policy_head = nn.Linear(dense_hidden_dim, out_pi)
            self.value_head = nn.Linear(dense_hidden_dim, 1)
        else:
            self.trunk = None
            self.policy_head = None
            self.value_head = None

    @property
    def pre_trunk_feature_dim(self) -> int:
        """Width of ``forward_fusion_hidden`` (CLS ‖ float token, each ``vis_d_model``)."""
        return self._in_trunk

    def _norm_float(self, x: Tensor) -> Tensor:
        return (x - self.float_inputs_mean) / self.float_inputs_std

    def _prepare_pixels(self, img: Tensor) -> Tensor:
        """img: (B, 1, H, W) float roughly in [-1, 1] from (uint8-128)/128 → map to processor space."""
        x = img
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        size = getattr(self.image_processor, "size", None)
        if isinstance(size, dict) and "height" in size:
            th, tw = int(size["height"]), int(size["width"])
        elif hasattr(self.image_processor, "size") and isinstance(self.image_processor.size, (tuple, list)):
            th, tw = int(self.image_processor.size[0]), int(self.image_processor.size[1])
        else:
            th, tw = 224, 224
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        x01 = (x.clamp(-1, 1) + 1.0) * 0.5
        proc = self.image_processor
        if hasattr(proc, "image_mean") and proc.image_mean is not None:
            mean = torch.tensor(proc.image_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor(proc.image_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            return (x01 - mean) / std
        return x01

    def forward_fusion_hidden(self, img: Tensor, float_inputs: Tensor) -> Tensor:
        """Concatenated CLS + float token (dim ``2 * vis_d_model``) before the shared MLP trunk; IQN uses this as state dim."""
        z = self._norm_float(float_inputs)
        z = self.float_feature_extractor(z)
        z = self.float_to_hidden(z)

        pix = self._prepare_pixels(img)
        out = self.backbone(pixel_values=pix)
        pool = out.last_hidden_state[:, 0]
        pool = self.cls_dropout(pool)
        pool = self.vis_proj(pool)
        if self.vis_refine is not None:
            pool = self.vis_refine(pool.unsqueeze(1)).squeeze(1)

        return torch.cat([pool, z], dim=1)

    def forward_features(self, img: Tensor, float_inputs: Tensor) -> Tensor:
        """Trunk output before policy/value heads (for BC multi-offset wrappers)."""
        h = self.forward_fusion_hidden(img, float_inputs)
        if self.trunk is None:
            return h
        return self.trunk(h)

    def forward(self, img: Tensor, float_inputs: Tensor) -> PolicyOutput:
        if self.policy_head is None or self.value_head is None:
            raise RuntimeError("HfActorCritic: policy heads omitted (IQN backbone-only build)")
        h = self.forward_features(img, float_inputs)
        logits = self.policy_head(h)
        v = self.value_head(h)
        return PolicyOutput(logits=logits, value=v)

    def evaluate_actions(
        self, img: Tensor, float_inputs: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, PolicyOutput]:
        out = self.forward(img, float_inputs)
        assert out.logits is not None
        if self.n_actions_per_block <= 1:
            logits = out.logits
            act = actions.reshape(-1)
        else:
            logits = out.logits.reshape(-1, self.n_actions_per_block, self.n_actions)
            act = actions.reshape(-1, self.n_actions_per_block)
        logp, ent = discrete_action_logprob_and_entropy(logits, act)
        return logp, ent, out.value.squeeze(-1), out


def build_hf_actor_critic(cfg, *, include_policy_heads: bool = True) -> HfActorCritic:
    AutoImageProcessor, AutoModel = _lazy_import_transformers()
    t = cfg.vis.transformer
    if t is None:
        raise ValueError("HfActorCritic requires nn.vis.transformer (ViT/HF backbone settings).")
    processor = AutoImageProcessor.from_pretrained(
        t.model_name_or_path,
        trust_remote_code=t.trust_remote_code,
    )
    backbone = AutoModel.from_pretrained(
        t.model_name_or_path,
        trust_remote_code=t.trust_remote_code,
    )
    cfg_h = getattr(backbone.config, "hidden_size", None)
    hidden = int(cfg_h) if cfg_h is not None else hf_vision_backbone_hidden_size(backbone)
    mean = torch.tensor(np.asarray(cfg.float_inputs_mean, dtype=np.float32))
    std = torch.tensor(np.asarray(cfg.float_inputs_std, dtype=np.float32))
    return HfActorCritic(
        backbone=backbone,
        image_processor=processor,
        float_inputs_dim=cfg.float_input_dim,
        float_hidden_dim=cfg.float_hidden_dim,
        dense_hidden_dim=cfg.dense_hidden_dimension,
        backbone_hidden_size=hidden,
        vis_d_model=int(t.d_model),
        vis_nhead=int(t.n_heads),
        vis_nlayers=int(t.n_layers),
        vis_ff_mult=int(t.ff_mult),
        vis_dropout=float(t.dropout),
        hidden_dropout_prob=float(t.hidden_dropout_prob),
        float_inputs_mean=mean,
        float_inputs_std=std,
        n_actions=len(cfg.inputs),
        n_actions_per_block=cfg.n_actions_per_block,
        include_policy_heads=bool(include_policy_heads),
    )


def make_hf_ppo_network_pair(cfg, jit: bool, is_inference: bool) -> Tuple[nn.Module, nn.Module]:
    uncompiled = build_hf_actor_critic(cfg)
    if not jit or not cfg.use_jit:
        model = uncompiled
    else:
        compile_mode = None if "rocm" in torch.__version__ else (
            "max-autotune" if is_inference else "max-autotune-no-cudagraphs"
        )
        model = torch.compile(uncompiled, mode=compile_mode)
    u = uncompiled.to(device="cuda", memory_format=torch.channels_last).train()
    m = model.to(device="cuda", memory_format=torch.channels_last).train()
    return m, u
