"""Shared helpers for Hugging Face vision backbones (ViT, DINO, TimmWrapper, …)."""

from __future__ import annotations

import torch.nn as nn


def hf_vision_backbone_hidden_size(hf_bb: nn.Module) -> int:
    """Feature width for HF vision checkpoints (ViT/DINO config.hidden_size, TimmWrapper timm_model.embed_dim, …)."""
    cfg = getattr(hf_bb, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    timm_model = getattr(hf_bb, "timm_model", None)
    if timm_model is not None:
        for attr in ("embed_dim", "num_features"):
            if hasattr(timm_model, attr):
                return int(getattr(timm_model, attr))
    raise TypeError(
        f"Cannot infer vision hidden size from {type(hf_bb).__name__}; "
        "expected config.hidden_size or timm_model.embed_dim / num_features."
    )
