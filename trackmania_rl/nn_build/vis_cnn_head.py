"""Single source of truth for the RL vision CNN stem (``_build_img_head`` kwargs).

Every path that builds the same conv stack as IQN — classic ``IQN_Network``, PPO CNN actor,
multimodal fusion with ``vis_branch=cnn``, BC full-IQN, encoder injection checks — should take
kwargs from here so ``nn.vis.cnn`` (after ``btr`` merge) is honored consistently.
"""

from __future__ import annotations

from typing import Any, Mapping

from config_files.nn_schema import VisCnnBodyConfig

# Baseline 4-conv stack (matches legacy multimodal hard-code and pretrain default encoder).
DEFAULT_VIS_CNN_HEAD_KW: dict[str, Any] = {
    "use_impala_cnn": False,
    "impala_model_size": 2,
    "use_spectral_norm": False,
    "use_adaptive_maxpool": False,
    "adaptive_maxpool_size": 6,
}


def default_vis_cnn_head_kw() -> dict[str, Any]:
    return dict(DEFAULT_VIS_CNN_HEAD_KW)


def vis_cnn_head_kw_from_body(cnn: VisCnnBodyConfig) -> dict[str, Any]:
    return {
        "use_impala_cnn": bool(cnn.use_impala_cnn),
        "impala_model_size": int(cnn.impala_model_size),
        "use_spectral_norm": bool(cnn.use_spectral_norm),
        "use_adaptive_maxpool": bool(cnn.use_adaptive_maxpool),
        "adaptive_maxpool_size": int(cnn.adaptive_maxpool_size),
    }


def vis_cnn_head_kw_from_nn_vis(vis: Any) -> dict[str, Any]:
    """Resolve kwargs from ``nn.vis`` (duck-typed: ``no_image``, ``cnn``)."""
    if getattr(vis, "no_image", False):
        return default_vis_cnn_head_kw()
    cnn = getattr(vis, "cnn", None)
    if cnn is None:
        return default_vis_cnn_head_kw()
    return vis_cnn_head_kw_from_body(cnn)


def merge_vis_cnn_head_kw(user: Mapping[str, Any] | None) -> dict[str, Any]:
    """Defaults + optional partial overrides (hub ctor, tests)."""
    out = default_vis_cnn_head_kw()
    if user is not None:
        out.update(dict(user))
    return out


# Backward-compatible name used in hub code paths
vis_cnn_head_kw_from_vis_cnn = vis_cnn_head_kw_from_body
