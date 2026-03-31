"""IQN dense-head knobs from flat RL config (``btr:`` → ``cfg.use_layer_norm`` etc.)."""

from __future__ import annotations

from typing import Any


def iqn_btr_mlp_head_kw_from_config(cfg: Any) -> dict[str, Any]:
    """Kwargs for ``IQN_Network`` / ``IQNSharedBackboneNetwork`` BTR-style MLP heads (NoisyNet, LayerNorm)."""
    return {
        "use_layer_norm": bool(cfg.use_layer_norm),
        "use_noisy_linear": bool(cfg.use_noisy_linear),
        "noisy_sigma0": float(cfg.noisy_sigma0),
    }
