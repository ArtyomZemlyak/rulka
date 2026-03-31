"""Build RL policy networks for BC pretrain (same topology as training, CPU, no compile / forced CUDA)."""

from __future__ import annotations

import torch.nn as nn

from config_files.config_loader import get_config
from trackmania_rl.agents.algorithms.ppo_wiring import build_ppo_policy_uncompiled
from trackmania_rl.agents.iqn import build_iqn_network_uncompiled
from trackmania_rl.pretrain.models import IQN_BC_MultiOffset, PpoPolicyBcMultiOffset


def freeze_ppo_heads_for_bc(model: nn.Module, *, freeze_policy: bool, freeze_value: bool) -> None:
    """Stop gradients into PPO actor-critic heads (BC uses CE on logits only)."""
    if not freeze_policy and not freeze_value:
        return
    for name, p in model.named_parameters():
        if freeze_value and "value_head" in name:
            p.requires_grad_(False)
        if freeze_policy and "policy_head" in name:
            p.requires_grad_(False)


def build_rl_policy_for_bc(*, n_bc_offsets: int, bc_multi_offset_mode: str) -> nn.Module:
    """Construct policy matching ``get_config()`` (RL YAML already loaded).

    Parameters
    ----------
    n_bc_offsets
        ``len(bc_time_offsets_ms)`` from BC config.
    bc_multi_offset_mode
        ``"separate_heads"`` or ``"fused"`` (IQN multi-offset only).
    """
    cfg = get_config()
    alg = cfg.algorithm
    if alg == "iqn":
        iqn = build_iqn_network_uncompiled()
        if n_bc_offsets <= 1:
            return iqn
        if bc_multi_offset_mode == "fused":
            if int(cfg.n_actions_per_block) != int(n_bc_offsets):
                raise ValueError(
                    "bc_multi_offset_mode=fused with bc_use_rl_architecture requires "
                    f"RL n_actions_per_block ({cfg.n_actions_per_block}) == "
                    f"len(bc_time_offsets_ms) ({n_bc_offsets})"
                )
            return iqn
        return IQN_BC_MultiOffset(iqn, n_bc_offsets)

    if alg != "ppo":
        raise ValueError(f"build_rl_policy_for_bc: unsupported training.algorithm {alg!r}")

    base = build_ppo_policy_uncompiled(cfg)

    n_act = len(cfg.inputs)
    if n_bc_offsets <= 1:
        freeze_ppo_heads_for_bc(base, freeze_policy=False, freeze_value=True)
        return base

    wrap = PpoPolicyBcMultiOffset(base, n_bc_offsets, n_act)
    freeze_ppo_heads_for_bc(base, freeze_policy=True, freeze_value=True)
    return wrap
