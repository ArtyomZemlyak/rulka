"""Group Relative Policy Optimization: per-group centered returns, REINFORCE-style policy gradient."""

from __future__ import annotations

import torch


def group_relative_advantages(returns: torch.Tensor, normalize: str) -> torch.Tensor:
    """returns: (K,) — higher is better. Returns detached advantages with zero mean within the group."""
    if normalize == "mean":
        adv = returns - returns.mean()
    elif normalize == "mean_std":
        adv = returns - returns.mean()
        std = returns.std(unbiased=False).clamp_min(1e-8)
        adv = adv / std
    else:
        raise ValueError(f"Unknown normalize_group {normalize!r}")
    return adv.detach()


def grpo_policy_objective(
    traj_log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Mean over group of -A_i * log π(τ_i), where traj_log_probs[i] = sum_t log π(a_t|s_t)."""
    return -(advantages * traj_log_probs).mean()
