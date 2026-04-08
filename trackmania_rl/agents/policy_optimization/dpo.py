"""Direct Preference Optimization loss for trajectory-level preferences (discrete actions).

Online pairing uses heuristic alignment: trajectory-level scalar score (sum of shaped rewards);
per-step log-probs are summed over min(T_win, T_lose) indices — see learner contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_preference_loss(
    logp_pi_win: torch.Tensor,
    logp_ref_win: torch.Tensor,
    logp_pi_lose: torch.Tensor,
    logp_ref_lose: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Scalar DPO loss: -log σ(β * (Δπ_win - Δπ_lose)) with Δ = log π_θ - log π_ref (trajectory totals).

    Args:
        logp_pi_* , logp_ref_*: 0-dim tensors (total log-prob under policy / ref for one trajectory).
    """
    z = beta * ((logp_pi_win - logp_ref_win) - (logp_pi_lose - logp_ref_lose))
    return -F.logsigmoid(z)


def sum_log_probs_evaluate(
    policy: torch.nn.Module,
    obs_img: torch.Tensor,
    obs_float: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sum of log π(a_t|s_t), mean entropy, and values for auxiliary critic."""
    logp, ent, vals, _ = policy.evaluate_actions(obs_img, obs_float, actions)
    return logp.sum(), ent.mean(), vals
