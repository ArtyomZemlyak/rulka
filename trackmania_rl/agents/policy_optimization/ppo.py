"""
PPO objective and GAE (Schulman et al., arXiv:1707.06347; see also Spinning Up PPO).

Notation (per transition t, batch flattened or vectorized):
  r_t(θ) = exp(log π_θ(a_t|s_t) - log π_old(a_t|s_t))
  L^CLIP(θ) = E_t [ min( r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]

Value target: MSE(V_θ, R) or PPO-style clipped value loss vs V_old (Schulman et al.; CleanRL ``clip_vloss``).

GAE (Schulman et al.): generalized advantage with λ, γ and done mask.
Aligned with common single-file references (e.g. CleanRL ppo.py) for masks and bootstrap.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        rewards: (T,)
        values: (T,) value predictions V(s_t) at collection time
        dones: (T,) 1.0 if episode ended after step t (no bootstrap from s_{t+1}), else 0.0
        next_value: scalar or (1,) V(s_T) after last state (often 0 if terminal)
        gamma, gae_lambda: discount and GAE λ

    Returns:
        advantages (T,), returns (T,) where returns = advantages + values
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            nextnonterminal = 1.0 - dones[t]
            next_v = next_value
        else:
            nextnonterminal = 1.0 - dones[t]
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def ppo_loss_components(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    *,
    old_values: Optional[torch.Tensor] = None,
    clip_coef_vf: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Clipped surrogate + value loss + entropy bonus (maximize entropy => minimize -entropy).

    Args:
        new_logprob, old_logprob: (B,) for taken actions
        advantages, returns: (B,) — advantages often normalized per minibatch outside
        new_values: (B,) or (B,1) squeezed
        entropy: (B,) mean entropy per sample, or scalar broadcast
        clip_coef: ε for policy ratio
        vf_coef, ent_coef: SB3-style coefficients
        old_values: V(s) at collection time; required with ``clip_coef_vf`` for clipped vf loss
        clip_coef_vf: if set and > 0 with ``old_values``, value loss uses
            0.5 * mean(max((V−R)², (clip(V)−R)²)) with clip(V)=V_old+clamp(V−V_old,±clip_coef_vf)

    Returns:
        total_loss scalar, dict of detached scalars for logging
    """
    logratio = new_logprob - old_logprob
    ratio = logratio.exp()

    adv = advantages
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = torch.mean(torch.maximum(pg_loss1, pg_loss2))

    new_v = new_values.reshape(-1)
    ret = returns.reshape(-1)
    use_vf_clip = (
        old_values is not None
        and clip_coef_vf is not None
        and float(clip_coef_vf) > 0.0
    )
    if use_vf_clip:
        old_v = old_values.reshape(-1).to(device=new_v.device, dtype=new_v.dtype)
        eps = float(clip_coef_vf)
        v_pred_clipped = old_v + torch.clamp(new_v - old_v, -eps, eps)
        v_loss_unclipped = (new_v - ret) ** 2
        v_loss_clipped = (v_pred_clipped - ret) ** 2
        v_loss = 0.5 * torch.mean(torch.maximum(v_loss_unclipped, v_loss_clipped))
    else:
        v_loss = 0.5 * torch.mean((new_v - ret) ** 2)

    if entropy.ndim > 0:
        ent = entropy.mean()
    else:
        ent = entropy

    loss = pg_loss + vf_coef * v_loss - ent_coef * ent

    with torch.no_grad():
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_coef).float())
        approx_kl = torch.mean((ratio - 1.0) - logratio)
        if use_vf_clip:
            old_vd = old_values.reshape(-1)
            vf_clipfrac = torch.mean(
                (torch.abs(new_v - old_vd.to(device=new_v.device, dtype=new_v.dtype)) > eps).float()
            )
        else:
            vf_clipfrac = torch.zeros((), device=new_v.device, dtype=new_v.dtype)

    metrics = {
        "loss_policy": pg_loss.detach(),
        "loss_value": v_loss.detach(),
        "loss_entropy": (-ent).detach(),
        "clipfrac": clipfrac,
        "approx_kl": approx_kl,
        "vf_clipfrac": vf_clipfrac,
    }
    return loss, metrics


def discrete_action_logprob_and_entropy(
    logits: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Categorical log π(a|s) and entropy.

    - logits (B, n_actions), actions (B,)
    - logits (B, k, n_actions), actions (B, k) — independent heads, sum log-prob and entropy.
    """
    if logits.dim() == 2:
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()
    if logits.dim() == 3:
        k = logits.shape[1]
        logps = []
        ents = []
        for i in range(k):
            d = torch.distributions.Categorical(logits=logits[:, i])
            logps.append(d.log_prob(actions[:, i]))
            ents.append(d.entropy())
        return torch.stack(logps, dim=1).sum(dim=1), torch.stack(ents, dim=1).sum(dim=1)
    raise ValueError(f"Unsupported logits shape {logits.shape} with actions {actions.shape}")
