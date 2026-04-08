"""Build GPU tensors from env rollout dicts (shared by PPO / DPO / GRPO learners)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from trackmania_rl import utilities
from trackmania_rl.agents.policy_optimization.rollout_rewards import ppo_rewards_and_dones_from_rollout


def _linear_schedule_scalar(cfg, scalar_attr: str, schedule_attr: str, step: int) -> float:
    sched = getattr(cfg, schedule_attr)
    scalar = float(getattr(cfg, scalar_attr))
    if sched:
        return float(utilities.from_linear_schedule(sched, step))
    return float(utilities.from_linear_schedule([[0, scalar]], step))


def ppo_scheduled_float(cfg, attr: str, schedule_attr: str, step: int) -> float:
    return _linear_schedule_scalar(cfg, attr, schedule_attr, step)


def grpo_scheduled_float(cfg, scalar_attr: str, schedule_attr: str, step: int) -> float:
    """GRPO hyperparams under ``grpo:`` — same linear schedule semantics as ``ppo_scheduled_float``."""
    return _linear_schedule_scalar(cfg, scalar_attr, schedule_attr, step)


def dpo_scheduled_float(cfg, scalar_attr: str, schedule_attr: str, step: int) -> float:
    """DPO hyperparams under ``dpo:`` — same linear schedule semantics as ``ppo_scheduled_float``."""
    return _linear_schedule_scalar(cfg, scalar_attr, schedule_attr, step)


def _scalar_rollout_shaping_gamma(cfg) -> float:
    """γ when no rollout γ schedule is configured (neither training nor ppo)."""
    pg = getattr(cfg, "policy_rollout_gamma", None)
    if pg is not None:
        return float(pg)
    return float(getattr(cfg, "gamma"))


def scheduled_rollout_shaping_gamma(cfg, schedule_step: int) -> float:
    """γ for potential-based shaping in on-policy rollouts (PPO / DPO / GRPO).

    Precedence: ``training.policy_rollout_gamma_schedule`` → ``ppo.ppo_gamma_schedule`` → scalar
    ``training.policy_rollout_gamma`` if set → ``ppo.gamma``.
    """
    pr_sched = getattr(cfg, "policy_rollout_gamma_schedule", None)
    if pr_sched:
        return float(utilities.from_linear_schedule(pr_sched, schedule_step))
    ppo_sched = getattr(cfg, "ppo_gamma_schedule", None)
    if ppo_sched:
        return ppo_scheduled_float(cfg, "gamma", "ppo_gamma_schedule", schedule_step)
    return _scalar_rollout_shaping_gamma(cfg)


def build_policy_rollout_tensors(
    rollout_results: dict[str, Any],
    end_race_stats: dict[str, Any],
    cfg,
    device: torch.device,
    schedule_step: int,
) -> dict[str, torch.Tensor] | None:
    """One on-policy rollout → tensors for policy optimization (PPO/DPO/GRPO)."""
    n_act = len(rollout_results.get("actions", []))
    n_lp = len(rollout_results.get("ppo_log_probs", []))
    if n_act < 2 or n_lp < 2:
        return None
    n = min(n_act, n_lp, len(rollout_results["frames"]), len(rollout_results["state_float"]))
    if n < 2:
        return None

    gamma_shaping = scheduled_rollout_shaping_gamma(cfg, schedule_step)
    eng_ss = utilities.from_linear_schedule(cfg.engineered_speedslide_reward_schedule, schedule_step)
    eng_ns = utilities.from_linear_schedule(cfg.engineered_neoslide_reward_schedule, schedule_step)
    eng_kk = utilities.from_linear_schedule(cfg.engineered_kamikaze_reward_schedule, schedule_step)
    eng_vcp = utilities.from_linear_schedule(cfg.engineered_close_to_vcp_reward_schedule, schedule_step)
    rewards_np, dones_np = ppo_rewards_and_dones_from_rollout(
        rollout_results,
        cfg,
        gamma=gamma_shaping,
        engineered_speedslide_reward=float(eng_ss),
        engineered_neoslide_reward=float(eng_ns),
        engineered_kamikaze_reward=float(eng_kk),
        engineered_close_to_vcp_reward=float(eng_vcp),
        race_finished=end_race_stats.get("race_finished"),
    )
    if len(rewards_np) < n:
        return None
    rewards_np = rewards_np[:n]
    dones_np = dones_np[:n]

    frames = rollout_results["frames"][:n]
    floats = np.stack(rollout_results["state_float"][:n]).astype(np.float32)

    img_list = []
    for fr in frames:
        x = np.asarray(fr, dtype=np.float32)
        if x.size == 1 or (x.ndim == 3 and x.shape[1] == 1 and x.shape[2] == 1):
            img_list.append(np.zeros((1, cfg.H_downsized, cfg.W_downsized), dtype=np.float32))
        else:
            img_list.append(((x.astype(np.float32) - 128.0) / 128.0))
    obs_img = np.stack(img_list, axis=0)
    obs_img_t = torch.from_numpy(obs_img).to(device=device, non_blocking=True)
    if not cfg.use_iqn_image_head:
        obs_img_t = torch.zeros((n, 1, cfg.H_downsized, cfg.W_downsized), device=device, dtype=torch.float32)

    obs_fl_t = torch.from_numpy(floats).to(device=device, non_blocking=True)

    old_vals = np.asarray(rollout_results["ppo_values"][:n], dtype=np.float32)
    old_logp = np.asarray(rollout_results["ppo_log_probs"][:n], dtype=np.float32)

    actions_raw = rollout_results["actions"][:n]
    if cfg.n_actions_per_block <= 1:
        actions_t = torch.tensor([int(a) for a in actions_raw], dtype=torch.long, device=device)
    else:
        actions_t = torch.tensor(np.stack([np.atleast_1d(a) for a in actions_raw]), dtype=torch.long, device=device)

    return {
        "obs_img": obs_img_t,
        "obs_float": obs_fl_t,
        "actions": actions_t,
        "old_logp": torch.from_numpy(old_logp).to(device=device),
        "old_values": torch.from_numpy(old_vals).to(device=device),
        "rewards": torch.from_numpy(rewards_np).to(device=device),
        "dones": torch.from_numpy(dones_np).to(device=device),
    }


def trajectory_return_scalar(
    rollout_results: dict[str, Any],
    end_race_stats: dict[str, Any],
    cfg,
    schedule_step: int,
) -> float:
    """Scalar for ranking trajectories (higher is better): sum of shaped step rewards."""
    batch = build_policy_rollout_tensors(rollout_results, end_race_stats, cfg, torch.device("cpu"), schedule_step)
    if batch is None:
        return float("-inf")
    return float(batch["rewards"].sum().item())
