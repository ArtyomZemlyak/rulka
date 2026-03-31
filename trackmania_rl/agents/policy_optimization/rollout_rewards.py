"""Per-decision rewards and terminal mask for PPO from a TM rollout dict (no IQN replay)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from trackmania_rl.reward_vectorized import compute_dense_reward_per_action_t


def _fold_potential_into_ppo_step_rewards(
    dense_rewards_per_t: np.ndarray,
    potentials: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Potential-based shaping on per-decision dense rewards (Ng et al.), IQN 1-step equivalent.

    r'_t = r_t + γ Φ(s_{t+1}) - Φ(s_t); last step uses Φ(s') = 0 (terminal), i.e. r'_{T-1} -= Φ(s_{T-1}).

    Pass ``dense_rewards_per_t`` from :func:`trackmania_rl.reward_vectorized.compute_dense_reward_per_action_t`.
    """
    r = np.asarray(dense_rewards_per_t, dtype=np.float64)
    phi = np.asarray(potentials, dtype=np.float64)
    out = r.copy()
    if len(out) > 1:
        out[:-1] += gamma * phi[1:] - phi[:-1]
    if len(out) > 0:
        out[-1] -= phi[-1]
    return out.astype(np.float32)


class _RewardCfg(Protocol):
    n_actions_per_block: int
    ms_per_block: int
    ms_per_action: int
    n_prev_actions_in_inputs: int
    n_contact_material_physics_behavior_types: int
    constant_reward_per_ms: float
    reward_per_m_advanced_along_centerline: float
    final_speed_reward_per_m_per_s: float
    shaped_reward_dist_to_cur_vcp: float
    shaped_reward_min_dist_to_cur_vcp: float
    shaped_reward_max_dist_to_cur_vcp: float
    shaped_reward_point_to_vcp_ahead: float
    engineered_reward_min_dist_to_cur_vcp: float
    engineered_reward_max_dist_to_cur_vcp: float


def ppo_rewards_and_dones_from_rollout(
    rollout_results: dict[str, Any],
    cfg: _RewardCfg,
    *,
    gamma: float,
    engineered_speedslide_reward: float,
    engineered_neoslide_reward: float,
    engineered_kamikaze_reward: float,
    engineered_close_to_vcp_reward: float,
    race_finished: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-step rewards aligned with ``ppo_log_probs`` / ``ppo_values`` length.

    Same vectorized reward as IQN (``compute_rewards_into_and_potentials``), then
    :func:`compute_dense_reward_per_action_t` splits meter/final_speed/middle-constant (credited to
    a_{k-1} in slot k) vs engineered (credited to a_k), so each PPO timestep t gets r_t for action a_t;
    then potential fold with γ matching PPO GAE.

    Args:
        rollout_results: ``state_float``, ``actions``, ``meters_advanced_along_centerline``, …
        cfg: flat config with reward fields and ``final_speed_reward_per_m_per_s`` (from RewardsConfig validator).
        gamma: MDP discount for Φ folding (use scheduled PPO γ for consistency).
        engineered_*: scheduled coefficients (same as learner uses for IQN buffer fill).
        race_finished: optional; if False, rollout ended by cutoff (still terminal bootstrap on last step).

    Returns:
        rewards (T,), dones (T,) float32 with dones[t]=1 on last step (episode end).
    """
    _ = race_finished
    n = len(rollout_results["actions"])
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    n = min(
        n,
        len(rollout_results.get("ppo_log_probs", [])),
        len(rollout_results["frames"]),
        len(rollout_results["state_float"]),
    )
    if n < 2:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    n_ab = cfg.n_actions_per_block
    ms = cfg.ms_per_block if n_ab > 1 else cfg.ms_per_action

    state_float = torch.tensor(
        np.stack(rollout_results["state_float"][:n]), dtype=torch.float32
    )
    meters = torch.tensor(
        np.asarray(rollout_results["meters_advanced_along_centerline"][:n], dtype=np.float32),
        dtype=torch.float32,
    )
    if n_ab > 1:
        actions_arr = np.stack(rollout_results["actions"][:n]).astype(np.float64)
        actions_arr[np.isnan(actions_arr) | np.isinf(actions_arr)] = 0
        actions = torch.from_numpy(actions_arr.astype(np.int64))
    else:
        actions_raw = np.array(rollout_results["actions"][:n], dtype=np.float64)
        actions_raw = np.atleast_1d(actions_raw)
        actions_raw[np.isnan(actions_raw) | np.isinf(actions_raw)] = 0
        actions = torch.from_numpy(actions_raw.astype(np.int64)).unsqueeze(1)

    race_time_finished = "race_time" in rollout_results
    race_time = float(rollout_results.get("race_time", 0.0))

    dense_t, potentials = compute_dense_reward_per_action_t(
        state_float,
        meters,
        actions,
        cfg,
        n,
        race_time_finished,
        race_time,
        ms,
        engineered_speedslide_reward,
        engineered_neoslide_reward,
        engineered_kamikaze_reward,
        engineered_close_to_vcp_reward,
    )
    r_folded = _fold_potential_into_ppo_step_rewards(
        dense_t.detach().cpu().numpy(),
        potentials.detach().cpu().numpy(),
        gamma,
    )

    dones = np.zeros(n, dtype=np.float32)
    dones[-1] = 1.0
    return r_folded, dones
