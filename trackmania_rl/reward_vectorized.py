"""Vectorized per-frame rewards and potentials (shared IQN buffer + PPO rollout)."""

from __future__ import annotations

from typing import Any

import torch


def state_float_slice_indices(cfg: Any) -> tuple[int, int, int, int, int, int, int]:
    """Layout: waypoint0, waypoint1, velocity, wheel contact blocks in state_float."""
    n_prev = cfg.n_prev_actions_in_inputs
    n_c = cfg.n_contact_material_physics_behavior_types
    prev_len = 4 * n_prev
    gear_len = 16 + 4 * n_c
    idx_gear_start = 1 + prev_len
    idx_wheels_start = idx_gear_start + 4
    idx_wheels_end = idx_gear_start + 8
    idx_vel_start = idx_gear_start + gear_len + 3
    idx_vel_end = idx_vel_start + 3
    idx_waypoint0_start = idx_vel_end + 3
    idx_waypoint0_end = idx_waypoint0_start + 3
    idx_waypoint1_end = idx_waypoint0_end + 3
    return (
        idx_waypoint0_start,
        idx_waypoint0_end,
        idx_waypoint1_end,
        idx_vel_start,
        idx_vel_end,
        idx_wheels_start,
        idx_wheels_end,
    )


def _per_slot_meter_reward(
    meters_advanced: torch.Tensor,
    reward_per_m: float,
    n_frames: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.zeros(n_frames, dtype=dtype, device=device)
    if n_frames >= 2 and float(reward_per_m) != 0.0:
        out[1:] = (meters_advanced[1:] - meters_advanced[:-1]) * float(reward_per_m)
    return out


def _per_slot_constant_reward_parts(
    cfg: Any,
    n_frames: int,
    race_time_finished: bool,
    race_time: float,
    ms_per_step: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Middle indices 1:-1 vs terminal slice on last frame (same as ``compute_rewards_into_and_potentials``)."""
    c = float(cfg.constant_reward_per_ms)
    mid = torch.zeros(n_frames, dtype=dtype, device=device)
    if n_frames >= 3:
        mid[1:-1] = c * float(ms_per_step)
    last = torch.zeros(n_frames, dtype=dtype, device=device)
    if n_frames >= 1:
        if race_time_finished:
            last[-1] = c * (float(race_time) - (n_frames - 2) * float(ms_per_step))
        else:
            last[-1] = c * float(ms_per_step)
    return mid, last


def _per_slot_final_speed_reward(
    state_float: torch.Tensor,
    cfg: Any,
    n_frames: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.zeros(n_frames, dtype=dtype, device=device)
    n_mid = min(n_frames - 2, state_float.shape[0] - 2)
    if n_mid <= 0 or float(cfg.final_speed_reward_per_m_per_s) == 0.0:
        return out
    w0_s, w0_e, w1_e, vel_s, vel_e, wh_s, wh_e = state_float_slice_indices(cfg)
    vel_forward = state_float[1 : 1 + n_mid, vel_s + 2]
    vel_norm_curr = torch.linalg.norm(state_float[1 : 1 + n_mid, vel_s:vel_e], dim=1)
    vel_norm_prev = torch.linalg.norm(state_float[:n_mid, vel_s:vel_e], dim=1)
    fwd_mask = vel_forward > 0
    out[1 : 1 + n_mid] += torch.where(
        fwd_mask,
        (vel_norm_curr - vel_norm_prev) * float(cfg.final_speed_reward_per_m_per_s),
        0.0,
    )
    return out


def compute_dense_reward_per_action_t(
    state_float: torch.Tensor,
    meters_advanced: torch.Tensor,
    actions: torch.Tensor,
    cfg: Any,
    n_frames: int,
    race_time_finished: bool,
    race_time: float,
    ms_per_step: float,
    engineered_speedslide_reward: float,
    engineered_neoslide_reward: float,
    engineered_kamikaze_reward: float,
    engineered_close_to_vcp_reward: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-decision dense reward r_t credited to (s_t, a_t) for PPO/GAE, plus Φ(s).

    Decomposes the same ``rewards_into`` as :func:`compute_rewards_into_and_potentials`:

    - At index k, meter / final_speed / constant (middle slots) follow the buffer convention
      (outcome of entering state k from a_{k-1}); they are shifted onto timestep k-1.
    - Engineered terms use (s_k, a_k) and stay on timestep k.
    - Terminal constant on the last frame stays on the last timestep.

    Sum over t equals sum of ``rewards_into`` (mass-preserving rearrangement).

    Returns:
        (dense_per_action_t, potentials) same shapes as ``rewards_into``.
    """
    rewards_into, potentials = compute_rewards_into_and_potentials(
        state_float,
        meters_advanced,
        actions,
        cfg,
        n_frames,
        race_time_finished,
        float(race_time),
        ms_per_step,
        engineered_speedslide_reward,
        engineered_neoslide_reward,
        engineered_kamikaze_reward,
        engineered_close_to_vcp_reward,
    )
    device = rewards_into.device
    dtype = rewards_into.dtype
    meter = _per_slot_meter_reward(
        meters_advanced, cfg.reward_per_m_advanced_along_centerline, n_frames, device=device, dtype=dtype
    )
    const_mid, const_last = _per_slot_constant_reward_parts(
        cfg, n_frames, race_time_finished, race_time, ms_per_step, device=device, dtype=dtype
    )
    speed = _per_slot_final_speed_reward(state_float, cfg, n_frames, device=device, dtype=dtype)
    shift = meter + speed + const_mid
    eng = rewards_into - shift - const_last
    dense = torch.zeros_like(rewards_into)
    if n_frames >= 2:
        dense[:-1] = shift[1:] + eng[:-1]
    if n_frames >= 1:
        dense[-1] = rewards_into[-1] - meter[-1] - speed[-1]
    return dense, potentials


def compute_rewards_into_and_potentials(
    state_float: torch.Tensor,
    meters_advanced: torch.Tensor,
    actions: torch.Tensor,
    cfg: Any,
    n_frames: int,
    race_time_finished: bool,
    race_time: float,
    ms_per_step: float,
    engineered_speedslide_reward: float,
    engineered_neoslide_reward: float,
    engineered_kamikaze_reward: float,
    engineered_close_to_vcp_reward: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Same semantics as the former inline block in buffer_management.fill_buffer_from_rollout.

    Returns:
        rewards_into: (n_frames,) immediate dense rewards indexed like the replay buffer
        potentials: (n_frames,) Φ(s) for potential-based shaping
    """
    n_states = state_float.shape[0]
    n_mid = min(n_frames - 2, n_states - 2)
    if n_mid < 0:
        n_mid = 0

    w0_s, w0_e, w1_e, vel_s, vel_e, wh_s, wh_e = state_float_slice_indices(cfg)

    rewards_into = torch.zeros(n_frames, dtype=torch.float32, device=state_float.device)

    rewards_into[1:-1] += cfg.constant_reward_per_ms * ms_per_step
    if race_time_finished:
        rewards_into[-1] += cfg.constant_reward_per_ms * (race_time - (n_frames - 2) * ms_per_step)
    else:
        rewards_into[-1] += cfg.constant_reward_per_ms * ms_per_step

    rewards_into[1:] += (meters_advanced[1:] - meters_advanced[:-1]) * cfg.reward_per_m_advanced_along_centerline

    if n_mid > 0:
        if cfg.final_speed_reward_per_m_per_s != 0:
            vel_forward = state_float[1 : 1 + n_mid, vel_s + 2]
            vel_norm_curr = torch.linalg.norm(state_float[1 : 1 + n_mid, vel_s:vel_e], dim=1)
            vel_norm_prev = torch.linalg.norm(state_float[:n_mid, vel_s:vel_e], dim=1)
            fwd_mask = vel_forward > 0
            rewards_into[1 : 1 + n_mid] += torch.where(
                fwd_mask,
                (vel_norm_curr - vel_norm_prev) * cfg.final_speed_reward_per_m_per_s,
                0.0,
            )

        if engineered_speedslide_reward != 0:
            wheels_ground_mask = torch.all(state_float[1 : 1 + n_mid, wh_s:wh_e] > 0, dim=1)
            lat = state_float[1 : 1 + n_mid, vel_s]
            fwd = state_float[1 : 1 + n_mid, vel_s + 2]
            ss_rewards = torch.zeros(n_mid, dtype=torch.float32, device=state_float.device)
            if wheels_ground_mask.any():
                from trackmania_rl.reward_shaping import speedslide_quality_tarmac_vectorized

                lat_np = lat[wheels_ground_mask].detach().cpu().numpy()
                fwd_np = fwd[wheels_ground_mask].detach().cpu().numpy()
                ss_qualities = torch.from_numpy(
                    speedslide_quality_tarmac_vectorized(lat_np, fwd_np).astype("float32")
                ).to(state_float.device)
                ss_rewards[wheels_ground_mask] = engineered_speedslide_reward * torch.clamp(
                    1.0 - torch.abs(ss_qualities - 1.0), min=0.0
                )
            rewards_into[1 : 1 + n_mid] += ss_rewards

        if engineered_neoslide_reward != 0:
            neo_mask = torch.abs(state_float[1 : 1 + n_mid, vel_s]) >= 2.0
            rewards_into[1 : 1 + n_mid] += torch.where(neo_mask, engineered_neoslide_reward, 0.0)

        if engineered_kamikaze_reward != 0:
            all_gas_only = torch.all(actions[1 : 1 + n_mid] <= 2, dim=1)
            kamikaze_mask = all_gas_only | (
                torch.sum(state_float[1 : 1 + n_mid, wh_s:wh_e] > 0, dim=1) <= 1
            )
            rewards_into[1 : 1 + n_mid] += torch.where(kamikaze_mask, engineered_kamikaze_reward, 0.0)

        if engineered_close_to_vcp_reward != 0:
            vcp_dist = torch.linalg.norm(state_float[1 : 1 + n_mid, w0_s:w0_e], dim=1)
            clamped_dist = torch.clamp(
                vcp_dist,
                min=cfg.engineered_reward_min_dist_to_cur_vcp,
                max=cfg.engineered_reward_max_dist_to_cur_vcp,
            )
            rewards_into[1 : 1 + n_mid] += engineered_close_to_vcp_reward * clamped_dist

    vcp_to_vcp = state_float[:, w0_e:w1_e] - state_float[:, w0_s:w0_e]
    vcp_to_vcp_norm = vcp_to_vcp / (torch.linalg.norm(vcp_to_vcp, dim=1, keepdim=True) + 1e-8)
    dist_cur_vcp = torch.linalg.norm(state_float[:, w0_s:w0_e], dim=1)
    clamped_dist_potential = torch.clamp(
        dist_cur_vcp,
        min=cfg.shaped_reward_min_dist_to_cur_vcp,
        max=cfg.shaped_reward_max_dist_to_cur_vcp,
    )
    potentials = (cfg.shaped_reward_dist_to_cur_vcp * clamped_dist_potential) + (
        cfg.shaped_reward_point_to_vcp_ahead * (vcp_to_vcp_norm[:, 2] - 1.0)
    )
    if len(potentials) < n_frames:
        potentials = torch.cat(
            [potentials, torch.zeros(n_frames - len(potentials), dtype=torch.float32, device=potentials.device)]
        )
    return rewards_into, potentials
