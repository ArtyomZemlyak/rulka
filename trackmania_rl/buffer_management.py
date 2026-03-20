"""
This file's main entry point is the function fill_buffer_from_rollout_with_n_steps_rule().
Its main inputs are a rollout_results object (obtained from a GameInstanceManager object), and a buffer to be filled.
It reassembles the rollout_results object into transitions, as defined in /trackmania_rl/experience_replay/experience_replay_interface.py
"""

import math
import random

import numpy as np
from torchrl.data import ReplayBuffer

from config_files.config_loader import get_config
from trackmania_rl.experience_replay.experience_replay_interface import Experience


def _state_float_slice_indices(cfg):
    """Return (waypoint0_start, waypoint0_end, waypoint1_end, vel_start, vel_end, wheels_start, wheels_end) for state_float layout.

    gear_and_wheels sub-layout: is_sliding(4), has_ground_contact(4), damper_absorb(4), gearbox/gear/rpm/counter(4), contact(4*n_c).
    """
    n_prev = cfg.n_prev_actions_in_inputs
    n_c = cfg.n_contact_material_physics_behavior_types
    prev_len = 4 * n_prev
    gear_len = 16 + 4 * n_c
    idx_gear_start = 1 + prev_len
    idx_wheels_start = idx_gear_start + 4  # skip is_sliding(4)
    idx_wheels_end = idx_gear_start + 8    # has_ground_contact(4)
    idx_vel_start = idx_gear_start + gear_len + 3  # +3 ang_vel
    idx_vel_end = idx_vel_start + 3
    idx_waypoint0_start = idx_vel_end + 3  # +3 y_map
    idx_waypoint0_end = idx_waypoint0_start + 3
    idx_waypoint1_end = idx_waypoint0_end + 3
    return idx_waypoint0_start, idx_waypoint0_end, idx_waypoint1_end, idx_vel_start, idx_vel_end, idx_wheels_start, idx_wheels_end


def fill_buffer_from_rollout_with_n_steps_rule(
    buffer: ReplayBuffer,
    buffer_test: ReplayBuffer,
    rollout_results: dict,
    n_steps_max: int,
    gamma: float,
    discard_non_greedy_actions_in_nsteps: bool,
    engineered_speedslide_reward: float,
    engineered_neoslide_reward: float,
    engineered_kamikaze_reward: float,
    engineered_close_to_vcp_reward: float,
):
    import torch
    
    cfg = get_config()
    assert len(rollout_results["frames"]) == len(rollout_results["current_zone_idx"])
    n_frames = len(rollout_results["frames"])
    n_states = len(rollout_results["state_float"])
    # Collector/TMI can produce off-by-one in edge cases (dummy frame, race end). Align all to same length.
    required_keys = ("frames", "state_float", "current_zone_idx", "actions", "action_was_greedy", "meters_advanced_along_centerline")
    n_align = min(len(rollout_results.get(k, [])) for k in required_keys if k in rollout_results)
    n_frames = n_align
    for key in required_keys:
        if key in rollout_results and len(rollout_results[key]) > n_frames:
            rollout_results[key] = rollout_results[key][:n_frames]
    
    if n_frames <= 1:
        return buffer, buffer_test, 0, 0

    number_memories_added_train = 0
    number_memories_added_test = 0

    # =========================================================================
    # 1. Convert all inputs to PyTorch tensors for vectorized processing
    # =========================================================================
    n_ab = cfg.n_actions_per_block
    ms_per_step = cfg.ms_per_block if n_ab > 1 else cfg.ms_per_action

    # state_float: shape (N, D)
    state_float = torch.tensor(np.stack(rollout_results["state_float"]), dtype=torch.float32)
    # rollout_results["actions"]: list of int (N=1) or list of np.ndarray shape (N,) (multi-action).
    if n_ab > 1:
        actions_arr = np.stack(rollout_results["actions"]).astype(np.float64)
        actions_arr[np.isnan(actions_arr) | np.isinf(actions_arr)] = 0
        actions = torch.from_numpy(actions_arr.astype(np.int64))  # (n_frames, N)
    else:
        actions_raw = np.array(rollout_results["actions"], dtype=np.float64)
        actions_raw = np.atleast_1d(actions_raw)
        actions_raw[np.isnan(actions_raw) | np.isinf(actions_raw)] = 0
        actions = torch.from_numpy(actions_raw.astype(np.int64)).unsqueeze(1)  # (n_frames, 1)

    action_was_greedy = torch.tensor(rollout_results["action_was_greedy"], dtype=torch.bool)
    meters_advanced = torch.tensor(rollout_results["meters_advanced_along_centerline"], dtype=torch.float32)
    race_time_finished = "race_time" in rollout_results
    race_time = rollout_results.get("race_time", 0.0)

    gammas_arr = (gamma ** np.linspace(1, n_steps_max, n_steps_max)).astype(np.float32)

    # =========================================================================
    # 2. Vectorized Step-by-Step Reward Calculation
    # =========================================================================
    n_states = state_float.shape[0]
    # Align length: frames and state_float can differ by 1 (e.g. T+1 frames vs T states)
    n_mid = min(n_frames - 2, n_states - 2)
    if n_mid < 0:
        n_mid = 0

    w0_s, w0_e, w1_e, vel_s, vel_e, wh_s, wh_e = _state_float_slice_indices(cfg)

    rewards_into = torch.zeros(n_frames, dtype=torch.float32)

    # Base constant reward (per decision step: one block when multi-action, one action when single)
    rewards_into[1:-1] += cfg.constant_reward_per_ms * ms_per_step
    if race_time_finished:
        rewards_into[-1] += cfg.constant_reward_per_ms * (race_time - (n_frames - 2) * ms_per_step)
    else:
        rewards_into[-1] += cfg.constant_reward_per_ms * ms_per_step
        
    # Reward for meters advanced
    rewards_into[1:] += (meters_advanced[1:] - meters_advanced[:-1]) * cfg.reward_per_m_advanced_along_centerline
    
    # Vectorized physics rewards (only applied to non-terminal frames: indices 1 to n_frames-2)
    if n_mid > 0:
        # V Forward diff
        if cfg.final_speed_reward_per_m_per_s != 0:
            vel_forward = state_float[1 : 1 + n_mid, vel_s + 2]  # forward component
            vel_norm_curr = torch.linalg.norm(state_float[1 : 1 + n_mid, vel_s:vel_e], dim=1)
            vel_norm_prev = torch.linalg.norm(state_float[:n_mid, vel_s:vel_e], dim=1)
            fwd_mask = vel_forward > 0
            rewards_into[1 : 1 + n_mid] += torch.where(fwd_mask, (vel_norm_curr - vel_norm_prev) * cfg.final_speed_reward_per_m_per_s, 0.0)

        # Speedslide reward
        if engineered_speedslide_reward != 0:
            wheels_ground_mask = torch.all(state_float[1 : 1 + n_mid, wh_s:wh_e] > 0, dim=1)
            lat = state_float[1 : 1 + n_mid, vel_s]
            fwd = state_float[1 : 1 + n_mid, vel_s + 2]

            ss_rewards = torch.zeros(n_mid, dtype=torch.float32)
            if wheels_ground_mask.any():
                from trackmania_rl.reward_shaping import speedslide_quality_tarmac_vectorized
                lat_np = lat[wheels_ground_mask].numpy()
                fwd_np = fwd[wheels_ground_mask].numpy()
                ss_qualities = torch.from_numpy(speedslide_quality_tarmac_vectorized(lat_np, fwd_np).astype(np.float32))
                ss_rewards[wheels_ground_mask] = engineered_speedslide_reward * torch.clamp(1.0 - torch.abs(ss_qualities - 1.0), min=0.0)
            rewards_into[1 : 1 + n_mid] += ss_rewards

        # Neoslide
        if engineered_neoslide_reward != 0:
            neo_mask = torch.abs(state_float[1 : 1 + n_mid, vel_s]) >= 2.0
            rewards_into[1 : 1 + n_mid] += torch.where(neo_mask, engineered_neoslide_reward, 0.0)

        # Kamikaze: penalize if ALL actions in block are gas-only (idx<=2) OR airborne
        if engineered_kamikaze_reward != 0:
            all_gas_only = torch.all(actions[1 : 1 + n_mid] <= 2, dim=1)
            kamikaze_mask = all_gas_only | (torch.sum(state_float[1 : 1 + n_mid, wh_s:wh_e] > 0, dim=1) <= 1)
            rewards_into[1 : 1 + n_mid] += torch.where(kamikaze_mask, engineered_kamikaze_reward, 0.0)

        # Close to VCP
        if engineered_close_to_vcp_reward != 0:
            vcp_dist = torch.linalg.norm(state_float[1 : 1 + n_mid, w0_s:w0_e], dim=1)
            clamped_dist = torch.clamp(vcp_dist, min=cfg.engineered_reward_min_dist_to_cur_vcp, max=cfg.engineered_reward_max_dist_to_cur_vcp)
            rewards_into[1 : 1 + n_mid] += engineered_close_to_vcp_reward * clamped_dist

    # =========================================================================
    # 3. Vectorized Potentials
    # =========================================================================
    vcp_to_vcp = state_float[:, w0_e:w1_e] - state_float[:, w0_s:w0_e]
    vcp_to_vcp_norm = vcp_to_vcp / (torch.linalg.norm(vcp_to_vcp, dim=1, keepdim=True) + 1e-8)
    dist_cur_vcp = torch.linalg.norm(state_float[:, w0_s:w0_e], dim=1)
    clamped_dist_potential = torch.clamp(dist_cur_vcp, min=cfg.shaped_reward_min_dist_to_cur_vcp, max=cfg.shaped_reward_max_dist_to_cur_vcp)
    
    potentials = (cfg.shaped_reward_dist_to_cur_vcp * clamped_dist_potential) + \
                 (cfg.shaped_reward_point_to_vcp_ahead * (vcp_to_vcp_norm[:, 2] - 1.0))
    if len(potentials) < n_frames:
        potentials = torch.cat([potentials, torch.zeros(n_frames - len(potentials), dtype=torch.float32)])

    # =========================================================================
    # 4. Extracting Experiences
    # =========================================================================
    # Build transition buffers
    ValidIdx = n_frames - 1
    
    Experiences_For_Buffer = []
    Experiences_For_Buffer_Test = []

    # Pre-compute N-step discounted rewards via unfold (zero-copy sliding window)
    gamma_vec = gamma ** torch.arange(n_steps_max, dtype=torch.float32)
    padded_rewards = torch.cat([rewards_into[1:], torch.zeros(n_steps_max, dtype=torch.float32)])
    raw_step_rewards = padded_rewards.unfold(0, n_steps_max, 1)[:n_frames]  # (n_frames, n_steps_max) — view, non-contiguous
    accum_discounted_rewards = torch.cumsum(raw_step_rewards.contiguous() * gamma_vec, dim=1)

    # Pre-convert to numpy once (avoid per-iteration torch→numpy bridge overhead)
    accum_rewards_np = accum_discounted_rewards.numpy()
    potentials_np = potentials.numpy()
    actions_np = actions.numpy()
    greedy_np = action_was_greedy.numpy() if discard_non_greedy_actions_in_nsteps else None
    frames = rollout_results["frames"]
    state_floats = rollout_results["state_float"]
    test_ratio = cfg.buffer_test_ratio

    for i in range(ValidIdx):
        n_steps = min(n_steps_max, n_frames - 1 - i)

        if greedy_np is not None:
            for j in range(1, n_steps):
                if not greedy_np[i + j]:
                    n_steps = j
                    break

        terminal_actions = float((n_frames - 1) - i) if race_time_finished else math.inf
        next_state_has_passed_finish = ((i + n_steps) == (n_frames - 1)) and race_time_finished
        next_idx = i + n_steps if not next_state_has_passed_finish else i

        is_test = random.random() < test_ratio or random.random() < 0.1
        list_to_fill = Experiences_For_Buffer_Test if is_test else Experiences_For_Buffer

        list_to_fill.append(Experience(
            frames[i],
            state_floats[i],
            float(potentials_np[i]),
            actions_np[i],
            n_steps,
            accum_rewards_np[i],
            frames[next_idx] if not next_state_has_passed_finish else frames[i],
            state_floats[next_idx] if not next_state_has_passed_finish else state_floats[i],
            float(potentials_np[next_idx]) if not next_state_has_passed_finish else 0.0,
            gammas_arr,
            terminal_actions,
        ))

    number_memories_added_train += len(Experiences_For_Buffer)
    if len(Experiences_For_Buffer) > 1:
        buffer.extend(Experiences_For_Buffer)
    elif len(Experiences_For_Buffer) == 1:
        buffer.add(Experiences_For_Buffer[0])
        
    number_memories_added_test += len(Experiences_For_Buffer_Test)
    if len(Experiences_For_Buffer_Test) > 1:
        buffer_test.extend(Experiences_For_Buffer_Test)
    elif len(Experiences_For_Buffer_Test) == 1:
        buffer_test.add(Experiences_For_Buffer_Test[0])

    return buffer, buffer_test, number_memories_added_train, number_memories_added_test

