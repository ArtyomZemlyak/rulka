"""
This file's main entry point is the function fill_buffer_from_rollout_with_n_steps_rule().
Its main inputs are a rollout_results object (obtained from a GameInstanceManager object), and a buffer to be filled.
It reassembles the rollout_results object into transitions, as defined in /trackmania_rl/experience_replay/experience_replay_interface.py
"""

import math
import random

import numpy as np
from numba import jit
from torchrl.data import ReplayBuffer

from config_files.config_loader import get_config
from trackmania_rl.experience_replay.experience_replay_interface import Experience
from trackmania_rl.reward_shaping import speedslide_quality_tarmac


@jit(nopython=True)
def get_potential(
    state_float,
    shaped_reward_dist_to_cur_vcp: float,
    shaped_reward_min_dist_to_cur_vcp: float,
    shaped_reward_max_dist_to_cur_vcp: float,
    shaped_reward_point_to_vcp_ahead: float,
):
    # https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    vector_vcp_to_vcp_further_ahead = state_float[65:68] - state_float[62:65]
    vector_vcp_to_vcp_further_ahead_normalized = vector_vcp_to_vcp_further_ahead / np.linalg.norm(vector_vcp_to_vcp_further_ahead)

    return (
        shaped_reward_dist_to_cur_vcp
        * max(
            shaped_reward_min_dist_to_cur_vcp,
            min(shaped_reward_max_dist_to_cur_vcp, np.linalg.norm(state_float[62:65])),
        )
    ) + (shaped_reward_point_to_vcp_ahead * (vector_vcp_to_vcp_further_ahead_normalized[2] - 1))


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
    # state_float: shape (N, D)
    state_float = torch.tensor(np.stack(rollout_results["state_float"]), dtype=torch.float32)
    # rollout_results["actions"] can sometimes contain floats or NaNs due to how TMI deals with invalid frames.
    # Convert safely by passing it to float first, fixing NaNs, and then casting to integer
    actions_raw = np.array(rollout_results["actions"], dtype=np.float32)
    actions_raw[np.isnan(actions_raw) | np.isinf(actions_raw)] = 0
    actions = torch.tensor(actions_raw, dtype=torch.int64)

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

    rewards_into = torch.zeros(n_frames, dtype=torch.float32)
    
    # Base constant reward
    rewards_into[1:-1] += cfg.constant_reward_per_ms * cfg.ms_per_action
    if race_time_finished:
        rewards_into[-1] += cfg.constant_reward_per_ms * (race_time - (n_frames - 2) * cfg.ms_per_action)
    else:
        rewards_into[-1] += cfg.constant_reward_per_ms * cfg.ms_per_action
        
    # Reward for meters advanced
    rewards_into[1:] += (meters_advanced[1:] - meters_advanced[:-1]) * cfg.reward_per_m_advanced_along_centerline
    
    # Vectorized physics rewards (only applied to non-terminal frames: indices 1 to n_frames-2)
    if n_mid > 0:
        # V Forward diff
        if cfg.final_speed_reward_per_m_per_s != 0:
            vel_forward = state_float[1 : 1 + n_mid, 58]
            vel_norm_curr = torch.linalg.norm(state_float[1 : 1 + n_mid, 56:59], dim=1)
            vel_norm_prev = torch.linalg.norm(state_float[:n_mid, 56:59], dim=1)
            fwd_mask = vel_forward > 0
            rewards_into[1 : 1 + n_mid] += torch.where(fwd_mask, (vel_norm_curr - vel_norm_prev) * cfg.final_speed_reward_per_m_per_s, 0.0)

        # Speedslide reward
        if engineered_speedslide_reward != 0:
            wheels_ground_mask = torch.all(state_float[1 : 1 + n_mid, 25:29] > 0, dim=1)
            lat = state_float[1 : 1 + n_mid, 56]
            fwd = state_float[1 : 1 + n_mid, 58]
            
            from trackmania_rl.reward_shaping import speedslide_quality_tarmac
            ss_qualities = torch.tensor([
                speedslide_quality_tarmac(l.item(), f.item())
                for (l, f, mask) in zip(lat, fwd, wheels_ground_mask) if mask.item()
            ], dtype=torch.float32)
            
            ss_rewards = torch.zeros(n_mid, dtype=torch.float32)
            if len(ss_qualities) > 0:
                ss_rewards[wheels_ground_mask] = engineered_speedslide_reward * torch.clamp(1.0 - torch.abs(ss_qualities - 1.0), min=0.0)
            rewards_into[1 : 1 + n_mid] += ss_rewards

        # Neoslide
        if engineered_neoslide_reward != 0:
            neo_mask = torch.abs(state_float[1 : 1 + n_mid, 56]) >= 2.0
            rewards_into[1 : 1 + n_mid] += torch.where(neo_mask, engineered_neoslide_reward, 0.0)
            
        # Kamikaze
        if engineered_kamikaze_reward != 0:
            kamikaze_mask = (actions[1 : 1 + n_mid] <= 2) | (torch.sum(state_float[1 : 1 + n_mid, 25:29] > 0, dim=1) <= 1)
            rewards_into[1 : 1 + n_mid] += torch.where(kamikaze_mask, engineered_kamikaze_reward, 0.0)

        # Close to VCP
        if engineered_close_to_vcp_reward != 0:
            vcp_dist = torch.linalg.norm(state_float[1 : 1 + n_mid, 62:65], dim=1)
            clamped_dist = torch.clamp(vcp_dist, min=cfg.engineered_reward_min_dist_to_cur_vcp, max=cfg.engineered_reward_max_dist_to_cur_vcp)
            rewards_into[1 : 1 + n_mid] += engineered_close_to_vcp_reward * clamped_dist

    # =========================================================================
    # 3. Vectorized Potentials
    # =========================================================================
    vcp_to_vcp = state_float[:, 65:68] - state_float[:, 62:65]
    vcp_to_vcp_norm = vcp_to_vcp / (torch.linalg.norm(vcp_to_vcp, dim=1, keepdim=True) + 1e-8)
    dist_cur_vcp = torch.linalg.norm(state_float[:, 62:65], dim=1)
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

    # Optimization: Pre-compute N-step rewards for all starting indices
    # We do a rolling sum discount. 
    # Because n_frames can be thousands and n_steps_max is small (e.g. 5), a loop over n_steps_max is super fast
    # Rewards_n_step_matrix initialized to max potential shape (ValidIdx, n_steps_max)
    R_matrix = torch.zeros((n_frames, n_steps_max), dtype=torch.float32)
    gamma_vec = gamma ** torch.arange(n_steps_max, dtype=torch.float32)
    
    # We want R_matrix[i, j] = reward_into[i + j + 1] (discounted later or sequentially accumulated)
    # R_out[i, k] = sum_{j=0}^{k} (gamma**j) * reward_into[i+1+j]
    padded_rewards = torch.cat([rewards_into, torch.zeros(n_steps_max, dtype=torch.float32)])
    raw_step_rewards = torch.stack([padded_rewards[j+1 : j+1+n_frames] for j in range(n_steps_max)], dim=1) # (n_frames, n_steps_max)
    raw_discounted = raw_step_rewards * gamma_vec.unsqueeze(0)
    accum_discounted_rewards = torch.cumsum(raw_discounted, dim=1) # (n_frames, n_steps_max)

    for i in range(ValidIdx):
        n_steps = min(n_steps_max, n_frames - 1 - i)
        
        if discard_non_greedy_actions_in_nsteps:
            # Find first non-greedy in the window
            window_greedy = action_was_greedy[(i + 1) : (i + n_steps)]
            first_false_idx = (window_greedy == False).nonzero(as_tuple=True)[0]
            if len(first_false_idx) > 0:
                n_steps = min(n_steps, first_false_idx[0].item() + 1)

        final_rewards = accum_discounted_rewards[i, :n_steps_max].numpy()
        
        terminal_actions = float((n_frames - 1) - i) if race_time_finished else math.inf
        next_state_has_passed_finish = ((i + n_steps) == (n_frames - 1)) and race_time_finished

        next_idx = i + n_steps if not next_state_has_passed_finish else i
        
        is_test = random.random() < cfg.buffer_test_ratio or random.random() < 0.1
        list_to_fill = Experiences_For_Buffer_Test if is_test else Experiences_For_Buffer
        
        list_to_fill.append( Experience(
            rollout_results["frames"][i],
            rollout_results["state_float"][i],
            potentials[i].item(),
            actions[i].item(),
            n_steps,
            final_rewards,
            rollout_results["frames"][next_idx] if not next_state_has_passed_finish else rollout_results["frames"][i],
            rollout_results["state_float"][next_idx] if not next_state_has_passed_finish else rollout_results["state_float"][i],
            potentials[next_idx].item() if not next_state_has_passed_finish else 0.0,
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

