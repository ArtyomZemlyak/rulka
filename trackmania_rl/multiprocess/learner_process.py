"""
This file implements the main training loop, tensorboard statistics tracking, etc...
"""

import copy
import math
import random
import sys
import time
import threading
import typing
from collections import defaultdict, deque
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import PrioritizedSampler

from config_files.config_loader import get_config
from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents import iqn as iqn
from trackmania_rl.agents.iqn import make_untrained_iqn_network
from trackmania_rl.analysis_metrics import (
    distribution_curves,
    highest_prio_transitions,
    loss_distribution,
    race_time_left_curves,
    tau_curves,
)
from trackmania_rl.buffer_utilities import make_buffers, resize_buffers
from trackmania_rl.map_reference_times import reference_times


def _get_frozen_param_prefixes():
    """Return parameter name prefixes to freeze (e.g. 'img_head.') from config _freeze flags."""
    cfg = get_config()
    prefixes = []
    if getattr(cfg, "pretrain_encoder_freeze", False):
        prefixes.append("img_head.")
    if getattr(cfg, "pretrain_float_head_freeze", False):
        prefixes.append("float_feature_extractor.")
    if getattr(cfg, "pretrain_iqn_fc_freeze", False):
        prefixes.append("iqn_fc.")
    if getattr(cfg, "pretrain_actions_head_freeze", False):
        prefixes.append("A_head.")
    if getattr(cfg, "pretrain_V_head_freeze", False):
        prefixes.append("V_head.")
    return prefixes


def _apply_pretrain_freeze(network):
    """Set requires_grad=False for parameters whose name starts with a frozen prefix."""
    prefixes = _get_frozen_param_prefixes()
    if not prefixes:
        return
    for name, param in network.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = False
    frozen_count = sum(1 for n, p in network.named_parameters() if not p.requires_grad)
    if frozen_count:
        print(f"[OK] Pretrain freeze: {frozen_count} parameter tensors frozen (prefixes: {prefixes})")


def learner_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
):
    layout_version = "lay_mono"
    SummaryWriter(log_dir=str(tensorboard_base_dir / layout_version)).add_custom_scalars(
        {
            layout_version: {
                # Race metrics
                "Race/eval_race_time_robust": [
                    "Multiline",
                    [
                        "Race/eval_race_time_robust.*",
                    ],
                ],
                "Race/explo_race_time_finished": [
                    "Multiline",
                    [
                        "Race/explo_race_time_finished.*",
                    ],
                ],
                # Training metrics
                "Training/loss": [
                    "Multiline",
                    [
                        "Training/loss$",
                        "Training/loss_test$",
                    ],
                ],
                # RL metrics
                "RL/avg_Q": [
                    "Multiline",
                    [
                        "RL/avg_Q.*",
                    ],
                ],
                "RL/single_zone_reached": [
                    "Multiline",
                    [
                        "RL/single_zone_reached.*",
                    ],
                ],
                # Gradient metrics
                "Gradients/norm": [
                    "Multiline",
                    [
                        "Gradients/norm_median",
                        "Gradients/norm_d9",
                        "Gradients/norm_d98",
                        "Gradients/norm_max",
                    ],
                ],
                "Gradients/norm_before_clip": [
                    "Multiline",
                    [
                        "Gradients/norm_before_clip_median",
                        "Gradients/norm_before_clip_d9",
                        "Gradients/norm_before_clip_d98",
                        "Gradients/norm_before_clip_max",
                    ],
                ],
                # Buffer metrics
                "Buffer/priorities": [
                    "Multiline",
                    [
                        "Buffer/priorities_median",
                        "Buffer/priorities_d9",
                        "Buffer/priorities_max",
                    ],
                ],
            },
        }
    )

    # ========================================================
    # Create new stuff
    # ========================================================

    online_network, uncompiled_online_network = make_untrained_iqn_network(get_config().use_jit, is_inference=False)
    target_network, _ = make_untrained_iqn_network(get_config().use_jit, is_inference=False)

    print("\n" + "="*80)
    print("  NETWORK ARCHITECTURE")
    print("="*80)
    utilities.count_parameters(online_network)
    print("="*80 + "\n")

    accumulated_stats: defaultdict[str, typing.Any] = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    previous_alltime_min = None
    time_last_save = time.perf_counter()
    queue_check_order = list(range(len(rollout_queues)))
    rollout_queue_readers = [q._reader for q in rollout_queues]
    time_waited_for_workers_since_last_tensorboard_write = 0
    time_training_since_last_tensorboard_write = 0
    time_testing_since_last_tensorboard_write = 0

    # ========================================================
    # Load existing stuff
    # ========================================================
    def _state_dict_for_model(loaded_sd: dict, model: torch.nn.Module) -> dict:
        """Remap checkpoint keys if it was saved without torch.compile (no _orig_mod. prefix)."""
        model_keys = list(model.state_dict().keys())
        loaded_keys = list(loaded_sd.keys())
        if not loaded_keys or not model_keys:
            return loaded_sd
        model_has_prefix = model_keys[0].startswith("_orig_mod.")
        loaded_has_prefix = loaded_keys[0].startswith("_orig_mod.")
        if model_has_prefix and not loaded_has_prefix:
            return {"_orig_mod." + k: v for k, v in loaded_sd.items()}
        return loaded_sd

    w1_path = save_dir / "weights1.torch"
    w2_path = save_dir / "weights2.torch"
    try:
        if not w1_path.exists() or not w2_path.exists():
            raise FileNotFoundError(
                f"Checkpoint files missing in {save_dir}: "
                f"weights1.torch exists={w1_path.exists()}, weights2.torch exists={w2_path.exists()}"
            )
        sd1 = torch.load(f=w1_path, weights_only=False)
        sd2 = torch.load(f=w2_path, weights_only=False)
        sd1 = _state_dict_for_model(sd1, online_network)
        sd2 = _state_dict_for_model(sd2, target_network)
        online_network.load_state_dict(sd1, strict=True)
        target_network.load_state_dict(sd2, strict=True)
        print("[OK] Weights loaded successfully")
        if get_config().pretrain_encoder_path:
            print("[OK] Pretrain: img_head loaded from checkpoint (initialized via pretrain_encoder_path).")
    except FileNotFoundError as e:
        print(f"[INFO] Starting with fresh weights: {e}")
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint from {save_dir}: {e}")
        print("[INFO] Starting with fresh weights (fix config/architecture or remove old .torch to avoid this message)")

    _apply_pretrain_freeze(online_network)
    _apply_pretrain_freeze(target_network)

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    # noinspection PyBroadException
    try:
        accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
        shared_steps.value = accumulated_stats["cumul_number_frames_played"]
        print(f"[OK] Stats loaded - resuming from {accumulated_stats['cumul_number_frames_played']:,} frames")
    except:
        print("[INFO] Starting fresh training session")

    if "rolling_mean_ms" not in accumulated_stats.keys():
        # Temporary to preserve compatibility with old runs that doesn't have this feature. To be removed later.
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats["cumul_number_single_memories_used"]
    transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]
    neural_net_reset_counter = 0
    single_reset_flag = get_config().single_reset_flag

    optimizer1 = torch.optim.RAdam(
        [p for p in online_network.parameters() if p.requires_grad],
        lr=utilities.from_exponential_schedule(get_config().lr_schedule, accumulated_stats["cumul_number_frames_played"]),
        eps=get_config().adam_epsilon,
        betas=(get_config().adam_beta1, get_config().adam_beta2),
    )
    # optimizer1 = torch_optimizer.Lookahead(optimizer1, k=5, alpha=0.5)

    scaler = torch.amp.GradScaler("cuda")
    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        get_config().memory_size_schedule, accumulated_stats["cumul_number_frames_played"]
    )
    buffer, buffer_test = make_buffers(memory_size)
    offset_cumul_number_single_memories_used = memory_size_start_learn * get_config().number_times_single_memory_is_used_before_discard

    # noinspection PyBroadException
    try:
        optimizer1.load_state_dict(torch.load(f=save_dir / "optimizer1.torch", weights_only=False))
        scaler.load_state_dict(torch.load(f=save_dir / "scaler.torch", weights_only=False))
        print("[OK] Optimizer loaded")
    except Exception:
        print("[INFO] Starting with fresh optimizer (or checkpoint was saved with different trainable params, e.g. freeze config)")

    tensorboard_suffix = utilities.from_staircase_schedule(
        get_config().tensorboard_suffix_schedule,
        accumulated_stats["cumul_number_frames_played"],
    )
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (get_config().run_name + tensorboard_suffix)))

    loss_history = []
    loss_test_history = []
    train_on_batch_duration_history = []
    grad_norm_history = []
    grad_norm_before_clip_history = []
    layer_grad_norm_history = defaultdict(list)

    # ========================================================
    # Make the trainer
    # ========================================================
    trainer = iqn.Trainer(
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer1,
        scaler=scaler,
        batch_size=get_config().batch_size,
        iqn_n=get_config().iqn_n,
    )

    inferer = iqn.Inferer(
        inference_network=online_network,
        iqn_k=get_config().iqn_k,
        tau_epsilon_boltzmann=get_config().tau_epsilon_boltzmann,
    )

    # ================================================================
    #   PHASE 3: IngestThread — offload rollout ingestion to background
    # ================================================================
    # The buffer stays in-process so train_on_batch(buffer) and
    # buffer.update_priority() work unchanged.
    # The IngestThread reads rollouts, fills the buffer, and pushes
    # race metrics into a deque for the main thread to log to TB.
    # GIL is released during PyTorch tensor ops (vectorized buffer fill),
    # so the thread truly runs in parallel with GPU training.
    # ================================================================

    buffer_lock = threading.Lock()  # protects buffer/buffer_test during concurrent access
    metrics_deque = deque(maxlen=1024)  # thread-safe; main thread pops for TB logging
    ingest_stats_lock = threading.Lock()  # protects accumulated_stats fields written by ingest thread

    # Mutable containers shared between threads
    ingest_state = {
        "buffer": buffer,
        "buffer_test": buffer_test,
        "memory_size": memory_size,
        "memory_size_start_learn": memory_size_start_learn,
        "neural_net_reset_counter": 0,
    }

    def _ingest_thread_fn():
        """Background thread: reads rollouts from collectors, fills the replay buffer."""
        nonlocal offset_cumul_number_single_memories_used
        queue_check_order_t = list(range(len(rollout_queues)))
        rollout_queue_readers_t = [q._reader for q in rollout_queues]

        while True:
            try:
                wait(rollout_queue_readers_t)
                for idx in queue_check_order_t:
                    if not rollout_queues[idx].empty():
                        (
                            rollout_results_t,
                            end_race_stats_t,
                            fill_buffer_t,
                            is_explo_t,
                            map_name_t,
                            map_status_t,
                            rollout_duration_t,
                            loop_number_t,
                        ) = rollout_queues[idx].get()
                        queue_check_order_t.append(queue_check_order_t.pop(queue_check_order_t.index(idx)))
                        break
                else:
                    continue

                # Push race metrics for TB logging by main thread
                metrics_deque.append((
                    rollout_results_t, end_race_stats_t, fill_buffer_t,
                    is_explo_t, map_name_t, map_status_t, rollout_duration_t, loop_number_t,
                ))

                if fill_buffer_t:
                    # Dynamic reward annealing (read shared counter)
                    frames_played = accumulated_stats["cumul_number_frames_played"]
                    gamma_t = utilities.from_linear_schedule(get_config().gamma_schedule, frames_played)
                    eng_ss = utilities.from_linear_schedule(get_config().engineered_speedslide_reward_schedule, frames_played)
                    eng_ns = utilities.from_linear_schedule(get_config().engineered_neoslide_reward_schedule, frames_played)
                    eng_kk = utilities.from_linear_schedule(get_config().engineered_kamikaze_reward_schedule, frames_played)
                    eng_vcp = utilities.from_linear_schedule(get_config().engineered_close_to_vcp_reward_schedule, frames_played)

                    # Check memory resize
                    new_msz, new_msl = utilities.from_staircase_schedule(get_config().memory_size_schedule, frames_played)
                    if new_msz != ingest_state["memory_size"]:
                        with buffer_lock:
                            ingest_state["buffer"], ingest_state["buffer_test"] = resize_buffers(
                                ingest_state["buffer"], ingest_state["buffer_test"], new_msz
                            )
                        with ingest_stats_lock:
                            offset_cumul_number_single_memories_used += (
                                new_msl - ingest_state["memory_size_start_learn"]
                            ) * get_config().number_times_single_memory_is_used_before_discard
                        ingest_state["memory_size_start_learn"] = new_msl
                        ingest_state["memory_size"] = new_msz

                    # Fill buffer (CPU-heavy, vectorized PyTorch — GIL released during tensor ops)
                    with buffer_lock:
                        (
                            ingest_state["buffer"],
                            ingest_state["buffer_test"],
                            n_added_train,
                            n_added_test,
                        ) = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
                            ingest_state["buffer"],
                            ingest_state["buffer_test"],
                            rollout_results_t,
                            get_config().n_steps,
                            gamma_t,
                            get_config().discard_non_greedy_actions_in_nsteps,
                            eng_ss, eng_ns, eng_kk, eng_vcp,
                        )

                    # Update shared counters (thread-safe)
                    with ingest_stats_lock:
                        accumulated_stats["cumul_number_memories_generated"] += n_added_train + n_added_test
                        accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
                            get_config().number_times_single_memory_is_used_before_discard * n_added_train
                        )
                        ingest_state["neural_net_reset_counter"] += n_added_train
                    # shared_steps is updated by the main thread after it adds cumul_number_frames_played

            except Exception as e:
                print(f"[IngestThread] Error: {e}")
                import traceback
                traceback.print_exc()

    ingest_thread = threading.Thread(target=_ingest_thread_fn, daemon=True, name="IngestThread")
    ingest_thread.start()
    print("[OK] IngestThread started (Phase 3: async buffer fill)")

    last_rollout_results = None  # Track last rollout for IQN spread calculation

    # ================================================================
    #   MAIN TRAINING LOOP
    # ================================================================
    while True:
        # -------------------------------------------------------------
        # 1. PROCESS RACE METRICS from IngestThread (non-blocking)
        # -------------------------------------------------------------
        processed_any_metrics = False
        while metrics_deque:
            try:
                (
                    rollout_results, end_race_stats, fill_buffer,
                    is_explo, map_name, map_status, rollout_duration, loop_number,
                ) = metrics_deque.popleft()
                processed_any_metrics = True
                last_rollout_results = rollout_results  # Save for IQN spread later
            except IndexError:
                break

            accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])
            shared_steps.value = accumulated_stats["cumul_number_frames_played"]

            new_tensorboard_suffix = utilities.from_staircase_schedule(
                get_config().tensorboard_suffix_schedule,
                accumulated_stats["cumul_number_frames_played"],
            )
            if new_tensorboard_suffix != tensorboard_suffix:
                tensorboard_suffix = new_tensorboard_suffix
                tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (get_config().run_name + tensorboard_suffix)))

            if get_config().plot_race_time_left_curves and not is_explo and (loop_number // 5) % 17 == 0:
                race_time_left_curves(rollout_results, inferer, save_dir, map_name)
                tau_curves(rollout_results, inferer, save_dir, map_name)
                with buffer_lock:
                    distribution_curves(ingest_state["buffer"], save_dir, online_network, target_network)
                    loss_distribution(ingest_state["buffer"], save_dir, online_network, target_network)

            # ===============================================
            #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
            # ===============================================
            race_stats_to_write = {
                f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / (rollout_duration * 1000),
                f"explo_race_time_{map_status}_{map_name}" if is_explo else f"eval_race_time_{map_status}_{map_name}": end_race_stats["race_time"] / 1000,
                f"explo_race_finished_{map_status}_{map_name}" if is_explo else f"eval_race_finished_{map_status}_{map_name}": end_race_stats["race_finished"],
                f"mean_action_gap_{map_name}": -(np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1, initial=None).reshape(-1, 1)).mean(),
                f"single_zone_reached_{map_status}_{map_name}": rollout_results["furthest_zone_idx"],
                "instrumentation__answer_normal_step": end_race_stats["instrumentation__answer_normal_step"],
                "instrumentation__answer_action_step": end_race_stats["instrumentation__answer_action_step"],
                "instrumentation__between_run_steps": end_race_stats["instrumentation__between_run_steps"],
                "instrumentation__grab_frame": end_race_stats["instrumentation__grab_frame"],
                "instrumentation__convert_frame": end_race_stats["instrumentation__convert_frame"],
                "instrumentation__grab_floats": end_race_stats["instrumentation__grab_floats"],
                "instrumentation__exploration_policy": end_race_stats["instrumentation__exploration_policy"],
                "instrumentation__request_inputs_and_speed": end_race_stats["instrumentation__request_inputs_and_speed"],
                "tmi_protection_cutoff": end_race_stats["tmi_protection_cutoff"],
                "worker_time_in_rollout_percentage": rollout_results["worker_time_in_rollout_percentage"],
            }

            if not is_explo:
                race_stats_to_write[f"avg_Q_{map_status}_{map_name}"] = np.mean(rollout_results["q_values"])

            if end_race_stats["race_finished"]:
                race_stats_to_write[f"{'explo' if is_explo else 'eval'}_race_time_finished_{map_status}_{map_name}"] = (
                    end_race_stats["race_time"] / 1000
                )
                if not is_explo:
                    accumulated_stats["rolling_mean_ms"][map_name] = (
                        accumulated_stats["rolling_mean_ms"].get(map_name, get_config().cutoff_rollout_if_race_not_finished_within_duration_ms)
                        * 0.9
                        + end_race_stats["race_time"] * 0.1
                    )
            if (
                (not is_explo)
                and end_race_stats["race_finished"]
                and end_race_stats["race_time"] < 1.02 * accumulated_stats["rolling_mean_ms"][map_name]
            ):
                race_stats_to_write[f"eval_race_time_robust_{map_status}_{map_name}"] = end_race_stats["race_time"] / 1000
                if map_name in reference_times:
                    for reference_time_name in ["author", "gold"]:
                        if reference_time_name in reference_times[map_name]:
                            reference_time = reference_times[map_name][reference_time_name]
                            race_stats_to_write[f"eval_ratio_{map_status}_{reference_time_name}_{map_name}"] = (
                                100 * (end_race_stats["race_time"] / 1000) / reference_time
                            )
                            race_stats_to_write[f"eval_agg_ratio_{map_status}_{reference_time_name}"] = (
                                100 * (end_race_stats["race_time"] / 1000) / reference_time
                            )

            for i in [0]:
                race_stats_to_write[f"q_value_{i}_starting_frame_{map_name}"] = end_race_stats[f"q_value_{i}_starting_frame"]
            if not is_explo:
                for i, split_time in enumerate(
                    [(e - s) / 1000 for s, e in zip(end_race_stats["cp_time_ms"][:-1], end_race_stats["cp_time_ms"][1:])]
                ):
                    race_stats_to_write[f"split_{map_name}_{i}"] = split_time

            walltime_tb = time.time()
            for tag, value in race_stats_to_write.items():
                if tag.startswith("eval_race_time") or tag.startswith("explo_race_time"):
                    grouped_tag = f"Race/{tag}"
                elif tag.startswith("eval_race_finished") or tag.startswith("explo_race_finished"):
                    grouped_tag = f"Race/{tag}"
                elif tag.startswith("race_time_ratio"):
                    grouped_tag = f"Race/{tag}"
                elif tag.startswith("avg_Q") or tag.startswith("single_zone_reached") or tag.startswith("mean_action_gap") or tag.startswith("q_value_"):
                    grouped_tag = f"RL/{tag}"
                elif tag.startswith("split_"):
                    grouped_tag = f"Race/{tag}"
                elif tag.startswith("instrumentation__") or tag.startswith("worker_time_") or tag.startswith("tmi_protection_"):
                    grouped_tag = f"Performance/{tag}"
                else:
                    grouped_tag = tag
                tensorboard_writer.add_scalar(
                    tag=grouped_tag, scalar_value=value,
                    global_step=accumulated_stats["cumul_number_frames_played"], walltime=walltime_tb,
                )

            # ===============================================
            #   SAVE STUFF IF THIS WAS A GOOD RACE
            # ===============================================
            is_new_record = end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"].get(map_name, 99999999999)
            if is_new_record:
                old_best = accumulated_stats["alltime_min_ms"].get(map_name, 99999999999)
                accumulated_stats["alltime_min_ms"][map_name] = end_race_stats["race_time"]
                improvement = (old_best - end_race_stats["race_time"]) / 1000
                race_time_s = end_race_stats["race_time"] / 1000
                race_finished_str = "FINISH" if end_race_stats["race_finished"] else "DNF"
                explo_str = "EXPLO" if is_explo else "EVAL"
                if old_best < 99999999:
                    print(f"\n>>> NEW RECORD! [{explo_str}] [{race_finished_str}] {map_name:15} {race_time_s:6.2f}s (improved by {improvement:.3f}s) <<<\n")
                else:
                    print(f"\n>>> FIRST FINISH! [{explo_str}] {map_name:15} {race_time_s:6.2f}s <<<\n")
                if accumulated_stats["cumul_number_frames_played"] > get_config().frames_before_save_best_runs:
                    name = f"{map_name}_{end_race_stats['race_time']}"
                    utilities.save_run(base_dir, save_dir / "best_runs" / name, rollout_results, f"{name}.inputs", inputs_only=False)
                    utilities.save_checkpoint(save_dir / "best_runs", online_network, target_network, optimizer1, scaler)

            if end_race_stats["race_time"] < get_config().threshold_to_save_all_runs_ms:
                name = f"{map_name}_{end_race_stats['race_time']}_{datetime.now().strftime('%m%d_%H%M%S')}_{accumulated_stats['cumul_number_frames_played']}_{'explo' if is_explo else 'eval'}"
                utilities.save_run(base_dir, save_dir / "good_runs", rollout_results, f"{name}.inputs", inputs_only=True)

        # -------------------------------------------------------------
        # 2. TRAINING ANNEALING (every iteration of main loop)
        # -------------------------------------------------------------
        learning_rate = utilities.from_exponential_schedule(get_config().lr_schedule, accumulated_stats["cumul_number_frames_played"])
        weight_decay = get_config().weight_decay_lr_ratio * learning_rate
        gamma = utilities.from_linear_schedule(get_config().gamma_schedule, accumulated_stats["cumul_number_frames_played"])

        for param_group in optimizer1.param_groups:
            param_group["lr"] = learning_rate
            param_group["epsilon"] = get_config().adam_epsilon
            param_group["betas"] = (get_config().adam_beta1, get_config().adam_beta2)

        with buffer_lock:
            if isinstance(ingest_state["buffer"]._sampler, PrioritizedSampler):
                ingest_state["buffer"]._sampler._alpha = get_config().prio_alpha
                ingest_state["buffer"]._sampler._beta = get_config().prio_beta
                ingest_state["buffer"]._sampler._eps = get_config().prio_epsilon

        # Memory size tracking for main loop
        memory_size = ingest_state["memory_size"]
        memory_size_start_learn = ingest_state["memory_size_start_learn"]

        # -------------------------------------------------------------
        # 3. PERIODIC NETWORK RESET (check counter from ingest thread)
        # -------------------------------------------------------------
        with ingest_stats_lock:
            neural_net_reset_counter = ingest_state["neural_net_reset_counter"]

        if neural_net_reset_counter >= get_config().reset_every_n_frames_generated or single_reset_flag != get_config().single_reset_flag:
            with ingest_stats_lock:
                ingest_state["neural_net_reset_counter"] = 0
            single_reset_flag = get_config().single_reset_flag
            accumulated_stats["cumul_number_single_memories_should_have_been_used"] += get_config().additional_transition_after_reset

            _, untrained_iqn_network = make_untrained_iqn_network(get_config().use_jit, False)
            frozen_prefixes = _get_frozen_param_prefixes()
            utilities.soft_copy_param(
                online_network, untrained_iqn_network, get_config().overall_reset_mul_factor,
                skip_key_prefixes=frozen_prefixes,
            )

            with torch.no_grad():
                if not get_config().pretrain_actions_head_freeze:
                    online_network.A_head[2].weight = utilities.linear_combination(
                        online_network.A_head[2].weight, untrained_iqn_network.A_head[2].weight, get_config().last_layer_reset_factor,
                    )
                    online_network.A_head[2].bias = utilities.linear_combination(
                        online_network.A_head[2].bias, untrained_iqn_network.A_head[2].bias, get_config().last_layer_reset_factor,
                    )
                if not get_config().pretrain_V_head_freeze:
                    online_network.V_head[2].weight = utilities.linear_combination(
                        online_network.V_head[2].weight, untrained_iqn_network.V_head[2].weight, get_config().last_layer_reset_factor,
                    )
                    online_network.V_head[2].bias = utilities.linear_combination(
                        online_network.V_head[2].bias, untrained_iqn_network.V_head[2].bias, get_config().last_layer_reset_factor,
                    )

        # -------------------------------------------------------------
        # 4. LEARN ON BATCH (fine-grained locking: lock only for sample + priority update)
        # -------------------------------------------------------------
        if not online_network.training:
            online_network.train()

        with buffer_lock:
            buf = ingest_state["buffer"]
            buf_test = ingest_state["buffer_test"]
            can_learn = len(buf) >= memory_size_start_learn

        if can_learn:
            while (
                accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used
                <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
            ):
                with buffer_lock:
                    buf = ingest_state["buffer"]
                    buf_test = ingest_state["buffer_test"]
                    buf_len = len(buf)
                    if buf_len < memory_size_start_learn:
                        break

                if (random.random() < get_config().buffer_test_ratio and len(buf_test) > 0) or buf_len == 0:
                    test_start_time = time.perf_counter()
                    # Phase A: sample under lock (~1ms)
                    with buffer_lock:
                        batch, batch_info = trainer.sample_batch(ingest_state["buffer_test"])
                    # Phase B: GPU compute — NO lock (~5-15ms)
                    loss, _, _, _ = trainer.train_on_data(batch, batch_info, do_learn=False)
                    # Phase C: no priority update for test batches
                    time_testing_since_last_tensorboard_write += time.perf_counter() - test_start_time
                    loss_test_history.append(loss)
                else:
                    train_start_time = time.perf_counter()
                    # Phase A: sample under lock (~1ms)
                    with buffer_lock:
                        batch, batch_info = trainer.sample_batch(ingest_state["buffer"])
                    # Phase B: GPU compute — NO lock (~5-15ms)
                    loss, grad_norm, grad_norm_before_clip, priority_update = trainer.train_on_data(batch, batch_info, do_learn=True)
                    # Phase C: priority update under lock (~0.5ms)
                    with buffer_lock:
                        trainer.apply_priority_update(ingest_state["buffer"], priority_update)
                    train_on_batch_duration_history.append(time.perf_counter() - train_start_time)
                    time_training_since_last_tensorboard_write += train_on_batch_duration_history[-1]
                    with buffer_lock:
                        accumulated_stats["cumul_number_single_memories_used"] += (
                            4 * get_config().batch_size
                            if (len(ingest_state["buffer"]) < ingest_state["buffer"]._storage.max_size and ingest_state["buffer"]._storage.max_size > 200_000)
                            else get_config().batch_size
                        )  # do fewer batches while memory is not full
                    loss_history.append(loss)
                    if not math.isinf(grad_norm):
                        grad_norm_history.append(grad_norm)
                    if not math.isinf(grad_norm_before_clip):
                        grad_norm_before_clip_history.append(grad_norm_before_clip)
                    utilities.log_gradient_norms(online_network, layer_grad_norm_history)

                    accumulated_stats["cumul_number_batches_done"] += 1

                    utilities.custom_weight_decay(online_network, 1 - weight_decay, only_trainable=True)
                    if accumulated_stats["cumul_number_batches_done"] % get_config().send_shared_network_every_n_batches == 0:
                        with shared_network_lock:
                            uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

                    # UPDATE TARGET NETWORK
                    if (
                        accumulated_stats["cumul_number_single_memories_used"]
                        >= accumulated_stats["cumul_number_single_memories_used_next_target_network_update"]
                    ):
                        accumulated_stats["cumul_number_target_network_updates"] += 1
                        accumulated_stats["cumul_number_single_memories_used_next_target_network_update"] += (
                            get_config().number_memories_trained_on_between_target_network_updates
                        )
                        utilities.soft_copy_param(target_network, online_network, get_config().soft_update_tau)
        else:
            # No data yet — wait a bit before spinning
            time.sleep(0.1)

        sys.stdout.flush()

        # ===============================================
        #   WRITE AGGREGATED STATISTICS TO TENSORBOARD EVERY 5 MINUTES
        # ===============================================
        save_frequency_s = 5 * 60
        if time.perf_counter() - time_last_save >= save_frequency_s:
            accumulated_stats["cumul_training_hours"] += (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save
            waited_percentage = time_waited_for_workers_since_last_tensorboard_write / time_since_last_save
            trained_percentage = time_training_since_last_tensorboard_write / time_since_last_save
            tested_percentage = time_testing_since_last_tensorboard_write / time_since_last_save
            time_waited_for_workers_since_last_tensorboard_write = 0
            time_training_since_last_tensorboard_write = 0
            time_testing_since_last_tensorboard_write = 0
            transitions_learned_per_second = (
                accumulated_stats["cumul_number_single_memories_used"] - transitions_learned_last_save
            ) / time_since_last_save
            time_last_save = time.perf_counter()
            transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]

            # ===============================================
            #   COLLECT VARIOUS STATISTICS
            # ===============================================
            step_stats = {
                # Training metrics
                "Training/learning_rate": learning_rate,
                "Training/weight_decay": weight_decay,
                "Training/batch_size": get_config().batch_size,
                "Training/n_steps": get_config().n_steps,
                "Training/discard_non_greedy_actions_in_nsteps": get_config().discard_non_greedy_actions_in_nsteps,
                
                # RL hyperparameters
                "RL/gamma": gamma,
                "RL/epsilon": utilities.from_exponential_schedule(get_config().epsilon_schedule, shared_steps.value),
                "RL/epsilon_boltzmann": utilities.from_exponential_schedule(get_config().epsilon_boltzmann_schedule, shared_steps.value),
                "RL/tau_epsilon_boltzmann": get_config().tau_epsilon_boltzmann,
                
                # Buffer metrics (read under lock)
                "Buffer/size": len(ingest_state["buffer"]),
                "Buffer/max_size": ingest_state["buffer"]._storage.max_size,
                "Buffer/number_times_single_memory_is_used_before_discard": get_config().number_times_single_memory_is_used_before_discard,
                
                # Performance metrics
                "Performance/learner_percentage_waiting_for_workers": waited_percentage,
                "Performance/learner_percentage_training": trained_percentage,
                "Performance/learner_percentage_testing": tested_percentage,
                "Performance/transitions_learned_per_second": transitions_learned_per_second,
            }
            if len(loss_history) > 0 and len(loss_test_history) > 0:
                step_stats.update(
                    {
                        # Training loss
                        "Training/loss": np.mean(loss_history),
                        "Training/loss_test": np.mean(loss_test_history),
                        "Training/train_on_batch_duration": np.median(train_on_batch_duration_history),
                        
                        # Gradient norms (after clipping)
                        "Gradients/norm_q1": np.quantile(grad_norm_history, 0.25),
                        "Gradients/norm_median": np.quantile(grad_norm_history, 0.5),
                        "Gradients/norm_q3": np.quantile(grad_norm_history, 0.75),
                        "Gradients/norm_d9": np.quantile(grad_norm_history, 0.9),
                        "Gradients/norm_d98": np.quantile(grad_norm_history, 0.98),
                        "Gradients/norm_max": np.max(grad_norm_history),
                        
                        # Gradient norms BEFORE clipping (for explosion detection)
                        "Gradients/norm_before_clip_q1": np.quantile(grad_norm_before_clip_history, 0.25) if len(grad_norm_before_clip_history) > 0 else 0.0,
                        "Gradients/norm_before_clip_median": np.quantile(grad_norm_before_clip_history, 0.5) if len(grad_norm_before_clip_history) > 0 else 0.0,
                        "Gradients/norm_before_clip_q3": np.quantile(grad_norm_before_clip_history, 0.75) if len(grad_norm_before_clip_history) > 0 else 0.0,
                        "Gradients/norm_before_clip_d9": np.quantile(grad_norm_before_clip_history, 0.9) if len(grad_norm_before_clip_history) > 0 else 0.0,
                        "Gradients/norm_before_clip_d98": np.quantile(grad_norm_before_clip_history, 0.98) if len(grad_norm_before_clip_history) > 0 else 0.0,
                        "Gradients/norm_before_clip_max": np.max(grad_norm_before_clip_history) if len(grad_norm_before_clip_history) > 0 else 0.0,
                    }
                )
                # Per-layer gradient norms
                for key, val in layer_grad_norm_history.items():
                    # Extract layer name from key (format: "L2_grad_norm_layer_name" or "Linf_grad_norm_layer_name")
                    if key.startswith("L2_grad_norm_"):
                        layer_name = key.replace("L2_grad_norm_", "")
                        step_stats.update(
                            {
                                f"Gradients/by_layer/{layer_name}/L2_median": np.quantile(val, 0.5),
                                f"Gradients/by_layer/{layer_name}/L2_q3": np.quantile(val, 0.75),
                                f"Gradients/by_layer/{layer_name}/L2_d9": np.quantile(val, 0.9),
                                f"Gradients/by_layer/{layer_name}/L2_max": np.max(val),
                            }
                        )
                    elif key.startswith("Linf_grad_norm_"):
                        layer_name = key.replace("Linf_grad_norm_", "")
                        step_stats.update(
                            {
                                f"Gradients/by_layer/{layer_name}/Linf_median": np.quantile(val, 0.5),
                                f"Gradients/by_layer/{layer_name}/Linf_q3": np.quantile(val, 0.75),
                                f"Gradients/by_layer/{layer_name}/Linf_d9": np.quantile(val, 0.9),
                                f"Gradients/by_layer/{layer_name}/Linf_max": np.max(val),
                            }
                        )
            with buffer_lock:
                if isinstance(ingest_state["buffer"]._sampler, PrioritizedSampler):
                    all_priorities = np.array([ingest_state["buffer"]._sampler._sum_tree.at(i) for i in range(len(ingest_state["buffer"]))])
                    step_stats.update(
                        {
                            "Buffer/priorities_min": np.min(all_priorities),
                            "Buffer/priorities_q1": np.quantile(all_priorities, 0.1),
                            "Buffer/priorities_mean": np.mean(all_priorities),
                            "Buffer/priorities_median": np.quantile(all_priorities, 0.5),
                            "Buffer/priorities_q3": np.quantile(all_priorities, 0.75),
                            "Buffer/priorities_d9": np.quantile(all_priorities, 0.9),
                            "Buffer/priorities_c98": np.quantile(all_priorities, 0.98),
                            "Buffer/priorities_max": np.max(all_priorities),
                        }
                    )
            for key, value in accumulated_stats.items():
                if key not in ["alltime_min_ms", "rolling_mean_ms"]:
                    step_stats[key] = value
            for key, value in accumulated_stats["alltime_min_ms"].items():
                step_stats[f"alltime_min_ms_{key}"] = value

            loss_history = []
            loss_test_history = []
            train_on_batch_duration_history = []
            grad_norm_history = []
            grad_norm_before_clip_history = []
            layer_grad_norm_history = defaultdict(list)

            # ===============================================
            #   COLLECT IQN SPREAD
            # ===============================================

            if online_network.training:
                online_network.eval()
            tau = torch.linspace(0.05, 0.95, get_config().iqn_k)[:, None].to("cuda")
            if last_rollout_results is not None:
                per_quantile_output = inferer.infer_network(last_rollout_results["frames"][0], last_rollout_results["state_float"][0], tau)
            # IQN-specific metrics: quantile spread per action
                for i, std in enumerate(list(per_quantile_output.std(axis=0))):
                    step_stats[f"IQN/quantile_std_action_{i}"] = std

            # ===============================================
            #   WRITE TO TENSORBOARD
            # ===============================================

            walltime_tb = time.time()
            # Log network weights
            for name, param in online_network.named_parameters():
                tensorboard_writer.add_scalar(
                    tag=f"Network/weights/{name}/L2",
                    scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
            assert len(optimizer1.param_groups) == 1
            try:
                # Log optimizer state (Adam/RAdam specific); only params that are in the optimizer (trainable)
                for name, param in online_network.named_parameters():
                    if not param.requires_grad or param not in optimizer1.state:
                        continue
                    state = optimizer1.state[param]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    mod_lr = 1 / (exp_avg_sq.sqrt() + 1e-4)
                    tensorboard_writer.add_scalar(
                        tag=f"Network/optimizer/{name}/adaptive_lr_L2",
                        scalar_value=np.sqrt((mod_lr**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"Network/optimizer/{name}/exp_avg_L2",
                        scalar_value=np.sqrt((exp_avg**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"Network/optimizer/{name}/exp_avg_sq_L2",
                        scalar_value=np.sqrt((exp_avg_sq**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
            except:
                pass

            # Log all aggregated statistics (already grouped with prefixes)
            for k, v in step_stats.items():
                tensorboard_writer.add_scalar(
                    tag=k,
                    scalar_value=v,
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )

            previous_alltime_min = previous_alltime_min or copy.deepcopy(accumulated_stats["alltime_min_ms"])

            tensorboard_writer.add_text(
                "times_summary",
                f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} "
                + " ".join(
                    [
                        f"{'**' if v < previous_alltime_min.get(k, 99999999) else ''}{k}: {v / 1000:.2f}{'**' if v < previous_alltime_min.get(k, 99999999) else ''}"
                        for k, v in accumulated_stats["alltime_min_ms"].items()
                    ]
                ),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

            previous_alltime_min = copy.deepcopy(accumulated_stats["alltime_min_ms"])

            # ===============================================
            #   BUFFER STATS
            # ===============================================

            with buffer_lock:
                state_floats = np.array([experience.state_float for experience in ingest_state["buffer"]._storage])
                mean_in_buffer = state_floats.mean(axis=0)
                std_in_buffer = state_floats.std(axis=0)

            # ===============================================
            #   BEAUTIFUL SUMMARY EVERY 5 MINUTES
            # ===============================================
            print("\n" + "="*80)
            print(f"  TRAINING SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            print(f"  Frames played: {accumulated_stats['cumul_number_frames_played']:,}")
            print(f"  Training hours: {accumulated_stats['cumul_training_hours']:.1f}h")
            print(f"  Buffer size: {len(ingest_state['buffer']):,} / {ingest_state['buffer']._storage.max_size:,}")
            if len(loss_history) > 0:
                print(f"  Avg loss: {np.mean(loss_history):.3e}  |  Grad norm: {np.median(grad_norm_history):.3e}")
            print(f"  Learning rate: {learning_rate:.2e}")
            print(f"  Transitions/sec: {transitions_learned_per_second:.1f}")
            print("-"*80)
            print("  BEST TIMES:")
            for map_name_iter, best_time_ms in sorted(accumulated_stats["alltime_min_ms"].items()):
                best_time_s = best_time_ms / 1000
                rolling_mean_s = accumulated_stats["rolling_mean_ms"].get(map_name_iter, best_time_ms) / 1000
                print(f"    {map_name_iter:15} {best_time_s:6.2f}s  (rolling avg: {rolling_mean_s:6.2f}s)")
            print("="*80 + "\n")

            # ===============================================
            #   HIGH PRIORITY TRANSITIONS
            # ===============================================
            with buffer_lock:
                if get_config().make_highest_prio_figures and isinstance(ingest_state["buffer"]._sampler, PrioritizedSampler):
                    highest_prio_transitions(ingest_state["buffer"], save_dir)

            # ===============================================
            #   SAVE
            # ===============================================
            utilities.save_checkpoint(save_dir, online_network, target_network, optimizer1, scaler)
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
