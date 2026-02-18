"""
This file implements the main training loop, tensorboard statistics tracking, etc...
"""

import copy
import math
import random
import sys
import time
import typing
from collections import defaultdict
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
    # noinspection PyBroadException
    try:
        online_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
        target_network.load_state_dict(torch.load(f=save_dir / "weights2.torch", weights_only=False))
        print("[OK] Weights loaded successfully")
        if get_config().pretrain_encoder_path:
            print("[OK] Pretrain: img_head loaded from checkpoint (initialized via pretrain_encoder_path).")
    except:
        print("[INFO] Starting with fresh weights (no checkpoint found)")

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
        online_network.parameters(),
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
    except:
        print("[INFO] Starting with fresh optimizer")

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

    while True:  # Trainer loop
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time
        if time_waited > 5:  # Only warn if waited more than 5 seconds
            print(f"[WARNING] Learner waited {time_waited:.1f}s for workers (workers might be slow)")
        time_waited_for_workers_since_last_tensorboard_write += time_waited
        for idx in queue_check_order:
            if not rollout_queues[idx].empty():
                (
                    rollout_results,
                    end_race_stats,
                    fill_buffer,
                    is_explo,
                    map_name,
                    map_status,
                    rollout_duration,
                    loop_number,
                ) = rollout_queues[idx].get()
                queue_check_order.append(queue_check_order.pop(queue_check_order.index(idx)))
                break

        new_tensorboard_suffix = utilities.from_staircase_schedule(
            get_config().tensorboard_suffix_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_tensorboard_suffix != tensorboard_suffix:
            tensorboard_suffix = new_tensorboard_suffix
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (get_config().run_name + tensorboard_suffix)))

        (
            new_memory_size,
            new_memory_size_start_learn,
        ) = utilities.from_staircase_schedule(
            get_config().memory_size_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_memory_size != memory_size:
            buffer, buffer_test = resize_buffers(buffer, buffer_test, new_memory_size)
            offset_cumul_number_single_memories_used += (
                new_memory_size_start_learn - memory_size_start_learn
            ) * get_config().number_times_single_memory_is_used_before_discard
            memory_size_start_learn = new_memory_size_start_learn
            memory_size = new_memory_size
        # ===============================================
        #   VERY BASIC TRAINING ANNEALING
        # ===============================================

        # LR and weight_decay calculation
        learning_rate = utilities.from_exponential_schedule(get_config().lr_schedule, accumulated_stats["cumul_number_frames_played"])
        weight_decay = get_config().weight_decay_lr_ratio * learning_rate
        engineered_speedslide_reward = utilities.from_linear_schedule(
            get_config().engineered_speedslide_reward_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        engineered_neoslide_reward = utilities.from_linear_schedule(
            get_config().engineered_neoslide_reward_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        engineered_kamikaze_reward = utilities.from_linear_schedule(
            get_config().engineered_kamikaze_reward_schedule, accumulated_stats["cumul_number_frames_played"]
        )
        engineered_close_to_vcp_reward = utilities.from_linear_schedule(
            get_config().engineered_close_to_vcp_reward_schedule, accumulated_stats["cumul_number_frames_played"]
        )
        gamma = utilities.from_linear_schedule(get_config().gamma_schedule, accumulated_stats["cumul_number_frames_played"])

        # ===============================================
        #   RELOAD
        # ===============================================

        for param_group in optimizer1.param_groups:
            param_group["lr"] = learning_rate
            param_group["epsilon"] = get_config().adam_epsilon
            param_group["betas"] = (get_config().adam_beta1, get_config().adam_beta2)

        if isinstance(buffer._sampler, PrioritizedSampler):
            buffer._sampler._alpha = get_config().prio_alpha
            buffer._sampler._beta = get_config().prio_beta
            buffer._sampler._eps = get_config().prio_epsilon

        if get_config().plot_race_time_left_curves and not is_explo and (loop_number // 5) % 17 == 0:
            race_time_left_curves(rollout_results, inferer, save_dir, map_name)
            tau_curves(rollout_results, inferer, save_dir, map_name)
            distribution_curves(buffer, save_dir, online_network, target_network)
            loss_distribution(buffer, save_dir, online_network, target_network)
            # patrick_curves(rollout_results, trainer, save_dir, map_name)

        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

        # ===============================================
        #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
        # ===============================================
        race_stats_to_write = {
            f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / (rollout_duration * 1000),
            f"explo_race_time_{map_status}_{map_name}" if is_explo else f"eval_race_time_{map_status}_{map_name}": end_race_stats[
                "race_time"
            ]
            / 1000,
            f"explo_race_finished_{map_status}_{map_name}" if is_explo else f"eval_race_finished_{map_status}_{map_name}": end_race_stats[
                "race_finished"
            ],
            f"mean_action_gap_{map_name}": -(
                np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1, initial=None).reshape(-1, 1)
            ).mean(),
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
        
        # Don't print every race - only NEW RECORDS and periodic summaries (every 5 minutes)

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
                [
                    (e - s) / 1000
                    for s, e in zip(
                        end_race_stats["cp_time_ms"][:-1],
                        end_race_stats["cp_time_ms"][1:],
                    )
                ]
            ):
                race_stats_to_write[f"split_{map_name}_{i}"] = split_time

        walltime_tb = time.time()
        for tag, value in race_stats_to_write.items():
            # Group race metrics
            if tag.startswith("eval_race_time") or tag.startswith("explo_race_time"):
                grouped_tag = f"Race/{tag}"
            elif tag.startswith("eval_race_finished") or tag.startswith("explo_race_finished"):
                grouped_tag = f"Race/{tag}"
            elif tag.startswith("race_time_ratio"):
                grouped_tag = f"Race/{tag}"
            elif tag.startswith("avg_Q"):
                grouped_tag = f"RL/{tag}"
            elif tag.startswith("single_zone_reached"):
                grouped_tag = f"RL/{tag}"
            elif tag.startswith("mean_action_gap"):
                grouped_tag = f"RL/{tag}"
            elif tag.startswith("q_value_"):
                grouped_tag = f"RL/{tag}"
            elif tag.startswith("split_"):
                grouped_tag = f"Race/{tag}"
            elif tag.startswith("instrumentation__") or tag.startswith("worker_time_") or tag.startswith("tmi_protection_"):
                grouped_tag = f"Performance/{tag}"
            else:
                grouped_tag = tag  # Keep original tag for unknown metrics
            
            tensorboard_writer.add_scalar(
                tag=grouped_tag,
                scalar_value=value,
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

        # ===============================================
        #   SAVE STUFF IF THIS WAS A GOOD RACE
        # ===============================================

        # Check for NEW RECORD!
        is_new_record = end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"].get(map_name, 99999999999)
        if is_new_record:
            # This is a new alltime_minimum
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
                utilities.save_run(
                    base_dir,
                    save_dir / "best_runs" / name,
                    rollout_results,
                    f"{name}.inputs",
                    inputs_only=False,
                )
                utilities.save_checkpoint(
                    save_dir / "best_runs",
                    online_network,
                    target_network,
                    optimizer1,
                    scaler,
                )

        if end_race_stats["race_time"] < get_config().threshold_to_save_all_runs_ms:
            name = f"{map_name}_{end_race_stats['race_time']}_{datetime.now().strftime('%m%d_%H%M%S')}_{accumulated_stats['cumul_number_frames_played']}_{'explo' if is_explo else 'eval'}"
            utilities.save_run(
                base_dir,
                save_dir / "good_runs",
                rollout_results,
                f"{name}.inputs",
                inputs_only=True,
            )

        # ===============================================
        #   FILL BUFFER WITH (S, A, R, S') transitions
        # ===============================================
        if fill_buffer:
            (
                buffer,
                buffer_test,
                number_memories_added_train,
                number_memories_added_test,
            ) = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
                buffer,
                buffer_test,
                rollout_results,
                get_config().n_steps,
                gamma,
                get_config().discard_non_greedy_actions_in_nsteps,
                engineered_speedslide_reward,
                engineered_neoslide_reward,
                engineered_kamikaze_reward,
                engineered_close_to_vcp_reward,
            )

            accumulated_stats["cumul_number_memories_generated"] += number_memories_added_train + number_memories_added_test
            shared_steps.value = accumulated_stats["cumul_number_frames_played"]
            neural_net_reset_counter += number_memories_added_train
            accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
                get_config().number_times_single_memory_is_used_before_discard * number_memories_added_train
            )
            # Removed noisy memory generation log

            # ===============================================
            #   PERIODIC RESET ?
            # ===============================================

            if neural_net_reset_counter >= get_config().reset_every_n_frames_generated or single_reset_flag != get_config().single_reset_flag:
                neural_net_reset_counter = 0
                single_reset_flag = get_config().single_reset_flag
                accumulated_stats["cumul_number_single_memories_should_have_been_used"] += get_config().additional_transition_after_reset

                _, untrained_iqn_network = make_untrained_iqn_network(get_config().use_jit, False)
                utilities.soft_copy_param(online_network, untrained_iqn_network, get_config().overall_reset_mul_factor)

                with torch.no_grad():
                    online_network.A_head[2].weight = utilities.linear_combination(
                        online_network.A_head[2].weight,
                        untrained_iqn_network.A_head[2].weight,
                        get_config().last_layer_reset_factor,
                    )
                    online_network.A_head[2].bias = utilities.linear_combination(
                        online_network.A_head[2].bias,
                        untrained_iqn_network.A_head[2].bias,
                        get_config().last_layer_reset_factor,
                    )
                    online_network.V_head[2].weight = utilities.linear_combination(
                        online_network.V_head[2].weight,
                        untrained_iqn_network.V_head[2].weight,
                        get_config().last_layer_reset_factor,
                    )
                    online_network.V_head[2].bias = utilities.linear_combination(
                        online_network.V_head[2].bias,
                        untrained_iqn_network.V_head[2].bias,
                        get_config().last_layer_reset_factor,
                    )

            # ===============================================
            #   LEARN ON BATCH
            # ===============================================

            if not online_network.training:
                online_network.train()

            while (
                len(buffer) >= memory_size_start_learn
                and accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used
                <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
            ):
                if (random.random() < get_config().buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                    test_start_time = time.perf_counter()
                    loss, _, _ = trainer.train_on_batch(buffer_test, do_learn=False)
                    time_testing_since_last_tensorboard_write += time.perf_counter() - test_start_time
                    loss_test_history.append(loss)
                    # Removed noisy test batch log
                else:
                    train_start_time = time.perf_counter()
                    loss, grad_norm, grad_norm_before_clip = trainer.train_on_batch(buffer, do_learn=True)
                    train_on_batch_duration_history.append(time.perf_counter() - train_start_time)
                    time_training_since_last_tensorboard_write += train_on_batch_duration_history[-1]
                    accumulated_stats["cumul_number_single_memories_used"] += (
                        4 * get_config().batch_size
                        if (len(buffer) < buffer._storage.max_size and buffer._storage.max_size > 200_000)
                        else get_config().batch_size
                    )  # do fewer batches while memory is not full
                    loss_history.append(loss)
                    if not math.isinf(grad_norm):
                        grad_norm_history.append(grad_norm)
                    if not math.isinf(grad_norm_before_clip):
                        grad_norm_before_clip_history.append(grad_norm_before_clip)
                    # Log gradient norms per layer for monitoring
                    utilities.log_gradient_norms(online_network, layer_grad_norm_history)

                    accumulated_stats["cumul_number_batches_done"] += 1
                    # Removed noisy training batch log - use TensorBoard for detailed metrics

                    utilities.custom_weight_decay(online_network, 1 - weight_decay)
                    if accumulated_stats["cumul_number_batches_done"] % get_config().send_shared_network_every_n_batches == 0:
                        with shared_network_lock:
                            uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

                    # ===============================================
                    #   UPDATE TARGET NETWORK
                    # ===============================================
                    if (
                        accumulated_stats["cumul_number_single_memories_used"]
                        >= accumulated_stats["cumul_number_single_memories_used_next_target_network_update"]
                    ):
                        accumulated_stats["cumul_number_target_network_updates"] += 1
                        accumulated_stats["cumul_number_single_memories_used_next_target_network_update"] += (
                            get_config().number_memories_trained_on_between_target_network_updates
                        )
                        # print("UPDATE")
                        utilities.soft_copy_param(target_network, online_network, get_config().soft_update_tau)
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
                
                # Buffer metrics
                "Buffer/size": len(buffer),
                "Buffer/max_size": buffer._storage.max_size,
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
            if isinstance(buffer._sampler, PrioritizedSampler):
                all_priorities = np.array([buffer._sampler._sum_tree.at(i) for i in range(len(buffer))])
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
            per_quantile_output = inferer.infer_network(rollout_results["frames"][0], rollout_results["state_float"][0], tau)
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
                # Log optimizer state (Adam/RAdam specific)
                for p, (name, _) in zip(
                    optimizer1.param_groups[0]["params"],
                    online_network.named_parameters(),
                ):
                    state = optimizer1.state[p]
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

            state_floats = np.array([experience.state_float for experience in buffer._storage])
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
            print(f"  Buffer size: {len(buffer):,} / {buffer._storage.max_size:,}")
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
            if get_config().make_highest_prio_figures and isinstance(buffer._sampler, PrioritizedSampler):
                highest_prio_transitions(buffer, save_dir)

            # ===============================================
            #   SAVE
            # ===============================================
            utilities.save_checkpoint(save_dir, online_network, target_network, optimizer1, scaler)
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
