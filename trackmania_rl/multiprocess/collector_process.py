"""
This file implements a single multithreaded worker that handles a Trackmania game instance and provides rollout results to the learner process.
"""

import copy
import time
from itertools import count, cycle
from pathlib import Path

import numpy as np
import torch
from torch import multiprocessing as mp

from config_files.config_loader import load_config, set_config, get_config
from trackmania_rl import utilities
from trackmania_rl.agents import iqn as iqn
from trackmania_rl.utilities import set_random_seed


def collector_process_fn(
    config_path: Path,
    rollout_queue,
    uncompiled_shared_network,
    shared_network_lock,
    game_spawning_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tmi_port: int,
    process_number: int,
):
    from trackmania_rl.map_loader import analyze_map_cycle, load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager

    # Load config once per process (no hot-reload)
    set_config(load_config(config_path))
    config = get_config()

    set_random_seed(process_number)

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        running_speed=config.running_speed,
        run_steps_per_action=config.tm_engine_step_per_action,
        max_overall_duration_ms=config.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=tmi_port,
    )

    inference_network, uncompiled_inference_network = iqn.make_untrained_iqn_network(config.use_jit, is_inference=True)
    try:
        inference_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
    except Exception as e:
        print(f"[INFO] Worker {process_number} starting with fresh weights")

    inferer = iqn.Inferer(inference_network, config.iqn_k, config.tau_epsilon_boltzmann)

    def update_network():
        # Update weights of the inference network
        with shared_network_lock:
            uncompiled_inference_network.load_state_dict(uncompiled_shared_network.state_dict())

    # ========================================================
    # Training loop
    # ========================================================
    inference_network.train()

    # Initialize map cycle
    print(f"[Collector {process_number}] Initializing map cycle...")
    print(f"[Collector {process_number}] map_cycle length: {len(config.map_cycle)}")
    
    if not config.map_cycle:
        print(f"[Collector {process_number}] ERROR: map_cycle is EMPTY at initialization!")
        print(f"[Collector {process_number}] Please configure maps in YAML config file.")
        raise ValueError("map_cycle cannot be empty. Configure at least one map.")
    
    map_cycle_str = str(config.map_cycle)
    set_maps_trained, set_maps_blind = analyze_map_cycle(config.map_cycle)
    map_cycle_iter = cycle(copy.deepcopy(config.map_cycle))
    
    print(f"[Collector {process_number}] Map cycle initialized successfully")
    print(f"[Collector {process_number}] Training maps: {set_maps_trained}")
    print(f"[Collector {process_number}] Blind test maps: {set_maps_blind}")

    zone_centers_filename = None

    # ========================================================
    # Warmup pytorch and numba
    # ========================================================
    for _ in range(5):
        inferer.infer_network(
            np.random.randint(low=0, high=255, size=(1, config.H_downsized, config.W_downsized), dtype=np.uint8),
            np.random.rand(config.float_input_dim).astype(np.float32),
        )
    # game_instance_manager.update_current_zone_idx(0, zone_centers, np.zeros(3))

    time_since_last_queue_push = time.perf_counter()
    for loop_number in count(1):
        # Config is fixed at startup (no hot-reload)
        tmi.max_minirace_duration_ms = config.cutoff_rollout_if_no_vcp_passed_within_duration_ms

        # ===============================================
        #   DID THE CYCLE CHANGE ? (not applicable - config fixed)
        # ===============================================
        if str(config.map_cycle) != map_cycle_str:
            map_cycle_str = str(config.map_cycle)
            
            # Validate map_cycle is not empty
            if not config.map_cycle:
                print(f"[Collector {process_number}] ERROR: map_cycle is EMPTY!")
                print(f"[Collector {process_number}] Check YAML config file.")
                raise ValueError("map_cycle cannot be empty. Please configure at least one map in config.")
            
            print(f"[Collector {process_number}] Map cycle updated. Number of cycle elements: {len(config.map_cycle)}")
            set_maps_trained, set_maps_blind = analyze_map_cycle(config.map_cycle)
            map_cycle_iter = cycle(copy.deepcopy(config.map_cycle))
            print(f"[Collector {process_number}] Maps for training: {set_maps_trained}")
            print(f"[Collector {process_number}] Maps for blind testing: {set_maps_blind}")

        # ===============================================
        #   GET NEXT MAP FROM CYCLE
        # ===============================================
        try:
            next_map_tuple = next(map_cycle_iter)
        except StopIteration:
            print(f"[Collector {process_number}] ERROR: StopIteration in map_cycle!")
            print(f"[Collector {process_number}] map_cycle length: {len(config.map_cycle)}")
            print(f"[Collector {process_number}] map_cycle contents: {config.map_cycle}")
            raise RuntimeError(f"map_cycle iterator exhausted unexpectedly. This should not happen with cycle().")
        if next_map_tuple[2] != zone_centers_filename:
            zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
        map_name, map_path, zone_centers_filename, is_explo, fill_buffer = next_map_tuple
        map_status = "trained" if map_name in set_maps_trained else "blind"

        inferer.epsilon = utilities.from_exponential_schedule(config.epsilon_schedule, shared_steps.value)
        inferer.epsilon_boltzmann = utilities.from_exponential_schedule(config.epsilon_boltzmann_schedule, shared_steps.value)
        inferer.tau_epsilon_boltzmann = config.tau_epsilon_boltzmann
        inferer.is_explo = is_explo

        # ===============================================
        #   PLAY ONE ROUND
        # ===============================================

        rollout_start_time = time.perf_counter()

        if inference_network.training and not is_explo:
            inference_network.eval()
        elif is_explo and not inference_network.training:
            inference_network.train()

        update_network()

        rollout_start_time = time.perf_counter()
        rollout_results, end_race_stats = tmi.rollout(
            exploration_policy=inferer.get_exploration_action,
            map_path=map_path,
            zone_centers=zone_centers,
            update_network=update_network,
        )
        rollout_end_time = time.perf_counter()
        rollout_duration = rollout_end_time - rollout_start_time
        rollout_results["worker_time_in_rollout_percentage"] = rollout_duration / (time.perf_counter() - time_since_last_queue_push)
        time_since_last_queue_push = time.perf_counter()

        if not tmi.last_rollout_crashed:
            rollout_queue.put(
                (
                    rollout_results,
                    end_race_stats,
                    fill_buffer,
                    is_explo,
                    map_name,
                    map_status,
                    rollout_duration,
                    loop_number,
                )
            )
