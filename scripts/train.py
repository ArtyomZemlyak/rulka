# =======================================================================================================================
# Train TrackMania RL agent. Configuration from YAML file (--config).
# =======================================================================================================================
# Config MUST be loaded before any trackmania_rl imports (iqn etc need get_config at import time)

import argparse
import ctypes
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

# Parse args and load config first
parser = argparse.ArgumentParser(description="Train TrackMania RL agent")
parser.add_argument(
    "--config",
    type=str,
    default="config_files/rl/config_default.yaml",
    help="Path to YAML config file",
)
args = parser.parse_args()
base_dir = Path(__file__).resolve().parents[1]
config_path = base_dir / args.config
if not config_path.is_file():
    print(f"ERROR: Config file not found: {config_path}")
    sys.exit(1)

from config_files.config_loader import load_config, set_config, get_config
set_config(load_config(config_path))

import numpy as np
import torch
import torch.multiprocessing as mp
from art import tprint
from torch.multiprocessing import Lock

from trackmania_rl.agents.iqn import make_untrained_iqn_network
from trackmania_rl.multiprocess.collector_process import collector_process_fn
from trackmania_rl.multiprocess.learner_process import learner_process_fn
from trackmania_rl.utilities import set_random_seed

# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")
random_seed = 444
set_random_seed(random_seed)


def signal_handler(sig, frame):
    print("Received SIGINT signal. Killing all open Trackmania instances.")
    clear_tm_instances()

    for child in mp.active_children():
        child.kill()

    tprint("Bye bye!", font="tarty1")
    sys.exit()


def clear_tm_instances():
    config = get_config()
    if config.is_linux:
        os.system("pkill -9 TmForever.exe")
    else:
        os.system("taskkill /F /IM TmForever.exe")


if __name__ == "__main__":
    config = get_config()  # Already loaded above

    signal.signal(signal.SIGINT, signal_handler)

    clear_tm_instances()

    save_dir = Path(base_dir) / "save" / config.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot to experiment folder
    shutil.copy(config_path, save_dir / "config_snapshot.yaml")

    # --- Pretrain encoder injection ---
    # If pretrain_encoder_path is set and weights1.torch does not yet exist,
    # inject the pretrained img_head into a fresh IQN network pair so that
    # the learner and collectors start from the pretrained visual backbone.
    # Skipped automatically on resumed runs (weights1.torch already present).
    weights_existed = (save_dir / "weights1.torch").exists()
    pretrain_injected = False
    if config.pretrain_encoder_path:
        from trackmania_rl.pretrain.export import inject_encoder_into_iqn
        pretrain_injected = inject_encoder_into_iqn(
            encoder_pt=Path(base_dir) / config.pretrain_encoder_path,
            save_dir=save_dir,
            overwrite=False,
        )
        if pretrain_injected:
            print("[OK] Pretrain encoder injected; training will start from pretrained img_head.")

    # --- Pretrain BC full IQN injection (from iqn_bc.pt) ---
    # Only on fresh run (weights did not exist at start): load full BC IQN state into current weights.
    bc_heads_injected = False
    if config.pretrain_bc_heads_path and not weights_existed:
        from trackmania_rl.pretrain.export import inject_bc_heads_into_iqn
        bc_heads_injected = inject_bc_heads_into_iqn(
            bc_heads_path=Path(base_dir) / config.pretrain_bc_heads_path,
            save_dir=save_dir,
        )
        if bc_heads_injected:
            print("[OK] Pretrain BC full IQN state (img_head, float_feature_extractor, iqn_fc, A_head, V_head) injected.")

    # --- Pretrain piecewise: float_head.pt -> float_feature_extractor, actions_head.pt -> A_head ---
    float_head_injected = False
    actions_head_injected = False
    if config.pretrain_float_head_path and not weights_existed:
        from trackmania_rl.pretrain.export import inject_float_head_into_iqn
        float_head_injected = inject_float_head_into_iqn(
            float_head_path=Path(base_dir) / config.pretrain_float_head_path,
            save_dir=save_dir,
        )
        if float_head_injected:
            print("[OK] Pretrain float_feature_extractor (float_head.pt) injected.")
    if config.pretrain_actions_head_path and not weights_existed:
        from trackmania_rl.pretrain.export import inject_actions_head_into_iqn
        actions_head_injected = inject_actions_head_into_iqn(
            actions_head_path=Path(base_dir) / config.pretrain_actions_head_path,
            save_dir=save_dir,
        )
        if actions_head_injected:
            print("[OK] Pretrain A_head (actions_head.pt) injected.")

    tensorboard_base_dir = Path(base_dir) / "tensorboard"

    # Copy Angelscript plugin to TMInterface dir
    shutil.copyfile(
        Path(base_dir) / "trackmania_rl" / "tmi_interaction" / "Python_Link.as",
        config.target_python_link_path,
    )

    print("\n" + "=" * 80)
    tprint("Rulka", font="tarty1")
    print("=" * 80)
    print(f"  Run name: {config.run_name}")
    print(f"  GPU collectors: {config.gpu_collectors_count}")
    print(f"  Base TMI port: {config.base_tmi_port}")
    print(f"  Save directory: {save_dir}")
    print(f"  Config: {config_path}")
    if config.pretrain_encoder_path:
        print(f"  Pretrain encoder: {config.pretrain_encoder_path}" + (" (injected)" if pretrain_injected else " (skipped — checkpoint exists)"))
    if config.pretrain_bc_heads_path:
        print(f"  Pretrain BC full IQN: {config.pretrain_bc_heads_path}" + (" (injected)" if bc_heads_injected else " (skipped — checkpoint exists)"))
    if config.pretrain_float_head_path:
        print(f"  Pretrain float head: {config.pretrain_float_head_path}" + (" (injected)" if float_head_injected else " (skipped)"))
    if config.pretrain_actions_head_path:
        print(f"  Pretrain actions head: {config.pretrain_actions_head_path}" + (" (injected)" if actions_head_injected else " (skipped)"))
    print("=" * 80)
    print("\n[INFO] Starting training...\n")

    if config.is_linux:
        os.system(f"chmod +x {config.linux_launch_game_path}")

    # Prepare multi process utilities
    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0
    rollout_queues = [
        mp.Queue(config.max_rollout_queue_size)
        for _ in range(config.gpu_collectors_count)
    ]
    shared_network_lock = Lock()
    game_spawning_lock = Lock()
    _, uncompiled_shared_network = make_untrained_iqn_network(
        jit=config.use_jit, is_inference=False
    )
    uncompiled_shared_network.share_memory()

    # Start worker processes (each loads config from config_path)
    collector_processes = [
        mp.Process(
            target=collector_process_fn,
            args=(
                config_path,
                rollout_queue,
                uncompiled_shared_network,
                shared_network_lock,
                game_spawning_lock,
                shared_steps,
                base_dir,
                save_dir,
                config.base_tmi_port + process_number,
                process_number,
            ),
        )
        for rollout_queue, process_number in zip(
            rollout_queues, range(config.gpu_collectors_count)
        )
    ]
    for collector_process in collector_processes:
        collector_process.start()

    # Start learner process (runs in main process, config already set)
    learner_process_fn(
        rollout_queues,
        uncompiled_shared_network,
        shared_network_lock,
        shared_steps,
        Path(base_dir),
        save_dir,
        tensorboard_base_dir,
    )

    for collector_process in collector_processes:
        collector_process.join()
