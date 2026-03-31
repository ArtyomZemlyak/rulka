# =======================================================================================================================
# Train TrackMania RL agent. Configuration from YAML file (--config).
# =======================================================================================================================
# Config MUST be loaded before any trackmania_rl.agents.* import: agents.algorithms pulls in iqn.py, which
# uses get_config() when building networks / training; set_config(load_config(...)) must run first.

import argparse
import ctypes
import logging
import os
import random

# Disable inductor/Triton autotune console spam (must be set before torch is imported)
os.environ["TORCH_LOGS"] = "-inductor"
# Skip Triton GEMM autotune configs that exceed per-block shared memory (avoids
# RuntimeError: No valid triton configs on some consumer GPUs with max-autotune).
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM", "1")
# Hub/tqdm weight bars; inherited by collector/learner subprocesses
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import shutil
import signal
import sys
import time
import warnings
from pathlib import Path

# Suppress PyTorch TypedStorage deprecation warning (from inductor; fixed in future PyTorch)
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*", category=UserWarning)
# Hugging Face Hub (wording varies by package version)
warnings.filterwarnings("ignore", message=".*[Uu]nauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*[Rr]ate limit.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*[Ff]aster download.*")
# Transformers image processor when torchvision backend unavailable
warnings.filterwarnings("ignore", message=".*[Rr]equested torchvision backend.*")
warnings.filterwarnings("ignore", message=".*[Ff]alling back to pil backend.*")

# HF Hub token / rate-limit nags (logger or warnings; wording varies by version)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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

from trackmania_rl.agents.algorithms import get_wiring
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

    weights_existed = (save_dir / "weights1.torch").exists()
    pretrain_injected = False
    bc_heads_injected = False
    float_head_injected = False
    actions_head_injected = False
    pretrain_ppo_injected = False
    ppo_bc_inject_candidate = False

    _pretrain_paths_set = bool(
        config.pretrain_encoder_path
        or config.pretrain_bc_heads_path
        or config.pretrain_float_head_path
        or config.pretrain_actions_head_path
    )
    if config.algorithm != "iqn" and _pretrain_paths_set:
        print(
            "[WARN] IQN-only pretrain paths are ignored when training.algorithm != 'iqn' "
            f"(algorithm={config.algorithm!r})."
        )
    elif config.algorithm == "iqn":
        # --- Pretrain encoder injection ---
        # If pretrain_encoder_path is set and weights1.torch does not yet exist,
        # inject the pretrained img_head into a fresh IQN network pair so that
        # the learner and collectors start from the pretrained visual backbone.
        # Skipped automatically on resumed runs (weights1.torch already present).
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
        if config.pretrain_bc_heads_path and not weights_existed:
            from trackmania_rl.pretrain.export import inject_bc_heads_into_iqn

            bc_heads_injected = inject_bc_heads_into_iqn(
                bc_heads_path=Path(base_dir) / config.pretrain_bc_heads_path,
                save_dir=save_dir,
            )
            if bc_heads_injected:
                print(
                    "[OK] Pretrain BC full IQN state (img_head, float_feature_extractor, iqn_fc, A_head, V_head) injected."
                )

        # --- Pretrain piecewise: float_head.pt -> float_feature_extractor, actions_head.pt -> A_head ---
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

    # PPO BC: if ppo_policy_bc.pt exists, we inject after the first make_network (see skip_multimodal_fusion_hub_init_from_pretrained).
    if config.algorithm == "ppo" and config.pretrain_ppo_policy_path and not weights_existed:
        _bc_raw = Path(base_dir) / config.pretrain_ppo_policy_path
        _bc_pt = _bc_raw / "ppo_policy_bc.pt" if _bc_raw.is_dir() else _bc_raw
        if _bc_pt.is_file():
            ppo_bc_inject_candidate = True

    tensorboard_base_dir = Path(base_dir) / "tensorboard"

    # Copy Angelscript plugin to TMInterface dir
    shutil.copyfile(
        Path(base_dir) / "trackmania_rl" / "tmi_interaction" / "Python_Link.as",
        config.target_python_link_path,
    )

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
    wiring = get_wiring()
    _, uncompiled_shared_network = wiring.make_network(jit=config.use_jit, is_inference=False)

    if ppo_bc_inject_candidate:
        from trackmania_rl.pretrain.export import inject_ppo_bc_policy_into_save_dir

        try:
            pretrain_ppo_injected = inject_ppo_bc_policy_into_save_dir(
                Path(base_dir) / config.pretrain_ppo_policy_path,
                save_dir,
                uncompiled=uncompiled_shared_network,
            )
            if pretrain_ppo_injected:
                print(
                    "[OK] PPO: BC policy (ppo_policy_bc.pt) → weights1.torch "
                    f"from {config.pretrain_ppo_policy_path!r}."
                )
                print(
                    "[INFO] PPO: if nn.init_from_pretrained is set, hub load is skipped when "
                    "weights1.torch or BC ppo_policy_bc.pt is present (automatic)."
                )
        except Exception as e:
            print(f"[ERROR] PPO BC pretrain injection failed: {e}")
            sys.exit(1)

    # Collectors start before the learner runs; their first ``update_network`` copies ``shared_state_dict``
    # into local policy. That dict aliases parent parameters only if they already match the checkpoint.
    # On resume, ``make_network`` above used fresh HF init — load ``weights1.torch`` here so shared sync is correct.
    if config.algorithm == "ppo":
        w1_align = save_dir / "weights1.torch"
        if w1_align.is_file():
            from trackmania_rl import utilities as _trl_utilities

            _sd0 = torch.load(w1_align, map_location="cpu", weights_only=False)
            _slice = bool(getattr(config, "pretrain_ppo_policy_slice_head_to_model", False))
            _prep0 = _trl_utilities.prepare_ppo_policy_state_dict_for_load(
                _sd0, uncompiled_shared_network, slice_policy_head_to_model=_slice
            )
            uncompiled_shared_network.load_state_dict(_prep0, strict=True)
            if weights_existed:
                print(
                    "[OK] PPO: parent policy synced from weights1.torch before collectors (resume / shared_state_dict)."
                )

    with shared_network_lock:
        uncompiled_shared_network.share_memory()
    # Snapshot of shared-memory tensors for cross-process weight sync.
    # Parametrized nn.Modules can't be pickled (Windows spawn), but this
    # plain dict of shared-memory tensors can. Tensors share storage with
    # the module's parameters, so learner writes are visible to collectors.
    shared_state_dict = uncompiled_shared_network.state_dict()

    print("\n" + "=" * 80)
    tprint("Rulka", font="tarty1")
    print("=" * 80)
    print(f"  Run name: {config.run_name}")
    print(f"  Algorithm: {config.algorithm}")
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
    if config.pretrain_ppo_policy_path:
        print(
            f"  Pretrain PPO (BC): {config.pretrain_ppo_policy_path}"
            + (" (injected)" if pretrain_ppo_injected else " (skipped — checkpoint exists or not used)")
        )
    print("=" * 80)
    print("\n[INFO] Starting training...\n")

    # --- Compilation Warmup (Windows Stability) ---
    wiring.warmup_compile(config)

    # Start worker processes (each loads config from config_path)
    collector_processes = [
        mp.Process(
            target=collector_process_fn,
            args=(
                config_path,
                rollout_queue,
                shared_state_dict,
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
