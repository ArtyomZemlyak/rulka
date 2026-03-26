"""IQN: thin delegates for multiprocess RL entry points."""

import torch

from config_files.config_loader import get_config
from trackmania_rl.agents import iqn as iqn_mod
from trackmania_rl.agents.iqn import make_untrained_iqn_network


def make_network(jit: bool, is_inference: bool):
    return make_untrained_iqn_network(jit, is_inference)


def make_trainer(online_network, target_network, optimizer, scaler, batch_size: int):
    return iqn_mod.Trainer(
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        scaler=scaler,
        batch_size=batch_size,
        iqn_n=get_config().iqn_n,
    )


def make_inferer(network):
    cfg = get_config()
    return iqn_mod.Inferer(
        inference_network=network,
        iqn_k=cfg.iqn_k,
        tau_epsilon_boltzmann=cfg.tau_epsilon_boltzmann,
    )


def freeze_prefixes_from_config(cfg):
    """Parameter name prefixes for pretrain freeze (IQN module names)."""
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


def warmup_compile(config) -> None:
    """Populate torch.compile cache in the main process before spawning collectors (Windows Triton)."""
    if not config.use_jit:
        return
    print("\n[INFO] Warming up torch.compile (Populating Triton cache)...")
    dummy_img = torch.zeros(
        (1, 1, config.H_downsized, config.W_downsized), device="cuda", dtype=torch.float32
    )
    dummy_float = torch.zeros((1, config.float_input_dim), device="cuda", dtype=torch.float32)

    inf_net, _ = make_untrained_iqn_network(jit=True, is_inference=True)
    inf_net.eval()
    with torch.no_grad():
        for _ in range(2):
            inf_net(dummy_img, dummy_float, config.iqn_k)

    train_net, _ = make_untrained_iqn_network(jit=True, is_inference=False)
    train_net.train()
    for _ in range(2):
        train_net(dummy_img, dummy_float, config.iqn_k)

    print("[OK] Warmup complete. Triton cache is now populated.\n")
