"""Repro / regression test for torch.compile + NoisyLinear + AMP (matches learner path).

The training bug was:
  RuntimeError: CompiledFunctionBackward ... got [512] but expected [1, 512]

It could appear only after many steps (Dynamo recompile, autotune, or autocast path).

Run (from repo root, with venv):
  .venv\\Scripts\\python.exe scripts/test_compile_backward.py

Args:
  --steps N   number of train-like iterations (default 300)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from config_files.config_loader import ConfigView, set_config


def _make_minimal_config():
    from config_files.config_schema import (
        BTRConfig,
        EnvironmentConfig,
        ExplorationConfig,
        InputsConfig,
        MapCycleConfig,
        MemoryConfig,
        NeuralNetworkConfig,
        PerformanceConfig,
        RewardsConfig,
        RulkaConfig,
        StateNormalizationConfig,
        TrainingConfig,
        UserConfig,
    )

    env = EnvironmentConfig()
    nn_cfg = NeuralNetworkConfig()
    nn_cfg.float_input_dim = (
        27
        + 3 * env.n_zone_centers_in_inputs
        + 4 * env.n_prev_actions_in_inputs
        + 4 * env.n_contact_material_physics_behavior_types
        + 1
    )
    tr = TrainingConfig()
    mem = MemoryConfig()
    exp = ExplorationConfig()
    rew = RewardsConfig()
    mc = MapCycleConfig(entries=[], map_cycle=[])
    perf = PerformanceConfig()
    inp = InputsConfig(actions=[], action_forward_idx=0, action_backward_idx=6)
    sn = StateNormalizationConfig(
        waypoint_mean_40cp=[0.0] * 120,
        waypoint_std_40cp=[1.0] * 120,
        float_inputs_mean=np.zeros(nn_cfg.float_input_dim),
        float_inputs_std=np.ones(nn_cfg.float_input_dim),
    )
    btr = BTRConfig(
        use_impala_cnn=True,
        impala_model_size=2,
        use_adaptive_maxpool=True,
        adaptive_maxpool_size=6,
        use_spectral_norm=True,
        use_layer_norm=True,
        use_noisy_linear=True,
        noisy_sigma0=0.5,
    )
    return RulkaConfig(
        environment=env,
        neural_network=nn_cfg,
        training=tr,
        memory=mem,
        exploration=exp,
        rewards=rew,
        map_cycle=mc,
        performance=perf,
        inputs=inp,
        state_normalization=sn,
        user=UserConfig(),
        btr=btr,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300, help="train-like iterations")
    args = ap.parse_args()

    raw = _make_minimal_config()
    set_config(ConfigView(raw))
    cfg = raw

    from trackmania_rl.agents.iqn import IQN_Network, _build_img_head, calculate_conv_output_dim

    H, W = 64, 64
    float_input_dim = cfg.neural_network.float_input_dim
    float_hidden_dim = cfg.neural_network.float_hidden_dim
    dense_hidden_dim = cfg.neural_network.dense_hidden_dimension
    iqn_embed_dim = cfg.neural_network.iqn_embedding_dimension
    n_actions = 12
    iqn_n = 8
    batch_size = 32

    tmp_head = _build_img_head(
        use_impala_cnn=True,
        impala_model_size=2,
        use_spectral_norm=True,
        use_adaptive_maxpool=True,
        adaptive_maxpool_size=6,
    )
    conv_dim = calculate_conv_output_dim(tmp_head, H, W)

    print(f"conv_dim={conv_dim} float_dim={float_input_dim} batch={batch_size} iqn_n={iqn_n}")
    print(f"steps={args.steps} (autocast fp16 + GradScaler + compile max-autotune-no-cudagraphs)\n")

    model = (
        IQN_Network(
            float_inputs_dim=float_input_dim,
            float_hidden_dim=float_hidden_dim,
            conv_head_output_dim=conv_dim,
            dense_hidden_dimension=dense_hidden_dim,
            iqn_embedding_dimension=iqn_embed_dim,
            n_actions=n_actions,
            float_inputs_mean=np.zeros(float_input_dim, dtype=np.float32),
            float_inputs_std=np.ones(float_input_dim, dtype=np.float32),
            use_impala_cnn=True,
            impala_model_size=2,
            use_adaptive_maxpool=True,
            adaptive_maxpool_size=6,
            use_spectral_norm=True,
            use_layer_norm=True,
            use_noisy_linear=True,
            noisy_sigma0=0.5,
        )
        .cuda()
        .train()
    )

    compiled = torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    dummy_img = torch.randn(batch_size, 1, H, W, device="cuda", dtype=torch.float32)
    dummy_float = torch.randn(batch_size, float_input_dim, device="cuda", dtype=torch.float32)

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        model.reset_noise()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            Q, _tau = compiled(dummy_img, dummy_float, iqn_n, tau=None)
            loss = Q.float().sum()

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        scaler.step(opt)
        scaler.update()

        if (step + 1) % 50 == 0:
            print(f"  step {step + 1}/{args.steps} OK")

    print("\nALL PASSED")


if __name__ == "__main__":
    main()
