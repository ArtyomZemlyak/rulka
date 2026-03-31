"""PPO: network factory, inferer, trainer hook, compile warmup (no-op for HF by default)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from config_files.config_loader import get_config
from trackmania_rl.agents.policy_models.ppo_actor_critic import make_ppo_network_pair


def build_ppo_policy_uncompiled(cfg=None):
    """Uncompiled PPO policy (multimodal / HF vision / CNN). Same routing as ``make_network`` without jit/CUDA."""
    c = cfg if cfg is not None else get_config()
    if c.transformers.fusion_mode != "none":
        from trackmania_rl.agents.policy_models.multimodal_torch_fusion import build_multimodal_fusion_uncompiled

        return build_multimodal_fusion_uncompiled(c)
    vt = c.vis.transformer
    if vt is not None and vt.use_hf_backbone:
        from trackmania_rl.agents.policy_models.hf_actor_critic import build_hf_actor_critic

        return build_hf_actor_critic(c)
    from trackmania_rl.agents.policy_models.ppo_actor_critic import build_ppo_actor_critic_uncompiled

    return build_ppo_actor_critic_uncompiled(c)


def make_network(jit: bool, is_inference: bool):
    cfg = get_config()
    if cfg.transformers.fusion_mode != "none":
        from trackmania_rl.agents.policy_models.multimodal_torch_fusion import make_multimodal_fusion_network_pair

        return make_multimodal_fusion_network_pair(jit, is_inference)
    vt = cfg.vis.transformer
    if vt is not None and vt.use_hf_backbone:
        from trackmania_rl.agents.policy_models.hf_actor_critic import make_hf_ppo_network_pair

        return make_hf_ppo_network_pair(cfg, jit, is_inference)
    return make_ppo_network_pair(jit, is_inference)


class PPOInferer:
    """Stochastic discrete policy; ``get_exploration_action`` returns a dict for game_instance_manager."""

    __slots__ = ("network", "is_explo")

    def __init__(self, network: nn.Module):
        self.network = network
        self.is_explo = True

    def get_exploration_action(
        self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray
    ) -> dict:
        cfg = get_config()
        with torch.no_grad():
            img = (
                torch.from_numpy(img_inputs_uint8)
                .unsqueeze(0)
                .to("cuda", memory_format=torch.channels_last, non_blocking=True, dtype=torch.float32)
                - 128
            ) / 128
            if not cfg.use_iqn_image_head:
                img = torch.zeros((1, 1, cfg.H_downsized, cfg.W_downsized), device="cuda", dtype=torch.float32)
            fl = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)

            out = self.network(img, fl)
            assert out.logits is not None
            logits = out.logits
            n_ab = cfg.n_actions_per_block
            n_act = len(cfg.inputs)
            value = float(out.value.item())

            if n_ab <= 1:
                dist = torch.distributions.Categorical(logits=logits[0])
                action = int(dist.sample().item())
                log_prob = float(dist.log_prob(torch.tensor(action, device=logits.device)).item())
                greedy = int(torch.argmax(logits[0]).item())
                q_proxy = logits[0].float().cpu().numpy()
                return {
                    "action": action,
                    "action_was_greedy": action == greedy,
                    "value": value,
                    "q_values": q_proxy,
                    "log_prob": log_prob,
                }

            lg = logits.reshape(1, n_ab, n_act)[0]
            actions = []
            log_probs = []
            greedy_actions = []
            for i in range(n_ab):
                d = torch.distributions.Categorical(logits=lg[i])
                a = int(d.sample().item())
                actions.append(a)
                log_probs.append(d.log_prob(torch.tensor(a, device=lg.device)))
                greedy_actions.append(int(torch.argmax(lg[i]).item()))
            action_arr = np.array(actions, dtype=np.int64)
            log_prob = float(torch.stack(log_probs).sum().item())
            greedy_arr = np.array(greedy_actions, dtype=np.int64)
            is_greedy = bool(np.all(action_arr == greedy_arr))
            return {
                "action": action_arr,
                "action_was_greedy": is_greedy,
                "value": value,
                "q_values": lg.float().cpu().numpy(),
                "log_prob": log_prob,
            }


def make_inferer(network):
    return PPOInferer(network)


class PPOLearnerOps:
    """So ``make_trainer`` matches IQN; PPO learner uses ``learner_ppo``."""

    __slots__ = ("policy", "optimizer", "scaler")

    def __init__(self, policy, optimizer, scaler):
        self.policy = policy
        self.optimizer = optimizer
        self.scaler = scaler


def make_trainer(online_network, target_network, optimizer, scaler, batch_size: int):
    _ = target_network
    _ = batch_size
    return PPOLearnerOps(online_network, optimizer, scaler)


def freeze_prefixes_from_config(cfg):
    """Parameter-name prefixes from ``nn.*.freeze`` (see ``trackmania_rl.param_freeze``)."""
    from trackmania_rl.param_freeze import collect_frozen_prefixes

    return collect_frozen_prefixes(cfg, wiring_algorithm="ppo")


def warmup_compile(config) -> None:
    if getattr(config, "algorithm", "iqn") != "ppo":
        return
    if not config.use_jit or (
        config.transformers.fusion_mode == "none"
        and (config.vis.transformer is not None and config.vis.transformer.use_hf_backbone)
    ):
        return
    print("\n[INFO] PPO: optional torch.compile warmup (backbone)...")
    c = get_config()
    net, _ = make_network(jit=True, is_inference=False)
    net.train()
    img = torch.zeros((1, 1, c.H_downsized, c.W_downsized), device="cuda", dtype=torch.float32)
    fl = torch.zeros((1, c.float_input_dim), device="cuda", dtype=torch.float32)
    for _ in range(2):
        net(img, fl)
    print("[OK] PPO warmup done.\n")
