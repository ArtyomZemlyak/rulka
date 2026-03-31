"""BC ``build_rl_policy_for_bc``: CPU forwards and tensor shapes (no forced CUDA)."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
_RL_IQN = _REPO / "config_files" / "rl" / "config_default.yaml"
_RL_PPO = _REPO / "config_files" / "rl" / "config_ppo.yaml"


class TestBcRlPolicyFactory(unittest.TestCase):
    def test_iqn_single_offset_forward(self):
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.pretrain.rl_policy_factory import build_rl_policy_for_bc

        self.assertTrue(_RL_IQN.is_file(), msg=f"missing {_RL_IQN}")
        set_config(load_config(_RL_IQN))
        cfg = get_config()
        self.assertEqual(cfg.algorithm, "iqn")

        model = build_rl_policy_for_bc(n_bc_offsets=1, bc_multi_offset_mode="separate_heads")
        model.eval()
        B = 2
        img = torch.zeros(B, 1, cfg.H_downsized, cfg.W_downsized, dtype=torch.float32)
        fl = torch.zeros(B, cfg.float_input_dim, dtype=torch.float32)
        tau = torch.full((B, 1), 0.5, dtype=torch.float32)
        Q, _tau = model(img, fl, 1, tau=tau)
        n_act = len(cfg.inputs)
        self.assertEqual(Q.shape, (B, n_act))

    def test_iqn_multi_offset_forward(self):
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.pretrain.rl_policy_factory import build_rl_policy_for_bc

        set_config(load_config(_RL_IQN))
        cfg = get_config()
        n_act = len(cfg.inputs)
        model = build_rl_policy_for_bc(n_bc_offsets=3, bc_multi_offset_mode="separate_heads")
        model.eval()
        B = 2
        img = torch.zeros(B, 1, cfg.H_downsized, cfg.W_downsized, dtype=torch.float32)
        fl = torch.zeros(B, cfg.float_input_dim, dtype=torch.float32)
        tau = torch.full((B, 1), 0.5, dtype=torch.float32)
        Q, _ = model(img, fl, 1, tau=tau)
        self.assertEqual(Q.shape, (B, 3, n_act))
        tgt = torch.zeros(B, 3, dtype=torch.long)
        loss = sum(F.cross_entropy(Q[:, i], tgt[:, i]) for i in range(3))
        self.assertTrue(torch.isfinite(loss))

    def test_ppo_single_offset_policy_output(self):
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.pretrain.rl_policy_factory import build_rl_policy_for_bc

        self.assertTrue(_RL_PPO.is_file(), msg=f"missing {_RL_PPO}")
        set_config(load_config(_RL_PPO))
        cfg = get_config()
        self.assertEqual(cfg.algorithm, "ppo")

        model = build_rl_policy_for_bc(n_bc_offsets=1, bc_multi_offset_mode="separate_heads")
        model.eval()
        B = 2
        img = torch.zeros(B, 1, cfg.H_downsized, cfg.W_downsized, dtype=torch.float32)
        fl = torch.zeros(B, cfg.float_input_dim, dtype=torch.float32)
        out = model(img, fl)
        n_act = len(cfg.inputs)
        self.assertTrue(hasattr(out, "logits"))
        if cfg.n_actions_per_block <= 1:
            self.assertEqual(out.logits.shape, (B, n_act))
        else:
            self.assertEqual(out.logits.shape[0], B)
            self.assertEqual(out.logits.numel(), B * cfg.n_actions_per_block * n_act)

    def test_ppo_multi_offset_bc_heads_shape(self):
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.pretrain.rl_policy_factory import build_rl_policy_for_bc

        set_config(load_config(_RL_PPO))
        cfg = get_config()
        n_act = len(cfg.inputs)
        n_off = 4
        model = build_rl_policy_for_bc(n_bc_offsets=n_off, bc_multi_offset_mode="separate_heads")
        model.eval()
        B = 2
        img = torch.zeros(B, 1, cfg.H_downsized, cfg.W_downsized, dtype=torch.float32)
        fl = torch.zeros(B, cfg.float_input_dim, dtype=torch.float32)
        logits = model(img, fl)
        self.assertEqual(logits.shape, (B, n_off, n_act))
        tgt = torch.randint(0, n_act, (B, n_off), dtype=torch.long)
        loss = sum(F.cross_entropy(logits[:, i], tgt[:, i]) for i in range(n_off))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
