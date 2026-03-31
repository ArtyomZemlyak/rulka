"""Smoke: RL config + IQN wiring (stdlib unittest; no pytest dependency)."""

import os
import tempfile
from pathlib import Path
import unittest

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RL_DEFAULT = _REPO_ROOT / "config_files" / "rl" / "config_default.yaml"
_RL_PPO = _REPO_ROOT / "config_files" / "rl" / "config_ppo.yaml"


@unittest.skipUnless(torch.cuda.is_available(), "PPO factory places weights on CUDA (same contract as IQN)")
class TestPPOWiringSmoke(unittest.TestCase):
    """PPO CNN policy forward + one loss backward on GPU."""

    def test_ppo_make_network_forward_and_loss(self):
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.agents.algorithms.registry import get_wiring
        from trackmania_rl.agents.policy_optimization.ppo import ppo_loss_components

        self.assertTrue(_RL_PPO.is_file(), msg=f"missing {_RL_PPO}")
        set_config(load_config(_RL_PPO))
        cfg = get_config()
        self.assertEqual(cfg.algorithm, "ppo")

        w = get_wiring("ppo")
        self.assertIsInstance(w.freeze_prefixes_from_config(cfg), list)
        net, _ = w.make_network(False, False)
        self.assertGreater(len(net.state_dict()), 0)

        img = torch.zeros(
            1, 1, cfg.H_downsized, cfg.W_downsized, device="cuda", dtype=torch.float32
        )
        fl = torch.zeros(1, cfg.float_input_dim, device="cuda", dtype=torch.float32)
        if cfg.n_actions_per_block <= 1:
            actions = torch.zeros(1, dtype=torch.long, device="cuda")
        else:
            actions = torch.zeros(1, cfg.n_actions_per_block, dtype=torch.long, device="cuda")

        logp, ent, vals, _ = net.evaluate_actions(img, fl, actions)
        old_logp = logp.detach()
        advantages = torch.zeros_like(logp)
        returns = vals.detach()
        loss, metrics = ppo_loss_components(
            logp,
            old_logp,
            advantages,
            vals,
            returns,
            ent,
            clip_coef=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("approx_kl", metrics)
        self.assertIn("vf_clipfrac", metrics)


@unittest.skipUnless(torch.cuda.is_available(), "IQN factory places weights on CUDA")
class TestRLWiringSmoke(unittest.TestCase):
    def test_get_wiring_iqn_make_network_state_dict(self):
        from config_files.config_loader import load_config, set_config
        from trackmania_rl.agents.algorithms.registry import get_wiring

        self.assertTrue(_RL_DEFAULT.is_file(), msg=f"missing {_RL_DEFAULT}")
        set_config(load_config(_RL_DEFAULT))
        w = get_wiring("iqn")
        net, _ = w.make_network(False, False)
        sd = net.state_dict()
        self.assertGreater(len(sd), 0)

    def test_freeze_prefixes_from_config_returns_list(self):
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.agents.algorithms.registry import get_wiring

        set_config(load_config(_RL_DEFAULT))
        prefixes = get_wiring("iqn").freeze_prefixes_from_config(get_config())
        self.assertIsInstance(prefixes, list)

    def test_iqn_vision_transformer_fusion_forward(self):
        """IQN on native multimodal backbone (no HF): same fusion graph minus policy heads."""
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.agents.algorithms.registry import get_wiring

        with open(_RL_DEFAULT, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data.setdefault("training", {})["algorithm"] = "iqn"
        data.setdefault("nn", {})["fusion_mode"] = "vision_transformer"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(data, tmp, sort_keys=False)
            path = tmp.name
        try:
            set_config(load_config(path))
            w = get_wiring("iqn")
            net, _ = w.make_network(False, False)
            cfg = get_config()
            img = torch.zeros(2, 1, cfg.H_downsized, cfg.W_downsized, device="cuda", dtype=torch.float32)
            fl = torch.zeros(2, cfg.float_input_dim, device="cuda", dtype=torch.float32)
            Q, _tau = net(img, fl, cfg.iqn_n)
            self.assertEqual(Q.shape[0], 2 * cfg.iqn_n)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
