"""Unit tests for PPO/GAE helpers (CPU, no game)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from config_files.config_loader import load_config
from trackmania_rl.agents.policy_optimization.ppo import compute_gae, ppo_loss_components
from trackmania_rl.agents.policy_optimization.rollout_rewards import (
    _fold_potential_into_ppo_step_rewards,
    ppo_rewards_and_dones_from_rollout,
)
from trackmania_rl.reward_vectorized import (
    compute_dense_reward_per_action_t,
    compute_rewards_into_and_potentials,
    state_float_slice_indices,
)


class TestComputeGAE(unittest.TestCase):
    def test_terminal_no_bootstrap(self):
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([1.0])
        next_value = torch.tensor(99.0)
        adv, ret = compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95)
        self.assertEqual(adv.shape, (1,))
        self.assertAlmostEqual(float(adv[0]), 0.5, places=5)
        self.assertAlmostEqual(float(ret[0]), 1.0, places=5)

    def test_two_steps_shape(self):
        rewards = torch.tensor([0.0, 1.0])
        values = torch.tensor([0.1, 0.2])
        dones = torch.tensor([0.0, 0.0])
        next_value = torch.tensor(0.3)
        adv, ret = compute_gae(rewards, values, dones, next_value, gamma=0.9, gae_lambda=1.0)
        self.assertEqual(adv.shape, (2,))
        self.assertEqual(ret.shape, (2,))
        self.assertTrue(torch.allclose(ret, adv + values))


class TestPPOLossComponents(unittest.TestCase):
    def test_clipped_value_loss_is_max_of_two_ms_es(self):
        B = 4
        new_lp = torch.zeros(B, requires_grad=True)
        old_lp = torch.zeros(B)
        advantages = torch.zeros(B)
        new_values = torch.tensor([10.0, 0.0, 5.0, 1.0], requires_grad=True)
        old_values = torch.tensor([0.0, 0.0, 0.0, 0.0])
        returns = torch.zeros(B)
        entropy = torch.zeros(B)
        loss_plain, _ = ppo_loss_components(
            new_lp,
            old_lp,
            advantages,
            new_values,
            returns,
            entropy,
            clip_coef=0.2,
            vf_coef=1.0,
            ent_coef=0.0,
        )
        loss_clip, m = ppo_loss_components(
            new_lp,
            old_lp,
            advantages,
            new_values,
            returns,
            entropy,
            clip_coef=0.2,
            vf_coef=1.0,
            ent_coef=0.0,
            old_values=old_values,
            clip_coef_vf=0.2,
        )
        # Per sample max(unclipped^2, clipped^2) >= unclipped^2 → aggregate vf loss >= plain MSE.
        self.assertGreaterEqual(float(loss_clip.detach()), float(loss_plain.detach()))
        self.assertIn("vf_clipfrac", m)
        self.assertGreater(float(m["vf_clipfrac"]), 0.0)

    def test_ratio_clips_policy_term(self):
        # log-ratio = 1.5 → ratio = e^1.5; clip(ratio, 0.8, 1.2) = 1.2; adv=+1 → term uses 1.2
        old_lp = torch.zeros(1)
        new_lp = torch.tensor([1.5])
        advantages = torch.ones(1)
        new_values = torch.zeros(1)
        returns = torch.zeros(1)
        entropy = torch.zeros(1)
        loss, m = ppo_loss_components(
            new_lp,
            old_lp,
            advantages,
            new_values,
            returns,
            entropy,
            clip_coef=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(m["loss_policy"]), -1.2, places=5)

    def test_backward_finite(self):
        B = 4
        old_lp = torch.zeros(B, requires_grad=False)
        new_lp = torch.randn(B, requires_grad=True)
        advantages = torch.randn(B)
        new_values = torch.randn(B, requires_grad=True)
        returns = torch.randn(B)
        entropy = torch.abs(torch.randn(B, requires_grad=True))
        loss, _ = ppo_loss_components(
            new_lp,
            old_lp,
            advantages,
            new_values,
            returns,
            entropy,
            clip_coef=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
        )
        loss.backward()
        self.assertIsNotNone(new_lp.grad)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RL_PPO_CFG = _REPO_ROOT / "config_files" / "rl" / "config_ppo.yaml"


class TestDenseRewardPerDecision(unittest.TestCase):
    """Mass-preserving split + meter/final_speed/constant shifted to action a_t."""

    def test_sum_matches_rewards_into(self):
        self.assertTrue(_RL_PPO_CFG.is_file())
        cfg = load_config(_RL_PPO_CFG)
        n = 11
        fd = cfg.float_input_dim
        rng = np.random.default_rng(42)
        w0s, w0e, *_ = state_float_slice_indices(cfg)
        sf = []
        for _ in range(n):
            v = rng.standard_normal(fd, dtype=np.float32)
            v[w0s:w0e] = [1.0, 4.0, 0.0]
            sf.append(v)
        meters = np.cumsum(np.abs(rng.standard_normal(n, dtype=np.float32)) * 0.2)
        actions = list(rng.integers(0, 8, size=n))
        n_ab = cfg.n_actions_per_block
        ms = cfg.ms_per_block if n_ab > 1 else cfg.ms_per_action
        st = torch.tensor(np.stack(sf), dtype=torch.float32)
        mt = torch.tensor(meters, dtype=torch.float32)
        act = (
            torch.from_numpy(np.stack(actions).astype(np.int64))
            if n_ab > 1
            else torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        )
        ri, _ = compute_rewards_into_and_potentials(
            st, mt, act, cfg, n, False, 0.0, ms, 0.0, 0.0, 0.0, 0.0
        )
        dense, _ = compute_dense_reward_per_action_t(
            st, mt, act, cfg, n, False, 0.0, ms, 0.0, 0.0, 0.0, 0.0
        )
        self.assertTrue(
            torch.allclose(dense.sum(), ri.sum(), rtol=1e-4, atol=1e-4),
            msg=f"{float(dense.sum())} vs {float(ri.sum())}",
        )


class TestPPORolloutRewardsMatchVectorized(unittest.TestCase):
    """PPO rollout rewards == reward_vectorized + same potential fold as IQN (n=1)."""

    def test_ppo_matches_compute_rewards_into_and_fold(self):
        self.assertTrue(_RL_PPO_CFG.is_file())
        cfg = load_config(_RL_PPO_CFG)
        n = 9
        fd = cfg.float_input_dim
        rng = np.random.default_rng(7)
        w0s, w0e, *_ = state_float_slice_indices(cfg)
        sf = []
        for _ in range(n):
            v = rng.standard_normal(fd, dtype=np.float32)
            v[w0s:w0e] = [1.0, 4.0, 0.0]
            sf.append(v)
        meters = np.cumsum(np.abs(rng.standard_normal(n, dtype=np.float32)) * 0.2)
        actions = list(rng.integers(0, 8, size=n))
        roll = {
            "actions": actions,
            "ppo_log_probs": [0.0] * n,
            "ppo_values": [0.0] * n,
            "frames": [np.zeros((1, 4, 4), np.float32)] * n,
            "state_float": sf,
            "meters_advanced_along_centerline": meters.tolist(),
        }
        n_ab = cfg.n_actions_per_block
        ms = cfg.ms_per_block if n_ab > 1 else cfg.ms_per_action
        st = torch.tensor(np.stack(sf), dtype=torch.float32)
        mt = torch.tensor(meters, dtype=torch.float32)
        if n_ab > 1:
            act = torch.from_numpy(np.stack(roll["actions"]).astype(np.int64))
        else:
            act = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        g = 0.99
        dense, pot = compute_dense_reward_per_action_t(
            st, mt, act, cfg, n, False, 0.0, ms, 0.0, 0.0, 0.0, 0.0
        )
        want = _fold_potential_into_ppo_step_rewards(dense.detach().numpy(), pot.detach().numpy(), g)
        got, dones = ppo_rewards_and_dones_from_rollout(
            roll,
            cfg,
            gamma=g,
            engineered_speedslide_reward=0.0,
            engineered_neoslide_reward=0.0,
            engineered_kamikaze_reward=0.0,
            engineered_close_to_vcp_reward=0.0,
        )
        self.assertTrue(np.allclose(want, got, rtol=1e-5, atol=1e-5))
        self.assertEqual(dones.shape, (n,))
        self.assertEqual(float(dones[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
