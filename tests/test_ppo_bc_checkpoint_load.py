"""BC PPO checkpoint → same RL policy: strict load after key preparation."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
_RL_PPO = _REPO / "config_files" / "rl" / "config_ppo.yaml"


class TestPpoBcCheckpointLoad(unittest.TestCase):
    def test_single_offset_bc_matches_base_policy_strict_load(self):
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.agents.policy_models.ppo_actor_critic import build_ppo_actor_critic_uncompiled
        from trackmania_rl.pretrain.rl_policy_factory import build_rl_policy_for_bc
        from trackmania_rl.utilities import prepare_ppo_policy_state_dict_for_load

        self.assertTrue(_RL_PPO.is_file(), msg=f"missing {_RL_PPO}")
        set_config(load_config(_RL_PPO))
        self.assertEqual(get_config().algorithm, "ppo")

        bc = build_rl_policy_for_bc(n_bc_offsets=1, bc_multi_offset_mode="separate_heads")
        rl = build_ppo_actor_critic_uncompiled(get_config())
        sd = bc.state_dict()
        prep = prepare_ppo_policy_state_dict_for_load(sd, rl)
        rl.load_state_dict(prep, strict=True)

    def test_multi_offset_bc_merges_bc_heads_into_policy_head(self):
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.agents.policy_models.ppo_actor_critic import build_ppo_actor_critic_uncompiled
        from trackmania_rl.pretrain.rl_policy_factory import build_rl_policy_for_bc
        from trackmania_rl.utilities import prepare_ppo_policy_state_dict_for_load

        set_config(load_config(_RL_PPO))
        wrap = build_rl_policy_for_bc(n_bc_offsets=3, bc_multi_offset_mode="separate_heads")
        sd = wrap.state_dict()
        self.assertTrue(any(k.startswith("bc_heads.") for k in sd))

        exp_w = torch.stack([wrap.bc_heads[i].weight.detach() for i in range(3)]).mean(dim=0)
        exp_b = torch.stack([wrap.bc_heads[i].bias.detach() for i in range(3)]).mean(dim=0)

        base = build_ppo_actor_critic_uncompiled(get_config())
        prep = prepare_ppo_policy_state_dict_for_load(sd, base)
        self.assertFalse(any(k.startswith("bc_heads.") for k in prep))
        self.assertIn("policy_head.weight", prep)
        base.load_state_dict(prep, strict=True)
        self.assertTrue(torch.allclose(base.policy_head.weight, exp_w))
        self.assertTrue(torch.allclose(base.policy_head.bias, exp_b))
        self.assertTrue(torch.allclose(base.trunk[0].weight, wrap.base.trunk[0].weight))

    def test_roundtrip_save_like_weights1_then_prepare(self):
        from config_files.config_loader import get_config, load_config, set_config
        from trackmania_rl.agents.policy_models.ppo_actor_critic import build_ppo_actor_critic_uncompiled
        from trackmania_rl.pretrain.rl_policy_factory import build_rl_policy_for_bc
        from trackmania_rl.utilities import prepare_ppo_policy_state_dict_for_load

        set_config(load_config(_RL_PPO))
        bc = build_rl_policy_for_bc(n_bc_offsets=1, bc_multi_offset_mode="separate_heads")
        import io

        buf = io.BytesIO()
        torch.save(bc.state_dict(), buf)
        buf.seek(0)
        loaded = torch.load(buf, map_location="cpu", weights_only=True)
        rl = build_ppo_actor_critic_uncompiled(get_config())
        rl.load_state_dict(prepare_ppo_policy_state_dict_for_load(loaded, rl), strict=True)

    def test_slice_policy_head_truncates_larger_checkpoint(self):
        from trackmania_rl.utilities import prepare_ppo_policy_state_dict_for_load

        class _PolicyStub(nn.Module):
            def __init__(self, n_pi: int) -> None:
                super().__init__()
                self.policy_head = nn.Linear(16, n_pi)
                self.value_head = nn.Linear(16, 1)

        big = _PolicyStub(12)
        small = _PolicyStub(6)
        torch.manual_seed(0)
        with torch.no_grad():
            nn.init.normal_(big.policy_head.weight, 0.0, 0.1)
            nn.init.normal_(big.policy_head.bias, 0.0, 0.1)
        prep = prepare_ppo_policy_state_dict_for_load(
            big.state_dict(), small, slice_policy_head_to_model=True
        )
        small.load_state_dict(prep, strict=True)
        self.assertTrue(torch.allclose(small.policy_head.weight, big.policy_head.weight[:6]))
        self.assertTrue(torch.allclose(small.policy_head.bias, big.policy_head.bias[:6]))

    def test_slice_policy_head_disabled_keeps_mismatch_for_strict_load(self):
        from trackmania_rl.utilities import prepare_ppo_policy_state_dict_for_load

        class _PolicyStub(nn.Module):
            def __init__(self, n_pi: int) -> None:
                super().__init__()
                self.policy_head = nn.Linear(8, n_pi)
                self.value_head = nn.Linear(8, 1)

        big = _PolicyStub(12)
        small = _PolicyStub(6)
        prep = prepare_ppo_policy_state_dict_for_load(
            big.state_dict(), small, slice_policy_head_to_model=False
        )
        with self.assertRaises(RuntimeError):
            small.load_state_dict(prep, strict=True)


if __name__ == "__main__":
    unittest.main()
