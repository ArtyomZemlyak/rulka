"""Load every RL YAML; nn vs legacy; BTR→vision merge; BC vs RL CNN kwargs; IQN/PPO wiring on CUDA."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RL_DIR = _REPO_ROOT / "config_files" / "rl"


def _rl_yaml_files() -> list[Path]:
    return sorted({p.resolve() for p in _RL_DIR.glob("config_*.yaml")})


class TestAllRlYamlLoad(unittest.TestCase):
    def test_each_config_file_loads(self):
        from config_files.config_loader import load_config

        for path in _rl_yaml_files():
            with self.subTest(msg=path.name):
                cfg = load_config(path)
                self.assertGreater(cfg.w_downsized, 0)
                self.assertGreater(cfg.h_downsized, 0)
                self.assertIn(cfg.algorithm, ("iqn", "ppo", "dpo", "grpo"))
                self.assertTrue(hasattr(cfg, "vis"), "ConfigView must expose nn.vis")
                v = cfg.vis
                self.assertTrue(v.no_image or v.cnn is not None or v.transformer is not None)

    def test_config_test_yaml_is_float_only_vision(self):
        from config_files.config_loader import load_config, set_config, get_config

        p = _RL_DIR / "config_test.yaml"
        self.assertTrue(p.is_file(), msg=f"missing {p}")
        set_config(load_config(p))
        cfg = get_config()
        self.assertTrue(cfg.vis.no_image)
        self.assertFalse(cfg.use_iqn_image_head)

    def test_bc_vision_cnn_kwargs_match_nn_vision(self):
        """BC full-IQN must use the same CNN dict as RL (nn.vis.cnn), not flat btr."""
        from config_files.config_loader import load_config, set_config, get_config
        from trackmania_rl.nn_build.vis_cnn_head import vis_cnn_head_kw_from_nn_vis

        for path in _rl_yaml_files():
            with self.subTest(msg=path.name):
                set_config(load_config(path))
                cfg = get_config()
                got = vis_cnn_head_kw_from_nn_vis(cfg.vis)
                v = cfg.vis
                if v.no_image or v.cnn is None:
                    self.assertFalse(got["use_impala_cnn"])
                    self.assertEqual(got["impala_model_size"], 2)
                    self.assertFalse(got["use_adaptive_maxpool"])
                    self.assertEqual(got["adaptive_maxpool_size"], 6)
                    self.assertFalse(got["use_spectral_norm"])
                else:
                    self.assertEqual(got["use_impala_cnn"], v.cnn.use_impala_cnn)
                    self.assertEqual(got["impala_model_size"], v.cnn.impala_model_size)
                    self.assertEqual(got["use_adaptive_maxpool"], v.cnn.use_adaptive_maxpool)
                    self.assertEqual(got["adaptive_maxpool_size"], v.cnn.adaptive_maxpool_size)
                    self.assertEqual(got["use_spectral_norm"], v.cnn.use_spectral_norm)


class TestMergeBtrIntoNnVision(unittest.TestCase):
    def test_fills_omitted_cnn_fields_from_btr(self):
        from config_files.config_loader import _merge_btr_cnn_into_vis
        from config_files.nn_schema import NnConfig

        nn_dict = NnConfig.model_validate({}).model_dump()
        nn_dict["vis"] = {"cnn": {}}
        btr = {
            "use_impala_cnn": True,
            "impala_model_size": 3,
            "use_adaptive_maxpool": True,
            "adaptive_maxpool_size": 7,
            "use_spectral_norm": True,
        }
        _merge_btr_cnn_into_vis(nn_dict, btr)
        n = NnConfig.model_validate(nn_dict)
        self.assertIsNotNone(n.vis.cnn)
        self.assertTrue(n.vis.cnn.use_impala_cnn)
        self.assertEqual(n.vis.cnn.impala_model_size, 3)
        self.assertTrue(n.vis.cnn.use_adaptive_maxpool)
        self.assertEqual(n.vis.cnn.adaptive_maxpool_size, 7)
        self.assertTrue(n.vis.cnn.use_spectral_norm)

    def test_does_not_override_explicit_vision_fields(self):
        from config_files.config_loader import _merge_btr_cnn_into_vis
        from config_files.nn_schema import NnConfig

        nn_dict = NnConfig.model_validate({}).model_dump()
        nn_dict["vis"] = {"cnn": {"use_impala_cnn": False, "impala_model_size": 2}}
        _merge_btr_cnn_into_vis(
            nn_dict,
            {"use_impala_cnn": True, "impala_model_size": 99},
        )
        n = NnConfig.model_validate(nn_dict)
        self.assertFalse(n.vis.cnn.use_impala_cnn)
        self.assertEqual(n.vis.cnn.impala_model_size, 2)

    def test_skips_non_cnn_vision(self):
        from config_files.config_loader import _merge_btr_cnn_into_vis
        from config_files.nn_schema import NnConfig

        nn_dict = NnConfig.model_validate({}).model_dump()
        nn_dict["vis"] = {"no_image": True}
        before = dict(nn_dict["vis"])
        _merge_btr_cnn_into_vis(nn_dict, {"use_impala_cnn": True})
        self.assertEqual(nn_dict["vis"], before)
        n = NnConfig.model_validate(nn_dict)
        self.assertTrue(n.vis.no_image)


@unittest.skipUnless(torch.cuda.is_available(), "IQN/PPO factories move modules to CUDA")
class TestWiringBuildsForEveryRlYaml(unittest.TestCase):
    def test_make_network_for_algorithm(self):
        from config_files.config_loader import load_config, set_config, get_config
        from trackmania_rl.agents.algorithms.registry import get_wiring

        for path in _rl_yaml_files():
            with self.subTest(msg=path.name):
                set_config(load_config(path))
                cfg = get_config()
                w = get_wiring(cfg.algorithm)
                net, _ = w.make_network(False, False)
                self.assertGreater(len(net.state_dict()), 0)


if __name__ == "__main__":
    unittest.main()
