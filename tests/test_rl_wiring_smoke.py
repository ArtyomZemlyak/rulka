"""Smoke: RL config + IQN wiring (stdlib unittest; no pytest dependency)."""

from pathlib import Path
import unittest

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RL_DEFAULT = _REPO_ROOT / "config_files" / "rl" / "config_default.yaml"


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


if __name__ == "__main__":
    unittest.main()
