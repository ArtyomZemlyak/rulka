"""soft_copy_param must honor freeze prefixes with torch.compile-style state_dict keys."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from trackmania_rl import utilities


class TestSoftCopyRespectsCompilePrefixedKeys(unittest.TestCase):
    def test_skips_frozen_prefix_under_orig_mod(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.img_head = nn.Linear(2, 2, bias=True)
                self.trunk = nn.Linear(2, 2, bias=True)

        a = M()
        b = M()
        w_img_before = a.img_head.weight.detach().clone()
        w_trunk_before = a.trunk.weight.detach().clone()

        # Fake compile-prefixed state dict (same layout as ``torch.compile`` checkpoints).
        def fake_sd(m: M) -> dict:
            d = m.state_dict()
            return {"_orig_mod." + k: v.clone() for k, v in d.items()}

        target = fake_sd(a)
        source = fake_sd(b)

        class Wrap(nn.Module):
            def __init__(self, sd: dict):
                super().__init__()
                self._sd = sd

            def state_dict(self, *args, **kwargs):  # noqa: ARG002
                return self._sd

        utilities.soft_copy_param(Wrap(target), Wrap(source), tau=1.0, skip_key_prefixes=["img_head."])

        # trunk updated (tau=1 → copy source), img_head skipped
        self.assertTrue(torch.allclose(target["_orig_mod.trunk.weight"], source["_orig_mod.trunk.weight"]))
        self.assertTrue(torch.allclose(target["_orig_mod.img_head.weight"], w_img_before))
        self.assertFalse(torch.allclose(target["_orig_mod.trunk.weight"], w_trunk_before))


if __name__ == "__main__":
    unittest.main()
