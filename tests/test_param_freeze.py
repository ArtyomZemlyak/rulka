"""Unit tests for ``trackmania_rl.param_freeze`` (``nn.*.freeze`` → prefixes)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from torch import nn

from trackmania_rl.param_freeze import (
    apply_frozen_prefixes,
    collect_frozen_prefixes,
    param_name_matches_any_prefix,
    prefixes_that_match_module,
    strip_compile_wrapper_from_param_name,
)


class TestParamFreezeRules(unittest.TestCase):
    def test_strip_compile_wrapper(self):
        self.assertEqual(strip_compile_wrapper_from_param_name("a.weight"), "a.weight")
        self.assertEqual(strip_compile_wrapper_from_param_name("_orig_mod.a.weight"), "a.weight")

    def test_prefix_with_trailing_dot(self):
        self.assertTrue(param_name_matches_any_prefix("img_head.0.weight", ("img_head.",)))
        self.assertFalse(param_name_matches_any_prefix("img_head_extra.0.weight", ("img_head.",)))

    def test_prefix_without_dot(self):
        self.assertTrue(param_name_matches_any_prefix("pos_vis", ("pos_vis",)))
        self.assertTrue(param_name_matches_any_prefix("pos_vis.foo", ("pos_vis",)))
        self.assertFalse(param_name_matches_any_prefix("pos_visible", ("pos_vis",)))

    def test_compile_wrapped_name(self):
        self.assertTrue(
            param_name_matches_any_prefix("_orig_mod.img_head.0.weight", ("img_head.",))
        )

    def test_apply_frozen_prefixes(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.img_head = nn.Linear(2, 2)
                self.trunk = nn.Linear(2, 2)

        m = M()
        n = apply_frozen_prefixes(m, ["img_head."])
        self.assertEqual(n, 2)
        self.assertFalse(m.img_head.weight.requires_grad)
        self.assertTrue(m.trunk.weight.requires_grad)

    def test_prefixes_that_match_module_filters_unused(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Linear(2, 2)

        m = M()
        wanted = ["patch_embed.", "cnn_to_vis.", "img_head."]
        self.assertEqual(prefixes_that_match_module(m, wanted), ["patch_embed."])


def _slot(freeze: bool = False) -> SimpleNamespace:
    return SimpleNamespace(freeze=freeze)


def _decoder(
    *,
    adv_f: bool = False,
    val_f: bool = False,
    trunk_f: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        advantage=_slot(adv_f),
        value=_slot(val_f),
        shared_trunk_freeze=trunk_f,
    )


def _cfg(
    *,
    algo_vis: bool = False,
    no_image: bool = False,
    float_f: bool = False,
    enc_f: bool = False,
    iqn_f: bool = False,
    decoder: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        vis=SimpleNamespace(freeze=algo_vis, no_image=no_image),
        float_branch=SimpleNamespace(freeze=float_f),
        encoder=SimpleNamespace(freeze=enc_f),
        iqn=SimpleNamespace(freeze=iqn_f),
        decoder=decoder or _decoder(),
    )


class TestCollectFromNnFreeze(unittest.TestCase):
    def test_iqn_vis_img_head_only(self):
        cfg = _cfg(algo_vis=True)
        p = collect_frozen_prefixes(cfg, wiring_algorithm="iqn")
        self.assertEqual([x for x in p if x.startswith("img")], ["img_head."])

    def test_ppo_vis_many_prefixes(self):
        cfg = _cfg(algo_vis=True)
        p = collect_frozen_prefixes(cfg, wiring_algorithm="ppo")
        self.assertIn("img_head.", p)
        self.assertIn("backbone.", p)
        self.assertIn("enc_vis.", p)

    def test_vis_ignored_when_no_image(self):
        cfg = _cfg(algo_vis=True, no_image=True)
        p = collect_frozen_prefixes(cfg, wiring_algorithm="iqn")
        self.assertNotIn("img_head.", p)

    def test_float_and_encoder_ppo(self):
        cfg = _cfg(float_f=True, enc_f=True)
        p = collect_frozen_prefixes(cfg, wiring_algorithm="ppo")
        self.assertIn("float_feature_extractor.", p)
        self.assertIn("float_to_hidden.", p)
        self.assertIn("bridge.", p)
        self.assertIn("enc_fusion_native.", p)

    def test_iqn_iqn_fc_and_heads(self):
        cfg = _cfg(
            iqn_f=True,
            decoder=_decoder(adv_f=True, val_f=True),
        )
        p = collect_frozen_prefixes(cfg, wiring_algorithm="iqn")
        self.assertIn("iqn_fc.", p)
        self.assertIn("A_head.", p)
        self.assertIn("A_head_multi.", p)
        self.assertIn("V_head.", p)

    def test_ppo_policy_value_trunk(self):
        cfg = _cfg(decoder=_decoder(adv_f=True, val_f=True, trunk_f=True))
        p = collect_frozen_prefixes(cfg, wiring_algorithm="ppo")
        self.assertIn("policy_head.", p)
        self.assertIn("value_head.", p)
        self.assertIn("trunk.", p)


if __name__ == "__main__":
    unittest.main()
