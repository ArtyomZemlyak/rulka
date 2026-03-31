"""``TransformersConfig`` / ``MultimodalTransformersConfig`` schema and fusion hub roundtrip."""

import unittest

from config_files.config_schema import NeuralNetworkConfig
from config_files.nn_schema import MultimodalTransformersConfig, NnConfig, NnEncoderConfig, TransformersConfig


class TestTransformersConfigSchema(unittest.TestCase):
    def test_defaults(self):
        n = NeuralNetworkConfig.model_validate({})
        self.assertEqual(n.transformers.fusion_mode, "none")
        self.assertIsNone(n.vis.transformer)
        self.assertEqual(n.transformers.init_from_pretrained, "")

    def test_d_model_divisible_by_heads(self):
        with self.assertRaises(ValueError):
            TransformersConfig.model_validate({"d_model": 100, "n_heads": 8})

    def test_unified_requires_matching_d_model(self):
        with self.assertRaises(ValueError):
            NnConfig.model_validate(
                {
                    "fusion_mode": "unified",
                    "vis": {
                        "cnn": None,
                        "transformer": {"d_model": 128, "n_heads": 4, "n_layers": 1, "ff_mult": 4, "patch_size": 8},
                    },
                    "encoder": {
                        "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 1, "ff_mult": 4, "patch_size": 8},
                    },
                }
            )

    def test_legacy_fusion_keys_under_encoder_hoisted(self):
        n = NnConfig.model_validate(
            {
                "encoder": {
                    "fusion_mode": "vision_transformer",
                    "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 1, "ff_mult": 4, "patch_size": 8},
                },
                "vis": {"cnn": None, "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 1, "ff_mult": 4, "patch_size": 8}},
            }
        )
        self.assertEqual(n.fusion_mode, "vision_transformer")
        self.assertEqual(n.transformers.fusion_mode, "vision_transformer")


class TestRulkaFusionHubRoundtrip(unittest.TestCase):
    def test_wrap_save_roundtrip_cpu(self):
        import importlib.util
        import tempfile

        if importlib.util.find_spec("transformers") is None:
            self.skipTest("transformers not installed")

        import numpy as np
        import torch

        from types import SimpleNamespace

        from trackmania_rl.agents.policy_models.multimodal_torch_fusion import build_multimodal_fusion_from_transformers
        from trackmania_rl.agents.policy_models.rulka_multimodal_fusion_hub import (
            load_fusion_policy_weights_from_hub,
            wrap_fusion_policy_for_hf_save,
        )

        d_in = 12
        vis_enc = TransformersConfig.model_validate(
            {
                "use_hf_backbone": False,
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "patch_size": 16,
            }
        )
        t = MultimodalTransformersConfig.model_validate(
            {
                "fusion_mode": "vision_transformer",
                "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 2, "patch_size": 16},
            }
        )
        mean = np.zeros(d_in, dtype=np.float32)
        std = np.ones(d_in, dtype=np.float32)
        pol = build_multimodal_fusion_from_transformers(
            t,
            vis_enc,
            float_inputs_dim=d_in,
            float_hidden_dim=32,
            dense_hidden_dim=64,
            image_h=64,
            image_w=64,
            use_image_head=True,
            vis_branch="native_transformer",
            float_inputs_mean=mean,
            float_inputs_std=std,
            n_actions=3,
            n_actions_per_block=1,
        )
        pol2 = build_multimodal_fusion_from_transformers(
            t,
            vis_enc,
            float_inputs_dim=d_in,
            float_hidden_dim=32,
            dense_hidden_dim=64,
            image_h=64,
            image_w=64,
            use_image_head=True,
            vis_branch="native_transformer",
            float_inputs_mean=mean,
            float_inputs_std=std,
            n_actions=3,
            n_actions_per_block=1,
        )

        enc_cfg = NnEncoderConfig(transformer=t.transformer)

        class Cfg:
            fusion_mode = t.fusion_mode
            init_from_pretrained = getattr(t, "init_from_pretrained", "") or ""
            encoder = enc_cfg
            vis = SimpleNamespace(transformer=vis_enc)

            def to_multimodal(self):
                return MultimodalTransformersConfig(
                    fusion_mode=self.fusion_mode,
                    transformer=self.encoder.transformer,
                    init_from_pretrained=self.init_from_pretrained,
                    fusion_encoder=self.encoder.fusion_encoder,
                    fusion_mlp=self.encoder.fusion_mlp,
                    fusion_cnn=self.encoder.fusion_cnn,
                    hf_embedding=self.encoder.hf_embedding,
                    post_concat_layout=self.encoder.post_concat_layout,
                    float_token_input=self.encoder.float_token_input,
                    float_token_layout=self.encoder.float_token_layout,
                )
            float_input_dim = d_in
            float_hidden_dim = 32
            dense_hidden_dimension = 64
            H_downsized = 64
            W_downsized = 64
            use_iqn_image_head = True
            inputs = [0, 0, 0]
            n_actions_per_block = 1
            float_inputs_mean = mean
            float_inputs_std = std

            def float_hidden_dim_effective(self):
                return int(self.float_hidden_dim)

        with tempfile.TemporaryDirectory() as td:
            w = wrap_fusion_policy_for_hf_save(pol, Cfg())
            w.save_pretrained(td)
            load_fusion_policy_weights_from_hub(pol2, td, trust_remote_code=True)
        x = torch.randn(1, 1, 64, 64)
        fl = torch.randn(1, d_in)
        with torch.no_grad():
            o1 = pol(x, fl)
            o2 = pol2(x, fl)
        self.assertTrue(torch.allclose(o1.value, o2.value, atol=1e-5))
        self.assertTrue(torch.allclose(o1.logits, o2.logits, atol=1e-4))

    def test_post_concat_token_sequence_native_forward(self):
        import numpy as np
        import torch

        from trackmania_rl.agents.policy_models.multimodal_torch_fusion import build_multimodal_fusion_from_transformers

        d_in = 8
        vis_enc = TransformersConfig.model_validate(
            {
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 0,
                "patch_size": 16,
                "fusion_tokens": "patch_tokens",
            }
        )
        enc_tr = TransformersConfig.model_validate(
            {
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 1,
                "ff_mult": 2,
                "unified_float_tokens": 2,
                "post_concat_seq_len": 4,
            }
        )
        t = MultimodalTransformersConfig.model_validate(
            {
                "fusion_mode": "post_concat",
                "transformer": enc_tr.model_dump(),
                "post_concat_layout": "token_sequence",
                "float_token_input": "raw",
            }
        )
        mean = np.zeros(d_in, dtype=np.float32)
        std = np.ones(d_in, dtype=np.float32)
        pol = build_multimodal_fusion_from_transformers(
            t,
            vis_enc,
            float_inputs_dim=d_in,
            float_hidden_dim=16,
            dense_hidden_dim=64,
            image_h=64,
            image_w=64,
            use_image_head=True,
            vis_branch="native_transformer",
            float_inputs_mean=mean,
            float_inputs_std=std,
            n_actions=2,
            n_actions_per_block=1,
        )
        pol.eval()
        with torch.no_grad():
            out = pol(torch.randn(1, 1, 64, 64), torch.randn(1, d_in))
        self.assertEqual(tuple(out.logits.shape), (1, 2))

    def test_post_concat_per_feature_float_forward(self):
        import numpy as np
        import torch

        from trackmania_rl.agents.policy_models.multimodal_torch_fusion import build_multimodal_fusion_from_transformers

        d_in = 5
        vis_enc = TransformersConfig.model_validate(
            {
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 0,
                "patch_size": 16,
                "fusion_tokens": "patch_tokens",
            }
        )
        enc_tr = TransformersConfig.model_validate(
            {
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 1,
                "ff_mult": 2,
                "unified_float_tokens": 1,
                "post_concat_seq_len": 4,
            }
        )
        t = MultimodalTransformersConfig.model_validate(
            {
                "fusion_mode": "post_concat",
                "transformer": enc_tr.model_dump(),
                "post_concat_layout": "token_sequence",
                "float_token_input": "raw",
                "float_token_layout": "per_feature",
            }
        )
        mean = np.zeros(d_in, dtype=np.float32)
        std = np.ones(d_in, dtype=np.float32)
        pol = build_multimodal_fusion_from_transformers(
            t,
            vis_enc,
            float_inputs_dim=d_in,
            float_hidden_dim=16,
            dense_hidden_dim=64,
            image_h=64,
            image_w=64,
            use_image_head=True,
            vis_branch="native_transformer",
            float_inputs_mean=mean,
            float_inputs_std=std,
            n_actions=2,
            n_actions_per_block=1,
        )
        pol.eval()
        with torch.no_grad():
            out = pol(torch.randn(1, 1, 64, 64), torch.randn(1, d_in))
        self.assertEqual(tuple(out.logits.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
