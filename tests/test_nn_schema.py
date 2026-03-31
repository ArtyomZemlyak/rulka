"""NnConfig validation and IQN decoder rules."""

import unittest

from config_files.nn_schema import (
    IqnDecoderConfig,
    MultimodalTransformersConfig,
    NnConfig,
    TransformersConfig,
    infer_fusion_encoder,
)


class TestNnSchema(unittest.TestCase):
    def test_empty_nn_defaults(self):
        n = NnConfig.model_validate({})
        self.assertEqual(n.image_size.width, 256)
        self.assertFalse(n.vis.no_image)
        self.assertFalse(n.vis.freeze)
        self.assertIsNotNone(n.vis.cnn)
        self.assertEqual(n.float_branch.mlp.hidden_dim, 256)
        self.assertFalse(n.float_branch.freeze)
        self.assertFalse(n.encoder.freeze)
        self.assertFalse(n.iqn.freeze)
        self.assertFalse(n.decoder.advantage.freeze)
        self.assertFalse(n.decoder.value.freeze)
        self.assertFalse(n.decoder.shared_trunk_freeze)
        self.assertIsNotNone(n.decoder.advantage.mlp)
        self.assertEqual(n.decoder.advantage.mlp.n_hidden_layers, 1)

    def test_decoder_mlp_yaml_aliases(self):
        d = IqnDecoderConfig.model_validate(
            {
                "shared_input": "post_tau",
                "dense_hidden_dimension": 512,
                "advantage": {"mlp": {"hidden": 200, "layers": 2}},
                "value": {"mlp": {"hidden_dim": 200, "n_hidden_layers": 2}},
            }
        )
        self.assertEqual(d.advantage.mlp.hidden_dim, 200)
        self.assertEqual(d.advantage.mlp.n_hidden_layers, 2)
        self.assertEqual(d.value.mlp.hidden_dim, 200)
        self.assertEqual(d.value.mlp.n_hidden_layers, 2)

    def test_transformer_head_requires_post_tau(self):
        with self.assertRaises(ValueError):
            IqnDecoderConfig.model_validate(
                {
                    "shared_input": "pre_tau",
                    "dense_hidden_dimension": 256,
                    "advantage": {"transformer": {"d_model": 64, "n_layers": 1, "n_heads": 4}},
                    "value": {"mlp": {"layers": 1}},
                }
            )

    def test_transformer_heads_post_tau_ok(self):
        IqnDecoderConfig.model_validate(
            {
                "shared_input": "post_tau",
                "dense_hidden_dimension": 256,
                "advantage": {"transformer": {"d_model": 64, "n_layers": 1, "n_heads": 4}},
                "value": {"mlp": {"layers": 1}},
            }
        )

    def test_transformer_hf_flag_rejected_for_now(self):
        with self.assertRaises(ValueError):
            IqnDecoderConfig.model_validate(
                {
                    "shared_input": "post_tau",
                    "dense_hidden_dimension": 256,
                    "advantage": {"transformer": {"d_model": 64, "n_layers": 1, "n_heads": 4, "use_hf_backbone": True}},
                    "value": {"mlp": {"layers": 1}},
                }
            )

    def test_legacy_transformer_encoder_yaml_key(self):
        d = IqnDecoderConfig.model_validate(
            {
                "shared_input": "post_tau",
                "dense_hidden_dimension": 256,
                "advantage": {"transformer_encoder": {"d_model": 64, "n_layers": 1, "n_heads": 4}},
                "value": {"mlp": {"layers": 1}},
            }
        )
        self.assertIsNotNone(d.advantage.transformer)
        self.assertEqual(d.advantage.transformer.d_model, 64)

    def test_infer_fusion_encoder_hf_from_encoder_transformer(self):
        tr = MultimodalTransformersConfig(
            fusion_mode="post_concat",
            transformer=TransformersConfig(
                use_hf_backbone=True,
                model_name_or_path="google/bert_uncased_L-2_H-128_A-2",
                d_model=128,
                n_layers=0,
                n_heads=4,
                post_concat_seq_len=4,
            ),
        )
        self.assertEqual(infer_fusion_encoder("post_concat", tr), "hf_embedding")

    def test_encoder_use_hf_backbone_requires_model_id(self):
        with self.assertRaises(ValueError):
            NnConfig.model_validate(
                {
                    "fusion_mode": "post_concat",
                    "encoder": {"transformer": {"use_hf_backbone": True, "model_name_or_path": ""}},
                }
            )

    def test_encoder_hf_backbone_conflicts_native_fusion_explicit(self):
        with self.assertRaises(ValueError):
            NnConfig.model_validate(
                {
                    "fusion_mode": "post_concat",
                    "encoder": {
                        "fusion_encoder": "native_transformer",
                        "transformer": {
                            "use_hf_backbone": True,
                            "model_name_or_path": "google/bert_uncased_L-2_H-128_A-2",
                            "d_model": 128,
                            "n_layers": 2,
                            "n_heads": 4,
                            "post_concat_seq_len": 4,
                        },
                    },
                }
            )

    def test_post_concat_patch_tokens_requires_token_sequence_layout(self):
        with self.assertRaises(ValueError):
            NnConfig.model_validate(
                {
                    "fusion_mode": "post_concat",
                    "encoder": {
                        "post_concat_layout": "fused_vector",
                        "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 0, "post_concat_seq_len": 4},
                    },
                    "vis": {
                        "cnn": None,
                        "transformer": {
                            "d_model": 64,
                            "n_heads": 4,
                            "n_layers": 0,
                            "patch_size": 8,
                            "fusion_tokens": "patch_tokens",
                        },
                    },
                }
            )

    def test_per_feature_float_requires_token_sequence(self):
        with self.assertRaises(ValueError):
            NnConfig.model_validate(
                {
                    "fusion_mode": "post_concat",
                    "encoder": {
                        "post_concat_layout": "fused_vector",
                        "float_token_layout": "per_feature",
                        "float_token_input": "raw",
                        "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 0, "post_concat_seq_len": 4},
                    },
                }
            )

    def test_per_feature_float_requires_raw_input(self):
        with self.assertRaises(ValueError):
            NnConfig.model_validate(
                {
                    "fusion_mode": "post_concat",
                    "encoder": {
                        "post_concat_layout": "token_sequence",
                        "float_token_layout": "per_feature",
                        "float_token_input": "mlp_hidden",
                        "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 0, "post_concat_seq_len": 4},
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
