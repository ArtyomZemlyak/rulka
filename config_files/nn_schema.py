"""
Hierarchical neural network configuration (YAML key ``nn``).

Each subtree uses **named branches** (what you declare is what gets built).
Optional ``freeze: true`` on ``vis``, ``float``, ``encoder``, ``iqn``, ``decoder`` slots
(and ``decoder.shared_trunk_freeze`` for PPO) maps to RL parameter freeze — see
``trackmania_rl.param_freeze``::

    nn:
      vis:
        image_size: {width, height}
        no_image: false
        freeze: false
        cnn: {...}              # XOR ``transformer``
        transformer: {...}
      float:
        mlp: {hidden_dim}
        freeze: false
      fusion_mode: none          # global (multimodal fusion / HF routing); not under ``encoder``
      init_from_pretrained: ""
      encoder:
        mlp: {hidden_dim}       # optional; else ``float.mlp`` width is used
        transformer: {...}     # multimodal fusion (PPO; IQN too when ``fusion_mode != none``)
      decoder:
        advantage:            # XOR: ``mlp`` | ``transformer``
          mlp:
            hidden_dim: ...  # alias: ``hidden``; default = dense_hidden_dimension // 2
            layers: 1        # hidden depth (alias: ``n_hidden_layers``)
          # transformer:
          #   d_model: 128
          #   n_layers: 1
        value: ...
      training: {...}           # optimizer / target-network knobs (``nn.training`` in YAML)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

MultimodalFusionModeLiteral = Literal["none", "vision_transformer", "post_concat", "unified"]
VisBranchLiteral = Literal["none", "cnn", "native_transformer", "hf_transformer"]
FusionEncoderKindLiteral = Literal["linear", "native_transformer", "mlp", "cnn", "hf_embedding"]
PostConcatLayoutLiteral = Literal["fused_vector", "token_sequence"]
FloatTokenInputLiteral = Literal["raw", "mlp_hidden"]
FloatTokenLayoutLiteral = Literal["dense", "per_feature"]
VisFusionTokensLiteral = Literal["summary", "patch_tokens"]


def infer_vis_branch(vis: Any) -> VisBranchLiteral:
    """How multimodal fusion resolves the image stem from ``nn.vis`` (any ``fusion_mode``)."""
    if getattr(vis, "no_image", False):
        return "none"
    tr = getattr(vis, "transformer", None)
    if tr is not None:
        if bool(getattr(tr, "use_hf_backbone", False)):
            return "hf_transformer"
        return "native_transformer"
    return "cnn"


class FusionMlpEncoderConfig(BaseModel):
    """MLP on flattened token sequence ``[B, L * fuse_d_model]`` → ``fuse_d_model``."""

    model_config = ConfigDict(extra="ignore")

    hidden_dim: int = Field(default=512, ge=32)
    n_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class FusionCnnEncoderConfig(BaseModel):
    """1D CNN over fusion sequence (channels = ``fuse_d_model``, length = ``L``)."""

    model_config = ConfigDict(extra="ignore")

    hidden_channels: list[int] = Field(default_factory=lambda: [256, 256])
    kernel_size: int = Field(default=3, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class HfEmbeddingEncoderConfig(BaseModel):
    """HF encoder fed via ``inputs_embeds`` (BERT/RoBERTa/GPT-2–style). Map any token dim → model hidden."""

    model_config = ConfigDict(extra="ignore")

    model_name_or_path: str = ""
    trust_remote_code: bool = False
    hidden_dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)


class TransformersConfig(BaseModel):
    """Transformer / HF backbone slot (image side or fusion encoder)."""

    use_hf_backbone: bool = False
    model_name_or_path: str = ""
    trust_remote_code: bool = False
    hidden_dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    d_model: int = Field(default=256, ge=32)
    n_layers: int = Field(default=2, ge=0)
    n_heads: int = Field(default=4, ge=1)
    ff_mult: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    patch_size: int = Field(default=8, ge=1)
    post_concat_seq_len: int = Field(default=4, ge=1)
    unified_float_tokens: int = Field(default=1, ge=1)
    fusion_tokens: VisFusionTokensLiteral = Field(
        default="summary",
        description="Vision output for multimodal fusion: single summary (CLS / mean / conv flat) vs full patch token sequence.",
    )

    @model_validator(mode="after")
    def _dim_heads(self) -> TransformersConfig:
        if self.n_layers > 0 and self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        return self


class MultimodalTransformersConfig(BaseModel):
    """Flat view for multimodal fusion factory and HF hub (from :meth:`NnConfig.to_multimodal`)."""

    fusion_mode: MultimodalFusionModeLiteral = "none"
    transformer: TransformersConfig = Field(default_factory=TransformersConfig)
    init_from_pretrained: str = Field(default="")
    fusion_encoder: FusionEncoderKindLiteral | None = None
    fusion_mlp: FusionMlpEncoderConfig | None = None
    fusion_cnn: FusionCnnEncoderConfig | None = None
    hf_embedding: HfEmbeddingEncoderConfig | None = None
    post_concat_layout: PostConcatLayoutLiteral = Field(
        default="fused_vector",
        description="post_concat: concat+Linear tokenization (legacy) vs explicit [vis tokens | float tokens] sequence.",
    )
    float_token_input: FloatTokenInputLiteral = Field(
        default="raw",
        description="token_sequence only: map normalized floats with Linear→K×d (raw) or after float MLP (mlp_hidden).",
    )
    float_token_layout: FloatTokenLayoutLiteral = Field(
        default="dense",
        description="token_sequence: dense Linear(float_dim→K·d) vs per_feature (one token per scalar: Linear(1,d)+slot emb).",
    )

    @model_validator(mode="before")
    @classmethod
    def _hub_dict_used_fuse_slot_key(cls, data: object) -> object:
        if isinstance(data, dict) and "fuse" in data and "transformer" not in data:
            out = dict(data)
            out["transformer"] = out.pop("fuse")
            return out
        return data


def infer_fusion_encoder(fusion_mode: str, tr: MultimodalTransformersConfig) -> FusionEncoderKindLiteral:
    """Default fusion trunk when ``fusion_encoder`` omitted in YAML / hub JSON.

    HF fusion (``inputs_embeds`` into a HF encoder) is selected when ``encoder.transformer.use_hf_backbone``
    is true — same idea as ``nn.vis.transformer.use_hf_backbone``. The internal kind name stays
    ``hf_embedding`` for the factory; the checkpoint lives on ``encoder.transformer.model_name_or_path``.
    """
    if tr.fusion_encoder is not None:
        return tr.fusion_encoder
    if tr.transformer.use_hf_backbone:
        return "hf_embedding"
    if fusion_mode == "vision_transformer":
        return "linear"
    return "native_transformer"


class FloatMlpBodyConfig(BaseModel):
    """Scalar float MLP width (no ``type`` discriminator — branch lives under ``float.mlp``)."""

    model_config = ConfigDict(extra="ignore")

    hidden_dim: int = Field(default=256, ge=1)


class FloatBranchConfig(BaseModel):
    """Named branch for float features: currently ``mlp`` only (future siblings stay orthogonal)."""

    model_config = ConfigDict(extra="ignore")

    freeze: bool = Field(default=False, description="RL: freeze float MLP (float_feature_extractor; + float_to_hidden on HF PPO).")
    mlp: FloatMlpBodyConfig | None = None

    @model_validator(mode="after")
    def _default_mlp(self) -> FloatBranchConfig:
        if self.mlp is None:
            object.__setattr__(self, "mlp", FloatMlpBodyConfig())
        return self


class MLPConfig(BaseModel):
    """Fully-connected stack: ``n_hidden_layers`` blocks of width ``hidden_dim``, then readout is built in code."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    hidden_dim: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("hidden_dim", "hidden"),
        description="Hidden width; if omitted, IQN heads use decoder.dense_hidden_dimension // 2.",
    )
    n_hidden_layers: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("n_hidden_layers", "layers"),
        serialization_alias="layers",
        description="Number of hidden Linear→(optional LayerNorm)→act blocks before the final readout layer.",
    )


class TransformerStackConfig(BaseModel):
    """Native ``TransformerEncoder`` stack (e.g. IQN decoder slot). HF backbones are not wired here yet."""

    model_config = ConfigDict(extra="ignore")

    d_model: int = Field(default=128, ge=32)
    n_layers: int = Field(default=1, ge=1)
    n_heads: int = Field(default=4, ge=1)
    ff_mult: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    use_hf_backbone: bool = False
    model_name_or_path: str = ""
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def _rules(self) -> TransformerStackConfig:
        if self.d_model % self.n_heads != 0:
            raise ValueError("transformer stack: d_model must be divisible by n_heads")
        if self.use_hf_backbone:
            raise ValueError(
                "decoder.transformer: use_hf_backbone is not supported for IQN head slots yet "
                "(native TransformerEncoder only; set use_hf_backbone: false)."
            )
        return self


class ImageSizeConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    width: int = Field(default=256, ge=1, validation_alias="w")
    height: int = Field(default=256, ge=1, validation_alias="h")


class VisCnnBodyConfig(BaseModel):
    """Convolutional vision stem (``nn.vis.cnn``).

    Runtime stem kwargs for ``trackmania_rl.agents.iqn._build_img_head`` are centralized in
    ``trackmania_rl.nn_build.vis_cnn_head`` (single source; IQN, PPO, multimodal, BC, pretrain).
    """

    model_config = ConfigDict(extra="ignore")

    use_impala_cnn: bool = False
    impala_model_size: int = 2
    use_adaptive_maxpool: bool = False
    adaptive_maxpool_size: int = 6
    use_spectral_norm: bool = False


class VisConfig(BaseModel):
    """Image branch: either ``cnn`` **or** ``transformer``; ``no_image`` for float-only."""

    model_config = ConfigDict(extra="ignore")

    no_image: bool = False
    freeze: bool = Field(default=False, description="RL: freeze vision weights (ignored when no_image).")
    image_size: ImageSizeConfig = Field(default_factory=ImageSizeConfig)
    cnn: VisCnnBodyConfig | None = None
    transformer: TransformersConfig | None = None

    @model_validator(mode="after")
    def _no_image_or_xor(self) -> VisConfig:
        if self.no_image:
            object.__setattr__(self, "cnn", None)
            object.__setattr__(self, "transformer", None)
            return self
        has_c = self.cnn is not None
        has_t = self.transformer is not None
        if has_c and has_t:
            raise ValueError("nn.vis: set only one of 'cnn' or 'transformer' for the image encoder")
        if not has_c and not has_t:
            object.__setattr__(self, "cnn", VisCnnBodyConfig())
        return self


class NnEncoderConfig(BaseModel):
    """Multimodal encoder stack: optional float MLP override + fusion trunk for PPO.

    ``fusion_encoder`` selects the trunk after early fusion tokens (or concat for ``vision_transformer``):

    - ``linear`` — single ``Linear`` from concat features to ``decoder.dense_hidden_dimension`` (default for ``vision_transformer``).
    - ``native_transformer`` — ``nn.TransformerEncoder`` on the sequence (default for ``post_concat`` / ``unified``).
    - ``mlp`` / ``cnn`` / ``hf_embedding`` — see ``fusion_mlp``, ``fusion_cnn``; HF path also via ``transformer`` below.
    - **HF encoder:** set ``encoder.transformer.use_hf_backbone: true`` and ``encoder.transformer.model_name_or_path``
      (mirrors ``nn.vis.transformer``). Alternate: ``fusion_encoder: hf_embedding`` + ``hf_embedding``.

    HF models must accept ``inputs_embeds`` (e.g. BERT, RoBERTa, small GPT-2).

    Global routing is on :class:`NnConfig` (``fusion_mode``, ``init_from_pretrained``).
    """

    model_config = ConfigDict(extra="ignore")

    freeze: bool = Field(
        default=False,
        description="RL PPO: freeze multimodal fusion stack. IQN: same when ``fusion_mode != none`` (submodule ``fusion.``).",
    )
    mlp: FloatMlpBodyConfig | None = None
    fusion_encoder: FusionEncoderKindLiteral | None = None
    post_concat_layout: PostConcatLayoutLiteral = Field(
        default="fused_vector",
        description="Mirrors multimodal bundle; only used when nn.fusion_mode == post_concat.",
    )
    float_token_input: FloatTokenInputLiteral = Field(
        default="raw",
        description="For post_concat token_sequence: raw normalized floats vs MLP hidden before float token Linear.",
    )
    float_token_layout: FloatTokenLayoutLiteral = Field(
        default="dense",
        description="token_sequence: dense K tokens from one Linear vs per_feature (= float_input_dim tokens).",
    )
    transformer: TransformersConfig = Field(default_factory=TransformersConfig)
    fusion_mlp: FusionMlpEncoderConfig | None = None
    fusion_cnn: FusionCnnEncoderConfig | None = None
    hf_embedding: HfEmbeddingEncoderConfig | None = None

    @model_validator(mode="after")
    def _fusion_encoder_subconfigs(self) -> NnEncoderConfig:
        fe = self.fusion_encoder
        if fe == "mlp" and self.fusion_mlp is None:
            object.__setattr__(self, "fusion_mlp", FusionMlpEncoderConfig())
        if fe == "cnn" and self.fusion_cnn is None:
            object.__setattr__(self, "fusion_cnn", FusionCnnEncoderConfig())
        if fe == "hf_embedding" and self.hf_embedding is None:
            object.__setattr__(self, "hf_embedding", HfEmbeddingEncoderConfig())
        return self

    @model_validator(mode="after")
    def _encoder_transformer_hf_backbone(self) -> NnEncoderConfig:
        tr = self.transformer
        if tr.use_hf_backbone and not (tr.model_name_or_path or "").strip():
            raise ValueError(
                "encoder.transformer.use_hf_backbone requires encoder.transformer.model_name_or_path "
                "(same pattern as nn.vis.transformer)"
            )
        if tr.use_hf_backbone and self.fusion_encoder == "native_transformer":
            raise ValueError(
                "encoder.fusion_encoder: native_transformer conflicts with encoder.transformer.use_hf_backbone; "
                "omit fusion_encoder or set use_hf_backbone: false for torch.nn.TransformerEncoder"
            )
        return self

    @model_validator(mode="after")
    def _per_feature_float_rules(self) -> NnEncoderConfig:
        if self.float_token_layout != "per_feature":
            return self
        if self.post_concat_layout != "token_sequence":
            raise ValueError(
                "encoder.float_token_layout: per_feature requires encoder.post_concat_layout: token_sequence"
            )
        if self.float_token_input != "raw":
            raise ValueError("encoder.float_token_layout: per_feature requires encoder.float_token_input: raw")
        return self

    @model_validator(mode="after")
    def _hf_embedding_path(self) -> NnEncoderConfig:
        if self.fusion_encoder != "hf_embedding":
            return self
        he = self.hf_embedding
        tr = self.transformer
        has_he = he is not None and bool((he.model_name_or_path or "").strip())
        has_tr_hf = tr.use_hf_backbone and bool((tr.model_name_or_path or "").strip())
        if not has_he and not has_tr_hf:
            raise ValueError(
                "encoder.fusion_encoder: hf_embedding requires either encoder.hf_embedding.model_name_or_path "
                "or encoder.transformer.use_hf_backbone with encoder.transformer.model_name_or_path"
            )
        return self


class IqnCoreConfig(BaseModel):
    embedding_dimension: int = Field(default=128, ge=1)
    n: int = Field(default=8, ge=1)
    k: int = Field(default=32, ge=1)
    kappa: float = Field(default=5e-3, gt=0)
    freeze: bool = Field(default=False, description="RL IQN: freeze iqn_fc (quantile MLP). PPO: ignored.")


class IqnHeadSlotConfig(BaseModel):
    """One IQN decoder slot: **either** ``mlp`` **or** ``transformer`` (mutually exclusive)."""

    model_config = ConfigDict(extra="ignore")

    freeze: bool = Field(
        default=False,
        description="RL: IQN → A_head + A_head_multi / V_head; PPO → policy_head / value_head.",
    )
    mlp: MLPConfig | None = None
    transformer: TransformerStackConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_transformer_encoder_key(cls, data: object) -> object:
        if isinstance(data, dict) and "transformer_encoder" in data and "transformer" not in data:
            out = dict(data)
            out["transformer"] = out.pop("transformer_encoder")
            return out
        return data

    @model_validator(mode="after")
    def _xor(self) -> IqnHeadSlotConfig:
        has_m = self.mlp is not None
        has_t = self.transformer is not None
        if has_m and has_t:
            raise ValueError("IQN decoder slot: set only one of 'mlp' or 'transformer'")
        if not has_m and not has_t:
            object.__setattr__(self, "mlp", MLPConfig())
        return self


class IqnDecoderConfig(BaseModel):
    """IQN policy decoder: advantage / value slots; ``shared_input`` selects transformer attachment."""

    shared_input: Literal["pre_tau", "post_tau"] = "post_tau"
    dense_hidden_dimension: int = Field(default=1024, ge=32)
    shared_trunk_freeze: bool = Field(
        default=False,
        description="RL PPO: freeze shared MLP trunk before heads. IQN: ignored.",
    )
    advantage: IqnHeadSlotConfig = Field(default_factory=IqnHeadSlotConfig)
    value: IqnHeadSlotConfig = Field(default_factory=IqnHeadSlotConfig)

    @model_validator(mode="after")
    def _transformer_only_post_tau(self) -> IqnDecoderConfig:
        if self.shared_input == "pre_tau":
            if self.advantage.transformer is not None or self.value.transformer is not None:
                raise ValueError(
                    "IQN decoder.transformer slots require decoder.shared_input == 'post_tau' "
                    "(chunked encoder on tau-modulated features). Use 'post_tau' or MLP."
                )
        return self


class NnTrainingBlock(BaseModel):
    """Optimizer-adjacent and reset knobs under ``nn.training``."""

    use_jit: bool = True
    use_ddqn: bool = True
    clip_grad_value: float = 1000
    clip_grad_norm: float = 30
    number_memories_trained_on_between_target_network_updates: int = 2048
    soft_update_tau: float = 0.02
    target_self_loss_clamp_ratio: float = 4
    single_reset_flag: int = 0
    reset_every_n_frames_generated: int = 400_000_00000000
    additional_transition_after_reset: int = 1_600_000
    last_layer_reset_factor: float = 0.8
    overall_reset_mul_factor: float = 0.01


class NnConfig(BaseModel):
    """Root ``nn`` block. Omitted keys use defaults; training knobs live under ``nn.training``."""

    model_config = ConfigDict(extra="ignore")

    fusion_mode: MultimodalFusionModeLiteral = "none"
    init_from_pretrained: str = Field(
        default="",
        description="Rulka fusion save_pretrained dir when fusion_mode != none; trust_remote_code from encoder.transformer.",
    )
    vis: VisConfig = Field(default_factory=VisConfig)
    float_branch: FloatBranchConfig = Field(
        default_factory=FloatBranchConfig,
        validation_alias="float",
        serialization_alias="float",
    )
    encoder: NnEncoderConfig = Field(default_factory=NnEncoderConfig)
    iqn: IqnCoreConfig = Field(default_factory=IqnCoreConfig)
    decoder: IqnDecoderConfig = Field(default_factory=IqnDecoderConfig)
    nn_training_ops: NnTrainingBlock = Field(
        default_factory=NnTrainingBlock,
        validation_alias="training",
    )

    float_input_dim: int = 0

    @model_validator(mode="before")
    @classmethod
    def _hoist_encoder_fusion_keys(cls, data: object) -> object:
        """Legacy YAML had ``fusion_mode`` / ``init_from_pretrained`` under ``encoder``; they now live on ``nn``."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        enc = out.get("encoder")
        if not isinstance(enc, dict):
            return out
        enc_copy = dict(enc)
        for k in ("fusion_mode", "init_from_pretrained"):
            if k not in enc_copy:
                continue
            v = enc_copy.pop(k)
            if k not in out:
                out[k] = v
            elif out[k] != v:
                raise ValueError(f"nn.{k} conflicts with nn.encoder.{k} (set only nn.{k})")
        out["encoder"] = enc_copy
        return out

    @model_validator(mode="after")
    def _fusion_agrees_with_vis_transformer(self) -> NnConfig:
        enc = self.encoder
        vt = self.vis.transformer
        if self.fusion_mode == "unified" and vt is not None and not vt.use_hf_backbone:
            if vt.d_model != enc.transformer.d_model:
                raise ValueError(
                    f"unified + native patch vision requires vis.transformer.d_model ({vt.d_model}) "
                    f"== encoder.transformer.d_model ({enc.transformer.d_model})"
                )
        if self.fusion_mode == "post_concat" and vt is not None:
            if vt.fusion_tokens == "patch_tokens" and enc.post_concat_layout != "token_sequence":
                raise ValueError(
                    "vis.transformer.fusion_tokens: patch_tokens requires encoder.post_concat_layout: token_sequence "
                    "(fused_vector uses a single vision vector before fusion tokenization)"
                )
        return self

    def to_multimodal(self) -> MultimodalTransformersConfig:
        enc = self.encoder
        return MultimodalTransformersConfig(
            fusion_mode=self.fusion_mode,
            transformer=enc.transformer,
            init_from_pretrained=self.init_from_pretrained,
            fusion_encoder=enc.fusion_encoder,
            fusion_mlp=enc.fusion_mlp,
            fusion_cnn=enc.fusion_cnn,
            hf_embedding=enc.hf_embedding,
            post_concat_layout=enc.post_concat_layout,
            float_token_input=enc.float_token_input,
            float_token_layout=enc.float_token_layout,
        )

    @property
    def fusion(self) -> MultimodalTransformersConfig:
        return self.to_multimodal()

    @property
    def image_size(self) -> ImageSizeConfig:
        return self.vis.image_size

    @property
    def w_downsized(self) -> int:
        return self.vis.image_size.width

    @property
    def h_downsized(self) -> int:
        return self.vis.image_size.height

    def float_hidden_dim_effective(self) -> int:
        """PPO may override float MLP width via ``encoder.mlp``."""
        if self.encoder.mlp is not None:
            return self.encoder.mlp.hidden_dim
        return self.float_branch.mlp.hidden_dim

    @property
    def float_hidden_dim(self) -> int:
        return self.float_branch.mlp.hidden_dim

    @property
    def dense_hidden_dimension(self) -> int:
        return self.decoder.dense_hidden_dimension

    @property
    def iqn_embedding_dimension(self) -> int:
        return self.iqn.embedding_dimension

    @property
    def iqn_n(self) -> int:
        return self.iqn.n

    @property
    def iqn_k(self) -> int:
        return self.iqn.k

    @property
    def iqn_kappa(self) -> float:
        return self.iqn.kappa

    @property
    def use_iqn_image_head(self) -> bool:
        return not self.vis.no_image

    @property
    def transformers(self) -> MultimodalTransformersConfig:
        return self.to_multimodal()

    @property
    def use_jit(self) -> bool:
        return self.nn_training_ops.use_jit

    @property
    def use_ddqn(self) -> bool:
        return self.nn_training_ops.use_ddqn

    @property
    def clip_grad_value(self) -> float:
        return self.nn_training_ops.clip_grad_value

    @property
    def clip_grad_norm(self) -> float:
        return self.nn_training_ops.clip_grad_norm

    @property
    def number_memories_trained_on_between_target_network_updates(self) -> int:
        return self.nn_training_ops.number_memories_trained_on_between_target_network_updates

    @property
    def soft_update_tau(self) -> float:
        return self.nn_training_ops.soft_update_tau

    @property
    def target_self_loss_clamp_ratio(self) -> float:
        return self.nn_training_ops.target_self_loss_clamp_ratio

    @property
    def single_reset_flag(self) -> int:
        return self.nn_training_ops.single_reset_flag

    @property
    def reset_every_n_frames_generated(self) -> int:
        return self.nn_training_ops.reset_every_n_frames_generated

    @property
    def additional_transition_after_reset(self) -> int:
        return self.nn_training_ops.additional_transition_after_reset

    @property
    def last_layer_reset_factor(self) -> float:
        return self.nn_training_ops.last_layer_reset_factor

    @property
    def overall_reset_mul_factor(self) -> float:
        return self.nn_training_ops.overall_reset_mul_factor


NeuralNetworkConfig = NnConfig

# Shared building-block names (same classes as vis.cnn, decoder.mlp, …)
CnnConfig = VisCnnBodyConfig
