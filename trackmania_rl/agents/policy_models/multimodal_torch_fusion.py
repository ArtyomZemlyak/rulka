"""Multimodal actor-critic: ``nn.fusion_mode`` + ``nn.encoder.transformer`` + ``nn.vis``.

For every fusion mode (``vision_transformer``, ``post_concat``, ``unified``), ``nn.vis`` can be:

- **CNN** (``nn.vis.cnn`` — default when no ``transformer`` block). The same ``_build_img_head`` flags as IQN/PPO
  (IMPALA, spectral norm, adaptive pool) are read from ``nn.vis.cnn`` when the model is built from a full Rulka config.
- **Native** patch embedding + ``nn.TransformerEncoder`` (``nn.vis.transformer`` with ``use_hf_backbone: false``),
- **Hugging Face** vision backbone (``nn.vis.transformer`` with ``use_hf_backbone: true``).

``nn.init_from_pretrained`` loads Rulka ``save_pretrained`` weights after build unless
:func:`trackmania_rl.utilities.skip_multimodal_fusion_hub_init_from_pretrained` applies (existing
``weights1.torch`` or BC ``ppo_policy_bc.pt`` for this run).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config_files.config_loader import get_config
from config_files.nn_schema import (
    FusionCnnEncoderConfig,
    FusionMlpEncoderConfig,
    HfEmbeddingEncoderConfig,
    MultimodalTransformersConfig,
    TransformersConfig,
    infer_fusion_encoder,
    infer_vis_branch,
)
from trackmania_rl.agents.iqn import _build_img_head, calculate_conv_output_dim
from trackmania_rl.nn_build.vis_cnn_head import merge_vis_cnn_head_kw, vis_cnn_head_kw_from_nn_vis
from trackmania_rl.agents.policy_models.hf_vision_utils import hf_vision_backbone_hidden_size
from trackmania_rl.agents.policy_optimization.ppo import discrete_action_logprob_and_entropy
from trackmania_rl.agents.policy_optimization.types import PolicyOutput
from trackmania_rl import utilities as tr_utilities


def _lazy_import_transformers():
    tr_utilities.quiet_transformers_weight_load_reports()
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as e:
        raise ImportError(
            'HF vision inside fusion requires transformers. Install: pip install -e ".[policy]"'
        ) from e
    return AutoImageProcessor, AutoModel


class PatchEmbed2d(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        b, _, h, w = x.shape
        return x.flatten(2).transpose(1, 2)


def _make_encoder(d_model: int, nhead: int, nlayers: int, dim_ff: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        dropout=dropout,
        batch_first=True,
        norm_first=False,
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=nlayers)


def _make_encoder_optional(
    d_model: int, nhead: int, nlayers: int, dim_ff: float, dropout: float
) -> nn.TransformerEncoder | None:
    if nlayers <= 0:
        return None
    return _make_encoder(d_model, nhead, nlayers, int(dim_ff), dropout)


def _build_fusion_mlp(flat_in: int, fuse_d: int, cfg: FusionMlpEncoderConfig) -> nn.Module:
    layers: list[nn.Module] = []
    d = flat_in
    for _ in range(max(0, cfg.n_layers - 1)):
        layers.extend([nn.Linear(d, cfg.hidden_dim), nn.ReLU(inplace=True)])
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        d = cfg.hidden_dim
    layers.append(nn.Linear(d, fuse_d))
    return nn.Sequential(*layers)


def _build_fusion_cnn(in_ch: int, fuse_d: int, cfg: FusionCnnEncoderConfig) -> nn.Module:
    layers: list[nn.Module] = []
    c = in_ch
    pad = cfg.kernel_size // 2
    for h in cfg.hidden_channels:
        layers.extend(
            [
                nn.Conv1d(c, h, kernel_size=cfg.kernel_size, padding=pad),
                nn.ReLU(inplace=True),
            ]
        )
        if cfg.dropout > 0:
            layers.append(nn.Dropout1d(cfg.dropout))
        c = h
    conv = nn.Sequential(*layers)
    tail = nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(c, fuse_d) if c != fuse_d else nn.Identity(),
    )
    return nn.Sequential(conv, tail)


def prepare_hf_pixels(proc: Any, img: Tensor) -> Tensor:
    x = img
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    size = getattr(proc, "size", None)
    if isinstance(size, dict) and "height" in size:
        th, tw = int(size["height"]), int(size["width"])
    elif hasattr(proc, "size") and isinstance(proc.size, (tuple, list)):
        th, tw = int(proc.size[0]), int(proc.size[1])
    else:
        th, tw = 224, 224
    x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    if hasattr(proc, "image_mean") and proc.image_mean is not None:
        mean = torch.tensor(proc.image_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(proc.image_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x01 - mean) / std
    return x01


def hf_backbone_num_image_tokens(
    hf_bb: nn.Module,
    proc: Any,
    *,
    image_h: int,
    image_w: int,
) -> int:
    hf_bb.eval()
    x = torch.zeros(1, 1, image_h, image_w, dtype=torch.float32)
    pix = prepare_hf_pixels(proc, x)
    with torch.no_grad():
        out = hf_bb(pixel_values=pix)
    return int(out.last_hidden_state.shape[1])


class TorchMultimodalActorCritic(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        vis_branch: str,
        fusion_encoder: str,
        float_inputs_dim: int,
        float_hidden_dim: int,
        dense_hidden_dim: int,
        vis_d_model: int,
        vis_nhead: int,
        vis_nlayers: int,
        vis_ff_mult: int,
        vis_dropout: float,
        vis_patch_size: int,
        fuse_d_model: int,
        fuse_nhead: int,
        fuse_nlayers: int,
        fuse_ff_mult: int,
        fuse_dropout: float,
        post_concat_seq_len: int,
        unified_float_tokens: int,
        unified_hf_n_tokens: int,
        image_h: int,
        image_w: int,
        use_image_head: bool,
        float_inputs_mean: Tensor,
        float_inputs_std: Tensor,
        n_actions: int,
        n_actions_per_block: int,
        hf_vis_backbone: Optional[nn.Module] = None,
        hf_vis_processor: Any = None,
        hf_vis_hidden_size: int = 0,
        fusion_mlp: FusionMlpEncoderConfig | None = None,
        fusion_cnn: FusionCnnEncoderConfig | None = None,
        hf_embedding_cfg: HfEmbeddingEncoderConfig | None = None,
        hf_fusion_backbone: Optional[nn.Module] = None,
        hf_fusion_hidden_size: int = 0,
        post_concat_layout: str = "fused_vector",
        vis_fusion_tokens: str = "summary",
        float_token_input: str = "raw",
        float_token_layout: str = "dense",
        n_hf_image_tokens_post: int = 0,
        vis_cnn_head_kw: Optional[Mapping[str, Any]] = None,
        include_policy_heads: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("vision_transformer", "post_concat", "unified"):
            raise ValueError(mode)
        vb = vis_branch
        if vb not in ("none", "cnn", "native_transformer", "hf_transformer"):
            raise ValueError(vis_branch)
        self.mode = mode
        self._vis_branch = vb
        self.n_actions = n_actions
        self.n_actions_per_block = n_actions_per_block
        self.include_policy_heads = bool(include_policy_heads)
        self.use_image_head = use_image_head
        self._unified_n_img = 0
        self.register_buffer("float_inputs_mean", float_inputs_mean)
        self.register_buffer("float_inputs_std", float_inputs_std)

        self._hf_vis_backbone = hf_vis_backbone
        self._hf_vis_processor = hf_vis_processor
        self._vis_d_model = int(vis_d_model)
        self._fuse_d_model = int(fuse_d_model)
        self._fe = fusion_encoder
        if self._fe not in ("linear", "native_transformer", "mlp", "cnn", "hf_embedding"):
            raise ValueError(fusion_encoder)
        fuse_dim_ff = max(64, fuse_d_model * fuse_ff_mult)
        self._fuse_cfg = (fuse_d_model, fuse_nhead, fuse_nlayers, fuse_dim_ff, fuse_dropout)
        self._fusion_mlp_cfg = fusion_mlp
        self._fusion_cnn_cfg = fusion_cnn
        self._hf_emb_cfg = hf_embedding_cfg
        self._hf_fusion_bb = hf_fusion_backbone
        self._hf_fusion_h = int(hf_fusion_hidden_size)
        self._post_concat_seq_len = int(post_concat_seq_len)
        self._post_concat_layout = post_concat_layout if mode == "post_concat" else "fused_vector"
        self._vis_fusion_tokens = vis_fusion_tokens
        self._float_token_input = float_token_input
        self._float_token_layout = float_token_layout
        self._n_hf_image_tokens_post = int(n_hf_image_tokens_post)
        self._post_concat_n_vis = 0
        self._post_concat_n_float = int(unified_float_tokens)
        self.vis_to_fuse: Optional[nn.Module] = None
        self.postconcat_cnn_to_fuse: Optional[nn.Module] = None

        if mode == "unified":
            self.float_feature_extractor = None
        elif mode == "post_concat" and self._post_concat_layout == "token_sequence" and float_token_input == "raw":
            self.float_feature_extractor = None
        else:
            self.float_feature_extractor = nn.Sequential(
                nn.Linear(float_inputs_dim, float_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(float_hidden_dim, float_hidden_dim),
                nn.ReLU(inplace=True),
            )

        self.patch_embed: Optional[PatchEmbed2d] = None
        self.pos_vis: Optional[nn.Parameter] = None
        self.enc_vis: Optional[nn.TransformerEncoder] = None
        self.img_head: Optional[nn.Module] = None
        self.conv_out_dim = 0
        self.cnn_to_vis: Optional[nn.Module] = None
        self.cnn_to_fuse: Optional[nn.Module] = None
        self.hf_vis_proj: Optional[nn.Module] = None
        self.vis_refine: Optional[nn.TransformerEncoder] = None
        self.pos_pc: Optional[nn.Parameter] = None
        self.fused_to_seq: Optional[nn.Linear] = None
        self.pos_uni: Optional[nn.Parameter] = None
        self.float_to_tokens: Optional[nn.Linear] = None
        self.float_scalar_to_tok: Optional[nn.Linear] = None
        self.float_per_feat_slot_emb: Optional[nn.Parameter] = None
        self.bridge: Optional[nn.Linear] = None
        self.visfloat_to_seq: Optional[nn.Linear] = None
        self.pos_vf: Optional[nn.Parameter] = None
        self.enc_fusion_native: Optional[nn.TransformerEncoder] = None
        self.fusion_mlp_mod: Optional[nn.Module] = None
        self.fusion_cnn_mod: Optional[nn.Module] = None
        self.hf_fusion_proj_in: Optional[nn.Linear] = None
        self.hf_fusion_proj_out: Optional[nn.Module] = None
        self._hf_fusion_dropout_p = 0.0

        vis_dim_ff = max(64, vis_d_model * vis_ff_mult)
        self._hf_hidden_dropout_p = 0.0
        self._img_head_cnn_kw = merge_vis_cnn_head_kw(vis_cnn_head_kw)

        if mode == "vision_transformer":
            self._init_vision_transformer_branch(
                vb=vb,
                vis_d_model=vis_d_model,
                vis_nhead=vis_nhead,
                vis_nlayers=vis_nlayers,
                vis_dim_ff=vis_dim_ff,
                vis_dropout=vis_dropout,
                vis_patch_size=vis_patch_size,
                float_hidden_dim=float_hidden_dim,
                dense_hidden_dim=dense_hidden_dim,
                image_h=image_h,
                image_w=image_w,
                use_image_head=use_image_head,
                hf_vis_backbone=hf_vis_backbone,
                hf_vis_hidden_size=hf_vis_hidden_size,
                img_head_cnn_kw=self._img_head_cnn_kw,
            )
        elif mode == "post_concat":
            self._init_post_concat_branch(
                vb=vb,
                vis_d_model=vis_d_model,
                vis_nhead=vis_nhead,
                vis_nlayers=vis_nlayers,
                vis_dim_ff=vis_dim_ff,
                vis_dropout=vis_dropout,
                vis_patch_size=vis_patch_size,
                float_inputs_dim=float_inputs_dim,
                float_hidden_dim=float_hidden_dim,
                dense_hidden_dim=dense_hidden_dim,
                fuse_d_model=fuse_d_model,
                fuse_nhead=fuse_nhead,
                fuse_nlayers=fuse_nlayers,
                fuse_ff_mult=fuse_ff_mult,
                fuse_dropout=fuse_dropout,
                post_concat_seq_len=post_concat_seq_len,
                unified_float_tokens=unified_float_tokens,
                image_h=image_h,
                image_w=image_w,
                use_image_head=use_image_head,
                hf_vis_backbone=hf_vis_backbone,
                hf_vis_hidden_size=hf_vis_hidden_size,
                post_concat_layout=self._post_concat_layout,
                vis_fusion_tokens=self._vis_fusion_tokens,
                float_token_input=self._float_token_input,
                float_token_layout=self._float_token_layout,
                n_hf_vis_tokens=self._n_hf_image_tokens_post,
                img_head_cnn_kw=self._img_head_cnn_kw,
            )
        else:
            self._init_unified_branch(
                vb=vb,
                vis_d_model=vis_d_model,
                vis_nhead=vis_nhead,
                vis_nlayers=vis_nlayers,
                vis_dim_ff=vis_dim_ff,
                vis_dropout=vis_dropout,
                vis_patch_size=vis_patch_size,
                float_inputs_dim=float_inputs_dim,
                dense_hidden_dim=dense_hidden_dim,
                fuse_d_model=fuse_d_model,
                fuse_nhead=fuse_nhead,
                fuse_nlayers=fuse_nlayers,
                fuse_ff_mult=fuse_ff_mult,
                fuse_dropout=fuse_dropout,
                unified_float_tokens=unified_float_tokens,
                unified_hf_n_tokens=unified_hf_n_tokens,
                image_h=image_h,
                image_w=image_w,
                use_image_head=use_image_head,
                hf_vis_backbone=hf_vis_backbone,
                hf_vis_hidden_size=hf_vis_hidden_size,
                img_head_cnn_kw=self._img_head_cnn_kw,
            )

        out_pi = n_actions * n_actions_per_block
        if self.include_policy_heads:
            self.trunk = nn.Sequential(
                nn.Linear(dense_hidden_dim, dense_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(dense_hidden_dim, dense_hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.policy_head = nn.Linear(dense_hidden_dim, out_pi)
            self.value_head = nn.Linear(dense_hidden_dim, 1)
        else:
            self.trunk = None
            self.policy_head = None
            self.value_head = None

    def _init_vision_transformer_branch(
        self,
        *,
        vb: str,
        vis_d_model: int,
        vis_nhead: int,
        vis_nlayers: int,
        vis_dim_ff: int,
        vis_dropout: float,
        vis_patch_size: int,
        float_hidden_dim: int,
        dense_hidden_dim: int,
        image_h: int,
        image_w: int,
        use_image_head: bool,
        hf_vis_backbone: Optional[nn.Module],
        hf_vis_hidden_size: int,
        img_head_cnn_kw: dict[str, Any],
    ) -> None:
        if vb == "cnn" and use_image_head:
            self.img_head = _build_img_head(**img_head_cnn_kw)
            self.conv_out_dim = int(calculate_conv_output_dim(self.img_head, image_h, image_w))
            self.cnn_to_vis = (
                nn.Linear(self.conv_out_dim, vis_d_model)
                if self.conv_out_dim != vis_d_model
                else nn.Identity()
            )
        elif vb == "native_transformer" and use_image_head:
            ps = vis_patch_size
            if image_h % ps != 0 or image_w % ps != 0:
                raise ValueError(
                    f"vis.patch_size={ps} must divide H_downsized,W_downsized ({image_h},{image_w}) "
                    f"for fusion_mode=vision_transformer"
                )
            n_vis = (image_h // ps) * (image_w // ps)
            self.patch_embed = PatchEmbed2d(1, vis_d_model, ps)
            self.pos_vis = nn.Parameter(torch.zeros(1, n_vis, vis_d_model))
            self.enc_vis = _make_encoder_optional(vis_d_model, vis_nhead, vis_nlayers, vis_dim_ff, vis_dropout)
        elif vb == "hf_transformer":
            if hf_vis_backbone is not None:
                bh = int(hf_vis_hidden_size)
                self.hf_vis_proj = nn.Linear(bh, vis_d_model) if bh != vis_d_model else nn.Identity()
                self.vis_refine = _make_encoder_optional(vis_d_model, vis_nhead, vis_nlayers, vis_dim_ff, vis_dropout)
        concat_dim = vis_d_model + float_hidden_dim
        if self._fe == "linear":
            self.bridge = nn.Linear(concat_dim, dense_hidden_dim)
        else:
            sl = self._post_concat_seq_len
            self.visfloat_to_seq = nn.Linear(concat_dim, sl * self._fuse_d_model)
            self.pos_vf = nn.Parameter(torch.zeros(1, sl, self._fuse_d_model))
            self.bridge = nn.Linear(self._fuse_d_model, dense_hidden_dim)
            self._install_fusion_trunk(sl)

    def _init_post_concat_branch(
        self,
        *,
        vb: str,
        vis_d_model: int,
        vis_nhead: int,
        vis_nlayers: int,
        vis_dim_ff: int,
        vis_dropout: float,
        vis_patch_size: int,
        float_inputs_dim: int,
        float_hidden_dim: int,
        dense_hidden_dim: int,
        fuse_d_model: int,
        fuse_nhead: int,
        fuse_nlayers: int,
        fuse_ff_mult: int,
        fuse_dropout: float,
        post_concat_seq_len: int,
        unified_float_tokens: int,
        image_h: int,
        image_w: int,
        use_image_head: bool,
        hf_vis_backbone: Optional[nn.Module],
        hf_vis_hidden_size: int,
        post_concat_layout: str,
        vis_fusion_tokens: str,
        float_token_input: str,
        float_token_layout: str,
        n_hf_vis_tokens: int,
        img_head_cnn_kw: dict[str, Any],
    ) -> None:
        if post_concat_layout == "token_sequence":
            self.fused_to_seq = None
            n_vis = 0
            if not use_image_head or vb == "none":
                n_vis = 0
            elif vb == "cnn":
                self.img_head = _build_img_head(**img_head_cnn_kw)
                self.conv_out_dim = int(calculate_conv_output_dim(self.img_head, image_h, image_w))
                self.postconcat_cnn_to_fuse = (
                    nn.Linear(self.conv_out_dim, fuse_d_model)
                    if self.conv_out_dim != fuse_d_model
                    else nn.Identity()
                )
                n_vis = 1
            elif vb == "native_transformer":
                ps = vis_patch_size
                if image_h % ps != 0 or image_w % ps != 0:
                    raise ValueError(
                        f"vis.patch_size={ps} must divide H_downsized,W_downsized ({image_h},{image_w}) "
                        f"for post_concat + native_transformer"
                    )
                n_patch = (image_h // ps) * (image_w // ps)
                self.patch_embed = PatchEmbed2d(1, vis_d_model, ps)
                self.pos_vis = nn.Parameter(torch.zeros(1, n_patch, vis_d_model))
                self.enc_vis = _make_encoder_optional(vis_d_model, vis_nhead, vis_nlayers, vis_dim_ff, vis_dropout)
                self.vis_to_fuse = (
                    nn.Linear(vis_d_model, fuse_d_model) if vis_d_model != fuse_d_model else nn.Identity()
                )
                n_vis = n_patch if vis_fusion_tokens == "patch_tokens" else 1
            elif vb == "hf_transformer":
                if hf_vis_backbone is not None:
                    bh = int(hf_vis_hidden_size)
                    self.hf_vis_proj = nn.Linear(bh, vis_d_model) if bh != vis_d_model else nn.Identity()
                    self.vis_refine = _make_encoder_optional(vis_d_model, vis_nhead, vis_nlayers, vis_dim_ff, vis_dropout)
                if use_image_head and hf_vis_backbone is not None:
                    self.vis_to_fuse = (
                        nn.Linear(vis_d_model, fuse_d_model) if vis_d_model != fuse_d_model else nn.Identity()
                    )
                    if vis_fusion_tokens == "patch_tokens":
                        n_vis = max(1, int(n_hf_vis_tokens))
                    else:
                        n_vis = 1
            else:
                raise ValueError(vb)

            self._post_concat_n_vis = n_vis
            if float_token_layout == "per_feature":
                if float_token_input != "raw":
                    raise ValueError("post_concat token_sequence per_feature requires float_token_input raw")
                k_float = int(float_inputs_dim)
                self._post_concat_n_float = k_float
                self.float_to_tokens = None
                self.float_scalar_to_tok = nn.Linear(1, fuse_d_model)
                self.float_per_feat_slot_emb = nn.Parameter(torch.zeros(1, k_float, fuse_d_model))
            else:
                k_float = int(unified_float_tokens)
                self._post_concat_n_float = k_float
                float_in = int(float_inputs_dim) if float_token_input == "raw" else int(float_hidden_dim)
                self.float_to_tokens = nn.Linear(float_in, k_float * fuse_d_model)
                self.float_scalar_to_tok = None
                self.float_per_feat_slot_emb = None
            seq_len = n_vis + self._post_concat_n_float
            if seq_len < 1:
                raise ValueError("post_concat token_sequence needs at least one vision or float token")
            self.pos_pc = nn.Parameter(torch.zeros(1, seq_len, fuse_d_model))
            self.bridge = nn.Linear(fuse_d_model, dense_hidden_dim)
            if self._fe != "linear":
                self._install_fusion_trunk(seq_len)
            return

        self.float_to_tokens = None
        self.float_scalar_to_tok = None
        self.float_per_feat_slot_emb = None
        iv_dim = 0
        if vb == "cnn" and use_image_head:
            self.img_head = _build_img_head(**img_head_cnn_kw)
            self.conv_out_dim = int(calculate_conv_output_dim(self.img_head, image_h, image_w))
            iv_dim = self.conv_out_dim
        elif vb == "native_transformer" and use_image_head:
            ps = vis_patch_size
            if image_h % ps != 0 or image_w % ps != 0:
                raise ValueError(
                    f"vis.patch_size={ps} must divide H_downsized,W_downsized ({image_h},{image_w}) "
                    f"for post_concat + native_transformer"
                )
            n_vis = (image_h // ps) * (image_w // ps)
            self.patch_embed = PatchEmbed2d(1, vis_d_model, ps)
            self.pos_vis = nn.Parameter(torch.zeros(1, n_vis, vis_d_model))
            self.enc_vis = _make_encoder_optional(vis_d_model, vis_nhead, vis_nlayers, vis_dim_ff, vis_dropout)
            iv_dim = vis_d_model
        elif vb == "hf_transformer":
            if hf_vis_backbone is not None:
                bh = int(hf_vis_hidden_size)
                self.hf_vis_proj = nn.Linear(bh, vis_d_model) if bh != vis_d_model else nn.Identity()
                self.vis_refine = _make_encoder_optional(vis_d_model, vis_nhead, vis_nlayers, vis_dim_ff, vis_dropout)
            if use_image_head and hf_vis_backbone is not None:
                iv_dim = vis_d_model

        fused_dim = iv_dim + float_hidden_dim
        self.fused_to_seq = nn.Linear(fused_dim, post_concat_seq_len * fuse_d_model)
        self.pos_pc = nn.Parameter(torch.zeros(1, post_concat_seq_len, fuse_d_model))
        self.bridge = nn.Linear(fuse_d_model, dense_hidden_dim)
        if self._fe != "linear":
            self._install_fusion_trunk(post_concat_seq_len)

    def _init_unified_branch(
        self,
        *,
        vb: str,
        vis_d_model: int,
        vis_nhead: int,
        vis_nlayers: int,
        vis_dim_ff: int,
        vis_dropout: float,
        vis_patch_size: int,
        float_inputs_dim: int,
        dense_hidden_dim: int,
        fuse_d_model: int,
        fuse_nhead: int,
        fuse_nlayers: int,
        fuse_ff_mult: int,
        fuse_dropout: float,
        unified_float_tokens: int,
        unified_hf_n_tokens: int,
        image_h: int,
        image_w: int,
        use_image_head: bool,
        hf_vis_backbone: Optional[nn.Module],
        hf_vis_hidden_size: int,
        img_head_cnn_kw: dict[str, Any],
    ) -> None:
        n_img = 0
        if vb == "cnn" and use_image_head:
            self.img_head = _build_img_head(**img_head_cnn_kw)
            self.conv_out_dim = int(calculate_conv_output_dim(self.img_head, image_h, image_w))
            self.cnn_to_fuse = (
                nn.Linear(self.conv_out_dim, fuse_d_model)
                if self.conv_out_dim != fuse_d_model
                else nn.Identity()
            )
            n_img = 1
        elif vb == "native_transformer" and use_image_head:
            if vis_d_model != fuse_d_model:
                raise ValueError("unified + native_transformer requires vis_d_model == fuse_d_model")
            ps = vis_patch_size
            if image_h % ps != 0 or image_w % ps != 0:
                raise ValueError(
                    f"vis.patch_size={ps} must divide H_downsized,W_downsized ({image_h},{image_w}) "
                    f"for unified + native_transformer"
                )
            n_img = (image_h // ps) * (image_w // ps)
            self.patch_embed = PatchEmbed2d(1, vis_d_model, ps)
        elif vb == "hf_transformer":
            if hf_vis_backbone is not None:
                bh = int(hf_vis_hidden_size)
                self.hf_vis_proj = nn.Linear(bh, fuse_d_model) if bh != fuse_d_model else nn.Identity()
                if use_image_head:
                    n_img = int(unified_hf_n_tokens)
                    if n_img < 1:
                        raise ValueError("unified_hf_n_tokens must be >= 1 for unified + HF")

        self._unified_n_img = n_img
        self.float_to_tokens = nn.Linear(float_inputs_dim, unified_float_tokens * fuse_d_model)
        self.pos_uni = nn.Parameter(torch.zeros(1, n_img + unified_float_tokens, fuse_d_model))
        self.bridge = nn.Linear(fuse_d_model, dense_hidden_dim)
        seq_len = n_img + unified_float_tokens
        if self._fe != "linear":
            self._install_fusion_trunk(seq_len)

    def _install_fusion_trunk(self, seq_len: int) -> None:
        fe = self._fe
        if fe == "linear":
            return
        fd, nh, nl, dff, do = self._fuse_cfg
        if fe == "native_transformer":
            self.enc_fusion_native = _make_encoder_optional(fd, nh, nl, dff, do)
        elif fe == "mlp":
            cfg = self._fusion_mlp_cfg
            assert cfg is not None
            self.fusion_mlp_mod = _build_fusion_mlp(seq_len * fd, fd, cfg)
        elif fe == "cnn":
            cfg = self._fusion_cnn_cfg
            assert cfg is not None
            self.fusion_cnn_mod = _build_fusion_cnn(fd, fd, cfg)
        elif fe == "hf_embedding":
            assert self._hf_fusion_bb is not None and self._hf_emb_cfg is not None
            hh = self._hf_fusion_h
            # Do not assign self.hf_fusion_base = self._hf_fusion_bb: that registers the same
            # submodule twice and breaks Hugging Face save_pretrained (spurious "shared tensors").
            self.hf_fusion_proj_in = nn.Linear(fd, hh)
            self.hf_fusion_proj_out = nn.Linear(hh, fd) if hh != fd else nn.Identity()
            self._hf_fusion_dropout_p = float(self._hf_emb_cfg.hidden_dropout_prob)
        else:
            raise RuntimeError(f"unknown fusion_encoder {fe!r}")

    def _fusion_trunk_forward(self, toks: Tensor) -> Tensor:
        fe = self._fe
        if fe == "native_transformer":
            if self.enc_fusion_native is not None:
                return self.enc_fusion_native(toks).mean(dim=1)
            return toks.mean(dim=1)
        if fe == "mlp":
            assert self.fusion_mlp_mod is not None
            b, seq_len, fdim = toks.shape
            return self.fusion_mlp_mod(toks.reshape(b, seq_len * fdim))
        if fe == "cnn":
            assert self.fusion_cnn_mod is not None
            return self.fusion_cnn_mod(toks.transpose(1, 2))
        if fe == "hf_embedding":
            assert self._hf_fusion_bb is not None and self.hf_fusion_proj_in is not None
            emb = self.hf_fusion_proj_in(toks)
            emb = F.dropout(emb, p=self._hf_fusion_dropout_p, training=self.training)
            mask = torch.ones(emb.shape[:2], device=emb.device, dtype=torch.long)
            out = self._hf_fusion_bb(inputs_embeds=emb, attention_mask=mask)
            hs = out.last_hidden_state
            pooled = hs.mean(dim=1)
            assert self.hf_fusion_proj_out is not None
            return self.hf_fusion_proj_out(pooled)
        raise RuntimeError(f"fusion trunk not implemented for {fe!r}")

    def _pool_fusion_tokens(self, toks: Tensor) -> Tensor:
        if self._fe == "linear":
            return toks.mean(dim=1)
        return self._fusion_trunk_forward(toks)

    def set_hf_hidden_dropout(self, p: float) -> None:
        self._hf_hidden_dropout_p = float(p)

    def _prepare_hf_pixels(self, img: Tensor) -> Tensor:
        proc = self._hf_vis_processor
        assert proc is not None
        return prepare_hf_pixels(proc, img)

    def _norm_float(self, x: Tensor) -> Tensor:
        return (x - self.float_inputs_mean) / self.float_inputs_std

    def _embed_hf_sequence(self, img: Tensor) -> Tensor:
        assert self._hf_vis_backbone is not None and self.hf_vis_proj is not None
        pix = self._prepare_hf_pixels(img)
        out = self._hf_vis_backbone(pixel_values=pix)
        toks = out.last_hidden_state
        toks = F.dropout(toks, p=self._hf_hidden_dropout_p, training=self.training)
        return self.hf_vis_proj(toks)

    def _post_concat_vis_tokens_token_sequence(self, img: Tensor, ref: Tensor) -> Tensor:
        """``post_concat`` + ``token_sequence``: vision → ``[B, n_vis, fuse_d]`` (then concat with float tokens)."""
        vb = self._vis_branch
        b = ref.shape[0]
        fd = self._fuse_d_model
        if not self.use_image_head or vb == "none":
            return ref.new_zeros(b, 0, fd)
        if vb == "cnn":
            assert self.img_head is not None and self.postconcat_cnn_to_fuse is not None
            x = self.img_head(img)
            return self.postconcat_cnn_to_fuse(x).unsqueeze(1)
        if vb == "native_transformer":
            assert self.patch_embed is not None and self.pos_vis is not None and self.vis_to_fuse is not None
            toks = self.patch_embed(img) + self.pos_vis
            if self.enc_vis is not None:
                toks = self.enc_vis(toks)
            if self._vis_fusion_tokens == "patch_tokens":
                return self.vis_to_fuse(toks)
            pooled = toks.mean(dim=1, keepdim=True)
            return self.vis_to_fuse(pooled)
        assert vb == "hf_transformer"
        if self._hf_vis_backbone is None or self.hf_vis_proj is None:
            return ref.new_zeros(b, 0, fd)
        assert self.vis_to_fuse is not None
        toks = self._embed_hf_sequence(img)
        if self._vis_fusion_tokens == "patch_tokens":
            if self.vis_refine is not None:
                toks = self.vis_refine(toks)
            return self.vis_to_fuse(toks)
        cls_t = toks[:, 0:1, :]
        if self.vis_refine is not None:
            cls_t = self.vis_refine(cls_t)
        return self.vis_to_fuse(cls_t)

    def _forward_bridge_out(self, img: Tensor, float_inputs: Tensor) -> Tensor:
        """Hidden vector after fusion bridge, before ``trunk`` (same input as ``policy_head``'s MLP stack)."""
        z = self._norm_float(float_inputs)
        if self.float_feature_extractor is not None:
            float_h = self.float_feature_extractor(z)
        else:
            float_h = None

        if self.mode == "vision_transformer":
            assert float_h is not None and self.bridge is not None
            vb = self._vis_branch
            if vb == "none" or not self.use_image_head:
                vis = float_h.new_zeros(float_h.shape[0], self._vis_d_model)
            elif vb == "cnn":
                assert self.img_head is not None and self.cnn_to_vis is not None
                vis = self.cnn_to_vis(self.img_head(img))
            elif vb == "native_transformer":
                assert self.patch_embed is not None and self.pos_vis is not None
                toks = self.patch_embed(img) + self.pos_vis
                if self.enc_vis is not None:
                    vis = self.enc_vis(toks).mean(dim=1)
                else:
                    vis = toks.mean(dim=1)
            else:
                assert vb == "hf_transformer"
                if self._hf_vis_backbone is None or self.hf_vis_proj is None:
                    vis = float_h.new_zeros(float_h.shape[0], self._vis_d_model)
                else:
                    vis = self._embed_hf_sequence(img)[:, 0]
                    if self.vis_refine is not None:
                        vis = self.vis_refine(vis.unsqueeze(1)).squeeze(1)
            z_cat = torch.cat([vis, float_h], dim=-1)
            if self._fe == "linear":
                assert self.bridge is not None
                h = self.bridge(z_cat)
            else:
                assert self.visfloat_to_seq is not None and self.pos_vf is not None and self.bridge is not None
                b = z_cat.shape[0]
                sl = self.pos_vf.shape[1]
                toks = self.visfloat_to_seq(z_cat).view(b, sl, self._fuse_d_model) + self.pos_vf
                pooled = self._pool_fusion_tokens(toks)
                h = self.bridge(pooled)
        elif self.mode == "post_concat":
            assert self.pos_pc is not None and self.bridge is not None
            if self._post_concat_layout == "fused_vector":
                assert float_h is not None and self.fused_to_seq is not None
                vb = self._vis_branch
                if vb == "none" or not self.use_image_head:
                    iv = float_h.new_zeros(float_h.shape[0], 0)
                elif vb == "cnn":
                    assert self.img_head is not None
                    iv = self.img_head(img)
                elif vb == "native_transformer":
                    assert self.patch_embed is not None and self.pos_vis is not None
                    toks = self.patch_embed(img) + self.pos_vis
                    if self.enc_vis is not None:
                        iv = self.enc_vis(toks).mean(dim=1)
                    else:
                        iv = toks.mean(dim=1)
                else:
                    assert vb == "hf_transformer"
                    if self._hf_vis_backbone is None or self.hf_vis_proj is None or not self.use_image_head:
                        iv = float_h.new_zeros(float_h.shape[0], 0)
                    else:
                        vis = self._embed_hf_sequence(img)[:, 0]
                        if self.vis_refine is not None:
                            vis = self.vis_refine(vis.unsqueeze(1)).squeeze(1)
                        iv = vis
                fused = torch.cat([iv, float_h], dim=-1)
                b = fused.shape[0]
                sl = self.pos_pc.shape[1]
                toks = self.fused_to_seq(fused).view(b, sl, self._fuse_d_model) + self.pos_pc
                pooled = self._pool_fusion_tokens(toks)
                h = self.bridge(pooled)
            else:
                fh = z if self.float_feature_extractor is None else float_h
                assert fh is not None
                b = fh.shape[0]
                if self._float_token_layout == "per_feature":
                    assert self.float_scalar_to_tok is not None and self.float_per_feat_slot_emb is not None
                    f = self._post_concat_n_float
                    if fh.shape[1] != f:
                        raise RuntimeError(
                            f"float_input dim {fh.shape[1]} != per_feature token count {f} "
                            "(encoder.float_token_layout per_feature uses float_input_dim tokens)"
                        )
                    ftoks = self.float_scalar_to_tok(fh.unsqueeze(-1)) + self.float_per_feat_slot_emb
                else:
                    assert self.float_to_tokens is not None
                    k = self._post_concat_n_float
                    ftoks = self.float_to_tokens(fh).view(b, k, self._fuse_d_model)
                vis_t = self._post_concat_vis_tokens_token_sequence(img, fh)
                seq = torch.cat([vis_t, ftoks], dim=1) + self.pos_pc
                pooled = self._pool_fusion_tokens(seq)
                h = self.bridge(pooled)
        else:
            assert self.float_to_tokens is not None and self.pos_uni is not None
            assert self.bridge is not None
            vb = self._vis_branch
            uf = self.float_to_tokens(z).view(z.shape[0], -1, self._fuse_d_model)
            if self._unified_n_img == 0 or not self.use_image_head or vb == "none":
                seq = uf + self.pos_uni
            elif vb == "cnn":
                assert self.img_head is not None and self.cnn_to_fuse is not None
                img_tok = self.cnn_to_fuse(self.img_head(img)).unsqueeze(1)
                seq = torch.cat([img_tok, uf], dim=1) + self.pos_uni
            elif vb == "native_transformer":
                assert self.patch_embed is not None
                img_tok = self.patch_embed(img)
                seq = torch.cat([img_tok, uf], dim=1) + self.pos_uni
            else:
                assert vb == "hf_transformer"
                if self._hf_vis_backbone is None or self.hf_vis_proj is None:
                    seq = uf + self.pos_uni
                else:
                    img_tok = self._embed_hf_sequence(img)
                    seq = torch.cat([img_tok, uf], dim=1) + self.pos_uni
            pooled = self._pool_fusion_tokens(seq)
            h = self.bridge(pooled)

        return h

    def forward_fusion_hidden(self, img: Tensor, float_inputs: Tensor) -> Tensor:
        """State vector after multimodal fusion (``bridge``), before the shared MLP ``trunk`` / policy heads.

        IQN uses this as the per-state embedding multiplied by quantile features (width
        ``nn.decoder.dense_hidden_dimension``).
        """
        return self._forward_bridge_out(img, float_inputs)

    def forward_features(self, img: Tensor, float_inputs: Tensor) -> Tensor:
        """Trunk output (dense hidden) used by policy and value heads — for BC multi-head wrappers."""
        h = self._forward_bridge_out(img, float_inputs)
        if self.trunk is None:
            return h
        return self.trunk(h)

    def forward(self, img: Tensor, float_inputs: Tensor) -> PolicyOutput:
        if self.policy_head is None or self.value_head is None:
            raise RuntimeError("TorchMultimodalActorCritic: policy heads omitted (backbone-only build; e.g. IQN)")
        h = self.forward_features(img, float_inputs)
        logits = self.policy_head(h)
        v = self.value_head(h)
        return PolicyOutput(logits=logits, value=v)

    def evaluate_actions(
        self, img: Tensor, float_inputs: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, PolicyOutput]:
        out = self.forward(img, float_inputs)
        assert out.logits is not None
        if self.n_actions_per_block <= 1:
            logits = out.logits
            act = actions.reshape(-1)
        else:
            logits = out.logits.reshape(-1, self.n_actions_per_block, self.n_actions)
            act = actions.reshape(-1, self.n_actions_per_block)
        logp, ent = discrete_action_logprob_and_entropy(logits, act)
        return logp, ent, out.value.squeeze(-1), out


def build_multimodal_fusion_from_transformers(
    tr: MultimodalTransformersConfig,
    vis_enc: TransformersConfig,
    *,
    float_inputs_dim: int,
    float_hidden_dim: int,
    dense_hidden_dim: int,
    image_h: int,
    image_w: int,
    use_image_head: bool,
    vis_branch: str,
    float_inputs_mean: np.ndarray,
    float_inputs_std: np.ndarray,
    n_actions: int,
    n_actions_per_block: int,
    vis_cnn_head_kw: Optional[Mapping[str, Any]] = None,
    include_policy_heads: bool = True,
) -> TorchMultimodalActorCritic:
    mode = tr.fusion_mode
    if mode == "none":
        raise ValueError("MultimodalTransformersConfig.fusion_mode must not be none for fusion policy")

    v = vis_enc
    f = tr.transformer

    vb = vis_branch

    fe = infer_fusion_encoder(mode, tr)
    fmlp = tr.fusion_mlp or (FusionMlpEncoderConfig() if fe == "mlp" else None)
    fcnn = tr.fusion_cnn or (FusionCnnEncoderConfig() if fe == "cnn" else None)
    hf_emb_cfg = tr.hf_embedding

    mean = torch.tensor(np.asarray(float_inputs_mean, dtype=np.float32))
    std = torch.tensor(np.asarray(float_inputs_std, dtype=np.float32))

    hf_fus = None
    hf_fus_h = 0
    hf_emb_effective: HfEmbeddingEncoderConfig | None = None
    if fe == "hf_embedding":
        he = hf_emb_cfg
        path = ""
        trust_rc = False
        drop_p = 0.0
        if f.use_hf_backbone and str(f.model_name_or_path or "").strip():
            path = str(f.model_name_or_path).strip()
            trust_rc = bool(f.trust_remote_code)
            drop_p = float(f.hidden_dropout_prob)
        elif he is not None and str(he.model_name_or_path or "").strip():
            path = str(he.model_name_or_path).strip()
            trust_rc = bool(he.trust_remote_code)
            drop_p = float(he.hidden_dropout_prob)
        if not path:
            raise ValueError(
                "HF fusion encoder: set nn.encoder.transformer.use_hf_backbone: true and "
                "nn.encoder.transformer.model_name_or_path (same idea as nn.vis.transformer), "
                "or nn.encoder.hf_embedding.model_name_or_path when fusion_encoder is hf_embedding. "
                "Models need inputs_embeds (e.g. BERT-class)."
            )
        hf_emb_effective = HfEmbeddingEncoderConfig(
            model_name_or_path=path,
            trust_remote_code=trust_rc,
            hidden_dropout_prob=drop_p,
        )
        _, AutoModel = _lazy_import_transformers()
        hf_fus = AutoModel.from_pretrained(path, trust_remote_code=trust_rc)
        hf_fus_h = int(hf_fus.config.hidden_size)

    hf_bb = None
    hf_proc = None
    hf_h = 0
    unified_hf_n = 0
    n_hf_post = 0
    if vb == "hf_transformer" and use_image_head:
        AutoImageProcessor, AutoModel = _lazy_import_transformers()
        hf_proc = AutoImageProcessor.from_pretrained(
            v.model_name_or_path,
            trust_remote_code=v.trust_remote_code,
        )
        hf_bb = AutoModel.from_pretrained(
            v.model_name_or_path,
            trust_remote_code=v.trust_remote_code,
        )
        hf_h = hf_vision_backbone_hidden_size(hf_bb)
        n_img_tok = hf_backbone_num_image_tokens(hf_bb, hf_proc, image_h=image_h, image_w=image_w)
        if mode == "unified":
            unified_hf_n = n_img_tok
        elif (
            mode == "post_concat"
            and tr.post_concat_layout == "token_sequence"
            and v.fusion_tokens == "patch_tokens"
        ):
            n_hf_post = n_img_tok

    model = TorchMultimodalActorCritic(
        mode=mode,
        vis_branch=vb,
        fusion_encoder=fe,
        float_inputs_dim=float_inputs_dim,
        float_hidden_dim=float_hidden_dim,
        dense_hidden_dim=dense_hidden_dim,
        vis_d_model=int(v.d_model),
        vis_nhead=int(v.n_heads),
        vis_nlayers=int(v.n_layers),
        vis_ff_mult=int(v.ff_mult),
        vis_dropout=float(v.dropout),
        vis_patch_size=int(v.patch_size),
        fuse_d_model=int(f.d_model),
        fuse_nhead=int(f.n_heads),
        fuse_nlayers=int(f.n_layers),
        fuse_ff_mult=int(f.ff_mult),
        fuse_dropout=float(f.dropout),
        post_concat_seq_len=int(f.post_concat_seq_len),
        unified_float_tokens=int(f.unified_float_tokens),
        unified_hf_n_tokens=unified_hf_n,
        image_h=image_h,
        image_w=image_w,
        use_image_head=use_image_head,
        float_inputs_mean=mean,
        float_inputs_std=std,
        n_actions=n_actions,
        n_actions_per_block=n_actions_per_block,
        hf_vis_backbone=hf_bb,
        hf_vis_processor=hf_proc,
        hf_vis_hidden_size=hf_h,
        fusion_mlp=fmlp,
        fusion_cnn=fcnn,
        hf_embedding_cfg=hf_emb_effective if fe == "hf_embedding" else None,
        hf_fusion_backbone=hf_fus,
        hf_fusion_hidden_size=hf_fus_h,
        post_concat_layout=str(tr.post_concat_layout),
        vis_fusion_tokens=str(v.fusion_tokens),
        float_token_input=str(tr.float_token_input),
        float_token_layout=str(tr.float_token_layout),
        n_hf_image_tokens_post=int(n_hf_post),
        vis_cnn_head_kw=vis_cnn_head_kw,
        include_policy_heads=bool(include_policy_heads),
    )
    if hf_bb is not None:
        model.set_hf_hidden_dropout(float(v.hidden_dropout_prob))
    return model


def build_multimodal_fusion_uncompiled(
    cfg: Any = None, *, include_policy_heads: bool = True
) -> TorchMultimodalActorCritic:
    c = cfg or get_config()
    t = c.transformers
    vb = infer_vis_branch(c.vis)
    return build_multimodal_fusion_from_transformers(
        t,
        c.vis.transformer or TransformersConfig(),
        float_inputs_dim=int(c.float_input_dim),
        float_hidden_dim=int(c.float_hidden_dim_effective()),
        dense_hidden_dim=int(c.dense_hidden_dimension),
        image_h=int(c.H_downsized),
        image_w=int(c.W_downsized),
        use_image_head=bool(c.use_iqn_image_head),
        vis_branch=vb,
        float_inputs_mean=np.asarray(c.float_inputs_mean, dtype=np.float32),
        float_inputs_std=np.asarray(c.float_inputs_std, dtype=np.float32),
        n_actions=len(c.inputs),
        n_actions_per_block=int(c.n_actions_per_block),
        vis_cnn_head_kw=vis_cnn_head_kw_from_nn_vis(c.vis),
        include_policy_heads=include_policy_heads,
    )


def make_multimodal_fusion_network_pair(jit: bool, is_inference: bool) -> Tuple[nn.Module, nn.Module]:
    cfg = get_config()
    uncompiled = build_multimodal_fusion_uncompiled(cfg)
    mp = (cfg.transformers.init_from_pretrained or "").strip()
    if mp and tr_utilities.skip_multimodal_fusion_hub_init_from_pretrained(cfg):
        mp = ""
    if mp:
        from trackmania_rl.agents.policy_models.rulka_multimodal_fusion_hub import load_fusion_policy_weights_from_hub

        he = cfg.transformers.hf_embedding
        trust_hub = bool(cfg.transformers.transformer.trust_remote_code) or bool(
            he is not None and he.trust_remote_code
        )
        load_fusion_policy_weights_from_hub(
            uncompiled,
            mp,
            trust_remote_code=trust_hub,
        )
    if not jit or not cfg.use_jit:
        model = uncompiled
    else:
        compile_mode = None if "rocm" in torch.__version__ else (
            "max-autotune" if is_inference else "max-autotune-no-cudagraphs"
        )
        model = torch.compile(uncompiled, mode=compile_mode)
    u = uncompiled.to(device="cuda", memory_format=torch.channels_last).train()
    m = model.to(device="cuda", memory_format=torch.channels_last).train()
    return m, u
