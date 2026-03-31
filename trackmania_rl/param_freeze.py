"""Map ``nn.*.freeze`` flags to parameter prefixes and apply ``requires_grad=False``.

Names are matched after stripping a leading ``_orig_mod.`` (``torch.compile``).
Prefix rules: ``"foo."`` matches submodule parameters; ``"foo"`` (no dot) matches
exactly ``foo`` or names starting with ``"foo."``.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from torch import nn

_COMPILE_WRAPPER = "_orig_mod."

# Multimodal / HF vision stem (not fusion ``encoder`` block, not ``trunk`` / heads).
_MULTIMODAL_VISION_STEM_PREFIXES: tuple[str, ...] = (
    "img_head.",
    "backbone.",
    "_hf_vis_backbone.",
    "patch_embed.",
    "enc_vis.",
    "vis_refine.",
    "hf_vis_proj.",
    "cnn_to_vis.",
    "cnn_to_fuse.",
    "postconcat_cnn_to_fuse.",
    "pos_vis",
    "vis_proj.",
)

# Multimodal fusion trunk (``nn.encoder`` side): between vision+float tokens and ``trunk.``.
_MULTIMODAL_FUSION_ENCODER_PREFIXES: tuple[str, ...] = (
    "bridge.",
    "visfloat_to_seq.",
    "pos_vf",
    "vis_to_fuse.",
    "fused_to_seq.",
    "pos_pc",
    "pos_uni",
    "float_to_tokens.",
    "float_scalar_to_tok.",
    "float_per_feat_slot_emb",
    "enc_fusion_native.",
    "fusion_mlp_mod.",
    "fusion_cnn_mod.",
    "hf_fusion_proj_in.",
    "hf_fusion_proj_out.",
    "_hf_fusion_bb.",
)


def strip_compile_wrapper_from_param_name(name: str) -> str:
    return name[len(_COMPILE_WRAPPER) :] if name.startswith(_COMPILE_WRAPPER) else name


def param_name_matches_prefix(normalized_name: str, prefix: str) -> bool:
    if not prefix:
        return False
    if prefix.endswith("."):
        return normalized_name.startswith(prefix)
    return normalized_name == prefix or normalized_name.startswith(prefix + ".")


def param_name_matches_any_prefix(param_name: str, prefixes: Sequence[str]) -> bool:
    n = strip_compile_wrapper_from_param_name(param_name)
    return any(param_name_matches_prefix(n, p) for p in prefixes if p)


def _dedupe_prefixes(prefixes: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in prefixes:
        s = (p or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _iqn_torch_fusion_backbone(cfg: Any) -> bool:
    tf = getattr(cfg, "transformers", None)
    return tf is not None and getattr(tf, "fusion_mode", "none") != "none"


def _iqn_hf_vision_only_backbone(cfg: Any) -> bool:
    if _iqn_torch_fusion_backbone(cfg):
        return False
    vis = getattr(cfg, "vis", None)
    if vis is None or getattr(vis, "no_image", False):
        return False
    vt = getattr(vis, "transformer", None)
    return vt is not None and getattr(vt, "use_hf_backbone", False)


def _iqn_shared_multimodal_actor_backbone(cfg: Any) -> bool:
    return _iqn_torch_fusion_backbone(cfg) or _iqn_hf_vision_only_backbone(cfg)


def collect_frozen_prefixes(
    cfg: Any, *, wiring_algorithm: Literal["iqn", "ppo"]
) -> list[str]:
    """Collect prefixes from ``nn.vis`` / ``nn.float`` / ``nn.encoder`` / ``nn.iqn`` / ``nn.decoder``."""
    parts: list[str] = []

    vis = getattr(cfg, "vis", None)
    if vis is not None and getattr(vis, "freeze", False) and not getattr(vis, "no_image", False):
        if wiring_algorithm == "iqn":
            if _iqn_shared_multimodal_actor_backbone(cfg):
                parts.extend(f"fusion.{p}" for p in _MULTIMODAL_VISION_STEM_PREFIXES)
            else:
                parts.append("img_head.")
        else:
            parts.extend(_MULTIMODAL_VISION_STEM_PREFIXES)

    fb = getattr(cfg, "float_branch", None)
    if fb is not None and getattr(fb, "freeze", False):
        if wiring_algorithm == "iqn":
            if _iqn_shared_multimodal_actor_backbone(cfg):
                parts.append("fusion.float_feature_extractor.")
                if _iqn_hf_vision_only_backbone(cfg):
                    parts.append("fusion.float_to_hidden.")
            else:
                parts.append("float_feature_extractor.")
        else:
            parts.append("float_feature_extractor.")
            parts.append("float_to_hidden.")

    enc = getattr(cfg, "encoder", None)
    if enc is not None and getattr(enc, "freeze", False):
        if wiring_algorithm == "ppo":
            parts.extend(_MULTIMODAL_FUSION_ENCODER_PREFIXES)
        elif wiring_algorithm == "iqn" and _iqn_torch_fusion_backbone(cfg):
            parts.extend(f"fusion.{p}" for p in _MULTIMODAL_FUSION_ENCODER_PREFIXES)

    iqn = getattr(cfg, "iqn", None)
    if iqn is not None and getattr(iqn, "freeze", False) and wiring_algorithm == "iqn":
        parts.append("iqn_fc.")

    dec = getattr(cfg, "decoder", None)
    if dec is not None:
        adv = getattr(dec, "advantage", None)
        val = getattr(dec, "value", None)
        if wiring_algorithm == "iqn":
            if adv is not None and getattr(adv, "freeze", False):
                parts.append("A_head.")
                # Multi-action (n_actions_per_block > 1): final linear is ``A_head_multi``, not inside ``A_head`` Sequential.
                parts.append("A_head_multi.")
            if val is not None and getattr(val, "freeze", False):
                parts.append("V_head.")
        else:
            if adv is not None and getattr(adv, "freeze", False):
                parts.append("policy_head.")
            if val is not None and getattr(val, "freeze", False):
                parts.append("value_head.")
            if getattr(dec, "shared_trunk_freeze", False):
                parts.append("trunk.")

    return _dedupe_prefixes(parts)


def prefixes_that_match_module(module: nn.Module, prefixes: Sequence[str]) -> list[str]:
    """Return which of ``prefixes`` match at least one ``named_parameters()`` name.

    ``vis.freeze`` for multimodal policies supplies a union of CNN / ViT / HF prefixes; only names present in
    this run appear here — useful for logs (avoid implying unused ``cnn_to_vis`` etc.).
    """
    if not prefixes:
        return []
    names = [n for n, _ in module.named_parameters()]
    return [p for p in prefixes if any(param_name_matches_any_prefix(n, (p,)) for n in names)]


def apply_frozen_prefixes(module: nn.Module, prefixes: Sequence[str]) -> int:
    if not prefixes:
        return 0
    n = 0
    for name, param in module.named_parameters():
        if param_name_matches_any_prefix(name, prefixes):
            param.requires_grad_(False)
            n += 1
    return n
