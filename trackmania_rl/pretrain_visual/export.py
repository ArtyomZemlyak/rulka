"""
Encoder artifact save / load / validate utilities.

The artifact contract is described in contract.py.  This module provides
the functions that write and read that contract on disk.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from trackmania_rl.pretrain_visual.contract import (
    ENCODER_FILE,
    META_FILE,
    METRICS_FILE,
    META_VERSION,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arch hash
# ---------------------------------------------------------------------------

def compute_arch_hash(state_dict: dict) -> str:
    """Compute a short hash from encoder layer names and shapes.

    Used to detect architecture drift (e.g. someone changed the CNN layout
    and tries to load an incompatible encoder into IQN).
    """
    parts = [f"{k}:{tuple(v.shape)}" for k, v in sorted(state_dict.items())]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# First-layer kernel averaging (N-channel → 1-channel for IQN load)
# ---------------------------------------------------------------------------

def average_first_layer_to_1ch(state_dict: dict) -> dict:
    """Average the first Conv2d weight from N input channels down to 1.

    Required when loading an encoder trained with ``stack_mode=channel`` (N-ch)
    into the IQN ``img_head`` which always expects 1-channel input.

    The averaging (rather than selecting channel 0) preserves gradient signal
    from all temporal frames while satisfying the 1-ch constraint.
    """
    new_sd = {k: v.clone() for k, v in state_dict.items()}
    for key, val in new_sd.items():
        if "weight" in key and val.dim() == 4:  # Conv2d: (out, in, kH, kW)
            if val.shape[1] > 1:
                new_sd[key] = val.mean(dim=1, keepdim=True)
                log.info(
                    "Averaged first Conv2d kernels from %d channels → 1 (key: %s)",
                    val.shape[1],
                    key,
                )
            break  # Only the first Conv2d layer needs treatment
    return new_sd


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_encoder_artifact(
    encoder: nn.Module,
    meta: dict[str, Any],
    out_dir: Path,
    metrics_rows: Optional[list[dict]] = None,
) -> None:
    """Write encoder.pt + pretrain_meta.json and optionally metrics.csv.

    Parameters
    ----------
    encoder:
        The backbone module whose ``state_dict()`` is saved.
    meta:
        Dictionary matching the pretrain_meta.json schema (see contract.py).
        ``version``, ``arch_hash``, and ``timestamp`` are added/overwritten here.
    out_dir:
        Output directory (created if needed).
    metrics_rows:
        Optional list of per-epoch dicts with at least ``{"epoch", "train_loss"}``.
        Written as metrics.csv if provided.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = encoder.state_dict()

    # Enrich meta
    meta["version"] = META_VERSION
    meta["arch_hash"] = compute_arch_hash(state_dict)
    meta["timestamp"] = datetime.now(timezone.utc).isoformat()

    # encoder.pt
    torch.save(state_dict, out_dir / ENCODER_FILE)
    log.info("Saved encoder weights → %s", out_dir / ENCODER_FILE)

    # pretrain_meta.json
    with open(out_dir / META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Saved metadata → %s", out_dir / META_FILE)

    # metrics.csv (optional)
    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        with open(out_dir / METRICS_FILE, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)
        log.info("Saved metrics → %s", out_dir / METRICS_FILE)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_encoder_artifact(
    artifact_dir: Path,
    map_location: Optional[str] = None,
) -> tuple[dict, dict]:
    """Load encoder state dict and metadata from an artifact directory.

    Returns
    -------
    (state_dict, meta)
    """
    artifact_dir = Path(artifact_dir)
    encoder_path = artifact_dir / ENCODER_FILE
    meta_path = artifact_dir / META_FILE

    if not encoder_path.exists():
        raise FileNotFoundError(f"encoder.pt not found in {artifact_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"pretrain_meta.json not found in {artifact_dir}")

    state_dict = torch.load(encoder_path, map_location=map_location, weights_only=True)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    return state_dict, meta


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_encoder_compatibility(
    state_dict: dict,
    meta: dict,
    strict: bool = True,
) -> None:
    """Check that the saved encoder is compatible with IQN_Network.img_head.

    Raises ``ValueError`` if the output dimension does not match what IQN expects
    for the given image size.  When ``in_channels > 1`` warns (or raises if
    ``strict=True``) because the encoder cannot be loaded directly into the 1-ch
    IQN head without prior kernel averaging.

    Parameters
    ----------
    state_dict:
        Encoder state dict as returned by ``load_encoder_artifact``.
    meta:
        Metadata dict as returned by ``load_encoder_artifact``.
    strict:
        If ``True``, raise on multi-channel encoder (caller must call
        ``average_first_layer_to_1ch`` before injecting into IQN).
        If ``False``, only warn.
    """
    from trackmania_rl.pretrain_visual.models import build_encoder_from_meta
    from trackmania_rl.agents.iqn import calculate_conv_output_dim

    image_size = meta["image_size"]
    in_channels = meta["in_channels"]

    # Architecture drift check
    current_hash = compute_arch_hash(state_dict)
    if current_hash != meta.get("arch_hash", current_hash):
        raise ValueError(
            f"Encoder arch_hash mismatch: saved={meta['arch_hash']}, current={current_hash}. "
            "The architecture may have changed since this encoder was saved."
        )

    # Build encoder and verify output shape
    encoder = build_encoder_from_meta(meta)
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    expected_dim = calculate_conv_output_dim(image_size, image_size)
    with torch.no_grad():
        test_out = encoder(torch.zeros(1, in_channels, image_size, image_size))
    actual_dim = test_out.shape[1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Encoder output dim {actual_dim} != IQN expected dim {expected_dim} "
            f"for image_size={image_size}. Pretrain image_size must match IQN config."
        )

    # Multi-channel warning / error
    if in_channels != 1:
        msg = (
            f"Encoder has {in_channels} input channels (stack_mode=channel). "
            "Loading directly into IQN img_head (1-ch) requires kernel-averaging. "
            "Call export.average_first_layer_to_1ch(state_dict) before injection, "
            "or retrain with --stack-mode concat to get a 1-ch encoder."
        )
        if strict:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg, stacklevel=2)

    log.info(
        "Encoder validated: task=%s  image_size=%d  in_channels=%d  enc_dim=%d",
        meta.get("task"),
        image_size,
        in_channels,
        actual_dim,
    )


# ---------------------------------------------------------------------------
# IQN injection helper (used by train.py and init_iqn_from_encoder.py)
# ---------------------------------------------------------------------------

def inject_encoder_into_iqn(
    encoder_pt: Path,
    save_dir: Path,
    *,
    overwrite: bool = False,
) -> bool:
    """Load ``encoder.pt`` and inject its weights into a fresh IQN checkpoint pair.

    Creates ``save_dir/weights1.torch`` and ``save_dir/weights2.torch`` with the
    pretrained ``img_head`` and random weights for all other layers.

    Parameters
    ----------
    encoder_pt:
        Path to ``encoder.pt`` produced by ``pretrain_visual_backbone.py``.
    save_dir:
        Directory where ``weights1.torch`` / ``weights2.torch`` will be written.
        This is the same directory the learner reads from.
    overwrite:
        If *False* (default), skips injection when ``weights1.torch`` already
        exists in ``save_dir`` (prevents overwriting resumed training progress).
        If *True*, always injects regardless of existing files.

    Returns
    -------
    bool
        *True* if injection was performed, *False* if skipped (weights already exist).
    """
    w1 = save_dir / "weights1.torch"
    if w1.exists() and not overwrite:
        log.info(
            "Pretrain injection skipped: %s already exists "
            "(existing training progress preserved; set overwrite=True to force).",
            w1,
        )
        return False

    encoder_pt = Path(encoder_pt)
    if not encoder_pt.exists():
        raise FileNotFoundError(f"encoder.pt not found: {encoder_pt}")

    # Load encoder state dict
    state_dict = torch.load(encoder_pt, map_location="cpu", weights_only=True)
    log.info("Loaded pretrained encoder: %s  (%d tensors)", encoder_pt, len(state_dict))

    # Load optional metadata for compatibility check + multi-channel detection
    meta: dict | None = None
    meta_path = encoder_pt.parent / META_FILE
    if meta_path.exists():
        import json
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        log.info(
            "Pretrain metadata: task=%s  image_size=%s  in_channels=%s",
            meta.get("task"), meta.get("image_size"), meta.get("in_channels"),
        )

    # Average first conv layer from N channels → 1 if needed
    in_channels = meta.get("in_channels", 1) if meta else 1
    if in_channels != 1:
        log.warning(
            "Encoder has %d input channels (stack_mode=channel). "
            "Averaging first Conv2d kernels to 1 channel for IQN img_head.",
            in_channels,
        )
        state_dict = average_first_layer_to_1ch(state_dict)

    # Build fresh IQN network pair and inject
    from trackmania_rl.agents.iqn import make_untrained_iqn_network
    online, _ = make_untrained_iqn_network(jit=False, is_inference=False)
    target, _ = make_untrained_iqn_network(jit=False, is_inference=False)

    encoder_sd_cuda = {k: v.to("cuda") for k, v in state_dict.items()}
    online.img_head.load_state_dict(encoder_sd_cuda, strict=True)
    target.img_head.load_state_dict(encoder_sd_cuda, strict=True)
    log.info(
        "[PRETRAIN] Loaded pretrain weights into IQN img_head (online + target); "
        "%d tensor keys.",
        len(state_dict),
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(online.state_dict(), save_dir / "weights1.torch")
    torch.save(target.state_dict(), save_dir / "weights2.torch")
    log.info(
        "[PRETRAIN] Saved weights1.torch + weights2.torch → %s  "
        "(learner and collectors will load these; img_head is pretrained).",
        save_dir,
    )
    return True
