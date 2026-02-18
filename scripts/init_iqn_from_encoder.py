"""
Initialize IQN checkpoint weights from a pretrained visual encoder (Level 0).

This script loads a pretrained encoder artifact (encoder.pt + pretrain_meta.json),
injects the encoder weights into a fresh (or existing) IQN network pair, and writes
weights1.torch / weights2.torch to the specified save directory.

The learner (learner_process.py) will then start training from these weights,
with the img_head already initialized from the self-supervised pretrain.

Usage
-----
# Fresh IQN + pretrained encoder (most common):
python scripts/init_iqn_from_encoder.py \
    --encoder-pt  pretrain_visual_out/encoder.pt \
    --save-dir    save/

# Inject into existing checkpoint (replace img_head only):
python scripts/init_iqn_from_encoder.py \
    --encoder-pt  pretrain_visual_out/encoder.pt \
    --save-dir    save/ \
    --no-fresh

# Validate only (no writing):
python scripts/init_iqn_from_encoder.py \
    --encoder-pt  pretrain_visual_out/encoder.pt \
    --dry-run

Notes
-----
* Requires CUDA (same as RL training).
* For multi-channel encoders (--stack-mode channel, n_stack > 1), the first
  Conv2d layer kernels are averaged across input channels to produce a 1-ch
  weight compatible with IQN's img_head.  A warning is printed when this happens.
* The optimizer checkpoint (optimizer1.torch) is NOT written by this script;
  the learner starts with a fresh optimizer when no optimizer checkpoint exists.
* Run this script from the project root so that config_files/ is on the Python path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _check_cuda() -> None:
    if not torch.cuda.is_available():
        log.error(
            "CUDA is not available.  IQN training requires CUDA; "
            "the IQN_Network constructor always places tensors on 'cuda'."
        )
        sys.exit(1)


def _load_artifact(encoder_pt: Path, meta_json: Path | None) -> tuple[dict, dict | None]:
    """Load encoder state dict and optional metadata."""
    if not encoder_pt.exists():
        log.error("encoder.pt not found: %s", encoder_pt)
        sys.exit(1)

    state_dict = torch.load(encoder_pt, map_location="cpu", weights_only=True)
    log.info("Loaded encoder: %s  (%d tensors)", encoder_pt, len(state_dict))

    meta: dict | None = None
    # Prefer explicit --meta-json; fall back to sibling pretrain_meta.json
    if meta_json is None:
        candidate = encoder_pt.parent / "pretrain_meta.json"
        if candidate.exists():
            meta_json = candidate

    if meta_json is not None and meta_json.exists():
        import json
        with open(meta_json, encoding="utf-8") as fh:
            meta = json.load(fh)
        log.info("Loaded metadata: task=%s  image_size=%s  in_channels=%s",
                 meta.get("task"), meta.get("image_size"), meta.get("in_channels"))
    else:
        log.info("No pretrain_meta.json found; skipping compatibility validation.")

    return state_dict, meta


def _validate(state_dict: dict, meta: dict | None) -> None:
    """Run compatibility check if metadata is available."""
    if meta is None:
        return
    try:
        from trackmania_rl.pretrain_visual.export import validate_encoder_compatibility
        validate_encoder_compatibility(state_dict, meta, strict=False)
        log.info("Compatibility check passed.")
    except Exception as exc:
        log.warning("Compatibility check warning: %s", exc)


def _maybe_avg_to_1ch(state_dict: dict, meta: dict | None) -> dict:
    """Average first Conv2d layer from N channels → 1 if encoder is multi-channel."""
    in_channels = 1
    if meta is not None:
        in_channels = meta.get("in_channels", 1)
    if in_channels == 1:
        return state_dict

    log.warning(
        "Encoder has %d input channels (stack_mode=channel). "
        "Averaging first Conv2d kernels to 1 channel for IQN compatibility.",
        in_channels,
    )
    from trackmania_rl.pretrain_visual.export import average_first_layer_to_1ch
    return average_first_layer_to_1ch(state_dict)


def _build_iqn_pair() -> tuple:
    """Create a fresh (online, target) IQN network pair on CUDA."""
    from trackmania_rl.agents.iqn import make_untrained_iqn_network
    online, _ = make_untrained_iqn_network(jit=False, is_inference=False)
    target, _ = make_untrained_iqn_network(jit=False, is_inference=False)
    return online, target


def _load_existing_pair(save_dir: Path) -> tuple | None:
    """Load existing weights from save_dir if they exist."""
    w1 = save_dir / "weights1.torch"
    w2 = save_dir / "weights2.torch"
    if not (w1.exists() and w2.exists()):
        return None

    from trackmania_rl.agents.iqn import make_untrained_iqn_network
    online, _ = make_untrained_iqn_network(jit=False, is_inference=False)
    target, _ = make_untrained_iqn_network(jit=False, is_inference=False)
    online.load_state_dict(torch.load(w1, weights_only=False))
    target.load_state_dict(torch.load(w2, weights_only=False))
    log.info("Loaded existing IQN checkpoints from %s", save_dir)
    return online, target


def _inject_encoder(networks: tuple, encoder_sd: dict) -> None:
    """Load encoder_sd into img_head of both online and target networks."""
    online, target = networks
    # Move to CUDA (same device as IQN)
    encoder_sd_cuda = {k: v.to("cuda") for k, v in encoder_sd.items()}
    online.img_head.load_state_dict(encoder_sd_cuda, strict=True)
    target.img_head.load_state_dict(encoder_sd_cuda, strict=True)
    log.info("Injected encoder weights into online and target img_head.")


def _save(networks: tuple, save_dir: Path) -> None:
    """Write weights1.torch and weights2.torch."""
    online, target = networks
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(online.state_dict(), save_dir / "weights1.torch")
    torch.save(target.state_dict(), save_dir / "weights2.torch")
    log.info("Saved IQN checkpoints → %s", save_dir)
    log.info("  weights1.torch  (online  network with pretrained img_head)")
    log.info("  weights2.torch  (target  network with pretrained img_head)")
    log.info("Start the learner normally; it will load these checkpoints automatically.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inject pretrained visual encoder into IQN checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--encoder-pt", type=Path, required=True,
                    help="Path to encoder.pt produced by pretrain_visual_backbone.py.")
    ap.add_argument("--save-dir", type=Path, default=Path("save"),
                    help="Directory where weights1.torch / weights2.torch will be written. "
                         "This is the same directory the learner reads from.")
    ap.add_argument("--meta-json", type=Path, default=None,
                    help="Path to pretrain_meta.json.  Auto-detected from encoder-pt parent if omitted.")
    ap.add_argument("--no-fresh", action="store_true",
                    help="If set and existing checkpoints are found in --save-dir, load them first "
                         "and only replace img_head (preserving other layer weights).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate encoder compatibility only; do not write any files.")

    args = ap.parse_args()

    _check_cuda()

    # 1. Load artifact
    state_dict, meta = _load_artifact(args.encoder_pt, args.meta_json)

    # 2. Validate
    _validate(state_dict, meta)

    if args.dry_run:
        log.info("Dry-run complete.  No files written.")
        return

    # 3. Average first layer if multi-channel
    state_dict = _maybe_avg_to_1ch(state_dict, meta)

    # 4. Build or load IQN network pair
    networks = None
    if not args.no_fresh:
        log.info("Creating fresh IQN network pair.")
        networks = _build_iqn_pair()
    else:
        networks = _load_existing_pair(args.save_dir)
        if networks is None:
            log.info("No existing checkpoints found in %s; creating fresh pair.", args.save_dir)
            networks = _build_iqn_pair()

    # 5. Inject encoder
    _inject_encoder(networks, state_dict)

    # 6. Save
    _save(networks, args.save_dir)


if __name__ == "__main__":
    main()
