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

# Validate only (no writing; no GPU required):
python scripts/init_iqn_from_encoder.py \
    --encoder-pt  pretrain_visual_out/encoder.pt \
    --dry-run

# Same topology as training (not default.yaml):
python scripts/init_iqn_from_encoder.py \
    --encoder-pt  pretrain_visual_out/encoder.pt \
    --save-dir save/my_run/ \
    --rl-config config_files/rl/config_btr.yaml

Notes
-----
* CUDA is required when writing checkpoints (IQN lives on GPU). ``--dry-run`` skips CUDA.
* For multi-channel encoders (--stack-mode channel, n_stack > 1), the first
  Conv2d layer kernels are averaged across input channels to produce a 1-ch
  weight compatible with IQN's img_head.  A warning is printed when this happens.
* The optimizer checkpoint (optimizer1.torch) is NOT written by this script;
  the learner starts with a fresh optimizer when no optimizer checkpoint exists.
* Run this script from the project root so that config_files/ is on the Python path.
* RL topology (IQN shape) comes from --rl-config (default: config_files/rl/config_default.yaml).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RL_CONFIG = _REPO_ROOT / "config_files" / "rl" / "config_default.yaml"


def _resolve_rl_config(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    return _DEFAULT_RL_CONFIG.resolve()


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
        from trackmania_rl.pretrain.export import validate_encoder_compatibility
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
    from trackmania_rl.pretrain.export import average_first_layer_to_1ch
    return average_first_layer_to_1ch(state_dict)


def _build_iqn_pair() -> tuple:
    """Create a fresh (online, target) IQN network pair on CUDA (same factory as RL train/export)."""
    from trackmania_rl.agents.algorithms.registry import get_wiring

    w = get_wiring("iqn")
    online, _ = w.make_network(False, False)
    target, _ = w.make_network(False, False)
    return online, target


def _load_existing_pair(save_dir: Path) -> tuple | None:
    """Load existing weights from save_dir if they exist."""
    w1 = save_dir / "weights1.torch"
    w2 = save_dir / "weights2.torch"
    if not (w1.exists() and w2.exists()):
        return None

    from trackmania_rl.agents.algorithms.registry import get_wiring

    w = get_wiring("iqn")
    online, _ = w.make_network(False, False)
    target, _ = w.make_network(False, False)
    online.load_state_dict(torch.load(w1, weights_only=False))
    target.load_state_dict(torch.load(w2, weights_only=False))
    log.info("Loaded existing IQN checkpoints from %s", save_dir)
    return online, target


def _inject_encoder(networks: tuple, encoder_sd: dict) -> None:
    """Load encoder_sd into CNN img_head of both online and target networks."""
    from trackmania_rl.pretrain.export import iqn_cnn_img_head_module

    online, target = networks
    encoder_sd_cuda = {k: v.to("cuda") for k, v in encoder_sd.items()}
    for name, net in (("online", online), ("target", target)):
        head = iqn_cnn_img_head_module(net)
        if head is None:
            log.error(
                "No CNN img_head on %s network (HF vision or float-only). encoder.pt targets fusion CNN only.",
                name,
            )
            sys.exit(1)
        head.load_state_dict(encoder_sd_cuda, strict=True)
    log.info("Injected encoder weights into online and target CNN img_head.")


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
    ap.add_argument(
        "--rl-config",
        type=Path,
        default=None,
        help="RL YAML (topology for IQN). Default: config_files/rl/config_default.yaml under repo root.",
    )

    args = ap.parse_args()

    rl_path = _resolve_rl_config(args.rl_config)
    if not rl_path.is_file():
        log.error("RL config not found: %s", rl_path)
        sys.exit(1)
    from config_files.config_loader import load_config, set_config

    set_config(load_config(rl_path))
    log.info("Loaded RL config: %s", rl_path)

    # 1. Load artifact (CPU) and validate — no CUDA required for --dry-run
    state_dict, meta = _load_artifact(args.encoder_pt, args.meta_json)
    _validate(state_dict, meta)

    if args.dry_run:
        log.info("Dry-run complete.  No files written.")
        return

    _check_cuda()

    # 2. Average first layer if multi-channel
    state_dict = _maybe_avg_to_1ch(state_dict, meta)

    # 3. Build or load IQN network pair
    networks = None
    if not args.no_fresh:
        log.info("Creating fresh IQN network pair.")
        networks = _build_iqn_pair()
    else:
        networks = _load_existing_pair(args.save_dir)
        if networks is None:
            log.info("No existing checkpoints found in %s; creating fresh pair.", args.save_dir)
            networks = _build_iqn_pair()

    # 4. Inject encoder
    _inject_encoder(networks, state_dict)

    # 5. Save
    _save(networks, args.save_dir)


if __name__ == "__main__":
    main()
