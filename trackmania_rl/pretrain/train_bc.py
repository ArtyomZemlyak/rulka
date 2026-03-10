"""
Level 1 BC (behavioral cloning) training.

Single entry: train_bc(cfg). Uses PyTorch Lightning only. BCPretrainConfig,
CachedBCDataModule or BCReplayDataModule, BCLightningModule, create_lightning_trainer,
save_encoder_artifact. Reuses Lightning setup from Level 0 (train.py).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from trackmania_rl.pretrain.export import load_encoder_into_bc, save_encoder_artifact
from trackmania_rl.pretrain.models import build_bc_network, get_enc_dim
from trackmania_rl.pretrain.preprocess import (
    build_bc_cache,
    build_bc_cache_floats_only,
    is_bc_cache_valid,
    is_bc_cache_valid_except_float_signature,
)
from trackmania_rl.pretrain.train import create_lightning_trainer

log = logging.getLogger(__name__)


def _resolve_run_dir(base_dir: Path, run_name: str | None) -> Path:
    """Return versioned run directory: base_dir/run_name or base_dir/run_001, run_002, ..."""
    if run_name:
        return base_dir / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        candidate = base_dir / f"run_{i:03d}"
        if not candidate.exists():
            return candidate
        i += 1


def train_bc(cfg) -> Path:
    """Run BC pretraining with PyTorch Lightning.

    Cache check/build, then Lightning DataModule + BCLightningModule,
    create_lightning_trainer (shared with Level 0), fit, save encoder.pt.

    cfg: BCPretrainConfig (from config_files.pretrain_bc_schema).
    Returns run_dir (where encoder.pt and pretrain_meta.json are written).
    """
    run_dir = _resolve_run_dir(Path(cfg.output_dir), cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("BC run directory: %s", run_dir)

    log.info("Training on device: %s", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Unlock Tensor Cores on Ampere / Ada / Blackwell GPUs (same as Level 0)
    prec = cfg.lightning.float32_matmul_precision
    if prec is not None:
        torch.set_float32_matmul_precision(prec)
        log.info("torch.set_float32_matmul_precision('%s')", prec)

    data_dir = Path(cfg.data_dir).resolve()
    track_ids = getattr(cfg, "track_ids", None)
    # Use cache when preprocess_cache_dir is set (build/validate for selected track_ids when provided).
    cache_dir = Path(cfg.preprocess_cache_dir) if cfg.preprocess_cache_dir else None

    # Load RL config (single source for all RL-matching params)
    from config_files.config_loader import load_config, set_config, get_config
    rl_path = Path(cfg.rl_config_path).resolve()
    if not rl_path.exists():
        raise FileNotFoundError(f"rl_config_path {rl_path} not found")
    set_config(load_config(rl_path))
    rl_cfg = get_config()

    image_size = rl_cfg.w_downsized
    n_actions = len(rl_cfg.inputs)
    dense_hidden_dimension = rl_cfg.dense_hidden_dimension
    iqn_embedding_dimension = rl_cfg.iqn_embedding_dimension

    if cache_dir is not None:
        use_floats = cfg.use_floats or cfg.use_full_iqn
        cache_valid = is_bc_cache_valid(
            cache_dir,
            data_dir,
            image_size,
            cfg.n_stack,
            cfg.val_fraction,
            cfg.seed,
            n_actions,
            cfg.bc_target,
            cfg.bc_time_offsets_ms,
            use_floats=use_floats,
            floats_config=rl_cfg,
            track_ids=track_ids,
        )
        if not cache_valid:
            base_valid = is_bc_cache_valid_except_float_signature(
                cache_dir, data_dir, image_size, cfg.n_stack,
                cfg.val_fraction, cfg.seed, n_actions,
                cfg.bc_target, cfg.bc_time_offsets_ms,
                track_ids=track_ids,
            )
            if base_valid and use_floats and rl_cfg is not None:
                log.info("BC cache base valid but float_config_signature changed. Rebuilding floats only...")
                build_bc_cache_floats_only(cache_dir, data_dir, rl_cfg)
            else:
                log.info("BC cache invalid or missing.")
                log.info("Building BC cache at %s ...", cache_dir)
                build_bc_cache(
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    image_size=image_size,
                    n_stack=cfg.n_stack,
                    val_fraction=cfg.val_fraction,
                    seed=cfg.seed,
                    n_actions=n_actions,
                    workers=cfg.cache_build_workers,
                    bc_target=cfg.bc_target,
                    bc_time_offsets_ms=cfg.bc_time_offsets_ms,
                    bc_offset_weights=cfg.bc_offset_weights,
                    floats_config=rl_cfg,
                    track_ids=track_ids,
                )
        elif use_floats and rl_cfg is not None:
            meta_path = Path(cache_dir) / "cache_meta.json"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if not meta.get("has_floats"):
                    log.info("BC cache valid but floats missing. Building float inputs...")
                    build_bc_cache_floats_only(cache_dir, data_dir, rl_cfg)

    enc_dim = get_enc_dim(1, image_size)
    float_dim = int(rl_cfg.float_input_dim) if (cfg.use_floats or cfg.use_full_iqn) else 0

    if cfg.use_full_iqn:
        from trackmania_rl.pretrain.models import build_iqn_for_bc, IQN_BC_MultiOffset
        iqn = build_iqn_for_bc(
            image_size=image_size,
            float_inputs_dim=float_dim,
            float_hidden_dim=rl_cfg.float_hidden_dim,
            n_actions=n_actions,
            dense_hidden_dimension=dense_hidden_dimension,
            iqn_embedding_dimension=iqn_embedding_dimension,
            float_inputs_mean=rl_cfg.float_inputs_mean.tolist(),
            float_inputs_std=rl_cfg.float_inputs_std.tolist(),
        )
        n_offsets = len(cfg.bc_time_offsets_ms)
        if n_offsets > 1:
            model = IQN_BC_MultiOffset(iqn, n_offsets)
        else:
            model = iqn
    else:
        model = build_bc_network(
            enc_dim=enc_dim,
            n_actions=n_actions,
            n_offsets=len(cfg.bc_time_offsets_ms),
            use_floats=cfg.use_floats,
            float_dim=float_dim,
            float_hidden_dim=rl_cfg.float_hidden_dim if rl_cfg else 256,
            float_inputs_mean=rl_cfg.float_inputs_mean.tolist() if rl_cfg else None,
            float_inputs_std=rl_cfg.float_inputs_std.tolist() if rl_cfg else None,
            use_actions_head=cfg.use_actions_head,
            dense_hidden_dimension=dense_hidden_dimension,
            dropout=getattr(cfg, "dropout", 0.0),
            action_head_dropout=getattr(cfg, "action_head_dropout", 0.0),
            in_channels=1,
            image_size=image_size,
        )

    # Load full BC model from a previous run (fine-tune) or only encoder (init)
    bc_resume_dir = Path(cfg.bc_resume_run_dir) if getattr(cfg, "bc_resume_run_dir", None) else None
    if bc_resume_dir and bc_resume_dir.exists() and cfg.use_full_iqn:
        # Prefer Lightning .ckpt (has full state including all A_heads); fallback to iqn_bc.pt
        ckpt_dir = bc_resume_dir / "checkpoints"
        ckpt_path = None
        if ckpt_dir.is_dir():
            ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpts:
                ckpt_path = ckpts[0]
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("state_dict", ckpt)
            # Lightning saves BCLightningModule: keys are "model.img_head...", "model.A_heads.0..."
            prefix = "model."
            if any(k.startswith(prefix) for k in state_dict):
                state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            model.load_state_dict(state_dict, strict=False)
            missing = set(model.state_dict().keys()) - set(state_dict.keys())
            if missing:
                log.warning("bc_resume (ckpt): keys not in checkpoint: %s", sorted(missing)[:10])
            log.info("Loaded full BC model from Lightning checkpoint %s for fine-tuning", ckpt_path)
        else:
            iqn_pt = bc_resume_dir / "iqn_bc.pt"
            if not iqn_pt.exists():
                raise FileNotFoundError(f"bc_resume_run_dir {bc_resume_dir} has no checkpoints/*.ckpt nor iqn_bc.pt")
            state_dict = torch.load(iqn_pt, map_location="cpu", weights_only=True)
            # iqn_bc.pt uses A_head.* and optionally A_head_offset_N.*; model has A_heads.0.*, A_heads.1.*, ...
            remapped = {}
            prefix_ah = "A_head_offset_"
            for k, v in state_dict.items():
                if k.startswith("A_head.") and not k.startswith(prefix_ah):
                    remapped["A_heads.0." + k[7:]] = v
                elif k.startswith(prefix_ah):
                    rest_after = k[len(prefix_ah) :]
                    dot = rest_after.index(".")
                    idx, rest = rest_after[:dot], rest_after[dot + 1 :]
                    remapped[f"A_heads.{idx}.{rest}"] = v
                else:
                    remapped[k] = v
            model.load_state_dict(remapped, strict=False)
            missing = set(model.state_dict().keys()) - set(remapped.keys())
            # If checkpoint had only A_head (old run with merge_all_heads=False), copy A_heads.0 to other heads
            head0_prefix = "A_heads.0."
            to_fill = [k for k in missing if k.startswith("A_heads.") and not k.startswith(head0_prefix)]
            if to_fill and hasattr(model, "A_heads") and isinstance(model.A_heads, nn.ModuleList):
                state = model.state_dict()
                for key in to_fill:
                    rest = key.split(".", 2)[-1]
                    src_key = head0_prefix + rest
                    if src_key in state:
                        state[key].copy_(state[src_key])
                model.load_state_dict(state, strict=False)
                n_heads_filled = len(set(k.split(".", 2)[1] for k in to_fill))
                log.info("bc_resume: copied A_heads.0 to %d missing offset heads (iqn_bc.pt had single head)", n_heads_filled)
                missing = missing - set(to_fill)
            if missing:
                log.warning("bc_resume: keys not in checkpoint (left as init): %s", sorted(missing)[:10])
            log.info("Loaded full BC model from %s for fine-tuning", iqn_pt)
    elif cfg.encoder_init_path and Path(cfg.encoder_init_path).exists():
        if cfg.use_full_iqn:
            from trackmania_rl.pretrain.export import average_first_layer_to_1ch
            from trackmania_rl.pretrain.contract import META_FILE
            enc_pt = Path(cfg.encoder_init_path)
            state_dict = torch.load(enc_pt, map_location="cpu", weights_only=True)
            meta_path = enc_pt.parent / META_FILE
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                if meta.get("in_channels", 1) != 1:
                    state_dict = average_first_layer_to_1ch(state_dict)
            model.img_head.load_state_dict(state_dict, strict=True)
            log.info("Loaded encoder init into IQN img_head from %s", enc_pt)
        else:
            load_encoder_into_bc(Path(cfg.encoder_init_path), model)

    from trackmania_rl.pretrain.tasks import BCLightningModule
    from trackmania_rl.pretrain.datasets import CachedBCDataModule, BCReplayDataModule

    bc_module = BCLightningModule(
        model,
        lr=cfg.lr,
        weight_decay=getattr(cfg, "weight_decay", 0.0),
        use_full_iqn=cfg.use_full_iqn,
        full_iqn_random_tau=cfg.full_iqn_random_tau,
        float_dim=float_dim,
        n_offsets=len(cfg.bc_time_offsets_ms),
        offset_weights=cfg.bc_offset_weights,
        bc_time_offsets_ms=cfg.bc_time_offsets_ms if len(cfg.bc_time_offsets_ms) > 1 else None,
    )

    if cache_dir is not None:
        log.info("BC DataModule: using preprocessed cache at %s", cache_dir)
        data_module = CachedBCDataModule(
            cache_dir=cache_dir,
            batch_size=cfg.batch_size,
            workers=cfg.workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            load_in_ram=cfg.cache_load_in_ram,
            expected_image_size=image_size,
            expected_n_stack=cfg.n_stack,
            image_normalization=cfg.image_normalization,
        )
    else:
        if track_ids is not None:
            log.info("BC DataModule: on-the-fly from data_dir (track_ids=%s)", track_ids)
        data_module = BCReplayDataModule(
            data_dir=data_dir,
            image_size=image_size,
            n_stack=cfg.n_stack,
            batch_size=cfg.batch_size,
            workers=cfg.workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
            bc_target=cfg.bc_target,
            bc_time_offsets_ms=cfg.bc_time_offsets_ms,
            bc_offset_weights=cfg.bc_offset_weights,
            image_normalization=cfg.image_normalization,
            track_ids=track_ids,
        )

    data_module.setup()
    n_train = data_module.n_train_samples
    n_val = data_module.n_val_samples

    # Confirm model + dataset alignment after cache build (user-visible checkpoint)
    if cache_dir is not None and (cfg.use_floats or cfg.use_full_iqn):
        meta_path = Path(cache_dir) / "cache_meta.json"
        has_floats = False
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                has_floats = json.load(f).get("has_floats", False)
        if float_dim > 0 and has_floats:
            log.info(
                "BC setup complete: model and dataset both use float inputs (float_input_dim=%d). Ready to train.",
                float_dim,
            )
        elif float_dim > 0 and not has_floats:
            log.warning(
                "BC setup: model expects floats (dim=%d) but cache has no float inputs. Training may fail.",
                float_dim,
            )

    n_train_batches = len(data_module.train_dataloader())
    trainer, metrics_cb, _ = create_lightning_trainer(
        cfg.lightning,
        run_dir,
        max_epochs=cfg.epochs,
        grad_clip=cfg.grad_clip,
        use_tqdm=cfg.use_tqdm,
        has_val=data_module.has_val,
        n_train_batches=n_train_batches,
    )

    try:
        trainer.fit(bc_module, datamodule=data_module)
    except KeyboardInterrupt:
        log.warning(
            "Training interrupted by user (Ctrl+C). "
            "%d epochs completed — saving partial encoder artifact.",
            len(metrics_cb.rows),
        )

    # Build metrics_rows: all callback metrics (loss, overall acc, per-class acc)
    metrics_rows = []
    for row in metrics_cb.rows:
        out = {k: v for k, v in row.items()}
        out["epoch"] = row.get("epoch", 0) + 1
        metrics_rows.append(out)

    backbone = bc_module.encoder
    meta = {
        "task": "bc",
        "framework": "lightning",
        "bc_mode": cfg.bc_mode,
        "bc_target": cfg.bc_target,
        "bc_time_offsets_ms": cfg.bc_time_offsets_ms,
        "image_size": image_size,
        "image_normalization": cfg.image_normalization,
        "n_stack": cfg.n_stack,
        "in_channels": 1,
        "enc_dim": enc_dim,
        "n_actions": n_actions,
        "encoder_init_path": str(cfg.encoder_init_path) if cfg.encoder_init_path else None,
        "epochs_trained": cfg.epochs,
        "train_loss_final": metrics_rows[-1].get("train_loss") if metrics_rows else None,
        "val_loss_final": metrics_rows[-1].get("val_loss") if metrics_rows else None,
        "train_acc_final": metrics_rows[-1].get("train_acc") if metrics_rows else None,
        "val_acc_final": metrics_rows[-1].get("val_acc") if metrics_rows else None,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "seed": cfg.seed,
    }
    if cfg.bc_offset_weights is not None:
        meta["bc_offset_weights"] = cfg.bc_offset_weights
    if cfg.use_full_iqn:
        meta["use_full_iqn"] = True
        meta["full_iqn_random_tau"] = cfg.full_iqn_random_tau
        meta["dense_hidden_dimension"] = dense_hidden_dimension
        meta["iqn_embedding_dimension"] = iqn_embedding_dimension
        full_iqn_path = run_dir / "iqn_bc.pt"
        n_offsets = len(cfg.bc_time_offsets_ms)
        if n_offsets > 1 and hasattr(model, "state_dict_for_iqn_transfer"):
            merge_all = getattr(cfg, "merge_actions_head", False)
            torch.save(model.state_dict_for_iqn_transfer(merge_all_heads=merge_all), full_iqn_path)
            if merge_all:
                meta["merge_actions_head"] = True
        else:
            torch.save(model.state_dict(), full_iqn_path)
        meta["full_iqn_file"] = "iqn_bc.pt"
        log.info("Saved full IQN state dict → %s (for RL transfer)", full_iqn_path)
    if cfg.use_floats and cfg.save_float_head and getattr(model, "float_head", None) is not None:
        float_head_path = run_dir / "float_head.pt"
        torch.save(model.float_head.state_dict(), float_head_path)
        meta["float_head_file"] = "float_head.pt"
        meta["float_hidden_dim"] = rl_cfg.float_hidden_dim
        log.info("Saved float head → %s", float_head_path)
    if cfg.use_actions_head and cfg.save_actions_head and getattr(model, "action_head", None) is not None:
        actions_head_path = run_dir / "actions_head.pt"
        ah = model.action_head
        n_offsets = len(cfg.bc_time_offsets_ms)
        merge_all = getattr(cfg, "merge_actions_head", False)

        def _action_head_state_for_iqn(module: nn.Module) -> dict:
            """State dict for IQN A_head: if head has dropout (keys 0,3), remap to (0,2)."""
            sd = module.state_dict()
            if "3.weight" in sd and "3.bias" in sd:
                return {"0.weight": sd["0.weight"], "0.bias": sd["0.bias"], "2.weight": sd["3.weight"], "2.bias": sd["3.bias"]}
            return sd

        if merge_all and n_offsets > 1 and isinstance(ah, nn.ModuleList):
            state = {f"offset_{i}": _action_head_state_for_iqn(ah[i]) for i in range(len(ah))}
            meta["actions_head_merged"] = True
            meta["actions_head_offsets"] = n_offsets
        else:
            if isinstance(ah, nn.ModuleList):
                state = _action_head_state_for_iqn(ah[0])
            else:
                state = _action_head_state_for_iqn(ah)
        torch.save(state, actions_head_path)
        meta["actions_head_file"] = "actions_head.pt"
        meta["dense_hidden_dimension"] = dense_hidden_dimension
        if merge_all and n_offsets > 1 and isinstance(ah, nn.ModuleList):
            log.info("Saved actions head (all %d offsets merged) → %s", len(ah), actions_head_path)
        else:
            log.info("Saved actions head (offset 0, IQN A_head layout) → %s", actions_head_path)
    save_encoder_artifact(backbone, meta, run_dir, metrics_rows=metrics_rows)
    log.info("BC pretrain complete. Artifact: %s", run_dir)
    return run_dir
