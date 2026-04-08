"""DPO learner: trajectory preferences vs reference policy; online pairing + optional offline JSONL."""

from __future__ import annotations

import copy
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing.connection import wait
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from config_files.config_loader import get_config
from trackmania_rl import utilities
from trackmania_rl.agents.algorithms import get_wiring
from trackmania_rl.agents.policy_optimization.dpo import dpo_preference_loss, sum_log_probs_evaluate
from trackmania_rl.multiprocess.policy_rollout_batch import build_policy_rollout_tensors, dpo_scheduled_float


def _align_trajectory_batches(
    bw: dict[str, torch.Tensor], bl: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    n = min(bw["actions"].shape[0], bl["actions"].shape[0])
    return {k: v[:n] for k, v in bw.items()}, {k: v[:n] for k, v in bl.items()}


def _offline_pair_paths_iterator(jsonl_path: Path) -> Iterator[tuple[Path, Path]]:
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield Path(rec["chosen"]), Path(rec["rejected"])


def _resolve_path(p: Path, base_dir: Path) -> Path:
    return p if p.is_absolute() else (base_dir / p)


def learner_dpo_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
):
    cfg = get_config()
    wiring = get_wiring()
    device = torch.device("cuda")

    ref_sync_every = max(1, int(cfg.dpo_ref_sync_every_updates))
    pair_buffer_max = max(2, int(cfg.dpo_pair_buffer_max))
    data_mode = str(cfg.dpo_data_mode)
    offline_jsonl = getattr(cfg, "dpo_offline_pairs_jsonl", None) or None
    update_epochs = max(1, int(cfg.dpo_update_epochs))

    policy, uncompiled_local = wiring.make_network(cfg.use_jit, is_inference=False)

    w1_path = save_dir / "weights1.torch"
    loaded_from_file = False
    try:
        if w1_path.exists():
            sd = torch.load(f=w1_path, weights_only=False)
            _slice = bool(getattr(cfg, "pretrain_ppo_policy_slice_head_to_model", False))
            policy.load_state_dict(
                utilities.prepare_ppo_policy_state_dict_for_load(sd, policy, slice_policy_head_to_model=_slice),
                strict=True,
            )
            uncompiled_local.load_state_dict(
                utilities.prepare_ppo_policy_state_dict_for_load(sd, uncompiled_local, slice_policy_head_to_model=_slice),
                strict=True,
            )
            loaded_from_file = True
            print("[OK] DPO: loaded weights1.torch")
    except Exception as e:
        print(f"[INFO] DPO: no usable checkpoint ({e}); will align to shared init from train.py")

    if not loaded_from_file:
        with shared_network_lock:
            uncompiled_local.load_state_dict(uncompiled_shared_network.state_dict())

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_local.state_dict())

    ref_policy = copy.deepcopy(uncompiled_local).eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    accumulated_stats: defaultdict[str, Any] = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    time_last_save = time.perf_counter()

    try:
        loaded = joblib.load(save_dir / "accumulated_stats.joblib")
        accumulated_stats.update(loaded)
        shared_steps.value = int(accumulated_stats.get("cumul_number_frames_played", 0))
        print(f"[OK] DPO: resumed stats frames={shared_steps.value:,}")
    except Exception:
        print("[INFO] DPO: fresh accumulated_stats")

    if "cumul_training_hours" not in accumulated_stats:
        accumulated_stats["cumul_training_hours"] = 0.0
    if not isinstance(accumulated_stats.get("alltime_min_ms"), dict):
        accumulated_stats["alltime_min_ms"] = {}
    if not isinstance(accumulated_stats.get("rolling_mean_ms"), dict):
        accumulated_stats["rolling_mean_ms"] = {}

    frames_at_last_periodic_save = int(accumulated_stats.get("cumul_number_frames_played", 0))

    if cfg.transformers.fusion_mode != "none":
        utilities.enable_all_parameters_trainable(uncompiled_local)

    from trackmania_rl.param_freeze import apply_frozen_prefixes, prefixes_that_match_module

    freeze_pfx = wiring.freeze_prefixes_from_config(cfg)
    n_u = apply_frozen_prefixes(uncompiled_local, freeze_pfx)
    n_p = apply_frozen_prefixes(policy, freeze_pfx)
    if freeze_pfx and (n_u or n_p):
        active = prefixes_that_match_module(uncompiled_local, freeze_pfx)
        print(f"[OK] DPO parameter freeze: {n_u} / {n_p} tensors — prefixes: {active}")

    optimizer = torch.optim.RAdam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=utilities.from_exponential_schedule(cfg.lr_schedule, shared_steps.value),
        eps=cfg.adam_epsilon,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )
    scaler = torch.amp.GradScaler("cuda")

    try:
        optimizer.load_state_dict(torch.load(save_dir / "optimizer1.torch", weights_only=False))
        scaler.load_state_dict(torch.load(save_dir / "scaler.torch", weights_only=False))
        print("[OK] DPO: loaded optimizer/scaler")
    except Exception:
        pass

    tb_suffix = utilities.from_staircase_schedule(cfg.tensorboard_suffix_schedule, shared_steps.value)
    writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (cfg.run_name + tb_suffix)))

    rollout_queue_readers = [q._reader for q in rollout_queues]
    queue_order = list(range(len(rollout_queues)))

    pair_buffer: list[dict[str, Any]] = []
    sched_step_anchor: int | None = None
    update_count = 0
    previous_alltime_min: dict[str, float] | None = None
    last_loss: float | None = None
    last_lr: float | None = None
    offline_cycle: list[tuple[Path, Path]] = []

    if offline_jsonl and data_mode in ("offline", "both"):
        jp = Path(offline_jsonl)
        if not jp.is_absolute():
            jp = base_dir / jp
        if jp.is_file():
            offline_cycle = list(_offline_pair_paths_iterator(jp))
            print(f"[OK] DPO: loaded {len(offline_cycle)} offline pair paths from {jp}")
        else:
            print(f"[WARN] DPO: offline JSONL not found: {jp}")
    if data_mode == "offline" and not offline_cycle:
        print(
            "[ERROR] DPO: dpo_data_mode=offline but no offline pairs loaded "
            "(missing/empty dpo_offline_pairs_jsonl). Learner will idle."
        )

    policy.train()
    while True:
        wait(rollout_queue_readers, timeout=0.5)
        for idx in list(queue_order):
            q = rollout_queues[idx]
            if q.empty():
                continue
            (
                rollout_results,
                end_race_stats,
                _fill_buffer,
                is_explo,
                map_name,
                map_status,
                rollout_duration,
                loop_number,
            ) = q.get()
            try:
                queue_order.remove(idx)
            except ValueError:
                pass
            queue_order.append(idx)

            n_frames = len(rollout_results.get("frames", []))
            if sched_step_anchor is None:
                sched_step_anchor = int(shared_steps.value)
            accumulated_stats["cumul_number_frames_played"] += n_frames
            shared_steps.value = int(accumulated_stats["cumul_number_frames_played"])

            step_s = sched_step_anchor
            batch = build_policy_rollout_tensors(rollout_results, end_race_stats, cfg, device, step_s)
            if batch is not None and data_mode in ("online", "both"):
                score = float(batch["rewards"].sum().item())
                pair_buffer.append({"batch": batch, "score": score})
                while len(pair_buffer) > pair_buffer_max:
                    pair_buffer.pop(0)

            if end_race_stats.get("race_time") is not None:
                rt_ms = float(end_race_stats["race_time"])
                rt = rt_ms / 1000.0
                key = f"eval_race_time_{map_status}_{map_name}" if not is_explo else f"explo_race_time_{map_status}_{map_name}"
                writer.add_scalar(key, rt, shared_steps.value)
                race_finished = bool(end_race_stats.get("race_finished"))
                fin_key = (
                    f"eval_race_finished_{map_status}_{map_name}"
                    if not is_explo
                    else f"explo_race_finished_{map_status}_{map_name}"
                )
                writer.add_scalar(fin_key, 1.0 if race_finished else 0.0, shared_steps.value)

                if not is_explo and race_finished:
                    accumulated_stats["rolling_mean_ms"][map_name] = (
                        accumulated_stats["rolling_mean_ms"].get(
                            map_name, cfg.cutoff_rollout_if_race_not_finished_within_duration_ms
                        )
                        * 0.9
                        + rt_ms * 0.1
                    )

                old_best = accumulated_stats["alltime_min_ms"].get(map_name, 99999999999)
                if rt_ms < old_best:
                    accumulated_stats["alltime_min_ms"][map_name] = rt_ms

        bw: dict[str, torch.Tensor] | None = None
        bl: dict[str, torch.Tensor] | None = None
        use_offline = False

        if data_mode == "offline" and offline_cycle:
            use_offline = True
        elif data_mode == "both" and offline_cycle and (update_count % 2 == 1 or len(pair_buffer) < 2):
            use_offline = True
        elif data_mode == "online" or not offline_cycle:
            use_offline = False

        if use_offline and offline_cycle:
            pair_idx = update_count % len(offline_cycle)
            cp, rp = offline_cycle[pair_idx]
            cp = _resolve_path(cp, base_dir)
            rp = _resolve_path(rp, base_dir)
            try:
                rw, ew = joblib.load(cp)
                rl, el = joblib.load(rp)
                bw = build_policy_rollout_tensors(rw, ew, cfg, device, int(shared_steps.value))
                bl = build_policy_rollout_tensors(rl, el, cfg, device, int(shared_steps.value))
            except Exception as e:
                print(f"[WARN] DPO offline pair load failed: {e}")
                bw, bl = None, None
        elif len(pair_buffer) >= 2:
            scores = [x["score"] for x in pair_buffer]
            i_best = max(range(len(pair_buffer)), key=lambda i: scores[i])
            i_worst = min(range(len(pair_buffer)), key=lambda i: scores[i])
            if i_best != i_worst:
                b_best = pair_buffer[i_best]["batch"]
                b_worst = pair_buffer[i_worst]["batch"]
                for ix in sorted([i_best, i_worst], reverse=True):
                    pair_buffer.pop(ix)
                bw, bl = b_best, b_worst

        if bw is None or bl is None:
            time.sleep(0.02)
            continue

        sched_step = sched_step_anchor if sched_step_anchor is not None else int(shared_steps.value)
        sched_step_anchor = None

        beta = dpo_scheduled_float(cfg, "dpo_beta", "dpo_beta_schedule", sched_step)
        vf_coef = dpo_scheduled_float(cfg, "dpo_vf_coef", "dpo_vf_coef_schedule", sched_step)
        max_grad_norm = dpo_scheduled_float(cfg, "dpo_max_grad_norm", "dpo_max_grad_norm_schedule", sched_step)

        bw, bl = _align_trajectory_batches(bw, bl)
        if bw["actions"].shape[0] < 2:
            continue

        total_loss_acc = 0.0
        opt_steps = 0
        for _ in range(update_epochs):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logp_pi_w, _ent_w, vals_w = sum_log_probs_evaluate(policy, bw["obs_img"], bw["obs_float"], bw["actions"])
                logp_pi_l, _ent_l, vals_l = sum_log_probs_evaluate(policy, bl["obs_img"], bl["obs_float"], bl["actions"])
                with torch.no_grad():
                    logp_ref_w, _, _ = sum_log_probs_evaluate(ref_policy, bw["obs_img"], bw["obs_float"], bw["actions"])
                    logp_ref_l, _, _ = sum_log_probs_evaluate(ref_policy, bl["obs_img"], bl["obs_float"], bl["actions"])
                loss_dpo = dpo_preference_loss(logp_pi_w, logp_ref_w, logp_pi_l, logp_ref_l, beta)
                ov_w = bw["old_values"].reshape(-1)
                ov_l = bl["old_values"].reshape(-1)
                loss_v = F.mse_loss(vals_w.reshape(-1), ov_w) + F.mse_loss(vals_l.reshape(-1), ov_l)
                loss = loss_dpo + vf_coef * loss_v

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            opt_steps += 1
            total_loss_acc += float(loss.detach())

        update_count += 1
        if update_count % ref_sync_every == 0:
            ref_policy.load_state_dict(uncompiled_local.state_dict())

        lr = utilities.from_exponential_schedule(cfg.lr_schedule, shared_steps.value)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with shared_network_lock:
            uncompiled_shared_network.load_state_dict(uncompiled_local.state_dict())

        loss_mean = total_loss_acc / max(1, opt_steps)
        gstep_updates = int(shared_steps.value)
        writer.add_scalar("Training/dpo_loss", loss_mean, gstep_updates)
        writer.add_scalar("Training/learning_rate", lr, gstep_updates)
        writer.add_scalar("DPO/beta", beta, gstep_updates)
        writer.add_scalar("DPO/vf_coef", vf_coef, gstep_updates)
        writer.add_scalar("DPO/max_grad_norm", max_grad_norm, gstep_updates)
        last_loss = loss_mean
        last_lr = lr

        if time.perf_counter() - time_last_save >= 300:
            now = time.perf_counter()
            delta_s = now - time_last_save
            time_last_save = now
            accumulated_stats["cumul_training_hours"] = float(accumulated_stats.get("cumul_training_hours", 0)) + delta_s / 3600.0

            walltime_tb = time.time()
            gstep = shared_steps.value
            cur_frames = int(accumulated_stats["cumul_number_frames_played"])
            env_frames_per_second = (
                (cur_frames - frames_at_last_periodic_save) / delta_s if delta_s > 0 else 0.0
            )
            frames_at_last_periodic_save = cur_frames

            writer.add_scalar("cumul_training_hours", accumulated_stats["cumul_training_hours"], gstep, walltime=walltime_tb)
            writer.add_scalar("Performance/env_frames_per_second", env_frames_per_second, gstep, walltime=walltime_tb)
            for mn, ms in accumulated_stats["alltime_min_ms"].items():
                writer.add_scalar(f"alltime_min_ms_{mn}", ms, gstep, walltime=walltime_tb)

            previous_alltime_min = previous_alltime_min or copy.deepcopy(accumulated_stats["alltime_min_ms"])
            writer.add_text(
                "times_summary",
                f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} "
                + " ".join(
                    [
                        f"{'**' if v < previous_alltime_min.get(k, 99999999) else ''}{k}: {v / 1000:.2f}"
                        f"{'**' if v < previous_alltime_min.get(k, 99999999) else ''}"
                        for k, v in sorted(accumulated_stats["alltime_min_ms"].items())
                    ]
                ),
                gstep,
                walltime=walltime_tb,
            )
            previous_alltime_min = copy.deepcopy(accumulated_stats["alltime_min_ms"])

            utilities.save_ppo_checkpoint(save_dir, policy, optimizer, scaler)
            joblib.dump(dict(accumulated_stats), save_dir / "accumulated_stats.joblib")

            loss_s = f"{last_loss:.4e}" if last_loss is not None else "n/a"
            lr_s = f"{last_lr:.2e}" if last_lr is not None else "n/a"
            print("\n" + "=" * 80)
            print(f"  DPO TRAINING SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print(f"  Frames played: {accumulated_stats['cumul_number_frames_played']:,}")
            print(f"  DPO updates: {update_count}  |  last loss: {loss_s}  |  lr: {lr_s}")
            print("-" * 80)
            print("  BEST TIMES:")
            if accumulated_stats["alltime_min_ms"]:
                for map_name_iter, best_time_ms in sorted(accumulated_stats["alltime_min_ms"].items()):
                    best_time_s = best_time_ms / 1000
                    rolling_mean_s = accumulated_stats["rolling_mean_ms"].get(map_name_iter, best_time_ms) / 1000
                    print(f"    {map_name_iter:15} {best_time_s:6.2f}s  (rolling avg: {rolling_mean_s:6.2f}s)")
            else:
                print("    (no finished race times yet)")
            print("=" * 80 + "\n")
            print(f"[OK] DPO checkpoint saved (update {update_count}, frames {shared_steps.value:,})")

        sys.stdout.flush()
