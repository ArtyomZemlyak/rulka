"""PPO learner: on-policy rollout aggregation, GAE, clipped objective (no IQN replay)."""

from __future__ import annotations

import copy
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from config_files.config_loader import get_config
from trackmania_rl import utilities
from trackmania_rl.agents.algorithms import get_wiring
from trackmania_rl.agents.policy_optimization.ppo import compute_gae, ppo_loss_components
from trackmania_rl.agents.policy_optimization.rollout_rewards import ppo_rewards_and_dones_from_rollout


def _ppo_scheduled_float(cfg, attr: str, schedule_attr: str, step: int) -> float:
    sched = getattr(cfg, schedule_attr)
    scalar = float(getattr(cfg, attr))
    if sched:
        return float(utilities.from_linear_schedule(sched, step))
    return float(utilities.from_linear_schedule([[0, scalar]], step))


def _rollout_tensors(
    rollout_results: dict[str, Any],
    end_race_stats: dict[str, Any],
    cfg,
    device: torch.device,
    schedule_step: int,
) -> dict[str, torch.Tensor] | None:
    """Build GPU tensors for one rollout; returns None if too short or missing PPO fields."""
    if cfg.algorithm != "ppo":
        return None
    n_act = len(rollout_results.get("actions", []))
    n_lp = len(rollout_results.get("ppo_log_probs", []))
    if n_act < 2 or n_lp < 2:
        return None
    n = min(n_act, n_lp, len(rollout_results["frames"]), len(rollout_results["state_float"]))
    if n < 2:
        return None

    gamma_shaping = _ppo_scheduled_float(cfg, "gamma", "ppo_gamma_schedule", schedule_step)
    eng_ss = utilities.from_linear_schedule(cfg.engineered_speedslide_reward_schedule, schedule_step)
    eng_ns = utilities.from_linear_schedule(cfg.engineered_neoslide_reward_schedule, schedule_step)
    eng_kk = utilities.from_linear_schedule(cfg.engineered_kamikaze_reward_schedule, schedule_step)
    eng_vcp = utilities.from_linear_schedule(cfg.engineered_close_to_vcp_reward_schedule, schedule_step)
    rewards_np, dones_np = ppo_rewards_and_dones_from_rollout(
        rollout_results,
        cfg,
        gamma=gamma_shaping,
        engineered_speedslide_reward=float(eng_ss),
        engineered_neoslide_reward=float(eng_ns),
        engineered_kamikaze_reward=float(eng_kk),
        engineered_close_to_vcp_reward=float(eng_vcp),
        race_finished=end_race_stats.get("race_finished"),
    )
    if len(rewards_np) < n:
        return None
    rewards_np = rewards_np[:n]
    dones_np = dones_np[:n]

    frames = rollout_results["frames"][:n]
    floats = np.stack(rollout_results["state_float"][:n]).astype(np.float32)

    img_list = []
    for fr in frames:
        x = np.asarray(fr, dtype=np.float32)
        if x.size == 1 or (x.ndim == 3 and x.shape[1] == 1 and x.shape[2] == 1):
            img_list.append(
                np.zeros((1, cfg.H_downsized, cfg.W_downsized), dtype=np.float32)
            )
        else:
            img_list.append(((x.astype(np.float32) - 128.0) / 128.0))
    obs_img = np.stack(img_list, axis=0)
    obs_img_t = torch.from_numpy(obs_img).to(device=device, non_blocking=True)
    if not cfg.use_iqn_image_head:
        obs_img_t = torch.zeros((n, 1, cfg.H_downsized, cfg.W_downsized), device=device, dtype=torch.float32)

    obs_fl_t = torch.from_numpy(floats).to(device=device, non_blocking=True)

    old_vals = np.asarray(rollout_results["ppo_values"][:n], dtype=np.float32)
    old_logp = np.asarray(rollout_results["ppo_log_probs"][:n], dtype=np.float32)

    actions_raw = rollout_results["actions"][:n]
    if cfg.n_actions_per_block <= 1:
        actions_t = torch.tensor([int(a) for a in actions_raw], dtype=torch.long, device=device)
    else:
        actions_t = torch.tensor(np.stack([np.atleast_1d(a) for a in actions_raw]), dtype=torch.long, device=device)

    return {
        "obs_img": obs_img_t,
        "obs_float": obs_fl_t,
        "actions": actions_t,
        "old_logp": torch.from_numpy(old_logp).to(device=device),
        "old_values": torch.from_numpy(old_vals).to(device=device),
        "rewards": torch.from_numpy(rewards_np).to(device=device),
        "dones": torch.from_numpy(dones_np).to(device=device),
    }


def _concat_batches(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in parts[0]:
        out[k] = torch.cat([p[k] for p in parts], dim=0)
    return out


def learner_ppo_process_fn(
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

    policy, uncompiled_local = wiring.make_network(cfg.use_jit, is_inference=False)

    w1_path = save_dir / "weights1.torch"
    loaded_from_file = False
    try:
        if w1_path.exists():
            sd = torch.load(f=w1_path, weights_only=False)
            _slice = bool(getattr(cfg, "pretrain_ppo_policy_slice_head_to_model", False))
            policy.load_state_dict(
                utilities.prepare_ppo_policy_state_dict_for_load(
                    sd, policy, slice_policy_head_to_model=_slice
                ),
                strict=True,
            )
            uncompiled_local.load_state_dict(
                utilities.prepare_ppo_policy_state_dict_for_load(
                    sd, uncompiled_local, slice_policy_head_to_model=_slice
                ),
                strict=True,
            )
            loaded_from_file = True
            print("[OK] PPO: loaded weights1.torch")
    except Exception as e:
        print(f"[INFO] PPO: no usable checkpoint ({e}); will align to shared init from train.py")

    # Collectors read `shared_state_dict` from the parent-built network before learner starts.
    # Without this, learner would use a new random init and overwrite shared → on-policy mismatch.
    if not loaded_from_file:
        with shared_network_lock:
            uncompiled_local.load_state_dict(uncompiled_shared_network.state_dict())

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_local.state_dict())

    accumulated_stats: defaultdict[str, Any] = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    time_last_save = time.perf_counter()

    try:
        loaded = joblib.load(save_dir / "accumulated_stats.joblib")
        accumulated_stats.update(loaded)
        shared_steps.value = int(accumulated_stats.get("cumul_number_frames_played", 0))
        print(f"[OK] PPO: resumed stats frames={shared_steps.value:,}")
    except Exception:
        print("[INFO] PPO: fresh accumulated_stats")

    if "cumul_training_hours" not in accumulated_stats:
        accumulated_stats["cumul_training_hours"] = 0.0
    if not isinstance(accumulated_stats.get("alltime_min_ms"), dict):
        accumulated_stats["alltime_min_ms"] = {}
    if not isinstance(accumulated_stats.get("rolling_mean_ms"), dict):
        accumulated_stats["rolling_mean_ms"] = {}

    frames_at_last_periodic_save = int(accumulated_stats.get("cumul_number_frames_played", 0))

    # Fusion: start from full trainability, then apply ``nn.*.freeze``.
    if cfg.transformers.fusion_mode != "none":
        utilities.enable_all_parameters_trainable(uncompiled_local)

    from trackmania_rl.param_freeze import apply_frozen_prefixes, prefixes_that_match_module

    freeze_pfx = wiring.freeze_prefixes_from_config(cfg)
    n_u = apply_frozen_prefixes(uncompiled_local, freeze_pfx)
    n_p = apply_frozen_prefixes(policy, freeze_pfx)
    if freeze_pfx and (n_u or n_p):
        active = prefixes_that_match_module(uncompiled_local, freeze_pfx)
        print(
            f"[OK] PPO parameter freeze: {n_u} (uncompiled) / {n_p} (policy) tensors "
            f"— active prefixes: {active}"
        )

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
        print("[OK] PPO: loaded optimizer/scaler")
    except Exception:
        pass

    tb_suffix = utilities.from_staircase_schedule(cfg.tensorboard_suffix_schedule, shared_steps.value)
    writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (cfg.run_name + tb_suffix)))

    rollout_queue_readers = [q._reader for q in rollout_queues]
    queue_order = list(range(len(rollout_queues)))

    pending: list[dict[str, torch.Tensor]] = []
    pending_steps = 0
    # One schedule anchor for the whole mega-batch: cumul frames *before* the first rollout
    # in this batch. Keeps γ in reward potential folding aligned with GAE γ for this update.
    pending_sched_step: int | None = None
    update_count = 0
    previous_alltime_min: dict[str, float] | None = None
    last_ppo_loss: float | None = None
    last_ppo_kl: float | None = None
    last_ppo_clipfrac: float | None = None
    last_ppo_vf_clipfrac: float | None = None
    last_lr: float | None = None

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
            if pending_steps == 0:
                pending_sched_step = int(shared_steps.value)
            accumulated_stats["cumul_number_frames_played"] += n_frames
            shared_steps.value = int(accumulated_stats["cumul_number_frames_played"])

            batch = _rollout_tensors(
                rollout_results,
                end_race_stats,
                cfg,
                device,
                pending_sched_step if pending_sched_step is not None else int(shared_steps.value),
            )
            if batch is not None:
                pending.append(batch)
                pending_steps += batch["actions"].shape[0]

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
                    race_time_s = rt_ms / 1000.0
                    race_finished_str = "FINISH" if race_finished else "DNF"
                    explo_str = "EXPLO" if is_explo else "EVAL"
                    if old_best < 99999999:
                        improvement = (old_best - rt_ms) / 1000.0
                        print(
                            f"\n>>> NEW RECORD! [{explo_str}] [{race_finished_str}] {map_name:15} "
                            f"{race_time_s:6.2f}s (improved by {improvement:.3f}s) <<<\n"
                        )
                    else:
                        print(f"\n>>> FIRST FINISH! [{explo_str}] {map_name:15} {race_time_s:6.2f}s <<<\n")

        need = cfg.rollout_steps_per_update
        if pending_steps < need:
            time.sleep(0.02)
            continue

        mega = _concat_batches(pending)
        pending.clear()
        pending_steps = 0
        sched_step = pending_sched_step if pending_sched_step is not None else int(shared_steps.value)
        pending_sched_step = None

        rewards = mega["rewards"]
        dones = mega["dones"]
        old_vals = mega["old_values"]
        old_logp = mega["old_logp"]
        obs_img = mega["obs_img"]
        obs_fl = mega["obs_float"]
        actions = mega["actions"]

        gamma_t = _ppo_scheduled_float(cfg, "gamma", "ppo_gamma_schedule", sched_step)
        gae_lambda_t = _ppo_scheduled_float(cfg, "gae_lambda", "gae_lambda_schedule", sched_step)
        ent_coef_t = _ppo_scheduled_float(cfg, "ent_coef", "ent_coef_schedule", sched_step)
        vf_coef_t = _ppo_scheduled_float(cfg, "vf_coef", "vf_coef_schedule", sched_step)

        T = rewards.shape[0]
        next_value = torch.zeros((), device=device, dtype=rewards.dtype)
        advantages, returns = compute_gae(
            rewards,
            old_vals,
            dones,
            next_value,
            gamma_t,
            gae_lambda_t,
        )
        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_mb = max(1, cfg.num_minibatches)
        mb = max(1, T // n_mb)
        update_epochs = cfg.update_epochs

        total_loss_acc = 0.0
        metrics_acc: dict[str, float] = defaultdict(float)
        opt_steps = 0

        for _ in range(update_epochs):
            idx = torch.randperm(T, device=device)
            for start in range(0, T, mb):
                sel = idx[start : start + mb]
                ob_i = obs_img[sel]
                of_i = obs_fl[sel]
                act_i = actions[sel]
                ol_i = old_logp[sel]
                adv_i = advantages[sel]
                ret_i = returns[sel]
                ov_i = old_vals[sel]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    if not hasattr(policy, "evaluate_actions"):
                        raise RuntimeError("PPO policy must implement evaluate_actions")
                    logp, ent, vals, _ = policy.evaluate_actions(ob_i, of_i, act_i)
                    loss, m = ppo_loss_components(
                        logp,
                        ol_i,
                        adv_i,
                        vals,
                        ret_i,
                        ent,
                        cfg.clip_coef,
                        vf_coef_t,
                        ent_coef_t,
                        old_values=ov_i,
                        clip_coef_vf=getattr(cfg, "clip_coef_vf", None),
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                opt_steps += 1
                total_loss_acc += float(loss.detach())
                for k, v in m.items():
                    metrics_acc[k] += float(v.detach())

        update_count += 1
        for k in metrics_acc:
            metrics_acc[k] /= max(1, opt_steps)

        lr = utilities.from_exponential_schedule(cfg.lr_schedule, shared_steps.value)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with shared_network_lock:
            uncompiled_shared_network.load_state_dict(uncompiled_local.state_dict())

        loss_mean = total_loss_acc / max(1, opt_steps)
        kl_mean = metrics_acc.get("approx_kl", 0.0)
        cf_mean = metrics_acc.get("clipfrac", 0.0)
        vfcf_mean = metrics_acc.get("vf_clipfrac", 0.0)
        gstep_updates = int(shared_steps.value)
        writer.add_scalar("Training/ppo_loss", loss_mean, gstep_updates)
        writer.add_scalar("Training/ppo_approx_kl", kl_mean, gstep_updates)
        writer.add_scalar("Training/ppo_clipfrac", cf_mean, gstep_updates)
        writer.add_scalar("Training/ppo_vf_clipfrac", vfcf_mean, gstep_updates)
        writer.add_scalar("Training/learning_rate", lr, gstep_updates)
        writer.add_scalar("PPO/gamma", gamma_t, gstep_updates)
        writer.add_scalar("PPO/gae_lambda", gae_lambda_t, gstep_updates)
        writer.add_scalar("PPO/ent_coef", ent_coef_t, gstep_updates)
        writer.add_scalar("PPO/vf_coef", vf_coef_t, gstep_updates)
        writer.add_scalar("PPO/rollout_size", T, gstep_updates)
        last_ppo_loss = loss_mean
        last_ppo_kl = kl_mean
        last_ppo_clipfrac = cf_mean
        last_ppo_vf_clipfrac = vfcf_mean
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

            loss_s = f"{last_ppo_loss:.4e}" if last_ppo_loss is not None else "n/a"
            kl_s = f"{last_ppo_kl:.5f}" if last_ppo_kl is not None else "n/a"
            cf_s = f"{last_ppo_clipfrac:.4f}" if last_ppo_clipfrac is not None else "n/a"
            vfcf_s = f"{last_ppo_vf_clipfrac:.4f}" if last_ppo_vf_clipfrac is not None else "n/a"
            lr_s = f"{last_lr:.2e}" if last_lr is not None else "n/a"
            print("\n" + "=" * 80)
            print(f"  PPO TRAINING SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print(f"  Frames played: {accumulated_stats['cumul_number_frames_played']:,}")
            print(f"  Training hours: {accumulated_stats['cumul_training_hours']:.2f}h")
            print(f"  Env frames/sec: {env_frames_per_second:.1f}")
            print(
                f"  PPO updates: {update_count}  |  last loss: {loss_s}  |  approx_kl: {kl_s}  |  "
                f"clipfrac: {cf_s}  |  vf_clipfrac: {vfcf_s}"
            )
            print(f"  Learning rate: {lr_s}")
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
            print(f"[OK] PPO checkpoint saved (update {update_count}, frames {shared_steps.value:,})")

        sys.stdout.flush()
