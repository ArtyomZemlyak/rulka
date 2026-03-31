"""
Various neural network & scheduling utilities.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import List, Tuple

import random
import joblib
import numpy as np
import torch
from prettytable import PrettyTable

from trackmania_rl import run_to_video

_transformers_load_report_quiet: bool = False

_log = logging.getLogger(__name__)


class _HfModelingUtilsNoiseFilter(logging.Filter):
    """Drop load-report spam without raising ``modeling_utils`` logger level.

    If ``transformers.modeling_utils`` is set to ERROR/WARNING, ``PreTrainedModel`` runs
    ``verify_tp_plan`` whenever ``logger.level >= WARNING`` — which incorrectly becomes true
    for ERROR (40), flooding "The following layers were not sharded" on every ``from_pretrained``.
    Filtering keeps the default level and only strips known noise.
    """

    _SUBSTRINGS = (
        "LOAD REPORT",
        "The following layers were not sharded",
        "The following TP rules were not applied",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(s in msg for s in self._SUBSTRINGS)


def apply_hf_hub_console_defaults() -> None:
    """Set process env before any ``huggingface_hub`` / download code runs (call from train entrypoint).

    Disables tqdm-style hub progress bars (e.g. ``Loading weights: …``). Child processes inherit the env.
    """
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def quiet_transformers_weight_load_reports() -> None:
    """Silence noisy HF weight load diagnostics on every ``from_pretrained`` (each worker process).

    RL uses BERT/timm backbones without LM/CLS heads, so UNEXPECTED keys are routine. Tables are
    WARNING on ``transformers.modeling_utils`` — use a log filter there, not ``setLevel(ERROR)``,
    to avoid triggering tensor-parallel ``verify_tp_plan`` spam (see ``_HfModelingUtilsNoiseFilter``).
    """
    global _transformers_load_report_quiet
    if _transformers_load_report_quiet:
        return
    apply_hf_hub_console_defaults()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").addFilter(_HfModelingUtilsNoiseFilter())
    _transformers_load_report_quiet = True


def _rulka_repo_root_candidates() -> list[Path]:
    """Likely project roots: package parent (dev layout), then cwd."""
    import trackmania_rl

    pkg = Path(trackmania_rl.__file__).resolve()
    return [pkg.parents[1], Path.cwd()]


def ppo_weights1_torch_paths(cfg) -> list[Path]:
    """Candidate ``save/<run_name>/weights1.torch`` paths (same layout as ``scripts/train.py``)."""
    rn = getattr(cfg, "run_name", "") or ""
    return [r / "save" / rn / "weights1.torch" for r in _rulka_repo_root_candidates()]


def ppo_pretrain_bc_checkpoint_file(cfg) -> Path | None:
    """BC file for ``training.pretrain_ppo_policy_path`` if it exists, else ``None``."""
    raw = getattr(cfg, "pretrain_ppo_policy_path", None)
    if raw is None or not str(raw).strip():
        return None
    rel = Path(str(raw).strip())
    bases = [rel] if rel.is_absolute() else [b / rel for b in _rulka_repo_root_candidates()]
    for base in bases:
        p = base / "ppo_policy_bc.pt" if base.is_dir() else base
        if p.is_file():
            return p
    return None


def skip_multimodal_fusion_hub_init_from_pretrained(cfg) -> bool:
    """Skip ``nn.init_from_pretrained`` when weights are supplied via checkpoint or BC (not hub dir).

    Avoids loading a second Rulka ``save_pretrained`` tree on top of ``weights1.torch`` / BC inject.
    """
    if getattr(cfg, "algorithm", "") != "ppo":
        return False
    tr = getattr(cfg, "transformers", None)
    if tr is None or not str(getattr(tr, "init_from_pretrained", "") or "").strip():
        return False
    if any(p.is_file() for p in ppo_weights1_torch_paths(cfg)):
        return True
    if ppo_pretrain_bc_checkpoint_file(cfg) is not None:
        return True
    return False


def init_kaiming(layer, neg_slope=0, nonlinearity="leaky_relu"):
    torch.nn.init.kaiming_normal_(layer.weight, a=neg_slope, mode="fan_out", nonlinearity=nonlinearity)
    torch.nn.init.zeros_(layer.bias)


def init_xavier(layer, gain=1.0):
    torch.nn.init.xavier_normal_(layer.weight, gain=gain)
    torch.nn.init.zeros_(layer.bias)


def init_orthogonal(layer, gain=1.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.zeros_(layer.bias)


def init_uniform(layer, a, b):
    torch.nn.init.uniform_(layer.weight, a=a, b=b)
    torch.nn.init.zeros_(layer.bias)


def init_normal(layer, mean, std):
    torch.nn.init.normal_(layer.weight, mean=mean, std=std)
    torch.nn.init.zeros_(layer.bias)


def log_gradient_norms(model, layer_grad_norm_history):
    l2_norms = []
    linf_norms = []
    param_names = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            l2_norms.append(torch.norm(grad))
            linf_norms.append(torch.max(grad))
            param_names.append(name)

    l2_norms_cpu = torch.stack(l2_norms).cpu().numpy()
    linf_norms_cpu = torch.stack(linf_norms).cpu().numpy()

    for name, l2_norm, linf_norm in zip(param_names, l2_norms_cpu, linf_norms_cpu):
        layer_grad_norm_history[f"L2_grad_norm_{name}"].append(l2_norm)
        layer_grad_norm_history[f"Linf_grad_norm_{name}"].append(linf_norm)


def linear_combination(a, b, alpha):
    assert a.shape == b.shape
    a.mul_(1 - alpha)
    a.add_(alpha * b)
    return a


# From https://github.com/pfnet/pfrl/blob/2ad3d51a7a971f3fe7f2711f024be11642990d61/pfrl/utils/copy_param.py#L37
def soft_copy_param(target_link, source_link, tau, skip_key_prefixes=None):
    """Soft-copy parameters of a link to another link.
    If skip_key_prefixes is provided (iterable of str), keys matching any prefix are skipped.
    Uses the same rules as ``trackmania_rl.param_freeze`` (including ``_orig_mod.`` from ``torch.compile``).
    """
    from trackmania_rl.param_freeze import param_name_matches_any_prefix

    target_dict = target_link.state_dict()
    source_dict = source_link.state_dict()
    skip_prefixes = tuple(skip_key_prefixes or ())
    for k, target_value in target_dict.items():
        if skip_prefixes and param_name_matches_any_prefix(k, skip_prefixes):
            continue
        source_value = source_dict[k]
        if source_value.dtype in [torch.float32, torch.float64, torch.float16]:
            linear_combination(target_value, source_value, tau)
        else:
            # Scalar type
            # Some modules such as BN has scalar value `num_batches_tracked`
            target_dict[k] = source_value
            assert False, "Soft scalar update should not happen"


def custom_weight_decay(target_link, decay_factor, only_trainable=False):
    """Apply decay_factor to parameters. If only_trainable=True, skip parameters with requires_grad=False."""
    if only_trainable:
        for p in target_link.parameters():
            if p.requires_grad:
                p.data.mul_(decay_factor)
    else:
        target_dict = target_link.state_dict()
        for k, target_value in target_dict.items():
            target_value.mul_(decay_factor)


def from_exponential_schedule(schedule: List[Tuple[int, float]], current_step: int):
    """
    Calculate the current scheduled value, with exponential interpolation between fixed setpoints at given steps.
    If current step is larger than the largest scheduled step, return the value prescribed by the largest scheduled step.

    Args:
        - schedule:         a list of (step, value) tuples. Must contain a value for step 0.
        - current_step:     an int representing... the current step

    Returns:
        value: the value defined by the schedule and current_step
    """
    schedule = sorted(schedule, key=lambda p: p[0])  # Sort by step in case it was not defined in sorted order
    assert schedule[0][0] == 0
    schedule_end_index = next((idx for idx, p in enumerate(schedule) if p[0] > current_step), -1)  # Returns -1 if none is found
    if schedule_end_index == -1:
        return schedule[-1][1]
    else:
        assert schedule_end_index >= 1
        schedule_end_step = schedule[schedule_end_index][0]
        schedule_begin_step = schedule[schedule_end_index - 1][0]
        annealing_period = schedule_end_step - schedule_begin_step
        end_value = schedule[schedule_end_index][1]
        begin_value = schedule[schedule_end_index - 1][1]
        ratio = begin_value / end_value
        assert annealing_period > 0
        return begin_value * math.exp(-math.log(ratio) * (current_step - schedule_begin_step) / annealing_period)


def from_linear_schedule(schedule, current_step):
    """
    Calculate the current scheduled value, with linear interpolation between fixed setpoints at given steps.
    If current step is larger than the largest scheduled step, return the value prescribed by the largest scheduled step.

    Args:
        - schedule:         a list of (step, value) tuples. Must contain a value for step 0.
        - current_step:     an int representing... the current step

    Returns:
        value: the value defined by the schedule and current_step
    """
    schedule = sorted(schedule, key=lambda p: p[0])  # Sort by step in case it was not defined in sorted order
    assert schedule[0][0] == 0
    return np.interp([current_step], [p[0] for p in schedule], [p[1] for p in schedule])[0]


def from_staircase_schedule(schedule, current_step):
    """
    Calculate the current scheduled value, with no interpolation between steps.

    Args:
        - schedule:         a list of (step, value) tuples. Must contain a value for step 0.
        - current_step:     an int representing... the current step

    Returns:
        value: the value defined by the schedule and current_step
    """
    schedule = sorted(schedule, key=lambda p: p[0])  # Sort by step in case it was not defined in sorted order
    assert schedule[0][0] == 0
    return next((p for p in reversed(schedule) if p[0] <= current_step))[1]


def count_parameters(model):
    # from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def save_run(
    base_dir: Path,
    run_dir: Path,
    rollout_results: dict,
    inputs_filename: str,
    inputs_only: bool,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    run_to_video.write_actions_in_tmi_format(rollout_results["actions"], run_dir / inputs_filename)
    if not inputs_only:
        joblib.dump(rollout_results["q_values"], run_dir / "q_values.joblib")


def save_checkpoint(
    checkpoint_dir: Path,
    online_network: torch.nn.Module,
    target_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(online_network.state_dict(), checkpoint_dir / "weights1.torch")
    torch.save(target_network.state_dict(), checkpoint_dir / "weights2.torch")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer1.torch")
    torch.save(scaler.state_dict(), checkpoint_dir / "scaler.torch")


def _ppo_policy_inner(policy: torch.nn.Module) -> torch.nn.Module:
    return getattr(policy, "_orig_mod", policy)


def enable_all_parameters_trainable(module: torch.nn.Module) -> None:
    """``requires_grad=True`` on every parameter (buffers unchanged)."""
    for p in module.parameters():
        p.requires_grad_(True)


def _strip_ppo_bc_multi_offset_base_prefix(key: str) -> str:
    """Map ``base.*`` / ``_orig_mod.base.*`` (``PpoPolicyBcMultiOffset``) to inner actor-critic keys."""
    p = "_orig_mod."
    if key.startswith(p):
        rest = key[len(p) :]
        if rest.startswith("base."):
            return p + rest[len("base.") :]
        return key
    if key.startswith("base."):
        return key[len("base.") :]
    return key


def _parse_ppo_bc_head_key(stripped_key: str) -> tuple[int, str] | None:
    """Return ``(index, 'weight'|'bias')`` for ``bc_heads.{i}.(weight|bias)`` (optional ``_orig_mod.``)."""
    k = stripped_key
    p = "_orig_mod."
    if k.startswith(p):
        k = k[len(p) :]
    parts = k.split(".")
    if len(parts) != 3 or parts[0] != "bc_heads":
        return None
    try:
        idx = int(parts[1])
    except ValueError:
        return None
    if parts[2] not in ("weight", "bias"):
        return None
    return idx, parts[2]


def _ppo_policy_head_keys_for_model(mk: set[str]) -> tuple[str, str] | None:
    for prefix in ("_orig_mod.", ""):
        wk = f"{prefix}policy_head.weight"
        bk = f"{prefix}policy_head.bias"
        if wk in mk and bk in mk:
            return wk, bk
    return None


def _maybe_slice_ppo_policy_head_to_model(out: dict, policy: torch.nn.Module) -> None:
    """Mutate *out* so ``policy_head`` matches the model if the checkpoint had more outputs (same in_features)."""
    mk = set(policy.state_dict().keys())
    ph = _ppo_policy_head_keys_for_model(mk)
    if ph is None:
        return
    w_key, b_key = ph
    if w_key not in out or b_key not in out:
        return
    target_w = policy.state_dict()[w_key]
    target_b = policy.state_dict()[b_key]
    cw = out[w_key]
    cb = out[b_key]
    if cw.shape == target_w.shape and cb.shape == target_b.shape:
        return
    if cw.dim() != 2 or target_w.dim() != 2:
        raise ValueError(
            f"policy_head.weight must be 2D; got checkpoint {tuple(cw.shape)} vs model {tuple(target_w.shape)}"
        )
    if cw.shape[1] != target_w.shape[1]:
        raise ValueError(
            f"policy_head in_features mismatch (checkpoint {tuple(cw.shape)} vs model {tuple(target_w.shape)}); "
            "cannot slice — trunk/head width differs."
        )
    if cw.shape[0] < target_w.shape[0]:
        raise ValueError(
            f"policy_head: checkpoint has fewer outputs ({cw.shape[0]}) than model ({target_w.shape[0]}). "
            "slice_head_to_model only truncates a larger BC head down to RL size."
        )
    if cw.shape[0] > target_w.shape[0]:
        _log.warning(
            "PPO: truncating policy_head from %d to %d outputs (prefix logits; only valid when BC action indices "
            "match RL for those outputs).",
            cw.shape[0],
            target_w.shape[0],
        )
        out[w_key] = cw[: target_w.shape[0]].clone().detach()
        out[b_key] = cb[: target_b.shape[0]].clone().detach()
        return
    if cb.shape != target_b.shape:
        raise ValueError(
            f"policy_head bias shape mismatch: checkpoint {tuple(cb.shape)} vs model {tuple(target_b.shape)}"
        )


def _mean_bc_heads_policy_tensors(sd: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Average ``bc_heads.{i}`` linears into one policy head (same as single ``nn.Linear``)."""
    weights: dict[int, torch.Tensor] = {}
    biases: dict[int, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        sk = _strip_ppo_bc_multi_offset_base_prefix(k)
        parsed = _parse_ppo_bc_head_key(sk)
        if parsed is None:
            continue
        idx, kind = parsed
        if kind == "weight":
            weights[idx] = v
        else:
            biases[idx] = v
    if not weights:
        return None, None
    idxs = sorted(weights.keys())
    if set(idxs) != set(biases.keys()):
        return None, None
    ref = weights[idxs[0]]
    w_stack = torch.stack([weights[i].float() for i in idxs])
    b_stack = torch.stack([biases[i].float() for i in idxs])
    w_mean = w_stack.mean(dim=0).to(dtype=ref.dtype, device=ref.device)
    b_mean = b_stack.mean(dim=0).to(dtype=ref.dtype, device=ref.device)
    return w_mean, b_mean


def align_ppo_checkpoint_keys_to_model(loaded_sd: dict, model: torch.nn.Module) -> dict:
    """Match checkpoint keys to ``model.state_dict()`` (e.g. torch.compile ``_orig_mod.`` wrappers)."""
    model_keys = list(model.state_dict().keys())
    loaded_keys = list(loaded_sd.keys())
    if not loaded_keys or not model_keys:
        sd = dict(loaded_sd)
    else:
        model_has_prefix = any(k.startswith("_orig_mod.") for k in model_keys)
        loaded_has_prefix = any(k.startswith("_orig_mod.") for k in loaded_keys)
        if model_has_prefix and not loaded_has_prefix:
            sd = {"_orig_mod." + k: v for k, v in loaded_sd.items()}
        elif loaded_has_prefix and not model_has_prefix:
            p = "_orig_mod."
            sd = {k[len(p) :]: v for k, v in loaded_sd.items() if k.startswith(p)}
        else:
            sd = dict(loaded_sd)
    mk = set(model.state_dict().keys())
    if mk:
        sd = {k: v for k, v in sd.items() if not ("hf_fusion_base." in k and k not in mk)}
    return sd


def prepare_ppo_policy_state_dict_for_load(
    loaded_sd: dict,
    policy: torch.nn.Module,
    *,
    slice_policy_head_to_model: bool = False,
) -> dict:
    """Prepare a flat checkpoint for ``policy.load_state_dict(..., strict=True)``.

    - Aligns ``_orig_mod.`` prefixes with the target module (compiled vs uncompiled).
    - Strips ``PpoPolicyBcMultiOffset`` ``base.*`` so inner trunk / heads keys match RL.
    - **BC multi-offset:** trains separate ``bc_heads.{i}``; base ``policy_head`` is frozen.
      Those trained heads are **averaged** into a single ``policy_head`` expected by RL.
    - Drops orphan ``hf_fusion_base.*`` keys (legacy duplicate registration).
    - If ``slice_policy_head_to_model`` (``training.pretrain_ppo_policy_slice_head_to_model``): when the checkpoint
      ``policy_head`` has more output logits than the target policy but the same ``in_features``, keep only the
      leading rows (assumes RL actions are a prefix of BC actions by index).

    Use for ``weights1.torch``, ``ppo_policy_bc.pt``, or policy shards from Lightning.
    """
    sd = align_ppo_checkpoint_keys_to_model(loaded_sd, policy)
    mk = set(policy.state_dict().keys())
    if not mk:
        return sd
    out: dict = {}
    for k, v in sd.items():
        if k in mk:
            out[k] = v
            continue
        sk = _strip_ppo_bc_multi_offset_base_prefix(k)
        if sk != k and sk in mk:
            out[sk] = v

    ph = _ppo_policy_head_keys_for_model(mk)
    if ph is not None:
        w_key, b_key = ph
        merged_w, merged_b = _mean_bc_heads_policy_tensors(sd)
        if merged_w is not None and merged_b is not None:
            out[w_key] = merged_w
            out[b_key] = merged_b

    if slice_policy_head_to_model:
        _maybe_slice_ppo_policy_head_to_model(out, policy)

    return out


def _save_ppo_hf_transformer_artifacts(checkpoint_dir: Path, policy: torch.nn.Module, cfg) -> None:
    """Write Hugging Face-style dirs beside weights (vision backbone + fusion policy)."""
    if getattr(cfg, "algorithm", "") != "ppo":
        return
    inner = _ppo_policy_inner(policy)
    if (
        cfg.transformers.fusion_mode == "none"
        and cfg.vis.transformer is not None
        and cfg.vis.transformer.use_hf_backbone
        and hasattr(inner, "backbone")
        and hasattr(inner.backbone, "save_pretrained")
    ):
        out = checkpoint_dir / "hf_transformer_vis"
        try:
            inner.backbone.save_pretrained(str(out))
            proc = getattr(inner, "image_processor", None)
            if proc is not None and hasattr(proc, "save_pretrained"):
                _vis_proc_cfgs = ("preprocessor_config.json", "processor_config.json")
                if not any((out / n).exists() for n in _vis_proc_cfgs):
                    proc.save_pretrained(str(out))
        except Exception as e:
            print(f"[WARN] PPO: hf_transformer_vis save failed: {e}")
    # TorchMultimodalActorCritic uses _vis_branch; fusion trunk is enc_fusion_native / MLP / CNN / HF, not enc_pc/enc_uni.
    if cfg.transformers.fusion_mode != "none" and hasattr(inner, "_vis_branch"):
        try:
            from trackmania_rl.agents.policy_models.rulka_multimodal_fusion_hub import wrap_fusion_policy_for_hf_save
        except ImportError as e:
            print(
                f"[WARN] PPO: hf_transformer_fusion not saved (transformers / hub): {e}. "
                'Install: pip install -e ".[policy]"'
            )
        else:
            try:
                fusion_dir = checkpoint_dir / "hf_transformer_fusion"
                w = wrap_fusion_policy_for_hf_save(inner, cfg)
                try:
                    w.save_pretrained(str(fusion_dir), safe_serialization=True)
                except Exception as e_safe:
                    print(
                        f"[INFO] PPO: hf_transformer_fusion safe_serialization save failed ({e_safe}); "
                        "retrying with safe_serialization=False."
                    )
                    w.save_pretrained(str(fusion_dir), safe_serialization=False)
                proc = getattr(inner, "_hf_vis_processor", None)
                if proc is not None and hasattr(proc, "save_pretrained"):
                    # Avoid Hub HEAD round-trips every periodic save (SSL timeouts); configs are static.
                    _proc_names = ("preprocessor_config.json", "processor_config.json")
                    if not any((fusion_dir / n).exists() for n in _proc_names):
                        try:
                            proc.save_pretrained(str(fusion_dir))
                        except Exception as e_proc:
                            print(f"[WARN] PPO: hf_transformer_fusion image_processor save failed: {e_proc}")
            except Exception as e:
                print(f"[WARN] PPO: hf_transformer_fusion save failed: {e}")


def save_ppo_checkpoint(
    checkpoint_dir: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
):
    """PPO: single policy weights (no target network file).

    Writes the full policy ``state_dict`` (vis + fusion encoder/decoder + trunk + heads),
    including nested Hugging Face submodules, using the uncompiled inner module when
    ``policy`` is ``torch.compile``-wrapped.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sd = _ppo_policy_inner(policy).state_dict()
    torch.save(sd, checkpoint_dir / "weights1.torch")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer1.torch")
    torch.save(scaler.state_dict(), checkpoint_dir / "scaler.torch")
    from config_files.config_loader import get_config

    _save_ppo_hf_transformer_artifacts(checkpoint_dir, policy, get_config())

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
