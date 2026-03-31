"""CNN + MLP actor-critic for PPO (default when HF backbone is off)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from config_files.config_loader import get_config
from trackmania_rl.agents.iqn import _build_img_head, calculate_conv_output_dim
from trackmania_rl.nn_build.vis_cnn_head import default_vis_cnn_head_kw, vis_cnn_head_kw_from_nn_vis
from trackmania_rl.agents.policy_optimization.ppo import discrete_action_logprob_and_entropy
from trackmania_rl.agents.policy_optimization.types import PolicyOutput


class _Cfg(Protocol):
    float_input_dim: int
    float_hidden_dim: int
    dense_hidden_dimension: int
    H_downsized: int
    W_downsized: int
    vision: object
    use_iqn_image_head: bool
    float_inputs_mean: np.ndarray
    float_inputs_std: np.ndarray
    n_actions_per_block: int
    inputs: list[dict[str, Any]]


class PpoActorCritic(nn.Module):
    """Shared trunk, policy logits (K × A), value head."""

    def __init__(
        self,
        *,
        float_inputs_dim: int,
        float_hidden_dim: int,
        dense_hidden_dim: int,
        conv_head_output_dim: int,
        use_image_head: bool,
        float_inputs_mean: Tensor,
        float_inputs_std: Tensor,
        n_actions: int,
        n_actions_per_block: int,
        img_cnn: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.use_image_head = use_image_head
        self.n_actions = n_actions
        self.n_actions_per_block = n_actions_per_block
        self.register_buffer("float_inputs_mean", float_inputs_mean)
        self.register_buffer("float_inputs_std", float_inputs_std)

        if use_image_head:
            kw = img_cnn or default_vis_cnn_head_kw()
            self.img_head = _build_img_head(**kw)
            in_trunk = conv_head_output_dim + float_hidden_dim
        else:
            self.img_head = None
            in_trunk = float_hidden_dim

        self.float_feature_extractor = nn.Sequential(
            nn.Linear(float_inputs_dim, float_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(float_hidden_dim, float_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(in_trunk, dense_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dense_hidden_dim, dense_hidden_dim),
            nn.ReLU(inplace=True),
        )
        out_pi = n_actions * n_actions_per_block
        self.policy_head = nn.Linear(dense_hidden_dim, out_pi)
        self.value_head = nn.Linear(dense_hidden_dim, 1)

    def _norm_float(self, x: Tensor) -> Tensor:
        return (x - self.float_inputs_mean) / self.float_inputs_std

    def forward_features(self, img: Tensor, float_inputs: Tensor) -> Tensor:
        """Trunk output before policy/value heads (for BC multi-offset wrappers)."""
        z = self._norm_float(float_inputs)
        z = self.float_feature_extractor(z)
        if self.use_image_head:
            assert self.img_head is not None
            vis = self.img_head(img)
            h = torch.cat([vis, z], dim=1)
        else:
            h = z
        return self.trunk(h)

    def forward(self, img: Tensor, float_inputs: Tensor) -> PolicyOutput:
        h = self.forward_features(img, float_inputs)
        logits = self.policy_head(h)
        v = self.value_head(h)
        return PolicyOutput(logits=logits, value=v)

    def evaluate_actions(
        self, img: Tensor, float_inputs: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, PolicyOutput]:
        """Return log_prob, entropy, value (B,1), and raw output."""
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


def build_ppo_actor_critic_uncompiled(cfg: _Cfg | None = None) -> PpoActorCritic:
    c = cfg or get_config()
    mean = torch.tensor(np.asarray(c.float_inputs_mean, dtype=np.float32))
    std = torch.tensor(np.asarray(c.float_inputs_std, dtype=np.float32))
    v = c.vis
    use_img = not v.no_image and v.cnn is not None
    img_cnn: Dict[str, Any] | None = None
    if use_img:
        assert v.cnn is not None
        img_cnn = dict(vis_cnn_head_kw_from_nn_vis(v))
        tmp = _build_img_head(**img_cnn)
        conv_dim = calculate_conv_output_dim(tmp, c.H_downsized, c.W_downsized)
    else:
        conv_dim = 0
    n_act = len(c.inputs)
    return PpoActorCritic(
        float_inputs_dim=c.float_input_dim,
        float_hidden_dim=c.float_hidden_dim,
        dense_hidden_dim=c.dense_hidden_dimension,
        conv_head_output_dim=conv_dim,
        use_image_head=use_img,
        float_inputs_mean=mean,
        float_inputs_std=std,
        n_actions=n_act,
        n_actions_per_block=c.n_actions_per_block,
        img_cnn=img_cnn,
    )


def make_ppo_network_pair(jit: bool, is_inference: bool) -> Tuple[nn.Module, nn.Module]:
    """Return (possibly compiled network, uncompiled copy for shared memory)."""
    uncompiled = build_ppo_actor_critic_uncompiled()
    if not jit or not get_config().use_jit:
        model = uncompiled
    else:
        compile_mode = None if "rocm" in torch.__version__ else ("max-autotune" if is_inference else "max-autotune-no-cudagraphs")
        model = torch.compile(uncompiled, mode=compile_mode)
    # Match IQN: collectors / inferer use CUDA tensors (see ppo_wiring.PPOInferer).
    u = uncompiled.to(device="cuda", memory_format=torch.channels_last).train()
    m = model.to(device="cuda", memory_format=torch.channels_last).train()
    return m, u
