"""
In this file, we define:
    - The IQN_Network class, which defines the neural network's structure.
    - The Trainer class, which implements the IQN training logic in method train_on_batch.
    - The Inferer class, which implements utilities for forward propagation with and without exploration.
"""

import copy
import math
import random
from math import sqrt
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init
from torchrl.data import ReplayBuffer

from config_files.config_loader import get_config
from trackmania_rl import utilities


# ---------------------------------------------------------------------------
#  BTR building blocks
# ---------------------------------------------------------------------------


class FactorizedNoisyLinear(nn.Module):
    """Factorized Gaussian noise layer (NoisyNet, Fortunato et al. 2018)."""

    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()
        self.disable_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        scale = 1.0 / sqrt(self.in_features)
        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)
        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _factored_noise(self, size: int) -> Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        eps_in = self._factored_noise(self.in_features)
        eps_out = self._factored_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon.zero_()
        self.bias_epsilon.zero_()

    def forward(self, x: Tensor) -> Tensor:
        # Use matmul instead of F.linear: torch.compile(max-autotune) can produce
        # wrong weight gradients for F.linear when out_features==1 (e.g. [512] vs [1, 512]),
        # and the bug can show up only under autocast / after recompilation.
        w = self.weight_mu + self.weight_sigma * self.weight_epsilon
        b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return x.matmul(w.T) + b


class MatmulLinear(nn.Linear):
    """Like nn.Linear but forward uses matmul, not F.linear.

    torch.compile(max-autotune + autocast) can mis-compile F.linear backward when
    out_features==1 (weight shape [1, in]); same class of bug as FactorizedNoisyLinear.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.matmul(self.weight.T) + self.bias


class ImpalaCNNResidual(nn.Module):
    """Single residual block used inside each IMPALA CNN stage."""

    def __init__(self, depth: int, norm_func, activation_cls=nn.ReLU):
        super().__init__()
        self.activation = activation_cls()
        self.conv_0 = norm_func(nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        x_ = self.conv_0(self.activation(x))
        x_ = self.conv_1(self.activation(x_))
        return x + x_


class ImpalaCNNBlock(nn.Module):
    """One IMPALA stage: conv -> maxpool -> 2x residual."""

    def __init__(self, depth_in: int, depth_out: int, norm_func, activation_cls=nn.ReLU):
        super().__init__()
        self.conv = norm_func(nn.Conv2d(depth_in, depth_out, kernel_size=3, stride=1, padding=1))
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func, activation_cls)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func, activation_cls)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


def calculate_conv_output_dim(img_head: nn.Module, height: int, width: int) -> int:
    """Return the flattened output dimension of *img_head* given (1, 1, H, W) input.

    Works on CPU so it can be called at config-load time without CUDA.
    """
    dummy = torch.zeros(1, 1, height, width)
    with torch.no_grad():
        out = img_head(dummy)
    return out.shape[1]


def _build_img_head(
    *,
    use_impala_cnn: bool,
    impala_model_size: int,
    use_spectral_norm: bool,
    use_adaptive_maxpool: bool,
    adaptive_maxpool_size: int,
) -> nn.Sequential:
    """Build the image encoder (CNN) based on config flags. Returns an nn.Sequential ending with Flatten."""
    identity = lambda m: m  # noqa: E731
    norm_func = torch.nn.utils.spectral_norm if use_spectral_norm else identity

    if use_impala_cnn:
        s = impala_model_size
        layers: list[nn.Module] = [
            ImpalaCNNBlock(1, 16 * s, norm_func),
            ImpalaCNNBlock(16 * s, 32 * s, norm_func),
            ImpalaCNNBlock(32 * s, 32 * s, norm_func),
            nn.ReLU(inplace=True),
        ]
    else:
        ch = [1, 16, 32, 64, 32]
        act = nn.LeakyReLU
        layers = [
            norm_func(nn.Conv2d(ch[0], ch[1], kernel_size=4, stride=2)), act(inplace=True),
            norm_func(nn.Conv2d(ch[1], ch[2], kernel_size=4, stride=2)), act(inplace=True),
            norm_func(nn.Conv2d(ch[2], ch[3], kernel_size=3, stride=2)), act(inplace=True),
            norm_func(nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=1)), act(inplace=True),
        ]

    if use_adaptive_maxpool:
        layers.append(nn.AdaptiveMaxPool2d((adaptive_maxpool_size, adaptive_maxpool_size)))

    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


class IQN_Network(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim: int,
        float_hidden_dim: int,
        conv_head_output_dim: int,
        dense_hidden_dimension: int,
        iqn_embedding_dimension: int,
        n_actions: int,
        float_inputs_mean: npt.NDArray,
        float_inputs_std: npt.NDArray,
        use_image_head: bool = True,
        n_actions_per_block: int = 1,
        # BTR flags
        use_impala_cnn: bool = False,
        impala_model_size: int = 2,
        use_adaptive_maxpool: bool = False,
        adaptive_maxpool_size: int = 6,
        use_spectral_norm: bool = False,
        use_layer_norm: bool = False,
        use_noisy_linear: bool = False,
        noisy_sigma0: float = 0.5,
    ):
        super().__init__()
        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.use_image_head = use_image_head
        self.n_actions_per_block = n_actions_per_block
        self.use_noisy_linear = use_noisy_linear
        activation_function = torch.nn.LeakyReLU

        # --- linear layer factory (NoisyLinear vs plain Linear) ---
        def _linear(in_f: int, out_f: int) -> nn.Module:
            if use_noisy_linear:
                return FactorizedNoisyLinear(in_f, out_f, sigma_0=noisy_sigma0)
            return nn.Linear(in_f, out_f)

        # --- Image head ---
        if use_image_head:
            self.img_head = _build_img_head(
                use_impala_cnn=use_impala_cnn,
                impala_model_size=impala_model_size,
                use_spectral_norm=use_spectral_norm,
                use_adaptive_maxpool=use_adaptive_maxpool,
                adaptive_maxpool_size=adaptive_maxpool_size,
            )
        else:
            self.img_head = None

        # --- Float feature extractor ---
        if use_layer_norm:
            self.float_feature_extractor = nn.Sequential(
                nn.Linear(float_inputs_dim, float_hidden_dim),
                nn.LayerNorm(float_hidden_dim),
                activation_function(inplace=True),
                nn.Linear(float_hidden_dim, float_hidden_dim),
                nn.LayerNorm(float_hidden_dim),
                activation_function(inplace=True),
            )
        else:
            self.float_feature_extractor = nn.Sequential(
                nn.Linear(float_inputs_dim, float_hidden_dim),
                activation_function(inplace=True),
                nn.Linear(float_hidden_dim, float_hidden_dim),
                activation_function(inplace=True),
            )

        dense_input_dimension = (conv_head_output_dim if use_image_head else 0) + float_hidden_dim
        a_head_hidden = dense_hidden_dimension // 2

        # --- Dueling A head ---
        if n_actions_per_block <= 1:
            if use_layer_norm:
                self.A_head = nn.Sequential(
                    _linear(dense_input_dimension, a_head_hidden),
                    nn.LayerNorm(a_head_hidden),
                    activation_function(inplace=True),
                    _linear(a_head_hidden, n_actions),
                )
            else:
                self.A_head = nn.Sequential(
                    _linear(dense_input_dimension, a_head_hidden),
                    activation_function(inplace=True),
                    _linear(a_head_hidden, n_actions),
                )
            self.A_head_multi = None
        else:
            if use_layer_norm:
                self.A_head = nn.Sequential(
                    _linear(dense_input_dimension, a_head_hidden),
                    nn.LayerNorm(a_head_hidden),
                    activation_function(inplace=True),
                )
            else:
                self.A_head = nn.Sequential(
                    _linear(dense_input_dimension, a_head_hidden),
                    activation_function(inplace=True),
                )
            self.A_head_multi = _linear(a_head_hidden, n_actions_per_block * n_actions)

        # --- Dueling V head ---
        # Last layer MatmulLinear: out_features=1 + plain nn.Linear uses F.linear internally,
        # which still breaks torch.compile backward (same [512] vs [1,512] gradient bug).
        if use_layer_norm:
            self.V_head = nn.Sequential(
                _linear(dense_input_dimension, dense_hidden_dimension // 2),
                nn.LayerNorm(dense_hidden_dimension // 2),
                activation_function(inplace=True),
                MatmulLinear(dense_hidden_dimension // 2, 1),
            )
        else:
            self.V_head = nn.Sequential(
                _linear(dense_input_dimension, dense_hidden_dimension // 2),
                activation_function(inplace=True),
                MatmulLinear(dense_hidden_dimension // 2, 1),
            )

        # --- IQN cosine embedding ---
        self.iqn_fc = nn.Sequential(
            nn.Linear(iqn_embedding_dimension, dense_input_dimension),
            nn.LeakyReLU(inplace=True),
        )

        self.initialize_weights()

        self.n_actions = n_actions

        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        lrelu_neg_slope = 1e-2
        activation_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)

        def _should_init(m: nn.Module) -> bool:
            if isinstance(m, FactorizedNoisyLinear):
                return False
            return isinstance(m, (nn.Conv2d, nn.Linear))

        def _orthogonal_init(m: nn.Module, gain: float):
            """Orthogonal init that targets weight_orig when spectral_norm hook is active."""
            w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
            torch.nn.init.orthogonal_(w, gain=gain)
            torch.nn.init.zeros_(m.bias)

        a_head_first = self.A_head[:-1] if self.A_head_multi is None else self.A_head
        modules_to_init = [self.float_feature_extractor, a_head_first, self.V_head[:-1]]
        if self.img_head is not None:
            modules_to_init.insert(0, self.img_head)
        for module in modules_to_init:
            for m in module.modules():
                if _should_init(m):
                    _orthogonal_init(m, activation_gain)

        utilities.init_orthogonal(
            self.iqn_fc[0], np.sqrt(2) * activation_gain
        )

        # Last layer(s) of A/V heads -- output layers get gain=1
        def _init_last(layer: nn.Module):
            if isinstance(layer, FactorizedNoisyLinear):
                return  # already initialized
            utilities.init_orthogonal(layer)

        if self.A_head_multi is None:
            _init_last(self.A_head[-1])
        else:
            _init_last(self.A_head_multi)
        _init_last(self.V_head[-1])

    # --- NoisyNet helpers ---

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()

    def disable_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.disable_noise()

    # --- Forward pass ---

    def forward(
        self, img: torch.Tensor, float_inputs: torch.Tensor, num_quantiles: int, tau: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = img.shape[0]
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        if self.img_head is not None:
            img_outputs = self.img_head(img)
            concat = torch.cat((img_outputs, float_outputs), 1)
        else:
            concat = float_outputs
        if tau is None:
            tau = (
                torch.arange(num_quantiles // 2, device="cuda", dtype=torch.float32).repeat_interleave(batch_size).unsqueeze(1)
                + torch.rand(size=(batch_size * num_quantiles // 2, 1), device="cuda", dtype=torch.float32)
            ) / num_quantiles
            tau = torch.cat((tau, 1 - tau), dim=0)
        quantile_net = torch.cos(
            torch.arange(1, self.iqn_embedding_dimension + 1, 1, device="cuda") * math.pi * tau
        )
        quantile_net = quantile_net.expand([-1, self.iqn_embedding_dimension])
        quantile_net = self.iqn_fc(quantile_net)
        concat = concat.repeat(num_quantiles, 1)
        concat = concat * quantile_net

        V = self.V_head(concat)
        if self.A_head_multi is None:
            A = self.A_head(concat)
            Q = V + A - A.mean(dim=-1).unsqueeze(-1)
            return Q, tau
        a_hidden = self.A_head(concat)
        A = self.A_head_multi(a_hidden).view(-1, self.n_actions_per_block, self.n_actions)
        Q = V.unsqueeze(1) + A - A.mean(dim=-1, keepdim=True)
        return Q, tau

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


# Decorator evaluated at import time; worker processes may not have config set — disable compile there.
# torch.compile works on Windows since PyTorch 2.3+ (requires `pip install triton-windows` for CUDA).
try:
    _iqn_compile_disable = not get_config().use_jit
except RuntimeError:
    _iqn_compile_disable = True


@torch.compile(disable=_iqn_compile_disable, dynamic=False)
def iqn_loss(targets: torch.Tensor, outputs: torch.Tensor, tau_outputs: torch.Tensor, num_quantiles: int, batch_size: int):
    """
    Implements the IQN loss as defined in the IQN paper (https://arxiv.org/pdf/1806.06923)

    Args:
        targets: a torch.Tensor of shape (batch_size, num_quantiles, 1)
        outputs: a torch.Tensor of shape (batch_size, num_quantiles, 1)
        tau_outputs: a torch.Tensor of shape (batch_size * num_quantiles, 1)
        num_quantiles: (int)
        batch_size: (int)

    Returns:
        loss: a torch.Tensor of shape (batch_size, )
    """
    TD_error = targets[:, :, None, :] - outputs[:, None, :, :]
    # (batch_size, iqn_n, iqn_n, 1)
    loss = torch.where(
        torch.lt(torch.abs(TD_error), get_config().iqn_kappa),
        (0.5 / get_config().iqn_kappa) * TD_error**2,
        (torch.abs(TD_error) - 0.5 * get_config().iqn_kappa),
    )
    tau = tau_outputs.reshape([num_quantiles, batch_size, 1]).transpose(0, 1)  # (batch_size, iqn_n, 1)
    tau = tau[:, None, :, :].expand([-1, num_quantiles, -1, -1])  # (batch_size, iqn_n, iqn_n, 1)
    loss = (torch.where(torch.lt(TD_error, 0), 1 - tau, tau) * loss).sum(dim=2).mean(dim=1)[:, 0]  # pinball loss # (batch_size, )
    return loss


class Trainer:
    __slots__ = (
        "online_network",
        "target_network",
        "optimizer",
        "scaler",
        "batch_size",
        "iqn_n",
        "typical_self_loss",
        "typical_clamped_self_loss",
    )

    def __init__(
        self,
        online_network: IQN_Network,
        target_network: IQN_Network,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        batch_size: int,
        iqn_n: int,
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.scaler = scaler
        self.batch_size = batch_size
        self.iqn_n = iqn_n
        self.typical_self_loss = 0.01
        self.typical_clamped_self_loss = 0.01

    def sample_batch(self, buffer: ReplayBuffer):
        """
        Phase A: Sample a batch from the replay buffer (CPU operation, needs buffer_lock).

        Returns:
            batch: tuple of tensors (already on GPU via collate_fn)
            batch_info: dict with sampling metadata (indices, weights)
        """
        return buffer.sample(self.batch_size, return_info=True)

    def train_on_data(self, batch, batch_info, do_learn: bool):
        """
        Phase B: GPU compute — forward pass, loss, backward, optimizer step.
        Does NOT touch the buffer. No lock needed.

        Args:
            batch: tuple from sample_batch (tensors already on GPU)
            batch_info: dict from sample_batch
            do_learn: whether to backprop and update weights

        Returns:
            total_loss: a float
            grad_norm: a float (after clipping)
            grad_norm_before_clip: a float (before clipping)
            priority_update: None or (indices, priorities) tuple for Phase C
        """
        cfg = get_config()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                (
                    state_img_tensor,
                    state_float_tensor,
                    actions,
                    rewards,
                    next_state_img_tensor,
                    next_state_float_tensor,
                    gammas_terminal,
                ) = batch
                if cfg.prio_alpha > 0:
                    IS_weights = torch.from_numpy(batch_info["_weight"]).to("cuda", non_blocking=True)

                n_ab = self.online_network.n_actions_per_block
                n_act = self.online_network.n_actions
                # Keep original (batch_size,) actions for Munchausen before repeat
                actions_orig = actions.clone()  # (batch_size, N) or (batch_size, 1)
                rewards = rewards.unsqueeze(-1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                gammas_terminal = gammas_terminal.unsqueeze(-1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                actions = actions.repeat([self.iqn_n, 1])  # (batch_size*iqn_n, N)

                q__stpo__target__quantiles_tau2, tau2 = self.target_network(
                    next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=None
                )

                if cfg.use_munchausen:
                    # --- Munchausen IQN targets ---
                    m_tau = cfg.munchausen_entropy_tau
                    m_alpha = cfg.munchausen_alpha
                    m_lo = cfg.munchausen_lo

                    if n_ab <= 1:
                        # q_next: (iqn_n*batch, n_actions) -> mean over quantiles -> (batch, n_actions)
                        q_next_mean = q__stpo__target__quantiles_tau2.reshape(
                            self.iqn_n, self.batch_size, n_act
                        ).mean(dim=0)

                        # Soft policy over next actions: log π = log softmax(Q/τ)
                        q_next_centered = q_next_mean - q_next_mean.max(1, keepdim=True)[0]
                        logsum_next = torch.logsumexp(q_next_centered / m_tau, dim=1, keepdim=True)
                        log_pi_next = q_next_centered / m_tau - logsum_next  # (batch, n_actions)
                        pi_next = F.softmax(q_next_mean / m_tau, dim=1)  # (batch, n_actions)

                        # Soft V(s') per quantile: Σ_a π(a) * (Q(a) - τ * log π(a))
                        q_next_per_quant = q__stpo__target__quantiles_tau2.reshape(
                            self.iqn_n, self.batch_size, n_act
                        )  # (iqn_n, batch, n_actions)
                        v_next = (pi_next.unsqueeze(0) * (q_next_per_quant - m_tau * log_pi_next.unsqueeze(0))).sum(dim=2)
                        v_next = v_next.reshape(self.iqn_n * self.batch_size, 1)  # (iqn_n*batch, 1)

                        # Munchausen bonus: α * τ * clamp(log π(a_t | s_t), lo, 0)
                        q_cur, _ = self.target_network(
                            state_img_tensor, state_float_tensor, self.iqn_n, tau=None
                        )
                        q_cur_mean = q_cur.reshape(self.iqn_n, self.batch_size, n_act).mean(dim=0)
                        q_cur_centered = q_cur_mean - q_cur_mean.max(1, keepdim=True)[0]
                        logsum_cur = torch.logsumexp(q_cur_centered / m_tau, dim=1, keepdim=True)
                        log_pi_cur = q_cur_centered / m_tau - logsum_cur
                        munch_bonus = m_alpha * m_tau * torch.clamp(
                            log_pi_cur.gather(1, actions_orig), min=m_lo, max=0
                        )  # (batch, 1)
                        munch_bonus = munch_bonus.repeat(self.iqn_n, 1)  # (iqn_n*batch, 1)

                        outputs_target_tau2 = (rewards + munch_bonus) + gammas_terminal * v_next
                    else:
                        # Multi-action Munchausen: per-head soft policy, sum over heads
                        # q_next: (iqn_n*batch, N, n_actions)
                        q_next_mean = q__stpo__target__quantiles_tau2.reshape(
                            self.iqn_n, self.batch_size, n_ab, n_act
                        ).mean(dim=0)  # (batch, N, n_actions)

                        q_next_centered = q_next_mean - q_next_mean.max(2, keepdim=True)[0]
                        logsum_next = torch.logsumexp(q_next_centered / m_tau, dim=2, keepdim=True)
                        log_pi_next = q_next_centered / m_tau - logsum_next  # actual log π
                        pi_next = F.softmax(q_next_mean / m_tau, dim=2)  # (batch, N, n_actions)

                        q_next_per_quant = q__stpo__target__quantiles_tau2.reshape(
                            self.iqn_n, self.batch_size, n_ab, n_act
                        )
                        v_next_per_head = (
                            pi_next.unsqueeze(0) * (q_next_per_quant - m_tau * log_pi_next.unsqueeze(0))
                        ).sum(dim=3)  # (iqn_n, batch, N)
                        v_next = v_next_per_head.sum(dim=2).reshape(self.iqn_n * self.batch_size, 1)

                        q_cur, _ = self.target_network(
                            state_img_tensor, state_float_tensor, self.iqn_n, tau=None
                        )
                        q_cur_mean = q_cur.reshape(self.iqn_n, self.batch_size, n_ab, n_act).mean(dim=0)
                        q_cur_centered = q_cur_mean - q_cur_mean.max(2, keepdim=True)[0]
                        logsum_cur = torch.logsumexp(q_cur_centered / m_tau, dim=2, keepdim=True)
                        log_pi_cur = q_cur_centered / m_tau - logsum_cur  # actual log π
                        munch_per_head = m_alpha * m_tau * torch.clamp(
                            torch.gather(log_pi_cur, 2, actions_orig.unsqueeze(-1)),
                            min=m_lo, max=0,
                        ).sum(dim=1)  # (batch, 1)
                        munch_bonus = munch_per_head.repeat(self.iqn_n, 1)

                        outputs_target_tau2 = (rewards + munch_bonus) + gammas_terminal * v_next
                else:
                    # --- Standard DDQN / max targets ---
                    if n_ab <= 1:
                        if cfg.use_ddqn:
                            a__tpo__online__reduced_repeated = (
                                self.online_network(
                                    next_state_img_tensor,
                                    next_state_float_tensor,
                                    self.iqn_n,
                                    tau=None,
                                )[0]
                                .reshape([self.iqn_n, self.batch_size, n_act])
                                .mean(dim=0)
                                .argmax(dim=1, keepdim=True)
                                .repeat([self.iqn_n, 1])
                            )
                            outputs_target_tau2 = rewards + gammas_terminal * q__stpo__target__quantiles_tau2.gather(
                                1, a__tpo__online__reduced_repeated
                            )
                        else:
                            outputs_target_tau2 = (
                                rewards + gammas_terminal * q__stpo__target__quantiles_tau2.max(dim=1, keepdim=True)[0]
                            )
                    else:
                        if cfg.use_ddqn:
                            q_online_next, _ = self.online_network(
                                next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=None
                            )
                            q_on_next = q_online_next.reshape(
                                [self.iqn_n, self.batch_size, n_ab, n_act]
                            ).mean(dim=0)
                            a_next = q_on_next.argmax(dim=2)
                            a_next_repeated = a_next.repeat([self.iqn_n, 1])
                            target_gather = torch.gather(
                                q__stpo__target__quantiles_tau2, 2, a_next_repeated.unsqueeze(-1)
                            ).sum(dim=1)
                            outputs_target_tau2 = rewards + gammas_terminal * target_gather
                        else:
                            target_sum = q__stpo__target__quantiles_tau2.max(dim=2)[0].sum(dim=1, keepdim=True)
                            outputs_target_tau2 = rewards + gammas_terminal * target_sum

                outputs_target_tau2 = outputs_target_tau2.reshape([self.iqn_n, self.batch_size, 1]).transpose(
                    0, 1
                )  # (batch_size, iqn_n, 1)

            q__st__online__quantiles_tau3, tau3 = self.online_network(
                state_img_tensor, state_float_tensor, self.iqn_n, tau=None
            )
            if n_ab <= 1:
                outputs_tau3 = (
                    q__st__online__quantiles_tau3.gather(1, actions).reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)
                )
            else:
                # Factorized Q: sum_i Q_i(s)[a_i] — single vectorized gather
                current_q = torch.gather(
                    q__st__online__quantiles_tau3, 2, actions.unsqueeze(-1)
                ).sum(dim=1)  # (batch*iqn_n, 1)
                outputs_tau3 = current_q.reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)

            loss = iqn_loss(outputs_target_tau2, outputs_tau3, tau3, cfg.iqn_n, cfg.batch_size)

            target_self_loss = torch.sqrt(
                iqn_loss(
                    outputs_target_tau2.detach(), outputs_target_tau2.detach(), tau2.detach(), cfg.iqn_n, cfg.batch_size
                )
            )

            self.typical_self_loss = 0.99 * self.typical_self_loss + 0.01 * target_self_loss.mean()

            correction_clamped = target_self_loss.clamp(min=self.typical_self_loss / cfg.target_self_loss_clamp_ratio)

            self.typical_clamped_self_loss = 0.99 * self.typical_clamped_self_loss + 0.01 * correction_clamped.mean()

            loss *= self.typical_clamped_self_loss / correction_clamped

            total_loss = torch.sum(IS_weights * loss if cfg.prio_alpha > 0 else loss)

            if do_learn:
                self.scaler.scale(total_loss).backward()

                # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self.scaler.unscale_(self.optimizer)
                
                # Calculate gradient norm BEFORE clipping for monitoring
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    self.online_network.parameters(), float('inf')
                ).detach().cpu().item()
                
                # Now clip gradients
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), cfg.clip_grad_norm).detach().cpu().item()
                )
                torch.nn.utils.clip_grad_value_(self.online_network.parameters(), cfg.clip_grad_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = 0
                grad_norm_before_clip = 0

            total_loss = total_loss.detach().cpu()

            # Compute priority update data (CPU tensors) but don't write to buffer yet
            priority_update = None
            if cfg.prio_alpha > 0:
                mask_update_priority = torch.lt(state_float_tensor[:, 0], cfg.min_horizon_to_update_priority_actions).detach().cpu()
                priority_indices = batch_info["index"][mask_update_priority]
                priority_values = (
                    (outputs_tau3.mean(axis=1) - outputs_target_tau2.mean(axis=1))
                    .abs()[mask_update_priority]
                    .detach()
                    .cpu()
                    .type(torch.float64)
                )
                priority_update = (priority_indices, priority_values)

        return total_loss, grad_norm, grad_norm_before_clip, priority_update

    @staticmethod
    def apply_priority_update(buffer: ReplayBuffer, priority_update):
        """
        Phase C: Write priority updates back to the buffer (CPU operation, needs buffer_lock).

        Args:
            buffer: the replay buffer
            priority_update: (indices, priorities) tuple from train_on_data, or None
        """
        if priority_update is not None:
            indices, priorities = priority_update
            buffer.update_priority(indices, priorities)

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        """
        Backward-compatible wrapper that calls all three phases.
        Use sample_batch + train_on_data + apply_priority_update for fine-grained locking.

        Returns:
            total_loss, grad_norm, grad_norm_before_clip
        """
        batch, batch_info = self.sample_batch(buffer)
        total_loss, grad_norm, grad_norm_before_clip, priority_update = self.train_on_data(batch, batch_info, do_learn)
        self.apply_priority_update(buffer, priority_update)
        return total_loss, grad_norm, grad_norm_before_clip


class Inferer:
    __slots__ = (
        "inference_network",
        "iqn_k",
        "epsilon",
        "epsilon_boltzmann",
        "tau_epsilon_boltzmann",
        "is_explo",
        "use_noisy_linear",
    )

    def __init__(self, inference_network, iqn_k, tau_epsilon_boltzmann):
        self.inference_network = inference_network
        self.iqn_k = iqn_k
        self.epsilon = None
        self.epsilon_boltzmann = None
        self.tau_epsilon_boltzmann = tau_epsilon_boltzmann
        self.is_explo = None
        self.use_noisy_linear = inference_network.use_noisy_linear

    def infer_network(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray, tau=None) -> npt.NDArray:
        """
        Perform inference of a single state through self.inference_network.

        Args:
            img_inputs_uint8:   a numpy array of shape (1, H, W) and dtype np.uint8
            float_inputs:       a numpy array of shape (float_input_dim, ) and dtype np.float32
            tau:                a torch.Tensor of shape (iqn_k,  1)

        Returns:
            q_values: when n_actions_per_block==1, shape (iqn_k, n_actions); when N>1, shape (iqn_k, N, n_actions).
        """
        with torch.no_grad():
            state_img_tensor = (
                torch.from_numpy(img_inputs_uint8)
                .unsqueeze(0)
                .to("cuda", memory_format=torch.channels_last, non_blocking=True, dtype=torch.float32)
                - 128
            ) / 128
            state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            q_values = (
                self.inference_network(
                    state_img_tensor,
                    state_float_tensor,
                    self.iqn_k,
                    tau=tau,
                )[0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return q_values

    def get_exploration_action(
        self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray
    ) -> Tuple[Union[int, npt.NDArray], bool, float, npt.NDArray]:
        """
        Selects an action (or block of N actions) according to the exploration strategy.

        When use_noisy_linear is enabled, NoisyNets provides state-dependent exploration:
        noise is active during explo (reset_noise called externally per step) and
        disabled during eval. Epsilon-greedy / Boltzmann perturbation is skipped.

        When use_noisy_linear is disabled, the existing epsilon-greedy + Boltzmann is used.
        """
        # NoisyNets: set noise state before inference
        if self.use_noisy_linear:
            if self.is_explo:
                self.inference_network.reset_noise()
            else:
                self.inference_network.disable_noise()

        n_actions_per_block = self.inference_network.n_actions_per_block
        q_raw = self.infer_network(img_inputs_uint8, float_inputs)
        q_values = q_raw.mean(axis=0)
        n_actions = self.inference_network.n_actions

        if n_actions_per_block <= 1:
            if self.use_noisy_linear:
                # NoisyNets exploration: noise is already baked into Q-values
                get_argmax_on = q_values
            else:
                r = random.random()
                if self.is_explo and r < self.epsilon:
                    get_argmax_on = np.random.randn(*q_values.shape)
                elif self.is_explo and r < self.epsilon + self.epsilon_boltzmann:
                    get_argmax_on = q_values + self.tau_epsilon_boltzmann * np.random.randn(*q_values.shape)
                else:
                    get_argmax_on = q_values
            action_chosen_idx = int(np.argmax(get_argmax_on))
            greedy_action_idx = int(np.argmax(q_values))
            return (
                action_chosen_idx,
                action_chosen_idx == greedy_action_idx,
                float(np.max(q_values)),
                q_values,
            )

        # Multi-action
        greedy_actions = np.argmax(q_values, axis=1)

        if self.use_noisy_linear:
            chosen = greedy_actions.copy()
        else:
            multi_mode = get_config().multi_action_exploration
            if multi_mode == "per_block":
                r = random.random()
                if self.is_explo and r < self.epsilon:
                    chosen = np.random.randint(0, n_actions, size=n_actions_per_block)
                elif self.is_explo and r < self.epsilon + self.epsilon_boltzmann:
                    perturbed = q_values + self.tau_epsilon_boltzmann * np.random.randn(*q_values.shape)
                    chosen = np.argmax(perturbed, axis=1)
                else:
                    chosen = greedy_actions.copy()
            else:
                rs = np.random.random(n_actions_per_block)
                random_mask = self.is_explo & (rs < self.epsilon)
                boltz_mask = self.is_explo & ~random_mask & (rs < self.epsilon + self.epsilon_boltzmann)

                chosen = greedy_actions.copy()
                if random_mask.any():
                    chosen[random_mask] = np.random.randint(0, n_actions, size=int(random_mask.sum()))
                if boltz_mask.any():
                    perturbed = q_values[boltz_mask] + self.tau_epsilon_boltzmann * np.random.randn(int(boltz_mask.sum()), n_actions)
                    chosen[boltz_mask] = np.argmax(perturbed, axis=1)

        actions_arr = chosen.astype(np.int64)
        is_greedy = bool(np.all(actions_arr == greedy_actions))
        value = float(np.sum(np.max(q_values, axis=1)))
        return (actions_arr, is_greedy, value, q_values)


def make_untrained_iqn_network(jit: bool, is_inference: bool) -> Tuple[IQN_Network, IQN_Network]:
    """
    Constructs two identical copies of the IQN network.

    The first copy is compiled (if jit == True) and is used for inference, for rollouts, for training, etc...
    The second copy is never compiled and **only** used to efficiently share a neural network's weights between processes.

    Args:
        jit: a boolean indicating whether compilation should be used
    """
    cfg = get_config()
    use_image_head = cfg.use_iqn_image_head

    # BTR flags
    btr_kwargs = dict(
        use_impala_cnn=cfg.use_impala_cnn,
        impala_model_size=cfg.impala_model_size,
        use_adaptive_maxpool=cfg.use_adaptive_maxpool,
        adaptive_maxpool_size=cfg.adaptive_maxpool_size,
        use_spectral_norm=cfg.use_spectral_norm,
        use_layer_norm=cfg.use_layer_norm,
        use_noisy_linear=cfg.use_noisy_linear,
        noisy_sigma0=cfg.noisy_sigma0,
    )

    if use_image_head:
        tmp_head = _build_img_head(
            use_impala_cnn=cfg.use_impala_cnn,
            impala_model_size=cfg.impala_model_size,
            use_spectral_norm=cfg.use_spectral_norm,
            use_adaptive_maxpool=cfg.use_adaptive_maxpool,
            adaptive_maxpool_size=cfg.adaptive_maxpool_size,
        )
        conv_head_output_dim = calculate_conv_output_dim(tmp_head, cfg.H_downsized, cfg.W_downsized)
    else:
        conv_head_output_dim = 0

    uncompiled_model = IQN_Network(
        float_inputs_dim=cfg.float_input_dim,
        float_hidden_dim=cfg.float_hidden_dim,
        conv_head_output_dim=conv_head_output_dim,
        dense_hidden_dimension=cfg.dense_hidden_dimension,
        iqn_embedding_dimension=cfg.iqn_embedding_dimension,
        n_actions=len(cfg.inputs),
        float_inputs_mean=cfg.float_inputs_mean,
        float_inputs_std=cfg.float_inputs_std,
        use_image_head=use_image_head,
        n_actions_per_block=cfg.n_actions_per_block,
        **btr_kwargs,
    )
    if jit:
        # torch.compile; multi-process stability is handled by warmup in main process + collector warmup under game_spawning_lock (see train.py, collector_process.py).

        # On ROCm, compile_mode is set to None (max-autotune not supported).
        compile_mode = None if "rocm" in torch.__version__ else ("max-autotune" if is_inference else "max-autotune-no-cudagraphs")
        
        try:
            model = torch.compile(uncompiled_model, dynamic=False, mode=compile_mode)
            print(f"[OK] torch.compile enabled (mode={compile_mode})")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}). Falling back to uncompiled model.")
            print(f"  Hint: On Windows, install Triton with: pip install triton-windows")
            model = copy.deepcopy(uncompiled_model)
    else:
        model = copy.deepcopy(uncompiled_model)
    return (
        model.to(device="cuda", memory_format=torch.channels_last).train(),
        uncompiled_model.to(device="cuda", memory_format=torch.channels_last).train(),
    )
