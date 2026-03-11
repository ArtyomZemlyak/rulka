"""
IQN for multi-label actions: the network predicts several actions at once (e.g. gas + left + brake
simultaneously). A_head outputs one logit per action dimension; greedy = all dimensions with
positive advantage (no softmax/argmax — multiple outputs can be 1). Q(s,a) uses the full action vector.
"""

import copy
import math
import random
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchrl.data import ReplayBuffer

from config_files.config_loader import get_config
from trackmania_rl import utilities


def calculate_conv_output_dim(height: int, width: int) -> int:
    """
    Dynamically calculate the output dimension of the CNN head based on input image dimensions.
    
    This function creates a temporary CNN head with the same architecture as IQN_Network
    and computes the output size by running a forward pass with a dummy input.
    
    The function works on CPU to avoid CUDA dependencies during configuration loading.
    
    Args:
        height: Input image height in pixels
        width: Input image width in pixels
        
    Returns:
        Output dimension after Flatten() (channels × final_height × final_width)
    """
    # Create the same CNN architecture as in IQN_Network
    img_head_channels = [1, 16, 32, 64, 32]
    activation_function = torch.nn.LeakyReLU
    img_head = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=img_head_channels[0], out_channels=img_head_channels[1], kernel_size=(4, 4), stride=2),
        activation_function(inplace=True),
        torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4), stride=2),
        activation_function(inplace=True),
        torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3), stride=2),
        activation_function(inplace=True),
        torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3), stride=1),
        activation_function(inplace=True),
        torch.nn.Flatten(),
    )
    
    # Create dummy input and run forward pass on CPU (no CUDA dependency)
    dummy_input = torch.zeros(1, 1, height, width)
    with torch.no_grad():
        output = img_head(dummy_input)
    
    return output.shape[1]  # Return the flattened dimension


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
    ):
        super().__init__()
        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.use_image_head = use_image_head
        activation_function = torch.nn.LeakyReLU
        if use_image_head:
            img_head_channels = [1, 16, 32, 64, 32]
            self.img_head = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=img_head_channels[0], out_channels=img_head_channels[1], kernel_size=(4, 4), stride=2),
                activation_function(inplace=True),
                torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4), stride=2),
                activation_function(inplace=True),
                torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3), stride=2),
                activation_function(inplace=True),
                torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3), stride=1),
                activation_function(inplace=True),
                torch.nn.Flatten(),
            )
        else:
            self.img_head = None
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            activation_function(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            activation_function(inplace=True),
        )

        dense_input_dimension = (conv_head_output_dim if use_image_head else 0) + float_hidden_dim

        # Multi-label head: one logit per dimension (accel, brake, left_1..N, right_1..N). No softmax —
        # greedy = (A > 0) per dim, so several actions can be active at once (e.g. gas+left+brake).
        self.A_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, n_actions),
        )
        self.V_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, 1),
        )
        self.iqn_fc = torch.nn.Sequential(torch.nn.Linear(iqn_embedding_dimension, dense_input_dimension), torch.nn.LeakyReLU(inplace=True))
        self.initialize_weights()

        self.n_actions = n_actions

        # States are not normalized when the method forward() is called. Normalization is done as the first step of the forward() method.
        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        lrelu_neg_slope = 1e-2
        activation_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)
        modules_to_init = [self.float_feature_extractor, self.A_head[:-1], self.V_head[:-1]]
        if self.img_head is not None:
            modules_to_init.insert(0, self.img_head)
        for module in modules_to_init:
            for m in module:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    utilities.init_orthogonal(m, activation_gain)
        utilities.init_orthogonal(
            self.iqn_fc[0], np.sqrt(2) * activation_gain
        )  # Since cosine has a variance of 1/2, and we would like to exit iqn_fc with a variance of 1, we need a weight variance double that of what a normal leaky relu would need
        for module in [self.A_head[-1], self.V_head[-1]]:
            utilities.init_orthogonal(module)

    def forward(
        self, img: torch.Tensor, float_inputs: torch.Tensor, num_quantiles: int, tau: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method implements the forward pass through the IQN neural network.

        The neural network is structured with two input heads:
            - one for images, with Conv2D layers
            - one for float features with Dense layers

        The embedding extracted by these two input heads are concatenated, mixed (Hadamard product) with an embedding for IQN quantiles.

        A dueling network architecture (https://arxiv.org/abs/1511.06581) is implemented, two output heads predict:
            - the value of a (state, quantile) pair
            - the advantage per action dimension (multi-label: several actions at once, e.g. gas+left+brake)

        A_head outputs one logit per dimension; Q(s,a) = V + sum(a_i*A_i) - sum(max(0,A_i)). No softmax.

        Args:
            img: a torch.Tensor of shape (batch_size, 1, H, W) and type float16 or float32, depending on context.
            float_inputs: a torch.Tensor of shape (batch_size, float_input_dim) and type float16 or float32, depending on context.
            num_quantiles: the number of quantiles, defined as N or N' in the IQN paper (https://arxiv.org/pdf/1806.06923).
            tau: if not None, a torch.Tensor of shape (batch_size * num_quantiles) the specifies the exact quantiles for which the neural network should return Q values
                 if None, the method will sample tau randomly in num_quantiles regularly spaced segments, and symmetrically around 0.5.

        Returns:
            V: (batch_size * num_quantiles, 1) value head output
            A: (batch_size * num_quantiles, n_action_dims) advantage per action dimension (multi-label dueling)
            tau: (batch_size * num_quantiles, 1) quantiles used. Use q_from_va(V, A, a) to get Q(s,a).
        """
        batch_size = img.shape[0]
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        if self.img_head is not None:
            img_outputs = self.img_head(img)
            concat = torch.cat((img_outputs, float_outputs), 1)  # (batch_size, dense_input_dimension)
        else:
            concat = float_outputs  # (batch_size, dense_input_dimension)
        if tau is None:
            tau = (
                torch.arange(num_quantiles // 2, device="cuda", dtype=torch.float32).repeat_interleave(batch_size).unsqueeze(1)
                + torch.rand(size=(batch_size * num_quantiles // 2, 1), device="cuda", dtype=torch.float32)
            ) / num_quantiles  # (batch_size * num_quantiles // 2, 1) (random numbers)
            tau = torch.cat((tau, 1 - tau), dim=0)  # ensure that tau are sampled symmetrically
        quantile_net = torch.cos(
            torch.arange(1, self.iqn_embedding_dimension + 1, 1, device="cuda") * math.pi * tau
        )  # (batch_size*num_quantiles, 1)
        quantile_net = quantile_net.expand(
            [-1, self.iqn_embedding_dimension]
        )  # (batch_size*num_quantiles, iqn_embedding_dimension) (still random numbers)
        # (8 or 32 initial random numbers, expanded with cos to iqn_embedding_dimension)
        # (batch_size*num_quantiles, dense_input_dimension)
        quantile_net = self.iqn_fc(quantile_net)
        # (batch_size*num_quantiles, dense_input_dimension)
        concat = concat.repeat(num_quantiles, 1)
        # (batch_size*num_quantiles, dense_input_dimension)
        concat = concat * quantile_net

        A = self.A_head(concat)  # (batch_size*num_quantiles, n_action_dims)
        V = self.V_head(concat)  # (batch_size*num_quantiles, 1)

        # Multi-label dueling: Q(s,a) = V + sum_i a_i*A_i - sum_i max(0, A_i); return (V, A) for caller to compute Q(s,a)
        return V, A, tau

    @staticmethod
    def q_from_va(V: torch.Tensor, A: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute Q(s,a) from dueling (V, A) and action vector a. a: (..., n_action_dims), A: (..., n_action_dims). Returns (..., 1)."""
        return (V + (A * a).sum(dim=-1, keepdim=True) - A.clamp(min=0).sum(dim=-1, keepdim=True))

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
                if get_config().prio_alpha > 0:
                    IS_weights = torch.from_numpy(batch_info["_weight"]).to("cuda", non_blocking=True)

                rewards = rewards.unsqueeze(-1).repeat(
                    [self.iqn_n, 1]
                )  # (batch_size*iqn_n, 1)
                gammas_terminal = gammas_terminal.unsqueeze(-1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                # actions: (batch_size, n_action_dims) -> repeat for quantiles (batch_size*iqn_n, n_action_dims)
                actions = actions.unsqueeze(0).repeat([self.iqn_n, 1, 1]).reshape(-1, actions.shape[-1])  # (batch_size*iqn_n, n_action_dims)
                #
                #   Target network: (V2, A2, tau2). Greedy a' = (A2 > 0).float(); Q(s', a') = q_from_va(V2, A2, a').
                #
                V2, A2, tau2 = self.target_network(
                    next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=None
                )  # V2 (batch*iqn_n, 1), A2 (batch*iqn_n, n_action_dims)
                if get_config().use_ddqn:
                    _, A_online, _ = self.online_network(
                        next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=None
                    )
                    A_online_mean = A_online.reshape([self.iqn_n, self.batch_size, -1]).mean(dim=0)  # (batch_size, n_action_dims)
                    a__tpo__greedy = (A_online_mean > 0).float().unsqueeze(0).repeat([self.iqn_n, 1, 1]).reshape(-1, A_online_mean.shape[-1])
                else:
                    a__tpo__greedy = (A2 > 0).float()  # (batch*iqn_n, n_action_dims)
                q__stpo__target__quantiles_tau2 = self.online_network.q_from_va(V2, A2, a__tpo__greedy)  # (batch*iqn_n, 1)
                outputs_target_tau2 = (
                    rewards + gammas_terminal * q__stpo__target__quantiles_tau2
                )  # (batch_size*iqn_n, 1)

                outputs_target_tau2 = outputs_target_tau2.reshape([self.iqn_n, self.batch_size, 1]).transpose(
                    0, 1
                )  # (batch_size, iqn_n, 1)

            V3, A3, tau3 = self.online_network(
                state_img_tensor, state_float_tensor, self.iqn_n, tau=None
            )  # V3 (batch*iqn_n, 1), A3 (batch*iqn_n, n_action_dims)
            q__st__online__quantiles_tau3 = self.online_network.q_from_va(V3, A3, actions)  # (batch_size*iqn_n, 1)
            outputs_tau3 = (
                q__st__online__quantiles_tau3.reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)
            )  # (batch_size, iqn_n, 1)

            loss = iqn_loss(outputs_target_tau2, outputs_tau3, tau3, get_config().iqn_n, get_config().batch_size)

            target_self_loss = torch.sqrt(
                iqn_loss(
                    outputs_target_tau2.detach(), outputs_target_tau2.detach(), tau2.detach(), get_config().iqn_n, get_config().batch_size
                )
            )

            self.typical_self_loss = 0.99 * self.typical_self_loss + 0.01 * target_self_loss.mean()

            correction_clamped = target_self_loss.clamp(min=self.typical_self_loss / get_config().target_self_loss_clamp_ratio)

            self.typical_clamped_self_loss = 0.99 * self.typical_clamped_self_loss + 0.01 * correction_clamped.mean()

            loss *= self.typical_clamped_self_loss / correction_clamped

            total_loss = torch.sum(IS_weights * loss if get_config().prio_alpha > 0 else loss)

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
                    torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), get_config().clip_grad_norm).detach().cpu().item()
                )
                torch.nn.utils.clip_grad_value_(self.online_network.parameters(), get_config().clip_grad_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = 0
                grad_norm_before_clip = 0

            total_loss = total_loss.detach().cpu()

            # Compute priority update data (CPU tensors) but don't write to buffer yet
            priority_update = None
            if get_config().prio_alpha > 0:
                mask_update_priority = torch.lt(state_float_tensor[:, 0], get_config().min_horizon_to_update_priority_actions).detach().cpu()
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
    )

    def __init__(self, inference_network, iqn_k, tau_epsilon_boltzmann):
        self.inference_network = inference_network
        self.iqn_k = iqn_k
        self.epsilon = None
        self.epsilon_boltzmann = None
        self.tau_epsilon_boltzmann = tau_epsilon_boltzmann
        self.is_explo = None

    def infer_network(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray, tau=None) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Perform inference of a single state through self.inference_network.

        Returns:
            V: (iqn_k, 1), A: (iqn_k, n_action_dims) — dueling outputs for multi-label Q(s,a) = V + (A*a).sum() - (A.clamp(min=0).sum()
        """
        with torch.no_grad():
            state_img_tensor = (
                torch.from_numpy(img_inputs_uint8)
                .unsqueeze(0)
                .to("cuda", memory_format=torch.channels_last, non_blocking=True, dtype=torch.float32)
                - 128
            ) / 128
            state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            V, A, _ = self.inference_network(
                state_img_tensor,
                state_float_tensor,
                self.iqn_k,
                tau=tau,
            )
            V = V.cpu().numpy().astype(np.float32)
            A = A.cpu().numpy().astype(np.float32)
            return V, A

    def get_exploration_action(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> Tuple[npt.NDArray, bool, float, npt.NDArray]:
        """
        Multi-label: returns action vector (n_action_dims,) where several entries can be 1 at once
        (e.g. gas + left + brake). Greedy = (A_mean > 0) per dimension — no argmax over actions.
        """
        V, A = self.infer_network(img_inputs_uint8, float_inputs)
        A_mean = A.mean(axis=0)  # (n_action_dims,)
        # Greedy = each dimension independently: positive advantage → 1 (allows gas+left+brake etc.)
        greedy_action = (A_mean > 0).astype(np.float32)
        r = random.random()

        if self.is_explo and r < self.epsilon:
            n_ad = self.inference_network.n_actions
            action_vector = (np.random.rand(n_ad) > 0.5).astype(np.float32)
        elif self.is_explo and r < self.epsilon + self.epsilon_boltzmann:
            get_on = A_mean + self.tau_epsilon_boltzmann * np.random.randn(*A_mean.shape)
            action_vector = (get_on > 0).astype(np.float32)
        else:
            action_vector = greedy_action

        is_greedy = np.allclose(action_vector, greedy_action)
        q_greedy = float(np.mean(V) + (A_mean * greedy_action).sum() - (np.maximum(A_mean, 0)).sum())
        return action_vector, is_greedy, q_greedy, A_mean


def make_untrained_iqn_network(jit: bool, is_inference: bool) -> Tuple[IQN_Network, IQN_Network]:
    """
    Constructs two identical copies of the IQN network.

    The first copy is compiled (if jit == True) and is used for inference, for rollouts, for training, etc...
    The second copy is never compiled and **only** used to efficiently share a neural network's weights between processes.

    Args:
        jit: a boolean indicating whether compilation should be used
    """
    use_image_head = get_config().use_iqn_image_head
    # When image head is disabled, conv_head_output_dim is 0; otherwise compute from image size
    conv_head_output_dim = (
        calculate_conv_output_dim(get_config().H_downsized, get_config().W_downsized)
        if use_image_head
        else 0
    )

    uncompiled_model = IQN_Network(
        float_inputs_dim=get_config().float_input_dim,
        float_hidden_dim=get_config().float_hidden_dim,
        conv_head_output_dim=conv_head_output_dim,
        dense_hidden_dimension=get_config().dense_hidden_dimension,
        iqn_embedding_dimension=get_config().iqn_embedding_dimension,
        n_actions=get_config().n_action_dims,
        float_inputs_mean=get_config().float_inputs_mean,
        float_inputs_std=get_config().float_inputs_std,
        use_image_head=use_image_head,
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
