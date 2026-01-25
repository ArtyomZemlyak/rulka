"""
Neural network architecture configuration.
See docs/source/configuration_guide.rst for detailed documentation.
"""

from config_files.environment_config import (
    n_zone_centers_in_inputs,
    n_prev_actions_in_inputs,
    n_contact_material_physics_behavior_types,
)

# Input dimensions
float_input_dim = (
    27  # Base features
    + 3 * n_zone_centers_in_inputs  # Waypoint coordinates (X,Y,Z)
    + 4 * n_prev_actions_in_inputs  # Action history
    + 4 * n_contact_material_physics_behavior_types  # Contact materials
    + 1  # Additional features
)

# Network architecture
float_hidden_dim = 256  # Hidden layer for scalar features
conv_head_output_dim = 5632  # CNN output dimension
dense_hidden_dimension = 1024  # Main hidden layer size

# IQN parameters
iqn_embedding_dimension = 64  # Quantile embedding dimension
iqn_n = 8  # Quantile samples during training (must be even)
iqn_k = 32  # Quantile samples during inference (must be even)
iqn_kappa = 5e-3  # Huber loss threshold

# Q-learning variant
use_ddqn = False  # Use Double DQN

# Gradient clipping
clip_grad_value = 1000  # Max absolute gradient value
clip_grad_norm = 30  # Max gradient L2 norm

# Network updates
number_memories_trained_on_between_target_network_updates = 2048  # Target network update frequency
soft_update_tau = 0.02  # Soft update coefficient
target_self_loss_clamp_ratio = 4  # Target loss clamping factor

# Network reset (experimental)
single_reset_flag = 0  # Reset counter
reset_every_n_frames_generated = 400_000_00000000  # Reset frequency (disabled)
additional_transition_after_reset = 1_600_000  # Extra transitions after reset
last_layer_reset_factor = 0.8  # Last layer preservation (0=full reset, 1=no reset)
overall_reset_mul_factor = 0.01  # Weight perturbation factor

# Additional features
use_jit = True  # Use PyTorch JIT compilation
