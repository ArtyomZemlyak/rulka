"""
Training hyperparameters configuration.
See docs/source/configuration_guide.rst for detailed documentation.
"""

import numpy as np
from config_files.environment_config import temporal_mini_race_duration_actions

# Run identification
run_name = "uni_4"  # Experiment name

# Schedule control
global_schedule_speed = 1  # Multiplier for all schedules

# Optimizer (Adam)
adam_epsilon = 1e-5  # Numerical stability constant
adam_beta1 = 0.9  # First moment decay rate
adam_beta2 = 0.999  # Second moment decay rate
weight_decay_lr_ratio = 0.1  # L2 regularization strength

# Batch size
batch_size = 512  # Number of transitions per training step

# Learning rate schedule
lr_schedule = [
    (0, 1e-3),
    (3_000_000 * global_schedule_speed, 5e-5),
    (12_000_000 * global_schedule_speed, 5e-5),
    (15_000_000 * global_schedule_speed, 1e-5),
]

# Gamma schedule
gamma_schedule = [
    (0, 0.999),
    (1_500_000, 0.999),
    (2_500_000, 1),
]

# N-step learning
n_steps = 3  # Number of steps for n-step returns
discard_non_greedy_actions_in_nsteps = True  # Exclude exploratory actions from n-step returns

# Tensorboard
tensorboard_suffix_schedule = [
    (0, ""),
    (6_000_000 * global_schedule_speed, "_2"),
    (15_000_000 * global_schedule_speed, "_3"),
    (30_000_000 * global_schedule_speed, "_4"),
    (45_000_000 * global_schedule_speed, "_5"),
    (80_000_000 * global_schedule_speed, "_6"),
    (150_000_000 * global_schedule_speed, "_7"),
]

# Temporal training parameters
oversample_long_term_steps = 40  # Oversample far future transitions
oversample_maximum_term_steps = 5  # Oversample episode end transitions
min_horizon_to_update_priority_actions = temporal_mini_race_duration_actions - 40  # Priority update horizon
