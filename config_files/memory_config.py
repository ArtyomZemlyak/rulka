"""
Memory and replay buffer configuration.
See docs/source/configuration_guide.rst for detailed documentation.
"""

import numpy as np
from config_files.training_config import global_schedule_speed

# Replay buffer size
memory_size_schedule = [
    (0, (50_000, 20_000)),  # (total_size, start_learning_size)
    (5_000_000 * global_schedule_speed, (100_000, 75_000)),
    (7_000_000 * global_schedule_speed, (200_000, 150_000)),
]

# Prioritized Experience Replay (PER)
prio_alpha = np.float32(0)  # Priority exponent (0=uniform, 1=full prioritization)
prio_epsilon = np.float32(2e-3)  # Priority offset
prio_beta = np.float32(1)  # Importance sampling correction exponent

# Memory usage
number_times_single_memory_is_used_before_discard = 32  # Min samples per transition

# Train/test split
buffer_test_ratio = 0.05  # Fraction of buffer for testing
