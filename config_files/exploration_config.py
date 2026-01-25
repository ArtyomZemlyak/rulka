"""
Exploration strategies configuration.
See docs/source/configuration_guide.rst for detailed documentation.
"""

from config_files.training_config import global_schedule_speed

# Epsilon-greedy
epsilon_schedule = [
    (0, 1),  # Pure random exploration
    (50_000, 1),
    (300_000, 0.1),
    (3_000_000 * global_schedule_speed, 0.03),
]

# Epsilon-Boltzmann
epsilon_boltzmann_schedule = [
    (0, 0.15),
    (3_000_000 * global_schedule_speed, 0.03),
]

tau_epsilon_boltzmann = 0.01  # Boltzmann temperature
