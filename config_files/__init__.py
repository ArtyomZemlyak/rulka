"""
Configuration files package for TrackMania RL.

This package contains all configuration settings organized into logical modules:
- environment_config: Environment and simulation settings
- neural_network_config: Network architecture parameters
- training_config: Training hyperparameters and schedules
- memory_config: Replay buffer configuration
- exploration_config: Exploration strategies
- rewards_config: Reward shaping parameters
- map_cycle_config: Map training cycles
- performance_config: System performance settings
- inputs_list: Action space definition
- state_normalization: State normalization parameters
- user_config: User-specific paths and settings

The main config.py file imports and re-exports all settings for backward compatibility.
"""

# Re-export all configuration for easy access
from config_files.config import *

__all__ = [
    'config',
    'config_copy',
    'environment_config',
    'neural_network_config',
    'training_config',
    'memory_config',
    'exploration_config',
    'rewards_config',
    'map_cycle_config',
    'performance_config',
    'inputs_list',
    'state_normalization',
    'user_config',
]
