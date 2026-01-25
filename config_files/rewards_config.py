"""
Reward shaping configuration.
See docs/source/configuration_guide.rst for detailed documentation.
"""

from config_files.environment_config import margin_to_announce_finish_meters
from config_files.training_config import global_schedule_speed

# Base rewards
constant_reward_per_ms = -6 / 5000  # Time penalty
reward_per_m_advanced_along_centerline = 5 / 500  # Progress reward

# Shaped rewards - distance to checkpoint
shaped_reward_dist_to_cur_vcp = -0.1  # Distance penalty
shaped_reward_min_dist_to_cur_vcp = 2  # Min distance threshold (meters)
shaped_reward_max_dist_to_cur_vcp = 25  # Max distance threshold (meters)
engineered_reward_min_dist_to_cur_vcp = 5
engineered_reward_max_dist_to_cur_vcp = 25

# Shaped rewards - orientation
shaped_reward_point_to_vcp_ahead = 0  # Orientation reward (disabled)

# Engineered rewards - advanced techniques
engineered_speedslide_reward_schedule = [(0, 0)]  # Speedslide bonus (disabled)
engineered_neoslide_reward_schedule = [(0, 0)]  # Neoslide bonus (disabled)
engineered_kamikaze_reward_schedule = [(0, 0)]  # Kamikaze bonus (disabled)
engineered_close_to_vcp_reward_schedule = [(0, 0)]  # Close CP bonus (disabled)

# Terminal rewards
final_speed_reward_as_if_duration_s = 0  # Finish speed multiplier (disabled)
final_speed_reward_per_m_per_s = reward_per_m_advanced_along_centerline * final_speed_reward_as_if_duration_s
