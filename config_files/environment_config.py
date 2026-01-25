"""
Environment and simulation configuration.
See docs/source/configuration_guide.rst for detailed documentation.
"""

import numpy as np

# Image processing
W_downsized = 160  # Width of captured game frames (pixels)
H_downsized = 120  # Height of captured game frames (pixels)

# Timing configuration
tm_engine_step_per_action = 5  # Number of simulation steps per action
ms_per_tm_engine_step = 10  # Milliseconds per simulation step (fixed by game)
ms_per_action = ms_per_tm_engine_step * tm_engine_step_per_action  # Total ms per action

# Spatial configuration - track representation
n_zone_centers_in_inputs = 40  # Number of waypoints used as input
one_every_n_zone_centers_in_inputs = 20  # Sample every N-th waypoint
n_zone_centers_extrapolate_after_end_of_map = 1000  # Virtual waypoints after finish
n_zone_centers_extrapolate_before_start_of_map = 20  # Virtual waypoints before start
distance_between_checkpoints = 0.5  # Spacing between checkpoints (meters)
road_width = 90  # Maximum lateral distance from centerline (meters)
max_allowable_distance_to_virtual_checkpoint = np.sqrt(
    (distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2
)

# Temporal configuration - mini-races
temporal_mini_race_duration_ms = 7000  # Duration of mini-races for reward shaping
temporal_mini_race_duration_actions = temporal_mini_race_duration_ms // ms_per_action
margin_to_announce_finish_meters = 700  # Distance to notify agent of finish line

# Contact and physics
n_contact_material_physics_behavior_types = 4  # Number of surface types (see contact_materials.py)
n_prev_actions_in_inputs = 5  # Number of previous actions in state

# Timeout configuration
cutoff_rollout_if_race_not_finished_within_duration_ms = 300_000  # Max race duration
cutoff_rollout_if_no_vcp_passed_within_duration_ms = 2_000  # Timeout if no checkpoint passed
timeout_during_run_ms = 10_100  # TMInterface timeout during active racing
timeout_between_runs_ms = 600_000_000  # Timeout during race setup
tmi_protection_timeout_s = 500  # Max time to wait for TMInterface response
game_reboot_interval = 3600 * 12  # Auto-restart game after N seconds

# Game settings
deck_height = -np.inf  # Minimum Y-coordinate for valid positions
game_camera_number = 2  # Camera view (1=behind, 2=first person, 3=top)
sync_virtual_and_real_checkpoints = True  # Align virtual CPs with game CPs
