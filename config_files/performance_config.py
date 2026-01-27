"""
Performance and system configuration.
See docs/source/configuration_guide.rst for detailed documentation.
"""

# Parallel processing
gpu_collectors_count = 8  # Number of parallel TrackMania instances
max_rollout_queue_size = 1  # Max rollouts waiting to be processed

# Network synchronization
send_shared_network_every_n_batches = 8  # Network update frequency to collectors
update_inference_network_every_n_actions = 8  # Collector local network update frequency

# Visualization and analysis
plot_race_time_left_curves = False  # Generate time-to-finish plots (slow)
n_transitions_to_plot_in_distribution_curves = 1000  # Sample size for distribution plots
make_highest_prio_figures = False  # Generate highest-priority transition plots (slow)

# Data augmentation
apply_randomcrop_augmentation = False  # Random crop augmentation
n_pixels_to_crop_on_each_side = 2  # Crop size (if augmentation enabled)

# Run saving
frames_before_save_best_runs = 1_500_000  # Min frames before saving replays
threshold_to_save_all_runs_ms = -1  # Auto-save threshold (disabled)

# Game speed
running_speed = 1024  # Simulation speed multiplier (160x real-time)

# Window focus management (LEGACY - kept for compatibility)
# NOTE: After analysis, constant focus switching causes "focus war" between multiple instances
# Solution: Focus is set once after map load (game_activated flag reset in request_map)
# This is sufficient - TMInterface uses socket API, not keyboard simulation
force_window_focus_on_input = False  # Deprecated - focus is managed automatically per map load
