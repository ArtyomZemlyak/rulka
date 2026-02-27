"""
Pydantic schemas for Rulka configuration.
All config sections with validation and computed fields.
"""

from pathlib import Path
from sys import platform
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_deck_height(v: Any) -> float:
    if isinstance(v, str) and v.lower() in ("-inf", "-infty"):
        return float("-inf")
    return float(v)


# --- Schedule types (list of [frame, value] or [frame, [a,b]]) ---
ScheduleStepFloat = list[Union[int, float]]
ScheduleStepTuple = list[Union[int, list[int]]]


# --- Environment ---
class EnvironmentConfig(BaseModel):
    tm_engine_step_per_action: int = 5
    ms_per_tm_engine_step: int = 10
    n_zone_centers_in_inputs: int = 40
    one_every_n_zone_centers_in_inputs: int = 20
    n_zone_centers_extrapolate_after_end_of_map: int = 1000
    n_zone_centers_extrapolate_before_start_of_map: int = 20
    distance_between_checkpoints: float = 0.5
    road_width: float = 90
    temporal_mini_race_duration_ms: int = 7000
    margin_to_announce_finish_meters: float = 700
    n_contact_material_physics_behavior_types: int = 4
    n_prev_actions_in_inputs: int = 5
    cutoff_rollout_if_race_not_finished_within_duration_ms: int = 300_000
    cutoff_rollout_if_no_vcp_passed_within_duration_ms: int = 2_000
    timeout_during_run_ms: int = 10_100
    timeout_between_runs_ms: int = 600_000_000
    tmi_protection_timeout_s: int = 500
    game_reboot_interval: int = 3600 * 12
    deck_height: Union[str, float] = "-inf"
    game_camera_number: int = 2
    sync_virtual_and_real_checkpoints: bool = True

    # Computed (filled by validator)
    ms_per_action: int = 0
    max_allowable_distance_to_virtual_checkpoint: float = 0.0
    temporal_mini_race_duration_actions: int = 0

    @field_validator("deck_height", mode="before")
    @classmethod
    def parse_deck_height(cls, v: Any) -> float:
        return _parse_deck_height(v)

    @model_validator(mode="after")
    def compute_derived(self) -> "EnvironmentConfig":
        self.ms_per_action = self.ms_per_tm_engine_step * self.tm_engine_step_per_action
        self.max_allowable_distance_to_virtual_checkpoint = float(
            np.sqrt(
                (self.distance_between_checkpoints / 2) ** 2
                + (self.road_width / 2) ** 2
            )
        )
        self.temporal_mini_race_duration_actions = (
            self.temporal_mini_race_duration_ms // self.ms_per_action
        )
        return self


# --- Neural Network ---
class NeuralNetworkConfig(BaseModel):
    w_downsized: int = 256
    h_downsized: int = 256
    float_hidden_dim: int = 256
    dense_hidden_dimension: int = 1024
    iqn_embedding_dimension: int = 128
    iqn_n: int = 8
    iqn_k: int = 32
    iqn_kappa: float = 5e-3
    use_ddqn: bool = True
    clip_grad_value: float = 1000
    clip_grad_norm: float = 30
    number_memories_trained_on_between_target_network_updates: int = 2048
    soft_update_tau: float = 0.02
    target_self_loss_clamp_ratio: float = 4
    single_reset_flag: int = 0
    reset_every_n_frames_generated: int = 400_000_00000000
    additional_transition_after_reset: int = 1_600_000
    last_layer_reset_factor: float = 0.8
    overall_reset_mul_factor: float = 0.01
    use_jit: bool = True

    # Computed by loader (depends on environment)
    float_input_dim: int = 0


# --- Training ---
class TrainingConfig(BaseModel):
    run_name: str = "uni_18"
    pretrain_encoder_path: Optional[str] = None
    # Optional: path to BC run dir or to iqn_bc.pt to load full IQN state into checkpoints.
    # All matching parts are loaded: img_head, float_feature_extractor, iqn_fc, A_head, V_head.
    # Applied on fresh run (after encoder injection if set). Requires iqn_bc.pt from BC with use_full_iqn.
    pretrain_bc_heads_path: Optional[str] = None

    # Optional: path to float_head.pt (BC run dir or file) to load only float_feature_extractor.
    pretrain_float_head_path: Optional[str] = None
    # Optional: path to actions_head.pt (BC run dir or file) to load only A_head.
    pretrain_actions_head_path: Optional[str] = None

    # Freeze pretrain parts during RL training (default: false). Only parameters of the
    # corresponding module are excluded from the optimizer and from resets/weight decay.
    pretrain_encoder_freeze: bool = False
    pretrain_float_head_freeze: bool = False
    pretrain_iqn_fc_freeze: bool = False
    pretrain_actions_head_freeze: bool = False
    pretrain_V_head_freeze: bool = False

    batch_size: int = 512
    adam_epsilon: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay_lr_ratio: float = 0.1
    global_schedule_speed: int = 1
    lr_schedule: list[ScheduleStepFloat] = Field(
        default_factory=lambda: [
            [0, 1e-3],
            [3_000_000, 5e-5],
            [12_000_000, 5e-5],
            [15_000_000, 1e-5],
        ]
    )
    gamma_schedule: list[ScheduleStepFloat] = Field(
        default_factory=lambda: [[0, 0.999], [1_500_000, 0.999], [2_500_000, 1]]
    )
    n_steps: int = 3
    discard_non_greedy_actions_in_nsteps: bool = True
    tensorboard_suffix_schedule: list[list[Union[int, float, str]]] = Field(
        default_factory=lambda: [
            [0, ""],
            [6_000_000, "_2"],
            [15_000_000, "_3"],
            [30_000_000, "_4"],
            [45_000_000, "_5"],
            [80_000_000, "_6"],
            [150_000_000, "_7"],
        ]
    )
    oversample_long_term_steps: int = 40
    oversample_maximum_term_steps: int = 5

    # Computed by loader (depends on environment)
    min_horizon_to_update_priority_actions: int = 0


# --- Memory ---
class MemoryConfig(BaseModel):
    memory_size_schedule: list[ScheduleStepTuple] = Field(
        default_factory=lambda: [
            [0, [50_000, 20_000]],
            [5_000_000, [100_000, 75_000]],
            [7_000_000, [200_000, 150_000]],
        ]
    )
    prio_alpha: float = 0
    prio_epsilon: float = 2e-3
    prio_beta: float = 1
    number_times_single_memory_is_used_before_discard: int = 32
    buffer_test_ratio: float = 0.05


# --- Exploration ---
class ExplorationConfig(BaseModel):
    epsilon_schedule: list[ScheduleStepFloat] = Field(
        default_factory=lambda: [
            [0, 1],
            [50_000, 1],
            [300_000, 0.1],
            [3_000_000, 0.03],
        ]
    )
    epsilon_boltzmann_schedule: list[ScheduleStepFloat] = Field(
        default_factory=lambda: [[0, 0.15], [3_000_000, 0.03]]
    )
    tau_epsilon_boltzmann: float = 0.01


# --- Rewards ---
class RewardsConfig(BaseModel):
    constant_reward_per_ms: float = -6 / 5000
    reward_per_m_advanced_along_centerline: float = 5 / 500
    shaped_reward_dist_to_cur_vcp: float = -0.1
    shaped_reward_min_dist_to_cur_vcp: float = 2
    shaped_reward_max_dist_to_cur_vcp: float = 25
    engineered_reward_min_dist_to_cur_vcp: float = 5
    engineered_reward_max_dist_to_cur_vcp: float = 25
    shaped_reward_point_to_vcp_ahead: float = 0
    engineered_speedslide_reward_schedule: list[ScheduleStepFloat] = Field(
        default_factory=lambda: [[0, 0]]
    )
    engineered_neoslide_reward_schedule: list[ScheduleStepFloat] = Field(
        default_factory=lambda: [[0, 0]]
    )
    engineered_kamikaze_reward_schedule: list[ScheduleStepFloat] = Field(
        default_factory=lambda: [[0, 0]]
    )
    engineered_close_to_vcp_reward_schedule: list[ScheduleStepFloat] = Field(
        default_factory=lambda: [[0, 0]]
    )
    final_speed_reward_as_if_duration_s: float = 0

    # Computed by validator
    final_speed_reward_per_m_per_s: float = 0

    @model_validator(mode="after")
    def compute_final_speed_reward(self) -> "RewardsConfig":
        self.final_speed_reward_per_m_per_s = (
            self.reward_per_m_advanced_along_centerline
            * self.final_speed_reward_as_if_duration_s
        )
        return self


# --- Map Cycle Entry ---
class MapCycleEntry(BaseModel):
    short_name: str
    map_path: str
    reference_line_path: str
    is_exploration: bool = True
    fill_buffer: bool = True
    repeat: int = 1


# --- Map Cycle ---
class MapCycleConfig(BaseModel):
    entries: list[MapCycleEntry] = Field(default_factory=list)

    # Expanded by loader to list of tuples
    map_cycle: list[tuple[str, str, str, bool, bool]] = Field(default_factory=list)


# --- Performance ---
class PerformanceConfig(BaseModel):
    gpu_collectors_count: int = 4
    max_rollout_queue_size: int = 1
    send_shared_network_every_n_batches: int = 8
    update_inference_network_every_n_actions: int = 8
    plot_race_time_left_curves: bool = False
    n_transitions_to_plot_in_distribution_curves: int = 1000
    make_highest_prio_figures: bool = False
    apply_randomcrop_augmentation: bool = False
    n_pixels_to_crop_on_each_side: int = 2
    frames_before_save_best_runs: int = 1_500_000
    threshold_to_save_all_runs_ms: int = -1
    running_speed: int = 512
    force_window_focus_on_input: bool = False


# --- Input Action ---
class InputAction(BaseModel):
    left: bool = False
    right: bool = False
    accelerate: bool = False
    brake: bool = False


# --- Inputs ---
class InputsConfig(BaseModel):
    actions: list[InputAction] = Field(default_factory=list)
    action_forward_idx: int = 0
    action_backward_idx: int = 6


# --- State Normalization ---
class StateNormalizationConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    waypoint_mean_40cp: list[float] = Field(default_factory=list)
    waypoint_std_40cp: list[float] = Field(default_factory=list)

    # Built by loader
    float_inputs_mean: np.ndarray = Field(default_factory=lambda: np.array([]))
    float_inputs_std: np.ndarray = Field(default_factory=lambda: np.array([]))


# --- User Config (from .env) ---
class UserConfig(BaseSettings):
    """Machine-specific settings loaded from .env. Env vars: USERNAME, TRACKMANIA_BASE_PATH, etc."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    username: str = "Player"
    trackmania_base_path: Path = Field(
        default_factory=lambda: Path.home() / "Documents" / "TrackMania"
    )
    target_python_link_path: Path = Field(
        default_factory=lambda: Path.home() / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"
    )
    base_tmi_port: int = 8478
    linux_launch_game_path: str = "path_to_be_filled_only_if_on_linux"
    windows_TMLoader_path: Path = Field(
        default_factory=lambda: Path.home() / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
    )
    windows_TMLoader_profile_name: str = "default"

    @property
    def is_linux(self) -> bool:
        return platform in ["linux", "linux2"]


# --- Root Config ---
class RulkaConfig(BaseModel):
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    neural_network: NeuralNetworkConfig = Field(default_factory=NeuralNetworkConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    rewards: RewardsConfig = Field(default_factory=RewardsConfig)
    map_cycle: MapCycleConfig = Field(default_factory=MapCycleConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    inputs: InputsConfig = Field(default_factory=InputsConfig)
    state_normalization: StateNormalizationConfig = Field(
        default_factory=StateNormalizationConfig
    )
    user: UserConfig = Field(default_factory=UserConfig)

    model_config = {"arbitrary_types_allowed": True}
