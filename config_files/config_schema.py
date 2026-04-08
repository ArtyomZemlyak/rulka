"""
Pydantic schemas for Rulka configuration.
All config sections with validation and computed fields.
"""

from pathlib import Path
from sys import platform
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config_files.nn_schema import (
    MultimodalTransformersConfig,
    NeuralNetworkConfig,
    TransformersConfig,
)

MultimodalFusionModeLiteral = Literal["none", "vision_transformer", "post_concat", "unified"]


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
    # Multi-action: offsets in ms from current moment, e.g. [0, 10, 20, 30, 40]. Empty or [0] = single action (current behavior).
    rl_action_offsets_ms: list[int] = Field(default_factory=lambda: [0])

    # Computed (filled by validator)
    ms_per_action: int = 0
    ms_per_block: int = 0  # N * 10 when multi-action, else same as ms_per_action
    max_allowable_distance_to_virtual_checkpoint: float = 0.0
    temporal_mini_race_duration_actions: int = 0
    n_actions_per_block: int = 1  # len(rl_action_offsets_ms) when multi-action, else 1

    @field_validator("deck_height", mode="before")
    @classmethod
    def parse_deck_height(cls, v: Any) -> float:
        return _parse_deck_height(v)

    @field_validator("rl_action_offsets_ms")
    @classmethod
    def validate_rl_action_offsets_ms(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError("rl_action_offsets_ms must not be empty")
        expected = [i * 10 for i in range(len(v))]
        if v != expected:
            raise ValueError(
                f"rl_action_offsets_ms must be consecutive 10ms steps starting from 0: "
                f"expected {expected}, got {v}. "
                f"The game engine applies actions every 10ms; non-uniform spacing is not supported."
            )
        return v

    @model_validator(mode="after")
    def compute_derived(self) -> "EnvironmentConfig":
        self.max_allowable_distance_to_virtual_checkpoint = float(
            np.sqrt(
                (self.distance_between_checkpoints / 2) ** 2
                + (self.road_width / 2) ** 2
            )
        )
        # Multi-action: rl_action_offsets_ms e.g. [0, 10, 20, 30, 40] -> N=5
        use_multi_action = (
            len(self.rl_action_offsets_ms) > 1
            or (len(self.rl_action_offsets_ms) == 1 and self.rl_action_offsets_ms[0] != 0)
        )
        if use_multi_action:
            self.n_actions_per_block = len(self.rl_action_offsets_ms)
            self.ms_per_action = self.ms_per_tm_engine_step  # 10 ms per env step
            self.ms_per_block = self.n_actions_per_block * self.ms_per_tm_engine_step
            self.temporal_mini_race_duration_actions = (
                self.temporal_mini_race_duration_ms // self.ms_per_block
            )
        else:
            self.n_actions_per_block = 1
            self.ms_per_action = self.ms_per_tm_engine_step * self.tm_engine_step_per_action
            self.ms_per_block = self.ms_per_action
            self.temporal_mini_race_duration_actions = (
                self.temporal_mini_race_duration_ms // self.ms_per_action
            )
        return self


# --- Training ---
class TrainingConfig(BaseModel):
    run_name: str = "uni_18"
    # RL stack key for get_wiring (train / learner / collector). Network shape comes from nn (flat neural_network) + btr + environment, not from this field.
    # Checkpoints under save/<run_name>/ must match the same algorithm + architecture as the run that wrote them.
    algorithm: Literal["iqn", "ppo", "dpo", "grpo"] = "iqn"
    pretrain_encoder_path: Optional[str] = None
    # Optional: path to BC run dir or to iqn_bc.pt to load full IQN state into checkpoints.
    # All matching parts are loaded: img_head, float_feature_extractor, iqn_fc, A_head, V_head.
    # Applied on fresh run (after encoder injection if set). Requires iqn_bc.pt from BC with use_full_iqn.
    pretrain_bc_heads_path: Optional[str] = None

    # PPO only: BC run directory (contains ppo_policy_bc.pt) or path to .pt from bc_use_rl_architecture.
    # On a fresh save/<run_name>/ (no weights1.torch), writes weights1.torch with key remap (multi-offset bc_heads → policy_head).
    pretrain_ppo_policy_path: Optional[str] = None
    # PPO only: if True, when BC policy_head has more outputs than the RL policy (same in_features), load only the
    # leading rows (logits 0..N-1). Assumes action indices match between BC and RL configs (e.g. RL is a prefix of BC).
    pretrain_ppo_policy_slice_head_to_model: bool = False

    # Optional: path to float_head.pt (BC run dir or file) to load only float_feature_extractor.
    pretrain_float_head_path: Optional[str] = None
    # Optional: path to actions_head.pt (BC run dir or file) to load only A_head.
    pretrain_actions_head_path: Optional[str] = None

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
    # On-policy rollout reward shaping (PPO / DPO / GRPO): discount in potential-based folding when
    # building per-step rewards from env rollouts. Same γ PPO uses for GAE on those rewards.
    # Not IQN ``gamma_schedule`` above (n-step return). Prefer these over legacy ``ppo.gamma`` /
    # ``ppo.ppo_gamma_schedule`` for clarity; if unset, code falls back to ``ppo:``.
    policy_rollout_gamma: Optional[float] = Field(
        default=None,
        description="Scalar γ when no policy_rollout_gamma_schedule. If None, uses ppo.gamma.",
    )
    policy_rollout_gamma_schedule: Optional[list[ScheduleStepFloat]] = Field(
        default=None,
        description="Piecewise-linear γ vs cumul frames for on-policy rollout shaping; overrides ppo.ppo_gamma_schedule when set.",
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
    # When n_actions_per_block > 1: "per_action" = epsilon per each of N actions; "per_block" = one draw for whole block.
    multi_action_exploration: Literal["per_action", "per_block"] = "per_action"


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
    # Pin each game client (TmForever.exe) to specific logical CPUs after launch.
    # collector_process_fn passes process_number as collector_index; see game_instance_manager.launch_game.
    pin_tm_forever_cpu_affinity: bool = False
    tm_forever_cpu_affinity_offset: int = 0
    # If set, collector i uses tm_forever_cpu_affinity_cpus[i] (must cover all collector indices).
    tm_forever_cpu_affinity_cpus: Optional[list[int]] = None


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


# --- PPO (algorithm == "ppo" for loss hyperparams; DPO/GRPO may omit this block if rollout γ is under training) ---
class PPOConfig(BaseModel):
    # Legacy fallback for on-policy rollout shaping + GAE γ when training.policy_rollout_* is unset.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    # None = MSE(V, returns) only; float ε = PPO2-style clip V to V_old ± ε then max of two squared errors.
    clip_coef_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    normalize_advantages: bool = True
    rollout_steps_per_update: int = 2048
    # Optional [[frame, value], ...] schedules; linear interpolation vs cumul_number_frames_played
    # (same step axis as training.lr_schedule). If omitted, the scalar above is used (implicit [[0, scalar]]).
    # Legacy: prefer ``training.policy_rollout_gamma_schedule`` for rollout shaping + PPO GAE γ.
    ppo_gamma_schedule: Optional[list[ScheduleStepFloat]] = None
    gae_lambda_schedule: Optional[list[ScheduleStepFloat]] = None
    ent_coef_schedule: Optional[list[ScheduleStepFloat]] = None
    vf_coef_schedule: Optional[list[ScheduleStepFloat]] = None


# --- DPO (training.algorithm == "dpo"; same actor-critic wiring as PPO) ---
# Field names prefixed where needed so ConfigView flat access does not shadow ``ppo:`` / ``training:``.
class DPOConfig(BaseModel):
    dpo_beta: float = 0.1
    dpo_ref_sync_every_updates: int = 1
    dpo_pair_buffer_max: int = 64
    dpo_data_mode: Literal["online", "offline", "both"] = "online"
    # Repo-relative or absolute path to JSONL; each line: {"chosen": "<joblib path>", "rejected": "<joblib path>"}
    # Each joblib file: tuple (rollout_results_dict, end_race_stats_dict).
    dpo_offline_pairs_jsonl: Optional[str] = None
    dpo_vf_coef: float = 0.1
    dpo_update_epochs: int = 4
    # Reserved for future variable-length pair minibatching (currently unused by learner_dpo).
    dpo_num_minibatches: int = 4
    dpo_max_grad_norm: float = 0.5
    # Optional [[frame, value], ...]; linear interpolation vs cumul_number_frames_played
    # (same axis as training.lr_schedule). If omitted, the scalar above is used (implicit [[0, scalar]]).
    dpo_beta_schedule: Optional[list[ScheduleStepFloat]] = None
    dpo_vf_coef_schedule: Optional[list[ScheduleStepFloat]] = None
    dpo_max_grad_norm_schedule: Optional[list[ScheduleStepFloat]] = None


# --- GRPO (training.algorithm == "grpo"; group-relative policy optimization) ---
class GRPOConfig(BaseModel):
    grpo_group_size: int = 4
    grpo_normalize_group: Literal["mean", "mean_std"] = "mean"
    grpo_ent_coef: float = 0.01
    grpo_max_grad_norm: float = 0.5
    grpo_update_epochs: int = 4
    # Reserved for future within-group minibatching (currently unused by learner_grpo).
    grpo_num_minibatches: int = 4
    grpo_ref_sync_every_updates: int = 50
    grpo_ref_kl_coef: float = 0.0  # 0 = disabled; else KL(ref) regularizer
    # Optional [[frame, value], ...]; linear interpolation vs cumul_number_frames_played
    # (same axis as training.lr_schedule). If omitted, the scalar above is used (implicit [[0, scalar]]).
    grpo_ent_coef_schedule: Optional[list[ScheduleStepFloat]] = None
    grpo_max_grad_norm_schedule: Optional[list[ScheduleStepFloat]] = None
    grpo_ref_kl_coef_schedule: Optional[list[ScheduleStepFloat]] = None


# --- BTR (Beyond The Rainbow) ---
class BTRConfig(BaseModel):
    """Optional IQN enhancements from the BTR paper — not a separate RL algorithm or wiring key.

    Same ``training.algorithm`` (iqn) and same ``IQN_Network`` class; flags only change internals
    (backbone, ``nn.decoder``, loss path). Vision CNN knobs belong in ``nn.vis.cnn``; the CNN fields
    below duplicate that block for backward compatibility and for :func:`config_files.config_loader._merge_btr_cnn_into_vis`
    (fills omitted ``nn.vis.cnn`` keys at load).

    LayerNorm / NoisyNet / ``noisy_sigma0`` are applied to IQN MLP heads using
    :func:`trackmania_rl.nn_build.iqn_btr_from_config.iqn_btr_mlp_head_kw_from_config` on the flat loaded config.
    """

    # Munchausen IQN: soft-policy targets instead of hard max
    use_munchausen: bool = False
    munchausen_alpha: float = 0.9
    munchausen_entropy_tau: float = 0.03
    munchausen_lo: float = -1.0

    # IMPALA-CNN: residual CNN encoder replacing the 4-layer conv
    use_impala_cnn: bool = False
    impala_model_size: int = 2

    # Adaptive MaxPooling: replaces Flatten after conv layers
    use_adaptive_maxpool: bool = False
    adaptive_maxpool_size: int = 6

    # Spectral Normalization on conv layers
    use_spectral_norm: bool = False

    # Layer Normalization on dense layers (V/A heads, float extractor)
    use_layer_norm: bool = False

    # NoisyNets: factorized noisy linear layers in V/A heads
    use_noisy_linear: bool = False
    noisy_sigma0: float = 0.5


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
    btr: BTRConfig = Field(default_factory=BTRConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)

    model_config = {"arbitrary_types_allowed": True}
