"""
Load configuration from YAML and .env.
Provides flat attribute access for backward compatibility with config_copy.xxx usage.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config_files.config_schema import (
    ACTION_INPUT_NAMES,
    ActionInputSpec,
    ActionSpaceConfig,
    EnvironmentConfig,
    ExplorationConfig,
    MapCycleConfig,
    MapCycleEntry,
    MemoryConfig,
    NeuralNetworkConfig,
    PerformanceConfig,
    RewardsConfig,
    StateNormalizationConfig,
    TrainingConfig,
    UserConfig,
)
from trackmania_rl.action_vector import ActionSpace
from trackmania_rl.float_inputs import (
    ALL_STATE_OBSERVATION_NAMES,
    compute_float_input_dim,
)


def _apply_schedule_speed(schedule: list, speed: int) -> list:
    """Multiply frame counts in schedule by global_schedule_speed."""
    if speed == 1:
        return schedule
    result = []
    for step in schedule:
        if isinstance(step[1], list):
            result.append([step[0] * speed, step[1]])
        else:
            result.append([step[0] * speed, step[1]])
    return result


def _waypoint_mean(data: list[float], n: int) -> list[float]:
    """Waypoint mean block of length n*3."""
    n_waypoints = n * 3
    if n == 40:
        return list(data)[:n_waypoints]
    if n < 40:
        return data[: n * 3]
    out = list(data)
    last_fwd = data[-3] if data else 0.0
    for _ in range(n - 40):
        out.extend([last_fwd, 0.0, 0.0])
    return out


def _waypoint_std(data: list[float], n: int) -> list[float]:
    """Waypoint std block of length n*3."""
    if n == 40:
        return list(data)[: n * 3]
    if n < 40:
        return data[: n * 3]
    out = list(data)
    for _ in range(n - 40):
        out.extend([50.0, 50.0, 50.0])
    return out


def _build_float_inputs_mean_std(
    env: EnvironmentConfig,
    state_norm: StateNormalizationConfig,
    include: dict[str, bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Build float_inputs_mean and float_inputs_std arrays for included segments only (same order as build_float_vector)."""
    from trackmania_rl.float_inputs import (
        CAR_TRACK_EXTRA_SEGMENT_DIMS,
        CAR_TRACK_EXTRA_SEGMENT_NAMES,
        STATE_OBSERVATION_MAIN_NAMES,
        _get_segment_dim,
    )

    n = env.n_zone_centers_in_inputs
    n_contact = env.n_contact_material_physics_behavior_types
    n_prev = env.n_prev_actions_in_inputs
    n_prev_actions = _get_segment_dim("prev_actions", env)
    n_action_dims = n_prev_actions // n_prev if n_prev else 4
    n_gear = _get_segment_dim("gear_and_wheels", env)
    n_waypoints = n * 3
    mini_race_half = env.temporal_mini_race_duration_actions // 2
    margin_mean = env.margin_to_announce_finish_meters
    margin_std = margin_mean / 2.0

    # One block per action: [accel, brake] + [0.3]*(n_action_dims-2) so length matches any action_space
    prev_actions_mean = ([0.8, 0.2] + [0.3] * max(0, n_action_dims - 2)) * n_prev
    prev_actions_std = [0.5] * n_prev_actions
    gear_mean = (
        [0.1, 0.1, 0.1, 0.1]
        + [0.9, 0.9, 0.9, 0.9]
        + [0.02, 0.02, 0.02, 0.02]
        + [0.3, 2.5, 7000.0, 0.1]
        + [0.5] * (4 * n_contact)
    )
    gear_std = (
        [0.5] * 12
        + [1, 2, 3000.0, 10]
        + [0.5] * (4 * n_contact)
    )
    assert len(gear_mean) == n_gear == len(gear_std), "gear mean/std length must match get_segment_dim(gear_and_wheels)"
    assert len(prev_actions_mean) == n_prev_actions == len(prev_actions_std), "prev_actions mean/std length must match get_segment_dim(prev_actions)"
    w_mean = state_norm.waypoint_mean_40cp
    w_std = state_norm.waypoint_std_40cp

    car_track_extra_mean = (
        [0.0] * 3 + [0.0] * 3 + [0.0] * 3
        + [0.5, 0.2, 0.0]
        + [55.0, 0.0]
        + [100.0]
        + [0.0, 0.0, 55.0]
        + [1.0, 0.0]
        + [0.0, 0.0]
        + [1.0, 0.5, 5000.0]
        + [1000.0, 50.0, 100.0, 20.0]
    )
    car_track_extra_std = (
        [5.0] * 3 + [5.0] * 3 + [100.0] * 3
        + [0.5, 0.5, 1.0]
        + [20.0, 10.0]
        + [30.0]
        + [10.0, 10.0, 20.0]
        + [0.5, 0.5]
        + [0.5, 0.5]
        + [0.3, 0.3, 3000.0]
        + [2000.0, 50.0, 50.0, 15.0]
    )

    mean_parts: list[list[float]] = []
    std_parts: list[list[float]] = []

    def add(name: str, m: list[float], s: list[float]) -> None:
        if include.get(name, True):
            mean_parts.append(m)
            std_parts.append(s)

    add(STATE_OBSERVATION_MAIN_NAMES[0], [float(mini_race_half)], [float(mini_race_half)])
    add(STATE_OBSERVATION_MAIN_NAMES[1], prev_actions_mean, prev_actions_std)
    add(STATE_OBSERVATION_MAIN_NAMES[2], gear_mean, gear_std)
    add(STATE_OBSERVATION_MAIN_NAMES[3], [0.0, 0.0, 0.0], [0.5, 1.0, 0.5])
    add(STATE_OBSERVATION_MAIN_NAMES[4], [0.0, 0.0, 55.0], [5.0, 5.0, 20.0])
    add(STATE_OBSERVATION_MAIN_NAMES[5], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5])
    add(STATE_OBSERVATION_MAIN_NAMES[6], _waypoint_mean(w_mean, n), _waypoint_std(w_std, n))
    add(STATE_OBSERVATION_MAIN_NAMES[7], [margin_mean], [margin_std])
    add(STATE_OBSERVATION_MAIN_NAMES[8], [0.0], [1.0])
    add(STATE_OBSERVATION_MAIN_NAMES[9], [0.0], [2.0])
    add(STATE_OBSERVATION_MAIN_NAMES[10], [0.0], [0.5])

    idx = 0
    for seg_name, dim in zip(CAR_TRACK_EXTRA_SEGMENT_NAMES, CAR_TRACK_EXTRA_SEGMENT_DIMS):
        if include.get(seg_name, True):
            mean_parts.append(car_track_extra_mean[idx : idx + dim])
            std_parts.append(car_track_extra_std[idx : idx + dim])
        idx += dim

    mean_arr = np.array([x for part in mean_parts for x in part], dtype=np.float64)
    std_arr = np.array([x for part in std_parts for x in part], dtype=np.float64)
    return mean_arr, std_arr


def _expand_map_cycle(entries: list[MapCycleEntry]) -> list[tuple[str, str, str, bool, bool]]:
    """Expand map cycle entries by repeat count."""
    result = []
    for e in entries:
        t = (e.short_name, e.map_path, e.reference_line_path, e.is_exploration, e.fill_buffer)
        for _ in range(e.repeat):
            result.append(t)
    return result


class ConfigView:
    """
    Flat view over nested RulkaConfig for backward compatibility.
    Provides config_copy.xxx style attribute access.
    """

    def __init__(self, cfg: "RulkaConfig"):
        self._cfg = cfg

    def __getattr__(self, name: str) -> Any:
        # Map flat names to nested config
        m = self._cfg
        e, n, t, mem, exp, r, mc, p, sn, u = (
            m.environment,
            m.neural_network,
            m.training,
            m.memory,
            m.exploration,
            m.rewards,
            m.map_cycle,
            m.performance,
            m.state_normalization,
            m.user,
        )
        # Aliases for backward compatibility
        if name == "W_downsized":
            return n.w_downsized
        if name == "H_downsized":
            return n.h_downsized
        if name == "n_actions":
            return n.n_action_dims
        # Action space: from config.action_space; legacy conversion uses STANDARD_12_ACTIONS.
        if name == "action_space":
            return getattr(m, "action_space", None)
        if name == "inputs":
            from trackmania_rl.action_vector import STANDARD_12_ACTIONS
            return STANDARD_12_ACTIONS
        if name == "action_forward_idx":
            return 0
        if name == "action_backward_idx":
            return 6
        # map_cycle: expanded list of tuples
        if name == "map_cycle":
            return mc.map_cycle
        # is_linux from user
        if name == "is_linux":
            return u.is_linux
        # User config
        if name in ("username", "trackmania_base_path", "target_python_link_path", "base_tmi_port",
                    "linux_launch_game_path", "windows_TMLoader_path", "windows_TMLoader_profile_name"):
            return getattr(u, name)
        # Environment (n_steer_parts: use action space effective value when action_space is set)
        if name == "n_steer_parts" and getattr(m, "action_space", None) is not None:
            return ActionSpace.from_config(m).n_steer_parts
        if hasattr(e, name):
            return getattr(e, name)
        # Neural network
        if hasattr(n, name):
            return getattr(n, name)
        # Training
        if hasattr(t, name):
            return getattr(t, name)
        # Memory
        if hasattr(mem, name):
            return getattr(mem, name)
        # Exploration
        if hasattr(exp, name):
            return getattr(exp, name)
        # Rewards
        if hasattr(r, name):
            return getattr(r, name)
        # Performance
        if hasattr(p, name):
            return getattr(p, name)
        # State normalization (and flattened state_observation_include for float_inputs)
        if name == "state_observation_include":
            return sn.state_observation_include
        if hasattr(sn, name):
            return getattr(sn, name)
        raise AttributeError(f"Config has no attribute '{name}'")


def load_config(config_path: Path | str) -> ConfigView:
    """
    Load configuration from YAML file.
    User settings from .env are merged via UserConfig.
    All computed fields are built at load time.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    speed = data.get("training", {}).get("global_schedule_speed", 1)

    # Apply global_schedule_speed to schedules
    for key in ("lr_schedule", "tensorboard_suffix_schedule"):
        if key in data.get("training", {}):
            data["training"][key] = _apply_schedule_speed(
                data["training"][key], speed
            )
    for key in ("epsilon_schedule", "epsilon_boltzmann_schedule"):
        if key in data.get("exploration", {}):
            data["exploration"][key] = _apply_schedule_speed(
                data["exploration"][key], speed
            )
    for key in (
        "engineered_speedslide_reward_schedule",
        "engineered_neoslide_reward_schedule",
        "engineered_kamikaze_reward_schedule",
        "engineered_close_to_vcp_reward_schedule",
    ):
        if key in data.get("rewards", {}):
            data["rewards"][key] = _apply_schedule_speed(
                data["rewards"][key], speed
            )
    if "memory_size_schedule" in data.get("memory", {}):
        data["memory"]["memory_size_schedule"] = _apply_schedule_speed(
            data["memory"]["memory_size_schedule"], speed
        )

    env = EnvironmentConfig.model_validate(data.get("environment", {}))
    neural = NeuralNetworkConfig.model_validate(data.get("neural_network", {}))
    training = TrainingConfig.model_validate(data.get("training", {}))
    memory = MemoryConfig.model_validate(data.get("memory", {}))
    exploration = ExplorationConfig.model_validate(data.get("exploration", {}))
    rewards = RewardsConfig.model_validate(data.get("rewards", {}))
    map_cycle_data = data.get("map_cycle", {})
    map_entries = [MapCycleEntry.model_validate(e) for e in map_cycle_data.get("entries", [])]
    map_cycle = MapCycleConfig(entries=map_entries, map_cycle=_expand_map_cycle(map_entries))
    performance = PerformanceConfig.model_validate(data.get("performance", {}))
    state_norm_data = data.get("state_normalization", {})
    include_raw = data.get("state_observation", {}).get("include", {})
    state_observation_include = {
        name: include_raw.get(name, True) for name in ALL_STATE_OBSERVATION_NAMES
    }
    state_norm = StateNormalizationConfig(
        waypoint_mean_40cp=state_norm_data.get("waypoint_mean_40cp", []),
        waypoint_std_40cp=state_norm_data.get("waypoint_std_40cp", []),
        state_observation_include=state_observation_include,
    )

    # Parse action_space (optional). Merge defaults for missing inputs.
    # When action_space.inputs is absent, left/right use environment.n_steer_parts for backward compatibility.
    action_space_data = data.get("action_space", {}) or {}
    inputs_data = action_space_data.get("inputs", {}) or {}
    n_steer_fallback = getattr(env, "n_steer_parts", 1)
    default_inputs = {name: ActionInputSpec(enabled=True, discretization=1) for name in ACTION_INPUT_NAMES}
    if inputs_data:
        for name in ACTION_INPUT_NAMES:
            if name in inputs_data:
                entry = inputs_data[name]
                if isinstance(entry, dict):
                    default_inputs[name] = ActionInputSpec(
                        enabled=entry.get("enabled", True),
                        discretization=int(entry.get("discretization", 1)),
                    )
                else:
                    default_inputs[name] = entry
    else:
        # No action_space.inputs in YAML: use n_steer_parts for left/right (legacy behavior).
        default_inputs["left"] = ActionInputSpec(enabled=True, discretization=n_steer_fallback)
        default_inputs["right"] = ActionInputSpec(enabled=True, discretization=n_steer_fallback)
    action_space_config = ActionSpaceConfig(inputs=default_inputs)

    # Config-like object with env + action_space so ActionSpace.from_config and float_inputs use it
    class _EnvWithActionSpace:
        pass

    env_with_action = _EnvWithActionSpace()
    for key in dir(env):
        if not key.startswith("_"):
            setattr(env_with_action, key, getattr(env, key))
    env_with_action.action_space = action_space_config

    action_space_obj = ActionSpace.from_config(env_with_action)
    neural.n_action_dims = action_space_obj.n_action_dims
    n_per_step = neural.n_action_dims
    neural.float_input_dim = compute_float_input_dim(env_with_action, state_observation_include)
    training.min_horizon_to_update_priority_actions = (
        env.temporal_mini_race_duration_actions - 40
    )

    mean_arr, std_arr = _build_float_inputs_mean_std(env_with_action, state_norm, state_observation_include)
    state_norm.float_inputs_mean = mean_arr
    state_norm.float_inputs_std = std_arr

    assert len(mean_arr) == neural.float_input_dim, (
        f"float_inputs_mean length {len(mean_arr)} != float_input_dim {neural.float_input_dim}"
    )

    user = UserConfig()

    from config_files.config_schema import RulkaConfig

    cfg = RulkaConfig(
        environment=env,
        neural_network=neural,
        training=training,
        memory=memory,
        exploration=exploration,
        rewards=rewards,
        map_cycle=map_cycle,
        performance=performance,
        state_normalization=state_norm,
        user=user,
        action_space=action_space_config,
    )
    return ConfigView(cfg)


# ---------------------------------------------------------------------------
# Module-level cache: set once per process, never reloaded in hot path
# ---------------------------------------------------------------------------
_config: ConfigView | None = None


def get_config() -> ConfigView:
    """Return the cached config. Must call set_config() first (at process startup)."""
    if _config is None:
        raise RuntimeError(
            "Config not initialized. Call set_config(load_config(path)) at process startup."
        )
    return _config


def set_config(cfg: ConfigView) -> None:
    """Set the cached config. Call once per process after load_config()."""
    global _config
    _config = cfg
