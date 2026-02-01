"""
Load configuration from YAML and .env.
Provides flat attribute access for backward compatibility with config_copy.xxx usage.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config_files.config_schema import (
    EnvironmentConfig,
    ExplorationConfig,
    InputAction,
    InputsConfig,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Build float_inputs_mean and float_inputs_std arrays."""
    n = env.n_zone_centers_in_inputs
    n_contact = env.n_contact_material_physics_behavior_types
    n_prev = env.n_prev_actions_in_inputs

    mini_race_half = env.temporal_mini_race_duration_actions // 2
    margin_mean = env.margin_to_announce_finish_meters
    margin_std = margin_mean / 2.0

    n_prev_actions = n_prev * 4
    n_gear = 4 + 4 + 4 + 1 + 1 + 1 + 1 + 4 * n_contact
    n_waypoints = n * 3

    prev_actions_mean = [0.8, 0.2, 0.3, 0.3] * n_prev
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

    ang_vel_mean = [0.0, 0.0, 0.0]
    ang_vel_std = [0.5, 1.0, 0.5]
    vel_mean = [0.0, 0.0, 55.0]
    vel_std = [5.0, 5.0, 20.0]
    y_map_mean = [0.0, 1.0, 0.0]
    y_map_std = [0.5, 0.5, 0.5]

    w_mean = state_norm.waypoint_mean_40cp
    w_std = state_norm.waypoint_std_40cp

    mean_arr = np.array(
        [
            float(mini_race_half),
            *prev_actions_mean,
            *gear_mean,
            *ang_vel_mean,
            *vel_mean,
            *y_map_mean,
            *_waypoint_mean(w_mean, n),
            margin_mean,
            0.0,
        ],
        dtype=np.float64,
    )
    std_arr = np.array(
        [
            float(mini_race_half),
            *prev_actions_std,
            *gear_std,
            *ang_vel_std,
            *vel_std,
            *y_map_std,
            *_waypoint_std(w_std, n),
            margin_std,
            1.0,
        ],
        dtype=np.float64,
    )
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
        e, n, t, mem, exp, r, mc, p, inp, sn, u = (
            m.environment,
            m.neural_network,
            m.training,
            m.memory,
            m.exploration,
            m.rewards,
            m.map_cycle,
            m.performance,
            m.inputs,
            m.state_normalization,
            m.user,
        )
        # Aliases for backward compatibility
        if name == "W_downsized":
            return n.w_downsized
        if name == "H_downsized":
            return n.h_downsized
        # inputs: list of dicts for set_input_state(**config.inputs[i])
        if name == "inputs":
            return [a.model_dump() for a in inp.actions]
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
        # Environment
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
        # Inputs (action_forward_idx, action_backward_idx)
        if hasattr(inp, name):
            return getattr(inp, name)
        # State normalization
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
    inputs_data = data.get("inputs", {})
    actions = [InputAction.model_validate(a) for a in inputs_data.get("actions", [])]
    inputs = InputsConfig(
        actions=actions,
        action_forward_idx=inputs_data.get("action_forward_idx", 0),
        action_backward_idx=inputs_data.get("action_backward_idx", 6),
    )
    state_norm_data = data.get("state_normalization", {})
    state_norm = StateNormalizationConfig(
        waypoint_mean_40cp=state_norm_data.get("waypoint_mean_40cp", []),
        waypoint_std_40cp=state_norm_data.get("waypoint_std_40cp", []),
    )

    # Compute cross-section values
    neural.float_input_dim = (
        27
        + 3 * env.n_zone_centers_in_inputs
        + 4 * env.n_prev_actions_in_inputs
        + 4 * env.n_contact_material_physics_behavior_types
        + 1
    )
    training.min_horizon_to_update_priority_actions = (
        env.temporal_mini_race_duration_actions - 40
    )

    mean_arr, std_arr = _build_float_inputs_mean_std(env, state_norm)
    state_norm.float_inputs_mean = mean_arr
    state_norm.float_inputs_std = std_arr

    # Sanity check: state length must match float_input_dim
    n_z = env.n_zone_centers_in_inputs
    n_p = env.n_prev_actions_in_inputs
    n_c = env.n_contact_material_physics_behavior_types
    expected_len = (
        1 + n_p * 4 + (4 + 4 + 4 + 1 + 1 + 1 + 1 + 4 * n_c)
        + 3 + 3 + 3 + n_z * 3 + 1 + 1
    )
    assert len(mean_arr) == expected_len == neural.float_input_dim, (
        f"float_inputs_mean length {len(mean_arr)} != expected {expected_len} != float_input_dim {neural.float_input_dim}"
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
        inputs=inputs,
        state_normalization=state_norm,
        user=user,
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
