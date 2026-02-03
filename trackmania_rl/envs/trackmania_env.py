"""
Gymnasium environment for TrackMania RL.

TrackManiaEnv wraps GameEnvBackend and implements the standard gym.Env interface.
- reset(seed, options): load map from options (map_path, zone_centers), return first obs and info.
- step(action): send action to game, block until next decision point, return (obs, reward, terminated, truncated, info).
- observation_space: Tuple(frame Box, state_float Box).
- action_space: Discrete(n_actions).
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from config_files.config_loader import get_config
from trackmania_rl.tmi_interaction.game_env_backend import GameEnvBackend
from trackmania_rl.tmi_interaction.game_instance_manager import GameInstanceManager


class TrackManiaEnv(gym.Env):
    """
    Gymnasium environment for TrackMania RL.
    All game interaction goes through GameEnvBackend; env does not call TMI directly.
    """

    def __init__(
        self,
        game_spawning_lock=None,
        config=None,
        tmi_port: Optional[int] = None,
    ):
        """
        Build env from config (or get_config() if config is None).
        game_spawning_lock: optional lock for multi-process; passed to GameInstanceManager.
        tmi_port: TMI port (default: config.base_tmi_port). Collector passes base_tmi_port + process_number.
        """
        super().__init__()
        cfg = config if config is not None else get_config()
        self._config = cfg
        if game_spawning_lock is None:
            import threading
            game_spawning_lock = threading.Lock()
        port = tmi_port if tmi_port is not None else cfg.base_tmi_port
        self._gim = GameInstanceManager(
            game_spawning_lock=game_spawning_lock,
            running_speed=cfg.running_speed,
            run_steps_per_action=cfg.tm_engine_step_per_action,
            max_overall_duration_ms=cfg.cutoff_rollout_if_race_not_finished_within_duration_ms,
            max_minirace_duration_ms=cfg.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
            tmi_port=port,
        )
        self._backend = GameEnvBackend(self._gim)
        # Observation: (frame (1, H, W) uint8, state_float (float_input_dim,) float32)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(
                low=0,
                high=255,
                shape=(1, cfg.H_downsized, cfg.W_downsized),
                dtype=np.uint8,
            ),
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(cfg.float_input_dim,),
                dtype=np.float32,
            ),
        ))
        self.action_space = gym.spaces.Discrete(len(cfg.inputs))

    @property
    def last_rollout_crashed(self) -> bool:
        """Whether the last episode ended due to TMI timeout/crash (for collector)."""
        return self._gim.last_rollout_crashed

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[npt.NDArray, npt.NDArray], Dict[str, Any]]:
        """
        Start or restart an episode. options may contain:
        - map_path: str
        - zone_centers: np.ndarray
        If omitted, caller must have called with them previously or use a default map.
        """
        super().reset(seed=seed)
        opts = options or {}
        map_path = opts.get("map_path")
        zone_centers = opts.get("zone_centers")
        if map_path is None or zone_centers is None:
            raise ValueError("reset(options=...) must provide 'map_path' and 'zone_centers'")
        obs, info = self._backend.start_episode(map_path, zone_centers)
        # Plan 11.3/11.4: info for collector (current_zone_idx, meters_advanced_along_centerline, state_float, etc.)
        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[Tuple[npt.NDArray, npt.NDArray], float, bool, bool, Dict[str, Any]]:
        """
        Send action to the game and block until the next decision point.
        Returns (obs, reward, terminated, truncated, info).
        """
        self._backend.send_action(action)
        obs, reward, terminated, truncated, info = self._backend.run_until_next_decision_point()
        return obs, reward, terminated, truncated, info
