"""
Gymnasium environment for TrackMania RL.

Exports TrackManiaEnv, which wraps GameEnvBackend and implements the standard
gym.Env interface (reset, step, observation_space, action_space).
"""

from trackmania_rl.envs.trackmania_env import TrackManiaEnv

__all__ = ["TrackManiaEnv"]
