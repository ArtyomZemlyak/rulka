"""Algorithm-specific wiring (network / trainer / inferer) selected by config.training.algorithm."""

from trackmania_rl.agents.algorithms.registry import get_wiring

__all__ = ["get_wiring"]
