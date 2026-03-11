"""
Action space is defined only by n_steer_parts: multi-label vector of length 2 + 2*n_steer_parts
(accelerate, brake, left_1..left_N, right_1..right_N). Steering is discretized into N parts per direction.
Conversion to/from game input (left, right, accelerate, brake) for set_input_state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class ActionSpace:
    """Forms the action space from a single parameter: n_steer_parts (steering discretization)."""

    __slots__ = ("n_steer_parts", "n_action_dims")

    def __init__(self, n_steer_parts: int):
        self.n_steer_parts = n_steer_parts
        self.n_action_dims = n_action_dims(n_steer_parts)

    def to_game_input(
        self,
        action_vec: npt.NDArray[np.floating] | list[float],
        threshold: float = 0.5,
    ) -> dict[str, bool]:
        """Convert action vector to game input dict for set_input_state."""
        return action_vector_to_game_input(action_vec, self.n_steer_parts, threshold)

    def from_game_input(self, inp: dict[str, Any]) -> npt.NDArray[np.float32]:
        """Convert game input dict to action vector."""
        return game_input_to_action_vector(inp, self.n_steer_parts)

    def random(self, rng: np.random.Generator | None = None) -> npt.NDArray[np.float32]:
        """Random binary action vector."""
        return random_action_vector(self.n_steer_parts, rng)


# Legacy: 12 discrete actions for conversion only (manifest action_idx → dict/vector). Not used by RL.
STANDARD_12_ACTIONS: list[dict[str, bool]] = [
    {"left": False, "right": False, "accelerate": True, "brake": False},
    {"left": True, "right": False, "accelerate": True, "brake": False},
    {"left": False, "right": True, "accelerate": True, "brake": False},
    {"left": False, "right": False, "accelerate": False, "brake": False},
    {"left": True, "right": False, "accelerate": False, "brake": False},
    {"left": False, "right": True, "accelerate": False, "brake": False},
    {"left": False, "right": False, "accelerate": False, "brake": True},
    {"left": True, "right": False, "accelerate": False, "brake": True},
    {"left": False, "right": True, "accelerate": False, "brake": True},
    {"left": False, "right": False, "accelerate": True, "brake": True},
    {"left": True, "right": False, "accelerate": True, "brake": True},
    {"left": False, "right": True, "accelerate": True, "brake": True},
]


def n_action_dims(n_steer_parts: int) -> int:
    """Number of action dimensions: 2 (accel, brake) + 2*n_steer_parts (left parts, right parts)."""
    return 2 + 2 * n_steer_parts


def action_vector_to_game_input(
    action_vec: npt.NDArray[np.floating] | list[float],
    n_steer_parts: int,
    threshold: float = 0.5,
) -> dict[str, bool]:
    """Convert action vector to game input dict for set_input_state.

    action_vec: length 2+2*n_steer_parts [accel, brake, left_1..left_N, right_1..right_N].
    left = any(left part >= threshold), right = any(right part >= threshold).
    """
    arr = np.asarray(action_vec, dtype=np.float64).ravel()
    n = n_action_dims(n_steer_parts)
    if len(arr) < n:
        arr = np.pad(arr, (0, max(0, n - len(arr))))
    arr = arr[:n]
    accel = float(arr[0]) >= threshold
    brake = float(arr[1]) >= threshold
    left = any(float(arr[2 + i]) >= threshold for i in range(n_steer_parts))
    right = any(float(arr[2 + n_steer_parts + i]) >= threshold for i in range(n_steer_parts))
    return {"accelerate": accel, "brake": brake, "left": left, "right": right}


def game_input_to_action_vector(
    inp: dict[str, Any],
    n_steer_parts: int,
) -> npt.NDArray[np.float32]:
    """Convert game input dict to action vector (e.g. from replay: accel, brake, left, right).

    Replay has binary left/right; we set all left parts = left, all right parts = right.
    """
    accel = float(inp.get("accelerate", False))
    brake = float(inp.get("brake", False))
    left = float(inp.get("left", False))
    right = float(inp.get("right", False))
    out = np.zeros(2 + 2 * n_steer_parts, dtype=np.float32)
    out[0] = accel
    out[1] = brake
    for i in range(n_steer_parts):
        out[2 + i] = left
    for i in range(n_steer_parts):
        out[2 + n_steer_parts + i] = right
    return out


def random_action_vector(n_steer_parts: int, rng: np.random.Generator | None = None) -> npt.NDArray[np.float32]:
    """Random binary action vector (each dimension 0 or 1)."""
    if rng is None:
        rng = np.random.default_rng()
    n = n_action_dims(n_steer_parts)
    return (rng.random(n) > 0.5).astype(np.float32)
