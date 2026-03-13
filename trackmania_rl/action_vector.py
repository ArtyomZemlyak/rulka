"""
Action space: parameterized by config (action_space.inputs or n_steer_parts fallback).
Multi-label vector: concatenation of enabled inputs, each with `discretization` dimensions.
Conversion to/from game input (left, right, accelerate, brake) for set_input_state.
Single source of truth for n_action_dims, to_game_input, from_game_input, and (for classification) index <-> vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# Order of inputs in the action vector. Must match game input keys.
ACTION_INPUT_ORDER: tuple[str, ...] = ("accelerate", "brake", "left", "right")


def _build_spec_from_config(config: Any) -> list[tuple[str, bool, int]]:
    """Build list of (name, enabled, n_bins) in ACTION_INPUT_ORDER. Uses action_space from config if present, else n_steer_parts for left/right."""
    action_space = getattr(config, "action_space", None)
    n_steer_parts = getattr(config, "n_steer_parts", None)
    if n_steer_parts is None:
        env = getattr(config, "environment", None)
        n_steer_parts = getattr(env, "n_steer_parts", 1) if env is not None else 1
    if action_space is not None and getattr(action_space, "inputs", None):
        inputs = action_space.inputs
        spec = []
        for name in ACTION_INPUT_ORDER:
            entry = inputs.get(name) if isinstance(inputs, dict) else getattr(inputs, name, None)
            if entry is None:
                enabled = True
                n_bins = n_steer_parts if name in ("left", "right") else 1
            else:
                enabled = getattr(entry, "enabled", True)
                n_bins = getattr(entry, "discretization", 1)
            spec.append((name, enabled, n_bins))
        return spec
    # Fallback: all 4 enabled, left/right use n_steer_parts
    return [
        ("accelerate", True, 1),
        ("brake", True, 1),
        ("left", True, n_steer_parts),
        ("right", True, n_steer_parts),
    ]


def _spec_to_layout(
    spec: list[tuple[str, bool, int]],
) -> tuple[int, list[tuple[str, int, int, int]], dict[str, tuple[int, int]]]:
    """From spec (name, enabled, n_bins), return (n_action_dims, [(name, start, end, n_bins), ...], name_to_slice)."""
    n_action_dims = 0
    layout: list[tuple[str, int, int, int]] = []
    name_to_slice: dict[str, tuple[int, int]] = {}
    for name, enabled, n_bins in spec:
        if enabled:
            start = n_action_dims
            end = n_action_dims + n_bins
            n_action_dims = end
            layout.append((name, start, end, n_bins))
            name_to_slice[name] = (start, end)
    return n_action_dims, layout, name_to_slice


class ActionSpace:
    """Action space from config: enabled inputs and discretization per input. Single source of truth for dims and conversion."""

    __slots__ = (
        "n_action_dims",
        "_layout",
        "_name_to_slice",
        "_spec",
        "n_steer_parts",
    )

    def __init__(self, spec: list[tuple[str, bool, int]] | None = None, n_steer_parts: int | None = None):
        if spec is None:
            if n_steer_parts is None:
                n_steer_parts = 1
            spec = [
                ("accelerate", True, 1),
                ("brake", True, 1),
                ("left", True, n_steer_parts),
                ("right", True, n_steer_parts),
            ]
        self._spec = spec
        nd, layout, name_to_slice = _spec_to_layout(spec)
        self.n_action_dims = nd
        self._layout = layout
        self._name_to_slice = name_to_slice
        # For backward compat: effective n_steer_parts (for left/right; 1 if not present)
        n_steer = 1
        for name, _, n_bins in spec:
            if name in ("left", "right") and n_bins > 0:
                n_steer = n_bins
                break
        self.n_steer_parts = n_steer

    @classmethod
    def from_config(cls, config: Any) -> ActionSpace:
        """Build ActionSpace from config (uses action_space if present, else n_steer_parts)."""
        spec = _build_spec_from_config(config)
        return cls(spec=spec)

    def to_game_input(
        self,
        action_vec: npt.NDArray[np.floating] | list[float],
        threshold: float = 0.5,
    ) -> dict[str, bool]:
        """Convert action vector to game input dict for set_input_state."""
        return action_vector_to_game_input_from_layout(action_vec, self._layout, self.n_action_dims, threshold)

    def from_game_input(self, inp: dict[str, Any]) -> npt.NDArray[np.float32]:
        """Convert game input dict to action vector."""
        return game_input_to_action_vector_from_layout(inp, self._layout, self.n_action_dims)

    def random(self, rng: np.random.Generator | None = None) -> npt.NDArray[np.float32]:
        """Random binary action vector."""
        if rng is None:
            rng = np.random.default_rng()
        return (rng.random(self.n_action_dims) > 0.5).astype(np.float32)

    def get_input_slice(self, name: str) -> tuple[int, int] | None:
        """Return (start, end) indices for an input in the action vector, or None if disabled."""
        return self._name_to_slice.get(name)

    @property
    def branch_dims(self) -> list[int]:
        """Number of dimensions (bins) per branch for BDQ-style multilabel. One branch per enabled input."""
        return [n_bins for (_, _, _, n_bins) in self._layout]

    # --- Classification mode: discrete set of joint actions (e.g. 12 canonical), index <-> vector ---
    @property
    def n_discrete_actions(self) -> int:
        """Number of discrete joint actions for classification head (e.g. 12)."""
        return len(STANDARD_12_ACTIONS)

    def discrete_action_vectors(self) -> list[npt.NDArray[np.float32]]:
        """List of action vectors for each discrete class (e.g. 12). Order matches STANDARD_12_ACTIONS."""
        return [self.from_game_input(d) for d in STANDARD_12_ACTIONS]

    def action_index_to_vector(self, k: int) -> npt.NDArray[np.float32]:
        """Map discrete action index (0 .. n_discrete_actions-1) to action vector."""
        if k < 0 or k >= len(STANDARD_12_ACTIONS):
            k = 0
        return self.from_game_input(STANDARD_12_ACTIONS[k])

    def action_vector_to_index(self, vec: npt.NDArray[np.floating] | list[float]) -> int:
        """Map action vector to discrete action index (exact match to one of discrete_action_vectors)."""
        arr = np.asarray(vec, dtype=np.float32).ravel()[: self.n_action_dims]
        vectors = self.discrete_action_vectors()
        for i, v in enumerate(vectors):
            if v.shape[0] == arr.shape[0] and np.allclose(v, arr):
                return i
        # Nearest by L1 or first match
        best = 0
        best_err = np.abs(arr - vectors[0]).sum()
        for i in range(1, len(vectors)):
            err = np.abs(arr - vectors[i]).sum()
            if err < best_err:
                best_err = err
                best = i
        return best


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
    """Number of action dimensions for legacy layout: 2 (accel, brake) + 2*n_steer_parts (left parts, right parts)."""
    return 2 + 2 * n_steer_parts


def action_vector_to_game_input_from_layout(
    action_vec: npt.NDArray[np.floating] | list[float],
    layout: list[tuple[str, int, int, int]],
    total_dim: int,
    threshold: float = 0.5,
) -> dict[str, bool]:
    """Convert action vector to game input using layout [(name, start, end, n_bins), ...]."""
    arr = np.asarray(action_vec, dtype=np.float64).ravel()
    if len(arr) < total_dim:
        arr = np.pad(arr, (0, max(0, total_dim - len(arr))))
    arr = arr[:total_dim]
    out: dict[str, bool] = {}
    for name, start, end, _ in layout:
        chunk = arr[start:end]
        if len(chunk) == 1:
            out[name] = float(chunk[0]) >= threshold
        else:
            out[name] = any(float(x) >= threshold for x in chunk)
    for name in ACTION_INPUT_ORDER:
        if name not in out:
            out[name] = False
    return out


def game_input_to_action_vector_from_layout(
    inp: dict[str, Any],
    layout: list[tuple[str, int, int, int]],
    total_dim: int,
) -> npt.NDArray[np.float32]:
    """Convert game input dict to action vector using layout."""
    out = np.zeros(total_dim, dtype=np.float32)
    for name, start, end, n_bins in layout:
        val = float(inp.get(name, False))
        out[start:end] = val
    return out


def action_vector_to_game_input(
    action_vec: npt.NDArray[np.floating] | list[float],
    n_steer_parts: int,
    threshold: float = 0.5,
) -> dict[str, bool]:
    """Legacy: convert action vector (2+2*n_steer_parts) to game input. Prefer ActionSpace.from_config(cfg).to_game_input(vec)."""
    layout = [
        ("accelerate", 0, 1, 1),
        ("brake", 1, 2, 1),
        ("left", 2, 2 + n_steer_parts, n_steer_parts),
        ("right", 2 + n_steer_parts, 2 + 2 * n_steer_parts, n_steer_parts),
    ]
    return action_vector_to_game_input_from_layout(
        action_vec, layout, 2 + 2 * n_steer_parts, threshold
    )


def game_input_to_action_vector(
    inp: dict[str, Any],
    n_steer_parts: int,
) -> npt.NDArray[np.float32]:
    """Convert game input dict to action vector (legacy layout). Prefer ActionSpace.from_config(cfg).from_game_input(inp)."""
    return game_input_to_action_vector_from_layout(
        inp,
        [
            ("accelerate", 0, 1, 1),
            ("brake", 1, 2, 1),
            ("left", 2, 2 + n_steer_parts, n_steer_parts),
            ("right", 2 + n_steer_parts, 2 + 2 * n_steer_parts, n_steer_parts),
        ],
        2 + 2 * n_steer_parts,
    )


def random_action_vector(n_steer_parts: int, rng: np.random.Generator | None = None) -> npt.NDArray[np.float32]:
    """Legacy: random binary action vector (2+2*n_steer_parts). Prefer ActionSpace.from_config(cfg).random()."""
    if rng is None:
        rng = np.random.default_rng()
    n = n_action_dims(n_steer_parts)
    return (rng.random(n) > 0.5).astype(np.float32)
