#!/usr/bin/env python
"""Verify unified float inputs module produces correct shape and layout."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_files.config_loader import load_config, set_config, get_config
from trackmania_rl.float_inputs import (
    build_float_vector,
    compute_float_input_dim,
    get_segment_dim,
    get_segment_start_index,
    prev_actions_flat_from_indices,
    state_dict_from_meta,
)


def main() -> int:
    set_config(load_config(Path(__file__).parents[1] / "config_files" / "rl" / "config_default.yaml"))
    cfg = get_config()

    # Build minimal valid meta dict (dimensions from float_inputs single source of truth)
    n_zone = cfg.n_zone_centers_in_inputs
    gear_len = get_segment_dim("gear_and_wheels", cfg)
    meta = {
        "gear_and_wheels": [0.0] * gear_len,
        "angular_velocity": [0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "orientation_flat": list(np.eye(3).ravel()),
        "zone_centers_in_car_frame": [0.0] * (n_zone * 3),
        "margin": 0.0,
        "is_freewheeling": 0.0,
    }

    state_dict = state_dict_from_meta(meta, cfg)
    prev_indices = [cfg.action_forward_idx] * cfg.n_prev_actions_in_inputs
    prev_flat = prev_actions_flat_from_indices(
        prev_indices, cfg.inputs, cfg.action_forward_idx, getattr(cfg, "n_steer_parts", 1), config=cfg
    )
    vec = build_float_vector(state_dict, prev_flat, 0.0, cfg)

    expected_dim = int(cfg.float_input_dim)
    if vec.shape != (expected_dim,):
        print(f"FAIL: vec.shape={vec.shape} != ({expected_dim},)")
        return 1
    if vec.dtype != np.float32:
        print(f"FAIL: vec.dtype={vec.dtype} != float32")
        return 1
    temporal_idx = get_segment_start_index("temporal", cfg)
    if temporal_idx is not None and vec[temporal_idx] != 0.0:
        print(f"FAIL: vec[temporal_idx={temporal_idx}] (temporal)={vec[temporal_idx]} != 0")
        return 1
    if compute_float_input_dim(cfg) != expected_dim:
        print(f"FAIL: compute_float_input_dim(cfg)={compute_float_input_dim(cfg)} != float_input_dim={expected_dim}")
        return 1

    print("PASS: vec.shape, dtype, temporal (if included), float_input_dim OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
