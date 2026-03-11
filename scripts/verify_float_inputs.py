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
    CAR_TRACK_EXTRA_DIM,
    build_float_vector,
    prev_actions_flat_from_indices,
    state_dict_from_meta,
)


def main() -> int:
    set_config(load_config(Path(__file__).parents[1] / "config_files" / "rl" / "config_default.yaml"))
    cfg = get_config()

    # Build minimal valid meta dict
    n_zone = cfg.n_zone_centers_in_inputs
    n_contact = cfg.n_contact_material_physics_behavior_types
    gear_len = 4 + 4 + 4 + 1 + 1 + 1 + 1 + 4 * n_contact
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
    prev_flat = prev_actions_flat_from_indices(prev_indices, cfg.inputs, cfg.action_forward_idx, cfg.n_steer_parts)
    vec = build_float_vector(state_dict, prev_flat, 0.0, cfg)

    expected_dim = int(cfg.float_input_dim)
    if vec.shape != (expected_dim,):
        print(f"FAIL: vec.shape={vec.shape} != ({expected_dim},)")
        return 1
    if vec.dtype != np.float32:
        print(f"FAIL: vec.dtype={vec.dtype} != float32")
        return 1
    if vec[0] != 0.0:
        print(f"FAIL: vec[0] (temporal)={vec[0]} != 0")
        return 1

    # Segment lengths: temporal(1), prev_actions(n_prev*n_action_dims), gear(32), ang_vel(3), vel(3), y_map(3),
    # zone_centers(n_zone*3), margin(1), freewheel(1), turning_rate(1), mobil_is_sliding(1), car_track_extra(29)
    n_ad = 2 + 2 * getattr(cfg, "n_steer_parts", 1)
    off = 1 + cfg.n_prev_actions_in_inputs * n_ad + gear_len + 3 + 3 + 3 + n_zone * 3 + 1 + 1 + 1 + 1 + CAR_TRACK_EXTRA_DIM
    if off != expected_dim:
        print(f"FAIL: segment sum={off} != float_input_dim={expected_dim}")
        return 1

    print("PASS: vec.shape, dtype, temporal, segments OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
