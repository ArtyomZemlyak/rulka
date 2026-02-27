"""
Utilities for parsing SimStateData and computing zone-dependent race state fields.

Used during capture to extract full race state (without previous_actions) for manifest meta.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from config_files.config_loader import get_config
from trackmania_rl import contact_materials, map_loader
from trackmania_rl.tmi_interaction.game_instance_manager import update_current_zone_idx


def sim_state_to_dict(sim_state: Any, last_gear_and_wheels: npt.NDArray[np.float32] | None = None) -> dict[str, Any]:
    """Parse SimStateData into a JSON-serializable dict (raw race state, no zone fields).

    Args:
        sim_state: SimStateData from get_simulation_state()
        last_gear_and_wheels: optional previous gear_and_wheels for counter_gearbox_state

    Returns:
        Dict with: race_time, position, velocity, orientation_flat, angular_velocity,
        gear_and_wheels, is_freewheeling. All arrays as lists for JSON.
    """
    cfg = get_config()
    dyna = sim_state.dyna.current_state
    mobil = sim_state.scene_mobil
    mobil_engine = mobil.engine
    simulation_wheels = sim_state.simulation_wheels
    wheel_state = [simulation_wheels[i].real_time_state for i in range(4)]

    position = np.array(dyna.position, dtype=np.float32)
    orientation = dyna.rotation.to_numpy().T  # (3, 3)
    velocity = np.array(dyna.linear_speed, dtype=np.float32)
    angular_velocity = np.array(dyna.angular_speed, dtype=np.float32)

    gearbox_state = mobil.gearbox_state
    counter_gearbox_state = 0
    if gearbox_state != 0 and last_gear_and_wheels is not None and len(last_gear_and_wheels) > 15:
        counter_gearbox_state = 1 + int(last_gear_and_wheels[15])

    gear_and_wheels = np.array(
        [
            *(float(ws.is_sliding) for ws in wheel_state),
            *(float(ws.has_ground_contact) for ws in wheel_state),
            *(float(ws.damper_absorb) for ws in wheel_state),
            float(gearbox_state),
            float(mobil_engine.gear),
            float(mobil_engine.actual_rpm),
            float(counter_gearbox_state),
            *(
                float(
                    i
                    == contact_materials.physics_behavior_fromint.get(
                        ws.contact_material_id & 0xFFFF, 0
                    )
                )
                for ws in wheel_state
                for i in range(cfg.n_contact_material_physics_behavior_types)
            ),
        ],
        dtype=np.float32,
    )

    return {
        "race_time": int(sim_state.race_time),
        "position": position.tolist(),
        "velocity": velocity.tolist(),
        "orientation_flat": orientation.ravel().tolist(),
        "angular_velocity": angular_velocity.tolist(),
        "gear_and_wheels": gear_and_wheels.tolist(),
        "is_freewheeling": float(mobil.is_freewheeling),
    }


def add_zone_fields(
    raw_dict: dict[str, Any],
    zone_centers: npt.NDArray[np.floating],
    zone_transitions: npt.NDArray[np.floating],
    distance_between_zone_transitions: npt.NDArray[np.floating],
    distance_from_start_track_to_prev_zone_transition: npt.NDArray[np.floating],
    normalized_vector_along_track_axis: npt.NDArray[np.floating],
    current_zone_idx: int,
    next_real_checkpoint_positions: npt.NDArray[np.floating],
    max_allowable_distance_to_real_checkpoint: npt.NDArray[np.floating],
) -> dict[str, Any]:
    """Add zone-dependent fields to raw dict. Modifies in place and returns it."""
    cfg = get_config()
    position = np.array(raw_dict["position"], dtype=np.float32)
    orientation = np.array(raw_dict["orientation_flat"], dtype=np.float32).reshape(3, 3)

    deck_height_val = (
        float(cfg.deck_height) if isinstance(cfg.deck_height, str) else cfg.deck_height
    )
    if position[1] > deck_height_val:
        current_zone_idx = update_current_zone_idx(
            current_zone_idx,
            zone_centers.astype(np.float32),
            position,
            cfg.max_allowable_distance_to_virtual_checkpoint,
            next_real_checkpoint_positions.astype(np.float32),
            max_allowable_distance_to_real_checkpoint.astype(np.float32),
            cfg.n_zone_centers_extrapolate_after_end_of_map,
        )

    meters_in_current_zone = np.clip(
        (position - zone_transitions[current_zone_idx - 1]).dot(
            normalized_vector_along_track_axis[current_zone_idx - 1]
        ),
        0,
        distance_between_zone_transitions[current_zone_idx - 1],
    )
    distance_since_track_begin = (
        distance_from_start_track_to_prev_zone_transition[current_zone_idx - 1] + meters_in_current_zone
    )
    margin = min(
        cfg.margin_to_announce_finish_meters,
        distance_from_start_track_to_prev_zone_transition[
            len(zone_centers) - cfg.n_zone_centers_extrapolate_after_end_of_map
        ]
        - distance_since_track_begin,
    )
    zone_centers_in_car_frame = orientation.dot(
        (
            zone_centers[
                current_zone_idx : current_zone_idx
                + cfg.one_every_n_zone_centers_in_inputs * cfg.n_zone_centers_in_inputs : cfg.one_every_n_zone_centers_in_inputs,
                :,
            ]
            - position
        ).T
    ).T.ravel()

    raw_dict["zone_centers_in_car_frame"] = zone_centers_in_car_frame.tolist()
    raw_dict["margin"] = float(margin)
    raw_dict["current_zone_idx"] = int(current_zone_idx)
    return raw_dict


def process_zone_centers(raw_zone_centers: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Apply extrapolation and smoothing (same as load_next_map_zone_centers)."""
    cfg = get_config()
    zone_centers = np.vstack(
        (
            raw_zone_centers[0]
            + np.expand_dims(raw_zone_centers[0] - raw_zone_centers[1], axis=0)
            * np.expand_dims(
                np.arange(cfg.n_zone_centers_extrapolate_before_start_of_map, 0, -1), axis=1
            ),
            raw_zone_centers,
            raw_zone_centers[-1]
            + np.expand_dims(raw_zone_centers[-1] - raw_zone_centers[-2], axis=0)
            * np.expand_dims(
                np.arange(1, 1 + cfg.n_zone_centers_extrapolate_after_end_of_map, 1), axis=1
            ),
        )
    )
    zone_centers[5:-5] = 0.5 * (zone_centers[:-10] + zone_centers[10:])
    return zone_centers


def load_zone_centers_from_vcp(vcp_path: Path) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Load VCP file and return zone data for add_zone_fields.

    Returns:
        (zone_centers, zone_transitions, distance_between_zone_transitions,
         distance_from_start_track_to_prev_zone_transition, normalized_vector_along_track_axis,
         next_real_checkpoint_positions, max_allowable_distance_to_real_checkpoint)
    """
    raw = np.load(str(vcp_path))
    zone_centers = process_zone_centers(raw)
    (
        zone_transitions,
        distance_between_zone_transitions,
        distance_from_start_track_to_prev_zone_transition,
        normalized_vector_along_track_axis,
    ) = map_loader.precalculate_virtual_checkpoints_information(zone_centers)
    n = len(zone_centers)
    next_real_checkpoint_positions = np.zeros((n, 3))
    max_allowable = 9999999.0 * np.ones(n)
    return (
        zone_centers,
        zone_transitions,
        distance_between_zone_transitions,
        distance_from_start_track_to_prev_zone_transition,
        normalized_vector_along_track_axis,
        next_real_checkpoint_positions,
        max_allowable,
    )
