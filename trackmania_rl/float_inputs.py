"""
Unified float vector construction for RL and BC.

Single source of truth for building the float observation vector.
Used by both RL (game_env_backend, game_instance_manager) and BC (preprocess).
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt


class FloatStateDict(TypedDict):
    """Canonical state dict for build_float_vector."""

    gear_and_wheels: npt.NDArray[np.float32]
    angular_velocity: npt.NDArray[np.float32]
    velocity: npt.NDArray[np.float32]
    y_map: npt.NDArray[np.float32]
    zone_centers_in_car_frame: npt.NDArray[np.float32]
    margin: float
    is_freewheeling: float


def build_float_vector(
    state: FloatStateDict,
    prev_actions_flat: npt.NDArray[np.floating],
    temporal: float,
    config: Any,
) -> npt.NDArray[np.float32]:
    """Build the canonical float vector from state dict and prev actions.

    Layout: [temporal, prev_actions, gear_and_wheels, ang_vel, vel, y_map, zone_centers, margin, freewheel]
    """
    n_zone = config.n_zone_centers_in_inputs
    zone_arr = np.asarray(state["zone_centers_in_car_frame"], dtype=np.float32).ravel()
    if len(zone_arr) > n_zone * 3:
        zone_arr = zone_arr[: n_zone * 3]
    elif len(zone_arr) < n_zone * 3:
        zone_arr = np.pad(zone_arr, (0, n_zone * 3 - len(zone_arr)))

    return np.hstack(
        (
            float(temporal),
            np.asarray(prev_actions_flat, dtype=np.float32).ravel(),
            np.asarray(state["gear_and_wheels"], dtype=np.float32).ravel(),
            np.asarray(state["angular_velocity"], dtype=np.float32).ravel(),
            np.asarray(state["velocity"], dtype=np.float32).ravel(),
            np.asarray(state["y_map"], dtype=np.float32).ravel(),
            zone_arr,
            float(state["margin"]),
            float(state["is_freewheeling"]),
        )
    ).astype(np.float32)


def prev_actions_flat_from_indices(
    prev_action_indices: list[int],
    inputs: list[dict],
    action_forward_idx: int,
) -> npt.NDArray[np.float32]:
    """Build flat [accel, brake, left, right]*n_prev from action indices."""
    out: list[float] = []
    for idx in prev_action_indices:
        if idx < 0 or idx >= len(inputs):
            act = inputs[action_forward_idx]
        else:
            act = inputs[idx]
        for k in ["accelerate", "brake", "left", "right"]:
            out.append(float(act.get(k, False)))
    return np.array(out, dtype=np.float32)


def prev_actions_flat_from_actions(
    previous_actions: list[dict],
) -> npt.NDArray[np.float32]:
    """Build flat [accel, brake, left, right]*n from list of action dicts."""
    out: list[float] = []
    for act in previous_actions:
        for k in ["accelerate", "brake", "left", "right"]:
            out.append(float(act.get(k, False)))
    return np.array(out, dtype=np.float32)


def state_dict_from_meta(meta_dict: dict[str, Any], config: Any) -> FloatStateDict:
    """Build FloatStateDict from BC manifest meta dict."""
    n_zone = config.n_zone_centers_in_inputs
    gear = np.array(meta_dict.get("gear_and_wheels", []), dtype=np.float32)
    ang_vel = np.array(meta_dict.get("angular_velocity", [0, 0, 0]), dtype=np.float32)
    vel = np.array(meta_dict.get("velocity", [0, 0, 0]), dtype=np.float32)
    ori = np.array(
        meta_dict.get("orientation_flat", list(np.eye(3).ravel())), dtype=np.float32
    ).reshape(3, 3)
    y_map = (ori @ np.array([0, 1, 0], dtype=np.float32)).ravel()
    zone_in_car = meta_dict.get("zone_centers_in_car_frame")
    if zone_in_car is not None:
        zone_arr = np.array(zone_in_car, dtype=np.float32)
        if len(zone_arr) > n_zone * 3:
            zone_arr = zone_arr[: n_zone * 3]
        elif len(zone_arr) < n_zone * 3:
            zone_arr = np.pad(zone_arr, (0, n_zone * 3 - len(zone_arr)))
    else:
        zone_arr = np.zeros(n_zone * 3, dtype=np.float32)
    return FloatStateDict(
        gear_and_wheels=gear,
        angular_velocity=ang_vel,
        velocity=vel,
        y_map=y_map,
        zone_centers_in_car_frame=zone_arr,
        margin=float(meta_dict.get("margin", 0.0)),
        is_freewheeling=float(meta_dict.get("is_freewheeling", 0.0)),
    )


def state_dict_from_sim_state(
    sim_state: Any,
    zone_centers: npt.NDArray[np.floating],
    zone_transitions: npt.NDArray[np.floating],
    distance_between_zone_transitions: npt.NDArray[np.floating],
    distance_from_start_track_to_prev_zone_transition: npt.NDArray[np.floating],
    normalized_vector_along_track_axis: npt.NDArray[np.floating],
    current_zone_idx: int,
    next_real_checkpoint_positions: npt.NDArray[np.floating],
    max_allowable_distance_to_real_checkpoint: npt.NDArray[np.floating],
    last_gear_and_wheels: npt.NDArray[np.float32] | None,
    config: Any,
) -> tuple[FloatStateDict, int, float]:
    """Build FloatStateDict from SimStateData and zone data for RL.

    Returns (state_dict, updated_current_zone_idx, distance_since_track_begin).
    The caller needs updated current_zone_idx and distance_since_track_begin for rollout bookkeeping.
    """
    from trackmania_rl import contact_materials
    from trackmania_rl.tmi_interaction.game_instance_manager import (
        update_current_zone_idx as _update_zone_idx,
    )

    dyna = sim_state.dyna.current_state
    mobil = sim_state.scene_mobil
    mobil_engine = mobil.engine
    simulation_wheels = sim_state.simulation_wheels
    wheel_state = [simulation_wheels[i].real_time_state for i in range(4)]

    position = np.array(dyna.position, dtype=np.float32)
    orientation = dyna.rotation.to_numpy().T
    velocity = np.array(dyna.linear_speed, dtype=np.float32)
    angular_velocity = np.array(dyna.angular_speed, dtype=np.float32)

    gearbox_state = mobil.gearbox_state
    counter_gearbox_state = 0
    if (
        gearbox_state != 0
        and last_gear_and_wheels is not None
        and len(last_gear_and_wheels) > 15
    ):
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
                    == contact_materials.physics_behavior_fromint[
                        ws.contact_material_id & 0xFFFF
                    ]
                )
                for ws in wheel_state
                for i in range(config.n_contact_material_physics_behavior_types)
            ),
        ],
        dtype=np.float32,
    )

    deck_height_val = (
        float(config.deck_height)
        if isinstance(config.deck_height, str)
        else config.deck_height
    )
    if position[1] > deck_height_val:
        current_zone_idx = _update_zone_idx(
            current_zone_idx,
            zone_centers.astype(np.float32),
            position,
            config.max_allowable_distance_to_virtual_checkpoint,
            next_real_checkpoint_positions.astype(np.float32),
            max_allowable_distance_to_real_checkpoint.astype(np.float32),
            config.n_zone_centers_extrapolate_after_end_of_map,
        )

    meters_in_current_zone = np.clip(
        (position - zone_transitions[current_zone_idx - 1]).dot(
            normalized_vector_along_track_axis[current_zone_idx - 1]
        ),
        0,
        distance_between_zone_transitions[current_zone_idx - 1],
    )
    distance_since_track_begin = (
        distance_from_start_track_to_prev_zone_transition[current_zone_idx - 1]
        + meters_in_current_zone
    )
    margin = min(
        config.margin_to_announce_finish_meters,
        distance_from_start_track_to_prev_zone_transition[
            len(zone_centers) - config.n_zone_centers_extrapolate_after_end_of_map
        ]
        - distance_since_track_begin,
    )
    zone_centers_in_car_frame = (
        orientation.dot(
            (
                zone_centers[
                    current_zone_idx : current_zone_idx
                    + config.one_every_n_zone_centers_in_inputs
                    * config.n_zone_centers_in_inputs : config.one_every_n_zone_centers_in_inputs,
                    :,
                ]
                - position
            ).T
        )
        .T.ravel()
        .astype(np.float32)
    )
    ang_vel_car = orientation.dot(angular_velocity).astype(np.float32)
    vel_car = orientation.dot(velocity).astype(np.float32)
    y_map = orientation.dot(np.array([0, 1, 0], dtype=np.float32)).ravel()

    state_dict: FloatStateDict = FloatStateDict(
        gear_and_wheels=gear_and_wheels,
        angular_velocity=ang_vel_car,
        velocity=vel_car,
        y_map=y_map,
        zone_centers_in_car_frame=zone_centers_in_car_frame,
        margin=float(margin),
        is_freewheeling=float(mobil.is_freewheeling),
    )
    return state_dict, current_zone_idx, float(distance_since_track_begin)
