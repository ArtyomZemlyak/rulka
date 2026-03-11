"""
Unified float vector construction for RL and BC.

Single source of truth for building the float observation vector.
Includes all car state and track state fields we get from the game (SimStateData / meta).
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

# Length of car_track_extra block: dyna (add_linear_speed 3, force 3, torque 3) + mobil inputs (3) +
# sync speed_forward/sideward (2) + max_linear_speed (1) + current_local_speed (3) + turbo (2) +
# has_any_lateral_contact (1) + burnout_state (1) + engine (slide_factor, braking_factor, clamped_rpm) (3) +
# track (distance_since_track_begin, meters_in_current_zone, segment_length_meters, current_zone_idx) (4)
CAR_TRACK_EXTRA_DIM = 29


class FloatStateDict(TypedDict):
    """Canonical state dict for build_float_vector. All game car/track state we use for RL/BC."""

    gear_and_wheels: npt.NDArray[np.float32]
    angular_velocity: npt.NDArray[np.float32]
    velocity: npt.NDArray[np.float32]
    y_map: npt.NDArray[np.float32]
    zone_centers_in_car_frame: npt.NDArray[np.float32]
    margin: float
    is_freewheeling: float
    turning_rate: float
    mobil_is_sliding: float
    # Order: add_linear_speed(3), force(3), torque(3), input_gas/brake/steer(3), speed_forward, speed_sideward(2),
    # max_linear_speed(1), current_local_speed(3), turbo_boost_factor, is_turbo(2), has_any_lateral_contact(1),
    # burnout_state(1), engine_slide_factor, engine_braking_factor, engine_clamped_rpm(3),
    # distance_since_track_begin, meters_in_current_zone, segment_length_meters, current_zone_idx(4)
    car_track_extra: npt.NDArray[np.float32]


def build_float_vector(
    state: FloatStateDict,
    prev_actions_flat: npt.NDArray[np.floating],
    temporal: float,
    config: Any,
) -> npt.NDArray[np.float32]:
    """Build the canonical float vector from state dict and prev actions.

    Layout: [temporal, prev_actions, gear_and_wheels, ang_vel, vel, y_map, zone_centers, margin, freewheel,
             turning_rate, mobil_is_sliding, car_track_extra(CAR_TRACK_EXTRA_DIM)].
    """
    n_zone = config.n_zone_centers_in_inputs
    zone_arr = np.asarray(state["zone_centers_in_car_frame"], dtype=np.float32).ravel()
    if len(zone_arr) > n_zone * 3:
        zone_arr = zone_arr[: n_zone * 3]
    elif len(zone_arr) < n_zone * 3:
        zone_arr = np.pad(zone_arr, (0, n_zone * 3 - len(zone_arr)))

    extra = np.asarray(state["car_track_extra"], dtype=np.float32).ravel()
    if len(extra) < CAR_TRACK_EXTRA_DIM:
        extra = np.pad(extra, (0, CAR_TRACK_EXTRA_DIM - len(extra)))
    else:
        extra = extra[:CAR_TRACK_EXTRA_DIM]

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
            float(state["turning_rate"]),
            float(state["mobil_is_sliding"]),
            extra,
        )
    ).astype(np.float32)


def prev_actions_flat_from_indices(
    prev_action_indices: list[int],
    inputs: list[dict],
    action_forward_idx: int,
    n_steer_parts: int,
) -> npt.NDArray[np.float32]:
    """Build flat action vector * n_prev from action indices (legacy: each index → inputs[idx] → vector)."""
    from trackmania_rl.action_vector import game_input_to_action_vector

    out: list[npt.NDArray[np.float32]] = []
    for idx in prev_action_indices:
        if idx < 0 or idx >= len(inputs):
            act = inputs[action_forward_idx]
        else:
            act = inputs[idx]
        out.append(game_input_to_action_vector(act, n_steer_parts))
    return np.concatenate(out, axis=0).astype(np.float32)


def prev_actions_flat_from_rollout_actions(
    actions_list: list[npt.NDArray[np.floating]],
    n_prev: int,
    n_steer_parts: int,
) -> npt.NDArray[np.float32]:
    """Build prev_actions flat from rollout list of action vectors. Pads with zeros if len(actions_list) < n_prev."""
    n_ad = 2 + 2 * n_steer_parts
    zero = np.zeros(n_ad, dtype=np.float32)
    if len(actions_list) >= n_prev:
        prev_actions = actions_list[-n_prev:]
    else:
        prev_actions = [zero] * (n_prev - len(actions_list)) + list(actions_list)
    return prev_actions_flat_from_actions(prev_actions, n_steer_parts)


def prev_actions_flat_from_actions(
    previous_actions: list[dict] | list[npt.NDArray[np.floating]],
    n_steer_parts: int,
) -> npt.NDArray[np.float32]:
    """Build flat [accel, brake, left_1..left_N, right_1..right_N]*n from list of action dicts or vectors."""
    from trackmania_rl.action_vector import game_input_to_action_vector

    out: list[npt.NDArray[np.float32]] = []
    for act in previous_actions:
        if isinstance(act, dict):
            out.append(game_input_to_action_vector(act, n_steer_parts))
        else:
            arr = np.asarray(act, dtype=np.float32).ravel()
            n = 2 + 2 * n_steer_parts
            if len(arr) < n:
                arr = np.pad(arr, (0, n - len(arr)))
            out.append(arr[:n])
    return np.concatenate(out, axis=0).astype(np.float32)


def _car_track_extra_from_meta(meta_dict: dict[str, Any]) -> npt.NDArray[np.float32]:
    """Build car_track_extra array (length CAR_TRACK_EXTRA_DIM) from manifest meta."""
    dyna = meta_dict.get("dyna_current") or {}
    mobil = meta_dict.get("mobil") or {}
    sync = mobil.get("sync_vehicle_state") or {}
    engine = mobil.get("engine") or {}

    def vec3(key: str, default: list[float] = (0.0, 0.0, 0.0)) -> list[float]:
        v = dyna.get(key, default)
        return list(v)[:3] if v else list(default)[:3]

    def flt(d: dict, key: str, default: float = 0.0) -> float:
        return float(d.get(key, default))

    add_linear_speed = vec3("add_linear_speed")
    force = vec3("force")
    torque = vec3("torque")
    input_gas = flt(mobil, "input_gas")
    input_brake = flt(mobil, "input_brake")
    input_steer = flt(mobil, "input_steer")
    speed_forward = flt(sync, "speed_forward")
    speed_sideward = flt(sync, "speed_sideward")
    max_linear_speed = flt(mobil, "max_linear_speed")
    cl = mobil.get("current_local_speed") or [0.0, 0.0, 0.0]
    current_local_speed = (list(cl)[:3] + [0.0, 0.0, 0.0])[:3]
    turbo_boost_factor = flt(mobil, "turbo_boost_factor")
    is_turbo = float(mobil.get("is_turbo", sync.get("is_turbo", False)))
    has_any_lateral_contact = float(mobil.get("has_any_lateral_contact", False))
    burnout_state = float(mobil.get("burnout_state", 0))
    engine_slide_factor = flt(engine, "slide_factor", 1.0)
    engine_braking_factor = flt(engine, "braking_factor")
    engine_clamped_rpm = flt(engine, "clamped_rpm")
    distance_since_track_begin = float(meta_dict.get("distance_since_track_begin", 0.0))
    meters_in_current_zone = float(meta_dict.get("meters_in_current_zone", 0.0))
    segment_length_meters = float(meta_dict.get("segment_length_meters", 0.0))
    current_zone_idx = float(meta_dict.get("current_zone_idx", 0))

    arr = np.array(
        add_linear_speed + force + torque
        + [input_gas, input_brake, input_steer]
        + [speed_forward, speed_sideward]
        + [max_linear_speed]
        + current_local_speed
        + [turbo_boost_factor, is_turbo, has_any_lateral_contact, burnout_state]
        + [engine_slide_factor, engine_braking_factor, engine_clamped_rpm]
        + [distance_since_track_begin, meters_in_current_zone, segment_length_meters, current_zone_idx],
        dtype=np.float32,
    )
    assert len(arr) == CAR_TRACK_EXTRA_DIM, f"car_track_extra length {len(arr)} != {CAR_TRACK_EXTRA_DIM}"
    return arr


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
    mobil = meta_dict.get("mobil") or {}
    turning_rate = float(mobil.get("turning_rate", 0.0))
    mobil_is_sliding = float(mobil.get("is_sliding", False))
    car_track_extra = _car_track_extra_from_meta(meta_dict)

    return FloatStateDict(
        gear_and_wheels=gear,
        angular_velocity=ang_vel,
        velocity=vel,
        y_map=y_map,
        zone_centers_in_car_frame=zone_arr,
        margin=float(meta_dict.get("margin", 0.0)),
        is_freewheeling=float(meta_dict.get("is_freewheeling", 0.0)),
        turning_rate=turning_rate,
        mobil_is_sliding=mobil_is_sliding,
        car_track_extra=car_track_extra,
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

    turning_rate = float(getattr(mobil, "turning_rate", 0.0))
    mobil_is_sliding = float(getattr(mobil, "is_sliding", False))

    add_linear_speed = np.array(getattr(dyna, "add_linear_speed", [0, 0, 0]), dtype=np.float32)
    force = np.array(getattr(dyna, "force", [0, 0, 0]), dtype=np.float32)
    torque = np.array(getattr(dyna, "torque", [0, 0, 0]), dtype=np.float32)
    input_gas = float(getattr(mobil, "input_gas", 0.0))
    input_brake = float(getattr(mobil, "input_brake", 0.0))
    input_steer = float(getattr(mobil, "input_steer", 0.0))
    sync_state = getattr(mobil, "sync_vehicle_state", None)
    speed_forward = float(getattr(sync_state, "speed_forward", 0.0)) if sync_state else 0.0
    speed_sideward = float(getattr(sync_state, "speed_sideward", 0.0)) if sync_state else 0.0
    max_linear_speed = float(getattr(mobil, "max_linear_speed", 0.0))
    current_local_speed = np.array(getattr(mobil, "current_local_speed", [0, 0, 0]), dtype=np.float32).ravel()[:3]
    if len(current_local_speed) < 3:
        current_local_speed = np.pad(current_local_speed, (0, 3 - len(current_local_speed)))
    turbo_boost_factor = float(getattr(mobil, "turbo_boost_factor", 0.0))
    is_turbo = float(getattr(sync_state, "is_turbo", False)) if sync_state else float(getattr(mobil, "turbo_type", 0) != 0)
    has_any_lateral_contact = float(getattr(mobil, "has_any_lateral_contact", False))
    burnout_state = float(getattr(mobil, "burnout_state", 0))
    engine_slide_factor = float(getattr(mobil_engine, "slide_factor", 1.0))
    engine_braking_factor = float(getattr(mobil_engine, "braking_factor", 0.0))
    engine_clamped_rpm = float(getattr(mobil_engine, "clamped_rpm", 0.0))
    segment_length_meters = float(distance_between_zone_transitions[current_zone_idx - 1])

    car_track_extra = np.array(
        [
            *add_linear_speed.ravel()[:3],
            *force.ravel()[:3],
            *torque.ravel()[:3],
            input_gas,
            input_brake,
            input_steer,
            speed_forward,
            speed_sideward,
            max_linear_speed,
            *current_local_speed.ravel()[:3],
            turbo_boost_factor,
            is_turbo,
            has_any_lateral_contact,
            burnout_state,
            engine_slide_factor,
            engine_braking_factor,
            engine_clamped_rpm,
            float(distance_since_track_begin),
            float(meters_in_current_zone),
            segment_length_meters,
            float(current_zone_idx),
        ],
        dtype=np.float32,
    )

    state_dict: FloatStateDict = FloatStateDict(
        gear_and_wheels=gear_and_wheels,
        angular_velocity=ang_vel_car,
        velocity=vel_car,
        y_map=y_map,
        zone_centers_in_car_frame=zone_centers_in_car_frame,
        margin=float(margin),
        is_freewheeling=float(mobil.is_freewheeling),
        turning_rate=turning_rate,
        mobil_is_sliding=mobil_is_sliding,
        car_track_extra=car_track_extra,
    )
    return state_dict, current_zone_idx, float(distance_since_track_begin)
