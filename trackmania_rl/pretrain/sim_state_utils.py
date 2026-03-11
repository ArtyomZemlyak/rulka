"""
Utilities for parsing SimStateData and computing zone-dependent race state fields.

Used during capture to extract full race state (without previous_actions) for manifest meta.
We save all car-state-related fields the game provides (dyna, scene_mobil, wheels)
so meta can be used for future extensions without re-capturing. We do NOT save
version, flags, timers, cp_data (not per-frame car state or duplicate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from config_files.config_loader import get_config
from trackmania_rl import contact_materials, map_loader
from trackmania_rl.tmi_interaction.game_instance_manager import update_current_zone_idx


def _to_list(arr: Any, default: list[float] | None = None) -> list[float]:
    """Convert array-like to list for JSON. Return default if missing or not convertible."""
    if arr is None:
        return default or []
    try:
        a = np.asarray(arr, dtype=np.float64)
        return a.ravel().tolist()
    except (TypeError, ValueError):
        return default or []


def _to_list_2d(arr: Any, shape: tuple[int, int] = (3, 3)) -> list[list[float]]:
    """Convert 2D array to list of lists."""
    if arr is None:
        return []
    try:
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1 and len(a) >= shape[0] * shape[1]:
            a = a.reshape(shape)
        return a.tolist()
    except (TypeError, ValueError):
        return []


def _dyna_state_to_dict(state: Any) -> dict[str, Any]:
    """Serialize one HmsDynaStateStruct (current, previous, or temp) to JSON-serializable dict."""
    if state is None:
        return {}
    out: dict[str, Any] = {}
    out["position"] = _to_list(getattr(state, "position", None), [0.0, 0.0, 0.0])
    out["quat"] = _to_list(getattr(state, "quat", None), [1.0, 0.0, 0.0, 0.0])
    rot = getattr(state, "rotation", None)
    if rot is not None and hasattr(rot, "to_numpy"):
        out["rotation"] = np.asarray(rot.to_numpy().T, dtype=np.float64).ravel().tolist()
    else:
        out["rotation"] = _to_list_2d(rot, (3, 3)) if rot is not None else []
    out["linear_speed"] = _to_list(getattr(state, "linear_speed", None), [0.0, 0.0, 0.0])
    out["add_linear_speed"] = _to_list(getattr(state, "add_linear_speed", None), [0.0, 0.0, 0.0])
    out["angular_speed"] = _to_list(getattr(state, "angular_speed", None), [0.0, 0.0, 0.0])
    out["force"] = _to_list(getattr(state, "force", None), [0.0, 0.0, 0.0])
    out["torque"] = _to_list(getattr(state, "torque", None), [0.0, 0.0, 0.0])
    iit = getattr(state, "inverse_inertia_tensor", None)
    out["inverse_inertia_tensor"] = _to_list_2d(iit, (3, 3)) if iit is not None else []
    out["unknown"] = float(getattr(state, "unknown", 0.0))
    out["not_tweaked_linear_speed"] = _to_list(getattr(state, "not_tweaked_linear_speed", None), [0.0, 0.0, 0.0])
    out["owner"] = int(getattr(state, "owner", 0))
    return out


def _scene_vehicle_state_to_dict(svs: Any) -> dict[str, Any]:
    """Serialize one SceneVehicleCarState (sync/async/prev) to dict. Skips binary rest."""
    if svs is None:
        return {}
    return {
        "speed_forward": float(getattr(svs, "speed_forward", 0.0)),
        "speed_sideward": float(getattr(svs, "speed_sideward", 0.0)),
        "input_steer": float(getattr(svs, "input_steer", 0.0)),
        "input_gas": float(getattr(svs, "input_gas", 0.0)),
        "input_brake": float(getattr(svs, "input_brake", 0.0)),
        "is_turbo": bool(getattr(svs, "is_turbo", False)),
        "rpm": float(getattr(svs, "rpm", 0.0)),
        "gearbox_state": int(getattr(svs, "gearbox_state", 0)),
    }


def _wheel_real_time_to_dict(rts: Any) -> dict[str, Any]:
    """Serialize RealTimeState of one wheel (extra fields beyond what we use in gear_and_wheels)."""
    if rts is None:
        return {}
    return {
        "damper_absorb": float(getattr(rts, "damper_absorb", 0.0)),
        "field_4": float(getattr(rts, "field_4", 0.0)),
        "field_8": float(getattr(rts, "field_8", 0.0)),
        "field_12": _to_list_2d(getattr(rts, "field_12", None), (3, 3)),
        "field_48": _to_list_2d(getattr(rts, "field_48", None), (3, 3)),
        "field_84": _to_list(getattr(rts, "field_84", None)),
        "field_108": float(getattr(rts, "field_108", 0.0)),
        "has_ground_contact": bool(getattr(rts, "has_ground_contact", False)),
        "contact_material_id": int(getattr(rts, "contact_material_id", 0)),
        "is_sliding": bool(getattr(rts, "is_sliding", False)),
        "relative_rotz_axis": _to_list(getattr(rts, "relative_rotz_axis", None), [0.0, 0.0, 1.0]),
        "nb_ground_contacts": int(getattr(rts, "nb_ground_contacts", 0)),
        "field_144": _to_list(getattr(rts, "field_144", None)),
    }


def _surface_handler_to_dict(sh: Any) -> dict[str, Any]:
    """Serialize SurfaceHandler of one wheel (position, rotation, unknown matrix)."""
    if sh is None:
        return {}
    return {
        "unknown": _to_list_2d(getattr(sh, "unknown", None), (4, 3)),
        "rotation": _to_list_2d(getattr(sh, "rotation", None), (3, 3)),
        "position": _to_list(getattr(sh, "position", None)),
    }


def _simulation_wheel_to_dict(sw: Any) -> dict[str, Any]:
    """Serialize one SimulationWheel (steerable, surface_handler, offset, real_time_state)."""
    if sw is None:
        return {}
    rts = getattr(sw, "real_time_state", None)
    sh = getattr(sw, "surface_handler", None)
    return {
        "steerable": bool(getattr(sw, "steerable", False)),
        "field_8": int(getattr(sw, "field_8", 0)),
        "surface_handler": _surface_handler_to_dict(sh),
        "field_112": _to_list_2d(getattr(sw, "field_112", None), (4, 3)),
        "field_160": int(getattr(sw, "field_160", 0)),
        "field_164": int(getattr(sw, "field_164", 0)),
        "offset_from_vehicle": _to_list(getattr(sw, "offset_from_vehicle", None)),
        "real_time_state": _wheel_real_time_to_dict(rts),
        "field_348": int(getattr(sw, "field_348", 0)),
        "contact_relative_local_distance": _to_list(getattr(sw, "contact_relative_local_distance", None)),
    }


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

    # --- Full dyna (current, previous, temp) for meta ---
    dyna = getattr(sim_state, "dyna", None)
    dyna_current = _dyna_state_to_dict(getattr(dyna, "current_state", None) if dyna else None)
    dyna_previous = _dyna_state_to_dict(getattr(dyna, "previous_state", None) if dyna else None)
    dyna_temp = _dyna_state_to_dict(getattr(dyna, "temp_state", None) if dyna else None)

    # --- Full scene_mobil (engine, vehicle states, all scalar/vector state) ---
    engine = getattr(mobil, "engine", None)
    engine_dict: dict[str, Any] = {}
    if engine is not None:
        engine_dict = {
            "max_rpm": float(getattr(engine, "max_rpm", 0.0)),
            "braking_factor": float(getattr(engine, "braking_factor", 0.0)),
            "clamped_rpm": float(getattr(engine, "clamped_rpm", 0.0)),
            "actual_rpm": float(getattr(engine, "actual_rpm", 0.0)),
            "slide_factor": float(getattr(engine, "slide_factor", 1.0)),
            "rear_gear": int(getattr(engine, "rear_gear", 0)),
            "gear": int(getattr(engine, "gear", 0)),
        }
    mobil_dict: dict[str, Any] = {
        "is_update_async": bool(getattr(mobil, "is_update_async", False)),
        "input_gas": float(getattr(mobil, "input_gas", 0.0)),
        "input_brake": float(getattr(mobil, "input_brake", 0.0)),
        "input_steer": float(getattr(mobil, "input_steer", 0.0)),
        "is_light_trials_set": bool(getattr(mobil, "is_light_trials_set", False)),
        "horn_limit": int(getattr(mobil, "horn_limit", 0)),
        "max_linear_speed": float(getattr(mobil, "max_linear_speed", 0.0)),
        "block_flags": int(getattr(mobil, "block_flags", 0)),
        "quality": int(getattr(mobil, "quality", 0)),
        "has_any_lateral_contact": bool(getattr(mobil, "has_any_lateral_contact", False)),
        "last_has_any_lateral_contact_time": int(getattr(mobil, "last_has_any_lateral_contact_time", -1)),
        "water_forces_applied": bool(getattr(mobil, "water_forces_applied", False)),
        "turning_rate": float(getattr(mobil, "turning_rate", 0.0)),
        "turbo_boost_factor": float(getattr(mobil, "turbo_boost_factor", 0.0)),
        "last_turbo_type_change_time": int(getattr(mobil, "last_turbo_type_change_time", 0)),
        "last_turbo_time": int(getattr(mobil, "last_turbo_time", 0)),
        "turbo_type": int(getattr(mobil, "turbo_type", 0)),
        "roulette_value": float(getattr(mobil, "roulette_value", 0.0)),
        "is_sliding": bool(getattr(mobil, "is_sliding", False)),
        "wheel_contact_absorb_counter": int(getattr(mobil, "wheel_contact_absorb_counter", 0)),
        "burnout_state": int(getattr(mobil, "burnout_state", 0)),
        "current_local_speed": _to_list(getattr(mobil, "current_local_speed", None)),
        "total_central_force_added": _to_list(getattr(mobil, "total_central_force_added", None)),
        "is_rubber_ball": bool(getattr(mobil, "is_rubber_ball", False)),
        "saved_state": _to_list_2d(getattr(mobil, "saved_state", None), (4, 3)),
        "engine": engine_dict,
        "prev_sync_vehicle_state": _scene_vehicle_state_to_dict(getattr(mobil, "prev_sync_vehicle_state", None)),
        "sync_vehicle_state": _scene_vehicle_state_to_dict(getattr(mobil, "sync_vehicle_state", None)),
        "async_vehicle_state": _scene_vehicle_state_to_dict(getattr(mobil, "async_vehicle_state", None)),
        "prev_async_vehicle_state": _scene_vehicle_state_to_dict(getattr(mobil, "prev_async_vehicle_state", None)),
    }

    # --- All four simulation wheels (full state per wheel) ---
    sim_wheels = getattr(sim_state, "simulation_wheels", None)
    wheels_list: list[dict[str, Any]] = []
    if sim_wheels is not None:
        for i in range(4):
            w = sim_wheels[i] if hasattr(sim_wheels, "__getitem__") else getattr(sim_wheels, f"wheel_{i}", None)
            wheels_list.append(_simulation_wheel_to_dict(w))

    return {
        "race_time": int(sim_state.race_time),
        "position": position.tolist(),
        "velocity": velocity.tolist(),
        "orientation_flat": orientation.ravel().tolist(),
        "angular_velocity": angular_velocity.tolist(),
        "gear_and_wheels": gear_and_wheels.tolist(),
        "is_freewheeling": float(mobil.is_freewheeling),
        "dyna_current": dyna_current,
        "dyna_previous": dyna_previous,
        "dyna_temp": dyna_temp,
        "mobil": mobil_dict,
        "simulation_wheels": wheels_list,
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
    raw_dict["distance_since_track_begin"] = float(distance_since_track_begin)
    raw_dict["meters_in_current_zone"] = float(meters_in_current_zone)
    raw_dict["segment_length_meters"] = float(distance_between_zone_transitions[current_zone_idx - 1])
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
