"""
Step-wise backend for TrackManiaEnv (Gymnasium).

Episode / run / map logic:
- One game instance per collector (one process, one TMI port). The game stays running; we do not
  close the connection between episodes.
- A "run" = one episode = one env.reset() then step() until terminated/truncated. The map is
  passed in reset(options={"map_path", "zone_centers"}); the collector gets the next map from
  map_cycle and can change the map each episode.
- start_episode() loads the map if needed (request_map), saves start state once per map, then
  rewind_to_state(start_states[map_path]) so the car is at start and returns the first observation.
- Reusing the same connection for the next episode: we respond to any pending TMI sync and clear
  _pending_* so the next read is from the socket (avoids desync from stale _pending_*).

API:
- start_episode(map_path, zone_centers) -> (obs, info): run until first observation is ready.
- send_action(action_idx): queue the action to send to the game.
- run_until_next_decision_point() -> (obs, reward, terminated, truncated, info): block until next tick.
"""

import math
import socket
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from config_files.config_loader import get_config
from trackmania_rl import contact_materials, map_loader
from trackmania_rl.tmi_interaction.game_instance_manager import (
    GameInstanceManager,
    ensure_not_minimized,
    ensure_window_focused,
    update_current_zone_idx,
)


def _build_info(
    current_zone_idx: int,
    meters_advanced_along_centerline: float,
    state_float: npt.NDArray,
    car_gear_and_wheels: npt.NDArray,
    input_w: float,
) -> Dict[str, Any]:
    """Build info dict for env step/reset (per plan 11.3/11.4)."""
    return {
        "current_zone_idx": current_zone_idx,
        "meters_advanced_along_centerline": meters_advanced_along_centerline,
        "state_float": state_float,
        "car_gear_and_wheels": car_gear_and_wheels,
        "input_w": input_w,
    }


class GameEnvBackend:
    """
    Backend that runs the TMI message loop step-wise for a Gymnasium env.
    Holds a reference to GameInstanceManager and reuses its iface, request_*, grab_screen, etc.
    """

    def __init__(self, gim: GameInstanceManager):
        self.gim = gim
        self._pending_action: int | None = None
        self._last_obs: Tuple[npt.NDArray, npt.NDArray] | None = None
        self._last_info: Dict[str, Any] | None = None
        self._episode_started = False
        # TMI protocol: after _respond_to_call we must read next message before get_simulation_state.
        # Shared between start_episode and run_until_next_decision_point when one returns and the other continues.
        self._pending_msgtype: int | None = None
        self._pending_time: int | None = None
        self._pending_current: int | None = None
        self._pending_target: int | None = None
        self._pending_frame_data: bytes | None = None

    def send_action(self, action_idx: int) -> None:
        """Queue the action to be sent when the loop next advances. Called by env.step(action)."""
        self._pending_action = action_idx

    def start_episode(self, map_path: str, zone_centers: npt.NDArray) -> Tuple[Tuple[npt.NDArray, npt.NDArray], Dict[str, Any]]:
        """
        Load map, run TMI loop until the first observation (frame + floats) is ready.
        Returns (obs, info). Does not send any action; the client will call send_action then run_until_next_decision_point.
        """
        gim = self.gim
        cfg = get_config()
        (
            zone_transitions,
            distance_between_zone_transitions,
            distance_from_start_track_to_prev_zone_transition,
            normalized_vector_along_track_axis,
        ) = map_loader.precalculate_virtual_checkpoints_information(zone_centers)

        gim.ensure_game_launched()
        if time.perf_counter() - gim.last_game_reboot > get_config().game_reboot_interval:
            gim.close_game()
            gim.iface = None
            gim.launch_game()

        end_race_stats: Dict[str, Any] = {"cp_time_ms": [0]}
        instrumentation__answer_normal_step = 0
        instrumentation__answer_action_step = 0
        instrumentation__between_run_steps = 0
        instrumentation__grab_frame = 0
        instrumentation__request_inputs_and_speed = 0
        instrumentation__exploration_policy = 0
        instrumentation__convert_frame = 0
        instrumentation__grab_floats = 0

        rollout_results: Dict[str, Any] = {
            "current_zone_idx": [],
            "frames": [],
            "input_w": [],
            "actions": [],
            "action_was_greedy": [],
            "car_gear_and_wheels": [],
            "q_values": [],
            "meters_advanced_along_centerline": [],
            "state_float": [],
            "furthest_zone_idx": 0,
        }

        last_progress_improvement_ms = 0
        max_sync_retries = 4  # 5 attempts total; backoff between retries gives game time to leave menus/loading
        last_sync_err_log_time = [0.0]  # mutable to share across retries

        for _sync_attempt in range(max_sync_retries + 1):
            if (gim.iface is None) or (not gim.iface.registered):
                assert gim.msgtype_response_to_wakeup_TMI is None
                if _sync_attempt == 0:
                    print("Initialize connection to TMInterface ")
                from trackmania_rl.tmi_interaction.tminterface2 import TMInterface

                gim.iface = TMInterface(gim.tmi_port)
                connection_attempts_start_time = time.perf_counter()
                last_connection_error_message_time = time.perf_counter()
                while True:
                    try:
                        gim.iface.register(cfg.tmi_protection_timeout_s)
                        break
                    except ConnectionRefusedError:
                        current_time = time.perf_counter()
                        if current_time - last_connection_error_message_time > 1:
                            print(
                                f"Connection to TMInterface unsuccessful for {current_time - connection_attempts_start_time:.1f}s"
                            )
                            last_connection_error_message_time = current_time
                # Wait for TMI to send SC_ON_CONNECT_SYNC and stabilize; avoids reading state before protocol is in sync.
                # After a sync-error reconnect, wait longer so the game can leave menus/loading.
                time.sleep(1.0 if _sync_attempt == 0 else 3.0)
                self._pending_msgtype = None
                self._pending_time = None
                self._pending_current = None
                self._pending_target = None
                self._pending_frame_data = None
            else:
                # After previous episode: respond to pending TMI sync if any, then clear _pending_* so we read
                # the next message from the socket (avoids desync from stale _pending_* from last run).
                gim.request_speed(gim.running_speed)
                if gim.msgtype_response_to_wakeup_TMI is not None:
                    gim.iface._respond_to_call(gim.msgtype_response_to_wakeup_TMI)
                    gim.msgtype_response_to_wakeup_TMI = None
                self._pending_msgtype = None
                self._pending_time = None
                self._pending_current = None
                self._pending_target = None
                self._pending_frame_data = None

            gim.last_rollout_crashed = False
            _time = -3000
            current_zone_idx = cfg.n_zone_centers_extrapolate_before_start_of_map
            give_up_signal_has_been_sent = False
            this_rollout_has_seen_t_negative = False
            this_rollout_is_finished = False
            compute_action_asap = False
            compute_action_asap_floats = False
            frame_expected = False
            map_change_requested_time = math.inf
            last_known_simulation_state = None
            pc = 0
            pc5 = 0
            floats = None
            distance_since_track_begin = 0
            sim_state_car_gear_and_wheels = None

            from trackmania_rl.tmi_interaction.tminterface2 import MessageType

            # TMI protocol: any 4-byte send is treated as response to current sync; then TMI sends next message.
            # So we must respond first with _respond_to_call(msgtype), then read next message into buffer, then get_simulation_state().
            responded_this_iteration = False

            try:
                while not this_rollout_is_finished:
                    responded_this_iteration = False
                    if self._pending_msgtype is not None:
                        msgtype = self._pending_msgtype
                        if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                            _time = self._pending_time
                        elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                            current = self._pending_current
                            target = self._pending_target
                            _time = -3000
                        else:
                            _time = -3000
                        self._pending_msgtype = None
                        self._pending_time = None
                        self._pending_current = None
                        self._pending_target = None
                    else:
                        msgtype = gim.iface._read_int32()
                        if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                            _time = gim.iface._read_int32()
                        elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                            current = gim.iface._read_int32()
                            target = gim.iface._read_int32()
                            _time = -3000
                        else:
                            _time = -3000

                    if compute_action_asap_floats:
                        pc2 = time.perf_counter_ns()
                        sim_state_race_time = last_known_simulation_state.race_time
                        sim_state_dyna_current = last_known_simulation_state.dyna.current_state
                        sim_state_mobil = last_known_simulation_state.scene_mobil
                        sim_state_mobil_engine = sim_state_mobil.engine
                        simulation_wheels = last_known_simulation_state.simulation_wheels
                        wheel_state = [simulation_wheels[i].real_time_state for i in range(4)]
                        sim_state_position = np.array(sim_state_dyna_current.position, dtype=np.float32)
                        sim_state_orientation = sim_state_dyna_current.rotation.to_numpy().T
                        sim_state_velocity = np.array(sim_state_dyna_current.linear_speed, dtype=np.float32)
                        sim_state_angular_speed = np.array(sim_state_dyna_current.angular_speed, dtype=np.float32)
                        gearbox_state = sim_state_mobil.gearbox_state
                        counter_gearbox_state = 0
                        if gearbox_state != 0 and len(rollout_results["car_gear_and_wheels"]) > 0:
                            counter_gearbox_state = 1 + rollout_results["car_gear_and_wheels"][-1][15]
                        sim_state_car_gear_and_wheels = np.array(
                            [
                                *(ws.is_sliding for ws in wheel_state),
                                *(ws.has_ground_contact for ws in wheel_state),
                                *(ws.damper_absorb for ws in wheel_state),
                                gearbox_state,
                                sim_state_mobil_engine.gear,
                                sim_state_mobil_engine.actual_rpm,
                                counter_gearbox_state,
                                *(
                                    i == contact_materials.physics_behavior_fromint[ws.contact_material_id & 0xFFFF]
                                    for ws in wheel_state
                                    for i in range(cfg.n_contact_material_physics_behavior_types)
                                ),
                            ],
                            dtype=np.float32,
                        )
                        if sim_state_position[1] > get_config().deck_height:
                            current_zone_idx = update_current_zone_idx(
                                current_zone_idx,
                                zone_centers,
                                sim_state_position,
                                cfg.max_allowable_distance_to_virtual_checkpoint,
                                gim.next_real_checkpoint_positions,
                                gim.max_allowable_distance_to_real_checkpoint,
                                cfg.n_zone_centers_extrapolate_after_end_of_map,
                            )
                        if current_zone_idx > rollout_results["furthest_zone_idx"]:
                            last_progress_improvement_ms = sim_state_race_time
                            rollout_results["furthest_zone_idx"] = current_zone_idx
                        rollout_results["current_zone_idx"].append(current_zone_idx)
                        meters_in_current_zone = np.clip(
                            (sim_state_position - zone_transitions[current_zone_idx - 1]).dot(
                                normalized_vector_along_track_axis[current_zone_idx - 1]
                            ),
                            0,
                            distance_between_zone_transitions[current_zone_idx - 1],
                        )
                        distance_since_track_begin = (
                            distance_from_start_track_to_prev_zone_transition[current_zone_idx - 1] + meters_in_current_zone
                        )
                        state_zone_center_coordinates_in_car_reference_system = sim_state_orientation.dot(
                            (
                                zone_centers[
                                    current_zone_idx : current_zone_idx
                                    + cfg.one_every_n_zone_centers_in_inputs * cfg.n_zone_centers_in_inputs : cfg.one_every_n_zone_centers_in_inputs,
                                    :,
                                ]
                                - sim_state_position
                            ).T
                        ).T
                        state_y_map_vector_in_car_reference_system = sim_state_orientation.dot(np.array([0, 1, 0]))
                        state_car_velocity_in_car_reference_system = sim_state_orientation.dot(sim_state_velocity)
                        state_car_angular_velocity_in_car_reference_system = sim_state_orientation.dot(sim_state_angular_speed)
                        previous_actions = [
                            get_config().inputs[
                                rollout_results["actions"][k] if k >= 0 else get_config().action_forward_idx
                            ]
                            for k in range(
                                len(rollout_results["actions"]) - cfg.n_prev_actions_in_inputs,
                                len(rollout_results["actions"]),
                            )
                        ]
                        floats = np.hstack(
                            (
                                0,
                                np.array(
                                    [
                                        pa[input_str]
                                        for pa in previous_actions
                                        for input_str in ["accelerate", "brake", "left", "right"]
                                    ]
                                ),
                                sim_state_car_gear_and_wheels.ravel(),
                                state_car_angular_velocity_in_car_reference_system.ravel(),
                                state_car_velocity_in_car_reference_system.ravel(),
                                state_y_map_vector_in_car_reference_system.ravel(),
                                state_zone_center_coordinates_in_car_reference_system.ravel(),
                                min(
                                    cfg.margin_to_announce_finish_meters,
                                    distance_from_start_track_to_prev_zone_transition[
                                        len(zone_centers) - cfg.n_zone_centers_extrapolate_after_end_of_map
                                    ]
                                    - distance_since_track_begin,
                                ),
                                sim_state_mobil.is_freewheeling,
                            )
                        ).astype(np.float32)
                        pc5 = time.perf_counter_ns()
                        instrumentation__grab_floats += pc5 - pc2
                        compute_action_asap_floats = False

                    if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                        if _time > 0 and this_rollout_has_seen_t_negative and _time % 50 == 0:
                            instrumentation__between_run_steps += time.perf_counter_ns() - pc
                        pc = time.perf_counter_ns()
                        if not gim.timeout_has_been_set:
                            gim.iface.set_timeout(get_config().timeout_during_run_ms)
                            gim.iface.execute_command(f"cam {cfg.game_camera_number}")
                            gim.timeout_has_been_set = True
                        if not gim.UI_disabled and _time < map_change_requested_time:
                            gim.iface.toggle_interface(False)
                            gim.UI_disabled = True
                        # TMI protocol: client must respond to current sync first; then we can send C_GET_SIMULATION_STATE.
                        # Otherwise TMI treats our 4-byte send as response and sends next message (we'd read msgtype=1 as state_length).
                        # Same as legacy: save when _time == 0 (race start), no this_rollout_has_seen_t_negative check.
                        if _time == 0 and (map_path not in gim.start_states):
                            gim.iface._respond_to_call(msgtype)
                            responded_this_iteration = True
                            next_mt = gim.iface._read_int32()
                            if next_mt == int(MessageType.SC_RUN_STEP_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_time = gim.iface._read_int32()
                            elif next_mt == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_current = gim.iface._read_int32()
                                self._pending_target = gim.iface._read_int32()
                                self._pending_time = -3000
                            elif next_mt == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_time = -3000
                                self._pending_frame_data = gim.iface.sock.recv(
                                    cfg.W_downsized * cfg.H_downsized * 4, socket.MSG_WAITALL
                                )
                            else:
                                self._pending_msgtype = next_mt
                                self._pending_time = -3000
                            gim.start_states[map_path] = gim.iface.get_simulation_state()
                            map_name = map_path.split("/")[-1].strip('"')
                            print(f"[OK] Start state saved for {map_name} - future runs will be automatic!")
                        if (not give_up_signal_has_been_sent) and (map_path != gim.latest_map_path_requested):
                            gim.request_map(map_path, zone_centers)
                            map_change_requested_time = _time
                            give_up_signal_has_been_sent = True
                        elif (not give_up_signal_has_been_sent) and (map_path not in gim.start_states):
                            gim.iface.give_up()
                            give_up_signal_has_been_sent = True
                        elif not give_up_signal_has_been_sent:
                            gim.iface.rewind_to_state(gim.start_states[map_path])
                            _time = 0
                            give_up_signal_has_been_sent = True
                            this_rollout_has_seen_t_negative = True
                        elif (
                            (_time > gim.max_overall_duration_ms or _time > last_progress_improvement_ms + gim.max_minirace_duration_ms)
                            and this_rollout_has_seen_t_negative
                            and not this_rollout_is_finished
                        ):
                            gim.iface._respond_to_call(msgtype)
                            responded_this_iteration = True
                            next_mt = gim.iface._read_int32()
                            if next_mt == int(MessageType.SC_RUN_STEP_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_time = gim.iface._read_int32()
                            elif next_mt == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_current = gim.iface._read_int32()
                                self._pending_target = gim.iface._read_int32()
                                self._pending_time = -3000
                            elif next_mt == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_time = -3000
                                self._pending_frame_data = gim.iface.sock.recv(
                                    cfg.W_downsized * cfg.H_downsized * 4, socket.MSG_WAITALL
                                )
                            else:
                                self._pending_msgtype = next_mt
                                self._pending_time = -3000
                            simulation_state = gim.iface.get_simulation_state()
                            race_time = max([simulation_state.race_time, 1e-12])
                            end_race_stats["race_finished"] = False
                            end_race_stats["race_time"] = get_config().cutoff_rollout_if_race_not_finished_within_duration_ms
                            end_race_stats["race_time_for_ratio"] = race_time
                            end_race_stats["instrumentation__answer_normal_step"] = instrumentation__answer_normal_step / race_time * 50
                            end_race_stats["instrumentation__answer_action_step"] = instrumentation__answer_action_step / race_time * 50
                            end_race_stats["instrumentation__between_run_steps"] = instrumentation__between_run_steps / race_time * 50
                            end_race_stats["instrumentation__grab_frame"] = instrumentation__grab_frame / race_time * 50
                            end_race_stats["instrumentation__convert_frame"] = instrumentation__convert_frame / race_time * 50
                            end_race_stats["instrumentation__grab_floats"] = instrumentation__grab_floats / race_time * 50
                            end_race_stats["instrumentation__exploration_policy"] = instrumentation__exploration_policy / race_time * 50
                            end_race_stats["instrumentation__request_inputs_and_speed"] = (
                                instrumentation__request_inputs_and_speed / race_time * 50
                            )
                            end_race_stats["tmi_protection_cutoff"] = False
                            gim.iface.rewind_to_current_state()
                            gim.msgtype_response_to_wakeup_TMI = msgtype
                            gim.iface.set_timeout(get_config().timeout_between_runs_ms)
                            if frame_expected:
                                gim.iface.unrequest_frame()
                                frame_expected = False
                            this_rollout_is_finished = True
    
                        if not this_rollout_is_finished:
                            this_rollout_has_seen_t_negative |= _time < 0
                            if _time >= 0 and _time % (10 * gim.run_steps_per_action) == 0 and this_rollout_has_seen_t_negative:
                                gim.iface._respond_to_call(msgtype)
                                responded_this_iteration = True
                                next_mt = gim.iface._read_int32()
                                if next_mt == int(MessageType.SC_RUN_STEP_SYNC):
                                    self._pending_msgtype = next_mt
                                    self._pending_time = gim.iface._read_int32()
                                elif next_mt == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                                    self._pending_msgtype = next_mt
                                    self._pending_time = -3000
                                    self._pending_frame_data = gim.iface.sock.recv(
                                        cfg.W_downsized * cfg.H_downsized * 4, socket.MSG_WAITALL
                                    )
                                else:
                                    self._pending_msgtype = next_mt
                                    self._pending_time = -3000
                                last_known_simulation_state = gim.iface.get_simulation_state()
                                gim.iface.rewind_to_current_state()
                                gim.request_speed(0)
                                compute_action_asap = True
                                if compute_action_asap:
                                    compute_action_asap_floats = True
                                    frame_expected = True
                                    gim.iface.request_frame(cfg.W_downsized, cfg.H_downsized)
    
                        if gim.msgtype_response_to_wakeup_TMI is None and not responded_this_iteration:
                            gim.iface._respond_to_call(msgtype)
                        if _time > 0 and this_rollout_has_seen_t_negative:
                            if _time % 40 == 0:
                                instrumentation__answer_normal_step += time.perf_counter_ns() - pc
                                pc = time.perf_counter_ns()
                            elif _time % 50 == 0:
                                instrumentation__answer_action_step += time.perf_counter_ns() - pc
                                pc = time.perf_counter_ns()
    
                    elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                        # current, target already set at start of iteration (from socket or pending)
                        gim.iface._respond_to_call(msgtype)
                        responded_this_iteration = True
                        next_mt = gim.iface._read_int32()
                        if next_mt == int(MessageType.SC_RUN_STEP_SYNC):
                            self._pending_msgtype = next_mt
                            self._pending_time = gim.iface._read_int32()
                        elif next_mt == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                            self._pending_msgtype = next_mt
                            self._pending_current = gim.iface._read_int32()
                            self._pending_target = gim.iface._read_int32()
                        elif next_mt == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                            self._pending_msgtype = next_mt
                            self._pending_time = -3000
                            self._pending_frame_data = gim.iface.sock.recv(
                                cfg.W_downsized * cfg.H_downsized * 4, socket.MSG_WAITALL
                            )
                        else:
                            self._pending_msgtype = next_mt
                            self._pending_time = -3000
                        simulation_state = gim.iface.get_simulation_state()
                        end_race_stats["cp_time_ms"].append(simulation_state.race_time)
                        if current == target:
                            cp_times_bug_handling_attempts = 0
                            while len(simulation_state.cp_data.cp_times) == 0 and cp_times_bug_handling_attempts < 5:
                                cp_times_bug_handling_attempts += 1
                            if len(simulation_state.cp_data.cp_times) != 0:
                                simulation_state.cp_data.cp_times[-1].time = -1
                                gim.iface.rewind_to_state(simulation_state)
                            else:
                                gim.iface.prevent_simulation_finish()
                            if this_rollout_has_seen_t_negative and not this_rollout_is_finished:
                                if len(rollout_results["current_zone_idx"]) == len(rollout_results["frames"]) + 1:
                                    rollout_results["current_zone_idx"].pop(-1)
                                end_race_stats["race_finished"] = True
                                end_race_stats["race_time"] = simulation_state.race_time
                                rollout_results["race_time"] = simulation_state.race_time
                                end_race_stats["race_time_for_ratio"] = simulation_state.race_time
                                end_race_stats["instrumentation__answer_normal_step"] = (
                                    instrumentation__answer_normal_step / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__answer_action_step"] = (
                                    instrumentation__answer_action_step / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__between_run_steps"] = (
                                    instrumentation__between_run_steps / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__grab_frame"] = instrumentation__grab_frame / simulation_state.race_time * 50
                                end_race_stats["instrumentation__convert_frame"] = (
                                    instrumentation__convert_frame / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__grab_floats"] = instrumentation__grab_floats / simulation_state.race_time * 50
                                end_race_stats["instrumentation__exploration_policy"] = (
                                    instrumentation__exploration_policy / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__request_inputs_and_speed"] = (
                                    instrumentation__request_inputs_and_speed / simulation_state.race_time * 50
                                )
                                end_race_stats["tmi_protection_cutoff"] = False
                                this_rollout_is_finished = True
                                gim.msgtype_response_to_wakeup_TMI = msgtype
                                gim.iface.set_timeout(get_config().timeout_between_runs_ms)
                                if frame_expected:
                                    gim.iface.unrequest_frame()
                                    frame_expected = False
                                rollout_results["current_zone_idx"].append(
                                    len(zone_centers) - cfg.n_zone_centers_extrapolate_after_end_of_map
                                )
                                rollout_results["frames"].append(np.nan)
                                rollout_results["input_w"].append(np.nan)
                                rollout_results["actions"].append(np.nan)
                                rollout_results["action_was_greedy"].append(np.nan)
                                rollout_results["car_gear_and_wheels"].append(np.nan)
                                rollout_results["meters_advanced_along_centerline"].append(
                                    distance_from_start_track_to_prev_zone_transition[
                                        len(zone_centers) - cfg.n_zone_centers_extrapolate_after_end_of_map
                                    ]
                                )
                        if gim.msgtype_response_to_wakeup_TMI is None and not responded_this_iteration:
                            gim.iface._respond_to_call(msgtype)
    
                    elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                        gim.iface._read_int32()
                        gim.iface._read_int32()
                        gim.iface._respond_to_call(msgtype)
    
                    elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                        if self._pending_frame_data is not None:
                            frame = np.frombuffer(
                                self._pending_frame_data, dtype=np.uint8
                            ).reshape((cfg.H_downsized, cfg.W_downsized, 4))
                            self._pending_frame_data = None
                        else:
                            frame = gim.grab_screen()
                        frame_expected = False
                        if (
                            give_up_signal_has_been_sent
                            and this_rollout_has_seen_t_negative
                            and not this_rollout_is_finished
                            and compute_action_asap
                        ):
                            pc6 = time.perf_counter_ns()
                            instrumentation__grab_frame += pc6 - pc5
                            assert gim.latest_tm_engine_speed_requested == 0
                            assert not compute_action_asap_floats
                            frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY), 0)
                            pc7 = time.perf_counter_ns()
                            instrumentation__convert_frame += pc7 - pc6
                            if not gim.game_activated:
                                ensure_window_focused(gim.tm_window_id)
                                gim.game_activated = True
                                if cfg.is_linux:
                                    gim.game_spawning_lock.release()
                            rollout_results["frames"].append(frame)
                            rollout_results["current_zone_idx"].append(current_zone_idx)
                            rollout_results["state_float"].append(floats)
                            rollout_results["car_gear_and_wheels"].append(sim_state_car_gear_and_wheels)
                            rollout_results["meters_advanced_along_centerline"].append(distance_since_track_begin)
                            instrumentation__request_inputs_and_speed += time.perf_counter_ns() - pc7
                            compute_action_asap = False
                            obs = (frame, floats)
                            info = _build_info(
                                current_zone_idx,
                                distance_since_track_begin,
                                floats,
                                sim_state_car_gear_and_wheels,
                                np.nan,
                            )
                            info["end_race_stats"] = end_race_stats
                            info["rollout_results"] = rollout_results
                            self._last_obs = obs
                            self._last_info = info
                            self._episode_started = True
                            self._store_episode_state(
                                zone_centers,
                                zone_transitions,
                                distance_between_zone_transitions,
                                distance_from_start_track_to_prev_zone_transition,
                                normalized_vector_along_track_axis,
                                map_path,
                                rollout_results,
                                end_race_stats,
                                last_progress_improvement_ms,
                                current_zone_idx,
                                last_known_simulation_state,
                                floats,
                                distance_since_track_begin,
                                sim_state_car_gear_and_wheels,
                                instrumentation_answer_normal_step=instrumentation__answer_normal_step,
                                instrumentation_answer_action_step=instrumentation__answer_action_step,
                                instrumentation_between_run_steps=instrumentation__between_run_steps,
                                instrumentation_grab_frame=instrumentation__grab_frame,
                                instrumentation_convert_frame=instrumentation__convert_frame,
                                instrumentation_grab_floats=instrumentation__grab_floats,
                                instrumentation_exploration_policy=instrumentation__exploration_policy,
                                instrumentation_request_inputs_and_speed=instrumentation__request_inputs_and_speed,
                            )
                            # CRITICAL: set input and speed BEFORE responding to the frame (same as legacy rollout).
                            # Otherwise the game stays at speed=0 and the car never moves.
                            gim.request_inputs(cfg.action_forward_idx, rollout_results)
                            gim.request_speed(gim.running_speed)
                            gim.iface._respond_to_call(msgtype)
                            return obs, info
                        gim.iface._respond_to_call(msgtype)
    
                    elif msgtype == int(MessageType.C_SHUTDOWN):
                        gim.iface.close()
                    elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                        if gim.latest_map_path_requested == -1:
                            gim.iface.execute_command("toggle_console")
                        gim.request_speed(1)
                        gim.iface.set_on_step_period(gim.run_steps_per_action * 10)
                        gim.iface.execute_command(f"set countdown_speed {gim.running_speed}")
                        gim.iface.execute_command(f"set autologin {get_config().username}")
                        gim.iface.execute_command("set unfocused_fps_limit false")
                        gim.iface.execute_command("set skip_map_load_screens true")
                        gim.iface.execute_command("set disable_forced_camera true")
                        gim.iface.execute_command("set autorewind false")
                        gim.iface.execute_command("set auto_reload_plugins false")
                        gim.iface.execute_command("set use_valseed false")
                        if gim.iface.is_in_menus() and map_path != gim.latest_map_path_requested:
                            print("Requested map load")
                            gim.request_map(map_path, zone_centers)
                        gim.iface._respond_to_call(msgtype)
                    else:
                        pass

            except (ConnectionError, ValueError, IndexError, OSError) as err:
                err_str = str(err)
                # Game in menus/loading: TMI sends tiny state. Per tminterface2, caller MUST close and reconnect:
                # the game may send more than 4+state_length bytes when in menus/loading, so retrying on the same
                # socket can read garbage as state_length. Same logic as run_until_next_decision_point and legacy.
                game_loading_err = (
                    "state_length" in err_str
                    and "SimStateData.min_size" in err_str
                    and "menus/loading" in err_str
                )
                if game_loading_err and gim.iface is not None:
                    now = time.perf_counter()
                    if now - last_sync_err_log_time[0] >= 30.0:
                        print("Game still loading (state too small), reconnecting after short wait:", err)
                        last_sync_err_log_time[0] = now
                    try:
                        gim.iface.close()
                    except Exception:
                        pass
                    gim.iface = None
                    if _sync_attempt < max_sync_retries:
                        backoff_s = min(2 ** _sync_attempt, 15)
                        time.sleep(backoff_s)
                        continue
                    raise
                sync_err = (
                    "state_length" in err_str
                    or "negative" in err_str.lower()
                    or "buffersize" in err_str.lower()
                    or "out of sync" in err_str.lower()
                    or "out of bounds" in err_str.lower()
                )
                if sync_err or isinstance(err, (ConnectionError, OSError)):
                    if gim.iface is not None:
                        try:
                            gim.iface.close()
                        except Exception:
                            pass
                        gim.iface = None
                    if _sync_attempt < max_sync_retries:
                        now = time.perf_counter()
                        if now - last_sync_err_log_time[0] >= 30.0:
                            print("TMI out of sync, reconnecting (retry):", err)
                            last_sync_err_log_time[0] = now
                        # Backoff before reconnect so game can leave menus/loading (state_length too small)
                        backoff_s = min(2 ** _sync_attempt, 15)
                        time.sleep(backoff_s)
                        continue
                raise
            except socket.timeout as err:
                print("Cutoff rollout due to TMI timeout", err)
                gim.iface.close()
                end_race_stats["tmi_protection_cutoff"] = True
                gim.last_rollout_crashed = True
                ensure_not_minimized(gim.tm_window_id)
                if self._last_obs is not None and self._last_info is not None:
                    info = {**self._last_info, "end_race_stats": end_race_stats}
                    return self._last_obs, 0.0, False, True, info
                raise RuntimeError("TMI timeout before first observation") from err

        raise RuntimeError("start_episode loop exited without returning first obs")

    def _store_episode_state(
        self,
        zone_centers: npt.NDArray,
        zone_transitions: npt.NDArray,
        distance_between_zone_transitions: npt.NDArray,
        distance_from_start_track_to_prev_zone_transition: npt.NDArray,
        normalized_vector_along_track_axis: npt.NDArray,
        map_path: str,
        rollout_results: Dict[str, Any],
        end_race_stats: Dict[str, Any],
        last_progress_improvement_ms: float,
        current_zone_idx: int,
        last_known_simulation_state: Any,
        floats: npt.NDArray,
        distance_since_track_begin: float,
        sim_state_car_gear_and_wheels: npt.NDArray,
        *,
        instrumentation_answer_normal_step: int = 0,
        instrumentation_answer_action_step: int = 0,
        instrumentation_between_run_steps: int = 0,
        instrumentation_grab_frame: int = 0,
        instrumentation_convert_frame: int = 0,
        instrumentation_grab_floats: int = 0,
        instrumentation_exploration_policy: int = 0,
        instrumentation_request_inputs_and_speed: int = 0,
    ) -> None:
        """Store state for run_until_next_decision_point (same loop needs these)."""
        self._zone_centers = zone_centers
        self._zone_transitions = zone_transitions
        self._distance_between_zone_transitions = distance_between_zone_transitions
        self._distance_from_start_track_to_prev_zone_transition = distance_from_start_track_to_prev_zone_transition
        self._normalized_vector_along_track_axis = normalized_vector_along_track_axis
        self._map_path = map_path
        self._rollout_results = rollout_results
        self._end_race_stats = end_race_stats
        self._last_progress_improvement_ms = last_progress_improvement_ms
        self._current_zone_idx = current_zone_idx
        self._last_known_simulation_state = last_known_simulation_state
        self._floats = floats
        self._distance_since_track_begin = distance_since_track_begin
        self._sim_state_car_gear_and_wheels = sim_state_car_gear_and_wheels
        self._instrumentation_ns = {
            "answer_normal_step": instrumentation_answer_normal_step,
            "answer_action_step": instrumentation_answer_action_step,
            "between_run_steps": instrumentation_between_run_steps,
            "grab_frame": instrumentation_grab_frame,
            "convert_frame": instrumentation_convert_frame,
            "grab_floats": instrumentation_grab_floats,
            "exploration_policy": instrumentation_exploration_policy,
            "request_inputs_and_speed": instrumentation_request_inputs_and_speed,
        }

    def run_until_next_decision_point(
        self,
    ) -> Tuple[Tuple[npt.NDArray, npt.NDArray], float, bool, bool, Dict[str, Any]]:
        """
        Send the pending action, then block until the next decision point (next frame).
        Returns (obs, reward, terminated, truncated, info).
        """
        if not self._episode_started or self._pending_action is None:
            raise RuntimeError("call start_episode then send_action before run_until_next_decision_point")
        gim = self.gim
        cfg = get_config()
        action_idx = self._pending_action
        self._pending_action = None
        gim.request_inputs(action_idx, self._rollout_results)
        gim.request_speed(gim.running_speed)
        rollout_results = self._rollout_results
        end_race_stats = self._end_race_stats
        zone_centers = self._zone_centers
        zone_transitions = self._zone_transitions
        distance_between_zone_transitions = self._distance_between_zone_transitions
        distance_from_start_track_to_prev_zone_transition = self._distance_from_start_track_to_prev_zone_transition
        normalized_vector_along_track_axis = self._normalized_vector_along_track_axis
        map_path = self._map_path
        last_progress_improvement_ms = self._last_progress_improvement_ms
        current_zone_idx = self._current_zone_idx
        last_known_simulation_state = self._last_known_simulation_state
        ins = self._instrumentation_ns
        instrumentation__answer_normal_step = ins["answer_normal_step"]
        instrumentation__answer_action_step = ins["answer_action_step"]
        instrumentation__between_run_steps = ins["between_run_steps"]
        instrumentation__grab_frame = ins["grab_frame"]
        instrumentation__convert_frame = ins["convert_frame"]
        instrumentation__grab_floats = ins["grab_floats"]
        instrumentation__exploration_policy = ins["exploration_policy"]
        instrumentation__request_inputs_and_speed = ins["request_inputs_and_speed"]
        give_up_signal_has_been_sent = True
        this_rollout_has_seen_t_negative = True
        this_rollout_is_finished = False
        compute_action_asap = False
        compute_action_asap_floats = False
        frame_expected = True
        map_change_requested_time = -1
        pc = time.perf_counter_ns()
        pc5 = pc
        floats = self._floats
        distance_since_track_begin = self._distance_since_track_begin
        sim_state_car_gear_and_wheels = self._sim_state_car_gear_and_wheels
        from trackmania_rl.tmi_interaction.tminterface2 import MessageType

        rollout_results["actions"].append(action_idx)
        rollout_results["input_w"].append(get_config().inputs[action_idx]["accelerate"])
        rollout_results["action_was_greedy"].append(np.nan)
        rollout_results["q_values"].append(np.nan)

        responded_this_iteration = False
        step_retry = True
        while step_retry:
            step_retry = False
            try:
                while not this_rollout_is_finished:
                    responded_this_iteration = False
                    # Consume buffered message from previous iteration (after respond→buffer→get_simulation_state).
                    if self._pending_msgtype is not None:
                        msgtype = self._pending_msgtype
                        if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                            _time = self._pending_time
                        elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                            current = self._pending_current
                            target = self._pending_target
                            _time = -3000
                        else:
                            _time = -3000
                        self._pending_msgtype = None
                        self._pending_time = None
                        self._pending_current = None
                        self._pending_target = None
                    else:
                        msgtype = gim.iface._read_int32()

                        if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                            _time = gim.iface._read_int32()
                        elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                            current = gim.iface._read_int32()
                            target = gim.iface._read_int32()
                            _time = -3000
                        else:
                            _time = -3000

                    if compute_action_asap_floats:
                        pc2 = time.perf_counter_ns()
                        sim_state_race_time = last_known_simulation_state.race_time
                        sim_state_dyna_current = last_known_simulation_state.dyna.current_state
                        sim_state_mobil = last_known_simulation_state.scene_mobil
                        sim_state_mobil_engine = sim_state_mobil.engine
                        simulation_wheels = last_known_simulation_state.simulation_wheels
                        wheel_state = [simulation_wheels[i].real_time_state for i in range(4)]
                        sim_state_position = np.array(sim_state_dyna_current.position, dtype=np.float32)
                        sim_state_orientation = sim_state_dyna_current.rotation.to_numpy().T
                        sim_state_velocity = np.array(sim_state_dyna_current.linear_speed, dtype=np.float32)
                        sim_state_angular_speed = np.array(sim_state_dyna_current.angular_speed, dtype=np.float32)
                        gearbox_state = sim_state_mobil.gearbox_state
                        counter_gearbox_state = 0
                        if gearbox_state != 0 and len(rollout_results["car_gear_and_wheels"]) > 0:
                            counter_gearbox_state = 1 + rollout_results["car_gear_and_wheels"][-1][15]
                        sim_state_car_gear_and_wheels = np.array(
                            [
                                *(ws.is_sliding for ws in wheel_state),
                                *(ws.has_ground_contact for ws in wheel_state),
                                *(ws.damper_absorb for ws in wheel_state),
                                gearbox_state,
                                sim_state_mobil_engine.gear,
                                sim_state_mobil_engine.actual_rpm,
                                counter_gearbox_state,
                                *(
                                    i == contact_materials.physics_behavior_fromint[ws.contact_material_id & 0xFFFF]
                                    for ws in wheel_state
                                    for i in range(cfg.n_contact_material_physics_behavior_types)
                                ),
                            ],
                            dtype=np.float32,
                        )
                        if sim_state_position[1] > get_config().deck_height:
                            current_zone_idx = update_current_zone_idx(
                                current_zone_idx,
                                zone_centers,
                                sim_state_position,
                                cfg.max_allowable_distance_to_virtual_checkpoint,
                                gim.next_real_checkpoint_positions,
                                gim.max_allowable_distance_to_real_checkpoint,
                                cfg.n_zone_centers_extrapolate_after_end_of_map,
                            )
                        if current_zone_idx > rollout_results["furthest_zone_idx"]:
                            last_progress_improvement_ms = sim_state_race_time
                            rollout_results["furthest_zone_idx"] = current_zone_idx
                        rollout_results["current_zone_idx"].append(current_zone_idx)
                        meters_in_current_zone = np.clip(
                            (sim_state_position - zone_transitions[current_zone_idx - 1]).dot(
                                normalized_vector_along_track_axis[current_zone_idx - 1]
                            ),
                            0,
                            distance_between_zone_transitions[current_zone_idx - 1],
                        )
                        distance_since_track_begin = (
                            distance_from_start_track_to_prev_zone_transition[current_zone_idx - 1] + meters_in_current_zone
                        )
                        state_zone_center_coordinates_in_car_reference_system = sim_state_orientation.dot(
                            (
                                zone_centers[
                                    current_zone_idx : current_zone_idx
                                    + cfg.one_every_n_zone_centers_in_inputs * cfg.n_zone_centers_in_inputs : cfg.one_every_n_zone_centers_in_inputs,
                                    :,
                                ]
                                - sim_state_position
                            ).T
                        ).T
                        state_y_map_vector_in_car_reference_system = sim_state_orientation.dot(np.array([0, 1, 0]))
                        state_car_velocity_in_car_reference_system = sim_state_orientation.dot(sim_state_velocity)
                        state_car_angular_velocity_in_car_reference_system = sim_state_orientation.dot(sim_state_angular_speed)
                        previous_actions = [
                            get_config().inputs[rollout_results["actions"][k] if k >= 0 else get_config().action_forward_idx]
                            for k in range(
                                len(rollout_results["actions"]) - cfg.n_prev_actions_in_inputs,
                                len(rollout_results["actions"]),
                            )
                        ]
                        floats = np.hstack(
                            (
                                0,
                                np.array(
                                    [
                                        pa[input_str]
                                        for pa in previous_actions
                                        for input_str in ["accelerate", "brake", "left", "right"]
                                    ]
                                ),
                                sim_state_car_gear_and_wheels.ravel(),
                                state_car_angular_velocity_in_car_reference_system.ravel(),
                                state_car_velocity_in_car_reference_system.ravel(),
                                state_y_map_vector_in_car_reference_system.ravel(),
                                state_zone_center_coordinates_in_car_reference_system.ravel(),
                                min(
                                    cfg.margin_to_announce_finish_meters,
                                    distance_from_start_track_to_prev_zone_transition[
                                        len(zone_centers) - cfg.n_zone_centers_extrapolate_after_end_of_map
                                    ]
                                    - distance_since_track_begin,
                                ),
                                sim_state_mobil.is_freewheeling,
                            )
                        ).astype(np.float32)
                        pc5 = time.perf_counter_ns()
                        instrumentation__grab_floats += pc5 - pc2
                        compute_action_asap_floats = False

                    if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                        if _time > 0 and _time % 50 == 0:
                            instrumentation__between_run_steps += time.perf_counter_ns() - pc
                        pc = time.perf_counter_ns()
                        if (
                            (_time > gim.max_overall_duration_ms or _time > last_progress_improvement_ms + gim.max_minirace_duration_ms)
                            and not this_rollout_is_finished
                        ):
                            gim.iface._respond_to_call(msgtype)
                            next_mt = gim.iface._read_int32()
                            if next_mt == int(MessageType.SC_RUN_STEP_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_time = gim.iface._read_int32()
                            elif next_mt == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_frame_data = gim.iface.sock.recv(
                                    cfg.W_downsized * cfg.H_downsized * 4, socket.MSG_WAITALL
                                )
                            else:
                                self._pending_msgtype = next_mt
                                self._pending_time = -3000
                            simulation_state = gim.iface.get_simulation_state()
                            race_time = max([simulation_state.race_time, 1e-12])
                            end_race_stats["race_finished"] = False
                            end_race_stats["race_time"] = get_config().cutoff_rollout_if_race_not_finished_within_duration_ms
                            end_race_stats["race_time_for_ratio"] = race_time
                            end_race_stats["instrumentation__answer_normal_step"] = instrumentation__answer_normal_step / race_time * 50
                            end_race_stats["instrumentation__answer_action_step"] = instrumentation__answer_action_step / race_time * 50
                            end_race_stats["instrumentation__between_run_steps"] = instrumentation__between_run_steps / race_time * 50
                            end_race_stats["instrumentation__grab_frame"] = instrumentation__grab_frame / race_time * 50
                            end_race_stats["instrumentation__convert_frame"] = instrumentation__convert_frame / race_time * 50
                            end_race_stats["instrumentation__grab_floats"] = instrumentation__grab_floats / race_time * 50
                            end_race_stats["instrumentation__exploration_policy"] = instrumentation__exploration_policy / race_time * 50
                            end_race_stats["instrumentation__request_inputs_and_speed"] = (
                                instrumentation__request_inputs_and_speed / race_time * 50
                            )
                            end_race_stats["tmi_protection_cutoff"] = False
                            self._instrumentation_ns = {
                                "answer_normal_step": instrumentation__answer_normal_step,
                                "answer_action_step": instrumentation__answer_action_step,
                                "between_run_steps": instrumentation__between_run_steps,
                                "grab_frame": instrumentation__grab_frame,
                                "convert_frame": instrumentation__convert_frame,
                                "grab_floats": instrumentation__grab_floats,
                                "exploration_policy": instrumentation__exploration_policy,
                                "request_inputs_and_speed": instrumentation__request_inputs_and_speed,
                            }
                            gim.iface.rewind_to_current_state()
                            gim.msgtype_response_to_wakeup_TMI = msgtype
                            gim.iface.set_timeout(get_config().timeout_between_runs_ms)
                            if frame_expected:
                                gim.iface.unrequest_frame()
                                frame_expected = False
                            this_rollout_is_finished = True
                            obs = self._last_obs
                            info = {**self._last_info, "end_race_stats": end_race_stats}
                            return obs, 0.0, False, True, info

                        if not this_rollout_is_finished and _time >= 0 and _time % (10 * gim.run_steps_per_action) == 0:
                            gim.iface._respond_to_call(msgtype)
                            responded_this_iteration = True
                            next_mt = gim.iface._read_int32()
                            if next_mt == int(MessageType.SC_RUN_STEP_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_time = gim.iface._read_int32()
                            elif next_mt == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                                self._pending_msgtype = next_mt
                                self._pending_frame_data = gim.iface.sock.recv(
                                    cfg.W_downsized * cfg.H_downsized * 4, socket.MSG_WAITALL
                                )
                            else:
                                self._pending_msgtype = next_mt
                                self._pending_time = -3000
                            last_known_simulation_state = gim.iface.get_simulation_state()
                            gim.iface.rewind_to_current_state()
                            gim.request_speed(0)
                            compute_action_asap = True
                            if compute_action_asap:
                                compute_action_asap_floats = True
                                frame_expected = True
                                gim.iface.request_frame(cfg.W_downsized, cfg.H_downsized)

                        if gim.msgtype_response_to_wakeup_TMI is None and not responded_this_iteration:
                            gim.iface._respond_to_call(msgtype)
                        if _time > 0:
                            if _time % 40 == 0:
                                instrumentation__answer_normal_step += time.perf_counter_ns() - pc
                                pc = time.perf_counter_ns()
                            elif _time % 50 == 0:
                                instrumentation__answer_action_step += time.perf_counter_ns() - pc
                                pc = time.perf_counter_ns()

                    elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                        gim.iface._respond_to_call(msgtype)
                        responded_this_iteration = True
                        next_mt = gim.iface._read_int32()
                        if next_mt == int(MessageType.SC_RUN_STEP_SYNC):
                            self._pending_msgtype = next_mt
                            self._pending_time = gim.iface._read_int32()
                        elif next_mt == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                            self._pending_msgtype = next_mt
                            self._pending_current = gim.iface._read_int32()
                            self._pending_target = gim.iface._read_int32()
                        elif next_mt == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                            self._pending_msgtype = next_mt
                            self._pending_time = -3000
                            self._pending_frame_data = gim.iface.sock.recv(
                                cfg.W_downsized * cfg.H_downsized * 4, socket.MSG_WAITALL
                            )
                        else:
                            self._pending_msgtype = next_mt
                            self._pending_time = -3000
                        simulation_state = gim.iface.get_simulation_state()
                        end_race_stats["cp_time_ms"].append(simulation_state.race_time)
                        if current == target:
                            cp_times_bug_handling_attempts = 0
                            while len(simulation_state.cp_data.cp_times) == 0 and cp_times_bug_handling_attempts < 5:
                                cp_times_bug_handling_attempts += 1
                            if len(simulation_state.cp_data.cp_times) != 0:
                                simulation_state.cp_data.cp_times[-1].time = -1
                                gim.iface.rewind_to_state(simulation_state)
                            else:
                                gim.iface.prevent_simulation_finish()
                            if not this_rollout_is_finished:
                                if len(rollout_results["current_zone_idx"]) == len(rollout_results["frames"]) + 1:
                                    rollout_results["current_zone_idx"].pop(-1)
                                    end_race_stats["race_finished"] = True
                                end_race_stats["race_time"] = simulation_state.race_time
                                rollout_results["race_time"] = simulation_state.race_time
                                end_race_stats["race_time_for_ratio"] = simulation_state.race_time
                                end_race_stats["instrumentation__answer_normal_step"] = (
                                    instrumentation__answer_normal_step / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__answer_action_step"] = (
                                    instrumentation__answer_action_step / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__between_run_steps"] = (
                                    instrumentation__between_run_steps / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__grab_frame"] = instrumentation__grab_frame / simulation_state.race_time * 50
                                end_race_stats["instrumentation__convert_frame"] = (
                                    instrumentation__convert_frame / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__grab_floats"] = instrumentation__grab_floats / simulation_state.race_time * 50
                                end_race_stats["instrumentation__exploration_policy"] = (
                                    instrumentation__exploration_policy / simulation_state.race_time * 50
                                )
                                end_race_stats["instrumentation__request_inputs_and_speed"] = (
                                    instrumentation__request_inputs_and_speed / simulation_state.race_time * 50
                                )
                                end_race_stats["tmi_protection_cutoff"] = False
                                this_rollout_is_finished = True
                                gim.msgtype_response_to_wakeup_TMI = msgtype
                                gim.iface.set_timeout(get_config().timeout_between_runs_ms)
                                if frame_expected:
                                    gim.iface.unrequest_frame()
                                    frame_expected = False
                                rollout_results["current_zone_idx"].append(
                                    len(zone_centers) - cfg.n_zone_centers_extrapolate_after_end_of_map
                                )
                                rollout_results["frames"].append(np.nan)
                                rollout_results["input_w"].append(np.nan)
                                rollout_results["actions"].append(np.nan)
                                rollout_results["action_was_greedy"].append(np.nan)
                                rollout_results["car_gear_and_wheels"].append(np.nan)
                                rollout_results["meters_advanced_along_centerline"].append(
                                    distance_from_start_track_to_prev_zone_transition[
                                        len(zone_centers) - cfg.n_zone_centers_extrapolate_after_end_of_map
                                    ]
                                )
                                self._instrumentation_ns = {
                                    "answer_normal_step": instrumentation__answer_normal_step,
                                    "answer_action_step": instrumentation__answer_action_step,
                                    "between_run_steps": instrumentation__between_run_steps,
                                    "grab_frame": instrumentation__grab_frame,
                                    "convert_frame": instrumentation__convert_frame,
                                    "grab_floats": instrumentation__grab_floats,
                                    "exploration_policy": instrumentation__exploration_policy,
                                    "request_inputs_and_speed": instrumentation__request_inputs_and_speed,
                                }
                                obs = (self._last_obs[0].copy(), floats.copy())
                                info = _build_info(
                                    len(zone_centers) - cfg.n_zone_centers_extrapolate_after_end_of_map,
                                    distance_from_start_track_to_prev_zone_transition[
                                        len(zone_centers) - cfg.n_zone_centers_extrapolate_after_end_of_map
                                    ],
                                    floats,
                                    sim_state_car_gear_and_wheels,
                                    get_config().inputs[action_idx]["accelerate"],
                                )
                                info["end_race_stats"] = end_race_stats
                                info["race_finished"] = True
                                info["race_time"] = simulation_state.race_time
                                self._last_obs = obs
                                self._last_info = info
                                return obs, 0.0, True, False, info
                        if gim.msgtype_response_to_wakeup_TMI is None and not responded_this_iteration:
                            gim.iface._respond_to_call(msgtype)

                    elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                        gim.iface._read_int32()
                        gim.iface._read_int32()
                        gim.iface._respond_to_call(msgtype)

                    elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                        if self._pending_frame_data is not None:
                            frame = np.frombuffer(
                                self._pending_frame_data, dtype=np.uint8
                            ).reshape((cfg.H_downsized, cfg.W_downsized, 4))
                            self._pending_frame_data = None
                        else:
                            frame = gim.grab_screen()
                        frame_expected = False
                        if (
                            give_up_signal_has_been_sent
                            and this_rollout_has_seen_t_negative
                            and not this_rollout_is_finished
                            and compute_action_asap
                        ):
                            pc6 = time.perf_counter_ns()
                            instrumentation__grab_frame += pc6 - pc5
                            assert gim.latest_tm_engine_speed_requested == 0
                            assert not compute_action_asap_floats
                            frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY), 0)
                            pc7 = time.perf_counter_ns()
                            instrumentation__convert_frame += pc7 - pc6
                            rollout_results["frames"].append(frame)
                            rollout_results["meters_advanced_along_centerline"].append(distance_since_track_begin)
                            rollout_results["car_gear_and_wheels"].append(sim_state_car_gear_and_wheels)
                            rollout_results["state_float"].append(floats)
                            instrumentation__request_inputs_and_speed += time.perf_counter_ns() - pc7
                            compute_action_asap = False
                            obs = (frame, floats)
                            info = _build_info(
                                current_zone_idx,
                                distance_since_track_begin,
                                floats,
                                sim_state_car_gear_and_wheels,
                                get_config().inputs[action_idx]["accelerate"],
                            )
                            info["end_race_stats"] = end_race_stats
                            self._last_obs = obs
                            self._last_info = info
                            self._last_progress_improvement_ms = last_progress_improvement_ms
                            self._current_zone_idx = current_zone_idx
                            self._last_known_simulation_state = last_known_simulation_state
                            self._floats = floats
                            self._distance_since_track_begin = distance_since_track_begin
                            self._sim_state_car_gear_and_wheels = sim_state_car_gear_and_wheels
                            self._end_race_stats = end_race_stats
                            self._instrumentation_ns = {
                                "answer_normal_step": instrumentation__answer_normal_step,
                                "answer_action_step": instrumentation__answer_action_step,
                                "between_run_steps": instrumentation__between_run_steps,
                                "grab_frame": instrumentation__grab_frame,
                                "convert_frame": instrumentation__convert_frame,
                                "grab_floats": instrumentation__grab_floats,
                                "exploration_policy": instrumentation__exploration_policy,
                                "request_inputs_and_speed": instrumentation__request_inputs_and_speed,
                            }
                            # Same as legacy: send input and speed BEFORE responding to the frame, so the game runs with them.
                            gim.request_inputs(action_idx, rollout_results)
                            gim.request_speed(gim.running_speed)
                            gim.iface._respond_to_call(msgtype)
                            return obs, 0.0, False, False, info
                        gim.iface._respond_to_call(msgtype)

                    elif msgtype == int(MessageType.C_SHUTDOWN):
                        gim.iface.close()
                    elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                        gim.iface._respond_to_call(msgtype)
                    else:
                        pass

            except (ConnectionError, ValueError, IndexError, OSError) as err:
                err_str = str(err)
                game_loading_err = (
                    "state_length" in err_str
                    and "SimStateData.min_size" in err_str
                    and "menus/loading" in err_str
                )
                if game_loading_err and gim.iface is not None:
                    # When state_length=1 the game may send more than 5 bytes; we only drain 1.
                    # Retrying get_simulation_state() on the same socket then reads leftover bytes
                    # as state_length (e.g. 16777249). So we must close and reconnect, and wait
                    # so this window has time to finish loading before reconnect.
                    print("Game still loading (state too small), reconnecting after short wait:", err)
                    try:
                        gim.iface.close()
                    except Exception:
                        pass
                    gim.iface = None
                    gim.last_rollout_crashed = True
                    time.sleep(5.0)
                    raise
                sync_err = (
                    "out of sync" in err_str.lower()
                    or "state_length" in err_str
                    or "out of bounds" in err_str.lower()
                    or "buffersize" in err_str.lower()
                )
                if sync_err or isinstance(err, (ConnectionError, OSError)):
                    print("TMI error in step, reconnecting:", err)
                    gim.last_rollout_crashed = True
                    if gim.iface is not None:
                        try:
                            gim.iface.close()
                        except Exception:
                            pass
                        gim.iface = None
                raise
            except socket.timeout as err:
                print("Cutoff rollout due to TMI timeout", err)
                gim.iface.close()
                end_race_stats["tmi_protection_cutoff"] = True
                gim.last_rollout_crashed = True
                ensure_not_minimized(gim.tm_window_id)
                if self._last_obs is not None and self._last_info is not None:
                    info = {**self._last_info, "end_race_stats": end_race_stats}
                    return self._last_obs, 0.0, False, True, info
                raise RuntimeError("TMI timeout in run_until_next_decision_point") from err

        raise RuntimeError("run_until_next_decision_point loop exited without return")
