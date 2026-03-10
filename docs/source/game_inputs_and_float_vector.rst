.. _game_inputs_and_float_vector:

Game inputs and float observation vector
=========================================

This document describes what data we take from the game (TMInterface / SimStateData), how it maps to our float observation vector, and **what the game provides but we do not use**. It is the single reference for "are we using everything?" and for extending the observation space.

Sources
-------

- **TMInterface 2** (socket API): we use ``get_simulation_state()`` → returns ``SimStateData`` (from package ``tminterface``).
- **Float vector** is built in ``trackmania_rl.float_inputs``: ``build_float_vector()`` and ``state_dict_from_sim_state()``.
- **RL rollout**: ``game_env_backend.py`` / ``game_instance_manager.py`` call ``get_simulation_state()``, then ``state_dict_from_sim_state()``, then ``build_float_vector()``.

Float vector layout (dimension and indices)
-------------------------------------------

**Formula** (see ``config_loader.py``):

.. code-block:: text

   float_input_dim = 27 + 3 * n_zone_centers_in_inputs
                    + 4 * n_prev_actions_in_inputs
                    + 4 * n_contact_material_physics_behavior_types
                    + 1

With defaults (n_zone=40, n_prev=5, n_contact=4): **184**.

**Order** (same as in ``float_inputs.build_float_vector()``):

.. list-table::
   :header-rows: 1
   :widths: 8 12 50 15

   * - Index
     - Count
     - Description
     - Source
   * - 0
     - 1
     - Temporal: time left in mini-race (overwritten in buffer collate)
     - Config / rollout
   * - 1–20
     - 20
     - Previous actions: [accel, brake, left, right] × 5 steps
     - Our action buffer (not from SimStateData)
   * - 21–52
     - 32
     - **gear_and_wheels**: 4×is_sliding, 4×has_ground_contact, 4×damper_absorb, gearbox_state, gear, actual_rpm, counter_gearbox_state, 4×4 contact_material one-hot per wheel
     - sim_state.scene_mobil, sim_state.simulation_wheels
   * - 53–55
     - 3
     - Angular velocity (car frame)
     - dyna.current_state.angular_speed × orientation
   * - 56–58
     - 3
     - Velocity (car frame); 56=lateral, 58=forward
     - dyna.current_state.linear_speed × orientation
   * - 59–61
     - 3
     - y_map (up in world, in car frame)
     - orientation @ [0,1,0]
   * - 62–181
     - 120
     - Zone centers in car frame (40 × 3)
     - Map VCP + position/orientation (not raw from game)
   * - 182
     - 1
     - Margin to finish (meters)
     - Zone math (not raw from game)
   * - 183
     - 1
     - is_freewheeling
     - scene_mobil.is_freewheeling

**Indices used elsewhere:** ``buffer_management.py`` and reward shaping use e.g. 25:29 (ground contact), 56:59 (velocity), 62:65 and 65:68 (first two zone centers). If you add/remove/reorder features, update those indices and ``float_input_dim``.

What we use from SimStateData
-----------------------------

- **dyna.current_state**: position, rotation (orientation), linear_speed, angular_speed.
- **scene_mobil**: engine.gear, engine.actual_rpm, gearbox_state, is_freewheeling.
- **simulation_wheels** (×4): real_time_state.is_sliding, has_ground_contact, damper_absorb, contact_material_id (mapped via ``contact_materials.py`` to physics behavior categories).
- **sim_state.race_time**: used for progress and finish detection (not in the float vector).

We do **not** read previous_state / temp_state from dyna; we only use current_state.

What the game provides but we do NOT use
----------------------------------------

Below is what is available on SimStateData (and related structs) but is **not** fed into the float vector or used in rollout logic. Adding any of these would require extending ``FloatStateDict``, ``state_dict_from_sim_state()``, ``build_float_vector()``, config (e.g. ``float_input_dim``, normalization), and any code that indexes by position (e.g. buffer_management reward logic).

**dyna (HmsDynaStruct)**

- previous_state, temp_state (full previous/temp physics state).
- add_linear_speed, force, torque, inverse_inertia_tensor.
- quat (we use rotation matrix only).
- unknown, not_tweaked_linear_speed, owner.

**scene_mobil (SceneVehicleCar)**

- input_gas, input_brake, input_steer (current inputs; we use our own prev_actions from the action buffer).
- max_linear_speed, quality, block_flags.
- prev_sync_vehicle_state, sync_vehicle_state, async_vehicle_state, prev_async_vehicle_state (speed_forward, speed_sideward, rpm, input_steer, input_gas, input_brake, is_turbo, gearbox_state — game-internal view; we use dyna velocity and engine instead).
- has_any_lateral_contact, last_has_any_lateral_contact_time.
- water_forces_applied, turning_rate, turbo_boost_factor, last_turbo_type_change_time, last_turbo_time, turbo_type, roulette_value.
- is_sliding (car-level; we use per-wheel is_sliding), wheel_contact_absorb_counter, burnout_state.
- current_local_speed, total_central_force_added, is_rubber_ball, saved_state.

**simulation_wheels (per wheel)**

- Everything except real_time_state: steerable, surface_handler (position, rotation, etc.), offset_from_vehicle, prev_sync_wheel_state, sync_wheel_state, async_wheel_state, contact_relative_local_distance, etc.
- real_time_state: we use is_sliding, has_ground_contact, damper_absorb, contact_material_id only; we do not use field_4, field_8, field_12, field_48, field_84, field_108, relative_rotz_axis, nb_ground_contacts, field_144, rest.

**SimStateData top-level**

- version, context_mode, flags, timers.
- cp_data (checkpoint data): we use our own zone/VCP math and do not feed raw checkpoint state into the float vector.

**TMInterface API**

- ``get_inputs()``: returns replay input script as a string (used in validation/capture scripts), not the current simulation state. We do not use it for the RL observation.

Outputs: commands we send to the car (binary vs analog)
------------------------------------------------------

**What we use today: binary only.**

We send commands via ``TMInterface.set_input_state(left, right, accelerate, brake)`` in ``tminterface2.py``. All four arguments are **booleans** (packed as ``uint8``). So in RL we only send one of the 12 discrete actions from ``config.inputs.actions`` (e.g. left+accel, right+brake, coast) — **no analog steering or throttle**.

**What TMInterface supports in script format:**

- **steer** — **analog**. Integer in ``[-65536, 65536]`` (negative = left, positive = right, 0 = no steer). Extended range with ``extended_steer``. Used in replay/script files (e.g. ``"8.43 steer 13292"``).
- **gas** — analog in script, but **TMNF does not use the value**: "TMNF/TMUF do not support analog acceleration." So gas is effectively binary in TMNF.
- **press** / **rel** — binary (up, down, left, right).

When we convert replays to TMInterface scripts we **convert analog steer to binary** left/right (deadzone); we do not preserve analog in our RL action space.

**Socket API we have:** Our client only implements ``C_SET_INPUT_STATE`` with 4 bytes (left, right, accelerate, brake). There is **no** ``set_steer_value(int)`` in ``tminterface2.py``. So for real-time control we are **limited to binary**. To use analog steer we would need a new socket message type (if the game plugin supports it) or per-step ``execute_command("steer ...")`` (brittle).

Summary
-------

- **Dimension**: 184 with default config; layout is fixed in ``float_inputs.build_float_vector()`` and must match ``float_input_dim`` and state normalization.
- **Used from game**: current dyna state (position, orientation, linear/angular speed), mobil engine/gearbox/freewheeling, and per-wheel sliding/contact/damper/contact_material.
- **Not used**: previous/temp dyna state, forces/torques, car-level lateral contact/turbo/burnout, game’s internal vehicle states (sync/async), wheel geometry/sync state, replay input string, and raw checkpoint data. Adding any of these would require extending the float vector and all dependent code (config, normalization, reward indices).
