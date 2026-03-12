.. _game_inputs_and_float_vector:

Game inputs and float observation vector
=========================================

This document describes what data we take from the game (TMInterface / SimStateData), how it maps to our **extended** float observation vector, and how actions are represented as a **multi-label** vector. It is the single reference for "are we using everything?" and for extending the observation/action space.

Sources
-------

- **TMInterface 2** (socket API): we use ``get_simulation_state()`` → returns ``SimStateData`` (from package ``tminterface``).
- **Float vector** is built in ``trackmania_rl.float_inputs``: ``build_float_vector()``, ``state_dict_from_sim_state()`` (RL), and ``state_dict_from_meta()`` (BC from manifest).
- **RL rollout**: ``game_env_backend.py`` / ``game_instance_manager.py`` call ``get_simulation_state()``, then ``state_dict_from_sim_state()``, then ``build_float_vector()``.

Action space (parameterized, multi-label or classification)
------------------------------------------------------------

The action space is **parameterized** by config: either the optional **``action_space``** section (per-input ``enabled`` and ``discretization``) or, when omitted, **``n_steer_parts``** (all four inputs, left/right discretization = ``n_steer_parts``).

- **``action_space.inputs``**: For each of ``accelerate``, ``brake``, ``left``, ``right``: ``enabled`` (bool), ``discretization`` (int). ``n_action_dims`` = sum of ``discretization`` over enabled inputs. Default per input: enabled true, discretization 1.
- **``n_steer_parts``**: Used when ``action_space`` is not set; also exposed as the effective steering resolution for backward compatibility.
- **``n_action_dims``**: Computed from action space (single source of truth: ``ActionSpace.from_config(config).n_action_dims``).
- **Head mode** (``neural_network.action_head_mode``): **``multilabel``** = one logit per dimension (or BDQ-style branching), greedy = (A > 0) per dim; **``classification``** = one head with N discrete classes (e.g. 12), dueling, greedy = argmax → index → ``action_index_to_vector``.
- **Conversion**: ``trackmania_rl.action_vector.ActionSpace.from_config(config)`` provides ``to_game_input()``, ``from_game_input()``, and (for classification) ``action_index_to_vector`` / ``action_vector_to_index``. Game commands are binary (left/right/accelerate/brake) via ``TMInterface.set_input_state()``.

Configurable segments (state_observation.include)
-------------------------------------------------

Which segments are included in the float vector is controlled by **``state_observation.include``** in the RL config (see ``config_default.yaml``). Each segment name maps to ``true`` (include) or ``false`` (exclude). Omitted keys default to **included**. This affects rollout, replay buffer, BC cache, and IQN input everywhere.

Segment names: ``temporal``, ``prev_actions``, ``gear_and_wheels``, ``angular_velocity``, ``velocity``, ``y_map``, ``zone_centers_in_car_frame``, ``margin``, ``is_freewheeling``, ``turning_rate``, ``mobil_is_sliding``; then the 21 car_track_extra sub-names (e.g. ``add_linear_speed``, ``force``, ``distance_since_track_begin``, ``current_zone_idx``). See ``trackmania_rl.float_inputs.ALL_STATE_OBSERVATION_NAMES``.

**``float_input_dim``** is the sum of dimensions of **included** segments only. Indices into the vector are computed by ``float_inputs.get_segment_start_indices(config)``; do not assume a fixed index (e.g. temporal at 0) if segments can be disabled.

Float vector layout (dimension and indices when all included)
-------------------------------------------------------------

**Formula** when all segments are included (see ``config_loader.py`` and ``float_inputs.compute_float_input_dim()``):

.. code-block:: text

   n_action_dims = 2 + 2 * n_steer_parts
   float_input_dim = 1
                    + n_prev_actions_in_inputs * n_action_dims
                    + (4 + 4 + 4 + 1 + 1 + 1 + 1 + 4 * n_contact_material_physics_behavior_types)
                    + 3 + 3 + 3
                    + n_zone_centers_in_inputs * 3
                    + 1 + 1
                    + 1 + 1
                    + 29

   (last two lines: margin, freewheel, turning_rate, mobil_is_sliding, car_track_extra)

With defaults (n_zone=40, n_prev=5, n_steer_parts=1 → n_action_dims=4, n_contact=4): **215**.

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
     - Previous actions: [accel, brake, left, right] × 5 steps (n_steer_parts=1)
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
   * - 184
     - 1
     - turning_rate
     - scene_mobil.turning_rate
   * - 185
     - 1
     - mobil_is_sliding (car-level)
     - scene_mobil.is_sliding
   * - 186–214
     - 29
     - **car_track_extra**: add_linear_speed(3), force(3), torque(3), input_gas/brake/steer(3), speed_forward/sideward(2), max_linear_speed(1), current_local_speed(3), turbo(2), has_any_lateral_contact, burnout_state, engine(3), track(4)
     - dyna, scene_mobil, sync_vehicle_state, engine, zone math

**Indices used elsewhere:** ``buffer_management._float_layout_indices(config)`` uses ``float_inputs.get_segment_start_indices(config)`` so indices respect ``state_observation.include``. Reward shaping requires ``gear_and_wheels``, ``velocity``, ``zone_centers_in_car_frame`` to be included.

What we use from SimStateData
-----------------------------

- **dyna.current_state**: position, rotation (orientation), linear_speed, angular_speed, **add_linear_speed, force, torque** (in car_track_extra).
- **scene_mobil**: engine.gear, engine.actual_rpm, gearbox_state, is_freewheeling, **turning_rate, is_sliding**, input_gas/brake/steer, max_linear_speed, current_local_speed, turbo_boost_factor, is_turbo, has_any_lateral_contact, burnout_state; **engine**: slide_factor, braking_factor, clamped_rpm.
- **sync_vehicle_state**: speed_forward, speed_sideward (in car_track_extra).
- **simulation_wheels** (×4): real_time_state.is_sliding, has_ground_contact, damper_absorb, contact_material_id (mapped via ``contact_materials.py``).
- **sim_state.race_time**: used for progress and finish detection (not in the float vector).
- **Zone/track**: distance_since_track_begin, meters_in_current_zone, segment_length_meters, current_zone_idx (in car_track_extra).

We do **not** read previous_state / temp_state from dyna; we only use current_state.

What the game provides but we do NOT use
----------------------------------------

- **dyna**: previous_state, temp_state, inverse_inertia_tensor, quat (we use rotation matrix), unknown, not_tweaked_linear_speed, owner.
- **scene_mobil**: quality, block_flags, prev_sync_vehicle_state, async_vehicle_state, water_forces_applied, last_has_any_lateral_contact_time, turbo_type, roulette_value, wheel_contact_absorb_counter, etc.
- **simulation_wheels**: everything except real_time_state (is_sliding, has_ground_contact, damper_absorb, contact_material_id).
- **SimStateData top-level**: version, context_mode, flags, timers, cp_data (we use our own zone/VCP math).
- **TMInterface**: ``get_inputs()`` returns replay script string; we do not use it for the RL observation.

Outputs: commands we send to the car
-------------------------------------

We send commands via ``TMInterface.set_input_state(left, right, accelerate, brake)`` — all four arguments are **booleans**. The agent outputs a **multi-label action vector** (length ``n_action_dims``); we convert it to game booleans with ``ActionSpace.from_config(cfg).to_game_input(vec)`` (or legacy ``action_vector_to_game_input()`` in ``trackmania_rl.action_vector``): left = any of left_1..left_N, right = any of right_1..right_N, accelerate/brake from the first two dimensions. So we still drive the car with binary inputs; the multi-label space allows the network to express "gas + slight left" or "brake + right" as separate dimensions for learning.

**Legacy:** Old replays or manifests may store a single integer ``action_idx`` (0..11). For backward compatibility we use ``STANDARD_12_ACTIONS`` and ``ActionSpace.from_config(cfg).from_game_input(inputs[idx])`` (or ``game_input_to_action_vector()``) to convert that index to an action vector when building BC cache or replaying.

Meta capture (full state in manifest)
-------------------------------------

When capturing replays with ``capture_replays_tmnf.py`` and ``--fps-meta``, we save **all** car-state-related fields into each meta snapshot (``sim_state_utils.sim_state_to_dict``): dyna (current/previous/temp), full scene_mobil (inputs, turbo, burnout, engine, sync/async states), and simulation_wheels. The float vector is built from this meta in BC via ``state_dict_from_meta()`` and includes **turning_rate**, **mobil_is_sliding**, and **car_track_extra** (29 values); missing keys default to 0. The dimension of the float vector is always the **extended** one (config ``float_input_dim``).

Summary
-------

- **Dimension**: 215 with default config (n_zone=40, n_prev=5, n_contact=4). Layout is fixed in ``float_inputs.build_float_vector()`` and must match ``float_input_dim`` and state normalization in config.
- **Actions**: Multi-label vector of length ``n_action_dims`` (from ``action_space.inputs`` or legacy ``2 + 2*n_steer_parts``); converted to game (left, right, accelerate, brake) via ``ActionSpace.from_config(cfg).to_game_input(vec)``.
- **Used from game**: dyna (position, orientation, speeds, add_linear_speed, force, torque), mobil (engine, gearbox, freewheeling, turning_rate, is_sliding, inputs, turbo, burnout, engine stats, sync speeds), per-wheel state, and track/zone metrics in car_track_extra.
- **Not used**: previous/temp dyna state, some mobil flags/timers, wheel geometry/sync state, replay input string, raw checkpoint data. Adding more would require extending ``FloatStateDict``, ``build_float_vector()``, config, and reward/buffer indices.
