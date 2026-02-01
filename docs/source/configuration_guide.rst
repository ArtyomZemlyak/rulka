====================
Configuration Guide
====================

This guide provides detailed documentation for all configuration parameters in TrackMania RL (Rulka).

Configuration files are located in ``config_files/`` and organized by category for easy editing.
Each setting includes a brief inline comment. This document provides comprehensive explanations.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
===========

Configuration is loaded from a single **YAML file** at startup and accessed via ``get_config()``:

.. code-block:: python

   from config_files.config_loader import get_config
   
   # Access any setting (flat attribute access)
   cfg = get_config()
   batch_size = cfg.batch_size
   learning_rate = cfg.lr_schedule

To run training with a specific config:

.. code-block:: bash

   python scripts/train.py --config config_files/config_default.yaml

You can version configs with separate YAML files (e.g. ``config_uni18.yaml``) and pass the path with ``--config``. User-specific settings (paths, usernames) are read from a ``.env`` file in the project root. Config is loaded once per process and cached; there is no hot-reload.

Configuration Structure (YAML)
==============================

The default YAML (``config_files/config_default.yaml``) is organized into sections that correspond to the former Python modules:

1. **environment** - Environment and simulation
2. **neural_network** - Network architecture
3. **training** - Training hyperparameters
4. **memory** - Replay buffer
5. **exploration** - Exploration strategies
6. **rewards** - Reward shaping
7. **map_cycle** - Map training cycle
8. **performance** - System performance

Environment Configuration
==========================

Located in the ``environment`` section of the config YAML.

Timing Configuration
--------------------

.. py:data:: tm_engine_step_per_action
   :type: int
   :value: 5

   **Number of game simulation steps per agent action**
   
   Controls temporal resolution of agent control. The game engine runs at 100Hz (10ms per step).
   
   - **Lower values (1-3)**: Finer control, more actions per second, slower data collection
   - **Higher values (5-10)**: Coarser control, faster data collection, easier learning
   - **Current**: 5 steps = 50ms per action = 20 actions/second
   
   .. tip::
      Start with 5 for most maps. Reduce to 3-4 for technical sections requiring precise control.

.. py:data:: ms_per_tm_engine_step
   :type: int
   :value: 10

   **Milliseconds per simulation step** (fixed by game engine)
   
   TrackMania engine runs at 100Hz. **Do not modify** unless you understand game internals.

.. py:data:: ms_per_action
   :type: int
   :value: 50

   **Total milliseconds per agent action** (computed)
   
   Calculated as: ``tm_engine_step_per_action × ms_per_tm_engine_step``
   
   With defaults: 5 × 10ms = 50ms (20 actions/second)

Spatial Configuration
---------------------

.. py:data:: n_zone_centers_in_inputs
   :type: int
   :value: 40

   **Number of waypoints** (zone centers) used as input to the agent
   
   The agent observes upcoming waypoints along the track centerline for spatial awareness.
   
   - **Purpose**: Provides lookahead vision of track geometry
   - **Lower values (20-30)**: Less memory, less forward vision
   - **Higher values (40-60)**: More forward vision, better long-term planning
   - **Memory impact**: Each waypoint adds 3 floats (X,Y,Z) to input
   - **Current**: 40 waypoints with 20-waypoint spacing covers ~400m ahead

.. py:data:: one_every_n_zone_centers_in_inputs
   :type: int
   :value: 20

   **Sampling rate for waypoints**
   
   Only every N-th waypoint is fed to the network to reduce input dimensionality.
   
   - **Lower values (10-15)**: Denser track representation, more inputs
   - **Higher values (20-30)**: Sparser representation, faster inference
   - **Current**: Sample every 20th waypoint
   
   With ``distance_between_checkpoints=0.5m``, this gives waypoints every 10 meters.

.. py:data:: distance_between_checkpoints
   :type: float
   :value: 0.5

   **Spacing between consecutive virtual checkpoints** (meters)
   
   Track is discretized into virtual checkpoints for progress tracking.
   
   - **Lower values (0.3-0.5m)**: Finer progress tracking, more checkpoints
   - **Higher values (0.5-1.0m)**: Coarser tracking, fewer checkpoints
   - **Current**: 0.5m provides good balance
   
   .. warning::
      Very low values (<0.3m) can cause performance issues with many checkpoints.

.. py:data:: n_zone_centers_extrapolate_before_start_of_map
   :type: int
   :value: 20

   **Virtual waypoints before the start line**
   
   Number of extrapolated zone centers added before the actual track start. These virtual waypoints extend the track centerline backwards from the start line.
   
   **Why needed:**
   
   - At race start, the car may be positioned before the start line (during countdown/initialization)
   - The zone tracking system needs valid zone indices even when the car is before the start
   - ``current_zone_idx`` is initialized to this value at rollout start
   
   **How it works:**
   
   - Virtual waypoints are created by extrapolating backwards along the direction from the first real checkpoint to the second
   - This creates a smooth continuation of the track before the start
   - The system can track position and progress even before crossing the start line
   
   **Typical values:** 10-30. Current: 20 provides sufficient buffer for initialization.

.. py:data:: n_zone_centers_extrapolate_after_end_of_map
   :type: int
   :value: 1000

   **Virtual waypoints after the finish line**
   
   Number of extrapolated zone centers added after the actual track finish. These virtual waypoints extend the track centerline forwards from the finish line.
   
   **Why needed:**
   
   - After crossing the finish, the car may continue moving forward
   - The system needs to track position and calculate distances even after finish
   - Used for calculating distance-to-finish notifications (see ``margin_to_announce_finish_meters``)
   - Prevents zone tracking from breaking when the car overshoots the finish
   
   **How it works:**
   
   - Virtual waypoints are created by extrapolating forwards along the direction from the last real checkpoint to the second-to-last
   - The agent cannot enter the final virtual zone (protected by a check in zone tracking)
   - Used to compute remaining distance to finish for reward shaping and state representation
   
   **Why so many (1000)?**
   
   - After finishing, the car may coast for significant distance
   - Need enough virtual waypoints to cover potential overshoot
   - With ``distance_between_checkpoints=0.5m``, 1000 waypoints = ~500 meters of virtual track
   - Ensures distance calculations remain valid even if the car travels far past finish
   
   **Typical values:** 500-2000. Current: 1000 provides generous buffer for post-finish tracking.

.. py:data:: road_width
   :type: int
   :value: 90

   **Maximum allowable lateral distance** from centerline (meters)
   
   Used to determine if car is on-track or off-track. Includes safety margin.
   
   - **Purpose**: Collision detection and checkpoint validation
   - **Current**: 90m is conservative (actual roads are 16-32m wide)
   - **Typical range**: 50-100m

Temporal Configuration - Mini-Races
------------------------------------

**What are mini-races?**

Mini-races are a key technique that allows the agent to learn with ``gamma = 1`` (no discounting) by reinterpreting each state as part of a fixed-duration "mini-race" rather than the full track trajectory. This simplifies learning and enables efficient credit assignment.

**How it works:**

1. During training, when sampling a batch from the replay buffer, each transition is reinterpreted as part of a random 7-second "mini-race"
2. A random "current time" (0 to 7 seconds) is sampled **independently for each transition** in the batch
3. The state is interpreted as "we are at time X in a mini-race"
4. If the next state would exceed 7 seconds, the transition becomes terminal
5. Q-values represent "expected sum of rewards in the next 7 seconds" instead of "expected sum of discounted rewards until finish"

**Important: How intervals are selected**

The 7-second intervals are **not sequential** (0-7, 7-14, 14-21...). Instead:

- Each transition from the real race can be reinterpreted as part of **any random 7-second window**
- For example, a transition at 15 seconds into the race might be interpreted as:
  - "Time 0 in a mini-race" (covering 15-22 seconds of real race)
  - "Time 3.5 in a mini-race" (covering 11.5-18.5 seconds)
  - "Time 6.5 in a mini-race" (covering 8.5-15.5 seconds)
  - Any other random position

- Intervals **overlap extensively** and are sampled randomly for each batch
- This means the same real transition can be trained on as part of many different mini-race contexts

**How does the agent learn the full track?**

Even though Q-values only predict 7 seconds ahead, the agent still learns to drive the entire track efficiently:

1. **Local optimization → global optimization**: By optimizing every 7-second segment along the track, the agent implicitly optimizes the full trajectory
2. **Overlapping coverage**: Since intervals overlap and are randomly sampled, transitions from all parts of the track are trained with various mini-race contexts
3. **Greedy policy**: At inference time, the agent greedily selects actions that maximize the 7-second Q-value, which naturally chains into a good full-track policy
4. **Reward structure**: The rewards (progress, time penalty) encourage forward progress, so optimizing 7-second segments leads to efficient full-track driving

**Benefits:**

- **Simplified learning**: No need to learn long-term value estimates for the entire track
- **Gamma = 1**: Can use undiscounted returns because the horizon is naturally limited
- **Better credit assignment**: Focuses learning on near-term consequences (7 seconds)
- **Stability**: Avoids issues with very long episodes and sparse rewards

**Implementation:**

The mini-race logic is implemented in ``buffer_utilities.buffer_collate_function()``, which is called during batch sampling. The first element of ``state_float`` contains the current time in the mini-race (in actions).

.. py:data:: temporal_mini_race_duration_ms
   :type: int
   :value: 7000

   **Duration of mini-races** (milliseconds)
   
   The fixed horizon for each mini-race. All Q-values are defined as "expected sum of rewards in the next N milliseconds".
   
   - **Purpose**: Defines the temporal horizon for value estimation
   - **Current**: 7000ms = 7 seconds
   - **Typical range**: 5000-10000ms (5-10 seconds)
   
   **Trade-offs:**
   
   - **Shorter (3-5s)**: Faster learning, but may miss long-term consequences
   - **Longer (10-15s)**: Better long-term planning, but slower learning and more variance
   - **Current (7s)**: Good balance for TrackMania's typical decision-making horizon
   
   With ``ms_per_action = 50ms``, this equals 140 actions per mini-race.

.. py:data:: temporal_mini_race_duration_actions
   :type: int
   :value: 140

   **Duration of mini-races in actions** (computed automatically)
   
   Automatically calculated as ``temporal_mini_race_duration_ms // ms_per_action``.
   
   Used internally for mini-race time calculations and terminal state detection.

.. py:data:: margin_to_announce_finish_meters
   :type: int
   :value: 700

   **Distance threshold to notify agent of finish line** (meters)
   
   When the agent is within this distance of the finish, the distance-to-finish feature in the state is capped at this value. This provides a consistent signal as the agent approaches the finish.
   
   **Why needed:**
   
   - The agent needs to know it's approaching the finish to adjust behavior (e.g., maintain speed, avoid unnecessary actions)
   - Without capping, the distance feature would decrease rapidly near finish, creating a non-linear signal
   - Capping provides a stable "finish approaching" signal within the last 700 meters
   
   **How it's used:**
   
   - Included in ``state_float`` as one of the input features
   - Value is ``min(margin_to_announce_finish_meters, actual_distance_to_finish)``
   - When far from finish (>700m), shows actual distance
   - When close (<700m), shows 700m (constant signal)
   
   **Typical values:** 500-1000 meters. Current: 700m provides good advance warning for finish approach.

Contact and Physics
-------------------

.. py:data:: n_contact_material_physics_behavior_types
   :type: int
   :value: 4

   **Number of surface physics categories** used as scalar input
   
   TrackMania has many surface types (Concrete, Grass, Ice, Turbo, Dirt, etc.). They are grouped into a small number of *physics behavior* categories for the agent's input. This parameter is the number of such categories that are explicitly encoded in the state.
   
   **How it works:**
   
   - Surface types are grouped in ``trackmania_rl/contact_materials.py`` (e.g. Asphalt-like, Grass, Dirt, Turbo, and "other")
   - For each of the 4 wheels, the game provides the current contact material ID
   - The state includes a one-hot-like encoding per wheel: for each category index ``0 .. n_contact_material_physics_behavior_types - 1``, one float indicates whether that wheel is on that surface category
   - Total floats from contact materials: **4 wheels × n_contact_material_physics_behavior_types** = 4 × 4 = 16
   
   **Why needed:**
   
   - Grip and behavior depend strongly on surface (asphalt vs grass vs turbo vs ice)
   - The agent needs to know which surface each wheel is on to predict handling and choose actions
   - Using a few physics groups keeps the input size small while preserving the main distinction (road vs off-road vs turbo vs other)
   
   **Current:** 4 categories. The mapping from game materials to these groups is defined in ``contact_materials.py`` (e.g. 0 = Asphalt-like, 1 = Grass, 2 = Dirt, 3 = Turbo; other materials map to an implicit "other" and are not encoded as a separate category index).
   
   **Do not change** unless you also change the grouping logic in ``contact_materials.py`` and the corresponding input dimension in the config.

.. py:data:: n_prev_actions_in_inputs
   :type: int
   :value: 5

   **Number of previous actions** included in the state (action history)
   
   The state includes the last N actions taken by the agent, each encoded as 4 binary flags: accelerate, brake, left, right (see ``config_files/inputs_list.py``). This gives the network a short history of what the car was doing.
   
   **How it works:**
   
   - At each step, the last ``n_prev_actions_in_inputs`` actions are taken from ``rollout_results["actions"]``
   - Each action is expanded to 4 floats: one per input name in ``["accelerate", "brake", "left", "right"]`` (1.0 if that input is pressed, 0.0 otherwise)
   - They are concatenated in order (oldest to newest)
   - Total floats from action history: **4 × n_prev_actions_in_inputs** = 4 × 5 = 20
   
   **Why needed:**
   
   - The MDP is not fully Markovian from a single frame: steering and acceleration have inertia, and the current command is partly a continuation of the previous ones
   - Including the last few actions makes the state closer to Markovian and helps the policy produce smooth, consistent control (e.g. sustained turns instead of jitter)
   - Without action history, the agent would have to infer "I was turning left" from the image/state alone, which is harder and noisier
   
   **Trade-offs:**
   
   - **Larger (6–8):** Longer memory of past actions, smoother behavior, more input dimensions
   - **Smaller (3–4):** Fewer parameters, but less context and possibly jerkier control
   
   **Current:** 5 actions. With 50 ms per action, this is 250 ms of action history (~0.25 s).
   
   Changing this value changes ``float_input_dim`` (add or subtract 4 per action); it is computed from config at load time.

Timeouts
--------

.. py:data:: cutoff_rollout_if_race_not_finished_within_duration_ms
   :type: int
   :value: 300000

   **Maximum race duration** before forced termination (milliseconds)
   
   Prevents infinite loops and stuck states. Race is cut off if not finished within this time.
   
   - **Purpose**: Prevent endless rollouts
   - **Current**: 300,000ms = 5 minutes
   - **Typical range**: 180,000-600,000ms (3-10 minutes)

.. py:data:: cutoff_rollout_if_no_vcp_passed_within_duration_ms
   :type: int
   :value: 2000

   **Timeout if no checkpoint passed** (milliseconds)
   
   Detects when agent is stuck or driving backwards.
   
   - **Purpose**: Early termination of unproductive rollouts
   - **Current**: 2,000ms = 2 seconds
   - **Typical range**: 1,000-5,000ms

.. py:data:: timeout_during_run_ms
   :type: int
   :value: 30000

   **TMInterface command timeout during active racing** (milliseconds)
   
   How long to wait for TMInterface responses while car is racing.
   
   - **Current**: 30,000ms = 30 seconds
   - **Previous**: 10,100ms (increased for stability)
   
   .. note::
      Higher values prevent timeout on lag spikes but slow error detection.

.. py:data:: game_reboot_interval
   :type: int
   :value: 43200

   **Auto-restart interval** for TrackMania (seconds)
   
   Automatically reboots game to prevent memory leaks during long training sessions.
   
   - **Current**: 43,200s = 12 hours
   - **Typical range**: 21,600-86,400s (6-24 hours)

Game Settings
-------------

.. py:data:: game_camera_number
   :type: int
   :value: 2

   **Camera view** in TrackMania
   
   - 1: Behind car
   - 2: First person (recommended for RL)
   - 3: Top view
   
   **Current**: 2 (first person) matches human driving perspective

.. py:data:: sync_virtual_and_real_checkpoints
   :type: bool
   :value: True

   **Align virtual checkpoints with game checkpoints**
   
   Ensures custom virtual checkpoint progress matches official game checkpoint progress.
   
   - **True**: Virtual CPs synchronized with game CPs (recommended)
   - **False**: Independent virtual checkpoint system

Neural Network Configuration
=============================

Located in the ``neural_network`` section of the config YAML.

Image Dimensions
----------------

.. py:data:: W_downsized
   :type: int
   :value: 160

   **Width of captured game frames** (pixels)
   
   Game screenshots are resized to this width before being fed to the neural network.
   
   - **Lower values**: Faster training, less memory, reduced visual detail
   - **Higher values**: Better visual quality, slower training, more memory
   - **Typical range**: 128-256 pixels
   - **Current**: 160 pixels provides good balance
   
   .. note::
      The CNN output dimension is automatically calculated from these dimensions when the network is created.
      No manual configuration needed.

.. py:data:: H_downsized
   :type: int
   :value: 120

   **Height of captured game frames** (pixels)
   
   Game screenshots are resized to this height before being fed to the neural network.
   
   - **Typical range**: 96-192 pixels
   - **Current**: 120 pixels (4:3 aspect ratio with W=160)
   
   .. note::
      The CNN output dimension is automatically calculated from these dimensions when the network is created.
      No manual configuration needed.

Input Dimensions
----------------

.. py:data:: float_input_dim
   :type: int
   :value: 191

   **Total dimension of scalar (non-image) inputs** (computed)
   
   Size of feature vector fed to network alongside images.
   
   **Breakdown of 27 base features:**
   
   - 1: Time remaining in mini-race
   - 20: Previous action encodings (5 actions × 4 binary flags)
   - 4: Car gear information
   - 2: Speed-related features
   
   **Dynamic features:**
   
   - 3 × ``n_zone_centers_in_inputs`` (40): Waypoint X,Y,Z coordinates
   - 4 × ``n_prev_actions_in_inputs`` (5): Recent action history
   - 4 × ``n_contact_material_physics_behavior_types`` (4): Wheel contact materials
   - 1: Additional feature (freewheeling flag)
   
   **Current**: 27 + 120 + 20 + 16 + 1 = 184 features

State Normalization (float_inputs_mean / float_inputs_std)
---------------------------------------------------------

Defined in the ``state_normalization`` section of the config YAML (or built from defaults in the loader).

Scalar inputs are normalized before the network: ``(float_inputs - float_inputs_mean) / float_inputs_std``. This keeps activations in a reasonable range and can speed up training.

**How were these values obtained?**

The repo does **not** include a script that computes them from data. The current values are a mix of:

1. **Domain-derived:** From config or known ranges. Examples:
   - First feature (mini-race time): mean = ``temporal_mini_race_duration_actions / 2`` (70), std = 70.
   - Distance to finish: mean = ``margin_to_announce_finish_meters`` (700), std = 350.

2. **Typical for binary/bounded inputs:** For action flags (0/1) and wheel/gear floats, mean and std are set to plausible "typical" values (e.g. 0.5 for symmetric binary, 0.8/0.2 for "often accelerate").

3. **Possibly empirical:** The 40 waypoint coordinates (120 floats) have non-round means and stds (e.g. -2.1, 9.5, 19.1, 28.5…), which suggests they may have been computed as sample mean and std over rollouts on one or more maps in the past, then hardcoded.

**How to recompute from your own data:**

1. Collect many ``state_float`` vectors (same order as in ``game_instance_manager.py``: placeholder, previous_actions, gear/wheels, angular velocity, velocity, y_map, zone_centers, distance_to_finish, freewheeling). During training, the first element is overwritten with mini-race time in ``buffer_collate_function``, so for normalization you can either use the "raw" state from the game or the post-collate state from the buffer.
2. Stack into a matrix of shape ``(N, float_input_dim)``.
3. Compute ``mean = np.nanmean(data, axis=0)`` and ``std = np.nanstd(data, axis=0)`` (or replace zeros in std with a small constant to avoid division by zero).
4. Update ``float_inputs_mean`` and ``float_inputs_std`` in ``state_normalization.py``.

If you change the number or order of float features (e.g. waypoints, actions), the length and indices in the config's state normalization must match ``float_input_dim`` and the order in ``game_instance_manager.py`` / ``buffer_utilities.py``.

Network Architecture
--------------------

.. py:data:: float_hidden_dim
   :type: int
   :value: 256

   **Hidden layer size for scalar feature processing**
   
   Transforms float_input_dim features before merging with visual features.
   
   - **Lower values (128-192)**: Faster, less capacity
   - **Higher values (256-512)**: More capacity, slower
   - **Typical range**: 128-512

.. py:data:: dense_hidden_dimension
   :type: int
   :value: 1024

   **Main hidden layer size**
   
   Primary representation layer after combining visual and scalar features.
   
   - **Lower values (512-768)**: Faster training, less capacity
   - **Higher values (1024-2048)**: More model capacity, slower
   - **Typical range**: 512-2048
   - **Current**: 1024 provides strong capacity for complex tracks

IQN Parameters
--------------

Implicit Quantile Networks (IQN) model the full return distribution rather than just expected value.

See: `Dabney et al. 2018 - Implicit Quantile Networks for Distributional RL <https://arxiv.org/abs/1806.06923>`_

.. py:data:: iqn_embedding_dimension
   :type: int
   :value: 64

   **Quantile embedding dimension**
   
   Controls how finely the return distribution is modeled.
   
   - **Higher values**: More expressive distribution modeling
   - **Typical range**: 32-128
   - **Paper recommendation**: 64

.. py:data:: iqn_n
   :type: int
   :value: 8

   **Number of quantile samples during training**
   
   How many quantile samples to draw when computing training loss.
   
   - **Must be even** (sampled symmetrically around 0.5)
   - **Higher values**: More stable gradients, slower training
   - **Typical range**: 8-32
   - **Paper recommendation**: 8 for training

.. py:data:: iqn_k
   :type: int
   :value: 32

   **Number of quantile samples during inference**
   
   How many quantile samples for action selection during rollouts.
   
   - **Must be even** (sampled symmetrically around 0.5)
   - **Higher values**: Better action selection, slower inference
   - **Typical range**: 8-64
   - **Paper recommendation**: 32 for evaluation

.. py:data:: iqn_kappa
   :type: float
   :value: 0.005

   **Huber loss threshold**
   
   Transition point between L1 and L2 loss in quantile Huber loss.
   
   - **Lower values**: More robust to outliers (L1-like)
   - **Higher values**: More sensitive to all errors (L2-like)
   - **Typical range**: 1e-3 to 1.0
   - **Paper default**: 1.0
   - **Current**: 5e-3 works better empirically

Q-learning variant
------------------

.. py:data:: use_ddqn
   :type: bool
   :value: False

   **Use Double DQN for target computation**
   
   Switches how the TD target for the next state is computed in the IQN training loop
   (see ``trackmania_rl/agents/iqn.py``, ``train_on_batch``).
   
   **When False (standard DQN-style):**
   
   - Target: ``reward + gamma * max_a Q_target(next_state, a)``
   - The target network both selects the best action and evaluates it, which can lead to
     overestimation of Q-values.
   
   **When True (Double DQN):**
   
   - The **online** network selects the action: ``a* = argmax_a Q_online(next_state, a)``
     (after averaging over quantiles).
   - The **target** network evaluates that action: target = ``reward + gamma * Q_target(next_state, a*)``.
   - Selecting and evaluating are decoupled, which reduces overestimation and often
     stabilizes training.
   
   **Effect:**
   
   - **True**: Usually more stable learning, less Q overestimation; one extra forward pass
     through the online network per batch.
   - **False**: Slightly faster per batch; may overestimate Q more.
   
   See: `van Hasselt et al. 2016 - Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/abs/1509.06461>`_

Gradient Clipping
-----------------

.. py:data:: clip_grad_value
   :type: float
   :value: 1000

   **Maximum absolute gradient value**
   
   Clips individual gradient elements to [-value, +value].
   
   - **Purpose**: Prevent exploding gradients
   - **Current**: 1000 (effectively disabled)
   - **Typical range**: 10-1000

.. py:data:: clip_grad_norm
   :type: float
   :value: 30

   **Maximum gradient L2 norm**
   
   Scales gradient if its L2 norm exceeds this value.
   
   - **Purpose**: Prevent exploding gradients by total magnitude
   - **Lower values**: More aggressive clipping, more stable
   - **Higher values**: Less clipping, faster learning, less stable
   - **Typical range**: 10-100

Target Network
--------------

.. py:data:: number_memories_trained_on_between_target_network_updates
   :type: int
   :value: 2048

   **Target network update frequency** (in transitions)
   
   How often to update target network from online network.
   
   - **Purpose**: Stable Q-value targets during training
   - **Higher values**: More stable but slower improvement propagation
   - **Current**: 2048 transitions ≈ 4 batches (with batch_size=512)
   - **Typical range**: 1000-10000

.. py:data:: soft_update_tau
   :type: float
   :value: 0.02

   **Soft update coefficient** (tau)
   
   Interpolation factor: ``target = tau × online + (1 - tau) × target``
   
   - **tau=1.0**: Hard update (full copy)
   - **tau=0.0**: No update
   - **Lower values**: Smoother, more stable updates
   - **Current**: 0.02 is relatively aggressive
   - **Typical range**: 0.001-0.1

Training Configuration
======================

Located in the ``training`` section of the config YAML.

Run Identification
------------------

.. py:data:: run_name
   :type: str
   :value: "uni_3"

   **Experiment identifier**
   
   Used for:
   
   - Tensorboard log directory naming
   - Model checkpoint naming
   - Distinguishing multiple experiments
   
   **Example**: ``"uni_3"``, ``"A02_training"``, ``"experiment_v2"``

Schedules
---------

All schedules are lists of ``(cumulative_frames, value)`` tuples.
Values are linearly interpolated between schedule points.

.. py:data:: global_schedule_speed
   :type: float
   :value: 1.0

   **Global schedule multiplier**
   
   Speeds up or slows down all frame-based schedules uniformly.
   
   - **1.0**: Normal speed
   - **>1.0**: Accelerated schedules
   - **<1.0**: Decelerated schedules
   
   Useful for adjusting training duration without editing all schedules.

Optimizer
---------

.. py:data:: adam_epsilon
   :type: float
   :value: 0.0001

   **Adam epsilon** for numerical stability
   
   Small constant added to denominator in Adam's adaptive learning rate.
   
   - **Paper default**: 1e-8
   - **Current**: 1e-4 helps with stability
   - **Typical range**: 1e-8 to 1e-3

.. py:data:: adam_beta1
   :type: float
   :value: 0.9

   **Adam beta1** (first moment decay)
   
   Controls gradient direction smoothing.
   
   - **Paper default**: 0.9 (recommended)
   - **Typical range**: 0.9-0.99

.. py:data:: adam_beta2
   :type: float
   :value: 0.999

   **Adam beta2** (second moment decay)
   
   Controls gradient magnitude smoothing.
   
   - **Paper default**: 0.999 (recommended)
   - **Typical range**: 0.99-0.999

.. py:data:: batch_size
   :type: int
   :value: 512

   **Training batch size**
   
   Number of transitions sampled from replay buffer per training step.
   
   **Larger batches:**
   
   - ✅ More stable gradient estimates
   - ✅ Better GPU utilization
   - ❌ More memory usage
   - ❌ Less frequent weight updates per frame
   
   - **Typical range**: 32-512
   - **Current**: 512 is aggressive but works with large buffers

.. py:data:: lr_schedule
   :type: list
   :value: [(0, 0.001), (3000000, 5e-05), (12000000, 5e-05), (15000000, 1e-05)]

   **Learning rate schedule**
   
   Controls gradient descent step size throughout training.
   
   **Strategy:**
   
   - Start high (1e-3) for rapid initial learning
   - Decay to 5e-5 at 3M frames for stable convergence
   - Maintain until 12M frames
   - Final decay to 1e-5 at 15M frames for fine-tuning
   
   **Rationale**: Large steps for exploration, small steps for exploitation.
   
   **Why a separate schedule is needed:**
   
   Even though RAdam optimizer is used (which has built-in variance reduction through rectification), a separate learning rate schedule is still necessary for the following reasons:
   
   - **RAdam does not have built-in warmup**: While RAdam's rectification mechanism reduces variance in early training stages, it does not provide explicit learning rate warmup or decay functionality.
   
   - **Decay is essential**: The schedule provides explicit learning rate decay throughout training, which is crucial for stable convergence and fine-tuning. The exponential decay between schedule points allows smooth transitions.
   
   - **Frame-based updates**: Unlike standard PyTorch schedulers (which update based on optimizer steps via ``scheduler.step()``), this schedule updates based on ``cumul_number_frames_played``, which better matches the RL training paradigm where learning rate should be tied to environment interactions rather than optimization steps.
   
   - **Custom interpolation**: The implementation uses exponential interpolation between schedule points, providing smoother decay than step-based schedulers.

.. py:data:: gamma_schedule
   :type: list
   :value: [(0, 0.999), (1500000, 0.999), (2500000, 1)]

   **Discount factor (gamma) schedule**
   
   How much to value future rewards vs immediate rewards.
   
   - **gamma → 1.0**: Value long-term consequences (far-sighted)
   - **gamma → 0.0**: Value immediate rewards (myopic)
   
   **Strategy:**
   
   - Start at 0.999 (slight discounting)
   - Transition to 1.0 by 2.5M frames (undiscounted)
   
   **Rationale**: TrackMania benefits from long-term planning. Gamma=1.0 treats all future rewards equally.

N-Step Learning
---------------

.. py:data:: n_steps
   :type: int
   :value: 3

   **Number of steps for n-step returns**
   
   How many steps ahead to bootstrap Q-value estimate.
   
   - **n=1**: TD(0) - high bias, low variance
   - **n>1**: Multi-step - lower bias, higher variance
   - **n=∞**: Monte Carlo - no bias, maximum variance
   
   **Higher n:**
   
   - ✅ Faster credit assignment
   - ✅ Lower bias
   - ❌ Higher variance
   
   - **Typical range**: 1-5
   - **Current**: 3 balances credit assignment and variance

.. py:data:: discard_non_greedy_actions_in_nsteps
   :type: bool
   :value: True

   **Exclude exploratory actions** from n-step returns
   
   - **True**: Only greedy actions in n-step backup (recommended)
   - **False**: Include all actions
   
   Reduces exploration bias with epsilon-greedy.

Temporal Training Parameters (Mini-Races)
-----------------------------------------

These parameters bias how the **current time in the mini-race** is sampled when forming batches in ``buffer_collate_function``, and (when PER is enabled) which transitions get their priority updated.

**What is "current time" in the mini-race?**

The mini-race is always 7 seconds (140 actions). For each transition in a batch we randomly choose *where* we are within that window: a number from 0 to about 139 (in actions). That is the "current time" in the mini-race: 0 = start of the window, 139 = near the end. This number drives how many steps are left until the "finish" of the mini-race and whether the transition is terminal.

**How is this time computed (step by step)?**

The code does **not** draw uniformly from [0, 140). It does:

1. **Extended range:** Draw an integer from **[low, high)** with  
   ``low = -oversample_long_term_steps + oversample_maximum_term_steps`` (= -35),  
   ``high = temporal_mini_race_duration_actions + oversample_maximum_term_steps`` (= 145).  
   So we draw from **[-35, 145)** — including negatives and slightly above 140. That is the "extended" range: wider than the "honest" 0..139.

2. **abs():** Take the absolute value. Negatives -35..-1 become 35..1. So values 1, 2, …, 35 now each appear **twice** (once from a positive draw, once from a negative), while 0, 36, 37, …, 144 appear once. So **the probability of values 1–35 is about twice** the probability of the rest.

3. **Shift (-5):** Subtract ``oversample_maximum_term_steps`` (= 5) from the result. The range shifts left; some values become negative.

4. **clip(min=0):** Replace all negatives by 0. The final "current time" is in **[0, 139]**.

So: the final time is still *inside* the 7-second window (0..139 actions). But the distribution is **not uniform**: values roughly 0–35 (start of the mini-race, "many steps left") are more likely — that is the **oversampling** of the "long horizon".

.. py:data:: oversample_long_term_steps
   :type: int
   :value: 40

   **Oversample "long horizon"** (many steps left in the mini-race)
   
   The larger this number, the wider the **early** part of the mini-race (times 0, 1, 2, …) that gets higher probability from the abs() step. With 40, the negative part of the extended range is -35..-1; after abs(), the probability of values 1..35 is doubled. So transitions with "many steps left" in the 7-second window appear in the batch more often than under uniform sampling — they are **oversampled**.
   
   **Why:** Long-horizon transitions are often more useful for learning value; oversampling them can improve credit assignment and stability.
   
   **Typical range:** 20–60. Current: 40 (about the first 1.4 s of the 7 s).

.. py:data:: oversample_maximum_term_steps
   :type: int
   :value: 5

   **Shift and upper bound of the extended range**
   
   Used in the same formula: (1) subtract 5 after abs(), and (2) upper bound of the extended range = 140 + 5 = 145. The shift keeps the final time in a sensible range after clip(min=0) and keeps the high end (139) reachable. Without the extra width and shift, the top values could be cut off.
   
   **Typical range:** 1–10. Current: 5.

.. py:data:: min_horizon_to_update_priority_actions
   :type: int
   :value: 100

   **Minimum horizon (in actions) for PER priority updates** (computed)
   
   Automatically set to ``temporal_mini_race_duration_actions - 40`` (e.g. 140 - 40 = 100). Used **only when PER is enabled** (``prio_alpha > 0``).
   
   **How it works:** After each training step, PER updates the priority of the sampled transitions from the TD-error. The code updates priority only for transitions whose **current time in the mini-race** (first element of ``state_float``) is **less than** ``min_horizon_to_update_priority_actions``. So transitions that were interpreted as "near the end" of the mini-race (e.g. time 100–140) do **not** get their priority updated.
   
   **Why:** Short-horizon transitions (few steps left in the mini-race) have small TD-targets and noisier TD-errors; using them to update PER priorities can be misleading. Restricting updates to "long-horizon" samples (at least 40 actions left) keeps priorities more meaningful.
   
   **Summary:** Only transitions with "current time" < 100 (i.e. at least 40 actions remaining in the mini-race) get their PER priority updated. Do not change unless you also change ``oversample_long_term_steps`` or the mini-race duration.

TensorBoard Logging
-------------------

.. py:data:: tensorboard_suffix_schedule
   :type: list
   :value: [(0, ""), (6000000, "_2"), (15000000, "_3"), (30000000, "_4"), (45000000, "_5"), (80000000, "_6"), (150000000, "_7")]

   **TensorBoard log directory suffix schedule**
   
   Controls when new TensorBoard log directories are created during long training runs.
   
   **How it works:**
   
   - When a schedule point is reached, a new ``SummaryWriter`` is created with a new suffix
   - Log directory format: ``{run_name}{suffix}/`` (e.g., ``uni_4/``, ``uni_4_2/``, ``uni_4_3/``)
   - Uses staircase schedule (no interpolation - switches at exact frame counts)
   
   **Why split logs:**
   
   - **Performance**: TensorBoard can slow down significantly with very large log files (>100M data points)
   - **Organization**: Easier to analyze separate training phases
   - **Comparison**: Compare different training stages side-by-side
   - **File size**: Prevents single log files from becoming too large
   
   **When to use:**
   
   - **Long training runs** (>30M frames): Recommended to split logs
   - **Short training runs** (<30M frames): Can use single log (set to ``[(0, "")]``)
   - **Very long runs** (>100M frames): Essential for TensorBoard performance
   
   **Example**: For ``run_name = "uni_4"``:
   
   - 0-6M frames: logs to ``uni_4/``
   - 6-15M frames: logs to ``uni_4_2/``
   - 15-30M frames: logs to ``uni_4_3/``
   - And so on...

Memory Configuration
====================

Located in the ``memory`` section of the config YAML.

Buffer Size
-----------

**Why the replay buffer is needed**

The agent uses off-policy RL (IQN/DQN): it learns from past experience stored in a replay buffer, not only from the current rollout. The buffer is needed for:

1. **Sample efficiency** — Each transition (state, action, reward, next state) is expensive: it comes from real-time play. The buffer lets the learner reuse each transition many times (see ``number_times_single_memory_is_used_before_discard``). Without a buffer, we would train only once per frame.

2. **Breaking temporal correlation** — Consecutive frames from one run are highly correlated. Training on them in sequence would destabilize learning. Sampling random mini-batches from the buffer decorrelates the data and stabilizes gradient updates.

3. **Stable gradients** — Training on diverse, randomly sampled transitions approximates i.i.d. data and helps the Q-network converge.

``memory_size_schedule`` controls how large this buffer is and when learning is allowed to start.

.. py:data:: memory_size_schedule
   :type: list
   :value: [(0, (50000, 20000)), (5000000, (100000, 75000)), (7000000, (200000, 150000))]

   **Replay buffer size schedule**
   
   Format: ``(frames, (total_size, start_learning_size))``
   
   - **total_size**: Maximum number of transitions in the buffer. Limits both RAM use and how much past experience is kept. Larger buffers store more diverse data and allow more reuse per transition, but use more memory and take longer to fill.
   
   - **start_learning_size**: Minimum number of transitions that must be in the buffer before training starts. Ensures the first updates use reasonably diverse data instead of a few early rollouts; avoids overfitting to the initial exploration.
   
   The schedule is applied by *cumulative frames played*: each entry is ``(frames_played, (total_size, start_learning_size))``. Sizes grow over training so that:
   
   - **Early** (small buffer): The buffer fills quickly, learning starts sooner, and RAM use stays low.
   - **Later** (larger buffer): More diverse experience is stored for harder phases and finer policy learning.
   
   **Typical strategy:**
   
   - Start small (50K total / 20K start) for rapid early training
   - Grow to 100K/75K at 5M frames for more diversity
   - Grow to 200K/150K at 7M frames for maximum diversity
   
   **Memory estimate** (~10KB per transition):
   
   - 50K ≈ 500MB
   - 100K ≈ 1GB
   - 200K ≈ 2GB

Prioritized Experience Replay
------------------------------

See: `Schaul et al. 2015 - Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_

**Why Prioritized Experience Replay (PER)?**

With uniform replay, all transitions are sampled with equal probability. Many of them are "easy": the network already predicts them well (low TD-error), and training on them adds little. PER uses a *priority* proportional to how wrong the network was on that transition (e.g. |Q_predicted − Q_target|). High-priority transitions are sampled more often, so the same buffer and the same number of batches are used mostly on transitions the agent still needs to learn from.

**Benefits:** Better sample efficiency — less waste on trivial transitions, more updates on informative ones. Can speed up learning when the distribution of TD-errors is very uneven.

**Trade-offs:** Prioritized sampling is biased (some transitions are over-represented). Importance sampling (``prio_beta``) corrects this in the loss; ``prio_epsilon`` ensures every transition keeps a non-zero chance of being sampled so no experience is completely ignored. Tuning PER (alpha, beta, epsilon) adds complexity and can cause instability, so it is often left off in favor of uniform replay.

**In this project:** ``prio_alpha=0`` disables PER; sampling is uniform. The parameters below are available for experiments with ``prio_alpha > 0``. Priorities are updated from the absolute error between predicted and target Q-values after each training step.

.. py:data:: prio_alpha
   :type: float
   :value: 0.0

   **Priority exponent**
   
   Controls how much prioritization affects sampling. Priority is stored as ``(td_error + eps)^alpha``; sampling probability is proportional to that value.
   
   - **alpha=0.0**: Uniform sampling (no prioritization). All transitions have equal probability.
   - **alpha=1.0**: Sampling probability proportional to priority (strong prioritization).
   - **0 < alpha < 1**: Smooth interpolation; higher alpha → more focus on high-error transitions.
   
   **Paper recommendations:**
   
   - Rainbow-IQN: 0.2
   - Rainbow: 0.5
   - PER: 0.6
   
   **Current**: 0.0 (uniform) for simplicity and to avoid bias/instability from prioritization.

.. py:data:: prio_epsilon
   :type: float
   :value: 0.002

   **Priority offset**
   
   Added to TD-error before computing priority: ``(td_error + prio_epsilon)^alpha``. Ensures that every transition has a strictly positive priority and thus a non-zero chance of being sampled.
   
   - Prevents transitions from never being revisited.
   - **Current**: 2e-3 (keeps sampling reasonably uniform when alpha is small).
   - **Typical range**: 1e-6 to 1e-2.

.. py:data:: prio_beta
   :type: float
   :value: 1.0

   **Importance sampling exponent**
   
   Prioritized sampling oversamples some transitions; the loss is weighted by inverse sampling probability (raised to beta) so that the expected gradient is unbiased.
   
   - **beta=0.0**: No correction (biased updates, prioritization effect is strongest).
   - **beta=1.0**: Full correction (unbiased). Often annealed from <1 to 1 over training in the PER paper.
   
   **Current**: 1.0 (full correction; when ``prio_alpha=0`` this has no effect because sampling is uniform).

Memory Usage Control
--------------------

.. py:data:: number_times_single_memory_is_used_before_discard
   :type: int
   :value: 32

   **Expected reuse per transition**
   
   Controls how many times each transition is expected to be used for training on average. This is a *global* limit, not a per-transition counter.
   
   **How it works:**
   
   - When N new transitions are added to the buffer, the system "allows" ``N * number_times_single_memory_is_used_before_discard`` total uses across all training batches
   - Training continues while the global usage counter is below this limit
   - Transitions are removed from the buffer only when it overflows (FIFO), not based on individual usage counts
   
   **Interaction with PER:**
   
   When ``prio_alpha > 0`` (PER enabled), high-priority transitions are sampled more often. This means:
   
   - High-priority transitions participate in more batches and contribute more to the global usage counter
   - However, they are **not** removed faster — removal is FIFO-based when the buffer overflows
   - Priorities are updated after each training step: as the network learns, TD-errors decrease, priorities drop, and sampling frequency self-balances
   - This creates a natural feedback loop: transitions that are hard to learn stay longer (high priority → frequent sampling → priority updates → if still hard, priority stays high)
   
   **Potential concern:** Could high-priority transitions "consume" the usage budget faster, leaving less training for low-priority ones?
   
   - In practice, this is mitigated by the self-balancing mechanism: as transitions are learned, their priority decreases
   - ``prio_epsilon`` ensures all transitions have a non-zero sampling probability
   - The global limit ensures overall training frequency matches data collection rate
   
   **Typical values:** 1-64. Higher values = more reuse per transition, better sample efficiency, but transitions may become stale if the policy changes significantly.

Exploration Configuration
==========================

Located in the ``exploration`` section of the config YAML.

The agent uses hybrid exploration: epsilon-greedy + epsilon-Boltzmann.

Epsilon-Greedy
--------------

.. py:data:: epsilon_schedule
   :type: list
   :value: [(0, 1), (50000, 1), (300000, 0.1), (3000000, 0.03)]

   **Epsilon-greedy exploration schedule**
   
   Probability of taking completely random action.
   
   **Strategy:**
   
   - Start at 1.0 (100% random) for buffer warmup
   - Maintain until 50K frames
   - Decay to 0.1 by 300K frames
   - Final decay to 0.03 by 3M frames
   
   **Interpretation:**
   
   - epsilon=1.0: Pure exploration
   - epsilon=0.1: 90% policy, 10% random
   - epsilon=0.03: 97% policy, 3% random

Epsilon-Boltzmann
-----------------

.. py:data:: epsilon_boltzmann_schedule
   :type: list
   :value: [(0, 0.15), (3000000, 0.03)]

   **Epsilon-Boltzmann exploration schedule**
   
   Probability of Boltzmann sampling (when not taking random action).
   
   **How the action is chosen** (one action per step):
   
   - **Random** (with probability epsilon): action is chosen uniformly among all actions (ignores Q-values).
   - **Boltzmann** (with probability (1−epsilon)×epsilon_boltzmann): to each Q(s,a) we add Gaussian noise, then take the action with the **maximum** of these noised values. So we still pick one action (argmax), but which one can differ from greedy because of the noise.
   - **Greedy** (with probability (1−epsilon)×(1−epsilon_boltzmann)): action with the **maximum** Q(s,a) is taken (no noise).
   
   So the difference: **greedy** = always the best Q; **Boltzmann** = best Q after adding random noise (often the same action when tau is small, sometimes another).
   
   **Combined behavior** (epsilon=0.1, epsilon_boltzmann=0.15):
   
   - 10% purely random
   - 90% × 15% = 13.5% Boltzmann (argmax of Q + noise)
   - 90% × 85% = 76.5% greedy (argmax of Q)

.. py:data:: tau_epsilon_boltzmann
   :type: float
   :value: 0.01

   **Boltzmann temperature**
   
   In the implementation, Gaussian noise is added to Q-values before taking argmax: ``argmax(Q + tau * randn)``. So tau controls noise scale:
   
   - **tau → 0**: Almost no noise → almost always greedy (max Q)
   - **tau large**: Large noise → sometimes a suboptimal action wins
   
   **Current**: 0.01 is very low (near-greedy)
   
   **Recommendations:**
   
   - Low (0.01-0.1): Well-calibrated Q-values
   - High (0.5-1.0): Noisy Q-values, need more exploration

Rewards Configuration
=====================

Located in the ``rewards`` section of the config YAML.

Base Rewards
------------

.. py:data:: constant_reward_per_ms
   :type: float
   :value: -0.0012

   **Time penalty** (per millisecond)
   
   Fixed negative reward at every timestep to encourage speed.
   
   - **Current**: -6/5000 = -0.0012 per ms
   - **Over 5 seconds**: Accumulates -6 total
   
   Balances with progress rewards to prevent reckless driving.

.. py:data:: reward_per_m_advanced_along_centerline
   :type: float
   :value: 0.01

   **Progress reward** (per meter)
   
   Primary positive reward for forward progress along racing line.
   
   - **Current**: 5/500 = 0.01 per meter
   - **Over 500 meters**: Earns +5 total
   
   **Balance example** (50 m/s speed over 10 seconds):
   
   - Distance: 500m → Progress: +5
   - Time: 10s → Time penalty: -12
   - Net: Must drive efficiently for positive reward

Shaped Rewards
--------------

Currently all shaped and engineered rewards are **disabled** (set to 0).

This creates a clean reward structure: **"go fast, go forward"**

Avoids reward hacking and unintended behaviors from complex shaping.

.. tip::
   If learning is too slow, consider enabling:
   
   - ``shaped_reward_dist_to_cur_vcp`` for denser checkpoint guidance
   - Engineered technique rewards (after basic driving is learned)

Map Cycle Configuration
========================

Located in the ``map_cycle`` section of the config YAML.

.. py:data:: map_cycle
   :type: list

   **Map training cycle**
   
   Under ``map_cycle.entries`` in YAML, each entry has:
   
   - **short_name** (str): Logging identifier (e.g., "hock", "A01")
   - **map_path** (str): Path to .Challenge.Gbx file
   - **reference_line_path** (str): Racing line .npy file in maps/
   - **is_exploration** (bool): Use exploration strategy
   - **fill_buffer** (bool): Add experiences to replay buffer
   - **repeat** (int): How many times to repeat this entry in the cycle
   
   **Common patterns (YAML):**
   
   .. code-block:: yaml
   
      map_cycle:
        entries:
          # 4 exploration + 1 evaluation
          - {short_name: A01, map_path: "A01-Race.Challenge.Gbx", reference_line_path: "A01_0.5m_cl.npy", is_exploration: true, fill_buffer: true, repeat: 4}
          - {short_name: A01, map_path: "A01-Race.Challenge.Gbx", reference_line_path: "A01_0.5m_cl.npy", is_exploration: false, fill_buffer: true, repeat: 1}

Performance Configuration
==========================

Located in the ``performance`` section of the config YAML.

Parallelization
---------------

.. py:data:: gpu_collectors_count
   :type: int
   :value: 4

   **Number of parallel TrackMania instances**
   
   More instances = faster data collection.
   
   **Tuning recommendations:**
   
   1. Start with 2
   2. Monitor CPU/GPU/RAM usage
   3. Gradually increase until bottleneck
   4. Measure batches trained per minute
   5. Optimal: Maximizes throughput without instability
   
   **Typical values:**
   
   - 4-core CPU: 2-4 collectors
   - 8-core CPU: 4-8 collectors
   - 16-core CPU: 8-16 collectors
   
   **Memory**: ~2GB RAM per instance

.. py:data:: running_speed
   :type: int
   :value: 160

   **Game simulation speed multiplier**
   
   Run game faster than real-time for rapid data collection.
   
   - **1**: Real-time (for debugging/visualization)
   - **10-50**: Fast training with visual observation
   - **100-200**: Maximum speed (no visual observation)
   
   **Current**: 160× real-time
   
   .. warning::
      Too fast may cause physics instability or inaccurate simulation.

Network Synchronization
-----------------------

Training uses one **learner process** (updates the policy) and several **collector processes** (run the game and select actions with the current policy). The learner and collectors share the same network weights via a **shared network** in shared memory. Two parameters control how often weights are pushed from the learner and pulled by the collectors.

**Data flow:**

1. **Learner** trains ``online_network`` every batch.
2. Periodically the learner copies ``online_network`` → **shared network** (in shared memory). This is the *push*.
3. Each **collector** has its own local **inference network** used to choose actions.
4. The collector copies **shared network** → **inference network** at the start of each rollout and periodically during long rollouts. This is the *pull*.

.. py:data:: send_shared_network_every_n_batches
   :type: int
   :value: 8

   **How often the learner pushes new weights to the shared network**
   
   Every N training batches, the learner copies the current ``online_network`` weights into the shared network (in shared memory). Collectors read from this shared copy when they update their local inference network.
   
   **How it works:**
   
   - After each batch, the learner checks ``cumul_number_batches_done % send_shared_network_every_n_batches == 0``
   - When true, it does: ``shared_network.load_state_dict(online_network.state_dict())`` under a lock
   - Collectors never write to the shared network; they only read from it when they pull
   
   **Why it matters:**
   
   - **Larger (e.g. 16–32):** Fewer copies, less lock contention, slightly less up-to-date policy in collectors
   - **Smaller (e.g. 2–4):** Collectors see new weights more often, but more frequent locking and copy cost
   
   **Trade-off:** Balance between "collectors use fresh policy" and "learner is not blocked by shared-memory writes". With 8, collectors are at most 8 batches behind; at ~512 batch size that is a few thousand transitions.
   
   **Current:** 8 batches. Typical range: 4–16.

.. py:data:: update_inference_network_every_n_actions
   :type: int
   :value: 8

   **How often each collector pulls the shared network into its local inference network during a rollout**
   
   Each collector updates its local inference network from the shared network at the start of every rollout. During a long rollout (e.g. a 2-minute race), it also updates every N **actions** (e.g. every 8 actions). So the policy used for action selection can be refreshed mid-race without waiting for the next rollout.
   
   **How it works:**
   
   - At rollout start: collector always calls ``update_network()`` (shared → inference) once before driving
   - During the rollout: the game loop runs with ``_time`` (race time in milliseconds). Every ``10 * run_steps_per_action * update_inference_network_every_n_actions`` ms, the collector calls ``update_network()`` again. That product equals ``ms_per_action * update_inference_network_every_n_actions`` (with 50 ms per action), so the interval is N actions in time.
   - With ``run_steps_per_action = 5``, ``ms_per_action = 50`` and ``update_inference_network_every_n_actions = 8``: interval = 50 × 8 = 400 ms = 8 actions
   
   **Why it matters:**
   
   - **Larger (e.g. 16–32):** Fewer copies during a run; inference network may be several hundred actions behind the learner
   - **Smaller (e.g. 2–4):** Inference network stays closer to the shared (and thus learner) policy during long races; more copy and lock usage
   
   **When it helps:** Long rollouts (e.g. 1–2 min). If you update only at rollout start, the first half of the race uses weights that may be hundreds of batches old by the time the rollout ends. Updating every N actions keeps inference policy closer to the current learner policy.
   
   **Current:** 8 actions. With 50 ms per action, that is an update every 400 ms during a rollout. Typical range: 4–16.

**Summary:** ``send_shared_network_every_n_batches`` controls how often the **learner** updates the shared copy. ``update_inference_network_every_n_actions`` controls how often each **collector** refreshes its local policy from that shared copy during a single run. Both are trade-offs between freshness of the policy used for data collection and the cost of copying/locking.

Visualization and Analysis
--------------------------

.. py:data:: make_highest_prio_figures
   :type: bool
   :value: False

   **Save images of highest-priority transitions** (only when PER is enabled)
   
   When ``True`` and the buffer uses Prioritized Experience Replay (``prio_alpha > 0``), the learner periodically saves PNG images of the transitions that have the **highest priority** in the replay buffer. These are written to ``save_dir / "high_prio_figures"``, **not** to TensorBoard.
   
   **What the figures show:**
   
   - For each of the **top 20 transitions by priority** (i.e. by TD-error), the code saves a small window of transitions: indices from ``high_error_idx - 4`` to ``high_error_idx + 5``
   - Each saved image is one transition: **state frame** and **next_state frame** side by side (horizontally), upscaled 4× for visibility
   - Filename format: ``{high_error_idx}_{idx}_{n_steps}_{priority:.2f}.png`` (e.g. ``123_120_3_0.45.png``)
   
   **What you can infer from them:**
   
   - **Which situations get high TD-error:** High priority ≈ “network was most wrong on this transition”. Looking at the frames tells you *where* (e.g. sharp turn, specific surface, near wall) and *what* (state → next_state) the agent is struggling to predict.
   - **Clustering:** If many high-priority transitions look similar (e.g. same turn, same surface), the agent may need more data or reward shaping there.
   - **Debugging PER and rewards:** Helps check that high TD-error corresponds to “hard” or “surprising” situations rather than noise or bugs.
   
   **When it runs:** Only at checkpoint save time (when the learner saves weights and stats), and only if ``get_config().make_highest_prio_figures`` is ``True`` and the buffer uses ``PrioritizedSampler``. If ``prio_alpha = 0`` (uniform replay), this option has no effect.
   
   **TensorBoard:** These figures are **not** sent to TensorBoard. They are only written as PNG files under ``save_dir / "high_prio_figures"``. To inspect them in TensorBoard you would need to add custom logging (e.g. ``writer.add_image(...)``) yourself.
   
   **Performance:** Generating them is relatively slow (iterating over the buffer and saving many images), so they are disabled by default. Enable only for occasional debugging or analysis.

Advanced Topics
===============

Schedules
---------

All schedules use linear interpolation between points:

.. code-block:: python

   schedule = [
       (0, 1.0),       # Start at 1.0
       (1000, 0.5),    # Linear decay to 0.5 at 1000 frames
       (2000, 0.1),    # Linear decay to 0.1 at 2000 frames
   ]
   
   # At frame 500: value = 0.75 (interpolated)
   # At frame 1500: value = 0.3 (interpolated)

Changing Configuration
----------------------

Config is loaded once at process startup (from the YAML path passed to ``train.py --config``). To change parameters you must edit the YAML file and restart training. A snapshot of the config used for each run is saved as ``config_snapshot.yaml`` in ``save/{run_name}/``.

.. warning::
   Don't change mid-run:
   
   - Network architecture parameters
   - Input dimensions
   - Action space

   These require restarting training.

Troubleshooting
===============

Training is slow
----------------

- Increase ``gpu_collectors_count``
- Increase ``running_speed``
- Reduce ``batch_size`` for more frequent updates
- Disable visualization options

Agent gets stuck
----------------

- Check ``cutoff_rollout_if_no_vcp_passed_within_duration_ms``
- Verify map reference line is correct
- Increase exploration (higher epsilon)

Memory issues
-------------

- Reduce ``memory_size_schedule`` values
- Reduce ``gpu_collectors_count``
- Reduce ``batch_size``

Further Reading
===============

- :doc:`first_training` - First training tutorial
- :doc:`second_training` - Advanced training guide
- :doc:`project_structure` - Codebase overview
- :doc:`main_objects` - Core classes documentation

Related Papers
--------------

- `DQN (Mnih et al. 2015) <https://www.nature.com/articles/nature14236>`_
- `Rainbow (Hessel et al. 2017) <https://arxiv.org/abs/1710.02298>`_
- `IQN (Dabney et al. 2018) <https://arxiv.org/abs/1806.06923>`_
- `PER (Schaul et al. 2015) <https://arxiv.org/abs/1511.05952>`_
- `DDQN (van Hasselt et al. 2015) <https://arxiv.org/abs/1509.06461>`_
