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

All configuration settings are in compact Python files:

.. code-block:: python

   from config_files import config_copy
   
   # Access any setting
   batch_size = config_copy.batch_size
   learning_rate = config_copy.lr_schedule

To modify settings during training:

1. Edit ``config_files/config_copy.py``
2. Changes apply automatically without restart
3. Replay buffer is preserved

Configuration Modules
=====================

Settings are organized into 8 modules:

1. **environment_config.py** - Environment and simulation
2. **neural_network_config.py** - Network architecture
3. **training_config.py** - Training hyperparameters
4. **memory_config.py** - Replay buffer
5. **exploration_config.py** - Exploration strategies
6. **rewards_config.py** - Reward shaping
7. **map_cycle_config.py** - Map training cycle
8. **performance_config.py** - System performance

Environment Configuration
==========================

Located in: ``config_files/environment_config.py``

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

.. py:data:: road_width
   :type: int
   :value: 90

   **Maximum allowable lateral distance** from centerline (meters)
   
   Used to determine if car is on-track or off-track. Includes safety margin.
   
   - **Purpose**: Collision detection and checkpoint validation
   - **Current**: 90m is conservative (actual roads are 16-32m wide)
   - **Typical range**: 50-100m

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

Located in: ``config_files/neural_network_config.py``

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

Located in: ``config_files/training_config.py``

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

Located in: ``config_files/memory_config.py``

Buffer Size
-----------

.. py:data:: memory_size_schedule
   :type: list
   :value: [(0, (50000, 20000)), (5000000, (100000, 75000)), (7000000, (200000, 150000))]

   **Replay buffer size schedule**
   
   Format: ``(frames, (total_size, start_learning_size))``
   
   - **total_size**: Maximum transitions in buffer
   - **start_learning_size**: Minimum transitions before training begins
   
   **Strategy:**
   
   - Start small (50K/20K) for rapid early training
   - Grow to 100K/75K at 5M frames for diversity
   - Grow to 200K/150K at 7M frames for maximum diversity
   
   **Memory estimate** (~10KB per transition):
   
   - 50K ≈ 500MB
   - 100K ≈ 1GB
   - 200K ≈ 2GB

Prioritized Experience Replay
------------------------------

See: `Schaul et al. 2015 - Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_

.. py:data:: prio_alpha
   :type: float
   :value: 0.0

   **Priority exponent**
   
   Controls how much prioritization affects sampling.
   
   - **alpha=0.0**: Uniform sampling (no prioritization)
   - **alpha=1.0**: Fully prioritized sampling
   
   **Paper recommendations:**
   
   - Rainbow-IQN: 0.2
   - Rainbow: 0.5
   - PER: 0.6
   
   **Current**: 0.0 (uniform) for simplicity and avoiding bias.

.. py:data:: prio_epsilon
   :type: float
   :value: 0.002

   **Priority offset**
   
   Added to priorities to ensure non-zero probability for all transitions.
   
   - **Current**: 2e-3 (ensures reasonable uniformity)
   - **Typical range**: 1e-6 to 1e-2

.. py:data:: prio_beta
   :type: float
   :value: 1.0

   **Importance sampling correction**
   
   Corrects bias from prioritized sampling.
   
   - **beta=0.0**: No correction (biased)
   - **beta=1.0**: Full correction (unbiased)
   
   **Current**: 1.0 (full correction, but has no effect with alpha=0)

Exploration Configuration
==========================

Located in: ``config_files/exploration_config.py``

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
   
   Boltzmann samples actions proportional to ``exp(Q(s,a) / temperature)``.
   
   **Combined behavior** (epsilon=0.1, epsilon_boltzmann=0.15):
   
   - 10% purely random
   - 90% × 15% = 13.5% Boltzmann
   - 90% × 85% = 76.5% greedy

.. py:data:: tau_epsilon_boltzmann
   :type: float
   :value: 0.01

   **Boltzmann temperature**
   
   Controls distribution sharpness: ``P(a) ∝ exp(Q(s,a) / tau)``
   
   - **tau → 0**: Nearly deterministic (always max Q-value)
   - **tau → ∞**: Uniform distribution
   
   **Current**: 0.01 is very low (near-greedy)
   
   **Recommendations:**
   
   - Low (0.01-0.1): Well-calibrated Q-values
   - High (0.5-1.0): Noisy Q-values, need more exploration

Rewards Configuration
=====================

Located in: ``config_files/rewards_config.py``

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

Located in: ``config_files/map_cycle_config.py``

.. py:data:: map_cycle
   :type: list

   **Map training cycle**
   
   List of iterators yielding tuples:
   
   .. code-block:: python
   
      (short_name, map_path, reference_line_path, is_exploration, fill_buffer)
   
   **Components:**
   
   1. **short_name** (str): Logging identifier (e.g., "hock", "A01")
   2. **map_path** (str): Path to .Challenge.Gbx file
   3. **reference_line_path** (str): Racing line .npy file in maps/
   4. **is_exploration** (bool): Use exploration strategy
   5. **fill_buffer** (bool): Add experiences to replay buffer
   
   **Common patterns:**
   
   .. code-block:: python
   
      # Standard training: 4 exploration + 1 evaluation
      repeat((name, path, line, True, True), 4),   # 4 exploration runs
      repeat((name, path, line, False, True), 1),  # 1 evaluation run
      
      # Pure evaluation (no training):
      repeat((name, path, line, False, False), 1),
   
   See inline documentation in config file for extensive examples.

Performance Configuration
==========================

Located in: ``config_files/performance_config.py``

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

Hot-Reloading
-------------

``config_copy.py`` is reloaded periodically during training.

To change parameters on-the-fly:

1. Edit ``config_copy.py`` (not ``config.py``)
2. Save the file
3. Changes apply at next reload interval
4. Replay buffer preserved

.. warning::
   Don't change:
   
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
