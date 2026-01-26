TensorBoard Metrics Reference
==============================

This document provides a comprehensive guide to all metrics logged to TensorBoard during training. Metrics are organized into groups for easier navigation.

Overview
--------

All metrics are logged with prefixes that group them into categories:
- ``Training/`` - Training process metrics
- ``RL/`` - Reinforcement learning metrics
- ``Race/`` - Race performance metrics
- ``Gradients/`` - Gradient monitoring
- ``Performance/`` - System performance metrics
- ``Buffer/`` - Replay buffer statistics
- ``Network/`` - Neural network weights and optimizer state
- ``IQN/`` - IQN-specific metrics

Training Metrics
----------------

**Training/loss**
    **Description**: Training loss computed on batches from the replay buffer.
    
    **Interpretation**:
    - In reinforcement learning, loss increasing early in training is **normal and expected**
    - This indicates the agent is discovering the environment and identifying inconsistencies in its value estimates
    - Loss should stabilize or decrease after ~1-2M frames
    - Values typically range from 0.01 to 10.0
    
    **What to watch for**:
    - Sudden spikes (>100) may indicate gradient explosions
    - Consistently increasing loss after 5M+ frames may indicate learning issues

**Training/loss_test**
    **Description**: Test loss computed on held-out test buffer (not used for training).
    
    **Interpretation**:
    - Should track training loss but be slightly higher
    - Large gap between training and test loss indicates overfitting
    - Useful for detecting when the model is memorizing rather than generalizing
    
    **What to watch for**:
    - Test loss much higher than training loss (>2x) suggests overfitting
    - Test loss decreasing while training loss increases suggests good generalization

**Training/learning_rate**
    **Description**: Current learning rate used by the optimizer.
    
    **Interpretation**:
    - Decays according to the learning rate schedule
    - Typical range: 1e-5 to 1e-3
    - Lower learning rates in later training allow fine-tuning
    
    **What to watch for**:
    - Should decrease smoothly over time
    - Abrupt changes indicate schedule issues

**Training/weight_decay**
    **Description**: L2 regularization strength (weight decay coefficient).
    
    **Interpretation**:
    - Prevents overfitting by penalizing large weights
    - Typically proportional to learning rate
    - Range: 1e-7 to 1e-5
    
    **What to watch for**:
    - Should track learning rate if using proportional weight decay
    - Too high values can prevent learning

**Training/batch_size**
    **Description**: Number of transitions sampled per training batch.
    
    **Interpretation**:
    - Larger batches provide more stable gradients but slower updates
    - Typical values: 32, 64, 128, 256
    
    **What to watch for**:
    - Should remain constant unless explicitly changed in config

**Training/n_steps**
    **Description**: N-step return horizon for bootstrapping.
    
    **Interpretation**:
    - Number of steps used in n-step returns
    - Higher values reduce bias but increase variance
    - Typical range: 1-5
    
    **What to watch for**:
    - Should remain constant unless explicitly changed in config

**Training/discard_non_greedy_actions_in_nsteps**
    **Description**: Whether non-greedy (exploratory) actions are excluded from n-step returns.
    
    **Interpretation**:
    - 1.0 = True (only greedy actions in n-step backup)
    - 0.0 = False (all actions included)
    - Recommended: True to reduce exploration bias
    
    **What to watch for**:
    - Should remain constant unless explicitly changed in config

**Training/train_on_batch_duration**
    **Description**: Median time (in seconds) to process one training batch.
    
    **Interpretation**:
    - Lower is better (faster training)
    - Typical range: 0.01-0.1 seconds
    - Affected by GPU speed, batch size, and network complexity
    
    **What to watch for**:
    - Sudden increases may indicate GPU throttling or system issues
    - Should be relatively stable

RL Metrics
----------

**RL/avg_Q**
    **Description**: Average Q-value (expected future reward) predicted by the network.
    
    **Interpretation**:
    - **Key indicator of learning progress**
    - Starts near zero for untrained agent
    - Initially decreases as agent discovers it plays poorly
    - Should increase as agent learns better strategies
    - Higher values indicate agent expects more reward
    
    **What to watch for**:
    - Should trend upward after initial exploration phase (~500K-1M frames)
    - Plateaus indicate agent has learned current strategy
    - Decreasing values may indicate learning instability

**RL/single_zone_reached**
    **Description**: Furthest virtual checkpoint (zone) reached during a race, as percentage of track.
    
    **Interpretation**:
    - 0.0 = agent didn't start
    - 1.0 = agent finished the track
    - Shows how far agent progresses along the track
    
    **What to watch for**:
    - Should increase over time
    - Takes ~300K steps to learn to press forward
    - Takes ~500K steps to finish map for first time
    - Takes ~1M steps to regularly finish map
    - Plateaus indicate agent is stuck at certain sections

**RL/gamma**
    **Description**: Discount factor for future rewards.
    
    **Interpretation**:
    - Controls how much future rewards are valued
    - Range: 0.0 (only immediate reward) to 1.0 (all future rewards equally)
    - Typically increases from 0.999 to 1.0 during training
    - Higher values make agent plan further ahead
    
    **What to watch for**:
    - Should increase according to schedule
    - Too low values make agent short-sighted
    - Too high values (1.0) can cause instability

**RL/epsilon**
    **Description**: Epsilon-greedy exploration rate.
    
    **Interpretation**:
    - Probability of taking random action instead of greedy action
    - Decays from 1.0 (fully random) to ~0.03 (mostly greedy)
    - Higher values = more exploration
    - Lower values = more exploitation
    
    **What to watch for**:
    - Should decay smoothly according to schedule
    - Too fast decay = insufficient exploration
    - Too slow decay = agent doesn't exploit learned strategies

**RL/epsilon_boltzmann**
    **Description**: Boltzmann exploration temperature parameter.
    
    **Interpretation**:
    - Controls softmax temperature for action selection
    - Higher values = more uniform action distribution (more exploration)
    - Lower values = more peaked distribution (more exploitation)
    - Used in combination with epsilon-greedy
    
    **What to watch for**:
    - Should decay according to schedule
    - Works together with epsilon for exploration strategy

**RL/tau_epsilon_boltzmann**
    **Description**: Tau parameter for Boltzmann exploration.
    
    **Interpretation**:
    - Additional temperature parameter for IQN quantile sampling
    - Affects exploration in distributional RL setting
    - Typically constant value
    
    **What to watch for**:
    - Should remain constant unless explicitly changed

**RL/mean_action_gap**
    **Description**: Average difference between best Q-value and other Q-values per state.
    
    **Interpretation**:
    - Measures how confident the agent is in its action selection
    - Higher values = agent has clear preference for one action
    - Lower values = agent is uncertain between actions
    - Negative values are possible (computed as negative gap)
    
    **What to watch for**:
    - Should increase as agent learns (becomes more confident)
    - Very low values indicate high uncertainty

**RL/q_value_{i}_starting_frame**
    **Description**: Q-value for action {i} at the starting frame of a race.
    
    **Interpretation**:
    - Shows agent's expected reward for each action at race start
    - Useful for understanding initial action preferences
    - Typically logged for action 0 (forward)
    
    **What to watch for**:
    - Should increase as agent learns
    - Can reveal if agent has learned good starting strategy

Race Metrics
------------

**Race/eval_race_time_robust**
    **Description**: **Most important performance metric!** Best evaluation race times (greedy policy, no exploration).
    
    **Interpretation**:
    - Time in seconds for evaluation runs that finished within 2% of rolling mean
    - Only includes "robust" runs (consistent performance)
    - Lower is better
    - This is the primary metric to track for agent performance
    
    **What to watch for**:
    - Should decrease over time (agent getting faster)
    - Plateaus indicate agent has learned current strategy
    - Compare with reference times (author/gold) if available
    - Most reliable indicator of actual performance

**Race/eval_race_time_{status}_{map}**
    **Description**: Evaluation race time for specific map and status.
    
    **Interpretation**:
    - Time in seconds for evaluation runs
    - Includes all evaluation runs (not just robust ones)
    - More variable than robust times
    - Status indicates run quality (e.g., "finished", "dnf")
    
    **What to watch for**:
    - More noisy than robust times
    - Useful for tracking completion rates

**Race/explo_race_time_finished**
    **Description**: Exploration race times for runs that finished.
    
    **Interpretation**:
    - Time in seconds for exploration runs that completed the track
    - Includes exploration, so more variable than evaluation times
    - Higher than evaluation times (exploration slows agent down)
    
    **What to watch for**:
    - Should trend downward but be more noisy
    - Useful for tracking exploration progress
    - Large gap with eval times indicates exploration is working

**Race/explo_race_time_{status}_{map}**
    **Description**: Exploration race time for specific map and status.
    
    **Interpretation**:
    - Time in seconds for exploration runs
    - Includes all exploration runs
    - More variable due to exploration
    
    **What to watch for**:
    - More noisy than finished times
    - Useful for understanding exploration behavior

**Race/eval_race_finished_{status}_{map}**
    **Description**: Whether evaluation race finished (1.0) or not (0.0).
    
    **Interpretation**:
    - Binary metric: 1.0 = finished, 0.0 = did not finish
    - Shows completion rate for evaluation runs
    - Should approach 1.0 as agent learns
    
    **What to watch for**:
    - Should increase to 1.0 as training progresses
    - Persistent 0.0 values indicate agent is stuck

**Race/explo_race_finished_{status}_{map}**
    **Description**: Whether exploration race finished (1.0) or not (0.0).
    
    **Interpretation**:
    - Binary metric: 1.0 = finished, 0.0 = did not finish
    - Shows completion rate for exploration runs
    - May be lower than eval completion rate
    
    **What to watch for**:
    - Should increase over time
    - Lower than eval rate is normal (exploration can cause crashes)

**Race/race_time_ratio_{map}**
    **Description**: Ratio of race time to total rollout duration.
    
    **Interpretation**:
    - Shows efficiency: how much of rollout time was spent racing
    - Values < 1.0 indicate time spent on loading, setup, etc.
    - Higher values = more efficient data collection
    
    **What to watch for**:
    - Should be relatively stable
    - Very low values indicate system overhead issues

**Race/split_{map}_{i}**
    **Description**: Time (in seconds) between checkpoint i and checkpoint i+1.
    
    **Interpretation**:
    - Shows performance on specific track segments
    - Useful for identifying which parts of track are slow
    - Only logged for evaluation runs
    
    **What to watch for**:
    - Should decrease over time for all splits
    - Large differences between splits indicate difficult sections
    - Useful for track-specific analysis

**Race/eval_ratio_{status}_{reference}_{map}**
    **Description**: Race time as percentage of reference time (author or gold).
    
    **Interpretation**:
    - 100% = matched reference time
    - <100% = faster than reference (rare, indicates very good performance)
    - >100% = slower than reference
    - Useful for comparing to human performance
    
    **What to watch for**:
    - Should decrease over time (approaching 100% or below)
    - Only available if reference times are configured

**Race/eval_agg_ratio_{status}_{reference}**
    **Description**: Aggregated ratio across all maps.
    
    **Interpretation**:
    - Average ratio across all maps with reference times
    - Useful for multi-map training
    
    **What to watch for**:
    - Should decrease over time
    - Only available if reference times are configured

Gradient Metrics
---------------

**Gradients/norm_median**
    **Description**: Median gradient norm after clipping.
    
    **Interpretation**:
    - Should be stable (typically <30)
    - Shows typical gradient magnitude
    - Stable values indicate healthy training
    
    **What to watch for**:
    - Should remain relatively constant
    - Sudden changes may indicate learning issues

**Gradients/norm_q1, norm_q3**
    **Description**: 25th and 75th percentile gradient norms after clipping.
    
    **Interpretation**:
    - Shows distribution of gradient magnitudes
    - Q1-Q3 range shows typical gradient spread
    - Useful for understanding gradient stability
    
    **What to watch for**:
    - Should be relatively stable
    - Large spread may indicate unstable gradients

**Gradients/norm_d9, norm_d98**
    **Description**: 90th and 98th percentile gradient norms after clipping.
    
    **Interpretation**:
    - Shows tail of gradient distribution
    - Higher percentiles reveal occasional large gradients
    - Useful for detecting outliers
    
    **What to watch for**:
    - Should be stable
    - Large values may indicate occasional gradient spikes

**Gradients/norm_max**
    **Description**: Maximum gradient norm after clipping.
    
    **Interpretation**:
    - Maximum gradient magnitude encountered
    - After clipping, should be bounded by clip value
    - Typical range: 10-50
    
    **What to watch for**:
    - Should be relatively stable
    - Consistently hitting clip value may indicate need for higher clip threshold

**Gradients/norm_before_clip_median**
    **Description**: Median gradient norm BEFORE clipping.
    
    **Interpretation**:
    - Shows typical gradient magnitude before clipping
    - Should be similar to after-clip median if clipping is not active
    - Useful for understanding if clipping is necessary
    
    **What to watch for**:
    - Should be stable
    - Much higher than after-clip indicates clipping is active

**Gradients/norm_before_clip_max**
    **Description**: **CRITICAL METRIC!** Maximum gradient norm BEFORE clipping.
    
    **Interpretation**:
    - **Watch this closely!** Values >100 indicate gradient explosions
    - Should typically be <50
    - Sudden spikes indicate training instability
    - Used to detect gradient explosion before clipping fixes it
    
    **What to watch for**:
    - **Most important gradient metric**
    - Values >100 = gradient explosion (bad!)
    - Values >200 = severe gradient explosion
    - Sudden spikes require investigation
    - Should be relatively stable

**Gradients/norm_before_clip_q1, q3, d9, d98**
    **Description**: Percentile gradient norms before clipping.
    
    **Interpretation**:
    - Shows distribution of unclipped gradients
    - Useful for understanding gradient behavior before clipping
    - Similar interpretation to after-clip percentiles
    
    **What to watch for**:
    - Should be stable
    - Large values indicate need for gradient clipping

**Gradients/by_layer/{layer_name}/L2_median, q3, d9, max**
    **Description**: Per-layer L2 gradient norms (Euclidean norm).
    
    **Interpretation**:
    - Shows gradient magnitude for each network layer
    - Useful for debugging which layers have gradient issues
    - L2 norm = sqrt(sum of squared gradients)
    
    **What to watch for**:
    - Some layers may have naturally larger gradients
    - Sudden spikes in specific layers indicate layer-specific issues
    - Useful for identifying problematic layers

**Gradients/by_layer/{layer_name}/Linf_median, q3, d9, max**
    **Description**: Per-layer Linf gradient norms (maximum absolute value).
    
    **Interpretation**:
    - Shows maximum gradient component for each layer
    - Useful for detecting individual parameter issues
    - Linf norm = max absolute gradient value
    
    **What to watch for**:
    - Can reveal issues in specific parameters
    - Large Linf with small L2 indicates sparse large gradients

Performance Metrics
-------------------

**Performance/transitions_learned_per_second**
    **Description**: Training throughput - number of transitions processed per second.
    
    **Interpretation**:
    - Higher is better (faster training)
    - Typical range: 100-1000 transitions/second
    - Affected by GPU speed, batch size, and system performance
    
    **What to watch for**:
    - Should be relatively stable
    - Sudden decreases may indicate system issues
    - Higher values = faster training progress

**Performance/learner_percentage_training**
    **Description**: Percentage of time learner process spends on training (vs waiting).
    
    **Interpretation**:
    - Should be high (>70%) for efficient training
    - Low values indicate learner is waiting for data
    - High values indicate good data collection rate
    
    **What to watch for**:
    - Should be >70% for efficient training
    - <50% indicates workers are too slow
    - 100% indicates perfect balance (rare)

**Performance/learner_percentage_waiting_for_workers**
    **Description**: Percentage of time learner process waits for worker data.
    
    **Interpretation**:
    - Should be low (<20%) for efficient training
    - High values indicate workers are too slow
    - Indicates data collection bottleneck
    
    **What to watch for**:
    - Should be <20% for efficient training
    - >50% indicates severe data collection bottleneck
    - May need more worker instances or faster workers

**Performance/learner_percentage_testing**
    **Description**: Percentage of time spent on test batches.
    
    **Interpretation**:
    - Typically small (<10%)
    - Time spent evaluating on test buffer
    - Useful for monitoring but not critical
    
    **What to watch for**:
    - Should be relatively small
    - Large values may indicate too much testing

**Performance/instrumentation__answer_normal_step**
    **Description**: Time spent in normal step processing (microseconds).
    
    **Interpretation**:
    - Low-level performance metric
    - Shows TMInterface communication overhead
    - Useful for debugging performance issues
    
    **What to watch for**:
    - Should be relatively stable
    - Sudden increases may indicate system issues

**Performance/instrumentation__answer_action_step**
    **Description**: Time spent in action step processing (microseconds).
    
    **Interpretation**:
    - Low-level performance metric
    - Shows action processing time
    - Useful for debugging performance issues
    
    **What to watch for**:
    - Should be relatively stable
    - Affects overall training speed

**Performance/instrumentation__between_run_steps**
    **Description**: Time spent between runs (microseconds).
    
    **Interpretation**:
    - Low-level performance metric
    - Shows overhead between race restarts
    - Includes map loading, reset, etc.
    
    **What to watch for**:
    - Should be relatively stable
    - Large values indicate slow map loading

**Performance/instrumentation__grab_frame**
    **Description**: Time spent grabbing frame from game (microseconds).
    
    **Interpretation**:
    - Low-level performance metric
    - Shows frame capture overhead
    - Affected by game rendering speed
    
    **What to watch for**:
    - Should be relatively stable
    - Large values may indicate rendering issues

**Performance/instrumentation__convert_frame**
    **Description**: Time spent converting frame format (microseconds).
    
    **Interpretation**:
    - Low-level performance metric
    - Shows image processing overhead
    - Affected by image resolution and format
    
    **What to watch for**:
    - Should be relatively stable
    - Can be optimized by reducing resolution

**Performance/instrumentation__grab_floats**
    **Description**: Time spent grabbing float data from game (microseconds).
    
    **Interpretation**:
    - Low-level performance metric
    - Shows data extraction overhead
    - Includes speed, position, etc.
    
    **What to watch for**:
    - Should be relatively stable
    - Typically very fast

**Performance/instrumentation__exploration_policy**
    **Description**: Time spent in exploration policy computation (microseconds).
    
    **Interpretation**:
    - Low-level performance metric
    - Shows action selection overhead
    - Includes Q-value computation and exploration
    
    **What to watch for**:
    - Should be relatively stable
    - Affected by network inference speed

**Performance/instrumentation__request_inputs_and_speed**
    **Description**: Time spent requesting inputs and speed from game (microseconds).
    
    **Interpretation**:
    - Low-level performance metric
    - Shows game communication overhead
    - Includes TMInterface API calls
    
    **What to watch for**:
    - Should be relatively stable
    - Large values may indicate communication issues

**Performance/tmi_protection_cutoff**
    **Description**: Number of times TMI protection cutoff was triggered.
    
    **Interpretation**:
    - Safety mechanism to prevent infinite loops
    - High values indicate agent is getting stuck frequently
    - Should be low for well-trained agent
    
    **What to watch for**:
    - Should decrease as agent learns
    - High values indicate learning issues
    - May need to adjust timeout settings

**Performance/worker_time_in_rollout_percentage**
    **Description**: Percentage of rollout time spent in worker processing.
    
    **Interpretation**:
    - Shows worker efficiency
    - Higher values = workers are busy (good)
    - Lower values = workers are waiting (bad)
    
    **What to watch for**:
    - Should be relatively high (>80%)
    - Low values indicate worker bottlenecks

Buffer Metrics
--------------

**Buffer/size**
    **Description**: Current number of transitions in replay buffer.
    
    **Interpretation**:
    - Grows from 0 to max_size during training
    - More transitions = more diverse training data
    - Typical range: 20K to 200K
    
    **What to watch for**:
    - Should increase until reaching max_size
    - Should remain at max_size once full
    - Sudden decreases may indicate buffer issues

**Buffer/max_size**
    **Description**: Maximum capacity of replay buffer.
    
    **Interpretation**:
    - Set by memory_size_schedule
    - Larger buffers = more memory but better diversity
    - Typical range: 50K to 200K
    
    **What to watch for**:
    - Should remain constant unless schedule changes
    - Changes according to memory_size_schedule

**Buffer/number_times_single_memory_is_used_before_discard**
    **Description**: How many times each transition is used before being discarded.
    
    **Interpretation**:
    - Controls transition reuse
    - Higher values = transitions used more times
    - Balances data efficiency with freshness
    
    **What to watch for**:
    - Should remain constant unless explicitly changed
    - Typical values: 1-4

**Buffer/priorities_min, q1, mean, median, q3, d9, c98, max**
    **Description**: Priority statistics for prioritized experience replay.
    
    **Interpretation**:
    - Only available if using prioritized replay (prio_alpha > 0)
    - Higher priorities = more important transitions
    - Priorities based on TD error
    - Shows distribution of transition importance
    
    **What to watch for**:
    - Large spread indicates some transitions are much more important
    - Should be relatively stable
    - Not available if using uniform sampling (prio_alpha = 0)

Network Metrics
---------------

**Network/weights/{layer_name}/L2**
    **Description**: L2 norm (Euclidean norm) of layer weights.
    
    **Interpretation**:
    - Shows magnitude of weights in each layer
    - Useful for detecting weight growth or decay
    - Should be relatively stable during training
    
    **What to watch for**:
    - Sudden increases may indicate instability
    - Gradual growth is normal
    - Very large values may indicate numerical issues

**Network/optimizer/{layer_name}/adaptive_lr_L2**
    **Description**: L2 norm of per-parameter adaptive learning rates (Adam/RAdam).
    
    **Interpretation**:
    - Shows magnitude of adaptive learning rates
    - Adam/RAdam adjust learning rate per parameter
    - Higher values = larger effective learning rates
    
    **What to watch for**:
    - Should be relatively stable
    - Useful for understanding optimizer behavior

**Network/optimizer/{layer_name}/exp_avg_L2**
    **Description**: L2 norm of first moment estimate (Adam/RAdam).
    
    **Interpretation**:
    - First moment (moving average of gradients)
    - Used by Adam/RAdam for momentum
    - Should track gradient magnitudes
    
    **What to watch for**:
    - Should be relatively stable
    - Useful for debugging optimizer state

**Network/optimizer/{layer_name}/exp_avg_sq_L2**
    **Description**: L2 norm of second moment estimate (Adam/RAdam).
    
    **Interpretation**:
    - Second moment (moving average of squared gradients)
    - Used by Adam/RAdam for adaptive learning rates
    - Should track gradient variance
    
    **What to watch for**:
    - Should be relatively stable
    - Useful for debugging optimizer state

IQN Metrics
-----------

**IQN/quantile_std_action_{i}**
    **Description**: Standard deviation of quantile predictions for action {i}.
    
    **Interpretation**:
    - Measures uncertainty in Q-value estimates for each action
    - Higher values = more uncertainty (wider distribution)
    - Lower values = more confidence (narrower distribution)
    - IQN-specific metric (distributional RL)
    
    **What to watch for**:
    - Should decrease as agent learns (becomes more confident)
    - High values indicate high uncertainty
    - Useful for understanding model confidence
    - Different actions may have different uncertainty levels

Other Metrics
-------------

**alltime_min_ms_{map}**
    **Description**: All-time best race time (in milliseconds) for each map.
    
    **Interpretation**:
    - Best time ever achieved on each map
    - Only decreases (new records)
    - Most important performance metric alongside eval_race_time_robust
    
    **What to watch for**:
    - Should decrease over time (new records)
    - Plateaus indicate agent has reached current limit
    - Compare with reference times if available

**cumul_number_frames_played**
    **Description**: Cumulative number of frames processed during training.
    
    **Interpretation**:
    - Total training progress
    - Used as x-axis in most TensorBoard plots
    - Typical training: 1M to 50M+ frames
    
    **What to watch for**:
    - Should increase steadily
    - Used to track training progress

**cumul_number_batches_done**
    **Description**: Cumulative number of training batches processed.
    
    **Interpretation**:
    - Total number of gradient updates
    - Related to frames_played but depends on buffer fill rate
    - Higher = more learning steps
    
    **What to watch for**:
    - Should increase steadily
    - Ratio to frames_played shows learning frequency

**cumul_number_single_memories_used**
    **Description**: Cumulative number of transitions used for training.
    
    **Interpretation**:
    - Total transitions sampled from buffer
    - May be higher than frames_played due to reuse
    - Shows total learning experience
    
    **What to watch for**:
    - Should increase steadily
    - Higher than frames_played indicates transition reuse

**cumul_number_memories_generated**
    **Description**: Cumulative number of transitions generated from rollouts.
    
    **Interpretation**:
    - Total transitions added to buffer
    - Includes n-step transitions
    - Shows data collection progress
    
    **What to watch for**:
    - Should increase steadily
    - Should be less than memories_used (due to reuse)

**cumul_training_hours**
    **Description**: Cumulative training time in hours.
    
    **Interpretation**:
    - Total wall-clock time spent training
    - Useful for estimating training duration
    - Includes all overhead (not just GPU time)
    
    **What to watch for**:
    - Should increase steadily
    - Useful for planning training schedules

**cumul_number_target_network_updates**
    **Description**: Cumulative number of target network updates.
    
    **Interpretation**:
    - Number of times target network was updated
    - Target network updated less frequently than online network
    - Used for stable Q-learning
    
    **What to watch for**:
    - Should increase steadily
    - Frequency depends on update schedule

**times_summary** (Text)
    **Description**: Text summary of best times for all maps.
    
    **Interpretation**:
    - Human-readable summary of performance
    - Shows best times with timestamps
    - Updated every 5 minutes
    
    **What to watch for**:
    - Useful for quick overview
    - Shows new records with ** markers

Tips for Using TensorBoard
--------------------------

1. **Filtering**: Use the search box in TensorBoard to filter metrics by prefix (e.g., type "Gradients/" to see all gradient metrics)

2. **Custom Scalars**: The "Custom Scalars" tab has pre-configured layouts for key metrics grouped together

3. **Smoothing**: Use the smoothing slider to reduce noise in plots (helpful for noisy metrics)

4. **Comparison**: Load multiple runs to compare different training configurations

5. **Key Metrics to Monitor**:
   - ``Race/eval_race_time_robust`` - Primary performance metric
   - ``RL/avg_Q`` - Learning progress indicator
   - ``Gradients/norm_before_clip_max`` - Training stability
   - ``Training/loss`` - Learning quality
   - ``Performance/transitions_learned_per_second`` - Training efficiency

6. **Early Training** (0-3M frames):
   - Watch ``RL/single_zone_reached`` - should increase to 1.0
   - Watch ``RL/avg_Q`` - may decrease then increase
   - Watch ``Training/loss`` - may increase (normal!)

7. **Mid Training** (3-10M frames):
   - Watch ``Race/eval_race_time_robust`` - should decrease
   - Watch ``RL/avg_Q`` - should increase
   - Watch ``Training/loss`` - should stabilize

8. **Late Training** (10M+ frames):
   - Watch ``Race/eval_race_time_robust`` - slow improvements
   - Watch for plateaus - may need longer training or hyperparameter changes
