============
User FAQ
============

General Questions
-----------------

**Q: What is distributional RL and why IQN?**

A: Distributional RL models the full distribution of returns rather than just expected values. IQN (Implicit Quantile Networks) is particularly effective for TrackMania because:

- Handles stochastic environments well
- Better exploration through uncertainty estimation
- More stable learning than standard DQN

**Q: How long does training take?**

A: Depends on the map complexity:

- Simple maps (A01-A05): 2-5M frames (~2-4 hours with 4 collectors)
- Medium maps (ESL-Hockolicious): 5-10M frames (~4-8 hours)
- Complex maps (map5, Endurance): 15-30M frames (~12-24 hours)

**Q: What hardware do I need?**

A: Minimum recommended:

- CPU: 4+ cores
- GPU: NVIDIA with 6GB+ VRAM (GTX 1060 or better)
- RAM: 20GB+ (16GB system + 4GB+ VRAM)
- Storage: 10GB+ for saves and tensorboard logs

**Q: Can I use AMD GPUs?**

A: PyTorch supports AMD GPUs via ROCm, but we haven't tested extensively. NVIDIA CUDA is recommended and well-tested.

**Q: Why do I need multiple game instances?**

A: Parallel game instances speed up data collection significantly. More instances = faster training, but requires more RAM/CPU.

Performance
-----------

**Q: How many game instances should I use?**

A: Start with 2-4 and monitor:

- CPU usage: should be 70-90% (not maxed)
- RAM usage: ~2GB per instance
- GPU usage: 80-95% is optimal
- Training throughput: batches per minute

Increase instances until you hit a bottleneck (usually CPU or RAM).

**Q: Linux vs Windows performance?**

A: On a system with Ryzen 7 5700G, 64GB RAM, RTX 4070 Ti:

- Linux: ~58% faster training
- Linux sweet spot: 4 collectors
- Windows sweet spot: 2 collectors

Linux is recommended for serious training due to better wine performance with DXVK.

**Q: My training is slow, what can I do?**

A: Optimization checklist:

1. Increase ``gpu_collectors_count`` (if RAM/CPU allows)
2. Increase ``running_speed`` to 160-200x
3. Lower game resolution and graphics quality
4. Reduce ``batch_size`` for more frequent updates
5. Disable visualization options in ``performance_config.py``
6. Close unnecessary background applications

Training
--------

**Q: My agent isn't learning / getting stuck**

A: Common causes and fixes:

- **Bad reference line**: Verify VCP file covers entire track
- **Too low exploration**: Increase epsilon in early training
- **Wrong map path**: Check map loads correctly
- **Timeout too short**: Increase ``cutoff_rollout_if_no_vcp_passed_within_duration_ms``
- **Reward imbalance**: Check time penalty vs progress reward ratio

**Q: Can I train on multiple maps simultaneously?**

A: Yes! Edit ``map_cycle_config.py`` to alternate between maps:

.. code-block:: python

    map_cycle = [
        repeat(("map1", "Map1.Gbx", "map1_0.5m_cl.npy", True, True), 4),
        repeat(("map2", "Map2.Gbx", "map2_0.5m_cl.npy", True, True), 4),
    ]

**Q: How do I resume training from a checkpoint?**

A: Checkpoints are saved automatically in ``save/{run_name}/``. To resume:

1. Ensure ``.torch`` files exist in ``save/{run_name}/``
2. Keep the same ``run_name`` in ``map_cycle_config.py``
3. Run ``python scripts/train.py``

Training will load the checkpoint automatically.

**Q: Can I change hyperparameters during training?**

A: Yes! Edit ``config_files/config_copy.py`` (not ``config.py``) and save. Changes apply at next reload interval (every few minutes) without losing the replay buffer.

⚠️ **Don't change**: Network architecture, input dimensions, action space - these require restart.

**Q: What's the difference between exploration and evaluation runs?**

A:

- **Exploration** (``is_explo=True``): Agent uses epsilon-greedy + Boltzmann exploration to discover new strategies
- **Evaluation** (``is_explo=False``): Agent plays greedily to measure current skill level

Typical ratio: 4 exploration runs per 1 evaluation run.

Maps & Replays
--------------

**Q: How do I create virtual checkpoints for a new map?**

A:

1. Drive the track manually and save a replay
2. Place replay in ``Documents/TrackMania/Tracks/Replays/``
3. Run: ``python scripts/tools/gbx_to_vcp.py "path/to/replay.Replay.Gbx"``
4. VCP file is saved to ``maps/`` folder (e.g., ``MapName_0.5m_cl.npy``)
5. Update ``map_cycle_config.py`` to reference the new VCP file

**Q: Does the reference line need to be fast?**

A: No! A slow centerline drive is perfectly fine. The reference line is only used to:

- Track progress along the track
- Define forward direction
- Provide waypoint lookahead to the agent

The agent will learn to drive faster than the reference line.

**Q: Can I use someone else's world record replay?**

A: Yes, but centerline is often better because:

- WR lines may use advanced techniques (wallbangs, cuts)
- Agent might struggle to discover these early in training
- Centerline provides more uniform progress tracking

**Q: My map won't load / stuck on loading screen**

A: Check:

- Map file is NOT in OneDrive/cloud storage
- Map path in config matches actual file location
- Map file is valid (``.Challenge.Gbx`` format)
- Game is in windowed mode (not minimized)

**Q: How do I replay agent runs?**

A: Best runs are saved in ``save/{run_name}/best_runs/{map}_{time}/``:

1. Copy ``.inputs`` file to ``Documents/TMInterface/Scripts/``
2. Open game and load the map
3. Open TMInterface console (F12)
4. Type: ``load filename.inputs``
5. Press Enter to play
6. Save replay if desired

Configuration
-------------

**Q: Where do I find all configuration options?**

A: Configuration is split across modules in ``config_files/``:

- Quick reference: Inline comments in each module
- Full docs: ``docs/source/configuration_guide.rst``
- Overview: ``config_files/README.md``

**Q: What's the recommended configuration for my first training?**

A: Default configuration is pre-tuned for ESL-Hockolicious. For other maps:

- **Easier than Hocko**: Set ``global_schedule_speed = 0.8``
- **Harder than Hocko**: Set ``global_schedule_speed = 1.5``
- **Very technical maps**: Reduce ``tm_engine_step_per_action`` to 3-4

**Q: What does global_schedule_speed do?**

A: Multiplier for all frame-based schedules:

- ``1.0``: Normal speed (default)
- ``0.8``: 20% faster schedule (for easier maps)
- ``1.5``: 50% slower schedule (for harder maps)

This uniformly speeds up/slows down learning rate decay, epsilon decay, buffer growth, etc.

Monitoring
----------

**Q: What metrics should I watch in TensorBoard?**

A: Key graphs in "Custom Scalars" tab:

- **avg_Q**: Expected reward (should increase after initial drop)
- **single_zones_reached**: How far agent drives (% of track completed)
- **eval_race_time_robust**: Best evaluation times
- **loss**: Training loss (can increase early - normal for RL)

**Q: Why is my loss increasing?**

A: In RL, loss increasing early in training is normal and expected! It means:

- Agent is discovering the environment
- Identifying inconsistencies in its value estimates
- Learning is progressing correctly

Loss should stabilize or decrease after ~1-2M frames.

**Q: My agent finishes the track but times aren't improving**

A: This is the "optimization phase" (after ~3-5M frames):

- Progress is slower now
- Agent is refining strategy details
- Continue training for 10-20M more frames
- Consider enabling shaped rewards for faster progress

Technical Issues
----------------

**Q: FileNotFoundError: Python_Link.as**

A: Copy the plugin:

.. code-block:: bash

    # Windows
    New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Documents\TMInterface\Plugins"
    Copy-Item "trackmania_rl\tmi_interaction\Python_Link.as" "$env:USERPROFILE\Documents\TMInterface\Plugins\"

**Q: Game stuck on login screen**

A: TMNF account must be an **online account**. Create one through the game launcher without TMInterface.

**Q: CUDA out of memory**

A: Reduce memory usage:

- Decrease ``batch_size`` in ``training_config.py``
- Decrease ``memory_size_schedule`` in ``memory_config.py``
- Reduce ``gpu_collectors_count`` in ``performance_config.py``
- Lower image resolution in ``environment_config.py``

**Q: Game crashes / TMInterface connection lost**

A: Common fixes:

- Increase ``timeout_during_run_ms`` in ``environment_config.py``
- Reduce ``running_speed`` (< 200x)
- Check TMLoader profile is correctly configured
- Verify no firewall blocking TMInterface
- Restart game instances (automatic every 12 hours)

Contributing
------------

**Q: How can I contribute to this project?**

A: Contributions welcome:

- Report issues and bugs
- Share your training results
- Improve documentation
- Add new features or algorithms
- Optimize performance

See ``DEVELOPMENT.md`` for development setup.

**Q: Can I share my trained models?**

A: Yes! Model weights are in ``save/{run_name}/weights*.torch``. Share with the community to help others.

⚠️ **Important**: All AI runs are Tool Assisted and must NOT be submitted to official leaderboards.

Additional Resources
--------------------

- **Documentation**: ``docs/source/``
- **Configuration Guide**: ``docs/source/configuration_guide.rst``
- **Original Linesight**: https://github.com/pb4git/linesight
- **TMInterface**: https://donadigo.com/tminterface/
- **TMNF Exchange**: https://tmnf.exchange/
