Experiment: Multi-action Offset Training (A01_as20_long v3 series)
===================================================================

Experiment Overview
-------------------

This experiment evaluates a new RL training mode where the agent learns with **multi-action time offsets**.

In multi-action mode (``rl_action_offsets_ms`` has more than one value), the policy makes a single forward pass and predicts ``N`` actions for offsets ``0, 10, 20, ...`` ms. The rollout then applies these actions on a 10 ms step period, and a replay transition corresponds to one **decision block** (N actions + aggregated reward over N steps). Exploration can be configured as either:

- ``multi_action_exploration: per_action``: epsilon is sampled independently per action inside the block.
- ``multi_action_exploration: per_block``: one epsilon draw applies to the whole block (either all greedy or all random).

Because a decision is made per block (N actions spanning multiple 10 ms steps), multi-action lookahead is applied at a lower decision frequency than single-action training, and in ``per_block`` mode the fully-random blocks become increasingly rare as epsilon decays.

Runs compared on map ``A01``:

- ``A01_as20_long_v3``: multi-action enabled, ``multi_action_exploration`` default (per_action), ``global_schedule_speed = 1``, no BC head pretrain.
- ``A01_as20_long_v3.1``: same multi-action setup, ``multi_action_exploration = per_block`` and faster schedules (``global_schedule_speed = 4``).
- ``A01_as20_long_v3.1_pretrained_bc``: same as v3.1 but initializes RL from BC heads with ``pretrain_bc_heads_path: output/ptretrain/bc/v5_multi_offset``.

Notes on why ``global_schedule_speed = 4``: this choice is based on the earlier ablation in ``docs/source/experiments/global_schedule_speed.rst`` (A01 long v2 series). The best saved A01 time is ``alltime_min_ms['A01'] = 24150`` (i.e. ~``24.15s``) in ``save\\A01_as20_long_v2``; in TensorBoard it shows up in the suffixed continuation run ``tensorboard\\A01_as20_long_v2_3``.

For “longest run” comparison (almost 100M+ training steps): ``A01_as20_long`` (single-map A01, trained with ``tensorboard_suffix_schedule`` up to ~150M steps).

Results
-------

Important: run durations differ, so the findings are primarily **by relative time**. Where step-based comparison is available, it is reported **by steps** (common overlap).

Key findings

- Multi-action schedule speedup (v3 -> v3.1) improves early learning: at 120 min, ``alltime_min_ms_A01`` is still at the maximum placeholder value for ``A01_as20_long_v3`` but has already dropped to ``27.230s`` for ``A01_as20_long_v3.1``.
- BC head pretraining (v3.1 -> v3.1_pretrained_bc) improves both peak time and reliability (finish rate), especially from the 3rd to 4th hour onward:
  - At 180 min: ``24.730s`` best eval time and ``56%`` eval finish rate for ``v3.1_pretrained_bc`` vs ``25.490s`` and ``40%`` for ``v3.1``.
  - At 240 min: ``24.570s`` and ``60%`` vs ``24.850s`` and ``46%``.
  - By the end of the shared window (up to ~1680 min): ``alltime_min_ms_A01`` reaches ``24.260s`` for ``v3.1_pretrained_bc`` vs ``24.410s`` for ``v3.1``.
- By steps (common overlap; step checkpoints shown by the analysis script):
  - At 20M steps: eval best time ``24.570s`` and finish rate ``59%`` for ``v3.1_pretrained_bc`` vs ``24.850s`` and ``45%`` for ``v3.1``.
  - At 80M steps: eval best time ``24.260s`` and finish rate ``73%`` for ``v3.1_pretrained_bc`` vs ``24.410s`` and ``67%`` for ``v3.1``.
- Comparison with the longest run ``A01_as20_long``:
  - Early (120 min): the longest run is ahead in peak time (``24.610s`` best vs ``25.580s`` for ``v3.1_pretrained_bc``) and in finish rate (``55%`` vs ``44%``).
  - Later (end of the shared window around 490 min): ``v3.1_pretrained_bc`` slightly overtakes the longest run in best time: ``alltime_min_ms_A01`` is ``24.400s`` vs ``24.510s``.
  - Note on the final number: the full ``v3.1_pretrained_bc`` run continues beyond the shared window; in your run log it reaches ``BEST TIMES: A01 24.26s``. That ``24.26s`` is therefore not visible in the overlap-with-``A01_as20_long`` comparison table.
  - Step-based overlap with ``A01_as20_long`` is limited (common overlap only reaches ~19.2M steps). At that shared step, ``v3.1_pretrained_bc`` has better best time (``24.570s`` vs ``24.510s``) but lower finish rate (``59%`` vs ``71%``). The main advantage shows up as training continues in the longer v3.1_pretrained run.
- Direct check requested: ``A01_as20_long_v3.1`` vs ``A01_as20_long_v2`` (both merged across ``run``, ``run_2``, ``run_3``):
  - By relative time, ``v2`` is consistently ahead on A01 best time over the shared window (e.g. 120 min: ``24.550s`` vs ``27.230s``; 240 min: ``24.430s`` vs ``24.850s``; 1680 min: ``24.150s`` vs ``24.410s``).
  - By steps (1M checkpoints), ``v2`` also stays ahead (e.g. 20M: ``24.460s`` vs ``24.850s``; 40M: ``24.300s`` vs ``24.470s``; 80M: ``24.200s`` vs ``24.410s``).
  - Final saved best (from ``save/<run>/accumulated_stats.joblib``): ``v2 = 24.150s`` (``24150`` ms), ``v3.1 = 24.410s`` (``24410`` ms).

Run Analysis
------------

- ``A01_as20_long_v3``: multi-action enabled, ``global_schedule_speed = 1``; trained ~1403 min.
- ``A01_as20_long_v3.1``: multi-action enabled, ``multi_action_exploration = per_block`` and ``global_schedule_speed = 4``; trained ~1681 min.
- ``A01_as20_long_v3.1_pretrained_bc``: v3.1 + BC heads from ``output/ptretrain/bc/v5_multi_offset``; trained ~1818 min.
- ``A01_as20_long`` (longest reference): single-map A01 long training with ``tensorboard_suffix_schedule`` up to ~150M steps; trained ~495 min.

Configuration Changes
---------------------

The runs share the same multi-action offsets:

- ``environment.rl_action_offsets_ms = [0, 10, 20, 30, 40]`` (N=5 actions per decision block; applied on 10 ms rollout cadence).

Differences:

- ``A01_as20_long_v3``:
  - ``training.global_schedule_speed = 1``
  - ``exploration.multi_action_exploration`` uses default (per_action)
  - ``training.pretrain_bc_heads_path = null``
- ``A01_as20_long_v3.1``:
  - ``training.global_schedule_speed = 4``
  - ``exploration.multi_action_exploration = per_block``
  - ``training.pretrain_bc_heads_path = null``
- ``A01_as20_long_v3.1_pretrained_bc``:
  - ``training.global_schedule_speed = 4``
  - ``exploration.multi_action_exploration = per_block``
  - ``training.pretrain_bc_heads_path = output/ptretrain/bc/v5_multi_offset``

Hardware
--------

- GPU: not extracted here (see individual run logs).
- Parallel instances: ``gpu_collectors_count = 8`` for the v3.1_pretrained_bc run.

Conclusions
-----------

- Multi-action offset training works and is sensitive to schedule speed and exploration granularity:
  - Going from v3 to v3.1 (faster schedule + per_block exploration) improves early learning and reaches the ~24.8-24.5 s range quickly.
- Pretraining the RL heads from the multi-offset BC run (v5_multi_offset) provides a durable benefit:
  - Higher peak time and higher finish rate at the same step levels (20M -> 80M).
  - Best time improvements continue over a long window (up to ~1680 min).
  - Likely reason: better temporal/action mapping between pretrain and RL. In BC pretrain, the model predicts 5 offset actions at 10 ms spacing; in multi-action RL, one decision is taken every ~50 ms and outputs a 5-action block. This alignment is much closer than the older single-action RL setup (one action every ~50 ms), where pretrain-to-RL mapping was weak.
- Against the longest existing baseline (``A01_as20_long`` trained up to ~150M steps), the offset+pretrained agent has a slower start but catches up later and slightly improves the final peak time in the shared window.

Recommendations
---------------

- If you adopt multi-action offsets, try ``global_schedule_speed = 4`` and ``multi_action_exploration = per_block`` as the first tuning pair.
- If you can afford it, initialize from a multi-offset BC run (here: ``v5_multi_offset``) to improve long-run finish rate and to raise the achievable best time at large step counts.
- For step-based comparisons against other “longest run” baselines, expect limited overlap if scalar/race tags stop being logged at different step ranges; use relative-time plots as the primary comparison in that case.

Analysis Tools
---------------

- Compare v3 variants (relative time + by steps; recommended checkpoint resolution: ``--step_interval 1000000``):

  ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long_v3 A01_as20_long_v3.1 A01_as20_long_v3.1_pretrained_bc --interval 60 --step_interval 1000000``

- Compare against the longest baseline ``A01_as20_long``:

  ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long_v3.1_pretrained_bc A01_as20_long --interval 10 --step_interval 1000000``

