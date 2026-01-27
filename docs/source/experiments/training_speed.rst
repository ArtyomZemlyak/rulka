Experiment: Batch Size and Running Speed
========================================

Experiment Overview
-------------------

This document covers experiments on **batch_size** and **running_speed**. Baseline: **uni_5** (batch 2048, speed 160×).

**Direction 1 — Larger batch:** Increasing batch to 8192 (uni_6) **worsened** convergence and loss; RTX 5090 hit a performance ceiling.

**Direction 2 — Smaller batch + faster speed:** Reducing batch to 512 and speed to 512× (uni_7) **improved** convergence, loss, and GPU utilization. Follow-up runs uni_8 and uni_9 (even smaller batch, higher speed) **reversed** those gains: over-tuning hurt. Conclusion: a **golden middle** between batch_size and running_speed is needed; uni_7 (512/512) is in that range.

Results
-------

**Key Findings:**

- **Larger batch (uni_6)**: Slower convergence, ~10× higher loss, more negative Q-values; GPU % unchanged (~53%). Do not go above 2048 for this setup.
- **Smaller batch + faster speed (uni_7)**: ~40% fewer steps to similar performance, 74% lower loss, better Q-values, GPU 78.4% (vs 53.8%). Best trade-off observed.
- **Over-tuning (uni_8, uni_9)**: Worse best times and Q-values than uni_7 despite higher GPU %. Golden middle is around 512/512; extremes (e.g. 64/1024) degrade quality.

Run Analysis
------------

All runs use the same hardware (RTX 5090, 8 collectors). TensorBoard logs: ``tensorboard\uni_<N>``.

- **uni_5**: Baseline — batch_size = 2048, running_speed = 160 (best Hock 24.560s, A01 24.560s)
- **uni_6**: Larger batch 8192, speed 160 — **worse** convergence (Hock 24.770s), loss 9.84× higher
- **uni_6_2**: Continuation of uni_6 (same config)
- **uni_7**: Smaller batch 512, speed 512 — **best trade-off** (Hock 24.630s, A01 24.580s; 40% fewer steps, 74% lower loss)
- **uni_8**: Further smaller batch + higher speed — **worse** than uni_7 (Hock 25.180s, Q ≈ -1.16)
- **uni_9**: Extreme (e.g. batch 64, speed 1024) — **worse** than uni_7 (Hock 25.100s, Q ≈ -1.10)

To reproduce metrics: activate venv, then from project root run ``python scripts/analyze_experiment.py uni_5 uni_6 uni_7 uni_8 uni_9`` (or ``--logdir "<path_to_tensorboard>"`` from another directory).

Detailed TensorBoard Metrics Analysis
-------------------------------------

Metrics below are from TensorBoard logs (``tensorboard\uni_<N>``). Baseline is uni_5 (2048 batch, 160 speed).

**Key metrics** (aligned with ``docs/source/tensorboard_metrics.rst``): ``Race/eval_race_time_robust_*`` and ``alltime_min_ms_{map}`` (performance), ``Training/loss`` (learning quality), ``RL/avg_Q_*`` (learning progress), ``Performance/learner_percentage_training`` (GPU efficiency). When comparing setups, also consider ``Performance/transitions_learned_per_second`` (throughput) and ``Gradients/norm_before_clip_max`` (stability; spikes >100 = gradient explosion). For interpretation and “what to watch for”, see that file.

Larger batch: uni_5 vs uni_6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Hock** (``alltime_min_ms_hock`` / ``Race/eval_race_time_robust_trained_hock``): uni_5 best 24.560s at step 5,507,000; uni_6 best 24.770s at step 5,806,096; uni_6_2 best 24.760s at step 9,291,263. Larger batch → **slower convergence**.
- **Training/loss**: uni_5 300.96 at step 5,871,020; uni_6 2,962.12 at step 5,989,582 → **9.84× higher** in uni_6.
- **``RL/avg_Q_*``**: uni_5 -0.5848; uni_6 -2.1154 → **more negative** in uni_6 (worse learning).
- **Performance/learner_percentage_training**: uni_5 53.8%; uni_6 52.6% → no gain; **performance ceiling**.

Smaller batch + faster speed: uni_5 vs uni_7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Hock** (``alltime_min_ms_hock``): uni_5 best 24.560s at step 3,658,108; uni_7 best 24.630s at step 2,192,167 → **~40% fewer steps** to similar performance.
- **A01** (``alltime_min_ms_A01``): uni_5 best 24.560s at step 4,391,205; uni_7 best 24.580s at step 2,862,282 → **~35% fewer steps**.
- **Training/loss**: uni_5 300.96; uni_7 77.72 → **0.26× (74% lower)**.
- **``RL/avg_Q_*``**: uni_5 -0.5848; uni_7 -0.5204 → **better** in uni_7.
- **Performance/learner_percentage_training**: uni_5 53.8%; uni_7 78.4% → **+24.6%**.

Over-tuning: uni_7 vs uni_8 vs uni_9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Hock** (``alltime_min_ms_hock``): uni_7 24.630s; uni_8 25.180s (step 1.39M); uni_9 25.100s (step 2.37M) — both **worse** than uni_7.
- **A01** (``alltime_min_ms_A01``): uni_7 24.580s; uni_8 25.180s (step 1.28M); uni_9 25.100s (step 2.30M) — **worse** than uni_7.
- **Training/loss**: uni_7 77.72; uni_8 52.41; uni_9 29.07 — lower raw loss but **worse** race times and Q-values; loss alone is misleading.
- **``RL/avg_Q_*``**: uni_7 -0.5204; uni_8 -1.1609; uni_9 -1.0951 — **more negative** ⇒ weaker learning.
- **Performance/learner_percentage_training**: uni_7 78.4%; uni_8 80.4%; uni_9 82.1% — higher % did **not** yield better policies.

Other metrics (from ``docs/source/tensorboard_metrics.rst``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When comparing batch/speed setups, also check in TensorBoard or via ``scripts/extract_tensorboard_data.py``:

- **Performance/transitions_learned_per_second** — training throughput; higher is better; reflects efficiency of the pipeline.
- **Gradients/norm_before_clip_max** — training stability; spikes >100 indicate gradient explosion; should stay relatively stable. Compare runs to ensure no setup introduces instability.

Full interpretation and “what to watch for” for all metrics: ``docs/source/tensorboard_metrics.rst``.

Configuration Changes
----------------------

**Training** (``config_files/training_config.py``): batch_size was varied (2048 baseline, 8192 in uni_6, 512 in uni_7, smaller in uni_8/uni_9). **Recommended:** 512.

**Performance** (``config_files/performance_config.py``): running_speed was varied (160 baseline, 512 in uni_7, higher in uni_8/uni_9). **Recommended:** 512.

.. code-block:: python

   # Recommended (uni_7-style)
   batch_size = 512   # not 2048, not 8192, not 64
   running_speed = 512  # not 160, not 1024

Hardware
--------

- **GPU**: RTX 5090
- **Parallel instances**: 8 collectors
- **System**: Same across all runs

Conclusions
-----------

1. **Larger batch (8192) hurts:** Fewer gradient updates per step, worse loss and convergence; GPU ceiling ~53%. Keep batch ≤ 2048 or use 512 for the “small batch + fast speed” regime.

2. **Smaller batch (512) + faster speed (512×) helps:** Fewer steps to similar performance, much lower loss, better Q-values, ~78% GPU. Best observed trade-off.

3. **Over-tuning (smaller batch + higher speed) hurts again:** uni_8/uni_9 did not reach uni_7-level times or Q-values. There is a **golden middle**; uni_7 (512/512) is in it; extremes (e.g. 64/1024) degrade sample efficiency and optimization.

4. **Trade-offs:** Smaller batches need more frequent sync; very high speed can affect physics; going beyond the sweet spot (batch/speed) degrades performance.

Recommendations
---------------

- **Use batch_size = 512** and **running_speed = 512** as the default “golden middle” for this setup (IQN, RTX 5090).
- **Avoid** very small batch (e.g. 64) with very high speed (e.g. 1024); it slows training and convergence.
- **Do not** increase batch beyond 2048 (or 512 if using the fast-speed regime) without evidence of benefit.
- **Monitor** training loss, map times, and Q-values; GPU % alone is not sufficient.
- **Tune** batch_size and running_speed together; uni_7 (512/512) is a proven working point.

**When to change:**

- Larger batches (1024–2048): if GPU memory or sync overhead is the bottleneck.
- Slower speed (160–200): if physics or stability is critical.
- Keep 512/512 unless you have a clear reason to move away.

**Analysis Tools:**

- Activate venv (Windows): ``.\.venv\Scripts\activate``
- From project root: ``python scripts/analyze_experiment.py uni_5 uni_6 uni_7 uni_8 uni_9`` (logs in ``tensorboard\uni_<N>``)
- From another directory: ``--logdir "C:\...\rulka\tensorboard"``
- ``scripts/extract_tensorboard_data.py`` for custom metrics (e.g. ``Gradients/norm_before_clip_max``, ``Performance/transitions_learned_per_second``)
- **Key metrics** (see ``docs/source/tensorboard_metrics.rst``): ``Training/loss``, ``Race/eval_race_time_robust_*``, ``alltime_min_ms_{map}``, ``RL/avg_Q_*``, ``Performance/learner_percentage_training``, ``Performance/transitions_learned_per_second``, ``Gradients/norm_before_clip_max`` (stability)
