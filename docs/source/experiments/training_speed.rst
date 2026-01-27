Experiment: Batch Size and Running Speed
========================================

Experiment Overview
-------------------

This document covers experiments on **batch_size** and **running_speed**. Baseline: **uni_5** (batch 2048, speed 160×).

**Direction 1 — Larger batch:** Increasing batch to 8192 (uni_6) **worsened** convergence and loss; RTX 5090 hit a performance ceiling.

**Direction 2 — Smaller batch + faster speed:** Reducing batch to 512 and speed to 512× (uni_7) **improved** convergence, loss, and GPU utilization. Follow-up runs uni_8 and uni_9 (even smaller batch, higher speed) **reversed** those gains: over-tuning hurt. Conclusion: a **golden middle** between batch_size and running_speed is needed; uni_7 (512/512) is in that range.

Results
-------

**Important:** Experiments had different durations (uni_5 ~160 min, uni_7 ~86 min, etc.), so comparing by “last value” is meaningless. All findings below are based on **relative time** — minutes from run start; metrics are compared at the same moments (5, 10, 20, … min), up to when the shortest compared run ends.

**Key Findings (by relative time):**

- **Larger batch (uni_6)**: At the same minutes from start — worse best times (at 160 min: Hock 24.77s vs 24.56s for uni_5), loss ~10× higher (2980 vs 301), GPU % no gain (~52% vs ~54%). Do not go above 2048.
- **Smaller batch + faster speed (uni_7)**: At the same minutes uni_7 reaches good times faster (at 60 min Hock 24.63s vs 24.69s for uni_5), loss much lower, Q better, GPU ~78% vs ~53%. Best trade-off.
- **Over-tuning (uni_8, uni_9)**: On the common window up to 70 min best times and Q are with uni_7 (at 70 min Hock 24.63s vs 25.18s / 25.25s); uni_8/9 have higher GPU %, but policies are worse.
- **Fewer collectors (uni_10, gpu_collectors_count=4)**: At the same 512/512 by relative time uni_10 is sometimes slightly faster to good times (at 20–30 min slightly better), by 70 min uni_7 is slightly ahead (24.63s vs 24.72s). **Main point** — throughput and GPU %: ~59% vs ~79%; fewer ops per unit time. For maximum training speed by wall clock, 8 collectors are better.

Run Analysis
------------

All runs use the same hardware (RTX 5090). uni_5–uni_9: 8 collectors; uni_10: 4 collectors. TensorBoard logs: ``tensorboard\uni_<N>``. **Durations (relative time):** uni_5 ~160 min, uni_6 ~165 min, uni_7 ~86 min, uni_8 ~76 min, uni_9 ~194 min, uni_10 ~71 min.

- **uni_5**: Baseline — batch_size = 2048, running_speed = 160, ~160 min
- **uni_6**: Larger batch 8192, speed 160, ~165 min — by relative time worse Hock and loss
- **uni_6_2**: Continuation of uni_6 (same config)
- **uni_7**: Smaller batch 512, speed 512, 8 collectors, ~86 min — **best trade-off** by relative time
- **uni_8**: Further smaller batch + higher speed, ~76 min — worse than uni_7 on times and Q on common window
- **uni_9**: Extreme (e.g. batch 64, speed 1024), ~194 min — worse than uni_7 on common window up to 70 min
- **uni_10**: Same as uni_7 (batch 512, speed 512) but **gpu_collectors_count = 4**, ~71 min — sometimes slightly faster on time early on, by 70 min slightly worse; **throughput and GPU % lower** (~59% vs ~79%).

To reproduce metrics by **relative time**: ``python scripts/analyze_experiment_by_relative_time.py uni_5 uni_7 --interval 5`` (or ``--logdir "<path_to_tensorboard>"``). For last-value comparison: ``python scripts/analyze_experiment.py uni_5 uni_6 uni_7 ...``.

Detailed TensorBoard Metrics Analysis
-------------------------------------

Metrics below are from TensorBoard logs (``tensorboard\uni_<N>``). Baseline is uni_5 (2048 batch, 160 speed).

**Methodology — Relative time:** Experiments had different durations (see Run Analysis), so comparing by “last value” is invalid. Metrics are aligned by **relative time** — minutes from run start. Values are taken at checkpoints 5, 10, 15, 20, … min; comparison runs only until the shortest compared run is still going. For race times at each checkpoint — **best so far** by that moment; for loss / Q / GPU % — **last value at that moment**. Tables: ``python scripts/analyze_experiment_by_relative_time.py <run1> <run2> [--interval 5]``.

**Key metrics** (aligned with ``docs/source/tensorboard_metrics.rst``): ``Race/eval_race_time_robust_*`` and ``alltime_min_ms_{map}`` (performance), ``Training/loss`` (learning quality), ``RL/avg_Q_*`` (learning progress), ``Performance/learner_percentage_training`` (GPU efficiency). Also: ``Performance/transitions_learned_per_second``, ``Gradients/norm_before_clip_max``. For interpretation see that file.

Larger batch: uni_5 vs uni_6 (relative time, common window up to 160 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Hock** (``alltime_min_ms_hock``): at 160 min — uni_5 24.560s, uni_6 24.770s; uni_6 **worse** throughout the window.
- **A01** (``alltime_min_ms_A01``): at 160 min — uni_5 24.560s, uni_6 24.770s.
- **Training/loss**: at 160 min — uni_5 301, uni_6 2979 → **~10× higher** in uni_6.
- **``RL/avg_Q_*``**: at 160 min — uni_5 -0.79; uni_6 -1.23 (uni_6 more negative over training).
- **Performance/learner_percentage_training**: ~53% uni_5, ~52% uni_6 → **no gain**; performance ceiling.

Smaller batch + faster speed: uni_5 vs uni_7 (relative time, common window up to 85 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Hock** (``alltime_min_ms_hock``): at 60 min — uni_5 24.69s, uni_7 24.63s; at 85 min — uni_5 24.58s, uni_7 24.63s. Uni_7 reaches good times faster during the run.
- **A01** (``alltime_min_ms_A01``): at 85 min — uni_5 24.58s, uni_7 24.58s (tie).
- **Training/loss**: at 85 min — uni_5 355, uni_7 77.7 → **much lower** in uni_7 at all checkpoints.
- **``RL/avg_Q_*``**: at 85 min — uni_5 -0.60, uni_7 -0.41 → **better** in uni_7.
- **Performance/learner_percentage_training**: ~53% uni_5, ~78% uni_7 over the window → **+25%** for uni_7.

Over-tuning: uni_7 vs uni_8 vs uni_9 (relative time, common window up to 70 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Hock** (``alltime_min_ms_hock``): at 70 min — uni_7 24.63s, uni_8 25.18s, uni_9 25.25s → **uni_7 better**.
- **A01** (``alltime_min_ms_A01``): at 70 min — uni_7 24.63s, uni_8 25.18s, uni_9 27.09s → **uni_7 better**.
- **Training/loss**: at 70 min — uni_7 92.4, uni_8 53.4, uni_9 45.4; uni_8/9 have lower raw loss but **race times and Q worse** ⇒ loss alone misleading.
- **``RL/avg_Q_*``**: at 70 min — uni_7 -0.85, uni_8 -1.37, uni_9 -2.31 → **uni_7 better** (less negative).
- **Performance/learner_percentage_training**: uni_7 ~79%, uni_8 ~80%, uni_9 ~82% — higher % **does not** yield better policies.

uni_7 vs uni_10 (gpu_collectors_count, relative time, common window up to 70 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

uni_10 — same batch_size 512 and running_speed 512 as uni_7, but **gpu_collectors_count = 4** instead of 8.

- **Hock** (``alltime_min_ms_hock``): at 20 min — uni_7 25.26s, uni_10 25.11s (uni_10 slightly faster); at 30 min — 25.14s vs 25.08s; at 70 min — uni_7 24.63s, uni_10 24.72s (uni_7 slightly better by end of common window).
- **A01** (``alltime_min_ms_A01``): at 70 min — uni_7 24.63s, uni_10 24.72s.
- **Training/loss**: at 70 min — uni_7 92.4, uni_10 100.7 — close; over the window difference is small.
- **``RL/avg_Q_*``**: at 70 min — uni_7 -0.85, uni_10 -0.86 — close.
- **Performance/learner_percentage_training**: uni_7 ~79%, uni_10 ~59% over the window → **throughput and GPU time share substantially lower** for uni_10; fewer ops per unit time.

**Conclusion for uni_10:** By relative time quality (times, Q) is close to uni_7; uni_10 sometimes reaches good time slightly faster early on. Main difference — **fewer ops per unit time** and lower GPU %; for maximum training speed by wall clock, 8 collectors are better.

Other metrics (from ``docs/source/tensorboard_metrics.rst``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When comparing batch/speed setups, also check in TensorBoard or via ``scripts/extract_tensorboard_data.py``:

- **Performance/transitions_learned_per_second** — training throughput; higher is better; reflects efficiency of the pipeline.
- **Gradients/norm_before_clip_max** — training stability; spikes >100 indicate gradient explosion; should stay relatively stable. Compare runs to ensure no setup introduces instability.

Full interpretation and “what to watch for” for all metrics: ``docs/source/tensorboard_metrics.rst``.

Configuration Changes
----------------------

**Training** (``config_files/training_config.py``): batch_size was varied (2048 baseline, 8192 in uni_6, 512 in uni_7, smaller in uni_8/uni_9). **Recommended:** 512.

**Performance** (``config_files/performance_config.py``): running_speed was varied (160 baseline, 512 in uni_7, higher in uni_8/uni_9). **Recommended:** 512. **gpu_collectors_count**: uni_10 used 4 with same 512/512 — throughput and GPU % drop; for maximum training speed keep 8.

.. code-block:: python

   # Recommended (uni_7-style)
   batch_size = 512   # not 2048, not 8192, not 64
   running_speed = 512  # not 160, not 1024
   gpu_collectors_count = 8  # 4 (uni_10) yields fewer ops/sec and lower GPU %

Hardware
--------

- **GPU**: RTX 5090
- **Parallel instances**: 8 collectors (uni_5–uni_9); uni_10 used 4 — less throughput
- **System**: Same across all runs

Conclusions
-----------

Conclusions are given by **relative time** (minutes from run start); comparing by “last value” when runs have different duration is invalid.

1. **Larger batch (8192) hurts:** On the common window by relative time — worse best times, ~10× higher loss, GPU % no gain (~52%). Keep batch ≤ 2048 or use 512 for the “small batch + fast speed” regime.

2. **Smaller batch (512) + faster speed (512×) helps:** At the same minutes from start uni_7 reaches good times faster, much lower loss, better Q, ~78% GPU. Best observed trade-off.

3. **Over-tuning (smaller batch + higher speed) hurts again:** On the common window up to 70 min uni_8/uni_9 do not reach uni_7-level times and Q. **Golden middle** — uni_7 (512/512); extremes (e.g. 64/1024) degrade quality.

4. **Trade-offs:** Smaller batches need more frequent sync; very high speed can affect physics; going beyond the sweet spot (batch/speed) degrades performance.

5. **gpu_collectors_count (uni_10):** By relative time quality is close to uni_7; uni_10 sometimes slightly faster early on. **Main point** — drop in throughput and GPU % (~59% vs ~79%); for maximum training speed by wall clock, 8 collectors are better.

Recommendations
---------------

- **Use batch_size = 512** and **running_speed = 512** as the default “golden middle” for this setup (IQN, RTX 5090).
- **Use gpu_collectors_count = 8** for maximum throughput; 4 collectors (uni_10) reduce ops per unit time and GPU share on training (~59% vs ~78%).
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
- **By relative time** (compare at the same minutes from start): ``python scripts/analyze_experiment_by_relative_time.py uni_5 uni_7 --interval 5`` (or ``uni_7 uni_10`` etc.; ``--logdir "<path>"`` if not from project root).
- By “last value” (less meaningful when durations differ): ``python scripts/analyze_experiment.py uni_5 uni_6 uni_7 ...``
- ``scripts/extract_tensorboard_data.py`` — selective metrics (``Gradients/norm_before_clip_max``, ``Performance/transitions_learned_per_second``, etc.).
- **Key metrics** (see ``docs/source/tensorboard_metrics.rst``): ``Training/loss``, ``Race/eval_race_time_robust_*``, ``alltime_min_ms_{map}``, ``RL/avg_Q_*``, ``Performance/learner_percentage_training``, ``Performance/transitions_learned_per_second``, ``Gradients/norm_before_clip_max`` (stability).
