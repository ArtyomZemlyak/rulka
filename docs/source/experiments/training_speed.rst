Experiment: Batch Size and Running Speed
========================================

Experiment Overview
-------------------

This document covers experiments on **batch_size** and **running_speed**. Baseline: **uni_5** (batch 2048, speed 160×).

**Direction 1 — Larger batch:** Increasing batch to 8192 (uni_6) **worsened** convergence and loss; RTX 5090 hit a performance ceiling.

**Direction 2 — Smaller batch + faster speed:** Reducing batch to 512 and speed to 512× (uni_7) **improved** convergence, loss, and GPU utilization. Follow-up runs uni_8 and uni_9 (even smaller batch, higher speed) **reversed** those gains: over-tuning hurt. Conclusion: a **golden middle** between batch_size and running_speed is needed; uni_7 (512/512) is in that range.

**Direction 3 — Map cycle length:** uni_11 and uni_12 kept the same batch/speed as uni_7 (512/512) but changed the map cycle. **uni_11**: 256 hock – 256 A01 — by relative time uni_7 was slightly better on times and GPU %; interpretation ambiguous. **uni_12**: 64 hock – 64 A01 — by relative time **uni_12 converges faster** (good Hock by 15 min, good A01 by 25 min), reaches **equal or better** race times and lower loss / better Q over the common 55 min window than uni_7; and is **clearly better** than uni_11 (256–256). **Conclusion:** 64–64 is a good compromise; prefer it over 4–4 and over 256–256.

Results
-------

**Important:** Experiments had different durations (uni_5 ~160 min, uni_7 ~86 min, etc.), so comparing by “last value” is meaningless. All findings below are based on **relative time** — minutes from run start; metrics are compared at the same moments (5, 10, 20, … min), up to when the shortest compared run ends.

**Key Findings (by relative time):**

- **Larger batch (uni_6)**: At the same minutes from start — worse best times (at 160 min: Hock 24.77s vs 24.56s for uni_5), loss ~10× higher (2980 vs 301), GPU % no gain (~52% vs ~54%). Do not go above 2048.
- **Smaller batch + faster speed (uni_7)**: At the same minutes uni_7 reaches good times faster (at 60 min Hock 24.63s vs 24.69s for uni_5), loss much lower, Q better, GPU ~78% vs ~53%. Best trade-off.
- **Over-tuning (uni_8, uni_9)**: On the common window up to 70 min best times and Q are with uni_7 (at 70 min Hock 24.63s vs 25.18s / 25.25s); uni_8/9 have higher GPU %, but policies are worse.
- **Fewer collectors (uni_10, gpu_collectors_count=4)**: At the same 512/512 by relative time uni_10 is sometimes slightly faster to good times (at 20–30 min slightly better), by 70 min uni_7 is slightly ahead (24.63s vs 24.72s). **Main point** — throughput and GPU %: ~59% vs ~79%; fewer ops per unit time. For maximum training speed by wall clock, 8 collectors are better.

- **Map cycle 256 hock – 256 A01 (uni_11 vs uni_7)**: Same batch/speed as uni_7; cycle changed from 4–4 to 256–256 (both with 1 eval on A01 at the end). Common window up to 55 min (uni_11 ~55 min, uni_7 ~86 min). By relative time: uni_7 slightly better on Hock at 55 min (24.72s vs 25.08s) and on A01 (24.95s vs 26.04s); loss and Q are close; GPU % uni_7 ~79% vs uni_11 ~70%. Slightly more events in uni_11, but metrics do not differ strongly — hard to interpret as clearly better or worse.

- **Map cycle 64 hock – 64 A01 (uni_12 vs uni_7, uni_11)**: Same batch/speed as uni_7; cycle **64–64** (+ 1 eval on A01 at the end). uni_12 ~55 min. By relative time over the common 55 min: **uni_12 converges faster** — Hock 24.85s by 15 min (uni_7 27.33s); A01 24.85s by 25 min (uni_7 reaches 24.95s only by 55 min). At 55 min: Hock uni_7 24.72s, uni_12 24.85s (uni_7 slightly better); A01 uni_12 24.85s, uni_7 24.95s (**uni_12 better**); loss 102.8 vs 113, Q -0.71 vs -1.24 (**uni_12 better**). GPU % uni_7 ~79%, uni_12 ~72%. Versus uni_11 (256–256): **uni_12 clearly better** — much faster convergence, better times and Q. **Conclusion:** 64–64 is **recommended** over 4–4 and over 256–256: faster convergence and equal or better quality in the same wall-clock window.

Run Analysis
------------

All runs use the same hardware (RTX 5090). uni_5–uni_9: 8 collectors; uni_10, uni_11, uni_12: 8 collectors. TensorBoard logs: ``tensorboard\uni_<N>``. **Durations (relative time):** uni_5 ~160 min, uni_6 ~165 min, uni_7 ~86 min, uni_8 ~76 min, uni_9 ~194 min, uni_10 ~71 min, uni_11 ~55 min, uni_12 ~55 min.

- **uni_5**: Baseline — batch_size = 2048, running_speed = 160, ~160 min
- **uni_6**: Larger batch 8192, speed 160, ~165 min — by relative time worse Hock and loss
- **uni_6_2**: Continuation of uni_6 (same config)
- **uni_7**: Smaller batch 512, speed 512, 8 collectors, map cycle **4 hock – 4 A01** (+ 1 eval A01 at end), ~86 min — **best trade-off** by relative time
- **uni_8**: Further smaller batch + higher speed, ~76 min — worse than uni_7 on times and Q on common window
- **uni_9**: Extreme (e.g. batch 64, speed 1024), ~194 min — worse than uni_7 on common window up to 70 min
- **uni_10**: Same as uni_7 (batch 512, speed 512) but **gpu_collectors_count = 4**, ~71 min — sometimes slightly faster on time early on, by 70 min slightly worse; **throughput and GPU % lower** (~59% vs ~79%).
- **uni_11**: Same as uni_7 (batch 512, speed 512, 8 collectors) but map cycle **256 hock – 256 A01** (+ 1 eval A01 at end), ~55 min — by relative time slightly more events than uni_7; metrics close, uni_7 slightly better on times and GPU % over common window up to 55 min.
- **uni_12**: Same as uni_7 (batch 512, speed 512, 8 collectors) but map cycle **64 hock – 64 A01** (+ 1 eval A01 at end), ~55 min — by relative time **faster convergence**, equal or better A01 and loss/Q than uni_7 over 55 min; **clearly better** than uni_11 (256–256). **Recommended** map cycle length for this setup.

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

uni_7 vs uni_11 (map cycle 4–4 vs 256–256, relative time, common window up to 55 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

uni_11 — same batch_size 512 and running_speed 512 as uni_7, but map cycle **256 hock – 256 A01** (+ 1 eval on A01 at the end) instead of **4 hock – 4 A01** (+ 1 eval on A01 at the end). uni_11 ran ~55 min; uni_7 ~86 min.

- **Hock** (``alltime_min_ms_hock``): uni_7 improves faster early (at 15 min 27.33s vs uni_11 64.58s); by 40 min they are close (24.95s vs 25.08s). At 55 min — uni_7 24.72s, uni_11 25.08s → **uni_7 slightly better** at end of common window.
- **A01** (``alltime_min_ms_A01``): uni_11 has A01 data earlier (longer A01 phase); at 55 min — uni_7 24.95s, uni_11 26.04s → **uni_7 better** on A01 over the window.
- **Training/loss**: at 55 min — uni_7 113.0, uni_11 106.0 — **close**; over the run differences are small.
- **``RL/avg_Q_trained_A01``**: at 55 min — uni_7 -1.24, uni_11 -1.10; uni_11 shows almost constant Q for many checkpoints (eval schedule differs). **Close**; not clearly better or worse.
- **Performance/learner_percentage_training**: uni_7 ~79% vs uni_11 ~70% over the window → **uni_7 higher GPU %**.

**Conclusion for uni_11:** Slightly more events in uni_11; by relative time metrics do not differ strongly. uni_7 is slightly better on race times and GPU % over the common window up to 55 min. **Interpretation is ambiguous** — the longer map cycle (256–256) is not clearly better or worse; more data or longer runs would be needed to draw a firmer conclusion.

uni_7 vs uni_12 (map cycle 4–4 vs 64–64, relative time, common window up to 55 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

uni_12 — same batch_size 512 and running_speed 512 as uni_7, but map cycle **64 hock – 64 A01** (+ 1 eval on A01 at the end) instead of **4 hock – 4 A01**. uni_12 ran ~55 min; uni_7 ~86 min.

- **Hock** (``alltime_min_ms_hock``): uni_12 reaches 24.85s **by 15 min** (uni_7 still 27.33s); uni_12 then stays at 24.85s. At 55 min — uni_7 24.72s, uni_12 24.85s → **uni_7 slightly better** on Hock at end of window; uni_12 **converges much faster** in the first half.
- **A01** (``alltime_min_ms_A01``): uni_12 reaches 24.85s by 25 min; uni_7 reaches 24.95s only by 55 min → **uni_12 better** on A01 over the window (24.85s vs 24.95s at 55 min).
- **Training/loss**: at 55 min — uni_7 113.0, uni_12 102.8 → **uni_12 lower** (better); trend holds over the run.
- **``RL/avg_Q_trained_A01``**: at 55 min — uni_7 -1.24, uni_12 -0.71 → **uni_12 better** (less negative); uni_12 also improves earlier (e.g. at 20 min -0.83 vs uni_7 -1.44).
- **Performance/learner_percentage_training**: uni_7 ~79% vs uni_12 ~72% over the window → uni_7 higher GPU %; uni_12 trades a bit of throughput for **faster convergence and better policy metrics**.

**Conclusion for uni_12:** On the common 55 min window, **64–64 yields faster convergence** (good times by 15–25 min) and **equal or better** quality: better A01, lower loss, better Q; uni_7 only edges ahead on Hock at 55 min (24.72s vs 24.85s). Versus uni_11 (256–256), uni_12 is **clearly better** on all metrics. **Recommendation:** Prefer **64 hock – 64 A01** over 4–4 and over 256–256 for this setup.

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

**Map cycle** (``config_files/map_cycle_config.py``): uni_7 used **4 hock – 4 A01** (+ 1 eval on A01 at the end); uni_11 **256 hock – 256 A01**; uni_12 **64 hock – 64 A01** (+ 1 eval on A01 at the end). Same batch/speed as uni_7. By relative time **uni_12 (64–64)** converges faster and gives equal or better A01, loss, and Q over the common 55 min window than uni_7; clearly better than uni_11. **Recommended:** 64–64.

.. code-block:: python

   # Recommended (uni_7-style)
   batch_size = 512   # not 2048, not 8192, not 64
   running_speed = 512  # not 160, not 1024
   gpu_collectors_count = 8  # 4 (uni_10) yields fewer ops/sec and lower GPU %

Hardware
--------

- **GPU**: RTX 5090
- **Parallel instances**: 8 collectors (uni_5–uni_9, uni_11, uni_12); uni_10 used 4 — less throughput
- **System**: Same across all runs

Conclusions
-----------

Conclusions are given by **relative time** (minutes from run start); comparing by “last value” when runs have different duration is invalid.

1. **Larger batch (8192) hurts:** On the common window by relative time — worse best times, ~10× higher loss, GPU % no gain (~52%). Keep batch ≤ 2048 or use 512 for the “small batch + fast speed” regime.

2. **Smaller batch (512) + faster speed (512×) helps:** At the same minutes from start uni_7 reaches good times faster, much lower loss, better Q, ~78% GPU. Best observed trade-off.

3. **Over-tuning (smaller batch + higher speed) hurts again:** On the common window up to 70 min uni_8/uni_9 do not reach uni_7-level times and Q. **Golden middle** — uni_7 (512/512); extremes (e.g. 64/1024) degrade quality.

4. **Trade-offs:** Smaller batches need more frequent sync; very high speed can affect physics; going beyond the sweet spot (batch/speed) degrades performance.

5. **gpu_collectors_count (uni_10):** By relative time quality is close to uni_7; uni_10 sometimes slightly faster early on. **Main point** — drop in throughput and GPU % (~59% vs ~79%); for maximum training speed by wall clock, 8 collectors are better.

6. **Map cycle length (uni_11, uni_12):** **256–256 (uni_11):** By relative time uni_7 slightly better on times and GPU %; ambiguous. **64–64 (uni_12):** By relative time **uni_12 converges faster** (good Hock by 15 min, A01 by 25 min), reaches **equal or better** A01, loss, and Q than uni_7 over 55 min, and is **clearly better** than uni_11. **Recommendation:** Use **64 hock – 64 A01** over 4–4 and over 256–256.

Recommendations
---------------

- **Use batch_size = 512** and **running_speed = 512** as the default “golden middle” for this setup (IQN, RTX 5090).
- **Use gpu_collectors_count = 8** for maximum throughput; 4 collectors (uni_10) reduce ops per unit time and GPU share on training (~59% vs ~78%).
- **Avoid** very small batch (e.g. 64) with very high speed (e.g. 1024); it slows training and convergence.
- **Do not** increase batch beyond 2048 (or 512 if using the fast-speed regime) without evidence of benefit.
- **Monitor** training loss, map times, and Q-values; GPU % alone is not sufficient.
- **Tune** batch_size and running_speed together; uni_7 (512/512) is a proven working point.
- **Map cycle:** Prefer **64 hock – 64 A01** (uni_12): by relative time it converges faster and gives equal or better A01, loss, and Q than 4–4 (uni_7), and is clearly better than 256–256 (uni_11). Use 64–64 as the default map cycle for this setup.

**When to change:**

- Larger batches (1024–2048): if GPU memory or sync overhead is the bottleneck.
- Slower speed (160–200): if physics or stability is critical.
- Keep 512/512 unless you have a clear reason to move away.

**Analysis Tools:**

- Activate venv (Windows): ``.\.venv\Scripts\activate``
- **By relative time** (compare at the same minutes from start): ``python scripts/analyze_experiment_by_relative_time.py uni_5 uni_7 --interval 5`` (or ``uni_7 uni_10``, ``uni_7 uni_11``, ``uni_7 uni_12``, ``uni_11 uni_12``, etc.; ``--logdir "<path>"`` if not from project root).
- By “last value” (less meaningful when durations differ): ``python scripts/analyze_experiment.py uni_5 uni_6 uni_7 ...``
- ``scripts/extract_tensorboard_data.py`` — selective metrics (``Gradients/norm_before_clip_max``, ``Performance/transitions_learned_per_second``, etc.).
- **Key metrics** (see ``docs/source/tensorboard_metrics.rst``): ``Training/loss``, ``Race/eval_race_time_robust_*``, ``alltime_min_ms_{map}``, ``RL/avg_Q_*``, ``Performance/learner_percentage_training``, ``Performance/transitions_learned_per_second``, ``Gradients/norm_before_clip_max`` (stability).
