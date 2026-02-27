.. _pretrain_bc:

BC pretraining
==============

Experiment Overview
-------------------

This experiment tested whether **adding behavioral cloning (BC) pretraining** on top of the visual backbone pretraining improves RL training. Two BC-based RL variants are compared:

- **A01_as20_long_vis_bc_pretrained** — three stages: (1) Visual pretrain (vis v1), (2) BC pretrain with **backbone only** (single-frame action prediction, encoder saved), (3) RL with that encoder.
- **A01_as20_long_vis_bc_ah_pretrained** — uses **both** the visual encoder (**img head**) and the **action head (A_head)** from the **v2_multi_offset_ahead_dropout_inner** BC experiment (see :ref:`pretrain_bc_behavioral_cloning`). That BC run uses multi-offset action prediction, IQN-style A_head (MLP) with dropout (0.2 on features, 0.1 inside head), vis v2 backbone, and achieves best val_acc 0.597 / val_loss 1.971. RL injects ``encoder.pt`` and ``actions_head.pt`` from ``output/ptretrain/bc/v2_multi_offset_ahead_dropout_inner/``.

Stages for the first variant:

1. **Visual pretrain** — same as A01_as20_long_vis_pretrained: autoencoder on replay frames (``config_files/pretrain/vis/pretrain_config.yaml``), producing ``output/ptretrain/vis/v1/encoder.pt``.
2. **BC pretrain** — train a CNN to predict actions from images, initialized from the vis encoder (``config_files/pretrain/bc/pretrain_config_bc.yaml``). BC mode ``backbone``: only the encoder is saved (``output/ptretrain/bc/v1.1/encoder.pt``) and injected into IQN. In this experiment BC was simply **action prediction from a single frame** (no temporal context).
3. **RL training** — same RL config as the other runs, with ``pretrain_encoder_path: "output/ptretrain/bc/v1.1/encoder.pt"`` so the BC-trained encoder is used as the IQN visual backbone.

**Hypothesis:** BC might provide a better initialization than visual-only pretraining by aligning the backbone with action-relevant features. Using **both** encoder and A_head (bc_ah) may give a stronger policy prior than encoder-only.

**Main question:** Does BC help over vis-only? Does adding the pretrained A_head (bc_ah) improve over encoder-only BC (bc_pretrained)?

Results
-------

**Important:** Run durations differed (A01_as20_long ~495 min, A01_as20_long_vis_pretrained ~275 min, A01_as20_long_vis_bc_pretrained ~262 min, A01_as20_long_vis_bc_ah_pretrained ~244 min). All findings below are by **relative time** (minutes from run start) and by **steps** (training step checkpoints). Comparing by "last value" is invalid. Common window for all four runs: **up to 244 min** (shortest = bc_ah).

**Key Findings:**

- **BC does not improve final performance over vis-only (encoder-only BC).** By the end of the common window (260 min for the first three runs), **vis_pretrained has the best A01 time (24.47s)**, then bc_pretrained (24.49s), then baseline (24.53s).
- **BC + A_head (bc_ah) matches the best A01 time.** At 240 min (end of bc_ah run), **bc_ah reaches 24.47s** — tied with vis_pretrained and better than bc_pretrained (24.49s) and baseline (24.53s). So the new experiment (encoder + A_head from v2_multi_offset_ahead_dropout_inner) is **as good as vis-only** on final best time and **better than encoder-only BC**.
- **BC gives the fastest early convergence (encoder-only):** First eval finish at **5.1 min** (bc) vs 8.3 min (baseline) vs 11.2 min (vis_pretrained). **bc_ah** has first eval finish at **15.2 min** (slowest of the four) — pretrained A_head does not help early first finish.
- **By steps:** At 10.85M steps (common max for first three), same ordering: vis_pretrained 24.47s, bc 24.49s, baseline 24.53s. bc_ah run is shorter; at equal relative time (244 min) bc_ah reaches 24.47s.
- **Eval finish rate at 240 min:** vis_pretrained 74%, bc_pretrained 70%, bc_ah 68%, baseline 63%. bc_ah is between bc and baseline.
- **Training loss at 240 min:** bc_pretrained 54.93 (lowest), bc_ah 62.58, baseline 63.89, vis_pretrained 64.33. bc_ah has lower loss than baseline and vis.
- **Conclusion:** **BC + A_head (bc_ah) is the only BC variant that matches vis-only on final A01 best time (24.47s)** and beats encoder-only BC (24.49s). It does not improve over vis on finish rate or early first finish; it is a viable alternative when using the multi-offset A_head pretrain.

**Does pretrain and transfer help? (Summary)**

**Question:** With the setups tried so far, does pretrain (BC encoder + A_head) and transfer actually help, or is it not working?

**Answer:** **Yes, pretrain helps when used correctly.** Evidence: **bc_ah** (encoder + A_head trainable) reaches **24.47s** — tied with vis-only. **enc_ah_freeze** (both frozen) fails (24.97s, 22% finish rate). **enc_ah_freeze_resume** (unfreeze + lower lr/epsilon) **recovers** to **24.51s** from the poor enc_ah_freeze checkpoint. So pretrain works when allowed to fine-tune; freezing A_head hurts; unfreezing with gentle lr/epsilon recovers. It does not beat vis-only on final time, but is a viable path. For best final time, use bc_ah or vis-only; avoid freezing A_head.

Run Analysis
------------

- **A01_as20_long** (baseline): No pretrain. ``pretrain_encoder_path: null``. IQN from random weights. **~495 min**, 3 TensorBoard log dirs merged.
- **A01_as20_long_vis_pretrained**: Visual pretrain only. ``pretrain_encoder_path: "output/ptretrain/vis/v1/encoder.pt"``. **~275 min**, 2 log dirs merged.
- **A01_as20_long_vis_bc_pretrained**: Vis pretrain then BC pretrain then RL. ``pretrain_encoder_path: "output/ptretrain/bc/v1.1/encoder.pt"``. BC encoder from ``config_files/pretrain/bc/pretrain_config_bc.yaml`` (encoder_init_path: vis v1, bc_mode: backbone, 50 epochs, action prediction from image). **~262 min**, 2 log dirs merged.
- **A01_as20_long_vis_bc_ah_pretrained**: RL with **encoder + A_head** from **v2_multi_offset_ahead_dropout_inner**. ``pretrain_encoder_path: "output/ptretrain/bc/v2_multi_offset_ahead_dropout_inner/encoder.pt"``, ``pretrain_actions_head_path: "output/ptretrain/bc/v2_multi_offset_ahead_dropout_inner/actions_head.pt"``. BC run: ``config_files/pretrain/bc/pretrain_config_bc_v2_multi_offset_ahead_dropout_inner.yaml`` — vis v2 backbone, multi-offset ``bc_time_offsets_ms: [-10, 0, 10, 100]``, ``use_actions_head: true``, dropout 0.2 on features, action_head_dropout 0.1; val_acc 0.597, val_loss 1.971 (best A_head variant). **~244 min**, 2 log dirs merged.
- **A01_as20_long_vis_bc_ah_pretrained_enc_freeze**: Same pretrain sources as bc_ah, but **encoder frozen** during RL: ``pretrain_encoder_freeze: true``. A_head, float_feature_extractor, iqn_fc, V_head trainable. **~158 min**, 2 log dirs merged.
- **A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze**: Same pretrain, but **both encoder and A_head frozen** during RL: ``pretrain_encoder_freeze: true``, ``pretrain_actions_head_freeze: true``. Only float_feature_extractor, iqn_fc, V_head trainable. **~228 min**, 2 log dirs merged.
- **A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze_resume**: Resumed from enc_ah_freeze weights; **encoder and A_head unfrozen**, lower lr (5e-5 at start, 1e-4 from 500k) and lower epsilon (0.1 at start vs 1.0). All parts trainable. **~259 min**, 2 log dirs merged.

Detailed TensorBoard Metrics Analysis
-------------------------------------

**Methodology — Relative time and by steps:** Metrics are compared (1) at checkpoints 5, 10, 15, 20, … min (only up to the shortest run; when including bc_ah, use the common window of all four runs) and (2) at step checkpoints 50k, 100k, … (only up to the smallest max step). The figures below show one metric per graph (runs as lines, by relative time). To generate plots including bc_ah, use the four-run command in Analysis Tools and regenerate with ``--experiments pretrain_bc``.

A01 Map Performance (common window up to 244 min for all four runs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Baseline (A01_as20_long):** at 35 min — 25.02s; at 85 min — 24.71s; at 150 min — 24.59s; at 240 min — 24.53s. First eval finish ~8.3 min.
- **Vis pretrained (A01_as20_long_vis_pretrained):** at 35 min — 24.79s; at 85 min — 24.55s; at 150 min — 24.50s; at 240 min — **24.47s**. First eval finish ~11.2 min.
- **BC pretrained (A01_as20_long_vis_bc_pretrained):** at 10 min — 24.92s (fastest early); at 35 min — 24.89s; at 85 min — 24.59s; at 150 min — 24.55s; at 240 min — 24.49s. First eval finish **~5.1 min** (earliest).
- **BC + A_head (A01_as20_long_vis_bc_ah_pretrained):** First eval finish **~15.2 min** (latest of the four). At 30 min — 24.83s; at 65 min — 24.59s; at 100 min — 24.55s; at 140 min — 24.53s; at 190 min — 24.47s; at 240 min — **24.47s** (tied with vis_pretrained for best). Eval finish rate at 240 min: 68%.

.. image:: ../_static/exp_pretrain_bc_A01_best.jpg
   :alt: A01 best time by relative time (A01_as20_long vs vis_pretrained vs vis_bc_pretrained)

Training Loss
~~~~~~~~~~~~~

- **Baseline:** at 90 min — 64.29; at 240 min — 63.89.
- **Vis pretrained:** at 90 min — 61.35; at 240 min — 64.33.
- **BC pretrained:** at 90 min — 71.70; at 240 min — 54.93 (lowest at 240 min).
- **BC + A_head (bc_ah):** at 90 min — 54.06; at 240 min — 62.58. Lower than baseline and vis at 240 min.

.. image:: ../_static/exp_pretrain_bc_loss.jpg
   :alt: Training loss by relative time (pretrain_bc three-way comparison)

Average Q-values
~~~~~~~~~~~~~~~~

- At 240 min: baseline -1.14, vis_pretrained -0.68, bc_pretrained -0.61, bc_ah -1.49. No clear winner; bc_ah is more negative (more conservative Q).

.. image:: ../_static/exp_pretrain_bc_avg_q.jpg
   :alt: Avg Q by relative time (pretrain_bc three-way comparison)

GPU Utilization
~~~~~~~~~~~~~~~

- All ~69–72% over the common window; no significant difference.

Experiment: Encoder freeze (bc_ah vs enc_freeze)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Compare **A01_as20_long_vis_bc_ah_pretrained** (encoder + A_head from v2_multi_offset_ahead_dropout_inner, **encoder trainable**) with **A01_as20_long_vis_bc_ah_pretrained_enc_freeze** (same pretrain, **img_head frozen** during RL). Only difference: ``pretrain_encoder_freeze: true`` in enc_freeze.

**Common window:** Up to **158 min** (shortest run = enc_freeze). All comparisons below are by relative time over this window.

**A01 map performance:**

- **bc_ah (encoder trainable):** First eval finish **~15.2 min**. At 60 min — 24.80s; at 100 min — 24.55s; at 155 min — **24.50s**. Eval finish rate at 155 min — 66%. Run continued to 244 min and reached 24.47s.
- **enc_freeze (encoder frozen):** First eval finish **~4.1 min** (much earlier than bc_ah). At 60 min — 24.81s; at 100 min — 24.60s; at 155 min — **24.55s**. Eval finish rate at 155 min — 64%. Run ended at ~158 min; best time in window 24.55s.

**Conclusion (encoder freeze):** Over the common window (158 min), **bc_ah has slightly better best A01 time** (24.50s vs 24.55s at 155 min) and **slightly higher eval finish rate** (66% vs 64%). **enc_freeze has much earlier first eval finish** (4.1 min vs 15.2 min). Training loss at 155 min: bc_ah 57.29, enc_freeze 63.33 (bc_ah lower). So **freezing the encoder** preserves a strong policy prior (very fast first finish) but does **not** match the trainable-encoder run on best time within 158 min; the trainable encoder (bc_ah) pulls ahead by ~50 ms and has lower loss. For best final A01 time in long runs, keep the encoder trainable; use enc_freeze if you want fastest early first finish or fewer parameters to train.

.. image:: ../_static/exp_pretrain_bc_enc_freeze_A01_best.jpg
   :alt: A01 best time by relative time (bc_ah vs enc_freeze)

**Training loss and avg Q (at 60, 100, 155 min):**

- **Loss:** bc_ah 74.89 / 60.07 / 57.29; enc_freeze 89.35 / 62.26 / 63.33. bc_ah lower at all checkpoints.
- **Avg Q:** Both fluctuate; at 155 min bc_ah -0.66, enc_freeze -1.34.

.. image:: ../_static/exp_pretrain_bc_enc_freeze_loss.jpg
   :alt: Training loss by relative time (bc_ah vs enc_freeze)

.. image:: ../_static/exp_pretrain_bc_enc_freeze_avg_q.jpg
   :alt: Avg Q by relative time (bc_ah vs enc_freeze)

**Reproduce:** ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long_vis_bc_ah_pretrained A01_as20_long_vis_bc_ah_pretrained_enc_freeze --interval 5 --step_interval 50000``

Experiment: Encoder + A_head freeze (three-way: bc_ah vs enc_freeze vs enc_ah_freeze)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Compare all three freeze variants with the same BC+AH pretrain: (1) **bc_ah** — both encoder and A_head trainable; (2) **enc_freeze** — encoder frozen, A_head trainable; (3) **enc_ah_freeze** — **both** encoder and A_head frozen (only float_feature_extractor, iqn_fc, V_head trainable).

**Common window:** Up to **158 min** (shortest = enc_freeze). All three runs have data up to 155 min.

**A01 map performance (at 155 min):**

| Run | Encoder | A_head | Best time (155 min) | First eval finish | Eval finish rate (155 min) | Loss (155 min) |
|-----|---------|--------|---------------------|-------------------|----------------------------|----------------|
| **bc_ah** | trainable | trainable | **24.50s** | 15.2 min | 66% | 57.29 |
| **enc_freeze** | frozen | trainable | **24.55s** | **4.1 min** | 64% | 63.33 |
| **enc_ah_freeze** | frozen | frozen | **24.97s** | 8.9 min | 22% | 279.12 |

**Conclusions (what to freeze):**

- **Freezing both encoder and A_head** (enc_ah_freeze) **severely hurts** RL: best time 24.97s (~470 ms worse than bc_ah), finish rate 22% (vs 66%), loss ~5x higher. The frozen A_head blocks policy adaptation; float_feature_extractor, iqn_fc, and V_head alone cannot compensate. **Do not freeze A_head.**
- **Freezing only the encoder** (enc_freeze) is **acceptable**: best time 24.55s (~50 ms worse than bc_ah), finish rate 64%, loss slightly higher. enc_freeze has **much earlier first eval finish** (4.1 min vs 15.2 min). Use when you want fastest early first finish or fewer trainable parameters.
- **Best final A01 time:** Keep **both encoder and A_head trainable** (bc_ah). For long runs aiming at 24.47s, do not freeze either.

**Summary:** Freeze **encoder only** if you prioritize early first finish; never freeze A_head. For best final time, train both.

.. image:: ../_static/exp_pretrain_bc_enc_ah_freeze_A01_best.jpg
   :alt: A01 best time by relative time (bc_ah vs enc_freeze vs enc_ah_freeze)

.. image:: ../_static/exp_pretrain_bc_enc_ah_freeze_loss.jpg
   :alt: Training loss by relative time (bc_ah vs enc_freeze vs enc_ah_freeze)

.. image:: ../_static/exp_pretrain_bc_enc_ah_freeze_avg_q.jpg
   :alt: Avg Q by relative time (bc_ah vs enc_freeze vs enc_ah_freeze)

**Reproduce:** ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long_vis_bc_ah_pretrained A01_as20_long_vis_bc_ah_pretrained_enc_freeze A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze --interval 5 --step_interval 50000``

Experiment: Resume from enc_ah_freeze (unfreeze + lower lr and epsilon)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Take enc_ah_freeze checkpoint (both encoder and A_head frozen during RL; best 24.97s, 22% finish rate), **resume** training with everything **unfrozen** and **reduced lr/epsilon**. Hypothesis: gentle fine-tuning of the pretrained parts will recover or improve performance.

**Setup:** Load weights from enc_ah_freeze; set ``pretrain_encoder_freeze: false``, ``pretrain_actions_head_freeze: false``; lr_schedule: 5e-5 (0–500k), 1e-4 (500k–3M), 5e-5 (3M+); epsilon: 0.1 at start (vs 1.0 in enc_ah_freeze), 0.5 at 300k, 0.03 by 3M.

**Common window:** Up to **228 min** (shortest = enc_ah_freeze). enc_ah_freeze_resume ran ~259 min.

**A01 map performance (by relative time):**

| Run | Best time (225 min) | First eval finish | Eval finish rate (225 min) | Loss (225 min) |
|-----|---------------------|-------------------|----------------------------|----------------|
| **enc_ah_freeze** | **24.97s** | 8.9 min | 29% | 229.71 |
| **enc_ah_freeze_resume** | **24.51s** | **6.0 min** | **56%** | 73.39 |

**By steps (at 9.65M steps, common for both):** enc_ah_freeze best 24.97s; enc_ah_freeze_resume best **24.51s** (460 ms better).

**Conclusions:**

- **Unfreezing + lower lr/epsilon recovers from enc_ah_freeze.** enc_ah_freeze_resume reaches **24.51s** vs 24.97s — ~460 ms improvement. Finish rate 56% vs 22%; loss ~73 vs ~230. The pretrained encoder and A_head, when allowed to fine-tune gently, quickly adapt and improve.
- **First eval finish:** resume 6.0 min vs freeze 8.9 min — slightly earlier.
- **Recommendation:** If you ran enc_ah_freeze and it plateaued poorly, resume with unfreeze + lower lr (5e-5–1e-4) and lower epsilon (0.1 at start) to recover.

.. image:: ../_static/exp_pretrain_bc_enc_ah_freeze_resume_A01_best.jpg
   :alt: A01 best time by relative time (enc_ah_freeze vs enc_ah_freeze_resume)

.. image:: ../_static/exp_pretrain_bc_enc_ah_freeze_resume_loss.jpg
   :alt: Training loss by relative time (enc_ah_freeze vs enc_ah_freeze_resume)

**Reproduce:** ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze_resume --interval 5 --step_interval 50000``

Configuration Changes
---------------------

**RL training** (encoder and optional A_head source):

.. code-block:: yaml

   # Baseline
   pretrain_encoder_path: null

   # Vis pretrained
   pretrain_encoder_path: "output/ptretrain/vis/v1/encoder.pt"

   # BC pretrained (vis -> BC -> RL, encoder only)
   pretrain_encoder_path: "output/ptretrain/bc/v1.1/encoder.pt"

   # BC + A_head (encoder and action head from v2_multi_offset_ahead_dropout_inner)
   pretrain_encoder_path: "output/ptretrain/bc/v2_multi_offset_ahead_dropout_inner/encoder.pt"
   pretrain_actions_head_path: "output/ptretrain/bc/v2_multi_offset_ahead_dropout_inner/actions_head.pt"

   # BC + A_head with encoder freeze (enc_freeze run; only difference: freeze img_head during RL)
   # pretrain_encoder_path and pretrain_actions_head_path same as above
   pretrain_encoder_freeze: true

   # BC + A_head with encoder AND A_head freeze (enc_ah_freeze run)
   pretrain_encoder_freeze: true
   pretrain_actions_head_freeze: true

   # enc_ah_freeze_resume: unfreeze and use lower lr/epsilon
   pretrain_encoder_freeze: false
   pretrain_actions_head_freeze: false
   lr_schedule: [[0, 0.00005], [500000, 0.0001], [3000000, 0.00005], ...]
   # epsilon_schedule: [[0, 0.1], [50000, 0.1], [300000, 0.5], [3000000, 0.03]]

**Visual pretrain** (``config_files/pretrain/vis/pretrain_config.yaml``): task ae, image_size 64, n_stack 1, epochs 50, batch_size 4096, output_dir output/ptretrain/vis, run_name v1.

**BC pretrain** (``config_files/pretrain/bc/pretrain_config_bc.yaml``):

.. code-block:: yaml

   encoder_init_path: "output/ptretrain/vis/v1/encoder.pt"
   bc_mode: backbone
   n_actions: 12
   image_size: 64
   n_stack: 1
   epochs: 50
   batch_size: 4096
   output_dir: output/ptretrain/bc
   run_name: v1.1

Hardware
--------

- **GPU:** Same as other A01 runs.
- **Parallel instances:** Same gpu_collectors_count (from config_default).
- **System:** Windows.

Conclusions
-----------

- **Does BC help in the current variant?** **No** for encoder-only BC: vis_pretrained still achieves the best A01 time (24.47s) and highest eval finish rate (73%) by 260 min among the first three runs.
- **BC + A_head (bc_ah) matches the best.** The run **A01_as20_long_vis_bc_ah_pretrained** (encoder + A_head from **v2_multi_offset_ahead_dropout_inner**) reaches **24.47s** at 240 min — **tied with vis_pretrained** and better than encoder-only bc_pretrained (24.49s) and baseline (24.53s). So **the new experiment is as good as the previous best (vis-only)** on final A01 time. bc_ah has slower first eval finish (15.2 min vs 5.1–11.2 min) and lower finish rate at 240 min (68% vs 74% vis); it is a viable alternative when using the multi-offset A_head pretrain.
- **BC advantage (encoder-only):** Only in the **very early** phase: earliest first finish (5.1 min), good initial time (24.92s by 10 min). So BC gives a faster "cold start" but vis-only catches up and slightly outperforms by 260 min.
- **Encoder freeze (bc_ah vs enc_freeze):** Over the **common window (158 min)**, **bc_ah** (encoder trainable) has **better best A01 time** (24.50s vs 24.55s at 155 min) and lower training loss; **enc_freeze** has **much earlier first eval finish** (4.1 min vs 15.2 min). So freezing the encoder keeps a strong policy prior for fast first finish but does not match the trainable-encoder run on final best time within the same wall-clock. For long runs aiming at best A01 time, keep encoder trainable (bc_ah); use enc_freeze if you need fastest early first finish or fewer trainable parameters.
- **Encoder + A_head freeze (enc_ah_freeze):** Freezing **both** encoder and A_head **severely hurts** RL: best time 24.97s (~470 ms worse than bc_ah), finish rate 22%, loss ~5x higher. The frozen A_head blocks policy adaptation. **Do not freeze A_head**; freeze encoder only if you prioritize early first finish.
- **Resume from enc_ah_freeze (enc_ah_freeze_resume):** Unfreezing everything and using **lower lr (5e-5 at start, 1e-4 from 500k) and lower epsilon (0.1 vs 1.0)** recovers: best time **24.51s** (vs 24.97s), finish rate 56% (vs 22%), loss ~73 (vs ~230). The pretrain prior is useful when allowed to fine-tune gently.

Recommendations
---------------

- **For best final A01 performance (current experiments):** Use **visual pretrain only** (``pretrain_encoder_path: "output/ptretrain/vis/v1/encoder.pt"``) or **BC + A_head (bc_ah)** — both reach 24.47s. Do not add the encoder-only BC stage (v1.1) if the goal is best final time; bc_ah matches vis and is better than encoder-only BC.
- **If you need fastest early convergence (e.g. for debugging):** BC pretrain (v1.1) gives the earliest first finish (5.1 min) and a good initial time in the first 10–20 min; then consider switching to vis-only or bc_ah for long runs.
- **BC + A_head (bc_ah):** Use ``pretrain_encoder_path`` and ``pretrain_actions_head_path`` from v2_multi_offset_ahead_dropout_inner; reaches 24.47s (tied with vis-only). To freeze encoder: ``pretrain_encoder_freeze: true`` (enc_freeze); **do not freeze A_head** — enc_ah_freeze performs poorly. If you have an enc_ah_freeze checkpoint, **resume with unfreeze + lower lr/epsilon** (enc_ah_freeze_resume) to recover to ~24.51s.

**Suggested RL variations to better understand pretrain contribution:**

- **Lower LR for pretrained parts:** Use a separate ``param_group`` with 0.1x or 0.01x LR for pretrained layers (encoder, A_head) vs random-initialized (float_feature_extractor, iqn_fc, V_head). Tests whether "gentle" fine-tuning of pretrain preserves or improves results.
- **Warmup with frozen pretrain:** Start RL with encoder (and optionally A_head) frozen for N steps (e.g. 50k–200k), then unfreeze. Compare with bc_ah and enc_freeze; tests if early stabilization helps.
- **Pretrain ablation:** Run RL from **random init** (baseline), **encoder only** (vis v2), **A_head only** (random encoder + BC A_head — would require loading only A_head), and **both** (bc_ah). Quantifies contribution of encoder vs A_head vs synergy.
- **Shorter runs with matched steps:** Run all variants for exactly the same number of gradient steps (e.g. 5M) to compare sample efficiency without wall-clock bias.
- **Different exploration schedules:** With pretrain, the policy is already reasonable; try faster epsilon decay (e.g. 0.1 by 100k instead of 3M) to reduce random actions and see if it helps or hurts.

**Analysis Tools:**

- By **relative time and by steps** (three runs): ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long A01_as20_long_vis_pretrained A01_as20_long_vis_bc_pretrained --interval 5 --step_interval 50000``
- By **relative time and by steps** (four runs, include bc_ah): ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long A01_as20_long_vis_pretrained A01_as20_long_vis_bc_pretrained A01_as20_long_vis_bc_ah_pretrained --interval 5 --step_interval 50000``
- **bc_ah vs enc_freeze (two runs):** ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long_vis_bc_ah_pretrained A01_as20_long_vis_bc_ah_pretrained_enc_freeze --interval 5 --step_interval 50000``
- **bc_ah vs enc_freeze vs enc_ah_freeze (three runs):** ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long_vis_bc_ah_pretrained A01_as20_long_vis_bc_ah_pretrained_enc_freeze A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze --interval 5 --step_interval 50000``
- **enc_ah_freeze vs enc_ah_freeze_resume:** ``python scripts/analyze_experiment_by_relative_time.py A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze A01_as20_long_vis_bc_ah_pretrained_enc_ah_freeze_resume --interval 5 --step_interval 50000``
- Plots: ``python scripts/generate_experiment_plots.py --experiments pretrain_bc pretrain_bc_enc_freeze pretrain_bc_enc_ah_freeze pretrain_bc_enc_ah_freeze_resume`` (generates comparison plots when tensorboard logs exist)
