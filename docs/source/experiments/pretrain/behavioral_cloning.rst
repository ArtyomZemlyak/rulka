.. _pretrain_bc_behavioral_cloning:

BC pretrain: training length & early stop (Level 1)
====================================================

Experiment Overview
-------------------

This experiment compares two BC (behavioral cloning) pretrain runs to answer: **how long to train**, and whether **early stopping** is beneficial. Both runs use the same data and architecture (backbone mode, encoder init from Level 0 visual), but:

- **v1**: Stopped at **25 epochs** (early stopping enabled in config, or 25-epoch cap). Lower final val_loss (1.038) at stop.
- **v1.1**: **No early stopping**, **lr=0.0005**, full **50 epochs**. Slightly higher final val_loss (1.119); train_acc a bit higher (0.699 vs 0.696).

Goal: understand if we can use early stopping to save time without hurting downstream RL, and how **per-action accuracy** (e.g. left, right, accel, brake) evolves.

Results
-------

**Important:** v1 has 25 epochs (0–24), v1.1 has 50 epochs (0–49). Comparisons below are **by epoch** over the common window (0–24). At epoch 24, v1 is at its final checkpoint; v1.1 continues to epoch 49.

**Key findings**

- **Val loss:** At epoch 24, v1 val_loss = 1.038, v1.1 val_loss = 1.045. v1 (early stop) has **slightly better** val_loss at the same epoch. By epoch 49, v1.1 val_loss = 1.119 (worse than v1 at 24), so **training past ~25 epochs increases val_loss** (overfitting).
- **Overall accuracy:** At epoch 24, v1 val_acc ≈ 0.624, v1.1 ≈ 0.619. Final v1.1 (epoch 49) val_acc ≈ 0.625 — almost no gain from extra epochs.
- **Per-action accuracy (validation):** Main predictive power is in **accel**, **left+accel**, **right+accel**, **coast**, **left+accel+brake**, **right+accel+brake**. Actions **left**, **right**, **brake**, **left+brake**, **right+brake**, **accel+brake** have near-zero or zero samples in validation, so val_acc_class_* is 0 or very small.
- **Combined (loss + per-action accuracy):** The script reports **best epoch by val_loss** and at that epoch: val_loss, val_acc, and **main_actions_val_acc** (mean over the six actions above). For v1 the minimum val_loss is at **epoch 14** (val_loss 0.997, main_actions_val_acc 0.49); for v1.1 at **epoch 11** (val_loss 0.994, main_actions_val_acc 0.42). At **later** epochs (e.g. 24), overall val_acc and per-action accuracies are **higher** (e.g. v1 at 24: val_acc 0.62, accel/left+accel ~0.71, coast ~0.58), while val_loss is slightly higher. So: stopping at **min val_loss** gives the lowest loss but **lower** per-action accuracy; stopping around **20–25 epochs** gives a better **trade-off** (good loss and higher accuracy on main actions). Prefer early stopping with **patience 5–10** so the run stops after val_loss flattens but before overfitting, typically around 20–25 epochs.
- **Conclusion:** **Early stopping around 20–25 epochs** is sufficient; training to 50 epochs (v1.1) does not improve val loss or val accuracy and leads to overfitting. Use early stopping (e.g. on val_loss with patience 5–10) to save time. When comparing runs, consider **both** val_loss and per-action accuracy (or main_actions_val_acc).

Run Analysis
------------

- **v1**: BC pretrain from ``output/ptretrain/vis/v1/encoder.pt``, bc_mode=backbone, **25 epochs** (stopped early or 25-epoch config). Val_loss final 1.038, val_acc 0.624. CSV: ``output/ptretrain/bc/v1/csv/metrics.csv`` (or ``csv/version_0/metrics.csv``).
- **v1.1**: Same encoder init and bc_mode, **no early stopping**, lr=0.0005, **50 epochs**. Val_loss final 1.119, val_acc 0.625. CSV: ``output/ptretrain/bc/v1.1/csv/metrics.csv``.

Detailed Metrics Analysis (by Epoch)
-------------------------------------

**Methodology:** Metrics are compared at epoch checkpoints (0, 5, 10, 15, 20, 24) over the **common epoch window** (0–24). Source: Lightning CSV; use ``scripts/analyze_pretrain_bc.py`` to reproduce. Per-action accuracy is reported with **action names** (accel, left+accel, right+accel, coast, left, right, brake, etc.), not only class indices.

Train / val loss and overall accuracy at checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **v1:** epoch 0 — train_loss 1.31, val_loss 1.07, val_acc 0.57; epoch 10 — 0.85 / 1.02 / 0.61; epoch 24 — 0.79 / 1.04 / 0.62.
- **v1.1:** epoch 0 — 1.30 / 1.09 / 0.59; epoch 10 — 0.89 / 1.00 / 0.59; epoch 24 — 0.83 / 1.05 / 0.62.

At epoch 24, v1 has lower val_loss (1.04 vs 1.05) and similar val_acc. v1.1 trained to 50 epochs ends with higher val_loss (1.12).

Combined analysis (loss + per-action accuracy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best epoch by val_loss** (script ``analyze_pretrain_bc.py``):

- **v1:** best epoch = 14 (val_loss 0.997, val_acc 0.61, main_actions_val_acc 0.49).
- **v1.1:** best epoch = 11 (val_loss 0.994, val_acc 0.60, main_actions_val_acc 0.42).

At these epochs val_loss is minimal but **main_actions_val_acc** (mean over accel, left+accel, right+accel, coast, left+accel+brake, right+accel+brake) is **lower** than at epoch 20–24. So stopping purely at min val_loss yields the best loss but **worse** per-action accuracy; a better compromise is to stop around **20–25 epochs** when both loss and per-action acc are good.

Per-action validation accuracy (action names)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At **last epoch** (v1 at 24, v1.1 at 49):

- **accel (0):** v1 0.71, v1.1 0.71
- **left+accel (1):** v1 0.71, v1.1 0.70
- **right+accel (2):** v1 0.52, v1.1 0.53
- **coast (3):** v1 0.58, v1.1 0.41 (v1.1 drops with more training)
- **left (4), right (5), brake (6), left+brake (7), right+brake (8), accel+brake (9):** near zero in both (very few or no val samples)
- **left+accel+brake (10):** v1 0.39, v1.1 0.41
- **right+accel+brake (11):** v1 0.33, v1.1 0.33

So the model learns **accel**, **left+accel**, **right+accel**, **coast**, and the two accel+brake turn actions; rare actions (brake-only, left/right without accel) stay at 0. Training longer (v1.1 to 50 epochs) does not improve these and can slightly hurt coast.

Per-action accuracy vs training (epoch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The figure below shows **validation accuracy for each action** (accel, left+accel, right+accel, coast, etc.) over **epochs** for v1 and v1.1. Main actions (0, 1, 2, 3, 10, 11) show clear learning curves; rare actions (4–9) stay near zero.

.. image:: ../../_static/exp_pretrain_bc_per_action_accuracy.jpg
   :alt: Per-action validation accuracy vs epoch (v1 and v1.1, 12 actions)

Configuration Changes
----------------------

**BC pretrain** (``config_files/pretrain_config_bc.yaml``):

- **v1:** early_stopping enabled (or epochs capped at 25); lr as in config (e.g. 0.0005).
- **v1.1:** early_stopping: false, lr: 0.0005, epochs: 50.

Other settings shared: encoder_init_path: ``output/ptretrain/vis/v1/encoder.pt``, bc_mode: backbone, n_actions: 12, batch_size: 4096, val_fraction: 0.1.

Hardware
--------

- Same as other pretrain/RL runs (single GPU, Windows).

Conclusions
-----------

- **Training length:** **~20–25 epochs** is enough; val_loss and val_acc do not improve after that and val_loss increases by 50 epochs (overfitting).
- **Early stopping:** **Recommended.** Use val_loss with patience 5–10 so runs stop around 20–25 epochs and avoid overfitting.
- **Per-action metrics:** Report and compare **per-action accuracy by name** (accel, left+accel, right+accel, coast, left+accel+brake, right+accel+brake). Rare actions (left, right, brake, left+brake, right+brake, accel+brake) have ~0 val samples; their val_acc_class_* are 0 or negligible.

Recommendations
---------------

- Use **early stopping** on val_loss (patience 5–10) for BC pretrain.
- Do not train beyond **~25 epochs** unless you have evidence of underfitting (e.g. val_loss still decreasing).
- For analysis, use **scripts/analyze_pretrain_bc.py** with run dirs or ``--base-dir output/ptretrain/bc v1 v1.1`` to get tables by epoch, **combined analysis** (best epoch by val_loss + main_actions_val_acc), and **per-action accuracy with human-readable names** (accel, left+accel, right+accel, etc.).
- To generate the per-action accuracy plot: ``python scripts/analyze_pretrain_bc.py output/ptretrain/bc/v1 output/ptretrain/bc/v1.1 --plot --output-dir docs/source/_static`` (saves ``exp_pretrain_bc_per_action_accuracy.jpg``).

**Analysis tools**

- **BC pretrain comparison:** ``python scripts/analyze_pretrain_bc.py output/ptretrain/bc/v1 output/ptretrain/bc/v1.1 --interval 5``
- With explicit CSV: ``python scripts/analyze_pretrain_bc.py --csv output/ptretrain/bc/v1/csv/metrics.csv output/ptretrain/bc/v1.1/csv/metrics.csv``
- With base dir: ``python scripts/analyze_pretrain_bc.py --base-dir output/ptretrain/bc v1 v1.1``
