Pretrain Experiments
===================

This section documents **pretraining** experiments (Level 0 visual, Level 1 BC, etc.).
Metrics and logs differ from RL: pretrain uses **epoch-based** training, **CSV** from PyTorch Lightning (and optionally TensorBoard), and metrics such as train/val loss, overall accuracy, and **per-action accuracy** (e.g. accel, left+accel, right+accel, brake).

**Analysis:** Use ``scripts/analyze_pretrain_bc.py`` for BC pretrain runs. It reads Lightning CSV from run directories and reports loss/accuracy by epoch and **per-action accuracy with human-readable names** (not just class 0, 1, 2). See the command doc for full options.

Contents
--------

.. toctree::
   :maxdepth: 2

   behavioral_cloning
