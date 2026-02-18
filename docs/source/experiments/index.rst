Experiments
===========

This section documents various experiments conducted in the project, their results, and conclusions.

**Analysis:** All comparisons are by **relative time** (minutes from run start). Use ``scripts/analyze_experiment_by_relative_time.py`` with **two or more runs** (e.g. ``uni_5 uni_7`` or ``uni_12 uni_13 uni_14``). The script prints per-race tables (best/mean/std, finish rate, first finish) from ``Race/eval_race_time_*`` and ``Race/explo_race_time_*``, then scalar metrics (``alltime_min_ms_*``, loss, Q, GPU %). For runs logged before the learner_process fix, prefer per-race tables for race-time comparison.

**Comparison plots:** Each experiment page embeds JPG graphs (one metric per graph, runs as lines) **next to the metric they illustrate** in "Detailed TensorBoard Metrics Analysis". Each image has an **alt text** (caption) describing the metric and runs. The image files (``exp_*.jpg`` in ``docs/source/_static/``) are **generated** by running ``python scripts/generate_experiment_plots.py`` (with TensorBoard logs present, e.g. ``tensorboard/uni_12``) and **should be committed** so the built docs include the plots. Use the project venv; if activation fails, run ``.venv\Scripts\python.exe scripts/generate_experiment_plots.py`` (Windows).

Contents
--------

.. toctree::
   :maxdepth: 2

   training_speed
   extended_training
   temporal_mini_race_duration
   exploration
   network_size_long_training
   pretrain_replay_roadmap
   pretrain_visual_backbone
   models/iqn