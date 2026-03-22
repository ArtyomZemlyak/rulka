Experiments
===========

This section documents various experiments conducted in the project, their results, and conclusions.

**Analysis — time axis:** Use ``scripts/analyze_experiment_by_relative_time.py`` with **two or more runs** (e.g. ``uni_5 uni_7``). Default ``--time-axis`` is **auto**: the script uses **cumulative training hours** (TensorBoard scalar ``cumul_training_hours``, same idea as console ``Training hours``) when every run logs it; otherwise it falls back to **wall-clock minutes** from the earliest TensorBoard ``wall_time`` in the merged run (includes idle gaps between restarts).

For any write-up, **do not** describe wall-span minutes as “training minutes” when the learner was stopped between sessions or logs are split across several TB folders. Check gaps with ``python scripts/audit_tensorboard_training_timeline.py`` (optional ``--runs …``). Short single-session ``uni_*`` runs usually have wall time close to active training time; long A01 runs with suffix merges often do **not**.

The script prints per-race tables (best/mean/std, finish rate, first finish) from ``Race/eval_race_time_*`` / ``Race/explo_race_time_*``, then scalar metrics (``alltime_min_ms_*``, loss, Q, GPU %). **BY STEP** tables compare equal environment steps regardless of wall time. For runs logged before the learner fix, prefer per-race tables for race-time comparison.

**Comparison plots:** Each experiment page embeds JPG graphs (one metric per graph, runs as lines) **next to the metric they illustrate** in "Detailed TensorBoard Metrics Analysis". Each image has an **alt text** (caption) describing the metric and runs. The image files (``exp_*.jpg`` in ``docs/source/_static/``) are **generated** by running ``python scripts/generate_experiment_plots.py`` (with TensorBoard logs present, e.g. ``tensorboard/uni_12``) and **should be committed** so the built docs include the plots. Use the project venv; if activation fails, run ``.venv\Scripts\python.exe scripts/generate_experiment_plots.py`` (Windows).

Contents
--------

.. toctree::
   :maxdepth: 2

   time_axis_conventions
   training_speed
   reward_shaping
   extended_training
   temporal_mini_race_duration
   exploration
   global_schedule_speed
   multi_action_prediction
   iqn_modernization_plan
   network_size_long_training
   pretrain_replay_roadmap
   pretrain_visual_backbone
   pretrain_bc
   pretrain/index
   iqn_no_image_head
   models/iqn_architecture
   models/iqn