Experiments
===========

This section documents various experiments conducted in the project, their results, and conclusions.

**Analysis:** All comparisons are by **relative time** (minutes from run start). Use ``scripts/analyze_experiment_by_relative_time.py`` with **two or more runs** (e.g. ``uni_5 uni_7`` or ``uni_12 uni_13 uni_14``). The script prints per-race tables (best/mean/std, finish rate, first finish) from ``Race/eval_race_time_*`` and ``Race/explo_race_time_*``, then scalar metrics (``alltime_min_ms_*``, loss, Q, GPU %). For runs logged before the learner_process fix, prefer per-race tables for race-time comparison.

Contents
--------

.. toctree::
   :maxdepth: 2

   training_speed
   temporal_mini_race_duration
   exploration
   models/iqn