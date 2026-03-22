.. Time-axis conventions (linked from experiment pages)

Time axis conventions (experiment write-ups)
==============================================

When this documentation cites **minutes** or a **5, 10, 15, …** checkpoint grid, unless it explicitly says **cumulative training hours**, those numbers are **TensorBoard wall-clock minutes**: minutes since the **earliest** ``wall_time`` in the **merged** run (all suffix dirs ``run``, ``run_2``, …). That timeline **includes** nights and idle time when the learner was not running, so it is **not** the same as “how many minutes the network trained.”

**Cumulative training time** (matches console ``Training hours``) is the scalar ``cumul_training_hours`` in TensorBoard and in ``save/<run>/accumulated_stats.joblib``. The analysis script defaults to ``--time-axis auto``: it uses cumulative hours when every compared run logs that scalar, otherwise wall minutes.

To **reproduce the classic 5-minute wall grid** printed in older write-ups, run explicitly::

   python scripts/analyze_experiment_by_relative_time.py RUN1 RUN2 --time-axis wall_minutes --interval 5

Several TensorBoard folders usually mean a **long** run with suffix rotation; they are **not** proof of restarts. Use ``python scripts/audit_tensorboard_training_timeline.py`` to see whether wall span and ``cumul_training_hours`` diverge.

See also :doc:`experiments/index` and the “Merged log folders” section in :doc:`tensorboard_metrics`.

Audit snapshot (documented runs, local ``tensorboard/``)
--------------------------------------------------------

Generated with ``python scripts/audit_tensorboard_training_timeline.py --runs …``. **wall_span_min** = minutes since earliest merged TB ``wall_time`` on ``Training/loss``. **cumul_train_h** = last ``cumul_training_hours``. **ratio** = wall_span_min / (cumul_train_h × 60). **wall>>training** means by-time conclusions that treat wall minutes as “training minutes” are **misleading**.

.. csv-table::
   :header: "run", "TB chunks", "wall min", "cumul h", "ratio", "flag"

   "A01_as20_long_v2", "3", "2898", "17.74", "2.72", "wall>>training"
   "A01_as20_long_v2.1", "3", "3494", "23.92", "2.43", "wall>>training"
   "A01_as20_long_v3.1_pretrained_bc", "4", "3222", "24.51", "2.19", "wall>>training"
   "A01_as20_long_full_iqn_bc_2", "2", "1018", "4.09", "4.15", "wall>>training"
   "A01_as20_long_full_iqn_bc_3", "5", "1162", "19.45", "1.00", "merged_dirs"
   "A01_as20_long_full_iqn_bc", "3", "1310", "3.25", "6.71", "wall>>training"
   "A01_as20_long", "3", "486", "8.18", "0.99", "merged_dirs"
   "A01_as20_long_vis_pretrained", "2", "270", "4.59", "0.98", "merged_dirs"
   "A01_as20_big_long", "3", "561", "9.43", "0.99", "merged_dirs"
   "uni_20_long", "4", "822", "15.87", "0.86", "merged_dirs"

Runs with ratio ≈ 1 still use **wall checkpoints** in older prose; prefer **``--time-axis cumul_training_hours``** or **BY STEP** for clarity.
