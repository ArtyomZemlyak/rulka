# Create Experiment Documentation

## Command: Create Experiment Page

When the user asks to document an experiment, follow this process. **All experiment docs must be in English.**

---

## Experiment Documents (one file per topic)

Experiments live in ``docs/source/experiments/``. **Each file covers one topic**; the toctree in ``docs/source/experiments/index.rst`` lists all of them.

- **training_speed.rst** — topic: batch_size, running_speed, gpu_collectors_count (uni_5, uni_6, uni_7, uni_8, uni_9, uni_10, etc.). If the experiment is about batch, game speed, or collector count, **add or edit this file**; do not create a separate file for the same topic.
- **Other topics** — for a new theme (e.g. learning rate, network size, maps), **create** ``docs/source/experiments/<experiment_name>.rst`` and add it to the index toctree.

**Rule:** same topic → same file (e.g. training_speed); new topic → new file + new index entry.

---

## Required Sections (checklist)

When creating or updating an experiment page, ensure it has:

1. **Experiment Overview** — what was tested, hypothesis/goal, key parameters changed  
2. **Results** — key findings (bullet points), main conclusions  
3. **Comparison plots** (optional but recommended) — JPG graphs: one metric per graph, runs as lines; generate with ``scripts/plot_experiment_comparison.py`` or ``scripts/generate_experiment_plots.py``; **embed in the right context with ``:alt:`` captions**. See "Comparison plots (JPG)" and **"Embedding plots in RST (quality)"** below.  
4. **Run Analysis** — list of runs compared, baseline vs experimental; **durations (relative time)** per run  
5. **Detailed TensorBoard Metrics Analysis** — by **relative time** and by **steps** (see below); not only “last value” when durations differ  
6. **Configuration Changes** — training section, performance section in config YAML, etc.  
7. **Hardware** — GPU, parallel instances, system  
8. **Conclusions** — what worked/didn’t, root causes, trade-offs  
9. **Recommendations** — optimal settings, when to change, analysis tools  

---

## Relative Time and By Steps (both mandatory)

**Experiments often run for different wall-clock times.** Comparing only “last value” or “final loss at step N” across runs is **invalid** when one run lasted 80 min and another 160 min.

- **Compare by relative time** — minutes from run start. Use checkpoints 5, 10, 15, 20, … min; compare only up to when the **shortest** of the compared runs is still going.
- **Script:** ``scripts/analyze_experiment_by_relative_time.py``. Primary tool; supports **2+ runs** (e.g. ``uni_5 uni_7`` or ``uni_12 uni_13 uni_14``). With ``--plot`` and ``--output-dir`` it also saves **comparison plots as JPG** (one metric per graph, multiple runs as lines; see "Comparison plots" below).
- **At each checkpoint:**
  - **Race times (preferred):** per-race events ``Race/eval_race_time_*``, ``Race/explo_race_time_*`` — script prints **best / mean / std**, best among finished, **finish rate**, **first finish** (one run-wide t0 per run). Use these tables for dynamics and stability.
  - **Scalar race times:** ``alltime_min_ms_{map}`` = best so far at that moment (for runs before learner_process fix, prefer per-race tables; scalar may be wrong).
  - **Loss, Q, GPU %**: use **last value at that moment**.
- **Always document run durations** in the RST, e.g. “uni_5 ~160 min, uni_7 ~86 min”, and state that conclusions are “by relative time”.
- **Wording in docs:** “at 60 min”, “at 70 min”, “common window up to 85 min”, “by relative time”.

- **Relative time and by steps (default for comparisons):**  
  ``python scripts/analyze_experiment_by_relative_time.py <run1> <run2> [<run3> ...] [--interval 5] [--step_interval 50000]``  
  Output: (1) tables by **min**, then (2) **BY STEP** tables by training step. Use for both "same wall-clock" and "same steps". Use when runs had different lengths or when you care about “same wall-clock time” comparison. Output: per-race tables (best/mean/std, finish rate, first finish) then scalar metrics.
### By steps (mandatory as well)

- **Compare by steps** — training step checkpoints (e.g. 50k, 100k, 150k). Use the **BY STEP** tables printed by the same script; compare only up to the **smallest** max step among runs (equal number of gradient updates).
- **At each checkpoint (step):** same metrics — best/mean/std race times, finish rate, first finish step; scalar loss, Q, GPU % at that step.
- **Wording in docs:** "at 100k steps", "common step window up to 200k steps", "by steps".

**Script (outputs both):**

- **Script:** ``scripts/analyze_experiment_by_relative_time.py``. **Outputs both** relative-time tables and **BY STEP** tables in one run.
- **Command:**  
  ``python scripts/analyze_experiment_by_relative_time.py <run1> <run2> [<run3> ...] [--interval 5] [--step_interval 50000]``  
  Use when runs had different lengths or when you care about "same wall-clock" and "same steps". Output: (1) per-race and scalar tables by **min**, then (2) same by **step** (BY STEP section).
- **With plots:** add ``--plot --output-dir docs/source/_static --prefix exp_<name>_<runs>`` to save JPG comparison graphs (one metric per file, runs as lines). Example: ``python scripts/analyze_experiment_by_relative_time.py uni_12 uni_15 --plot --output-dir docs/source/_static --prefix exp_exploration_uni12_uni15``.
- **Optional:** ``python scripts/analyze_experiment.py <run1> <run2> ...`` for last-value-only comparison (less meaningful when durations differ).

---

## Comparison plots (JPG)

**Rule:** One graph = **one metric** (e.g. loss vs time, or A01 best vs time); **multiple runs** = multiple lines on the same graph. Plots are saved as **compressed JPG** to keep repo size small. The plotting script uses **robust y-axis scaling** (percentiles) so outliers (e.g. 300s at start dropping to 30s) do not squash the readable range.

- **Single comparison:** ``python scripts/plot_experiment_comparison.py <run1> <run2> [--prefix exp_<name>] [--output-dir docs/source/_static]`` — builds graphs from TensorBoard data, saves JPG. Use ``--by-step`` to also generate by-step plots.
- **All documented experiments:** ``python scripts/generate_experiment_plots.py`` — runs plot for every experiment in ``EXPERIMENTS`` (exploration, temporal_mini_race_duration, training_speed sets, iqn pairs). Default output: ``docs/source/_static``. Use ``--experiments exploration temporal_mini_race_duration`` to limit.
- **Regenerate after script changes:** If the plotting script (e.g. ``experiment_plot_utils.py`` or ``analyze_experiment_by_relative_time.py``) was updated (e.g. better scaling, new metrics), run ``generate_experiment_plots.py`` again and **commit** the updated ``docs/source/_static/exp_*.jpg`` so the docs use the new graphs.
- **Generate before building docs:** Run ``generate_experiment_plots.py`` (with TensorBoard logs present) so that ``_static/*.jpg`` exist. **Commit the generated JPG files** to the repo so the built docs include the plots; TensorBoard logs and ``save/`` are not committed, but the plot images are.
- **Run script:** Use the project venv. If PowerShell blocks ``.\.venv\Scripts\Activate.ps1``, run directly: ``.venv\Scripts\python.exe scripts/generate_experiment_plots.py`` (Windows) or ``.venv/bin/python scripts/generate_experiment_plots.py`` (Unix).

---

## Embedding plots in RST (quality)

When adding or updating comparison plots in experiment pages, follow these rules so the next agent and readers get **correct, contextual, and explained** figures.

**1. Place each image next to the metric it illustrates**

- Insert the ``.. image::`` block **immediately after** the subsection that describes that metric (e.g. after "A01 (per-race eval_race_time_trained_A01)", after "Training loss", after "Average Q-values").
- Do **not** group all images at the end of the page; each figure should sit right under the text that discusses the same metric.

**2. Every image must have an alt caption (``:alt:``)**

- Add ``:alt:`` under ``.. image::`` with a **short descriptive caption**: metric name, runs compared, and optionally experiment context.
- Example: ``:alt: A01 eval best time by relative time (uni_12 vs uni_15)`` or ``:alt: Training loss by relative time (uni_5 vs uni_7)`` or ``:alt: Avg Q by relative time (uni_12 vs uni_16, DDQN reduces overestimation)``.
- This improves accessibility and makes it clear what the figure shows when the image is missing or in a different build.

**3. Intro sentence in "Detailed TensorBoard Metrics Analysis"**

- At the **start** of the "Detailed TensorBoard Metrics Analysis" section (right after the methodology paragraph), add **one sentence** explaining that the figures below show one metric per graph, runs as lines, by relative time.
- Example: "The figures below show one metric per graph (runs as lines, by relative time)." or "The figures below illustrate each metric (one graph per metric, runs as lines, by relative time)."

**4. Paths and file names**

- From ``docs/source/experiments/<file>.rst`` use ``.. image:: ../_static/<prefix>_<metric>.jpg``.
- From ``docs/source/experiments/models/<file>.rst`` use ``.. image:: ../../_static/<prefix>_<metric>.jpg``.
- File names are generated by the script: e.g. ``exp_exploration_uni12_uni15_A01_best.jpg``, ``exp_exploration_uni12_uni15_loss.jpg``, ``exp_iqn_uni12_uni16_avg_q.jpg``. Metric slugs: ``A01_best``, ``hock_best``, ``loss``, ``avg_q``, ``training_pct``, ``A01_rate``, ``hock_rate``, etc. (see ``scripts/experiment_plot_utils.py`` and generated files in ``_static``).

**5. Which plots to embed**

- Prefer embedding **key metrics** discussed in the text: e.g. A01 best, Hock best, loss, avg_q for the runs being compared in that subsection. If the page has multiple comparisons (e.g. uni_5 vs uni_7 and uni_7 vs uni_12), add the **relevant** plot after each comparison (e.g. ``exp_training_speed_uni5_uni7_*`` after uni_5 vs uni_7, ``exp_training_speed_uni7_uni12_*`` after uni_7 vs uni_12).
- For multi-experiment pages (e.g. ``models/iqn.rst`` with Exp 1–5), add images for **each** experiment in the right subsection (Exp 1: uni_12 vs uni_16; Exp 2: uni_16 vs uni_17; etc.).

**6. Checklist before finishing**

- [ ] Every ``.. image::`` has a ``:alt:`` line.
- [ ] Each image is placed right after the metric subsection it illustrates.
- [ ] "Detailed TensorBoard Metrics Analysis" starts with a sentence that the figures below show one metric per graph, runs as lines, by relative time.
- [ ] Paths are correct (``../_static/`` vs ``../../_static/`` depending on file location).
- [ ] File names match the generated JPGs (run ``generate_experiment_plots.py`` if new experiments were added).

---

## Step 1: Gather Information

Ask or extract (prefer extract first):

- **Run names**: Baseline and experimental runs (e.g. uni_5, uni_7)
- **Experiment description**: What was tested
- **Configuration changes**: Which parameters changed
- **Hardware**: GPU model, number of collectors, system specs
- **Results**: Key findings from the user
- **Run durations**: If unknown, get them from the relative-time script (it prints duration per run).

---

## Step 2: Extract TensorBoard Data

**Environment:** Use the project venv. Activate it before running any Python script.

- **Windows (PowerShell):** ``.\.venv\Scripts\Activate.ps1`` or ``.\.venv\Scripts\activate``  
- **Unix:** ``source .venv/bin/activate``

**Paths:** TensorBoard logs are in ``tensorboard\uni_<N>`` (e.g. ``tensorboard\uni_7``). Scripts assume **project root** by default. From another directory, pass ``--logdir "C:\...\rulka\tensorboard"``.

**Preferred: comparison by relative time**

From project root:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/analyze_experiment_by_relative_time.py uni_5 uni_7 --interval 5
# or 3+ runs: python scripts/analyze_experiment_by_relative_time.py uni_12 uni_13 uni_14 --interval 5
```

With explicit log dir:

```bash
python scripts/analyze_experiment_by_relative_time.py --logdir "C:\...\rulka\tensorboard" uni_5 uni_7 --interval 5
```

Save the full console output. Use it to fill the RST: for each comparison subsection, report values **at 5, 10, 20, … min** (relative time) **and at 50k, 100k, … steps** (by steps; use ``--step_interval 50000``), and note the **common window** for both (e.g. “common window up to 85 min”).

**Optional: last-value comparison** (only if runs have the same duration or you need step-based numbers):

```powershell
python scripts/analyze_experiment.py uni_5 uni_7 uni_8 uni_9
```

---

## Step 3: Check Configuration Files

Read the relevant configs to document parameter changes:

- ``config_files/config_default.yaml`` (or the YAML used for the run): ``training`` section (batch_size, learning rates, schedules), ``performance`` (running_speed, gpu_collectors_count), ``environment`` (if env params changed), ``neural_network`` (if architecture changed)  

---

## Step 4: Create or Update RST File

**Choose the target file:**

- **Batch / running_speed / gpu_collectors_count** → use or update ``docs/source/experiments/training_speed.rst`` (add new runs, subsections, conclusions; do not duplicate existing runs).
- **Any other topic** → create ``docs/source/experiments/<experiment_name>.rst`` and add it to the index toctree.

**Template for a new experiment page** (new topic only; for training_speed, adapt and append):

```rst
Experiment: <Title>
==================================

Experiment Overview
-------------------

This experiment tested the effect of <changes> on <metrics>.

<Brief hypothesis or goal>

Results
-------

**Important:** If run durations differed, all findings below are by **relative time** (minutes from run start). Comparing by “last value” is invalid.

**Key Findings:**

- Finding 1
- Finding 2
- Finding 3

Run Analysis
------------

- **<baseline_run>**: Config summary, **~X min** (relative time)
- **<experimental_run>**: Config summary, **~Y min** (relative time)

Detailed TensorBoard Metrics Analysis
-------------------------------------

**Methodology — Relative time and by steps:** Metrics are compared (1) at checkpoints 5, 10, 15, 20, … min (only up to the shortest run) and (2) at step checkpoints 50k, 100k, … (only up to the smallest max step). For race times — best so far at that checkpoint; for loss/Q/GPU% — last value at that moment. Tables: ``python scripts/analyze_experiment_by_relative_time.py <run1> <run2> [--interval 5] [--step_interval 50000]`` (outputs both relative-time and BY STEP tables). The figures below show one metric per graph (runs as lines, by relative time).

<Map> Map Performance (e.g. common window up to 85 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **<baseline>**: at 60 min — X.XXs; at 85 min — X.XXs
- **<experimental>**: at 60 min — X.XXs; at 85 min — X.XXs

.. image:: ../_static/exp_<name>_<runs>_<map_slug>_best.jpg
   :alt: <Map> best time by relative time (<run1> vs <run2>)

Training Loss
~~~~~~~~~~~~~~

- **<baseline>**: at 85 min — XXX.X
- **<experimental>**: at 85 min — XXX.X

.. image:: ../_static/exp_<name>_<runs>_loss.jpg
   :alt: Training loss by relative time (<run1> vs <run2>)

Average Q-values
~~~~~~~~~~~~~~~~

- **<baseline>**: at 85 min — -X.XX
- **<experimental>**: at 85 min — -X.XX

.. image:: ../_static/exp_<name>_<runs>_avg_q.jpg
   :alt: Avg Q by relative time (<run1> vs <run2>)

GPU Utilization
~~~~~~~~~~~~~~~

- **<baseline>**: ~XX% over the window
- **<experimental>**: ~XX% over the window

Configuration Changes
----------------------

**Training** (``training`` section in config YAML):

.. code-block:: python

   <parameter> = <value>  # e.g. batch_size = 512

**Performance** (``performance`` section in config YAML):

.. code-block:: python

   <parameter> = <value>  # e.g. running_speed = 512, gpu_collectors_count = 8

Hardware
--------

- **GPU**: <model>
- **Parallel instances**: <count> (e.g. 8 collectors; uni_10 used 4)
- **System**: <specs if relevant>

Conclusions
-----------

<Conclusions by relative time; do not rely on “last value” when durations differ.>

Recommendations
---------------

<Recommendations and when to change settings.>

**Analysis Tools:**

- By **relative time and by steps**: ``python scripts/analyze_experiment_by_relative_time.py <run1> <run2> [--interval 5] [--step_interval 50000]`` (``--logdir "<path>"`` if not from project root). Outputs both relative-time and BY STEP tables.
- By “last value”: ``python scripts/analyze_experiment.py <run1> <run2> ...`` (less meaningful when durations differ).
- ``scripts/extract_tensorboard_data.py`` — specific metrics (e.g. ``Gradients/norm_before_clip_max``, ``Performance/transitions_learned_per_second``).
- Key metrics (see ``docs/source/tensorboard_metrics.rst``): Per-race ``Race/eval_race_time_*``, ``Race/explo_race_time_*``; scalars ``Training/loss``, ``alltime_min_ms_{map}``, ``RL/avg_Q_*``, ``Performance/learner_percentage_training``, ``Gradients/norm_before_clip_max``.
```

---

## Step 5: Update Index (only for new topics)

The toctree in ``docs/source/experiments/index.rst`` must list every experiment page. When you **add a new topic** (new .rst file), add one line:

```rst
.. toctree::
   :maxdepth: 2

   training_speed
   <new_experiment_name>
```

When you **only update** an existing file (e.g. training_speed.rst), do **not** change the index.

---

## Step 6: Verify

- All numbers come from script output (relative-time **and** step-based, or last-value as appropriate), not placeholders.
- Configuration values match the actual config files.
- Wording uses “at X min”, “common window up to X min”, “by relative time” when run durations differed.
- Docs are in **English**.
- RST formatting is correct.
- **Comparison plots:** Every ``.. image::`` has ``:alt:``; each image is placed right after the metric subsection it illustrates; "Detailed TensorBoard Metrics Analysis" has an intro sentence that the figures show one metric per graph, runs as lines, by relative time; paths are correct (``../_static/`` from ``experiments/*.rst``, ``../../_static/`` from ``experiments/models/*.rst``). See "Embedding plots in RST (quality)".

---

## Analysis Scripts

| Purpose | Command |
|--------|--------|
| **Compare by relative time and by steps** (primary when durations differ; 2+ runs) | ``python scripts/analyze_experiment_by_relative_time.py <run1> <run2> [<run3> ...] [--interval 5] [--step_interval 50000]`` — outputs both relative-time tables and BY STEP tables |
| **Same + save comparison plots (JPG)** | Add ``--plot --output-dir docs/source/_static --prefix exp_<name>`` to the command above; one metric per graph, runs as lines |
| **Build comparison plots only** (one metric per graph, runs as lines) | ``python scripts/plot_experiment_comparison.py <run1> <run2> [--prefix exp_<name>] [--output-dir docs/source/_static]`` — optional ``--by-step`` |
| **Generate plots for all documented experiments** | ``python scripts/generate_experiment_plots.py`` (default output ``docs/source/_static``); ``--experiments exploration ...`` to limit |
| Compare by last value only | ``python scripts/analyze_experiment.py <run1> <run2> ...`` |
| Extract specific metrics | ``python scripts/extract_tensorboard_data.py --runs <run1> <run2> --metrics "Race/eval_race_time_robust_hock" "Training/loss"`` |
| Batch-size–specific (legacy) | ``python scripts/analyze_batch_experiment.py`` |

Use ``--logdir "C:\...\rulka\tensorboard"`` when not running from project root.

---

## Key Metrics to Include

Align with ``docs/source/tensorboard_metrics.rst``. Priority:

1. **Map performance (preferred):** Per-race ``Race/eval_race_time_*``, ``Race/explo_race_time_*`` (best/mean/std, finish rate, first finish). Scalar: ``alltime_min_ms_{map}``  
2. **Training loss**: ``Training/loss``  
3. **Q-values**: ``RL/avg_Q_*``  
4. **GPU / throughput**: ``Performance/learner_percentage_training`` (>70% target), ``Performance/transitions_learned_per_second``  
5. **Stability**: ``Gradients/norm_before_clip_max`` — watch for spikes >100  

When comparing, report these **at the same checkpoints** both by relative time (e.g. 20, 40, 60, 70 min) and by steps (e.g. 50k, 100k steps), not only “at last step”.

---

## Common Pitfalls (for the agent)

1. **Python/venv:** Use the project venv. Run ``.\.venv\Scripts\Activate.ps1`` (Windows) or ``source .venv/bin/activate`` (Unix) before any ``python scripts/...``. If PowerShell blocks script execution, run directly: ``.venv\Scripts\python.exe scripts/<script>.py`` (Windows).

2. **PowerShell:** On Windows, ``cmd1 && cmd2`` often fails. Use ``cmd1; cmd2`` or run commands separately.

3. **TensorBoard path:** Logs are in ``tensorboard\uni_<N>``. Scripts use cwd by default. If cwd is not project root, pass ``--logdir "<abs_path_to_tensorboard>"``.

4. **Run the script:** Execute the analysis in the same session (venv + correct cwd or ``--logdir``), then paste real numbers into the RST. Do not leave “run the script to get numbers” placeholders.

5. **One topic = one file:** Batch/speed/collectors → ``training_speed.rst`` only. Do not add ``training_speed_2.rst`` or one file per run for the same topic. New file only for a **new** topic; then add it to the index.

6. **Comparing when durations differ:** Do **not** compare “last value” or “final loss at step N” across runs of different length. Use ``analyze_experiment_by_relative_time.py`` and document metrics **at 5, 10, 20, … min** (relative time) **and at 50k, 100k, … steps** (by steps), and the **common window** for both. State clearly when conclusions are “by relative time”.

7. **Language:** All experiment docs in ``docs/source/experiments/`` must be in **English**.

8. **Unicode in scripts:** If you add or change scripts that print to the console on Windows, avoid characters like ``≈`` or ``—`` in output (cp1252 can fail). Use ``~`` and ``-`` instead.

9. **Comparison plot JPGs:** After running ``generate_experiment_plots.py`` (with TensorBoard logs present), **commit** the new/updated ``docs/source/_static/exp_*.jpg`` files so the built docs show the graphs. TensorBoard and ``save/`` are not committed; the generated plot images are. After **any change to the plotting script** (e.g. robust y-axis, new metrics), regenerate all plots with ``generate_experiment_plots.py`` and commit the updated JPGs.

10. **Embedding plots:** When adding or updating experiment docs, **embed** comparison plots correctly: place each image **right after** the metric subsection it illustrates; add **``:alt:``** to every image with a short caption (metric + runs); add **one intro sentence** at the start of "Detailed TensorBoard Metrics Analysis" that the figures show one metric per graph, runs as lines, by relative time. See "Embedding plots in RST (quality)".

---

## Example Usage

**Example 1 — batch/speed (same topic, existing file):**

User: “Documented uni_7 — batch 512, speed 512, converges faster; add to docs.”

1. Activate venv, run: ``python scripts/analyze_experiment_by_relative_time.py uni_5 uni_7 --interval 5`` or ``uni_12 uni_13 uni_14`` for 3+ runs (from project root or with ``--logdir``).
2. Read config files for exact values.
3. **Update** ``training_speed.rst``: add metrics and conclusions for uni_7 **by relative time** (at X min, common window, durations). Do not create a new file.
4. Do **not** change ``experiments/index.rst``.

**Example 2 — new topic:**

User: “Document experiment on learning rate, runs lr_1 and lr_2.”

1. Run ``analyze_experiment_by_relative_time.py lr_1 lr_2`` (or ``analyze_experiment.py`` if durations are equal); read configs.
2. **Create** ``docs/source/experiments/learning_rate.rst`` from the template.
3. **Add** ``learning_rate`` to the toctree in ``docs/source/experiments/index.rst``.
4. Fill metrics and conclusions; use relative time if run lengths differ.
5. **Add comparison plots:** Run ``plot_experiment_comparison.py lr_1 lr_2 --prefix exp_learning_rate --output-dir docs/source/_static`` (or add the run pair to ``generate_experiment_plots.py`` and run it). Embed each JPG in the RST **right after** the metric subsection it illustrates, with **``:alt:``** caption; add the intro sentence at the start of "Detailed TensorBoard Metrics Analysis". See "Embedding plots in RST (quality)".