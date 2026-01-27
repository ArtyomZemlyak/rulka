# Create Experiment Documentation

## Command: Create Experiment Page

When the user asks to document an experiment, follow these steps.

## Experiment Documents (one file per topic)

Experiments live in ``docs/source/experiments/``. **Each file covers one topic**; the toctree in ``index.rst`` lists all of them.

- **training_speed.rst** — topic: batch_size and running_speed (uni_5, uni_6, uni_7, uni_8, uni_9, etc.). If the experiment is about batch or game speed, **add or edit this file**; do not create a separate file for the same topic.
- **Other topics** — for a new theme (e.g. learning rate, network size, maps), **create** ``docs/source/experiments/<experiment_name>.rst`` and add it to the index toctree.

So: same topic → same file (e.g. training_speed); new topic → new file + new index entry.

### Step 1: Gather Information

Ask or extract (firstly try extract):
- **Run names**: Baseline and experimental runs (e.g., uni_5, uni_7)
- **Experiment description**: What was tested
- **Configuration changes**: What parameters were changed
- **Hardware**: GPU model, system specs
- **Results**: Key findings from the user

### Step 2: Extract TensorBoard Data

**Environment:** Python is usually available only inside the project venv. Always activate it before running scripts.

- **Windows (PowerShell):** ``.\.venv\Scripts\Activate.ps1`` or ``.\.venv\Scripts\activate``
- **Unix:** ``source .venv/bin/activate``

**Paths:** TensorBoard logs live in subdirs ``tensorboard\uni_<N>`` (e.g. ``tensorboard\uni_9``). The script expects these under the **project root** by default. If running from another directory, pass ``--logdir "C:\...\rulka\tensorboard"``.

**Run the analysis:**

From project root (recommended):

```powershell
# Windows PowerShell — chain with ; not &&
cd "C:\...\rulka"
.\.venv\Scripts\Activate.ps1
python scripts/analyze_experiment.py <baseline_run> <experimental_run> [additional_runs]
```

Or in one line (PowerShell): ``.\.venv\Scripts\Activate.ps1; python scripts/analyze_experiment.py uni_5 uni_7 uni_8 uni_9``

With explicit log dir (from any cwd):

```bash
python scripts/analyze_experiment.py --logdir "C:\...\rulka\tensorboard" uni_5 uni_7 uni_8 uni_9
```

Save the full console output; use it to fill metrics in the RST (Hock/A01 best times, loss, Q-values, GPU %).

### Step 3: Check Configuration Files

Read relevant config files to document changes:
- `config_files/training_config.py` - batch_size, learning rates, schedules
- `config_files/performance_config.py` - running_speed, gpu_collectors_count
- `config_files/environment_config.py` - if environment params changed
- `config_files/neural_network_config.py` - if network architecture changed

### Step 4: Create or Update RST File

**Choose the target file:**

- **Batch / running_speed** → use or update ``docs/source/experiments/training_speed.rst`` (add new runs, new subsections, new conclusions; avoid duplicating existing runs).
- **Any other topic** → create ``docs/source/experiments/<experiment_name>.rst`` (e.g. ``learning_rate.rst``, ``network_size.rst``) and fill it from the template below.

Template for a **new** experiment page (for a new topic; for training_speed, adapt structure and append without duplicating):

```rst
Experiment: <Title>
==================================

Experiment Overview
-------------------

This experiment tested the effect of <changes> on <metrics>.

<Brief hypothesis or goal>

Results
-------

**Key Findings:**

- Finding 1
- Finding 2
- Finding 3

Run Analysis
------------

- **<baseline_run>**: Description with config values
- **<experimental_run>**: Description with config values

Detailed TensorBoard Metrics Analysis
-------------------------------------

The following analysis is based on extracted metrics from TensorBoard logs.

<Map> Map Performance
~~~~~~~~~~~~~~~~~~~~

- **<baseline>**: Best time: X.XXXs at step N,NNN,NNN
- **<experimental>**: Best time: X.XXXs at step N,NNN,NNN

**Convergence speed improvement**: <experimental> reached similar performance in **X% fewer steps**.

Training Loss
~~~~~~~~~~~~~~

- **<baseline>**: Final loss: XXX.XX at step N,NNN,NNN
- **<experimental>**: Final loss: XXX.XX at step N,NNN,NNN
- **Loss ratio**: X.XX× (if applicable)

Average Q-values
~~~~~~~~~~~~~~~

- **<baseline>**: -X.XXXX at step N,NNN,NNN
- **<experimental>**: -X.XXXX at step N,NNN,NNN

GPU Utilization
~~~~~~~~~~~~~~~~

- **<baseline>**: XX.X% training time
- **<experimental>**: XX.X% training time
- **Difference**: ±X.X%

Configuration Changes
----------------------

**Training Configuration** (``config_files/training_config.py``):

.. code-block:: python

   <parameter> = <value>  # Changed from <old_value>

**Performance Configuration** (``config_files/performance_config.py``):

.. code-block:: python

   <parameter> = <value>  # Changed from <old_value>

Hardware
--------

- **GPU**: <model>
- **Parallel instances**: <count>
- **System**: <specs if relevant>

Conclusions
-----------

<Detailed conclusions with root causes>

Recommendations
---------------

<Recommendations based on results>

**Analysis Tools:**
- Activate venv first (Windows: ``.\.venv\Scripts\activate``).
- From project root: ``python scripts/analyze_experiment.py <run1> <run2> ...`` (logs in ``tensorboard\uni_<N>``).
- From another directory: ``--logdir "C:\...\rulka\tensorboard"``.
- Use ``scripts/extract_tensorboard_data.py`` to extract specific metrics.
- Key metrics (see ``docs/source/tensorboard_metrics.rst``): ``Training/loss``, ``Race/eval_race_time_robust_*``, ``RL/avg_Q_*``, ``Performance/learner_percentage_training``, ``Performance/transitions_learned_per_second``, ``Gradients/norm_before_clip_max`` (stability).
```

### Step 5: Update Index (only for new topics)

The toctree in ``docs/source/experiments/index.rst`` must list **every** experiment page. When you **add a new topic** (new .rst file), add one line to the toctree:

```rst
.. toctree::
   :maxdepth: 2

   training_speed
   <new_experiment_name>   # add this line when you created a new file
```

When you **only update** an existing file (e.g. training_speed.rst), do **not** change the index — the entry is already there.

### Step 6: Verify

- Check that all metrics are extracted correctly
- Verify configuration values match actual config files
- Ensure step numbers and values are accurate
- Check formatting (RST syntax)

## Example Usage

**Example 1 — batch/speed (same topic, existing file):**

User: "Провел эксп uni_7 - в нем уменьшил до 512 батч и еще увеличил с 160 до 512 скорость игры. В итоге сходится быстрее относительно шагов - причем на обоих картах. Давай это тоже добавим"

Response:
1. Activate venv, then run: `python scripts/analyze_experiment.py uni_5 uni_7` (from project root, or with `--logdir` path to `tensorboard`)
2. Read config files to get exact values
3. **Update** ``training_speed.rst`` — add metrics and conclusions for uni_7; do not create a new file (batch/speed → training_speed).
4. Do **not** change ``experiments/index.rst`` (training_speed is already in the toctree).
5. Document findings with specific numbers from TensorBoard.

**Example 2 — new topic (new file + index):**

User: "Провел эксперимент по learning rate, runs lr_1 и lr_2, добавь в доки"

Response:
1. Run analysis for lr_1, lr_2; read configs.
2. **Create** ``docs/source/experiments/learning_rate.rst`` from the template (new topic = new file).
3. **Add** ``learning_rate`` to the toctree in ``docs/source/experiments/index.rst``.
4. Fill with metrics and conclusions.

## Key Metrics to Always Include

Align with ``docs/source/tensorboard_metrics.rst``. Prioritise:

1. **Map performance**: ``Race/eval_race_time_robust_*`` (primary), ``alltime_min_ms_{map}`` — best times per map (hock, A01, etc.)
2. **Training loss**: ``Training/loss`` — learning quality; early rise is normal, then stabilize/decrease
3. **Q-values**: ``RL/avg_Q_*`` — learning progress; trend up after exploration
4. **GPU / throughput**: ``Performance/learner_percentage_training`` (>70% target), ``Performance/transitions_learned_per_second`` — efficiency
5. **Convergence speed**: Steps/frames to reach target performance
6. **Training stability** (from tensorboard_metrics): ``Gradients/norm_before_clip_max`` — watch for spikes >100 (explosion); keep in doc when comparing setups

**From tensorboard_metrics.rst “Key Metrics to Monitor”:** ``Race/eval_race_time_robust``, ``RL/avg_Q``, ``Gradients/norm_before_clip_max``, ``Training/loss``, ``Performance/transitions_learned_per_second``. For interpretation and “what to watch for”, see that file.

## Notes

- Always use actual data from TensorBoard, not estimates
- Include step numbers for reproducibility
- Document both absolute values and relative improvements
- Note any hardware differences
- Explain root causes of results, not just numbers

## Common Pitfalls (for the agent)

1. **Python not found:** The environment often has no global `python`; use the project venv. Run ``.\.venv\Scripts\Activate.ps1`` (Windows) or ``source .venv/bin/activate`` (Unix) before ``python scripts/...``.

2. **PowerShell vs bash:** On Windows PowerShell, ``cmd1 && cmd2`` fails. Use ``cmd1; cmd2`` or run commands separately.

3. **TensorBoard path:** Logs are in ``tensorboard\uni_5``, ``tensorboard\uni_7``, etc. The script resolves them relative to cwd by default. If cwd is not the project root, pass ``--logdir "<abs_path_to_tensorboard>"``.

4. **Running the script yourself:** When documenting, run the analysis script *in the same session* (with venv active and correct cwd or ``--logdir``), then paste the printed metrics into the RST. Do not skip running the script and leave placeholder text like "run the script to get numbers."

5. **One topic = one file:** Experiments on batch_size / running_speed go into ``training_speed.rst`` only. Do not create ``training_speed_2.rst`` or separate files for new runs of the same topic; add sections/runs to the existing file. Create a **new** file only for a **new** topic (e.g. learning rate, network size), and then add it to the index toctree.
