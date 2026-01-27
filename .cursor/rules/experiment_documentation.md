# Experiment Documentation Rule

When documenting experiments in the `docs/source/experiments/` directory, follow this structure and process.

## Creating a New Experiment Page

1. **Create the RST file**: `docs/source/experiments/<experiment_name>.rst`
2. **Add to index**: Update `docs/source/experiments/index.rst` to include the new page
3. **Extract TensorBoard data**: Use analysis scripts to get metrics
4. **Document configuration**: Include relevant config changes
5. **Document hardware**: Note GPU, CPU, and system specs

## Required Sections

### 1. Experiment Overview
- Brief description of what was tested
- Hypothesis or goal
- Key parameters changed

### 2. Results
- Key findings (bullet points)
- Main conclusions

### 3. Run Analysis
- List of runs compared
- Baseline vs experimental runs

### 4. Detailed TensorBoard Metrics Analysis
Use `scripts/analyze_experiment.py` to extract data:

```bash
python scripts/analyze_experiment.py <baseline_run> <experimental_run>
```

Include comparisons for:
- **Map Performance**: Best times for each map (hock, A01, etc.)
- **Training Loss**: Final loss values and trends
- **Average Q-values**: Learning progress indicator
- **GPU Utilization**: Training time percentage

### 5. Configuration Changes
Document changes in:
- `config_files/training_config.py` (batch_size, learning rates, etc.)
- `config_files/performance_config.py` (running_speed, collectors, etc.)
- `config_files/environment_config.py` (if changed)
- `config_files/neural_network_config.py` (if changed)

### 6. Hardware
- GPU model
- Number of parallel instances
- System specifications (if relevant)

### 7. Conclusions
- What worked/didn't work
- Root causes of results
- Trade-offs

### 8. Recommendations
- Optimal settings
- When to use different configurations
- Analysis tools to use

## Template

```rst
Experiment: <Title>
==================================

Experiment Overview
-------------------

This experiment tested the effect of <changes> on <metrics>.

Results
-------

**Key Findings:**

- Finding 1
- Finding 2
- Finding 3

Run Analysis
------------

- **baseline_run**: Description
- **experimental_run**: Description

Detailed TensorBoard Metrics Analysis
-------------------------------------

<Map> Map Performance
~~~~~~~~~~~~~~~~~~~~

- **baseline**: Best time: X.XXXs at step N,NNN,NNN
- **experimental**: Best time: X.XXXs at step N,NNN,NNN

Training Loss
~~~~~~~~~~~~~~

- **baseline**: Final loss: XXX.XX at step N,NNN,NNN
- **experimental**: Final loss: XXX.XX at step N,NNN,NNN

Average Q-values
~~~~~~~~~~~~~~~

- **baseline**: -X.XXXX at step N,NNN,NNN
- **experimental**: -X.XXXX at step N,NNN,NNN

GPU Utilization
~~~~~~~~~~~~~~~~

- **baseline**: XX.X% training time
- **experimental**: XX.X% training time

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

<Detailed conclusions>

Recommendations
---------------

<Recommendations>
```

## Analysis Scripts

### Compare Runs
```bash
python scripts/analyze_experiment.py <run1> <run2> [run3] ...
```

### Extract Specific Metrics
```bash
python scripts/extract_tensorboard_data.py --runs <run1> <run2> --metrics "Race/eval_race_time_robust_hock" "Training/loss"
```

### Batch Size Experiment Analysis
```bash
python scripts/analyze_batch_experiment.py
```

## Key Metrics to Monitor

- ``Race/eval_race_time_robust_*`` - Race completion times (primary performance)
- ``Training/loss`` - Training loss (optimization quality)
- ``RL/avg_Q_*`` - Average Q-values (learning progress)
- ``Performance/learner_percentage_training`` - GPU utilization
- ``alltime_min_ms_*`` - Best times per map

## Notes

- Always compare against a baseline run
- Include step numbers for reproducibility
- Document both absolute values and ratios/percentages
- Note any hardware differences between runs
- Include convergence speed analysis when relevant
