# Experiment Documentation

**All experiment documentation instructions live in a single place:**

**.cursor/commands/doc_exp.md**

Use that file for:

- How to create or update experiment pages in `docs/source/experiments/`
- Relative-time vs last-value comparison (when run durations differ)
- Which scripts to run (`analyze_experiment_by_relative_time.py`, `analyze_experiment.py`, etc.)
- Required sections, template, key metrics, common pitfalls
- One-topic-per-file rule (e.g. training_speed.rst for batch/speed/collectors)

Do not duplicate or override those instructions here.

---

**For running new experiments (config, run, analyze, document):**

**.cursor/commands/run_exp.md**

Use that file when you run a new experiment: new config only (no editing existing configs), always use .venv, mandatory full analysis comparing all relevant runs, and doc updates. See run_exp.md for the full checklist and BC pretrain commands.
