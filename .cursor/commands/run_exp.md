# Run Experiment (Full Flow)

## Command: Run New Experiment and Analyze

When the user asks to run an experiment (e.g. new BC pretrain variant, new hyperparameters), follow this flow. **Do not skip steps.** These rules exist so experiments are reproducible, comparable, and documented without errors.

---

## 1. New config only — never edit existing configs in place

**Rule:** For every new experiment you **create a new YAML config file**. Do **not** modify an existing config (e.g. `pretrain_config_bc_v2_multi_offset.yaml`) to add a new variant.

- **Reason:** Existing configs are the baseline for other runs and for doc comparisons. Changing them in place breaks reproducibility and confuses “what was run for run_name X”.

**How:**

- Copy the **closest existing config** (the one you want to differ from by one or few params).
- Save as a **new file** with a descriptive name, e.g.:
  - `pretrain_config_bc_v2_multi_offset_ahead_dropout_inner.yaml`
  - `pretrain_config_bc_v2_multi_offset_ahead_reg.yaml`
- In the new file, change **as few keys as possible** — ideally **one logical change** (e.g. add `dropout: 0.2`, or add `action_head_dropout: 0.1`, or change `lr` + `weight_decay`).
- **Always set a new `run_name`** in the new config so the run is written to a **new directory** (e.g. `run_name: v2_multi_offset_ahead_dropout_inner`). The directory name must be unique and recognizable (no overwriting previous runs).

**Checklist:**

  - [ ] New file created under `config_files/pretrain/bc/`.
- [ ] Only the intended parameters differ from the base config; everything else is identical (data paths, epochs, batch_size, etc.) unless you explicitly want to change them.
- [ ] `run_name` is set and unique.

---

## 2. Always use the project .venv

**Rule:** All Python commands (training, analysis, plotting) must run with the **project virtualenv**. Do not use system `python` or `py` — they may be missing or point to another environment.

**Commands:**

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\python.exe scripts/pretrain_bc.py --config config_files/pretrain/bc/<your_config>.yaml
  .\.venv\Scripts\python.exe scripts/analyze_pretrain_bc.py output/ptretrain/bc/run1 output/ptretrain/bc/run2 --interval 5
  ```
- **Unix:**
  ```bash
  .venv/bin/python scripts/pretrain_bc.py --config config_files/pretrain/bc/<your_config>.yaml
  .venv/bin/python scripts/analyze_pretrain_bc.py output/ptretrain/bc/run1 output/ptretrain/bc/run2 --interval 5
  ```

**PowerShell note:** Chaining with `&&` often fails. Use `;` instead, or run one command per invocation:
- Wrong: `cd ... && .\.venv\Scripts\python.exe ...`
- Right: `cd "C:\...\rulka"; .\.venv\Scripts\python.exe ...`

**Checklist:**

- [ ] Every `python` call uses `.\.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Unix).

---

## 3. Run the experiment

- Execute the training (or RL) command with the **new** config.
- Use the same .venv as above.
- Do not cap epochs or steps unless the user asked (e.g. “50 epochs” is in the config — run all 50 unless early_stopping stops earlier).
- If the run fails (e.g. missing encoder, cache invalid), fix the config or data and re-run; do not document a failed run as success.

**BC pretrain example:**

```powershell
.\.venv\Scripts\python.exe scripts/pretrain_bc.py --config config_files/pretrain/bc/pretrain_config_bc_v2_multi_offset_ahead_dropout_inner.yaml
```

**Checklist:**

- [ ] Training completed (or stopped by early_stopping as configured).
- [ ] Run directory exists, e.g. `output/ptretrain/bc/<run_name>/` with `encoder.pt`, `pretrain_meta.json`, `metrics.csv` (and optionally `actions_head.pt`).

---

## 4. Mandatory: full analysis with comparison to all relevant runs

**Rule:** After every new experiment you **must** run the appropriate analysis script and compare **all relevant** runs (baseline + previous variants + the new run). Do not report only the new run’s metrics without comparison.

**BC pretrain:**

- Script: `scripts/analyze_pretrain_bc.py`.
- Pass **full paths** to run directories (or use `--base-dir output/ptretrain/bc run1 run2 run3 ...`).
- Always use `--interval 5` for epoch checkpoints.
- **Include every run that belongs to the same “family”** (e.g. when testing a new A_head variant, include at least: `v2_multi_offset`, `v2_multi_offset_ahead`, and all tuned/reg/dropout variants you are comparing to).

**Example (A_head family):**

```powershell
.\.venv\Scripts\python.exe scripts/analyze_pretrain_bc.py output/ptretrain/bc/v2_multi_offset output/ptretrain/bc/v2_multi_offset_ahead output/ptretrain/bc/v2_multi_offset_ahead_dropout output/ptretrain/bc/v2_multi_offset_ahead_dropout_inner --interval 5
```

- Save or use the full console output: Run summary, per-offset validation accuracy, train/val loss and acc at checkpoints, best epoch by val_loss, per-action validation accuracy. Use these **exact numbers** in the docs.

**Checklist:**

- [ ] Analysis script was run with **all** relevant run dirs (not only the new one).
- [ ] Output was captured and used for documentation (no invented numbers).

---

## 5. Update documentation

**Rule:** Every new experiment and its comparison must be reflected in the experiment docs. Do not leave the new run only in the config and terminal output.

**Where:**

- **BC / pretrain:** `docs/source/experiments/pretrain/behavioral_cloning.rst` (or the relevant pretrain topic file).
- **RL:** The relevant file under `docs/source/experiments/` (e.g. `training_speed.rst`) and/or `docs/source/experiments/pretrain/`.

**What to add/update:**

1. **Run Analysis** — One line per run: run name, config file, main params (e.g. lr, weight_decay, dropout), key result (e.g. val_acc, val_loss, best epoch), and path (e.g. `output/ptretrain/bc/<run_name>/`).
2. **Comparison table(s)** — If the page has a table (e.g. val_acc, val_loss, per-offset, best epoch), add a column or row for the new run with **numbers from the analysis script**.
3. **Subsection or paragraph** — Short block for the new variant: config name, what changed (single sentence), main metrics (val_acc, val_loss, per-offset, best epoch), and when to use it (e.g. “Recommended for RL merge”).
4. **Recommendations** — Update the “Recommendations” and “Analysis tools” sections: add the new config to the recommended list if it is best; update the **exact** analysis command to include the new run so the next person can reproduce the full comparison.

**Checklist:**

- [ ] New run appears in Run Analysis with config path and key metrics.
- [ ] Comparison tables include the new run with real numbers from the analysis output.
- [ ] New variant has a dedicated subsection or paragraph with config name, change, and metrics.
- [ ] Recommendations and “Analysis tools” command are updated (including the new run in the list of dirs for `analyze_pretrain_bc.py`).

---

## 6. Pretrain-specific rules (BC)

- **Epoch-based:** Comparisons are by **epoch** (e.g. 0, 5, 10, … 49), not by wall-clock time. Use `--interval 5` for checkpoints.
- **Per-action accuracy:** In the doc use **action names** (accel, left+accel, right+accel, coast, left+accel+brake, right+accel+brake), not only class indices. The script `analyze_pretrain_bc.py` prints these names.
- **Best epoch:** Report “best epoch by val_loss” and metrics at that epoch (val_acc, main_actions_val_acc); for overfitting runs (e.g. untuned A_head) the best epoch can be much earlier than the last.
- **Multi-offset:** When comparing multi-offset runs, report **per-offset** val accuracy (e.g. val_acc_offset_ms_-10, val_acc_offset_ms_0, val_acc_offset_ms_10, val_acc_offset_ms_100) from the script output.
- **Val loss:** Multi-offset BC uses a weighted sum of several cross-entropies; val_loss is not directly comparable to single-head BC. Prefer val_acc and per-offset accuracies for comparison.

---

## 7. Common mistakes to avoid

1. **Editing an existing config** instead of creating a new one → breaks baselines and reproducibility.
2. **Using `python` or `py`** instead of `.\.venv\Scripts\python.exe` → wrong env or “command not found”.
3. **Skipping the analysis step** or running analysis only for the new run → no comparison, no tables, no doc updates.
4. **Documenting without running the analysis** → placeholder or wrong numbers in the doc.
5. **Forgetting to add the new run to the analysis command** in the doc → next person cannot reproduce the full comparison.
6. **Comparing “last epoch” only** when one run overfits (best epoch much earlier) → misleading; always report best epoch by val_loss and metrics at that epoch when relevant.
7. **PowerShell `&&`** → use `;` or separate commands.

---

## 8. Quick checklist (full flow)

- [ ] **New config file** created; minimal/single change vs base; unique `run_name`.
- [ ] **All commands** use `.\.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Unix).
- [ ] **Training** run to completion (or early_stop as configured).
- [ ] **Analysis** run with **all relevant** run dirs and `--interval 5`; output captured.
- [ ] **Docs updated:** Run Analysis line, comparison table(s), subsection for new variant, Recommendations and analysis command including the new run.

---

## 9. Reference: BC pretrain commands

| Step            | Command |
|-----------------|--------|
| Run BC pretrain | `.\.venv\Scripts\python.exe scripts/pretrain_bc.py --config config_files/pretrain/bc/<config>.yaml` |
| Analyze N runs  | `.\.venv\Scripts\python.exe scripts/analyze_pretrain_bc.py output/ptretrain/bc/run1 output/ptretrain/bc/run2 ... output/ptretrain/bc/runN --interval 5` |
| With base dir   | `.\.venv\Scripts\python.exe scripts/analyze_pretrain_bc.py --base-dir output/ptretrain/bc run1 run2 run3 --interval 5` |

Configs and run names used in the doc (as of last update) for A_head multi-offset family: `v2_multi_offset`, `v2_multi_offset_ahead`, `v2_multi_offset_ahead_tuned`, `v2_multi_offset_ahead_reg`, `v2_multi_offset_ahead_dropout`, `v2_multi_offset_ahead_dropout_inner`. When adding a new variant, add it to the analysis command and to the Run Analysis / tables / recommendations in `docs/source/experiments/pretrain/behavioral_cloning.rst`.
