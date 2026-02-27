# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **IQN architecture documentation** — `docs/source/experiments/models/iqn_architecture.rst` with high-level and per-block Graphviz diagrams (inputs/outputs, image head, float head, IQN quantile mixing, dueling heads); link from main_objects; `sphinx.ext.graphviz` enabled
- **Pretrain BC** — behavioral cloning pretraining (`trackmania_rl.pretrain`, `scripts/pretrain_bc.py`, `config_files/pretrain/bc/pretrain_config_bc.yaml`), experiment docs and plots; `pretrain_visual` renamed to `pretrain`

### Changed
- **Doc build** — optional dependency group `doc` comment: Graphviz (system) required for architecture diagrams; all architecture page text in English

## [1.4.0] - 2026-02-18

### Added
- **Pretrain visual backbone** — optional visual pretraining (encoder injection into IQN), config and scripts
- **TMNF replays** — replay capture and related tooling
- Config updates and `.gitignore` for cache/output

## [1.3.0] - 2026-02-16

### Added
- **Replay capture** — scripts and support for capturing replays (e.g. `--exclude-respawn-maps`), `replay_has_respawn` fixes
- **Experiments plots** and **game_env_backend**-related updates

### Changed
- Config, docs, and scripts updates

## [1.2.0] - 2026-02-01

### Added
- **Experiment comparison plots** — scripts `generate_experiment_plots.py`, `plot_experiment_comparison.py`, `experiment_plot_utils.py`; JPG graphs (one metric per graph, runs as lines) saved to `docs/source/_static/`, embedded in experiment RST with `:alt:` captions
- **Relative-time comparison** — `analyze_experiment_by_relative_time.py` with `--plot`, `compute_comparison_data()` for tables and plots; optional `--all-scalars`, `--metrics` for arbitrary TensorBoard tags
- **CI: docs build and deploy** — GitHub Actions workflow builds Sphinx docs and deploys to GitHub Pages (venv python, doc extra only)
- **doc_exp (Create Experiment)** — rule "Embedding plots in RST (quality)": place image after metric subsection, `:alt:` on every image, intro sentence in Detailed TensorBoard Metrics Analysis

### Changed
- **Config** — migrated to YAML (`config_files/`); IQN experiments (uni_19, uni_20) documented
- **Best-time plot Y-axis** — scale from **min(time)** to **mean(time) + 1s** so initial 300s spike is off-scale and improvement is readable; robust (percentile) scaling kept for loss, Q, finish rate
- **Experiment docs** — exploration (uni_12 vs uni_15), temporal_mini_race_duration, training_speed, models/iqn: plots for all experiments (incl. IQN Exp 2–5), intro sentence and alt text for every figure
- **Universal scalar metrics** — `load_run_metrics(tags_to_load)`, `get_available_scalar_tags()`, `use_all_scalars` / `extra_scalar_tags` in `compute_comparison_data()`; scalar tag → kind/unit inferred for any tag

## [1.1.0] - 2026-01-27

### Added
- **Experiment analysis scripts** — `scripts/analyze_experiment.py`, `scripts/analyze_batch_experiment.py`, `scripts/extract_tensorboard_data.py` for extracting and comparing TensorBoard metrics across runs
- **Experiments documentation** — `docs/source/experiments/` section (training_speed and index) with toctree in main docs

### Changed
- **performance_config.py** — parameter names aligned with code (e.g. `plot_race_time_left_curves`, `update_inference_network_every_n_actions`); `force_window_focus_on_input` deprecated (focus managed once per map load)
- **training_config.py** — extended `tensorboard_suffix_schedule`, `oversample_long_term_steps` / `oversample_maximum_term_steps`, `min_horizon_to_update_priority_actions`
- **docs/source/index.rst** — added Experiments toctree

## [1.0.0] - 2026-01-25

### Added
- Modern `pyproject.toml` with hatchling build backend
- Support for `uv` package manager - single command installation
- Comprehensive [User FAQ](https://artyomzemlyak.github.io/rulka/user_faq.html) with 30+ questions
- Comprehensive [Dev FAQ](https://artyomzemlyak.github.io/rulka/dev_faq.html) for contributors
- Hot-reload configuration system documentation

### Changed
- **README.md** - rewritten as concise English quick-start guide
- **Documentation** - significantly expanded and reorganized
  - Installation guide with uv support
  - Extended troubleshooting section
  - Virtual checkpoint creation guide
  - Modular configuration fully documented
- **Dependencies** - updated to PyTorch 2.7+, TorchRL 0.6+, CUDA 12.6
- **Build system** - migrated from setup.py to modern pyproject.toml
- **Language** - all documentation in English

### Removed
- `setup.py` - replaced by pyproject.toml
- `requirements_pip.txt` - merged into pyproject.toml
- `requirements_conda.txt` - merged into pyproject.toml
- Original Linesight branding from README

### Fixed
- NumPy version pinned to 1.26.4 (2.0 breaks numba)
- Documentation reflects actual modular config structure
- Installation instructions updated for modern tools

---

## Comparison with Original Linesight

| Aspect | Original Linesight | This Fork |
|--------|-------------------|-----------|
| Build System | setup.py + requirements.txt | pyproject.toml + uv |
| Installation | Multi-step conda/pip | `uv sync` |
| Dependencies | requirements files | pyproject.toml |
| Documentation | Basic | Comprehensive FAQ + guides |
| Language | Mixed | English |
| Focus | Research project | Personal experimentation |

---

## Original Linesight

This is a fork of the original Linesight project:
- Repository: https://github.com/pb4git/linesight
