# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
