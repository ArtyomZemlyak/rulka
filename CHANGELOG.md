# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-01-25

### Added
- Modern `pyproject.toml` with hatchling build backend
- Support for `uv` package manager - single command installation
- Comprehensive [User FAQ](docs/source/user_faq.rst) with 30+ questions
- Comprehensive [Dev FAQ](docs/source/dev_faq.rst) for contributors
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
- Documentation: https://linesight-rl.github.io/linesight/
