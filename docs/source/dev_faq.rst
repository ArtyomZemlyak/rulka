============
Dev FAQ
============

This page provides answers to common questions for developers working on the project.

Project Setup
=============

**Q: How do I set up the development environment?**

A: We recommend using `uv` for fast installation:

.. code-block:: bash

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
    irm https://astral.sh/uv/install.ps1 | iex       # Windows

    # Clone and setup
    git clone https://github.com/ArtyomZemlyak/rulka.git
    cd rulka
    uv sync

    # Activate environment
    source .venv/bin/activate  # Linux/macOS
    .\.venv\Scripts\activate   # Windows

**Q: Can I use pip instead of uv?**

A: Yes:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
    pip install -e .

Dependencies
============

**Q: How do I add a new dependency?**

A: All dependencies are managed in ``pyproject.toml``:

.. code-block:: bash

    # Using uv (recommended)
    uv add package-name

    # Using pip
    # 1. Edit pyproject.toml manually
    # 2. pip install -e .

**Q: Where are the requirements.txt files?**

A: They've been replaced by ``pyproject.toml``. All dependencies (PyTorch, TorchRL, etc.) are specified in the ``[project.dependencies]`` section.

Code Quality
============

**Q: How do I run linters?**

A: The project uses Ruff for linting and formatting:

.. code-block:: bash

    # Install dev dependencies
    uv sync --group dev  # or pip install -e ".[dev]"

    # Run linter
    ruff check .
    ruff format .

**Q: Are there pre-commit hooks?**

A: Not currently set up. Contributions to add pre-commit hooks are welcome!

Testing & Verification
======================

**Q: How do I test my installation?**

A:

.. code-block:: bash

    # Verify config loads (from project root)
    python -c "from config_files.config_loader import load_config; c = load_config('config_files/config_default.yaml'); print('Config OK')"
    python scripts/check_plugin.py

    # Verify packages
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import torchrl; print(f'TorchRL: {torchrl.__version__}')"
    python -c "import trackmania_rl; print('OK')"

**Q: Are there unit tests?**

A: Not currently. The project uses integration testing by running training. Unit tests would be a valuable contribution!

Configuration Development
=========================

**Q: How do I add a new configuration parameter?**

A:

1. Add the parameter to the appropriate Pydantic model in ``config_files/config_schema.py``
2. Add it to the corresponding section in ``config_files/config_default.yaml``
3. Update the Configuration Guide (:doc:`configuration_guide`) with full documentation
4. Verify with: ``python -c "from config_files.config_loader import load_config; load_config('config_files/config_default.yaml')"``

**Q: Can I modify config during training?**

A: Config is loaded once at startup. To change parameters, edit the YAML file and restart training. A snapshot is saved in ``save/{run_name}/config_snapshot.yaml``.

.. warning::
   Don't change network architecture, input dimensions, or action space - these require restart.

Documentation
=============

**Q: How do I build the documentation?**

A:

.. code-block:: bash

    # Install doc dependencies
    pip install -e ".[doc]"

    # Build docs
    cd docs
    make html          # Linux/macOS
    .\make.bat html    # Windows

    # View docs
    firefox build/html/index.html  # Linux
    open build/html/index.html     # macOS
    start build/html/index.html    # Windows

**Q: How do I add a new documentation page?**

A:

1. Create ``.rst`` file in ``docs/source/``
2. Add it to the toctree in ``docs/source/index.rst``
3. Build and verify: ``make html``
4. Check links: ``make linkcheck``

Troubleshooting
===============

**Q: Import errors after installation**

A:

.. code-block:: bash

    # Reinstall in editable mode
    pip install -e .

**Q: CUDA not available in development**

A:

.. code-block:: bash

    # Check CUDA
    nvidia-smi
    python -c "import torch; print(torch.cuda.is_available())"

    # Reinstall PyTorch
    uv sync --reinstall

**Q: Game connection issues during development**

A: Checklist:

1. Verify TMInterface is running
2. Check port in ``.env`` (``BASE_TMI_PORT``, default 8478)
3. Ensure ``Python_Link.as`` is copied to ``TMInterface/Plugins/``
4. Check TMLoader profile includes TMInterface

Performance
===========

**Q: How can I speed up training during development?**

A:

- Increase ``gpu_collectors_count`` in the ``performance`` section
- Increase ``running_speed`` (up to 200x)
- Reduce image resolution (``w_downsized``, ``h_downsized``) in the ``neural_network`` section
- Disable visualization in the ``performance`` section

**Q: How can I reduce memory usage?**

A:

- Reduce ``memory_size_schedule`` in the ``memory`` section
- Reduce ``batch_size`` in the ``training`` section
- Reduce ``gpu_collectors_count`` in the ``performance`` section

Virtual Checkpoints
===================

**Q: How do I create virtual checkpoints for a new map?**

A:

.. code-block:: bash

    # 1. Drive the track and save replay
    # 2. Convert replay to VCP
    python scripts/tools/gbx_to_vcp.py "path/to/replay.Replay.Gbx"

    # 3. File is saved to maps/ folder
    # 4. Update map_cycle.entries in config YAML to use new VCP file

**Q: What's the format of VCP files?**

A: VCP files are NumPy arrays (`.npy`) with shape (N, 3) containing waypoint coordinates (X, Y, Z). Points are sampled every 0.5m (configurable via ``distance_between_checkpoints``) along the replay trajectory.

Project Structure
=================

**Q: Where is the main code located?**

A:

- ``trackmania_rl/`` - Core RL implementation
  
  - ``agents/`` - IQN agent implementation
  - ``multiprocess/`` - Parallel training (collector & learner processes)
  - ``tmi_interaction/`` - Game interface (TMInterface communication)

- ``config_files/`` - YAML configuration (config_default.yaml, config_schema.py, config_loader.py)
- ``scripts/`` - Training script and tools

**Q: How is configuration structured?**

A:

- Configuration is loaded from a YAML file (e.g. ``config_files/config_default.yaml``) at startup via ``train.py --config <path>``
- Use ``from config_files.config_loader import get_config`` and ``get_config().<attribute>`` to access settings in code
- User-specific settings (paths, username) come from ``.env`` in the project root
- A snapshot of the config used for each run is saved in ``save/{run_name}/config_snapshot.yaml``

Build System
============

**Q: Why was setup.py removed?**

A: Migrated to modern ``pyproject.toml`` with hatchling build backend (PEP 517/518 compliant). Benefits:

- Single source of truth for dependencies
- Native uv support
- Faster installation
- Modern Python packaging standards

**Q: What's in pyproject.toml?**

A:

- Project metadata (name, version, description)
- All dependencies (PyTorch, TorchRL, etc.)
- Optional dependencies (dev, doc)
- Build system configuration (hatchling)
- Tool configuration (ruff)
- uv-specific settings (PyTorch CUDA index)

Contributing
============

**Q: How do I contribute to this fork?**

A:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Follow coding guidelines (use Ruff)
5. Update documentation
6. Test thoroughly
7. Submit a pull request

See :doc:`contributions` for detailed guidelines.

**Q: What coding style should I follow?**

A:

- Use Ruff for formatting: ``ruff format .``
- Add docstrings to new functions/classes
- Follow existing code patterns
- Update documentation for new features

Links
=====

- **Original Linesight**: https://github.com/pb4git/linesight
- **TMInterface**: https://donadigo.com/tminterface/
- **TMNF Exchange**: https://tmnf.exchange/
- **PyTorch Docs**: https://pytorch.org/docs/
- **TorchRL Docs**: https://pytorch.org/rl/
- **uv Package Manager**: https://github.com/astral-sh/uv
- **Ruff Linter**: https://github.com/astral-sh/ruff
