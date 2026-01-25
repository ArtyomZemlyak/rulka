============
Installation
============

Prerequisites
-------------

TrackMania RL requires:
    - Python >=3.10 and <3.12
    - PyTorch >=2.7 with CUDA 12.x (for GPU acceleration)
    - 20 GB RAM
    - `Trackmania Nations Forever <https://store.steampowered.com/app/11020/TrackMania_Nations_Forever/>`_ (free)
    - `TMLoader (ModLoader) <https://tomashu.dev/software/tmloader/>`_
    - `TMInterface 2.1.0 <https://www.donadigo.com/tminterface/>`_

This project is compatible with Windows and Linux. The authors primarily use Nvidia GPUs with CUDA support.

Python project setup
--------------------

We recommend using `uv <https://github.com/astral-sh/uv>`_ (fast Python package manager) for installation.

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

**1. Install uv**

On Windows (PowerShell):

.. code-block:: powershell

    irm https://astral.sh/uv/install.ps1 | iex

After installation, restart terminal or add to PATH:

.. code-block:: powershell

    $env:Path = "C:\Users\$env:USERNAME\.local\bin;$env:Path"

On Linux/macOS:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

**2. Clone the repository and install**

.. code-block:: bash

    git clone <your-repository-url>
    cd rulka
    uv sync

This single command will:
    - Create a `.venv` virtual environment with Python 3.11
    - Install PyTorch 2.7+ with CUDA 12.6
    - Install TorchRL with C++ binaries
    - Install all dependencies from `pyproject.toml`
    - Install `trackmania_rl` package in editable mode

**3. Activate the environment**

.. code-block:: bash

    # Windows
    .\.venv\Scripts\activate
    
    # Linux/macOS
    source .venv/bin/activate

Using pip (Alternative)
~~~~~~~~~~~~~~~~~~~~~~~

If not using uv, you can install with standard pip:

.. code-block:: bash

    git clone <your-repository-url>
    cd rulka
    python -m venv .venv
    
    # Activate environment
    # Windows: .\.venv\Scripts\activate
    # Linux/macOS: source .venv/bin/activate
    
    # Install PyTorch (check https://pytorch.org for your system)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install the project
    pip install -e .

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
    python -c "import torchrl; print(f'TorchRL: {torchrl.__version__}')"
    python -c "import trackmania_rl; print('Installation OK')"


Linux-specific instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the folder ``/scripts/``, create a bash script that takes an integer ``port`` as its single argument. The script should start the game configured to listen on port ``port`` for TMInterface communications. The scripts ``/scripts/launch_game_pb.sh`` and ``/scripts/launch_game_agade.sh`` are working examples on the authors' systems.

.. note::
    The authors have experienced improved FPS when executing TMNF within wine with the following setup:

    .. code-block:: bash

        sudo apt install winetricks
        winetricks dxvk

    and launching the game with ``exec gamemoderun wine (...)`` (see ``/scripts/launch_game_pb.sh`` for example).

Game configuration
------------------

**1. Configure TrackMania Nations Forever**

The game must be configured (via ``TmForeverLauncher.exe`` in the game's installation directory) to run in **windowed mode**.
We recommend adjusting game settings to run at the lowest resolution available with low graphics quality.

Create an **online account** in TrackMania (offline accounts are not supported by TMInterface).

.. note::
   There is a compromise to be made between *training speed* which increases with FPS and *trained performance* which increases with image quality. Users can experiment with their setup.

**2. Configure TMLoader**

- Launch TMLoader
- Create a profile named ``default`` (or use a custom name - update ``user_config.py`` accordingly)
- Enable TMInterface in the profile

**3. Copy Python_Link.as plugin**

The Python_Link.as plugin enables communication between Python and TMInterface.

On Windows (PowerShell):

.. code-block:: powershell

    # Create directory if it doesn't exist
    New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Documents\TMInterface\Plugins"
    
    # Copy the plugin
    Copy-Item "trackmania_rl\tmi_interaction\Python_Link.as" "$env:USERPROFILE\Documents\TMInterface\Plugins\"

On Linux:

.. code-block:: bash

    mkdir -p ~/Documents/TMInterface/Plugins
    cp trackmania_rl/tmi_interaction/Python_Link.as ~/Documents/TMInterface/Plugins/

User config
-----------

Open the file ``config_files/user_config.py`` and make modifications relevant to your system:

.. code-block:: python

    username = "tmnf_account_username"  # Username of your ONLINE TMNF account

    # Path where Python_Link.as should be placed so that it can be loaded in TMInterface.
    # Usually Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"
    target_python_link_path = Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"

    # Typically Path(os.path.expanduser("~")) / "Documents" / "TrackMania"
    trackmania_base_path = Path(os.path.expanduser("~")) / "Documents" / "TrackMania"

    # Communication port for the first TMInterface instance that will be launched.
    # If using multiple instances, the ports used will be base_tmi_port + 1, +2, +3, etc...
    base_tmi_port = 8478

    # If on Linux, path of a shell script that launches the game, with the TMInterface port as first argument
    linux_launch_game_path = "path_to_be_filled_only_if_on_linux"

    # If on Windows, path where TMLoader can be found.
    # Usually Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
    windows_TMLoader_path = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"

    # If on Windows, name of the TMLoader profile that will launch TmForever + TMInterface
    windows_TMLoader_profile_name = "default"

.. warning::
   **Important:** Map files and replay files must NOT be stored in OneDrive or other cloud storage directories. 
   Cloud synchronization interferes with the map loading mechanism.
