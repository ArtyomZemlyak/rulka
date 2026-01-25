===============
Troubleshooting
===============

Common Issues
-------------

**Game stuck on login screen:**

The TMNF account must be an **online account**. If your account is offline:

- Start the game without TMInterface and create an online account
- Set your `username` to that online account in `user_config.py`

**FileNotFoundError: Python_Link.as**

The plugin wasn't copied to the correct location. Run:

.. code-block:: powershell

    # Windows
    New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Documents\TMInterface\Plugins"
    Copy-Item "trackmania_rl\tmi_interaction\Python_Link.as" "$env:USERPROFILE\Documents\TMInterface\Plugins\"

.. code-block:: bash

    # Linux
    mkdir -p ~/Documents/TMInterface/Plugins
    cp trackmania_rl/tmi_interaction/Python_Link.as ~/Documents/TMInterface/Plugins/

**CUDA not available**

1. Ensure NVIDIA driver 525+ is installed: ``nvidia-smi``
2. Check PyTorch installation: ``python -c "import torch; print(torch.cuda.is_available())"``
3. Reinstall PyTorch: ``uv sync --reinstall`` or follow `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_

**TorchRL warning about C++ binaries**

Ensure you're using PyTorch 2.7+ with matching TorchRL version. The ``uv sync`` command installs compatible versions automatically.

**Map not loading**

- Ensure map files are NOT in OneDrive or cloud storage directories
- Verify map path in `map_cycle_config.py` matches actual location
- Check that map file exists in ``~/Documents/TrackMania/Tracks/Challenges/``

**Low FPS / Slow training**

- Increase ``gpu_collectors_count`` in ``performance_config.py``
- Increase ``running_speed`` (up to 200x real-time)
- Lower game graphics settings and resolution
- Close unnecessary background applications

**Memory issues**

- Reduce ``memory_size_schedule`` in ``memory_config.py``
- Reduce ``gpu_collectors_count`` in ``performance_config.py``
- Reduce ``batch_size`` in ``training_config.py``
- Check RAM usage with Task Manager / htop

**Agent gets stuck or doesn't progress**

- Verify virtual checkpoint file (`.npy`) is correctly generated
- Check ``cutoff_rollout_if_no_vcp_passed_within_duration_ms`` timeout
- Increase exploration (higher epsilon in ``exploration_config.py``)
- Verify reference line covers the entire track

**Cars stop moving during training (Window Focus Issue)** 🆕

*Symptoms:* Cars "twitch" at start but don't move forward. Clicking on a window makes it work again.

*Cause:* TMInterface requires window focus to process inputs, even with ``unfocused_fps_limit false``.

*Solution:* This is now **automatically fixed** in the code. The game window receives focus once per map load, which is sufficient. No manual intervention needed.

*Technical details:*

- Window focus is set automatically when loading a new map
- For multiple instances (8+), focus is managed to avoid "focus war"
- Minimal performance impact (<0.01%)
- Works correctly with map cycling

*If issue persists:*

1. Check that windows are not minimized (game pauses when minimized)
2. Verify ``force_window_focus_on_input = False`` in ``performance_config.py``
3. With multiple maps, ensure smooth transitions between maps in logs

**Game crashes on startup**

- Check TMLoader profile is correctly configured
- Verify TMInterface 2.1.0 is installed
- Try launching game manually first to verify it works
- Check Windows firewall isn't blocking TMLoader/TMInterface

Linux-specific:
---------------

**Linux installation checklist:**
This list is not exhaustive. It contains the main setup steps the authors use on their machine. It may need to be adapted for your own machine.

1. Update `winehq-staging`
2. Download Steam. Install TMNF from Steam
3. Check that the game can be launched with `wine TmForever.exe` from the installation directory
4. Download the ModLoader zip file, made available on Tomashu's website for linux setups
5. `wine TMLoader.exe` to configure the default profile
6. Check that the game runs with: `wine ~/path/to/TMLoader-win32/TMLoader.exe run TmForever "default" /configstring="set custom_port 8483"`
7. Modify `launch_game_pb.sh` in the repository with the path to ModLoader on your system
8. Install `winetricks`. Apply `winetricks dxvk` for performance.

**Missing OpenAL32.dll**

Install `OpenAL <https://www.openal.org/downloads/>`_ with `wine`.