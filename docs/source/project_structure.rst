=================
Project Structure
=================

This page provides a high-level overview of the structure of the Rulka repository.

Repository Overview
-------------------

The repository is organized into the following directories:

    - ``config_files/``: Contains configuration files for the project overall, and for AI runs.
    - ``maps/``: Contains reference trajectories used to train on each map.
    - ``save/``: Contains saved model checkpoints.
    - ``scripts/``: Contains scripts for training, and general interaction with the game.
    - ``tensorboard/``: Contains tensorboard logs.
    - ``trackmania_rl/``: Contains the main project code.

config_files
------------

The ``config_files/`` folder holds configuration loaded from **YAML** at startup:

**Core Files:**

    - ``config_default.yaml``: Default configuration (versioned). Use ``scripts/train.py --config config_files/rl/config_default.yaml``. You can add more YAML files (e.g. ``config_uni18.yaml``) in ``config_files/rl/`` and pass them with ``--config``.
    - ``config_loader.py``: Loads YAML, validates with Pydantic, and exposes ``load_config(path)``, ``get_config()``, and ``set_config(cfg)``. Config is loaded once per process and cached; there is no hot-reload.
    - ``config_schema.py``: Pydantic models for all config sections (environment, neural_network, training, memory, exploration, rewards, map_cycle, performance, state_normalization, user from ``.env``).

**Supporting Files:**

    - ``inputs_list.py``: Defines discrete action space (forward, brake, steering combinations), used by the loader/schema.
    - ``state_normalization.py``: Helpers if needed; main normalization data can live in the YAML.

User-specific settings (paths, usernames) are read from a ``.env`` file in the project root. In code, use ``from config_files.config_loader import get_config`` and then ``get_config().<attribute>`` for flat access to any setting.


maps
----

The ``maps/`` folder contains ``{map}.npy`` files which define a reference trajectory for a given map. These are binary dumps of numpy arrays of shape (N, 3) with N the number of virtual checkpoints along the reference line. Virtual checkpoints are spaced evenly according to the config (e.g. ``distance_between_checkpoints``, typically 0.5m).

save
----

The ``save/`` folder contains information collected during training.

    - ``save/{run_name}/weights1.torch`` and ``weights2.torch`` and ``scaler.torch`` and ``optimizer1.torch`` are checkpoints containing the latest version of the neural network.
    - ``save/{run_name}/accumulated_stats.joblib`` a dictionary containing various stats for this run (number of frames, number of batches, training duration, best racing time, etc...)
    - ``save/{run_name}/best_runs/{map_name}_{time}/config.bak.py`` contains a backup copy of the training hyperparameters used for this run.
    - ``save/{run_name}/best_runs/{map_name}_{time}/{map_name}_{time}.inputs`` is a text file that contains the inputs to replay that run. It can be loaded in the TMInterface in-game console.
    - ``save/{run_name}/best_runs/{map_name}_{time}/q_values.joblib`` is a joblib dump of q_values expected by the agent during the run. They are typically used to produce the visual input widget in trackmania_rl.run_to_video.make_widget_video_from_q_values()
    - ``save/{run_name}/best_runs/weights1.torch`` and ``weights2.torch`` and ``scaler.torch`` and ``optimizer1.torch`` are checkpoints saved anytime the agent improves its personal best.

When the script ``scripts/train.py`` is launched, it attempts to load the checkpoint saved in ``save/{run_name}/``. To resume training from a given checkpoint, paste the ``*.torch`` files in ``save/{run_name}``.

scripts
-------

The ``scripts/`` folder contains the training script as well as various utility scripts to interact with the game. Each script is documented with a docstring explaining its purpose and usage in the first few lines.

tensorboard
-----------

The ``tensorboard/`` folder contains tensorboard logs for all runs.

trackmania_rl
-------------

The ``trackmania_rl/`` folder contains the core project code, which is intended to be imported and utilized within scripts rather than executed directly. We've documented the key modules, classes, and functions in the code, and we encourage developers who wish to get a comprehensive understanding of the project to read the docstrings directly in the codebase.

The main modules are listed here:

    - ``agents/``: Contains implementations of reinforcement learning agents. Currently contains only IQN.py, but has contained various agents such as DDQN or SAC-Discrete.
    - ``buffer_management.py``: Implements ``fill_buffer_from_rollout_with_n_steps_rule()``, the function that creates and stores transitions in a replay buffer given a ``rollout_results`` object provided by the method ``GameInstanceManager.rollout()``.
    - ``buffer_utilities.py``: Implements ``buffer_collate_function()``, used to customize torchrl's ``ReplayBuffer.sample()`` method. The most important customization is our implementation of *mini-races*, a trick to define Q values as the *expected sum of undiscounted rewards in the next 7 seconds*.
    - ``experience_replay/experience_replay_interface.py``: Defines the structure of transitions stored in a ReplayBuffer.
    - ``multiprocess/collector_process.py``: Implements the behavior of a single process that handles a game instance, and feeds ``rollout_results`` objects to the learner process. Multiple collector processes may run in parallel.
    - ``multiprocess/learner_process.py``: Implements the behavior of the (unique) learner process. It receives ``rollout_results`` objects from collector_processes, via a ``multiprocessing.Queue`` object. It sends updated neural network weights to collector processes weights ``torch.nn.Module.share_memory()``
    - ``tmi_interaction/game_instance_manager.py``: This file implements the main logic to interact with the game, via the GameInstanceManager class. There is a lot of legacy code, implemented when only TMInterface 1.4.3 was available.
    - ``tmi_interaction/tminterface2.py``: Implements the TMInterface class. It is designed to (mostly) reproduce the original Python client provided by Donadigo to communicate with TMInterface 1.4.3 via memory-mapping.

In addition to the modules described above, the project includes several other modules that provide supplementary functionality.
