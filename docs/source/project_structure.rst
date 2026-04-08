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
    - ``config_ppo.yaml``: PPO example (CNN by default; comments for HF ViT / fusion). ``ppo:`` block; no replay.
    - ``config_dpo.yaml`` / ``config_grpo.yaml``: DPO and GRPO examples (same policy stack as PPO; ``dpo:`` / ``grpo:`` blocks).
    - ``config_ppo_cnn_mlp.yaml``: PPO baseline — ``nn.vis.cnn`` + ``nn.float.mlp``, ``nn.fusion_mode: none``.
    - ``config_ppo_transformer.yaml``: PPO ``post_concat`` multimodal (HF timm vision + HF fusion encoder; see file header).
    - ``config_btr.yaml``: IQN + full BTR bundle; CNN under ``nn.vis.cnn``. See :ref:`ppo-config`, :ref:`nn-yaml-reference`, :ref:`btr-yaml-reference` in :doc:`configuration_guide`.
    - ``config_loader.py``: Loads YAML, validates with Pydantic, and exposes ``load_config(path)``, ``get_config()``, and ``set_config(cfg)``. Config is loaded once per process and cached; there is no hot-reload.
    - ``config_schema.py``: Pydantic models for most config sections; hierarchical ``nn`` (network) lives in ``nn_schema.py`` (environment, training, memory, exploration, rewards, map_cycle, performance, state_normalization, user from ``.env``, ``ppo``, ``dpo``, ``grpo`` for algorithm-specific hyperparameters).

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

    - **IQN (default):** ``save/{run_name}/weights1.torch`` (online network), ``weights2.torch`` (target network), plus ``scaler.torch`` and ``optimizer1.torch``.
    - **PPO / DPO / GRPO:** ``weights1.torch`` (single policy), ``optimizer1.torch``, ``scaler.torch`` — there is **no** ``weights2.torch`` (no target network).
    - ``save/{run_name}/accumulated_stats.joblib`` a dictionary containing various stats for this run (number of frames, number of batches, training duration, best racing time, etc...)
    - ``save/{run_name}/best_runs/{map_name}_{time}/config.bak.py`` contains a backup copy of the training hyperparameters used for this run.
    - ``save/{run_name}/best_runs/{map_name}_{time}/{map_name}_{time}.inputs`` is a text file that contains the inputs to replay that run. It can be loaded in the TMInterface in-game console.
    - ``save/{run_name}/best_runs/{map_name}_{time}/q_values.joblib`` is a joblib dump of q_values expected by the agent during the run. They are typically used to produce the visual input widget in trackmania_rl.run_to_video.make_widget_video_from_q_values()
    - ``save/{run_name}/best_runs/`` IQN checkpoints on personal best: ``weights1.torch`` / ``weights2.torch`` (PPO best-runs naming may mirror ``weights1`` only where applicable).

When the script ``scripts/train.py`` is launched, collectors and the learner load checkpoints from ``save/{run_name}/`` when present. To resume, place the expected ``*.torch`` files for your algorithm (IQN: both weights files; PPO / DPO / GRPO: ``weights1`` only) in ``save/{run_name}``.

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

    - ``agents/``: RL agents and wiring. ``iqn.py`` implements IQN; ``algorithms/registry.py`` maps ``training.algorithm`` (``iqn`` | ``ppo`` | ``dpo`` | ``grpo``) to ``iqn_wiring`` or ``ppo_wiring`` (DPO and GRPO reuse ``ppo_wiring``). ``policy_models/`` holds discrete policies: ``ppo_actor_critic``, optional ``hf_actor_critic``, and ``multimodal_torch_fusion`` (``TorchMultimodalActorCritic``) when ``nn.fusion_mode != none`` (shared IQN backbone without heads). ``policy_optimization/`` holds PPO math (GAE, clipped loss), DPO preference loss, and GRPO group-relative objectives.
    - ``buffer_management.py``: Implements ``fill_buffer_from_rollout_with_n_steps_rule()``, the function that creates and stores transitions in a replay buffer given a ``rollout_results`` object provided by the method ``GameInstanceManager.rollout()``.
    - ``buffer_utilities.py``: Implements ``buffer_collate_function()``, used to customize torchrl's ``ReplayBuffer.sample()`` method. The most important customization is our implementation of *mini-races*, a trick to define Q values as the *expected sum of undiscounted rewards in the next 7 seconds*.
    - ``experience_replay/experience_replay_interface.py``: Defines the structure of transitions stored in a ReplayBuffer.
    - ``multiprocess/collector_process.py``: Implements the behavior of a single process that handles a game instance, and feeds ``rollout_results`` objects to the learner process. Multiple collector processes may run in parallel.
    - ``multiprocess/learner_process.py``: IQN learner (replay buffer, target network, priorities). If ``training.algorithm`` is ``ppo``, ``dpo``, or ``grpo``, training is delegated to ``learner_ppo.py``, ``learner_dpo.py``, or ``learner_grpo.py`` respectively (no IQN replay on those paths).
    - ``multiprocess/learner_ppo.py``: PPO learner loop (rollout aggregation, GAE, minibatch PPO updates).
    - ``multiprocess/learner_dpo.py`` / ``learner_grpo.py``: DPO and GRPO learner loops (preference pairs vs grouped rollouts).
    - ``tmi_interaction/game_instance_manager.py``: This file implements the main logic to interact with the game, via the GameInstanceManager class. There is a lot of legacy code, implemented when only TMInterface 1.4.3 was available.
    - ``tmi_interaction/tminterface2.py``: Implements the TMInterface class. It is designed to (mostly) reproduce the original Python client provided by Donadigo to communicate with TMInterface 1.4.3 via memory-mapping.

In addition to the modules described above, the project includes several other modules that provide supplementary functionality.
