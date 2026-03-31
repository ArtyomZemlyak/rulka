=====================================
TrackMania RL - Documentation
=====================================

Welcome to the TrackMania RL project documentation!

This is a fork and extension of the original `Linesight <https://github.com/pb4git/linesight>`_ project, adapted for reinforcement learning experiments in *Trackmania Nations Forever*.

The project trains an AI agent to drive in *Trackmania Nations Forever* using reinforcement learning. The default stack is **IQN** (Implicit Quantile Networks, distributional off-policy RL). A second stack is **PPO** (on-policy actor-critic). Both can use a **CNN** image head, **Hugging Face** vision, or a shared **native multimodal fusion** graph (``nn.fusion_mode``); see :doc:`models/index` and :doc:`configuration_guide`.

**Key Features:**

- Distributional RL with IQN (Implicit Quantile Network), default
- Optional on-policy **PPO** with shared TM rollout pipeline (see :ref:`ppo-config` in :doc:`configuration_guide`)
- Modular configuration system for easy experimentation
- Support for multiple parallel game instances
- Hot-reloadable training parameters
- TensorBoard integration for monitoring
- Virtual checkpoint system for dense progress tracking

**All runs produced by this project are Tool Assisted. They must not be submitted to the Official Leaderboards.**

.. toctree::
   :maxdepth: 2
   :caption: User Documentation:

   installation
   first_training
   second_training
   configuration_guide
   game_inputs_and_float_vector
   tmnf_replays
   hf_dataset
   tensorboard_metrics
   user_faq
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Dev Documentation:

   project_structure
   main_objects
   contributions
   documentation
   dev_faq
   reading_list

.. toctree::
   :maxdepth: 2
   :caption: Model Architectures:

   models/index

.. toctree::
   :maxdepth: 2
   :caption: Experiments:

   experiments/index

.. toctree::
   :maxdepth: 2
   :caption: Community tips & tricks

   empty_page
