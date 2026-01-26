=====================================
TrackMania RL - Documentation
=====================================

Welcome to the TrackMania RL project documentation!

This is a fork and extension of the original `Linesight <https://github.com/pb4git/linesight>`_ project, adapted for reinforcement learning experiments in *Trackmania Nations Forever*.

The project uses distributional reinforcement learning (IQN - Implicit Quantile Networks) to train an AI agent to drive in TrackMania. The goal is to explore RL algorithms, reward shaping, and training techniques in a complex racing environment.

**Key Features:**

- Distributional RL with IQN (Implicit Quantile Network)
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
   :caption: Community tips & tricks

   empty_page
