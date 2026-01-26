===============
Custom training
===============

In this page, it is expected that the user has successfully run :doc:`a first reference training on Hockolicious <first_training>`.

The objective is now to train the AI on another map, not preconfigured by the authors. As an exercise, we recommend setting up a simple map such as A02-Race.


Define a Reference Line
-----------------------

We define a reference trajectory along which the agent's progress is measured.

.. image:: _static/reference_line.png
  :width: 600
  :align: center


This reference trajectory **does not need to be fast**. The authors usually drive along the centerline of the map.

Use the following command to create a file ``maps/map.npy`` for your map from a replay.

.. code-block:: bash

   python scripts/tools/gbx_to_vcp.py {path_to_replay}.Replay.Gbx

Rename the file to your liking so that it is not overwritten in the future.

The authors recommend the following nomenclature: ``{map_name}_0.5m_{trajectory_description}.npy``. Trajectory description may be ``cl`` when the replay was driven on the centerline, or ``{player_name}`` if a known player's trajectory was used as reference.

.. note::
    To define the reference line, points are sampled every 50cm along the replay's trajectory. Over the course of the project, the terms "Virtual Checkpoints", "VCP", "zones" have been used to refer to these points.

Check the Reference Line
------------------------

This step is optional, but recommended. We want to visualize the reference line to check its position on the map.

Launch a game instance with the argument ``/configstring="set custom_port 8477"``.

Load the map.

Run the following command:

.. code-block:: bash

    python scripts/tools/tmi2/add_vcp_as_triggers.py maps/A02-Race_0.5m_cl.npy -p 8477

.. note::
    The ``-p`` argument allows you to choose the port used to connect to the game instance. You can freely change ``8477`` to any other available port.


Edit the configuration file
---------------------------

Configuration is now organized in separate modules for better maintainability. To train on a new map:

**1. Edit map_cycle_config.py**

Open ``config_files/map_cycle_config.py`` in a text editor. This file contains the map training cycle.

Find where the variable ``map_cycle`` is defined. Read the comment that describes how a map_cycle is defined. Edit the map_cycle so that the AI trains on A02-Race.

.. code-block:: python

    """
    Map cycle defines the training sequence.
    
    Each iterator returns tuples with:
        - short map name        (string):     for logging purposes
        - map path              (string):     to automatically load the map in game.
                                              This is the same map name as the "map" command in the TMInterface console.
        - reference line path   (string):     where to find the reference line for this map (in maps/ folder)
        - is_explo              (boolean):    whether the policy should be exploratory on this map
        - fill_buffer           (boolean):    whether the memories should be placed in the replay buffer
    
    Example: Simple cycle with 4 exploration runs + 1 evaluation run
    
    map_cycle = [
        repeat(("A02", '"A02-Race.Challenge.Gbx"', "A02_0.5m_cl.npy", True, True), 4),
        repeat(("A02", '"A02-Race.Challenge.Gbx"', "A02_0.5m_cl.npy", False, True), 1),
    ]
    """

**2. Adjust training schedule (optional)**

Open ``config_files/training_config.py``.

Locate the variable ``global_schedule_speed``. For easier maps like A02-Race, you can use a faster annealing schedule:

.. code-block:: python

    global_schedule_speed = 0.8

For harder maps like `map5 <https://tmnf.exchange/trackshow/10460245>`_ or E03-Endurance, use a slower schedule (e.g., 1.5).

**3. Update run name (optional)**

In ``config_files/training_config.py``, change the run name to identify this experiment:

.. code-block:: python

    run_name = "A02_training"

This affects tensorboard logs and save file locations.