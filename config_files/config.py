"""
===========================================================================================================
TRACKMANIA RL CONFIGURATION - MAIN CONFIG FILE
===========================================================================================================

This file contains a run's configuration.
It is expected that this file contains all relevant information about a run.

Two files named "config.py" and "config_copy.py" coexist in the same folder.

At the beginning of training, parameters are copied from config.py to config_copy.py
During training, config_copy.py will be reloaded at regular time intervals.
config_copy.py is NOT tracked in git, as it is essentially a temporary file.

Training parameters modifications made during training in config_copy.py will be applied on the fly
without losing the existing content of the replay buffer.

The content of config.py may be modified after starting a run: it will have no effect on the ongoing run.
This setup provides the possibility to:
1) Modify training parameters on the fly
2) Continue to code, use git, and modify config.py without impacting an ongoing run.

===========================================================================================================
CONFIGURATION ORGANIZATION
===========================================================================================================

Configuration settings are now organized into separate modules for better maintainability:

1. environment_config.py - Game environment and simulation settings
   - Image dimensions, timing, spatial parameters, timeouts

2. neural_network_config.py - Neural network architecture
   - Layer dimensions, IQN parameters, gradient clipping

3. training_config.py - Training hyperparameters
   - Run identification (run_name)
   - Learning rates, optimizer settings, schedules

4. memory_config.py - Replay buffer configuration
   - Buffer sizes, prioritization, sampling

5. exploration_config.py - Exploration strategies
   - Epsilon-greedy, Boltzmann exploration

6. rewards_config.py - Reward shaping
   - Progress rewards, time penalties, engineered rewards

7. map_cycle_config.py - Map training cycle
   - Map selection and training sequences

8. performance_config.py - System performance settings
   - Parallelization, visualization, augmentation

9. inputs_list.py - Action space definition
   - Available actions (forward, brake, steering combinations)

10. state_normalization.py - Input normalization
    - Mean and standard deviation for state features

11. user_config.py - User-specific configuration
    - Paths, usernames, system-specific settings

All settings are re-exported below for backward compatibility.
You can still import from this file as before: `from config_files.config import setting_name`
Or import from specific modules: `from config_files.environment_config import W_downsized`

===========================================================================================================
"""

# ===========================================================================================================
# IMPORT ALL CONFIGURATION SETTINGS
# ===========================================================================================================

# Import existing configuration modules (kept as-is)
from config_files.inputs_list import *
from config_files.state_normalization import *
from config_files.user_config import *

# Import from new organized configuration modules
from config_files.environment_config import *
from config_files.neural_network_config import *
from config_files.training_config import *
from config_files.memory_config import *
from config_files.exploration_config import *
from config_files.rewards_config import *
from config_files.map_cycle_config import *
from config_files.performance_config import *


# ===========================================================================================================
# CONFIGURATION SUMMARY
# ===========================================================================================================

"""
This configuration is preconfigured with sensible hyperparameters for the map ESL-Hockolicious, 
assuming the user has a computer with 16GB RAM.

Quick Reference - Key Settings:
-------------------------------

Environment:
- Image size: 160×120 pixels
- Action interval: 50ms (20 actions/second)
- Game speed: 160× real-time

Neural Network:
- Architecture: IQN (Implicit Quantile Network)
- Main hidden layer: 1024 units
- IQN samples: n=8 (training), k=32 (inference)

Training:
- Batch size: 512
- Initial learning rate: 1e-3 → 5e-5 → 1e-5
- Gamma: 0.999 → 1.0 (undiscounted)
- N-step: 3

Memory:
- Buffer size: 50K → 100K → 200K transitions
- Prioritization: Disabled (uniform sampling)
- Usage: 32 samples per transition

Exploration:
- Epsilon: 1.0 → 0.1 → 0.03
- Boltzmann: 0.15 → 0.03

Rewards:
- Time penalty: -0.0012 per ms
- Progress reward: +0.01 per meter
- Shaped rewards: Disabled

Maps:
- Primary: ESL-Hockolicious (4 exploration runs)
- Secondary: A01-Race (4 exploration + 1 evaluation)

Performance:
- Parallel instances: 4 collectors
- Network updates: Every 10 batches

To modify these settings:
1. Edit the specific module file (e.g., environment_config.py)
2. Or edit this file directly (changes will be propagated)
3. During training, edit config_copy.py for on-the-fly changes
"""
