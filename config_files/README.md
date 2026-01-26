# Configuration Files

## Quick Start

All configuration settings are in compact Python files for easy editing:

```python
from config_files import config_copy

# Access any setting
batch_size = config_copy.batch_size
epsilon = config_copy.epsilon_schedule
```

## File Structure

```
config_files/
├── config.py                    # Main config (imports all modules)
├── config_copy.py               # Runtime copy (auto-updated)
│
├── environment_config.py        # 🎮 Environment & simulation (70 lines)
├── neural_network_config.py     # 🧠 Network architecture (70 lines)
├── training_config.py           # 📊 Training hyperparameters (70 lines)
├── memory_config.py             # 💾 Replay buffer (40 lines)
├── exploration_config.py        # 🔍 Exploration strategies (30 lines)
├── rewards_config.py            # 🎁 Reward shaping (45 lines)
├── map_cycle_config.py          # 🗺️ Map training cycle (50 lines)
├── performance_config.py        # ⚡ Performance settings (45 lines)
│
├── inputs_list.py               # Action space
├── state_normalization.py       # State normalization
└── user_config.py               # User-specific settings
```

## Editing Configurations

### For Quick Changes

All config files are now **compact** (30-70 lines each) with inline comments.

Open any file and edit directly:

```python
# training_config.py
batch_size = 512  # Number of transitions per training step
```

### For Documentation

See comprehensive documentation at:

**📚 [Configuration Guide](../../docs/source/configuration_guide.rst)**

Or build Sphinx docs:

```bash
cd docs
make html
# Open build/html/configuration_guide.html
```

## On-the-Fly Changes

To modify settings during training:

1. Edit `config_copy.py` (not `config.py`)
2. Save the file
3. Changes apply automatically
4. Replay buffer preserved

## Benefits of New Structure

✅ **Compact files** - All settings visible on one screen  
✅ **Easy editing** - No scrolling through documentation  
✅ **Documentation preserved** - In Sphinx docs for learning  
✅ **Professional** - Separates code from documentation  
✅ **Fast** - Find and edit settings quickly

## Configuration Modules

### 1. Environment (environment_config.py)

Image processing, timing, spatial parameters, timeouts, game settings

Key settings:
- `tm_engine_step_per_action` - Control frequency
- `n_zone_centers_in_inputs` - Lookahead waypoints

### 2. Neural Network (neural_network_config.py)

Network architecture, IQN parameters, gradient clipping, image dimensions

Key settings:
- `W_downsized`, `H_downsized` - Image dimensions (affects CNN output size)
- `dense_hidden_dimension` - Main layer size
- `iqn_n`, `iqn_k` - Quantile samples
- `clip_grad_norm` - Gradient clipping

### 3. Training (training_config.py)

Learning rates, optimizer, schedules, batch size

Key settings:
- `run_name` - Experiment identifier
- `batch_size` - Training batch size
- `lr_schedule` - Learning rate schedule
- `gamma_schedule` - Discount factor schedule

### 4. Memory (memory_config.py)

Replay buffer size, prioritization, sampling

Key settings:
- `memory_size_schedule` - Buffer capacity
- `prio_alpha` - Prioritization strength
- `number_times_single_memory_is_used_before_discard` - Sample frequency

### 5. Exploration (exploration_config.py)

Epsilon-greedy, Boltzmann exploration

Key settings:
- `epsilon_schedule` - Random exploration decay
- `epsilon_boltzmann_schedule` - Boltzmann exploration
- `tau_epsilon_boltzmann` - Temperature

### 6. Rewards (rewards_config.py)

Progress rewards, time penalties, shaped rewards

Key settings:
- `constant_reward_per_ms` - Time penalty
- `reward_per_m_advanced_along_centerline` - Progress reward
- All shaped rewards (currently disabled)

### 7. Map Cycle (map_cycle_config.py)

Map selection and training sequences

Key settings:
- `map_cycle` - Training map rotation

### 8. Performance (performance_config.py)

Parallelization, visualization, augmentation

Key settings:
- `gpu_collectors_count` - Parallel game instances
- `running_speed` - Simulation speed
- Visualization flags

## Migration from Old Structure

Old config files (200+ lines each with embedded documentation) have been replaced with:

1. **Compact config files** (30-70 lines) - for editing
2. **Sphinx documentation** (configuration_guide.rst) - for learning

All settings remain accessible through `config` and `config_copy` modules.

## Examples

### Change learning rate during training

```python
# Edit config_copy.py
lr_schedule = [
    (0, 1e-3),
    (3_000_000, 1e-4),  # Changed from 5e-5
]
```

### Add new map to cycle

```python
# Edit map_cycle_config.py
map_cycle += [
    repeat(("newmap", "NewMap.Challenge.Gbx", "newmap_0.5m_cl.npy", True, True), 4),
]
```

### Adjust exploration

```python
# Edit exploration_config.py
epsilon_schedule = [
    (0, 1),
    (100_000, 0.2),  # More exploration
]
```

## Documentation

For detailed explanations of each parameter:

1. **Inline comments** - Quick reference in config files
2. **Configuration Guide** - Comprehensive Sphinx documentation
3. **README.md** - This file (overview)

Build documentation:

```bash
cd docs
make html
firefox build/html/configuration_guide.html
```

## Support

- Documentation: [Configuration Guide](../../docs/source/configuration_guide.rst)
- Discord: [Join discussion](https://discord.gg/PvWYGkGKqd)
- GitHub: [Report issues](https://github.com/ArtyomZemlyak/rulka)
