# Configuration Files

## Quick Start

Configuration is loaded from a **single YAML file** at startup. Use `get_config()` to access settings in code:

```python
from config_files.config_loader import get_config

cfg = get_config()
batch_size = cfg.batch_size
epsilon = cfg.epsilon_schedule
```

Run training with a specific config:

```bash
python scripts/train.py --config config_files/rl/config_default.yaml
```

User-specific settings (paths, usernames) are read from a `.env` file in the project root. Config is loaded once per process; there is no hot-reload.

## File Structure

```
config_files/
├── rl/
│   └── config_default.yaml  # Default YAML config (versioned, tracked in git)
├── pretrain/
│   ├── vis/                 # Visual pretrain (AE, SimCLR): pretrain_config.yaml, etc.
│   └── bc/                  # BC pretrain: pretrain_config_bc.yaml, etc.
├── config_schema.py         # Pydantic models for validation
├── config_loader.py         # load_config(), get_config(), set_config()
├── pretrain_schema.py       # PretrainConfig (vis)
├── pretrain_bc_schema.py    # BCPretrainConfig
└── README.md
```

You can add more YAML files (e.g. `config_uni18.yaml`) and pass them with `--config` to version experiments.

## Editing Configuration

1. Edit the YAML file (e.g. `rl/config_default.yaml` or your own `config_*.yaml`).
2. Restart training so the new config is loaded.
3. A snapshot of the config used for each run is saved in `save/{run_name}/config_snapshot.yaml`.

For detailed parameter descriptions, see:

**📚 [Configuration Guide](https://artyomzemlyak.github.io/rulka/configuration_guide.html)**

Or build Sphinx docs:

```bash
cd docs
make html
# Open build/html/configuration_guide.html
```

## YAML Sections

The default YAML is organized into sections:

- **environment** – Timing, spatial params, timeouts, game settings
- **neural_network** – Image size, layer sizes, IQN params, gradient clipping
- **training** – run_name, batch_size, lr_schedule, gamma_schedule, n_steps, etc.
- **memory** – Buffer size schedule, PER (prio_*), usage control
- **exploration** – Epsilon and Boltzmann schedules
- **rewards** – Time penalty, progress reward, shaped rewards
- **map_cycle** – Map training cycle (short_name, map_path, reference_line_path, is_exploration, fill_buffer)
- **performance** – gpu_collectors_count, running_speed, network sync, visualization

## Examples

### Change learning rate

Edit your YAML under `training:`:

```yaml
training:
  lr_schedule:
    - [0, 0.001]
    - [3000000, 0.0001]
    - [12000000, 5e-05]
```

### Use a different config file

```bash
python scripts/train.py --config config_files/rl/config_uni18.yaml
```

### User-specific settings (.env)

Create `.env` in the project root (not tracked in git):

```
USERNAME=YourTrackManiaLogin
# other keys as defined in config_schema.UserConfig
```

## Documentation

- **Configuration Guide** – Full parameter descriptions (Sphinx)
- **README.md** – This file (overview)

Build docs: `cd docs && make html`

## Support

- Documentation: [Configuration Guide](https://artyomzemlyak.github.io/rulka/configuration_guide.html)
- Discord: [Join discussion](https://discord.gg/PvWYGkGKqd)
- GitHub: [Report issues](https://github.com/ArtyomZemlyak/rulka)
