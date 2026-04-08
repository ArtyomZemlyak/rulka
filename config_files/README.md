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
python scripts/train.py --config config_files/rl/config_default.yaml   # IQN
python scripts/train.py --config config_files/rl/config_ppo.yaml        # PPO
python scripts/train.py --config config_files/rl/config_dpo.yaml      # DPO
python scripts/train.py --config config_files/rl/config_grpo.yaml     # GRPO
```

User-specific settings (paths, usernames) are read from a `.env` file in the project root. Config is loaded once per process; there is no hot-reload.

## File Structure

```
config_files/
├── rl/
│   ├── config_default.yaml  # IQN (default)
│   ├── config_ppo.yaml      # PPO on-policy example
│   ├── config_dpo.yaml      # DPO (preference learning; same policy stack as PPO)
│   ├── config_grpo.yaml     # GRPO (group-relative updates)
│   └── config_*.yaml        # Other experiment YAMLs
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
- **nn** – Vision (`vis`: `image_size`, `cnn` / `transformer`), float MLP width, PPO `fusion_mode`, IQN quantiles (`iqn`), dueling heads (`decoder`), IQN training ops (`nn.training`), etc. (full tree in the [Configuration Guide](https://artyomzemlyak.github.io/rulka/configuration_guide.html) under *Neural network YAML*). After load, the same fields also appear on the flat object `RulkaConfig.neural_network` for code that uses `cfg.neural_network.*`.
- **training** – `run_name`, `algorithm` (`iqn` \| `ppo` \| `dpo` \| `grpo`), `batch_size` (IQN), `lr_schedule`, `gamma_schedule` (IQN n-step), n-step knobs, etc. On-policy: optional `policy_rollout_gamma` / `policy_rollout_gamma_schedule` for the **shared rollout reward builder** (PPO/DPO/GRPO) and **PPO GAE γ** (not the same as IQN `gamma_schedule`). Legacy fallback: `ppo.gamma` / `ppo.ppo_gamma_schedule`.
- **ppo** – PPO loss hyperparameters (GAE λ schedules, clip, entropy, `rollout_steps_per_update`, …). The **clipped objective** runs only for `algorithm: ppo`. Details: [Configuration Guide — PPO](https://artyomzemlyak.github.io/rulka/configuration_guide.html#ppo-config).
- **dpo** – DPO-only keys (prefixed `dpo_*`), e.g. `dpo_beta`, `dpo_data_mode`, `dpo_offline_pairs_jsonl`. [DPO section](https://artyomzemlyak.github.io/rulka/configuration_guide.html#dpo-config).
- **grpo** – GRPO-only keys (`grpo_*`), e.g. `grpo_group_size`, `grpo_normalize_group`. [GRPO section](https://artyomzemlyak.github.io/rulka/configuration_guide.html#grpo-config).
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
