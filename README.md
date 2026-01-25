# TrackMania RL

Reinforcement Learning for training AI in TrackMania Nations Forever using IQN (Implicit Quantile Networks).

> Personal fork of [Linesight](https://github.com/pb4git/linesight) adapted for RL experimentation.

## Quick Start

### Installation

```bash
# 1. Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
irm https://astral.sh/uv/install.ps1 | iex       # Windows

# 2. Clone and install
git clone <your-repo-url>
cd rulka
uv sync

# 3. Activate environment
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows
```

### Game Setup

1. Install [TrackMania Nations Forever](https://store.steampowered.com/app/11020/TrackMania_Nations_Forever/) (free)
2. Install [TMLoader](https://tomashu.dev/software/tmloader/) and [TMInterface 2.1.0](https://donadigo.com/tminterface/)
3. Configure game in **windowed mode** and create **online account**
4. Edit `config_files/user_config.py`:
   ```python
   username = "your_tmnf_account"
   ```
5. Copy plugin:
   ```bash
   # Windows
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Documents\TMInterface\Plugins"
   Copy-Item "trackmania_rl\tmi_interaction\Python_Link.as" "$env:USERPROFILE\Documents\TMInterface\Plugins\"
   
   # Linux
   mkdir -p ~/Documents/TMInterface/Plugins
   cp trackmania_rl/tmi_interaction/Python_Link.as ~/Documents/TMInterface/Plugins/
   ```

### Training

```bash
python scripts/train.py

# Monitor (in separate terminal)
tensorboard --logdir=tensorboard
# Open http://localhost:6006
```

## Key Changes

Compared to original Linesight:

- ✅ **Modern build system** - `pyproject.toml` instead of `setup.py` + requirements.txt
- ✅ **uv support** - fast installation with `uv sync`
- ✅ **Updated dependencies** - PyTorch 2.7+, TorchRL 0.6+, CUDA 12.6
- ✅ **Expanded docs** - comprehensive FAQ, troubleshooting, dev guide
- ✅ **Modular config** - 8 separate modules for easier editing

## Documentation

- **[Installation Guide](docs/source/installation.rst)** - detailed setup instructions
- **[First Training](docs/source/first_training.rst)** - get started with training
- **[Configuration Guide](docs/source/configuration_guide.rst)** - all config parameters
- **[User FAQ](docs/source/user_faq.rst)** - 30+ common questions
- **[Dev FAQ](docs/source/dev_faq.rst)** - developer questions
- **[Troubleshooting](docs/source/troubleshooting.rst)** - common issues

**Build HTML docs:**
```bash
cd docs && pip install -e ".[doc]" && make html
```

## Requirements

- **OS:** Windows 10/11 or Linux
- **GPU:** NVIDIA with CUDA 12.x (6GB+ VRAM)
- **RAM:** 20 GB+
- **Python:** 3.10 or 3.11

## Project Structure

```
rulka/
├── config_files/       # Modular configuration (8 modules)
├── trackmania_rl/      # Core RL code (IQN agent, multiprocess)
├── scripts/            # train.py and utilities
├── maps/               # Reference lines (.npy)
├── docs/               # Documentation
└── save/               # Checkpoints and replays
```

## License

MIT License

## Credits

- Original [Linesight](https://github.com/pb4git/linesight) by pb4git
- [donadigo](https://github.com/donadigo) for TMInterface
- TrackMania community

---

⚠️ **Important:** All AI runs are Tool Assisted and must NOT be submitted to official leaderboards.
