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
git clone https://github.com/ArtyomZemlyak/rulka.git
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
4. Create a `.env` file in the project root (user-specific settings):
   ```
   USERNAME=your_tmnf_account
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
python scripts/train.py --config config_files/config_default.yaml

# Monitor (in separate terminal)
tensorboard --logdir=tensorboard
python -m tensorboard.main --logdir=tensorboard
# Open http://localhost:6006
```

## Key Changes

Compared to original Linesight:

- ✅ **Modern build system** - `pyproject.toml` instead of `setup.py` + requirements.txt
- ✅ **uv support** - fast installation with `uv sync`
- ✅ **Updated dependencies** - PyTorch 2.7+, TorchRL 0.6+, CUDA 12.6
- ✅ **Expanded docs** - comprehensive FAQ, troubleshooting, dev guide
- ✅ **YAML config** - single file with sections, versioned per experiment

## Documentation

**Online:** [https://artyomzemlyak.github.io/rulka/](https://artyomzemlyak.github.io/rulka/)

- **[Installation Guide](https://artyomzemlyak.github.io/rulka/installation.html)** - detailed setup instructions
- **[First Training](https://artyomzemlyak.github.io/rulka/first_training.html)** - get started with training
- **[Configuration Guide](https://artyomzemlyak.github.io/rulka/configuration_guide.html)** - all config parameters
- **[User FAQ](https://artyomzemlyak.github.io/rulka/user_faq.html)** - 30+ common questions
- **[Dev FAQ](https://artyomzemlyak.github.io/rulka/dev_faq.html)** - developer questions
- **[Troubleshooting](https://artyomzemlyak.github.io/rulka/troubleshooting.html)** - common issues

**Build HTML docs:**
```bash
# Install documentation dependencies
uv sync --extra docs

# Build documentation
cd docs
python -m sphinx -b html source build/html

# View locally (Windows)
start build/html/index.html
```

**Note:** `docs/build/` is git-ignored (built files are not committed). Documentation is built by GitHub Actions on every push to `main` and deployed to GitHub Pages. To enable it: **Settings → Pages → Source: GitHub Actions**.

## Requirements

- **OS:** Windows 10/11 or Linux
- **GPU:** NVIDIA with CUDA 12.x (6GB+ VRAM)
- **RAM:** 20 GB+
- **Python:** 3.10 or 3.11

## Visual pretraining (replays from leaderboard)

To pretrain the visual backbone on top players’ replays:

Activate the environment first (Windows: `.\.venv\Scripts\activate`; Linux/macOS: `source .venv/bin/activate`).

1. **Download top replays**:
   - **TMNF (Nations Forever):** ManiaExchange (TMNF-X), no auth:
     ```bash
     python scripts/download_top_replays_tmnf.py --track-id 100 --output-dir ./replays_tmnf --top 50
     python scripts/download_top_replays_tmnf.py --track-name "A01" --output-dir ./replays_tmnf
     ```
     **Many popular tracks at once:** discover tracks with good replays and optionally download:
     ```bash
     python scripts/list_popular_tracks_tmnf.py --output track_ids.txt --pages 200 --per-page 100
     python scripts/list_popular_tracks_tmnf.py --output track_ids.txt --min-replays 10
     python scripts/list_popular_tracks_tmnf.py --output maps/track_ids.txt --download-replays --replays-dir ./maps/replays --replays-per-track 10
     ```
     Track IDs from [tmnf.exchange](https://tmnf.exchange). Replays saved as `.replay.gbx`.
   - **TrackMania 2020:** Nadeo API (Ubisoft auth):
     ```bash
     set NADEO_UBI_EMAIL=your@email
     set NADEO_UBI_PASSWORD=your_password
     python scripts/download_top_replays.py --map-uid "MAP_UID" --output-dir ./replays_downloaded --top 50
     ```
     Use [Openplanet API](https://webservices.openplanet.dev/) for map UIDs.

2. **Capture frames** from replays:
   - For `.replay.gbx`: in game, load the map, watch the replay, record the screen (e.g. OBS), then extract frames: `ffmpeg -i video.mp4 -vf fps=10 frames/%05d.png`.
   - For `.inputs` files: `python scripts/capture_frames_from_replays.py --inputs-dir ./best_runs --output-dir ./parsed_actions` (parses to action lists; capture via game + TMI or training pipeline).

3. **Pretrain the backbone** (unsupervised: AE, VAE, SimCLR):
   ```bash
   python scripts/pretrain_visual_backbone.py --data-dir ./frames --task ae --epochs 50 --batch-size 128
   python scripts/pretrain_visual_backbone.py --data-dir ./frames --task simclr --framework lightly  # optional: pip install lightly
   ```
   **Stacked frames** (several consecutive images; temporal order = file sort):
   ```bash
   python scripts/pretrain_visual_backbone.py --data-dir ./frames --n-stack 4 --stack-mode concat --task ae
   ```
   `--stack-mode channel`: N input channels (saves N-ch encoder). `--stack-mode concat`: 1-ch encoder per frame + fusion (saves IQN-compatible 1-ch encoder).
   Saves `encoder.pt` loadable into `IQN_Network.img_head`.

## Project Structure

```
rulka/
├── config_files/       # YAML configuration (config_default.yaml, config_schema.py)
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

*The Contributors list on GitHub includes original Linesight authors; see above for attribution.*

---

⚠️ **Important:** All AI runs are Tool Assisted and must NOT be submitted to official leaderboards.
