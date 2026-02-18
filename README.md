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
- **[TMNF replay download & frame capture](https://artyomzemlyak.github.io/rulka/tmnf_replays.html)** - download replays from TMNF-X, capture frames via TMInterface
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

**TMNF (recommended):** Download replays from TMNF-X and capture frames via TMInterface (deterministic). Full guide: [TMNF replay download & frame capture](https://artyomzemlyak.github.io/rulka/tmnf_replays.html). Quick run: `set PYTHONPATH=scripts & python -m replays_tmnf.download --list-popular --output maps/track_ids.txt --per-page 1000`, then pipeline with `--track-ids` and `--extract-tracks-from-replays`; filter with `python scripts/filter_track_ids_no_respawn.py` and optionally `python scripts/filter_track_ids_custom_maptype.py` (removes non-standard MapType/environment); fix replay filenames with `python scripts/fix_replay_filenames.py` (replaces non-ASCII and spaces with `_`); capture with `python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir maps/img --running-speed 10`. Map previews are handled automatically via TMInterface retry (give_up/press_delete every 3s). Use `--running-speed 10`–`20` (higher values cause the game to skip inputs).

1. **Download top replays** (alternative scripts):
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

3. **Pretrain the backbone** (Level 0 — unsupervised: AE, VAE, SimCLR):

   All settings live in `config_files/pretrain_config.yaml` (`framework: lightning` by default).
   Install the pretrain dependencies once:
   ```bash
   pip install -e ".[pretrain]"   # lightning, lightly, timm
   ```

   **AE (YAML defaults, auto-versioned run dir):**
   ```bash
   python scripts/pretrain_visual_backbone.py --data-dir maps/img
   ```
   **SimCLR with track-level val split:**
   ```bash
   python scripts/pretrain_visual_backbone.py \
       --data-dir maps/img --task simclr --val-fraction 0.1
   ```
   **Stacked frames** (IQN-compatible 1-ch encoder):
   ```bash
   python scripts/pretrain_visual_backbone.py \
       --data-dir maps/img --n-stack 4 --stack-mode concat
   ```
   Override any field via env var (PowerShell: `$env:PRETRAIN_TASK='simclr'`) or `--config my.yaml`.

   Each run creates a versioned subdirectory:
   ```
   output/ptretrain/vis/run_001/
       encoder.pt          ← CNN weights only → goes into IQN
       pretrain_meta.json  ← reproducibility record
       metrics.csv         ← per-epoch loss
       checkpoints/        ← Lightning .ckpt for resuming (not for IQN)
       tensorboard/
       csv/
   ```
   Use `--run-name ae_baseline` for a fixed name instead of `run_001`.

4. **Inject encoder into IQN** (creates `weights1.torch` / `weights2.torch`):
   ```bash
   python scripts/init_iqn_from_encoder.py \
       --encoder-pt output/ptretrain/vis/run_001/encoder.pt \
       --save-dir   save/
   ```
   Multi-channel encoders are automatically averaged to 1-channel.
   Start the learner normally afterwards — it loads the checkpoint on startup.

   Dry-run validation (no files written):
   ```bash
   python scripts/init_iqn_from_encoder.py \
       --encoder-pt output/ptretrain/vis/run_001/encoder.pt --dry-run
   ```

   See `docs/source/experiments/pretrain_replay_roadmap.rst` for the full experiment
   matrix and KPI tracking guide (Level 0 → Level 1/2/4 roadmap).

## Project Structure

```
rulka/
├── config_files/
│   ├── config_default.yaml           # RL training config
│   ├── config_schema.py              # Pydantic schema for RL config
│   ├── config_loader.py              # loader + get_config()
│   ├── pretrain_config.yaml          # Level 0 pretrain config (edit defaults here)
│   └── pretrain_schema.py            # PretrainConfig(BaseSettings) + load_pretrain_config()
├── trackmania_rl/
│   ├── pretrain_visual/       # Level 0 pretraining package
│   │   ├── contract.py        #   artifact schema constants
│   │   ├── datasets.py        #   ReplayFrameDataset + Lightning DataModule
│   │   ├── models.py          #   encoder/decoder factories (IQN-compatible)
│   │   ├── tasks.py           #   Lightning modules: AE, VAE, SimCLR
│   │   ├── train.py           #   PretrainConfig + train_pretrain dispatcher
│   │   └── export.py          #   save/load/validate artifact
│   └── ...                    # IQN agent, multiprocess, utilities
├── scripts/
│   ├── pretrain_visual_backbone.py   # Level 0 train entry point (CLI)
│   ├── init_iqn_from_encoder.py      # inject encoder.pt into IQN checkpoint
│   └── ...                           # train.py and other utilities
├── maps/               # Reference lines (.npy) + captured frames (maps/img/)
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
