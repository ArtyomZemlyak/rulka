# TMNF replay download

Download replays and maps from TMNF-X (ManiaExchange).

## Quick start

**Step 1.** Download the track list (once). Use **1000 or 10000 tracks per page** (`--per-page`): 10000 — fewer API requests, 1000 — reasonable tradeoff.

```bash
# Windows (from project root)
set PYTHONPATH=scripts & python -m replays_tmnf.download --list-popular --output maps/track_ids.txt --per-page 1000

# Linux / macOS
PYTHONPATH=scripts python -m replays_tmnf.download --list-popular --output maps/track_ids.txt --per-page 1000
```

**Step 2.** Download replays and extract maps (resumable pipeline, Ctrl+C and progress bar):

```bash
# Windows (--api-workers 32 speeds up replay list fetching from API)
set PYTHONPATH=scripts & python -m replays_tmnf.download --track-ids maps/track_ids.txt --replays-dir ./maps/replays --tracks-dir ./maps/tracks --extract-tracks-from-replays --workers 64 --api-workers 64

# Linux / macOS
PYTHONPATH=scripts python -m replays_tmnf.download --track-ids maps/track_ids.txt --replays-dir ./maps/replays --tracks-dir ./maps/tracks --extract-tracks-from-replays --workers 256 --api-workers 32
```

After restart, the same step 2 continues from where it left off. Progress is stored in `./maps/replays/.replay_progress`.

---

## Module layout

- **api.py** — API: track search, replay list, download replays/maps.
- **list_popular.py** — list popular tracks (`run_list_popular`). Used by download when `--list-popular`.
- **download.py** — **entry point**: CLI for `--track-id` / `--track-name` / `--track-ids` or `--list-popular`.
- **pipeline.py** — pipeline for `--track-ids`: producer → replay queue → download workers → map queue → map workers; resume via `.replay_progress`.

## Modes and options

- **`--list-popular`** — fetch popular track list from TMNF-X, write IDs to `--output`. Optionally download replays (`--download-replays`) and/or maps (`--download-tracks`) right away.
- **`--track-ids <file>`** — track list already exists; without `--dry-run` runs the **pipeline** (parallel download with queues and resume). Replays go to `--replays-dir`, maps from replays via `--extract-tracks-from-replays` to `--tracks-dir`.
- **`--track-id` / `--track-name`** — single track: search by name or one ID, then plain download (no pipeline). Replays go to `--output-dir` (default `replays_tmnf`).

Options `--download-replays` and `--download-tracks` apply only in `--list-popular` mode. In the pipeline (`--track-ids`) maps are fetched only with `--extract-tracks-from-replays`.

## Pipeline (--track-ids)

1. **API workers** (`--api-workers`, default same as `--workers`) — in parallel request replay lists per track (one API request per track) and push tasks to the download queue. **Bottleneck was here**: one thread gave ~2 tracks/s; many threads (e.g. 32) speed it up a lot.
2. **Download workers** — take replays from the queue and save under `--replays-dir/track_id/...`.
3. **Map workers** — with `--extract-tracks-from-replays` extract the map to `--tracks-dir` (if not already present).

Everything runs in parallel as tasks appear. `--workers` sets the number of download and map workers; `--api-workers` sets parallel API requests for replay lists (0 = use `--workers`). Resume: `replays-dir/.replay_progress` holds the next track index; on restart the run continues from there. Ctrl+C stops the pipeline and saves progress.

## Main arguments

| Argument | Effect |
|----------|--------|
| `--list-popular` | Mode: fetch popular track list from API, write to `--output`. |
| `--track-ids <file>` | Mode: read IDs from file and run download pipeline (replays + optional maps). |
| `--output` | File for ID list when using `--list-popular` (e.g. `maps/track_ids.txt`). |
| `--output-dir` | Directory for replays when using `--track-id` or `--track-name` (single track; default `replays_tmnf`). Layout: `output-dir/track_id/...`. |
| `--replays-dir` | Replay directory (layout `replays-dir/track_id/...`). For pipeline and for `--list-popular --download-replays`. |
| `--tracks-dir` | Map directory (`.Challenge.Gbx`). For `--extract-tracks-from-replays` or `--download-tracks`. |
| `--per-page` | Tracks per page for `--list-popular` (e.g. 1000 or 10000 — fewer requests). |
| `--top` | How many top replays per track to download (default 50). |
| `--workers` | Number of parallel downloads (and map extraction workers in the pipeline). |
| `--api-workers` | Number of parallel API requests for replay lists (0 = same as `--workers`). Speeds up pipeline with large track lists. |
| `--extract-tracks-from-replays` | Extract map from replays into `--tracks-dir` (requires pygbx). In pipeline: one file per track as `{track_id}.Challenge.Gbx` in `--tracks-dir`. |
| `--rate-delay` | Delay in seconds between API requests (default 0). |
| `--no-tqdm` | Disable progress bar, logs only. |
| `--dry-run` | Only list what would be downloaded (no files written). |

## Running from project root

```bash
# Track list only (step 1)
set PYTHONPATH=scripts & python -m replays_tmnf.download --list-popular --output maps/track_ids.txt --per-page 1000

# Pipeline: replays + maps (step 2)
set PYTHONPATH=scripts & python -m replays_tmnf.download --track-ids maps/track_ids.txt --replays-dir ./maps/replays --tracks-dir ./maps/tracks --extract-tracks-from-replays --workers 256 --api-workers 32

# Single track by ID
set PYTHONPATH=scripts & python -m replays_tmnf.download --track-id 100 --output-dir ./replays_tmnf --top 50
```

**From the scripts/ directory** (no PYTHONPATH needed):

```bash
cd scripts
python -m replays_tmnf.download --list-popular --output ../maps/track_ids.txt --per-page 1000
python -m replays_tmnf.download --track-ids ../maps/track_ids.txt --replays-dir ../maps/replays --tracks-dir ../maps/tracks --extract-tracks-from-replays --workers 256
```

## Pagination and batching

- **Track listing** — pagination with `--per-page` and cursor; prefetch next page. 1000 or 10000 recommended for fast large lists.
- **Replay list** — one request per track (API has no batch for multiple tracks); parallelism via `--api-workers` (list requests) and `--workers` (file downloads).
- **Replay/map download** — one URL per file; speedup only via parallel requests.

## API (ManiaExchange / TMNF-X)

- Overview v2: https://api2.mania.exchange/
- Search Tracks: https://api2.mania.exchange/Method/Index/43 (TMNF-X: `https://tmnf.exchange/api/tracks?{params}`)
- Get Track Replays: https://api2.mania.exchange/Method/Index/45
- Replay download on site: `/recordgbx/{ReplayId}`; maps: `/trackgbx/{TrackId}`.

## Extracting map from replay

A TMNF replay GBX embeds the map (challenge). With `--extract-tracks-from-replays`, the map file is extracted from downloaded replays into `--tracks-dir` as `{TrackId}.Challenge.Gbx`. Uses **pygbx** (project dependency).

## Frame capture from replays (capture_replays_tmnf.py)

The script **scripts/capture_replays_tmnf.py** runs replays from `maps/replays` (structure: `replays_dir/track_id/*.replay.gbx`) via TMInterface in one or more game windows, captures screenshots at a given FPS and resolution, and saves frames with timing and metadata for recovery. It expects maps in `--tracks-dir` (default `maps/tracks`): either `tracks-dir/track_id/*.Challenge.Gbx` or `tracks-dir/*.Challenge.Gbx` — the same layout as produced by the download pipeline with `--extract-tracks-from-replays`.

**Output (per replay):** `output_dir/track_id/replay_name/`

- **metadata.json** — global run parameters: track_id, replay_name, challenge_name, fps, width, height, capture_interval_ms, step_ms, run_steps_per_action, race_time_ms, total_frames.
- **manifest.json** — array of per-frame entries: file, step, time_ms, inputs, state (position, speeds, rotation, cp_times_ms, current_checkpoint), capture_timestamp_utc (ISO 8601).
- **frame_&lt;step&gt;_&lt;time_ms&gt;ms.jpeg** — images; naming uses step (frame index) and time_ms (simulation race time in ms) for unambiguous recovery.
- **frame_&lt;step&gt;_&lt;time_ms&gt;ms.json** — optional (with `--per-frame-json`); one file per frame with the same content as the corresponding manifest entry for quick single-frame lookup.

**Examples (from project root):**

```bash
# Single worker, 64x64 frames, 10 FPS
python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir maps/img

# Two game windows, 128x128, 10 FPS
python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir maps/img --workers 1 --width 256 --height 256 --fps 100  --running-speed 1 --input-time-offset-ms "-10" --step-ms 10

# With per-frame JSON files for each screenshot
python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir out --per-frame-json

# Single track only
python scripts/capture_replays_tmnf.py --track-id 12345 --replays-dir maps/replays --output-dir out

# Only tracks from file
python scripts/capture_replays_tmnf.py --track-ids maps/track_ids.txt --replays-dir maps/replays --output-dir out
```

**Arguments:** `--replays-dir`, `--output-dir`, `--tracks-dir`, `--width`, `--height`, `--fps`, `--workers`, `--base-tmi-port`, `--track-ids`, `--track-id`, `--per-frame-json`, `--config`. **Port:** `BASE_TMI_PORT` in `.env` (or `--base-tmi-port`) must match the port shown in the game's TMInterface console (e.g. "Port set to 8480"). If the script hangs at "Waiting for game to finish loading...", check that the port in the game matches.

**Track vs replay loading (same as training):** **Track** — we send only the track **filename** (e.g. `1000074.Challenge.Gbx`) via the console command `map ...`; the game looks for it in its own `Tracks/Challenges` folder (no full path). **Replay** — loading the replay is different: the game does not take a path. The script copies the replay file into the game's Autosaves folder; when the track is (re)loaded, the game picks up the ghost from there by file location and naming, so no path is passed for the replay.

**Multiple workers:** All game instances share one Autosaves folder (one `trackmania_base_path`). To avoid overwriting the same ghost file, tasks are **grouped by track**: each worker takes a whole track and processes all its replays one after another; different workers process different tracks in parallel. So you get parallelism across tracks and no conflict on Autosaves.

**Replay folder:** So that the game loads the replay as a ghost, the script copies each replay into the game’s user data folder: `trackmania_base_path/Tracks/Replays/Autosaves` as `{Username}_{MapName}.Replay.gbx`. Set **trackmania_base_path** in config or `.env` (e.g. `TRACKMANIA_BASE_PATH=C:\Users\artyo\Documents\TrackMania`) to the folder that contains `Tracks\Replays` (the same folder where you see Autosaves in Explorer). Default is `Documents\TrackMania`. The script logs "Replay copied to: ... (base_path=...)" so you can confirm. If no files appear, the base path was wrong or the copy was skipped (the script now always copies before loading the track).
