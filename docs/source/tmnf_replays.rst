.. _tmnf_replays:

TMNF replay download and frame capture
======================================

Download replays and maps from TMNF-X (ManiaExchange), then capture frames via TMInterface for visual pretraining or dataset building.

Pipeline: steps to run (in order)
---------------------------------

Run the steps in order. All commands from the project root.

**Step 1.** Download the track list (once). Use 1000 or 10000 tracks per page (``--per-page``).

.. code-block:: bash

   # Windows (from project root)
   set PYTHONPATH=scripts & python -m replays_tmnf.download --list-popular --output maps/track_ids.txt --per-page 1000

   # Linux / macOS
   PYTHONPATH=scripts python -m replays_tmnf.download --list-popular --output maps/track_ids.txt --per-page 1000

**Step 2.** Download replays and extract maps (resumable pipeline; Ctrl+C saves progress).

.. code-block:: bash

   # Windows
   set PYTHONPATH=scripts & python -m replays_tmnf.download --track-ids maps/track_ids.txt --replays-dir ./maps/replays --tracks-dir ./maps/tracks --extract-tracks-from-replays --workers 64 --api-workers 64

   # Linux / macOS
   PYTHONPATH=scripts python -m replays_tmnf.download --track-ids maps/track_ids.txt --replays-dir ./maps/replays --tracks-dir ./maps/tracks --extract-tracks-from-replays --workers 256 --api-workers 32

After restart, step 2 continues from ``./maps/replays/.replay_progress``.

**Step 3.** Filter tracks with no respawn (keeps only tracks where replays do not respawn — for stable frame capture).

.. code-block:: bash

   python scripts/filter_track_ids_no_respawn.py --input maps/track_ids.txt --output maps/track_ids_no_respawn.txt --workers 16 -r maps/replays

**Step 3a.** (Optional) Filter tracks with non-standard MapType or long preview (removes tracks with custom environments or "Press Enter to start" screens).

.. code-block:: bash

   # Filter by environment (e.g., remove Stunts, custom MapType)
   python scripts/filter_track_ids_custom_maptype.py --input maps/track_ids_no_respawn.txt --output maps/track_ids_standard.txt --tracks-dir maps/tracks --jobs 16

   # (Future) Filter by MediaTracker preview duration (not yet implemented)
   python scripts/filter_track_ids_custom_maptype.py --input maps/track_ids_no_respawn.txt --output maps/track_ids_standard.txt --tracks-dir maps/tracks --max-preview-duration 15.0 --jobs 16

This removes maps with non-standard environments (e.g., not Stadium/Speed/Alpine/Rally/Bay/Island/Coast/Desert) and, when implemented, maps with long MediaTracker intros. Use ``--only-with-maps`` to skip tracks without Challenge.Gbx files.

**Step 3b.** Fix replay filenames (replaces non-ASCII characters and spaces with ``_``). TMInterface cannot load script files with special characters in their names.

.. code-block:: bash

   python scripts/fix_replay_filenames.py --dry-run   # preview changes
   python scripts/fix_replay_filenames.py              # apply

**Step 4.** Capture frames from replays (TMInterface; game must be running). Use the filtered track list.

.. code-block:: bash

   python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir maps/img --workers 1 --width 256 --height 256 --running-speed 16 --fps 1 --track-ids maps/track_ids_no_respawn.txt --max-replays-per-track 1

**Map preview handling:** Previews and "Press Enter to start" screens are handled automatically via ``disable_forced_camera`` + ``skip_map_load_screens``. If the game still doesn't start within 3 seconds (no RUN_STEP messages), the script sends TMInterface ``give_up`` / ``press delete`` commands to restart the race every 3 seconds (up to 25 seconds total), then skips the map. Use ``--write-enter-maps`` to collect track IDs of maps that didn't start, then ``--exclude-enter-maps`` on the next run.

**Note on --running-speed:** Values above ~20 may cause the game to skip reading inputs, making the car stand still. For reliable replay, use 10–20.

Output: ``maps/img/<track_id>/<replay_name>/`` — frames (jpeg), ``metadata.json``, ``manifest.json``. Details below.

---

How it works (details)
----------------------

The following sections describe the download modules and pipeline, the filter script, and frame capture.

Module layout (replays_tmnf)
----------------------------

- **api.py** — TMNF-X API: track search, replay list, download replays/maps.
- **list_popular.py** — list popular tracks; used by download when ``--list-popular``.
- **download.py** — entry point: CLI for ``--track-id`` / ``--track-name`` / ``--track-ids`` or ``--list-popular``.
- **pipeline.py** — pipeline for ``--track-ids``: producer → replay queue → download workers → map workers; resume via ``.replay_progress``.

Modes and options (download)
----------------------------

- **``--list-popular``** — fetch popular track list from TMNF-X, write to ``--output``. Optionally ``--download-replays`` and/or ``--download-tracks``.
- **``--track-ids <file>``** — run pipeline: replays to ``--replays-dir``, maps via ``--extract-tracks-from-replays`` to ``--tracks-dir``.
- **``--track-id`` / ``--track-name``** — single track; replays to ``--output-dir`` (default ``replays_tmnf``).

Pipeline (--track-ids)
----------------------

1. **API workers** (``--api-workers``) — request replay lists per track in parallel; often the bottleneck without many workers.
2. **Download workers** — save replays under ``replays-dir/track_id/...``.
3. **Map workers** — with ``--extract-tracks-from-replays``, extract map to ``tracks-dir`` as ``{TrackId}.Challenge.Gbx``.

Resume: ``replays-dir/.replay_progress`` stores the next track index. Ctrl+C stops and saves progress.

Filter tracks (step 3): filter_track_ids_no_respawn.py
-----------------------------------------------------

**scripts/filter_track_ids_no_respawn.py** reads a track ID list (e.g. ``maps/track_ids.txt``) and replays in ``--replays-dir`` (or ``-r``), detects tracks where any replay respawns, and writes a new list (e.g. ``maps/track_ids_no_respawn.txt``) containing only tracks with no respawn. Use this list in step 4 for more stable capture (no respawns during replay). Arguments: ``--input``, ``--output``, ``-r`` / ``--replays-dir``, ``--workers``.

Filter tracks (step 3a): filter_track_ids_custom_maptype.py
-----------------------------------------------------------

**scripts/filter_track_ids_custom_maptype.py** reads a track ID list and Challenge.Gbx files in ``--tracks-dir`` (or ``-t``), detects tracks with non-standard environments (e.g., Stunts, custom MapType) or long MediaTracker previews (not yet implemented), and writes a new list (e.g. ``maps/track_ids_standard.txt``) containing only standard tracks. Use this list in step 4 to avoid tracks with "Press Enter to start" screens or custom map scripts.

**Currently checks:**
  - Environment field: excludes maps with environment not in {Stadium, Speed, Alpine, Rally, Bay, Island, Coast, Desert}

**Future checks (not yet implemented):**
  - MediaTracker intro duration > threshold (``--max-preview-duration N.0``): requires pygbx MediaTracker parsing support

Arguments: ``--input`` (default ``maps/track_ids.txt``), ``--output`` (default ``maps/track_ids_standard.txt``), ``-t`` / ``--tracks-dir`` (default ``maps/tracks_tmnf``), ``-j`` / ``--jobs`` (default cpu_count), ``--only-with-maps`` (skip tracks without Challenge.Gbx), ``--max-preview-duration`` (not yet implemented).

Example:

.. code-block:: bash

   python scripts/filter_track_ids_custom_maptype.py --input maps/track_ids_no_respawn.txt --output maps/track_ids_standard.txt --tracks-dir maps/tracks --jobs 16

Main arguments (download)
------------------------

+---------------------------+------------------------------------------------------------------+
| Argument                  | Effect                                                           |
+===========================+==================================================================+
| ``--list-popular``        | Fetch popular track list, write to ``--output``.                  |
+---------------------------+------------------------------------------------------------------+
| ``--track-ids <file>``     | Run pipeline from file.                                          |
+---------------------------+------------------------------------------------------------------+
| ``--output``              | Output file for track ID list (``--list-popular``).              |
+---------------------------+------------------------------------------------------------------+
| ``--output-dir``          | Replay directory for single track (default ``replays_tmnf``).   |
+---------------------------+------------------------------------------------------------------+
| ``--replays-dir``         | Replay directory (layout ``replays-dir/track_id/...``).         |
+---------------------------+------------------------------------------------------------------+
| ``--tracks-dir``          | Map directory for ``--extract-tracks-from-replays``.            |
+---------------------------+------------------------------------------------------------------+
| ``--per-page``            | Tracks per page (e.g. 1000, 10000).                              |
+---------------------------+------------------------------------------------------------------+
| ``--top``                 | Top replays per track (default 50).                             |
+---------------------------+------------------------------------------------------------------+
| ``--workers``             | Parallel download and map workers.                               |
+---------------------------+------------------------------------------------------------------+
| ``--api-workers``         | Parallel API requests for replay lists (0 = use ``--workers``).  |
+---------------------------+------------------------------------------------------------------+
| ``--extract-tracks-from-replays`` | Extract map from replays (requires pygbx).                |
+---------------------------+------------------------------------------------------------------+
| ``--dry-run``             | List what would be downloaded; no files written.                 |
+---------------------------+------------------------------------------------------------------+

Extracting map from replay
--------------------------

A TMNF replay GBX embeds the map. With ``--extract-tracks-from-replays``, the map is extracted into ``--tracks-dir`` as ``{TrackId}.Challenge.Gbx`` using **pygbx** (project dependency).

Frame capture (capture_replays_tmnf.py)
--------------------------------------

**scripts/capture_replays_tmnf.py** runs replays from ``maps/replays`` (layout: ``replays_dir/track_id/*.replay.gbx``) via TMInterface, captures screenshots at a given FPS and resolution, and saves frames with timing and metadata. Maps are expected in ``--tracks-dir`` (default ``maps/tracks``): ``tracks-dir/track_id/*.Challenge.Gbx`` or ``tracks-dir/*.Challenge.Gbx`` — same layout as the download pipeline with ``--extract-tracks-from-replays``.

**Method:** TMInterface native script loading. Each replay is converted to a TMInterface input script (``.replay.gbx`` → ``.txt`` with ``press``/``steer`` commands); the script is loaded with ``load script.txt`` and the game replays inputs deterministically. Order is **load script before map** (per TMInterface docs). Finish is handled **by time** (switch to next replay/map shortly before nominal finish to avoid the medal screen and connection issues).

**Output (per replay):** ``output_dir/track_id/replay_name/``

- **metadata.json** — track_id, replay_name, challenge_name, fps, width, height, capture_interval_ms, step_ms, race_time_ms, total_frames.
- **manifest.json** — per-frame entries: file, step, time_ms, inputs, state (position, speeds, rotation, cp_times_ms, current_checkpoint), capture_timestamp_utc.
- **frame_<step>_<time_ms>ms.jpeg** — images (step = frame index, time_ms = simulation time).
- **frame_<step>_<time_ms>ms.json** — optional (``--per-frame-json``).

**Arguments:** ``--replays-dir``, ``--output-dir``, ``--tracks-dir``, ``--width``, ``--height``, ``--fps``, ``--workers``, ``--base-tmi-port``, ``--track-ids``, ``--track-id``, ``--max-replays-per-track``, ``--per-frame-json``, ``--running-speed``, ``--write-enter-maps``, ``--exclude-enter-maps``, ``--log-level``, ``--config``.

**Port:** ``BASE_TMI_PORT`` in ``.env`` (or ``--base-tmi-port``) must match the game’s TMInterface console (e.g. "Port set to 8480"). If the script hangs at "Waiting for game to finish loading...", check the port.

**Track vs replay loading:** The script sends only the track **filename** (e.g. ``1000074.Challenge.Gbx``) via ``map ...``; the game looks in its ``Tracks/Challenges`` folder. The replay is not passed by path: the script copies the replay into the game’s **Autosaves** folder (``trackmania_base_path/Tracks/Replays/Autosaves`` as ``{Username}_{MapName}.Replay.gbx``). Set ``trackmania_base_path`` in config or ``.env`` (e.g. ``TRACKMANIA_BASE_PATH=C:\Users\...\TrackMania``). Default is ``Documents\TrackMania``.

**Multiple workers:** Tasks are grouped by track so each worker processes all replays of one track sequentially; different workers handle different tracks in parallel. This avoids overwriting the same Autosaves ghost file.

**Connection handling:** If TMInterface disconnects (e.g. game closed), the script clears the connection and the next replay will reconnect automatically.

Examples (from project root)
---------------------------

.. code-block:: bash

   # Single worker, 64x64, 10 FPS
   python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir maps/img

   # 256x256, 1 FPS, specific track list, one replay per track
   python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir maps/img --workers 1 --width 256 --height 256 --running-speed 10 --fps 1 --track-ids maps/track_ids_no_respawn.txt --max-replays-per-track 1

   # Validate single replay
   python scripts/capture_replays_tmnf_validate.py --replay-path maps/replays/924307/pos1_ben3847_89250ms.replay.gbx --output-dir maps/img_validate --fps 10 --step-ms 10

   # Per-frame JSON
   python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir out --per-frame-json

   # Single track
   python scripts/capture_replays_tmnf.py --track-id 12345 --replays-dir maps/replays --output-dir out

API (TMNF-X / ManiaExchange)
-----------------------------

- Track search / replay list: TMNF-X API (e.g. ``https://tmnf.exchange/api/...``).
- Replay download: ``/recordgbx/{ReplayId}``; maps: ``/trackgbx/{TrackId}``.
