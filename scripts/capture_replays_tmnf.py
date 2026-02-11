"""
Capture frames from replays in maps/replays (TMNF .replay.gbx).

**Method: TMInterface Native Script Loading (Maximum Determinism)**
  Instead of manually calling set_input_state every tick (which requires
  rewind_to_current_state and causes floating-point drift), we use TMInterface's
  native replay validation system:
  
  1. Convert .replay.gbx → TMInterface script format (.txt)
     - Parse control_entries from replay using pygbx
     - Generate TMInterface commands: "0.68 press right", "0.83 rel right", etc.
  
  2. Load script via execute_command("load script.txt")
     - TMInterface injects inputs internally (fully deterministic!)
     - No manual set_input_state calls, no rewind_to_current_state drift
  
  3. Capture frames via on_step + request_frame
     - We only observe and record, not control inputs
  
  This approach matches TMInterface's native "Validation" feature and achieves
  exact replay reproduction, identical to the human run.

**Determinism (use_valseed false)**:
  TMInterface's `use_valseed` controls whether the car's initial state is
  randomized. We set it false for deterministic physics.

**TMInterface Prerequisites** (from official docs):
  Replays MUST finish the race. Incomplete replays cannot be validated.
  See: https://donadigo.com/tminterface/input-extraction

Input: directory with structure maps/replays/<track_id>/*.replay.gbx.

Output (per replay): output_dir/<track_id>/<replay_name>/
  - metadata.json   — track_id, replay_name, fps, width, height, step_ms,
                     race_time_ms, total_frames, race_finished, etc.
  - manifest.json   — per-frame entries: file, step, time_ms, inputs, action_idx.
  - frame_*.jpeg    — grayscale frames captured at the configured resolution.

Multiple windows: --workers N runs N game instances (one per worker, each on
base_tmi_port + worker_id).  Tasks are grouped by track so each track is handled
by one worker at a time.

Usage:
  python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir capture_frames_out
  python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir out --step-ms 10  # finer TMNF timing
  python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir out --workers 2 --fps 100
  python scripts/capture_replays_tmnf.py --replays-dir maps/replays --output-dir out --max-replays-per-track 5  # top 5 only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Project root and config
_script_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_script_root))
sys.path.insert(0, str(_script_root / "scripts"))

from config_files.config_loader import get_config, load_config, set_config

_default_yaml = _script_root / "config_files" / "config_default.yaml"
if _default_yaml.exists():
    set_config(load_config(_default_yaml))

# Reuse action mapping from the existing capture_frames_from_replays script
from capture_frames_from_replays import action_dict_to_index

log = logging.getLogger(__name__)


# ---------- 1. Replay list: collect (track_id, path) ----------

def collect_replay_tasks(
    replays_dir: Path,
    track_ids_path: Path | None = None,
    track_id_single: str | None = None,
    max_replays_per_track: int | None = None,
) -> list[tuple[str, Path]]:
    """
    Collect (track_id, replay_path) under replays_dir for *.replay.gbx (and *.Replay.gbx).
    
    If track_ids_path is set: only include track_ids from that file, in ORDER (popularity).
    If track_id_single is set: only that track_id.
    If max_replays_per_track is set: limit to top N replays per track (by filename, e.g. pos1, pos2, ...).
    
    Returns list preserving track_ids.txt order (most popular first).
    """
    tasks: list[tuple[str, Path]] = []
    
    # Build ordered list of track_ids to process
    if track_id_single is not None:
        track_ids_ordered = [str(track_id_single)]
    elif track_ids_path and track_ids_path.exists():
        # Preserve order from file (popularity ranking)
        track_ids_ordered = [
            line.strip() for line in track_ids_path.read_text().splitlines()
            if line.strip()
        ]
    else:
        # No filter — all tracks in arbitrary order
        if not replays_dir.exists():
            return tasks
        track_ids_ordered = sorted(d.name for d in replays_dir.iterdir() if d.is_dir())

    if not replays_dir.exists():
        return tasks

    # Process each track in order
    for track_id in track_ids_ordered:
        track_dir = replays_dir / track_id
        if not track_dir.is_dir():
            continue
        
        # Collect all replay files for this track
        replays = []
        for p in track_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() == ".gbx" and ".replay" in p.name.lower():
                replays.append(p)
        
        # Sort by filename (pos1, pos2, ... or alphabetically)
        # Replays from download_top_replays are typically named pos1_..., pos10_..., etc.
        replays.sort(key=lambda p: p.name)
        
        # Limit to top N if requested
        if max_replays_per_track is not None:
            replays = replays[:max_replays_per_track]
        
        for p in replays:
            tasks.append((track_id, p))

    return tasks


def group_replay_tasks_by_track(
    tasks: list[tuple[str, Path]],
) -> list[tuple[str, list[Path]]]:
    """
    Group (track_id, replay_path) into (track_id, [replay_path, ...]).
    Ensures only one worker processes a track at a time (no Autosaves overwrite).
    """
    from itertools import groupby
    grouped: list[tuple[str, list[Path]]] = []
    for track_id, group in groupby(tasks, key=lambda x: x[0]):
        replay_paths = [p for _, p in group]
        grouped.append((track_id, replay_paths))
    return grouped


# ---------- 2. Extract inputs from .replay.gbx (pygbx) ----------

# --- Control name mapping ---
# Binary (keyboard) events: each name maps to one input key.
_BINARY_EVENT_TO_KEY: dict[str, str] = {
    "accelerate": "accelerate",
    "gas": "accelerate",
    "up": "accelerate",
    "brake": "brake",
    "down": "brake",
    "steerleft": "left",
    "steer_left": "left",
    "steer left": "left",
    "left": "left",
    "steerright": "right",
    "steer_right": "right",
    "steer right": "right",
    "right": "right",
}
# Analog steer axis names: value from (flags<<16)|enabled, interpreted as signed int32.
# Convention (matches TM-Gbx-input-visualizer): value > 0 = right, value < 0 = left.
_ANALOG_STEER_NAMES: frozenset[str] = frozenset({
    "steer", "_fakepad_steer", "fakepad_steer",
})
_STEER_DEADZONE = 1000  # values within ±deadzone of 0 → no steer
_DEBUG_EVENTS_TO_LOG = 30
_DEBUG_POLICY_ACTIONS_TO_LOG = 40


def _event_to_analog_value(ce) -> int:
    """
    Compute signed analog value from ControlEntry (matches TM-Gbx-input-visualizer).
    Value = -sign_extend24((flags << 16) | enabled)
    """
    flags_raw = int(getattr(ce, "flags", getattr(ce, "Flags", 0)))
    enabled_raw = int(getattr(ce, "enabled", getattr(ce, "Enabled", 0)))
    combined = (flags_raw << 16) | enabled_raw
    val = combined & 0xFFFFFF
    if val & 0x800000:  # sign extend from 24 bits
        val -= 0x1000000
    return -val


def convert_replay_to_tmi_script(
    replay_path: Path,
    output_txt_path: Path,
) -> dict[str, Any] | None:
    """
    Convert .replay.gbx to TMInterface script format (.txt).
    
    TMInterface script format (time in seconds):
        0 press up              # Start accelerating at t=0s
        0.68 press right        # Steer right at t=0.68s
        0.83 rel right          # Release right at t=0.83s
        ...
    
    Returns metadata dict {race_time_ms, n_events, ghost_uid, game_version} or None on error.
    """
    try:
        from pygbx import Gbx, GbxType
    except ImportError:
        log.warning("pygbx not available")
        return None

    import logging as _logmod
    _saved = []
    for _name in ("pygbx", "pygbx.gbx"):
        _log = _logmod.getLogger(_name)
        _saved.append((_log, _log.level))
        _log.setLevel(_logmod.CRITICAL)
    
    try:
        gbx = Gbx(str(replay_path))
        ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
        if not ghosts:
            return None
        ghost = min(ghosts, key=lambda g: getattr(g, "cp_times", [0])[-1] if getattr(g, "cp_times", None) else 0)
        race_time = int(getattr(ghost, "race_time", 0) or 0)
        control_entries = list(getattr(ghost, "control_entries", []) or [])
        control_names = list(getattr(ghost, "control_names", []) or [])
        cp_times = list(getattr(ghost, "cp_times", []) or [])
    except Exception as e:
        log.debug("Failed to parse replay %s: %s", replay_path, e)
        return None
    finally:
        for _log, _lev in _saved:
            _log.setLevel(_lev)

    if race_time <= 0:
        return None
    if not control_entries:
        log.warning("Replay %s: incomplete (no inputs)", replay_path.name)
        return None

    if control_names:
        log.info("  Control names: %s", control_names)

    # Check for axis inversion
    cancel_default_steer_inversion = any(
        getattr(ce, "event_name", "") == "_FakeDontInverseAxis"
        for ce in control_entries
    )

    # Convert events to TMInterface script commands
    commands: list[tuple[float, str]] = []  # (time_seconds, command_string)
    key_state: dict[str, bool] = {
        "accelerate": False, "brake": False, "left": False, "right": False,
    }
    
    for ce in control_entries:
        t_ms = getattr(ce, "time", getattr(ce, "Time", None))
        if t_ms is None or t_ms < 0:
            continue
        
        event_name = getattr(ce, "event_name", getattr(ce, "EventName", "")).lower().strip()
        if not event_name or event_name.startswith("_fake"):
            continue
        
        t_sec = t_ms / 1000.0  # Convert milliseconds to seconds
        
        # Binary keys
        key = _BINARY_EVENT_TO_KEY.get(event_name)
        if key is not None:
            pressed = _event_to_analog_value(ce) != 0
            old_state = key_state[key]
            if pressed != old_state:
                key_state[key] = pressed
                action = "press" if pressed else "rel"
                tmi_key = {"accelerate": "up", "brake": "down", "left": "left", "right": "right"}[key]
                commands.append((t_sec, f"{action} {tmi_key}"))
            continue
        
        # Analog steer - use TMInterface 'steer' command with analog value
        if event_name in _ANALOG_STEER_NAMES:
            analog_val = _event_to_analog_value(ce)
            if cancel_default_steer_inversion:
                analog_val = -analog_val
            
            # Use steer command with analog value (range: -65536 to 65536)
            # TMInterface expects integer values in this range
            steer_val = int(analog_val)
            commands.append((t_sec, f"steer {steer_val}"))
            continue

    # Sort by time
    commands.sort(key=lambda x: x[0])
    
    # Write TMInterface script (time in seconds with 2 decimal places)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_txt_path.open("w", encoding="utf-8") as f:
        for t_sec, cmd in commands:
            f.write(f"{t_sec:.2f} {cmd}\n")
    
    # Log first few commands for verification
    log.info(
        "  Converted %s → %s (%d commands, race_time=%.2fs)",
        replay_path.name, output_txt_path.name, len(commands), race_time / 1000.0,
    )
    if commands:
        log.info("  First 10 commands:")
        for i, (t_sec, cmd) in enumerate(commands[:10]):
            log.info("    %d: %.2f %s", i, t_sec, cmd)
    
    metadata = {
        "race_time_ms": race_time,
        "n_events": len(commands),
        "race_finished": bool(cp_times),
        "n_checkpoints": len(cp_times),
    }
    uid_val = getattr(ghost, "uid", getattr(ghost, "UID", None))
    if uid_val:
        metadata["ghost_uid"] = str(uid_val)
    gv = getattr(ghost, "game_version", getattr(ghost, "GameVersion", None))
    if gv:
        metadata["game_version"] = str(gv)
    
    return metadata


def _sample_inputs_at_steps(
    race_time_ms: int,
    sample_period_ms: int,
    control_entries: list,
    step_ms: int,
    input_time_offset_ms: int = 0,
    control_names: list[str] | None = None,
) -> list[dict[str, bool]]:
    """
    Build list of {accelerate, brake, left, right} per step at 0, step_ms, 2*step_ms, ...

    pygbx ``ControlEntry`` objects are **events** (key press / release)::

        ControlEntry(time, event_name, enabled, flags)

    ``event_name`` comes from the ghost's ``control_names`` list.

    For binary (keyboard) events:
        ``event_name`` is e.g. "Accelerate", "Brake", "SteerLeft", "SteerRight".
        ``enabled`` is 0 (released) or non-zero/65535 (pressed).

    For analog steer axis:
        ``event_name`` is "Steer" (or "_FakePad_Steer").
        ``enabled`` is a uint16: 0=full-left, 32768=center, 65535=full-right.

    We accumulate key state across events and sample at each step.
    """
    if control_names:
        log.info("  Control names in replay: %s", control_names)

    # _FakeDontInverseAxis: when present, cancels the default sign-inversion
    # applied by _event_to_analog_value().  Without it the default convention
    # is: _event_to_analog_value returns ``-raw``, so positive raw → negative
    # output.  With the flag we negate again → back to raw sign.
    cancel_default_steer_inversion = any(
        getattr(ce, "event_name", "") == "_FakeDontInverseAxis"
        for ce in control_entries
    )
    if cancel_default_steer_inversion:
        log.info("  Replay has _FakeDontInverseAxis: cancelling default steer inversion")

    # --- collect and sort events ---
    events: list[tuple[int, str, bool]] = []
    unknown_names: set[str] = set()

    for ce in control_entries:
        t_ce = getattr(ce, "time", getattr(ce, "Time", None))
        if t_ce is None:
            continue

        event_name_raw = getattr(ce, "event_name", getattr(ce, "EventName", ""))
        if not event_name_raw:
            continue
        event_name = event_name_raw.lower().strip()

        # Optional global timing offset for replay events (diagnostics/tuning):
        # -10 means "apply inputs 1 tick earlier", +10 means "1 tick later".
        t_ev = t_ce + input_time_offset_ms

        # Binary key events (Accelerate, Brake, SteerLeft, SteerRight) — TMNF keyboard
        key = _BINARY_EVENT_TO_KEY.get(event_name)
        if key is not None:
            # Pressed = value != 0 (same as visualizer; handles flags+enabled combo)
            pressed = _event_to_analog_value(ce) != 0
            events.append((t_ev, key, pressed))
            continue

        # Analog steer axis → convert to binary left/right
        # Convention (TM-Gbx-input-visualizer): value > 0 = right, value < 0 = left
        if event_name in _ANALOG_STEER_NAMES:
            analog_val = _event_to_analog_value(ce)
            if cancel_default_steer_inversion:
                analog_val = -analog_val
            if analog_val > _STEER_DEADZONE:
                events.append((t_ev, "left", False))
                events.append((t_ev, "right", True))
            elif analog_val < -_STEER_DEADZONE:
                events.append((t_ev, "left", True))
                events.append((t_ev, "right", False))
            else:
                events.append((t_ev, "left", False))
                events.append((t_ev, "right", False))
            continue

        # Unknown — collect for diagnostics (skip _Fake* internal names silently)
        if not event_name.startswith("_fake"):
            unknown_names.add(event_name_raw)

    if unknown_names:
        log.info("  Unknown control names (ignored): %s", unknown_names)

    events.sort(key=lambda x: x[0])

    if events:
        t_min = events[0][0]
        t_max = events[-1][0]
        log.info(
            "  Parsed %d input events, time range [%d .. %d] ms, step_ms=%d",
            len(events), t_min, t_max, step_ms,
        )
        for j, (t_ev, key, enabled) in enumerate(events[:_DEBUG_EVENTS_TO_LOG]):
            log.info("  Event %02d: t=%dms key=%s state=%s", j, t_ev, key, int(enabled))

    # --- sample key state at each step ---
    n_steps = int(race_time_ms / step_ms) + 1
    key_state: dict[str, bool] = {
        "accelerate": False, "brake": False, "left": False, "right": False,
    }
    event_idx = 0
    out: list[dict[str, bool]] = []
    n_left_right_conflicts = 0

    for i in range(n_steps):
        t = i * step_ms
        # advance through all events up to and including this step time
        while event_idx < len(events) and events[event_idx][0] <= t:
            _, key, enabled = events[event_idx]
            key_state[key] = enabled
            event_idx += 1

        # Sanitize impossible state: left+right pressed simultaneously.
        # This can happen with keyboard replays if both SteerLeft and
        # SteerRight events overlap.  In the real game both cancel out,
        # so we set both to False to avoid an unmapped action combination.
        if key_state["left"] and key_state["right"]:
            n_left_right_conflicts += 1
            snap = dict(key_state)
            snap["left"] = False
            snap["right"] = False
            out.append(snap)
        else:
            out.append(dict(key_state))

    if n_left_right_conflicts:
        log.warning(
            "  %d steps had left+right pressed simultaneously — both cancelled to False",
            n_left_right_conflicts,
        )

    return out


def get_challenge_map_uid(challenge_path: Path) -> str | None:
    """Read map_uid from .Challenge.Gbx. Returns None if unavailable."""
    try:
        from pygbx import Gbx, GbxType
    except ImportError:
        return None
    try:
        gbx = Gbx(str(challenge_path))
        ch = gbx.get_class_by_id(GbxType.CHALLENGE)
        if ch is None:
            return None
        uid = getattr(ch, "map_uid", getattr(ch, "mapUid", None))
        return str(uid) if uid else None
    except Exception:
        return None


def get_sample_period_from_replay(replay_path: Path) -> int:
    """
    Read sample_period from .replay.gbx (game tick in ms). Default 10 if unavailable.
    """
    try:
        from pygbx import Gbx, GbxType
    except ImportError:
        return 10
    import logging as _logmod
    _saved = []
    for _name in ("pygbx", "pygbx.gbx"):
        _log = _logmod.getLogger(_name)
        _saved.append((_log, _log.level))
        _log.setLevel(_logmod.CRITICAL)
    try:
        gbx = Gbx(str(replay_path))
        ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
        if not ghosts:
            return 10
        ghost = min(ghosts, key=lambda g: getattr(g, "cp_times", [0])[-1] if getattr(g, "cp_times", None) else 0)
        return int(getattr(ghost, "sample_period", 10) or 10)
    except Exception:
        return 10
    finally:
        for _log, _lev in _saved:
            _log.setLevel(_lev)


def extract_inputs_from_replay_gbx(
    replay_path: Path,
    step_ms: int | None = None,
    input_time_offset_ms: int = 0,
) -> tuple[list[dict[str, bool]], int, dict] | None:
    """
    Extract inputs from .replay.gbx using pygbx. Returns (inputs_per_step, race_time_ms, metadata) or None.
    
    **TMInterface Prerequisite** (from official docs):
    The replay MUST finish the race. Incomplete replays (did not reach finish line)
    do not contain input data in the replay file, and cannot be extracted/validated.
    
    metadata: {ghost_uid, game_version, race_finished} for Map UID check and diagnostics.
    When step_ms is None, defaults to 10 ms (TMNF physics tick).
    """
    try:
        from pygbx import Gbx, GbxType
    except ImportError:
        log.warning("pygbx not available; cannot extract inputs from .replay.gbx")
        return None

    import logging as _logmod
    _saved = []
    for _name in ("pygbx", "pygbx.gbx"):
        _log = _logmod.getLogger(_name)
        _saved.append((_log, _log.level))
        _log.setLevel(_logmod.CRITICAL)
    try:
        gbx = Gbx(str(replay_path))
        ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
        if not ghosts:
            return None
        ghost = min(ghosts, key=lambda g: getattr(g, "cp_times", [0])[-1] if getattr(g, "cp_times", None) else 0)
        race_time = int(getattr(ghost, "race_time", 0) or 0)
        sample_period = int(getattr(ghost, "sample_period", 10) or 10)
        if step_ms is None:
            step_ms = 10
        control_entries = list(getattr(ghost, "control_entries", []) or [])
        control_names = list(getattr(ghost, "control_names", []) or [])
        cp_times = list(getattr(ghost, "cp_times", []) or [])
    except Exception as e:
        log.debug("Failed to parse replay %s: %s", replay_path, e)
        return None
    finally:
        for _log, _lev in _saved:
            _log.setLevel(_lev)

    # Validate replay is complete (TMInterface requirement)
    if race_time <= 0:
        log.debug("Replay %s: race_time=%d <= 0 (incomplete replay)", replay_path.name, race_time)
        return None
    
    # Check if replay finished (has checkpoint times and inputs)
    # Incomplete replays don't contain control_entries
    if not control_entries:
        log.warning(
            "Replay %s: no control_entries found (race_time=%dms). "
            "This is an incomplete replay that did not finish the race. "
            "TMInterface requires replays to finish for input extraction.",
            replay_path.name, race_time,
        )
        return None

    # Diagnostics: show control entry format for first replay
    if control_entries:
        ce0 = control_entries[0]
        log.info(
            "Replay %s: %d control_entries, race_time=%d ms. "
            "First entry attrs: time=%s event_name=%s enabled=%s flags=%s",
            replay_path.name,
            len(control_entries),
            race_time,
            getattr(ce0, "time", "?"),
            getattr(ce0, "event_name", "?"),
            getattr(ce0, "enabled", "?"),
            getattr(ce0, "flags", "?"),
        )
    else:
        log.info("Replay %s: 0 control_entries, race_time=%d ms", replay_path.name, race_time)

    # Build inputs from events
    if control_entries:
        inputs = _sample_inputs_at_steps(
            race_time, sample_period, control_entries, step_ms,
            input_time_offset_ms=input_time_offset_ms,
            control_names=control_names,
        )
        # Diagnostics: count how many steps have at least one active key
        n_active = sum(1 for inp in inputs if any(inp.values()))
        log.info(
            "  -> %d steps total (step_ms=%d), %d with active inputs (%.0f%%)",
            len(inputs), step_ms, n_active, 100 * n_active / max(len(inputs), 1),
        )
        # Log first 5 steps for verification (TMNF: accelerate usually from step 0)
        for i in range(min(5, len(inputs))):
            log.info("  Step %d (t=%dms): %s", i, i * step_ms, inputs[i])
    else:
        n_steps = int(race_time / step_ms) + 1
        inputs = [{"accelerate": False, "brake": False, "left": False, "right": False} for _ in range(n_steps)]

    metadata: dict[str, Any] = {}
    uid_val = getattr(ghost, "uid", getattr(ghost, "UID", None))
    if uid_val is not None:
        metadata["ghost_uid"] = str(uid_val)
    gv = getattr(ghost, "game_version", getattr(ghost, "GameVersion", None))
    if gv is not None:
        metadata["game_version"] = str(gv)
    metadata["race_finished"] = bool(cp_times) and race_time > 0
    metadata["n_checkpoints"] = len(cp_times)
    return (inputs, race_time, metadata)


# ---------- 3. Track ID -> challenge path ----------

def get_challenge_path_for_track(
    track_id: str,
    replay_path: Path,
    tracks_dir: Path,
    extract_challenge_from_replay: Any,
) -> Path | None:
    """
    Return path to .Challenge.Gbx for *this specific* track.

    Search order:
      1. tracks_dir/<track_id>/*.Challenge.Gbx  — any file in the track's own folder
      2. tracks_dir/<track_id>.Challenge.Gbx     — exact-name match in parent (pipeline layout)
      3. tracks_dir/*<track_id>*.Challenge.Gbx   — fuzzy match by track_id in parent
      4. Extract from replay into tracks_dir/<track_id>/<track_id>.Challenge.Gbx

    IMPORTANT: Never fall back to an arbitrary *.Challenge.Gbx in the parent
    directory — that would load a *different* track's map.
    """
    tid = str(track_id)

    # 1. Track's own subdirectory — any challenge file is fine
    track_subdir = tracks_dir / tid
    if track_subdir.is_dir():
        for p in track_subdir.glob("*.Challenge.Gbx"):
            return p
        for p in track_subdir.glob("*.challenge.gbx"):
            return p

    # 2. Exact name in parent: <track_id>.Challenge.Gbx
    exact = tracks_dir / f"{tid}.Challenge.Gbx"
    if exact.is_file():
        return exact
    exact_lower = tracks_dir / f"{tid}.challenge.gbx"
    if exact_lower.is_file():
        return exact_lower

    # 3. Fuzzy match in parent: filename contains the track_id
    if tracks_dir.is_dir():
        for p in tracks_dir.glob(f"*{tid}*.Challenge.Gbx"):
            return p
        for p in tracks_dir.glob(f"*{tid}*.challenge.gbx"):
            return p

    # 4. Extract from replay
    out_dir = tracks_dir / tid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tid}.Challenge.Gbx"
    if extract_challenge_from_replay(replay_path, out_path):
        return out_path
    return None


# ---------- 4. Replay policy & helpers for direct rollout() reuse ----------


def _make_replay_policy(
    inputs_per_step: list[dict[str, bool]],
    step_ms: int,
):
    """
    Build an exploration_policy callback that replays pre-extracted actions.

    Signature matches what rollout() expects (same as inferer.get_exploration_action
    in the training pipeline)::

        policy(frame, floats) -> (action_idx, action_was_greedy, q_value, q_values)

    **Timing contract** (verified by deep analysis of rollout()):

    rollout() calls ``on_step(T)`` at every game tick ``T = 0, 10, 20, …``
    (when ``run_steps_per_action=1``).  Inside ``on_step(T)`` it pauses the
    game, captures a frame, and calls ``exploration_policy(frame, floats)``.
    The returned action is applied via ``set_input_state(…)`` **before** the
    game computes the physics tick ``T → T+10``.

    Therefore policy-step *k* corresponds to game-time ``k * step_ms`` and the
    input ``inputs_per_step[k]`` is active during the tick
    ``k*step_ms → (k+1)*step_ms``.

    This matches the replay event semantics: ``Accelerate time=10`` means
    "accelerate key is pressed starting at T=10", affecting tick 10→20.
    Our ``_sample_inputs_at_steps`` processes events with ``time <= T`` (inclusive),
    so ``inputs_per_step[1]`` (T=10) includes the Accelerate event.  ✓
    """
    # Reuses action_dict_to_index imported from capture_frames_from_replays
    action_indices = [action_dict_to_index(inp) for inp in inputs_per_step]
    n_actions = len(get_config().inputs)
    dummy_q = np.zeros(n_actions, dtype=np.float32)
    step = [0]
    max_step = len(action_indices) - 1 if action_indices else 0
    clamped_warned = [False]

    log.info(
        "  Replay policy: %d steps available (step_ms=%d, covers 0..%d ms)",
        len(action_indices), step_ms, max_step * step_ms,
    )

    def policy(frame, floats):
        k = step[0]
        i = min(k, max_step)
        act = action_indices[i] if action_indices else get_config().action_forward_idx

        # Detect when rollout exceeds the replay's input range (clamping to last input).
        # This happens when our car is slightly slower than the original replay
        # (e.g. due to rewind_to_current_state drift) and hasn't finished yet.
        if k > max_step and not clamped_warned[0]:
            clamped_warned[0] = True
            log.warning(
                "  Policy step=%d exceeds available inputs (%d). "
                "Clamping to last input (idx=%d, action=%d). "
                "Car may be drifting from original replay trajectory.",
                k, len(action_indices), i, act,
            )

        if k < _DEBUG_POLICY_ACTIONS_TO_LOG:
            inp = inputs_per_step[i] if inputs_per_step else {}
            game_time_ms = k * step_ms
            log.info(
                "  Policy step=%d  game_time=%dms  input_src_idx=%d  action_idx=%d  inputs=%s",
                k, game_time_ms, i, act, inp,
            )
        step[0] += 1
        return act, True, 0.0, dummy_q

    return policy


def _make_dummy_zone_centers() -> np.ndarray:
    """
    Create dummy zone centers for rollout() when progress tracking is not needed.

    Returns a straight line array (N, 3) with enough points for all internal
    indexing in rollout() and the numba update_current_zone_idx function.
    The car will never match these virtual checkpoints, so the zone index stays
    constant — which is fine because the replay policy ignores floats entirely.
    """
    cfg = get_config()
    n = max(
        3000,
        cfg.n_zone_centers_extrapolate_before_start_of_map
        + cfg.one_every_n_zone_centers_in_inputs * cfg.n_zone_centers_in_inputs
        + cfg.n_zone_centers_extrapolate_after_end_of_map
        + 200,
    )
    return np.column_stack([
        np.zeros(n),
        np.full(n, 500.0),
        np.arange(n, dtype=np.float64) * 10.0,
    ])


# ---------- 5. Save captured frames from rollout results ----------


def _save_rollout_frames(
    rollout_results: dict,
    out_dir: Path,
    track_id: str,
    replay_name: str,
    challenge_name: str,
    race_time_ms: int,
    fps: float,
    width: int,
    height: int,
    step_ms: int,
    run_steps_per_action: int,
    inputs_per_step: list[dict[str, bool]],
    per_frame_json: bool = False,
) -> int:
    """Save captured frames from a rollout() result to *out_dir*, respecting *fps*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_interval_ms = 1000.0 / fps if fps > 0 else float(step_ms)
    next_capture_time_ms = 0.0
    manifest_entries: list[dict[str, Any]] = []
    frame_count = 0

    for i, frame in enumerate(rollout_results["frames"]):
        if not isinstance(frame, np.ndarray):
            continue  # terminal nan appended by rollout on race finish
        time_ms = i * step_ms
        if time_ms < next_capture_time_ms:
            continue
        fname = f"frame_{frame_count:05d}_{time_ms}ms.jpeg"
        # rollout() stores frames as (1, H, W) grayscale uint8
        cv2.imwrite(str(out_dir / fname), frame[0], [cv2.IMWRITE_JPEG_QUALITY, 95])

        inp_idx = min(i, len(inputs_per_step) - 1) if inputs_per_step else 0
        action_raw = rollout_results["actions"][i] if i < len(rollout_results["actions"]) else None
        entry = {
            "file": fname,
            "step": frame_count,
            "time_ms": time_ms,
            "inputs": inputs_per_step[inp_idx] if inputs_per_step else {},
            "action_idx": (
                int(action_raw)
                if action_raw is not None
                and not (isinstance(action_raw, float) and np.isnan(action_raw))
                else None
            ),
            "capture_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_entries.append(entry)
        if per_frame_json:
            (out_dir / f"frame_{frame_count:05d}_{time_ms}ms.json").write_text(
                json.dumps(entry, indent=2), encoding="utf-8",
            )
        frame_count += 1
        next_capture_time_ms += capture_interval_ms

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_entries, indent=2), encoding="utf-8",
    )
    race_finished = rollout_results.get("race_time") is not None
    metadata: dict[str, Any] = {
        "track_id": track_id,
        "replay_name": replay_name,
        "challenge_name": challenge_name,
        "fps": fps,
        "width": width,
        "height": height,
        "step_ms": step_ms,
        "run_steps_per_action": run_steps_per_action,
        "race_time_ms": race_time_ms,
        "total_frames": frame_count,
        "race_finished": race_finished,
    }
    if race_finished:
        metadata["actual_race_time_ms"] = rollout_results["race_time"]
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    return frame_count


# ---------- 6. Worker: replay via gim.rollout() ----------


def rollout_with_tmi_script(
    gim,
    script_path: Path,
    map_name: str,
    race_time_ms: int,
    width: int,
    height: int,
    fps: float,
    step_ms: int,
) -> dict[str, Any] | None:
    """
    Replay using TMInterface native script loading (deterministic, no drift).
    
    Instead of manually calling set_input_state every tick (which requires
    rewind_to_current_state and introduces floating-point drift), we:
      1. Load inputs via execute_command("load script.txt")
      2. Start the race
      3. TMInterface natively injects inputs (fully deterministic!)
      4. We only capture frames via on_step + request_frame
    
    Returns rollout_results dict with frames, times, or None on error.
    """
    from trackmania_rl.tmi_interaction.tminterface2 import MessageType
    
    # Ensure game is running and connected
    gim.ensure_game_launched()
    if gim.iface is None or not gim.iface.registered:
        log.info("Connecting to TMInterface...")
        from trackmania_rl.tmi_interaction.tminterface2 import TMInterface
        gim.iface = TMInterface(gim.tmi_port)
        import time
        start = time.perf_counter()
        while True:
            try:
                gim.iface.register(get_config().tmi_protection_timeout_s)
                break
            except ConnectionRefusedError:
                if time.perf_counter() - start > 30:
                    log.error("Cannot connect to TMInterface")
                    return None
                time.sleep(0.5)
    
    # Prepare frame capture state
    capture_interval_ms = 1000.0 / fps if fps > 0 else float(step_ms)
    next_capture_time_ms = 0.0
    frames = []
    frame_times_ms = []
    frame_expected = False
    race_started = False
    race_finished = False
    race_start_requested = False
    current_time = -3000
    timeout_start = None
    import time as time_module
    import socket
    
    try:
        while not race_finished:
            msgtype = gim.iface._read_int32()
            
            if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                # Initial connection - configure TMInterface
                gim.iface.execute_command(f"set countdown_speed {gim.running_speed}")
                gim.iface.execute_command("set use_valseed false")
                gim.iface.execute_command("set skip_map_load_screens true")
                gim.iface.execute_command("set unfocused_fps_limit false")
                gim.iface.execute_command(f"cam {get_config().game_camera_number}")
                gim.iface.set_on_step_period(step_ms)
                
                # CRITICAL: Load script BEFORE entering map
                # TMInterface docs: script must be loaded before starting race
                # Also verify execute_commands is enabled
                log.info("  Loading TMI script: %s", script_path.name)
                gim.iface.execute_command("set execute_commands true")
                gim.iface.execute_command(f"load {script_path.name}")
                time_module.sleep(0.3)
                
                # Load map (script already loaded, will auto-inject)
                log.info("  Loading map: %s", map_name)
                gim.iface.execute_command(f"map {map_name}")
                
                gim.iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                current_time = gim.iface._read_int32()
                
                # Start race (give_up restarts countdown)
                if current_time <= -2990 and not race_start_requested:
                    log.info("  Starting race (give_up)...")
                    gim.iface.give_up()
                    race_start_requested = True
                
                if not race_started and current_time >= 0:
                    race_started = True
                    timeout_start = time_module.perf_counter()
                    log.info("  Race started (t=0), capturing frames...")
                
                # Capture frames at specified FPS
                if race_started and current_time >= next_capture_time_ms:
                    if not frame_expected:
                        gim.iface.request_frame(width, height)
                        frame_expected = True
                        next_capture_time_ms += capture_interval_ms
                
                # Timeout protection
                if race_started and timeout_start:
                    if time_module.perf_counter() - timeout_start > (race_time_ms / 1000.0) + 30:
                        log.warning("  Timeout after %.1fs - race did not finish", time_module.perf_counter() - timeout_start)
                        break
                
                gim.iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                frame = gim.iface.get_frame(width, height)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                frames.append(np.expand_dims(frame_gray, 0))
                frame_times_ms.append(current_time)
                frame_expected = False
                gim.iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                current_cp = gim.iface._read_int32()
                target_cp = gim.iface._read_int32()
                if current_cp == target_cp:
                    log.info("  Race finished at t=%dms!", current_time)
                    race_finished = True
                gim.iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                gim.iface._read_int32()
                gim.iface._read_int32()
                gim.iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.C_SHUTDOWN):
                gim.iface.close()
                return None
            else:
                pass
                
    except socket.timeout:
        log.warning("  TMInterface timeout")
        return None
    except Exception as e:
        log.exception("  Rollout failed: %s", e)
        return None
    finally:
        # Unload script
        try:
            gim.iface.execute_command("unload")
        except:
            pass
    
    return {
        "frames": frames,
        "frame_times_ms": frame_times_ms,
        "race_finished": race_finished,
        "final_time_ms": current_time,
    }


def worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    game_spawning_lock: Any,
    base_tmi_port: int,
    replays_dir: Path,
    output_dir: Path,
    tracks_dir: Path,
    width: int,
    height: int,
    fps: float,
    run_steps_per_action: int,
    step_ms: int,
    input_time_offset_ms: int,
    running_speed: int,
    config_path: Path,
    per_frame_json: bool = False,
) -> None:
    """Single worker: convert .replay.gbx → TMI script, load via TMInterface, capture frames."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    set_config(load_config(config_path))
    cfg = get_config()

    log.info(
        "Worker %d: step_ms=%d, running_speed=%d, capture_fps=%.1f",
        worker_id, step_ms, running_speed, fps,
    )

    cfg.W_downsized = width
    cfg.H_downsized = height

    from trackmania_rl.tmi_interaction import game_instance_manager
    from replays_tmnf.api import extract_challenge_from_replay

    tmi_port = base_tmi_port + worker_id
    
    # Create scripts directory for TMInterface
    # TMInterface looks for scripts in Documents\TMInterface\Scripts, not in game folder
    import os
    user_docs = Path(os.path.expanduser("~")) / "Documents"
    scripts_dir = user_docs / "TMInterface" / "Scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    log.info("Worker %d: TMI scripts directory: %s", worker_id, scripts_dir)

    gim = game_instance_manager.GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        running_speed=running_speed,
        run_steps_per_action=run_steps_per_action,
        max_overall_duration_ms=600_000,
        max_minirace_duration_ms=600_000,
        tmi_port=tmi_port,
    )

    try:
        while True:
            try:
                task = task_queue.get(timeout=2.0)
            except queue.Empty:
                continue
            if task is None:
                break

            track_id, replay_paths = task
            for replay_path in replay_paths:
                replay_name = replay_path.stem
                out_dir = output_dir / str(track_id) / replay_name
                if (out_dir / "manifest.json").exists():
                    log.info("Skip (already done): %s / %s", track_id, replay_name)
                    continue

                # 1. Convert replay to TMInterface script format
                script_path = scripts_dir / f"replay_{track_id}_{replay_name}.txt"
                replay_metadata = convert_replay_to_tmi_script(replay_path, script_path)
                if replay_metadata is None:
                    log.warning("Skip (conversion failed): %s", replay_path)
                    continue
                
                race_time_ms = replay_metadata["race_time_ms"]
                if replay_metadata.get("game_version"):
                    log.info("  Game version: %s", replay_metadata["game_version"])

                # 2. Ensure challenge is in the game folder
                challenge_path = get_challenge_path_for_track(
                    track_id, replay_path, tracks_dir, extract_challenge_from_replay,
                )
                # Byte-level verification
                forced_out_dir = tracks_dir / str(track_id)
                forced_out_dir.mkdir(parents=True, exist_ok=True)
                embedded_challenge = forced_out_dir / f"{track_id}__from_replay.Challenge.Gbx"
                embedded_ok = extract_challenge_from_replay(replay_path, embedded_challenge)

                if challenge_path is None and not embedded_ok:
                    log.warning("Skip (no challenge): %s", replay_path)
                    continue
                if challenge_path is None and embedded_ok:
                    challenge_path = embedded_challenge
                elif challenge_path is not None and embedded_ok:
                    try:
                        local_bytes = challenge_path.read_bytes()
                        embedded_bytes = embedded_challenge.read_bytes()
                        if local_bytes != embedded_bytes:
                            log.warning("Map bytes differ - using embedded")
                            challenge_path = embedded_challenge
                        else:
                            log.info("  Map bytes match")
                    except Exception:
                        challenge_path = embedded_challenge

                # Copy challenge to game directory
                game_challenges = Path(cfg.trackmania_base_path) / "Tracks" / "Challenges"
                game_challenges.mkdir(parents=True, exist_ok=True)
                dest_challenge = game_challenges / challenge_path.name
                if dest_challenge != challenge_path.resolve():
                    import shutil
                    shutil.copy2(challenge_path, dest_challenge)

                # 3. Run replay with TMInterface native input loading (DETERMINISTIC!)
                try:
                    rollout_results = rollout_with_tmi_script(
                        gim=gim,
                        script_path=script_path,
                        map_name=dest_challenge.name,
                        race_time_ms=race_time_ms,
                        width=width,
                        height=height,
                        fps=fps,
                        step_ms=step_ms,
                    )
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError) as e:
                    log.warning("TMI connection broken: %s", e)
                    if gim.iface is not None:
                        try:
                            gim.iface.close()
                        except:
                            pass
                    gim.iface = None
                    gim.last_rollout_crashed = True
                    continue
                except Exception as e:
                    log.exception("Rollout failed: %s", e)
                    if gim.iface is not None:
                        try:
                            gim.iface.close()
                        except:
                            pass
                    gim.iface = None
                    gim.last_rollout_crashed = True
                    continue

                if rollout_results is None:
                    log.warning("Replay %s: rollout returned None", replay_path)
                    continue

                # 4. Save frames
                out_dir.mkdir(parents=True, exist_ok=True)
                manifest_entries = []
                for i, (frame, t_ms) in enumerate(zip(rollout_results["frames"], rollout_results["frame_times_ms"])):
                    fname = f"frame_{i:05d}_{t_ms}ms.jpeg"
                    cv2.imwrite(str(out_dir / fname), frame[0], [cv2.IMWRITE_JPEG_QUALITY, 95])
                    entry = {
                        "file": fname,
                        "step": i,
                        "time_ms": t_ms,
                        "capture_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    manifest_entries.append(entry)
                    if per_frame_json:
                        (out_dir / f"frame_{i:05d}_{t_ms}ms.json").write_text(
                            json.dumps(entry, indent=2), encoding="utf-8",
                        )
                
                (out_dir / "manifest.json").write_text(
                    json.dumps(manifest_entries, indent=2), encoding="utf-8",
                )
                
                metadata = {
                    "track_id": track_id,
                    "replay_name": replay_name,
                    "challenge_name": dest_challenge.name,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "step_ms": step_ms,
                    "race_time_ms": race_time_ms,
                    "total_frames": len(rollout_results["frames"]),
                    "race_finished": rollout_results["race_finished"],
                    "tmi_script_used": script_path.name,
                }
                if rollout_results["race_finished"]:
                    metadata["actual_race_time_ms"] = rollout_results["final_time_ms"]
                
                (out_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8",
                )
                
                log.info(
                    "Captured %s / %s: %d frames, finished=%s",
                    track_id, replay_name, len(rollout_results["frames"]), rollout_results["race_finished"],
                )
    except KeyboardInterrupt:
        log.info("Worker %d: interrupted", worker_id)
    finally:
        if gim.iface is not None:
            try:
                gim.iface.close()
            except Exception:
                pass


def _kill_tm_instances() -> None:
    """Kill all TmForever processes (same as train.py → clear_tm_instances)."""
    try:
        cfg = get_config()
        if cfg.is_linux:
            os.system("pkill -9 TmForever.exe 2>/dev/null || true")
        else:
            os.system("taskkill /F /IM TmForever.exe 2>nul")
    except Exception:
        pass


def _signal_handler(sig, frame):
    """Ctrl+C handler — identical pattern to train.py signal_handler."""
    print("\nReceived SIGINT. Killing all TM instances and workers.")
    _kill_tm_instances()
    for child in mp.active_children():
        child.kill()
    sys.exit(0)


# ---------- Main: CLI, queue, workers ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Capture frames from TMNF replays (maps/replays) via TMI.")
    parser.add_argument("--replays-dir", type=Path, default=Path("maps/replays"), help="Directory with track_id/*.replay.gbx")
    parser.add_argument("--output-dir", type=Path, default=Path("capture_frames_out"), help="Output root: output_dir/track_id/replay_name/")
    parser.add_argument("--tracks-dir", type=Path, default=Path("maps/tracks"), help="Extracted challenges or place to extract")
    parser.add_argument("--width", type=int, default=None, help="Frame width (default: config w_downsized)")
    parser.add_argument("--height", type=int, default=None, help="Frame height (default: config h_downsized)")
    parser.add_argument("--fps", type=float, default=10.0, help="Capture FPS (interval in sim time)")
    parser.add_argument("--workers", type=int, default=1, help="Number of game instances (TMI ports)")
    parser.add_argument("--base-tmi-port", type=int, default=None, help="Base TMI port (default: from config)")
    parser.add_argument("--track-ids", type=Path, default=Path("maps/track_ids.txt"), help="File with track IDs (one per line, ordered by popularity). Default: maps/track_ids.txt")
    parser.add_argument("--track-id", type=str, default=None, help="Single track ID to process")
    parser.add_argument("--max-replays-per-track", type=int, default=10, help="Max replays per track (top N by filename: pos1, pos2, ...). Default: 10")
    parser.add_argument("--per-frame-json", action="store_true", help="Write one JSON file per frame (same content as manifest entry) for quick single-frame lookup")
    parser.add_argument(
        "--step-ms",
        type=int,
        default=None,
        help="Action step in ms. Must be a positive multiple of 10 (TM physics tick). Default: 10 for TMNF replays.",
    )
    parser.add_argument(
        "--input-time-offset-ms",
        type=int,
        default=0,
        help="Global timing offset for replay input events in ms (e.g. -10, 0, +10). Must be multiple of 10.",
    )
    parser.add_argument("--running-speed", type=int, default=None, help="Override running_speed from config (e.g. 512 = 512x, 1 = real-time).")
    parser.add_argument("--config", type=Path, default=_default_yaml, help="Config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if _default_yaml.exists():
        set_config(load_config(args.config))
    cfg = get_config()

    base_tmi_port = args.base_tmi_port
    if base_tmi_port is None:
        base_tmi_port = cfg.base_tmi_port if _default_yaml.exists() else 8478
    width = args.width if args.width is not None else cfg.W_downsized
    height = args.height if args.height is not None else cfg.H_downsized

    tasks = collect_replay_tasks(
        args.replays_dir,
        args.track_ids,
        args.track_id,
        args.max_replays_per_track,
    )
    if not tasks:
        log.info("No replay tasks found under %s", args.replays_dir)
        return
    grouped = group_replay_tasks_by_track(tasks)
    log.info(
        "Filtered to %d replay tasks (%d tracks, max %d replays/track)",
        len(tasks), len(grouped), args.max_replays_per_track,
    )

    # step_ms and run_steps_per_action must match: step_ms = run_steps_per_action * 10
    if args.step_ms is not None:
        if args.step_ms <= 0 or args.step_ms % 10 != 0:
            raise ValueError(
                f"Invalid --step-ms={args.step_ms}. It must be a positive multiple of 10 "
                "(10, 20, 30, ...). Values like 3 or 3.5 cannot be represented by TMInterface timing."
            )
        step_ms = args.step_ms
        run_steps_per_action = step_ms // 10
        log.info("Using --step-ms %d (run_steps_per_action=%d) for input timing", step_ms, run_steps_per_action)
    else:
        step_ms = 10
        run_steps_per_action = 1
        log.info("Using step_ms=10 (default for TMNF replay)")

    if args.input_time_offset_ms % 10 != 0:
        raise ValueError(
            f"Invalid --input-time-offset-ms={args.input_time_offset_ms}. "
            "It must be a multiple of 10."
        )
    log.info("Using input_time_offset_ms=%d", args.input_time_offset_ms)

    running_speed = args.running_speed if args.running_speed is not None else cfg.running_speed
    if args.running_speed is not None:
        log.info("Using --running-speed %d (override)", running_speed)

    # Register Ctrl+C handler (same pattern as train.py)
    signal.signal(signal.SIGINT, _signal_handler)

    task_queue: mp.Queue = mp.Queue()
    for track_id, replay_paths in grouped:
        task_queue.put((track_id, replay_paths))
    for _ in range(args.workers):
        task_queue.put(None)  # sentinel: worker exits when it gets None

    manager = mp.Manager()
    game_spawning_lock = manager.Lock()

    procs = []
    for w in range(args.workers):
        p = mp.Process(
            target=worker_process,
            args=(
                w,
                task_queue,
                game_spawning_lock,
                base_tmi_port,
                args.replays_dir.resolve(),
                args.output_dir.resolve(),
                args.tracks_dir.resolve(),
                width,
                height,
                args.fps,
                run_steps_per_action,
                step_ms,
                args.input_time_offset_ms,
                running_speed,
                args.config.resolve(),
                getattr(args, "per_frame_json", False),
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    log.info("Done.")


if __name__ == "__main__":
    main()
