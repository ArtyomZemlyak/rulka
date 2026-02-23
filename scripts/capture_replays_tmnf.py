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
  # Custom track list (order = processing order). E.g. after filtering out respawn maps:
  python scripts/filter_track_ids_no_respawn.py -o maps/track_ids_no_respawn.txt
  python scripts/capture_replays_tmnf.py --track-ids maps/track_ids_no_respawn.txt --replays-dir maps/replays --output-dir out
  # Карты с превью/intro автоматически обрабатываются: disable_forced_camera + skip_map_load_screens.
  # Если гонка не стартовала за 3с (нет RUN_STEP), скрипт шлёт TMInterface give_up/press_delete
  # для рестарта (retry каждые 3с, всего до 25с). Флаг --press-enter-after-map-load убран.
  # --write-enter-maps собирает track_id карт, которые так и не стартовали.
  # --exclude-enter-maps чтобы пропустить эти карты при следующем запуске.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import queue
import signal
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_script_basename(replay_name: str) -> str:
    """Safe filename for TMInterface script: no quotes/spaces (breaks 'load' command)."""
    return re.sub(r"[\s'\"]+", "_", replay_name).strip("_") or "replay"

import cv2
import numpy as np

# Project root and config
_script_root = Path(__file__).resolve().parents[1]
_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_root))
sys.path.insert(0, str(_scripts_dir))

from config_files.config_loader import get_config, load_config, set_config

_default_yaml = _script_root / "config_files" / "config_default.yaml"
if _default_yaml.exists():
    set_config(load_config(_default_yaml))

# Reuse action mapping from capture_frames_from_replays (same directory)
try:
    from capture_frames_from_replays import action_dict_to_index
except ModuleNotFoundError:
    from importlib.util import spec_from_file_location, module_from_spec
    _spec = spec_from_file_location("capture_frames_from_replays", _scripts_dir / "capture_frames_from_replays.py")
    _mod = module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    action_dict_to_index = _mod.action_dict_to_index

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

    Each element is a single task for the queue; each worker gets whole tasks via
    task_queue.get(). So no two workers ever process the same track (no shared
    Autosaves/Challenges overwrite, no duplicate work).
    """
    from itertools import groupby
    grouped: list[tuple[str, list[Path]]] = []
    for track_id, group in groupby(tasks, key=lambda x: x[0]):
        replay_paths = [p for _, p in group]
        grouped.append((track_id, replay_paths))
    return grouped


# ---------- 2. Extract inputs from .replay.gbx (pygbx) ----------
#
# Conversion logic aligned with:
#   - gbxtools generate_input_file.py (donadigo): https://github.com/donadigo/gbxtools
#   - TM-Gbx-input-visualizer inputs.py: get_event_time, event_to_analog_value, _FakeDontInverseAxis
# TMInterface script format: https://donadigo.com/tminterface/script-syntax

# --- Control name mapping ---
# Binary (keyboard) events: each name maps to one input key.
# AccelerateReal/BrakeReal: only when flags==1 (see _should_skip_event).
_BINARY_EVENT_TO_KEY: dict[str, str] = {
    "accelerate": "accelerate",
    "acceleratereal": "accelerate",
    "gas": "accelerate",
    "up": "accelerate",
    "brake": "brake",
    "brakereal": "brake",
    "down": "brake",
    "steerleft": "left",
    "steer_left": "left",
    "steer left": "left",
    "left": "left",
    "steerright": "right",
    "steer_right": "right",
    "steer right": "right",
    "right": "right",
    # TMNF checkpoint respawn (ingame key: Backspace; TMInterface command: enter)
    "respawn": "respawn",
    "enter": "respawn",
    # Horn is intentionally not mapped (skipped, same as gbxtools)
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
    Compute signed analog value from ControlEntry.
    Matches gbxtools/TM-Gbx-input-visualizer: val = -sign_extend24((flags << 16) | enabled)
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
    input_time_offset_ms: int = 0,
    respawn_action: str = "enter",
    respawn_after_cp_ms: int = 0,
) -> dict[str, Any] | None:
    """
    Convert .replay.gbx to TMInterface script format (.txt).
    
    TMInterface script format (time in milliseconds):
        0 press up              # Start accelerating at t=0ms
        680 press right         # Steer right at t=680ms
        830 rel right           # Release right at t=830ms
        ...
    
    Returns metadata dict {race_time_ms, n_events, ghost_uid, game_version} or None on error.
    """
    try:
        from pygbx import Gbx, GbxType
    except ImportError:
        log.warning("pygbx not available")
        return None

    _UNBOUND_SENTINEL = 4294967295  # 0xFFFFFFFF, gbxtools: treat as key never released

    import logging as _logmod
    _saved = []
    for _name in ("pygbx", "pygbx.gbx"):
        _log = _logmod.getLogger(_name)
        _saved.append((_log, _log.level))
        _log.setLevel(_logmod.CRITICAL)
    
    try:
        gbx = Gbx(str(replay_path))
        ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
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

    # Replay dumped from TMInterface validate can carry a time offset in events.
    # Match gbxtools behavior and normalize when this marker is present.
    is_iface_timing = any(
        (
            str(getattr(ce, "event_name", getattr(ce, "EventName", ""))) == "_FakeIsRaceRunning"
            and int(getattr(ce, "time", getattr(ce, "Time", 0)) or 0) % 10 == 5
        )
        for ce in control_entries
    )
    iface_time_shift = 0xFFFF if is_iface_timing else 0

    def _event_name_raw(ce: Any) -> str:
        return str(getattr(ce, "event_name", getattr(ce, "EventName", "")) or "").strip()

    def _event_name_lc(ce: Any) -> str:
        return _event_name_raw(ce).lower()

    def _event_time_ms(ce: Any) -> int:
        t_raw = int(getattr(ce, "time", getattr(ce, "Time", 0)) or 0) - iface_time_shift
        name = _event_name_lc(ce)
        if name == "respawn":
            t = int(t_raw / 10) * 10
            if t_raw % 10 == 0:
                t -= 10
        else:
            t = int(t_raw / 10) * 10 - 10
        t += input_time_offset_ms
        return int(t / 10) * 10

    def _should_skip_event(ce: Any) -> bool:
        name = _event_name_lc(ce)
        enabled = int(getattr(ce, "enabled", getattr(ce, "Enabled", 0)) or 0)
        flags = int(getattr(ce, "flags", getattr(ce, "Flags", 0)) or 0)
        if name in ("acceleratereal", "brakereal"):
            return flags != 1
        if name in _ANALOG_STEER_NAMES or name == "gas":
            return False
        if name.startswith("_fake"):
            return True
        return enabled == 0

    def _find_event_end(from_index: int, target_name: str) -> Any | None:
        for j in range(from_index, len(control_entries)):
            if _event_name_lc(control_entries[j]) == target_name:
                return control_entries[j]
        return None

    # CP times from replay are used to avoid sending respawn too early.
    cp_times_ms = sorted(int(t) for t in cp_times if int(t) >= 0)
    respawn_shifted = 0

    # Convert events to TMInterface commands while preserving source order.
    commands: list[tuple[int, str]] = []  # (time_ms_for_log, command_line)

    for i, ce in enumerate(control_entries):
        if _should_skip_event(ce):
            continue

        event_name = _event_name_lc(ce)
        t_from = _event_time_ms(ce)

        # Analog events are immediate samples.
        if event_name in _ANALOG_STEER_NAMES:
            analog_val = _event_to_analog_value(ce)
            if cancel_default_steer_inversion:
                analog_val = -analog_val
            commands.append((max(0, t_from), f"{max(0, t_from)} steer {int(analog_val)}"))
            continue

        if event_name == "gas":
            analog_val = _event_to_analog_value(ce)
            if cancel_default_steer_inversion:
                analog_val = -analog_val
            commands.append((max(0, t_from), f"{max(0, t_from)} gas {int(analog_val)}"))
            continue

        # Binary keys are emitted as range commands (from-to press key), matching gbxtools.
        key = _BINARY_EVENT_TO_KEY.get(event_name)
        if key is None:
            continue
        tmi_key = {
            "accelerate": "up",
            "brake": "down",
            "left": "left",
            "right": "right",
            "respawn": respawn_action,
        }[key]

        to_event = _find_event_end(i + 1, event_name)
        is_unbound = to_event is None
        if to_event is not None:
            t_to = _event_time_ms(to_event)
            if t_to >= _UNBOUND_SENTINEL:
                is_unbound = True
        else:
            t_to = int((int(race_time) + input_time_offset_ms) / 10) * 10
            if t_to >= _UNBOUND_SENTINEL:
                is_unbound = True

        if t_from < 0:
            if t_to < 0 and not is_unbound:
                continue
            t_from = 0

        # In TMNF, respawn before CP registration effectively becomes restart.
        # When replay has CP metadata, keep respawn at least a little after
        # the latest reached checkpoint time.
        if key == "respawn" and respawn_after_cp_ms > 0 and cp_times_ms:
            latest_cp = next((cp for cp in reversed(cp_times_ms) if cp <= t_from), None)
            if latest_cp is not None:
                min_respawn_time = int((latest_cp + respawn_after_cp_ms) / 10) * 10
                if t_from < min_respawn_time:
                    delta = min_respawn_time - t_from
                    t_from = min_respawn_time
                    if not is_unbound:
                        t_to += delta
                    respawn_shifted += 1

        t_from = int(t_from / 10) * 10
        t_to = int(t_to / 10) * 10

        if is_unbound or t_to <= t_from:
            commands.append((max(0, t_from), f"{max(0, t_from)} press {tmi_key}"))
        else:
            commands.append((max(0, t_from), f"{max(0, t_from)}-{max(0, t_to)} press {tmi_key}"))
    
    # Write TMInterface script in integer-millisecond format.
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_txt_path.open("w", encoding="utf-8") as f:
        for _, cmd_line in commands:
            f.write(f"{cmd_line}\n")
    
    # Log first few commands for verification
    log.info(
        "  Converted %s → %s (%d commands, race_time=%.2fs)",
        replay_path.name, output_txt_path.name, len(commands), race_time / 1000.0,
    )
    if commands:
        log.info("  First 10 commands:")
        for i, (t_cmd_ms, cmd) in enumerate(commands[:10]):
            log.info("    %d: %d %s", i, t_cmd_ms, cmd)
    if respawn_shifted:
        log.info("  Shifted %d respawn events to be after checkpoint times", respawn_shifted)
    
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
        log.debug("  Control names in replay: %s", control_names)

    # _FakeDontInverseAxis: when present, cancels the default sign-inversion
    # applied by _event_to_analog_value().  Without it the default convention
    # is: _event_to_analog_value returns ``-raw``, so positive raw → negative
    # output.  With the flag we negate again → back to raw sign.
    cancel_default_steer_inversion = any(
        getattr(ce, "event_name", "") == "_FakeDontInverseAxis"
        for ce in control_entries
    )
    if cancel_default_steer_inversion:
        log.debug("  Replay has _FakeDontInverseAxis: cancelling default steer inversion")

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
        log.debug("  Unknown control names (ignored): %s", unknown_names)

    events.sort(key=lambda x: x[0])

    if events:
        t_min = events[0][0]
        t_max = events[-1][0]
        log.debug(
            "  Parsed %d input events, time range [%d .. %d] ms, step_ms=%d",
            len(events), t_min, t_max, step_ms,
        )
        for j, (t_ev, key, enabled) in enumerate(events[:_DEBUG_EVENTS_TO_LOG]):
            log.debug("  Event %02d: t=%dms key=%s state=%s", j, t_ev, key, int(enabled))

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
        ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
        if not ghosts:
            return 10
        ghost = min(ghosts, key=lambda g: getattr(g, "cp_times", [0])[-1] if getattr(g, "cp_times", None) else 0)
        return int(getattr(ghost, "sample_period", 10) or 10)
    except Exception:
        return 10
    finally:
        for _log, _lev in _saved:
            _log.setLevel(_lev)


def replay_has_respawn(replay_path: Path) -> bool:
    """
    Return True if the replay file contains any respawn (or enter) control events.
    Used with --exclude-respawn-maps to skip tracks that have respawn in any replay.
    """
    try:
        from pygbx import Gbx, GbxType
    except ImportError:
        return False
    import logging as _logmod
    _saved = []
    for _name in ("pygbx", "pygbx.gbx"):
        _log = _logmod.getLogger(_name)
        _saved.append((_log, _log.level))
        _log.setLevel(_logmod.CRITICAL)
    try:
        gbx = Gbx(str(replay_path))
        ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
        if not ghosts:
            return False
        ghost = min(ghosts, key=lambda g: getattr(g, "cp_times", [0])[-1] if getattr(g, "cp_times", None) else 0)
        control_entries = list(getattr(ghost, "control_entries", []) or [])
        for ce in control_entries:
            name = (str(getattr(ce, "event_name", getattr(ce, "EventName", ""))) or "").strip().lower()
            if name in ("respawn", "enter"):
                return True
        return False
    except Exception:
        return False
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
        ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
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

    # Diagnostics: show control entry format (debug only)
    if control_entries:
        ce0 = control_entries[0]
        log.debug(
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
        log.debug("Replay %s: 0 control_entries, race_time=%d ms", replay_path.name, race_time)

    # Build inputs from events
    if control_entries:
        inputs = _sample_inputs_at_steps(
            race_time, sample_period, control_entries, step_ms,
            input_time_offset_ms=input_time_offset_ms,
            control_names=control_names,
        )
        n_active = sum(1 for inp in inputs if any(inp.values()))
        log.debug(
            "  -> %d steps total (step_ms=%d), %d with active inputs (%.0f%%)",
            len(inputs), step_ms, n_active, 100 * n_active / max(len(inputs), 1),
        )
        for i in range(min(5, len(inputs))):
            log.debug("  Step %d (t=%dms): %s", i, i * step_ms, inputs[i])
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


def get_map_filename_for_task(
    task: tuple[str, list[Path]],
    tracks_dir: Path,
    extract_challenge_from_replay: Any,
) -> str | None:
    """Имя файла карты для первой реплеи задачи (то, что уйдёт в game Challenges)."""
    track_id, replay_paths = task
    if not replay_paths:
        return None
    replay_path = replay_paths[0]
    challenge_path = get_challenge_path_for_track(
        track_id, replay_path, tracks_dir, extract_challenge_from_replay,
    )
    forced_out_dir = tracks_dir / str(track_id)
    forced_out_dir.mkdir(parents=True, exist_ok=True)
    embedded_challenge = forced_out_dir / f"{track_id}__from_replay.Challenge.Gbx"
    embedded_ok = extract_challenge_from_replay(replay_path, embedded_challenge)
    if challenge_path is None and not embedded_ok:
        return None
    if challenge_path is None and embedded_ok:
        return embedded_challenge.name
    if challenge_path is not None and embedded_ok:
        try:
            if challenge_path.read_bytes() != embedded_challenge.read_bytes():
                return embedded_challenge.name
        except Exception:
            return embedded_challenge.name
    return challenge_path.name


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
    last_map_requested: list[str | None],
    next_map_name: str | None = None,
    next_script_path: Path | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """
    Replay using TMInterface native script loading (deterministic, no drift).
    
    Instead of manually calling set_input_state every tick (which requires
    rewind_to_current_state and introduces floating-point drift), we:
      1. Load inputs via execute_command("load script.txt")
      2. Start the race
      3. TMInterface natively injects inputs (fully deterministic!)
      4. We only capture frames via on_step + request_frame
    
    Returns (rollout_results dict with frames, times, or None on error, need_enter).
    need_enter is True when the run was aborted because the map did not start (e.g. "Press Enter to start").
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
                    return (None, False)
                time.sleep(0.5)
    
    # Prepare frame capture state. FPS = кадров в секунду гонки (simulation second), не в реальном времени.
    # Интервал в симе: 1000/fps мс (e.g. 64 fps → ~15.6 ms между кадрами).
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
    connect_seen_this_rollout = False
    console_hidden = False  # Флаг: консоль скрыта один раз
    max_cp_reached = 0  # Для диагностики: 0 чекпоинтов = машина не ехала
    # После смены карты TMInterface может ещё присылать run step с временем предыдущей гонки (51s).
    # Не считать время "нашим" пока не увидели countdown/старт новой карты (current_time <= 0).
    time_from_current_map = False
    # Финиш по таймингу: переключаемся до пересечения финиша, чтобы успеть до экрана медалей.
    FINISH_MARGIN_MS = -800  # минус = за N ms до race_time_ms делаем unload и переход на след. реплей/карту
    # Короткий таймаут сокета для retry-loop: при каждом таймауте шлём TMInterface команды (give_up, press delete)
    # чтобы скипнуть превью средствами самого движка, а не симуляцией клавиатуры.
    ENTER_SOCKET_TIMEOUT_S = 3.0
    # Полный таймаут после которого сдаёмся и пропускаем карту.
    ENTER_TOTAL_TIMEOUT_S = 25.0
    map_load_wall_time = 0.0
    need_enter = False
    import time as time_module
    import socket

    # Спам Enter в отдельном потоке: скипает превью на картах, где RUN_STEP не приходят во время превью.
    # Спам начинается через 1.5с после map (на картах без превью countdown уже начнётся и спам не отправится).
    # Останавливается при первом RUN_STEP с countdown (current_time <= 0).
    # После остановки спама — give_up() перезапускает гонку, чтобы TMI скрипт выполнился чисто.
    enter_spam_stop_event = threading.Event()
    enter_spam_thread = None

    def enter_spam_worker():
        """Spam Enter every 100ms until stopped by first RUN_STEP countdown.
        
        Starts with a 1.5s delay so maps without previews reach countdown
        before any Enter is sent (avoiding spurious keypresses during countdown).
        Maps WITH previews still get Enter after the delay.
        """
        from trackmania_rl.tmi_interaction import game_instance_manager
        log.info("  Enter spam: started (1.5s initial delay)")
        if enter_spam_stop_event.wait(1.5):
            log.info("  Enter spam: stopped during initial delay (0 sends)")
            return
        attempt = 0
        while not enter_spam_stop_event.is_set():
            attempt += 1
            game_instance_manager.send_enter_to_game_window(gim, log)
            if attempt <= 3 or attempt % 20 == 0:
                log.info("    Enter #%d", attempt)
            enter_spam_stop_event.wait(0.1)  # 100ms between sends, instant wake on stop
        log.info("  Enter spam: stopped after %d sends", attempt)

    default_socket_timeout_s = get_config().tmi_protection_timeout_s
    skip_retry_count = 0  # сколько раз пытались скипнуть превью через TMInterface команды
    try:
        while not race_finished:
            # Короткий таймаут сокета, когда ждём старт после map — если RUN_STEP перестали приходить, не висеть 500 сек.
            if map_load_wall_time > 0 and not race_started:
                gim.iface.set_socket_timeout(ENTER_SOCKET_TIMEOUT_S)
            else:
                gim.iface.set_socket_timeout(default_socket_timeout_s)

            try:
                msgtype = gim.iface._read_int32()
            except socket.timeout:
                # Сокет не получил данных за ENTER_SOCKET_TIMEOUT_S секунд.
                # Если ждём старт гонки — возможно карта показывает превью/intro.
                # Пробуем скипнуть средствами TMInterface (give_up, press delete) —
                # это работает на уровне движка, не зависит от клавиатуры.
                if map_load_wall_time > 0 and not race_started:
                    elapsed = time_module.perf_counter() - map_load_wall_time
                    if elapsed > ENTER_TOTAL_TIMEOUT_S:
                        log.warning("  Race did not start within %.0fs — skipping map", ENTER_TOTAL_TIMEOUT_S)
                        need_enter = True
                        try:
                            gim.iface.execute_command("unload")
                        except Exception:
                            pass
                        break
                    skip_retry_count += 1
                    log.info("  No messages for %.0fs (retry #%d), Enter spam running", elapsed, skip_retry_count)
                    continue  # retry _read_int32
                else:
                    # Timeout during race or other state — propagate as before
                    raise
            
            if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                connect_seen_this_rollout = True
                # skip_map_load_screens: пропускает "Press any key to continue" (docs: donadigo.com/tminterface/variables)
                gim.iface.execute_command("set skip_map_load_screens true")
                gim.iface.execute_command("set disable_forced_camera true")
                # Скрываем консоль TMInterface один раз при первом подключении
                if not console_hidden:
                    time_module.sleep(0.3)
                    from trackmania_rl.tmi_interaction import game_instance_manager
                    if game_instance_manager.send_tilde_to_game_window(gim, log):
                        log.info("  TMInterface console hidden")
                        console_hidden = True
                # Переменные и скрипт — каждый connect. Map шлём только если в меню (is_in_menus).
                gim.iface.execute_command(f"set countdown_speed {gim.running_speed}")
                gim.request_speed(gim.running_speed)  # скорость гонки (speed); countdown_speed — только обратный отсчёт
                try:
                    gim.iface.execute_command("set use_valseed false")
                except Exception:
                    pass
                gim.iface.execute_command("set unfocused_fps_limit false")
                gim.iface.execute_command(f"cam {get_config().game_camera_number}")
                gim.iface.set_on_step_period(step_ms)
                # Каждый connect — load скрипта реплея для текущей карты (важно для 2-й и далее карт).
                log.info("  Loading TMI script: %s", script_path.name)
                gim.iface.execute_command(f"load {script_path.name}")
                gim.iface.execute_command("set execute_commands true")
                if gim.iface.is_in_menus() and map_name != last_map_requested[0]:
                    last_map_requested[0] = map_name
                    race_start_requested = False  # give_up должен сработать для новой карты
                    max_cp_reached = 0
                    map_load_wall_time = time_module.perf_counter()
                    time_module.sleep(0.2)
                    log.info("  Loading map: %s", map_name)
                    # ВАЖНО: disable_forced_camera и skip_map_load_screens должны быть установлены ДО map команды
                    # (переустанавливаем на всякий случай, хотя они уже были в ON_CONNECT)
                    gim.iface.execute_command("set skip_map_load_screens true")
                    gim.iface.execute_command("set disable_forced_camera true")
                    gim.iface.execute_command(f"map {map_name}")
                    # Запускаем спам Enter (1.5с задержка, потом каждые 100мс)
                    if enter_spam_thread is None or not enter_spam_thread.is_alive():
                        enter_spam_stop_event.clear()
                        enter_spam_thread = threading.Thread(target=enter_spam_worker, daemon=True)
                        enter_spam_thread.start()
                elif map_name != last_map_requested[0]:
                    log.info("  Not in menus, will request map in run step: %s", map_name)
                gim.iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                current_time = gim.iface._read_int32()
                # Как только видим время countdown/старта (<= 0), считаем что приходят тики уже от текущей карты.
                if current_time <= 0 and not time_from_current_map:
                    # Первый RUN_STEP с countdown — СТОП спам немедленно
                    time_from_current_map = True
                    skip_retry_count = 0
                    if enter_spam_thread is not None and enter_spam_thread.is_alive():
                        enter_spam_stop_event.set()
                        log.info("  Spam stopped (countdown t=%d)", current_time)
                    # Активируем скрипт: reload + give_up перезапускает гонку.
                    # Без give_up TMInterface может не активировать скрипт для первого
                    # обратного отсчёта (скрипт загружен в ON_CONNECT, но команды могут
                    # не применяться до перезапуска). give_up() гарантирует чистый старт.
                    if not race_start_requested:
                        log.info("  Activating script (reload + give_up at t=%d)...", current_time)
                        gim.request_speed(gim.running_speed)
                        gim.iface.execute_command(f"load {script_path.name}")
                        gim.iface.execute_command("set execute_commands true")
                        gim.iface.give_up()
                        race_start_requested = True
                
                # По документации: load скрипта — до map. Карту шлём из run step (не из callback финиша).
                # Для второй карты: сначала load скрипта реплея этой карты, потом map.
                if map_name != last_map_requested[0]:
                    log.info("  Map changed, loading script then map: %s", map_name)
                    time_from_current_map = False  # после смены карты ждём снова countdown/старт
                    race_start_requested = False  # give_up должен сработать для новой карты
                    max_cp_reached = 0
                    map_load_wall_time = time_module.perf_counter()
                    # ВАЖНО: disable_forced_camera и skip_map_load_screens должны быть установлены ДО map команды
                    gim.iface.execute_command("set skip_map_load_screens true")
                    gim.iface.execute_command("set disable_forced_camera true")
                    gim.iface.execute_command(f"load {script_path.name}")
                    gim.iface.execute_command("set execute_commands true")
                    last_map_requested[0] = map_name
                    time_module.sleep(0.1)
                    gim.iface.execute_command(f"map {map_name}")
                    # Остановим старый спам и запустим новый
                    if enter_spam_thread is not None and enter_spam_thread.is_alive():
                        enter_spam_stop_event.set()
                        enter_spam_thread.join(timeout=0.5)
                    enter_spam_stop_event.clear()
                    enter_spam_thread = threading.Thread(target=enter_spam_worker, daemon=True)
                    enter_spam_thread.start()
                elif map_name == last_map_requested[0]:
                    # Карты с "Press Enter to start": гонка не стартует, current_time не доходит до 0. Таймаут → пропуск.
                    if not race_started and map_load_wall_time > 0 and (time_module.perf_counter() - map_load_wall_time) > ENTER_TOTAL_TIMEOUT_S:
                        log.warning("  Race did not start within %.0fs (e.g. 'Press Enter to start') — skipping", ENTER_TOTAL_TIMEOUT_S)
                        need_enter = True
                        try:
                            gim.iface.execute_command("unload")
                        except Exception:
                            pass
                        break
                    # Финиш по таймингу: только если время уже от текущей карты (не старый тик после смены карты).
                    if time_from_current_map and race_started and current_time >= race_time_ms + FINISH_MARGIN_MS:
                        log.info("  Race finished by time (t=%dms >= %dms), switching...", current_time, race_time_ms + FINISH_MARGIN_MS)
                        race_finished = True
                        try:
                            gim.iface.execute_command("unload")
                        except Exception:
                            pass
                        # Не шлём press enter: мы переключаемся до финиша (negative margin), медалей нет.
                        # Enter в момент гонки мог приводить к обрыву соединения TMInterface.
                    # give_up() fired at first countdown tick (see above) — not here.
                    # Old condition (current_time <= -2990) never triggered because the
                    # first RUN_STEP arrives at ~-2600 due to ON_CONNECT processing time.
                    
                    if not race_started and time_from_current_map and current_time >= 0:
                        race_started = True
                        timeout_start = time_module.perf_counter()
                        log.info("  Race started (t=0), capturing frames...")
                    
                    # Capture frames at specified FPS
                    if race_started and time_from_current_map and current_time >= next_capture_time_ms:
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
                if race_started and current_cp > max_cp_reached:
                    max_cp_reached = current_cp
                    log.info("  Checkpoint %d/%d at t=%dms", current_cp, target_cp, current_time)
                if current_cp == target_cp and race_started:
                    log.debug("  Checkpoint finish event at t=%dms (we use time-based finish)", current_time)
                gim.iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                gim.iface._read_int32()
                gim.iface._read_int32()
                gim.iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.C_SHUTDOWN):
                gim.iface.close()
                return (None, False)
            else:
                pass
                
    except socket.timeout:
        # Таймаут вне фазы ожидания старта — фатальная ошибка
        log.warning("  TMInterface socket timeout (not during map load wait)")
        return (None, False)
    except ConnectionError as e:
        log.warning("  TMInterface connection closed: %s", e)
        try:
            if gim.iface is not None:
                gim.iface.close()
        except Exception:
            pass
        gim.iface = None
        return (None, False)
    except Exception as e:
        log.exception("  Rollout failed: %s", e)
        return (None, False)
    finally:
        if enter_spam_thread is not None and enter_spam_thread.is_alive():
            enter_spam_stop_event.set()
            enter_spam_thread.join(timeout=0.5)
    # Не вызываем unload: следующий реплей загрузит свой скрипт поверх; unload после первого
    # реплея даёт "unknown message 3" и к моменту загрузки карты скрипта уже нет — машина не едет.

    # --- Диагностика: машина ехала? ---
    if race_started and max_cp_reached == 0:
        log.warning(
            "  CAR STUCK? Race started but 0 checkpoints reached (t=%dms, %d frames). "
            "Script may not have executed. Check TMI console for 'File does not exist'.",
            current_time, len(frames),
        )
    elif race_started:
        log.info("  Race summary: %d checkpoints, %d frames, final t=%dms", max_cp_reached, len(frames), current_time)

    if need_enter:
        return (None, True)
    return ({
        "frames": frames,
        "frame_times_ms": frame_times_ms,
        "race_finished": race_finished,
        "final_time_ms": current_time,
    }, False)


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
    respawn_action: str,
    respawn_after_cp_ms: int,
    running_speed: int,
    config_path: Path,
    per_frame_json: bool = False,
    track_ids_need_enter: Any = None,
    multi_instance: bool = False,
) -> None:
    """Single worker: convert .replay.gbx → TMI script, load via TMInterface, capture frames.
    When multi_instance is True, keys (Enter, Tilde) are sent only via PostMessage to this
    worker's window, so other game instances are not affected by focus stealing."""
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
        send_keys_to_window_only=multi_instance,
    )
    # Как в training: latest_map_path_requested — при смене трека запрашиваем новую карту.
    last_map_requested: list[str | None] = [None]

    try:
        try:
            task = task_queue.get(timeout=2.0)
        except queue.Empty:
            task = None
        if task is None:
            pass  # no tasks
        else:
            while True:
                try:
                    next_task = task_queue.get(timeout=2.0)
                except queue.Empty:
                    next_task = None
                track_id, replay_paths = task
                runs_done_this_track = 0  # only count actually run rollouts (skipped "already done" don't count)
                for replay_idx, replay_path in enumerate(replay_paths):
                    is_last_replay_this_task = replay_idx == len(replay_paths) - 1
                    replay_name = replay_path.stem
                    out_dir = output_dir / str(track_id) / replay_name
                    if (out_dir / "manifest.json").exists():
                        log.info("Skip (already done): %s / %s", track_id, replay_name)
                        # Ограничение: после skip игра остаётся на карте пропущенного реплея (её запросили в конце прошлого заезда). Следующий rollout может быть для другой карты — тогда в connect не шлём map (не в меню), получается неверная карта. Не skip'айте подряд реплеи на разных треках.
                        continue

                    # 1. Convert replay to TMInterface script format (safe name: no apostrophe/spaces for TMI load)
                    script_basename = f"replay_{track_id}_{_safe_script_basename(replay_name)}.txt"
                    script_path = scripts_dir / script_basename
                    replay_metadata = convert_replay_to_tmi_script(
                        replay_path,
                        script_path,
                        input_time_offset_ms=input_time_offset_ms,
                        respawn_action=respawn_action,
                        respawn_after_cp_ms=respawn_after_cp_ms,
                    )
                    if replay_metadata is None:
                        log.warning("Skip (conversion failed): %s", replay_path)
                        continue
                    
                    race_time_ms = replay_metadata["race_time_ms"]
                    if replay_metadata.get("game_version"):
                        log.info("  Game version: %s", replay_metadata["game_version"])

                    # Extract inputs from GBX for manifest (action_idx + full actions list)
                    inputs_per_step: list[dict[str, bool]] | None = None
                    action_indices: list[int] | None = None
                    inputs_result = extract_inputs_from_replay_gbx(
                        replay_path, step_ms, input_time_offset_ms=input_time_offset_ms,
                    )
                    if inputs_result is not None:
                        inputs_per_step, _, _ = inputs_result
                        action_indices = [action_dict_to_index(inp) for inp in inputs_per_step]
                    else:
                        log.debug("No inputs extracted from GBX for %s (manifest will have no action_idx)", replay_path.name)

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

                    # Следующая карта и скрипт после финиша: для цепочки unload → load → map.
                    if is_last_replay_this_task and next_task is not None:
                        next_map_name = get_map_filename_for_task(
                            next_task, tracks_dir, extract_challenge_from_replay
                        )
                        next_track_id, next_replay_paths = next_task
                        next_replay_path = next_replay_paths[0]
                    elif not is_last_replay_this_task:
                        next_map_name = dest_challenge.name
                        next_track_id = track_id
                        next_replay_path = replay_paths[replay_idx + 1]
                    else:
                        next_map_name = None
                        next_track_id = None
                        next_replay_path = None

                    next_script_path = None
                    if next_replay_path is not None:
                        next_script_basename = f"replay_{next_track_id}_{_safe_script_basename(next_replay_path.stem)}.txt"
                        next_script_path = scripts_dir / next_script_basename
                        if convert_replay_to_tmi_script(
                            next_replay_path,
                            next_script_path,
                            input_time_offset_ms=input_time_offset_ms,
                            respawn_action=respawn_action,
                            respawn_after_cp_ms=respawn_after_cp_ms,
                        ) is None:
                            next_script_path = None

                    # 3. Run replay with TMInterface native input loading (DETERMINISTIC!)
                    try:
                        rollout_results, need_enter = rollout_with_tmi_script(
                            gim=gim,
                            script_path=script_path,
                            map_name=dest_challenge.name,
                            race_time_ms=race_time_ms,
                            width=width,
                            height=height,
                            fps=fps,
                            step_ms=step_ms,
                            last_map_requested=last_map_requested,
                            next_map_name=next_map_name,
                            next_script_path=next_script_path,
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

                    if need_enter and track_ids_need_enter is not None:
                        track_ids_need_enter.append(track_id)
                        log.info("Track %s: needs 'Press Enter to start' (added to list)", track_id)
                    if rollout_results is None:
                        log.warning("Replay %s: rollout returned None", replay_path)
                        if gim.iface is not None:
                            try:
                                gim.iface.close()
                            except Exception:
                                pass
                            gim.iface = None
                        continue

                    runs_done_this_track += 1
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
                        if action_indices is not None and inputs_per_step is not None and len(action_indices) > 0:
                            step_idx = min(max(0, int(round(t_ms / step_ms))), len(action_indices) - 1)
                            entry["action_idx"] = action_indices[step_idx]
                            entry["inputs"] = dict(inputs_per_step[step_idx])
                        manifest_entries.append(entry)
                        if per_frame_json:
                            (out_dir / f"frame_{i:05d}_{t_ms}ms.json").write_text(
                                json.dumps(entry, indent=2), encoding="utf-8",
                            )
                    
                    manifest_obj: dict | list = (
                        {"entries": manifest_entries, "actions": action_indices}
                        if action_indices is not None
                        else manifest_entries
                    )
                    (out_dir / "manifest.json").write_text(
                        json.dumps(manifest_obj, indent=2), encoding="utf-8",
                    )
                    
                    capture_interval_ms = 1000.0 / fps if fps > 0 else float(step_ms)
                    metadata = {
                        "track_id": track_id,
                        "replay_name": replay_name,
                        "challenge_name": dest_challenge.name,
                        "fps": fps,
                        "running_speed": running_speed,
                        "capture_interval_ms": capture_interval_ms,
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
                task = next_task
                if task is None:
                    break
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
    parser.add_argument("--fps", type=float, default=10.0, help="Capture FPS (frames per real second; respects --running-speed)")
    parser.add_argument("--workers", type=int, default=1, help="Number of game instances (TMI ports)")
    parser.add_argument("--base-tmi-port", type=int, default=None, help="Base TMI port (default: from config)")
    parser.add_argument(
        "--track-ids",
        type=Path,
        default=Path("maps/track_ids.txt"),
        help=".txt file with track IDs, one per line. Order of lines = processing order. Default: maps/track_ids.txt",
    )
    parser.add_argument("--track-id", type=str, default=None, help="Single track ID to process")
    parser.add_argument(
        "--max-replays-per-track",
        type=int,
        default=None,
        metavar="N",
        help="Max replays per track (top N by filename: pos1, pos2, ...). If not set, process all replays per track.",
    )
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
    parser.add_argument(
        "--respawn-action",
        type=str,
        default="enter",
        help="TMInterface action used for replay Respawn events (e.g. enter, delete, backspace).",
    )
    parser.add_argument(
        "--respawn-after-cp-ms",
        type=int,
        default=0,
        help="Optional delay after latest replay checkpoint before Respawn (ms). 0 = exact replay timing.",
    )
    parser.add_argument(
        "--exclude-respawn-maps",
        action="store_true",
        help="Skip tracks that have at least one replay with respawn (checkpoint respawn) events.",
    )
    parser.add_argument(
        "--exclude-enter-maps",
        type=Path,
        default=None,
        metavar="FILE",
        help="Skip track IDs listed in FILE (one per line). Use with --write-enter-maps to build the list.",
    )
    parser.add_argument(
        "--write-enter-maps",
        type=Path,
        default=None,
        metavar="FILE",
        help="Append track IDs that did not start (e.g. 'Press Enter to start') to FILE. Next run use --exclude-enter-maps FILE.",
    )
    parser.add_argument("--running-speed", type=int, default=None, help="Override running_speed from config (e.g. 512 = 512x, 1 = real-time).")
    parser.add_argument("--config", type=Path, default=_default_yaml, help="Config YAML")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO). Use DEBUG for detailed Enter spam logs.",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

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

    if args.exclude_respawn_maps:
        track_ids_with_respawn: set[str] = set()
        for track_id, replay_path in tasks:
            if replay_has_respawn(replay_path):
                track_ids_with_respawn.add(track_id)
        if track_ids_with_respawn:
            tasks = [(tid, path) for tid, path in tasks if tid not in track_ids_with_respawn]
            log.info(
                "Excluded %d tracks with respawn (--exclude-respawn-maps): %s",
                len(track_ids_with_respawn),
                sorted(track_ids_with_respawn)[:20],
            )
            if len(track_ids_with_respawn) > 20:
                log.info("  ... and %d more", len(track_ids_with_respawn) - 20)
        if not tasks:
            log.info("No replay tasks left after excluding respawn maps.")
            return

    if args.exclude_enter_maps is not None and args.exclude_enter_maps.exists():
        exclude_enter: set[str] = set()
        for line in args.exclude_enter_maps.read_text(encoding="utf-8").splitlines():
            tid = line.strip()
            if tid and not tid.startswith("#"):
                exclude_enter.add(tid)
        if exclude_enter:
            tasks = [(tid, path) for tid, path in tasks if tid not in exclude_enter]
            log.info("Excluded %d tracks (--exclude-enter-maps %s)", len(exclude_enter), args.exclude_enter_maps)
        if not tasks:
            log.info("No replay tasks left after excluding enter-maps.")
            return

    grouped = group_replay_tasks_by_track(tasks)
    replays_limit_str = f"max {args.max_replays_per_track} replays/track" if args.max_replays_per_track is not None else "all replays/track"
    log.info("Filtered to %d replay tasks (%d tracks, %s)", len(tasks), len(grouped), replays_limit_str)

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
    if args.respawn_after_cp_ms < 0 or args.respawn_after_cp_ms % 10 != 0:
        raise ValueError(
            f"Invalid --respawn-after-cp-ms={args.respawn_after_cp_ms}. "
            "It must be a non-negative multiple of 10."
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
    track_ids_need_enter_list = manager.list() if args.write_enter_maps is not None else None

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
                args.respawn_action,
                args.respawn_after_cp_ms,
                running_speed,
                args.config.resolve(),
                getattr(args, "per_frame_json", False),
                track_ids_need_enter_list,
                args.workers > 1,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    if args.write_enter_maps is not None and track_ids_need_enter_list is not None and len(track_ids_need_enter_list) > 0:
        existing: set[str] = set()
        if args.write_enter_maps.exists():
            for line in args.write_enter_maps.read_text(encoding="utf-8").splitlines():
                tid = line.strip()
                if tid and not tid.startswith("#"):
                    existing.add(tid)
        to_append = sorted(set(track_ids_need_enter_list) - existing)
        if to_append:
            args.write_enter_maps.parent.mkdir(parents=True, exist_ok=True)
            with open(args.write_enter_maps, "a", encoding="utf-8") as f:
                for tid in to_append:
                    f.write(tid + "\n")
            log.info("Appended %d track IDs (need Enter) to %s", len(to_append), args.write_enter_maps)

    log.info("Done.")


if __name__ == "__main__":
    main()
