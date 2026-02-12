"""
Capture frames using TMInterface's Validate + replay workflow.

This script is intended for debugging and high-fidelity reproduction on a
single replay/map pair when text-script conversion is suspect.

Workflow:
  1) Put replay into the game's Replays folder.
  2) In game: Replays -> Edit a replay -> Launch -> Validate.
  3) Run this script. It sends `replay` (to load validated inputs), then loads
     the map and captures frames during the run.

Notes:
  - TMInterface `replay` reuses currently validated replay inputs.
  - If no replay was validated in game beforehand, this script will not be able
    to recover those inputs automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

_script_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_script_root))
sys.path.insert(0, str(_script_root / "scripts"))

from config_files.config_loader import get_config, load_config, set_config

_default_yaml = _script_root / "config_files" / "config_default.yaml"
if _default_yaml.exists():
    set_config(load_config(_default_yaml))

from trackmania_rl.tmi_interaction import game_instance_manager
from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface

from capture_replays_tmnf import get_challenge_path_for_track
from replays_tmnf.api import extract_challenge_from_replay

log = logging.getLogger(__name__)


def _copy_replay_to_game_replays(replay_path: Path, game_base_path: Path) -> Path:
    import shutil

    replays_dir = game_base_path / "Tracks" / "Replays"
    replays_dir.mkdir(parents=True, exist_ok=True)
    dst = replays_dir / replay_path.name
    shutil.copy2(replay_path, dst)
    return dst


def _candidate_game_base_paths(config_base_path: Path) -> list[Path]:
    docs = Path.home() / "Documents"
    candidates = [
        config_base_path,
        docs / "TmForever",
        docs / "TrackMania",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        k = str(p.resolve()) if p.exists() else str(p)
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return unique


def _prepare_map_for_game(
    replay_path: Path,
    track_id: str,
    tracks_dir: Path,
    game_base_path: Path,
) -> Path:
    challenge_path = get_challenge_path_for_track(
        track_id=track_id,
        replay_path=replay_path,
        tracks_dir=tracks_dir,
        extract_challenge_from_replay=extract_challenge_from_replay,
    )
    if challenge_path is None:
        raise RuntimeError(f"Cannot find/extract challenge for replay: {replay_path}")

    import shutil

    game_challenges = game_base_path / "Tracks" / "Challenges"
    game_challenges.mkdir(parents=True, exist_ok=True)
    dst_challenge = game_challenges / challenge_path.name
    if dst_challenge.resolve() != challenge_path.resolve():
        shutil.copy2(challenge_path, dst_challenge)
    return dst_challenge


def _copy_replay_to_all_candidate_replays(replay_path: Path, config_base_path: Path) -> list[Path]:
    copied: list[Path] = []
    for base in _candidate_game_base_paths(config_base_path):
        try:
            copied.append(_copy_replay_to_game_replays(replay_path, base))
        except Exception:
            continue
    return copied


def _prepare_map_for_all_candidate_paths(
    replay_path: Path,
    track_id: str,
    tracks_dir: Path,
    config_base_path: Path,
) -> list[Path]:
    prepared: list[Path] = []
    for base in _candidate_game_base_paths(config_base_path):
        try:
            prepared.append(_prepare_map_for_game(replay_path, track_id, tracks_dir, base))
        except Exception:
            continue
    return prepared


def run_validate_replay_capture(
    map_name: str,
    output_dir: Path,
    width: int,
    height: int,
    fps: float,
    step_ms: int,
    running_speed: int,
    tmi_port: int,
    replay_speed: float,
    send_replay_command: bool,
    validate_replay_path: Path | None,
) -> dict[str, Any] | None:
    gim = game_instance_manager.GameInstanceManager(
        game_spawning_lock=None,
        running_speed=running_speed,
        run_steps_per_action=max(1, step_ms // 10),
        max_overall_duration_ms=600_000,
        max_minirace_duration_ms=600_000,
        tmi_port=tmi_port,
    )

    def _connect_iface_with_retry(timeout_s: float = 20.0) -> None:
        started = time.perf_counter()
        last_err: Exception | None = None
        while time.perf_counter() - started < timeout_s:
            try:
                gim.iface = TMInterface(tmi_port)
                gim.iface.register(timeout=2000)
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.5)
        raise RuntimeError(f"Cannot connect to TMInterface on port {tmi_port}: {last_err}")

    def _launch_game_with_validatepath_if_requested() -> None:
        if validate_replay_path is None:
            gim.launch_game()
            return
        cfg = get_config()
        if cfg.is_linux:
            log.warning("Automatic validatepath launch is not implemented on Linux; using regular launch.")
            gim.launch_game()
            return

        validate_arg = str(validate_replay_path.resolve()).replace("/", "\\")
        launch_string = (
            'powershell -executionPolicy bypass -command "& {'
            f" $process = start-process -FilePath '{cfg.windows_TMLoader_path}'"
            " -PassThru -ArgumentList "
            f'\'run TmForever "{cfg.windows_TMLoader_profile_name}" '
            f'/validatepath="{validate_arg}"\';'
            ' echo exit $process.id}"'
        )
        tmi_process_id = int(subprocess.check_output(launch_string).decode().split("\r\n")[1])
        while gim.tm_process_id is None:
            for process in psutil.process_iter(["pid", "ppid", "name"]):
                try:
                    info = process.info
                    if info["name"].startswith("TmForever") and info["ppid"] == tmi_process_id:
                        gim.tm_process_id = info["pid"]
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        while not gim.is_game_running():
            time.sleep(0)
        gim.get_tm_window_id()

    _launch_game_with_validatepath_if_requested()
    _connect_iface_with_retry()

    capture_interval_ms = 1000.0 / fps if fps > 0 else float(step_ms)
    next_capture_time_ms = 0.0
    frames: list[np.ndarray] = []
    frame_times_ms: list[int] = []

    race_started = False
    race_finished = False
    race_start_requested = False
    frame_expected = False
    current_time = -3000
    timeout_start = None
    replay_inputs_loaded = False

    def _has_loaded_inputs() -> bool:
        try:
            inp = gim.iface.get_inputs()
            return bool(inp and inp.strip())
        except Exception:
            return False

    try:
        while not race_finished:
            try:
                msgtype = gim.iface._read_int32()
            except (ConnectionResetError, OSError):
                log.warning("TMInterface connection dropped; reconnecting...")
                _connect_iface_with_retry()
                race_started = False
                race_start_requested = False
                frame_expected = False
                continue

            if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                gim.iface.execute_command(f"set countdown_speed {running_speed}")
                gim.iface.execute_command("set skip_map_load_screens true")
                gim.iface.execute_command("set unfocused_fps_limit false")
                gim.iface.execute_command("set execute_commands true")
                gim.iface.execute_command(f"set replay_speed {replay_speed}")
                gim.iface.execute_command(f"cam {get_config().game_camera_number}")
                gim.iface.set_on_step_period(step_ms)

                if send_replay_command:
                    log.info("Sending `replay` command (expects validated replay in game).")
                    gim.iface.execute_command("replay")
                    time.sleep(0.2)
                    replay_inputs_loaded = _has_loaded_inputs()
                    if not replay_inputs_loaded:
                        log.warning("No inputs loaded after `replay` yet; will retry before race start.")

                log.info("Loading map: %s", map_name)
                gim.iface.execute_command(f"map {map_name}")
                gim.iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                current_time = gim.iface._read_int32()

                if current_time <= -2990 and not race_start_requested:
                    if send_replay_command and not replay_inputs_loaded:
                        log.info("Retrying `replay` before race start...")
                        gim.iface.execute_command("replay")
                        time.sleep(0.15)
                        replay_inputs_loaded = _has_loaded_inputs()

                    if send_replay_command and not replay_inputs_loaded:
                        log.error(
                            "Validated replay inputs are unavailable. "
                            "Automatic path requires startup validatepath to succeed."
                        )
                        gim.iface._respond_to_call(msgtype)
                        return None

                    log.info("Starting race (give_up)...")
                    gim.iface.give_up()
                    race_start_requested = True

                if not race_started and current_time >= 0:
                    race_started = True
                    timeout_start = time.perf_counter()
                    log.info("Race started, capturing frames...")

                if race_started and current_time >= next_capture_time_ms:
                    if not frame_expected:
                        gim.iface.request_frame(width, height)
                        frame_expected = True
                        next_capture_time_ms += capture_interval_ms

                if race_started and timeout_start:
                    if time.perf_counter() - timeout_start > 180:
                        log.warning("Timeout while waiting for finish.")
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
                    race_finished = True
                    log.info("Race finished at t=%dms", current_time)
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

    finally:
        try:
            gim.iface.execute_command("unload")
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for i, (frame, t_ms) in enumerate(zip(frames, frame_times_ms)):
        frame_name = f"frame_{i:06d}_{int(t_ms)}ms.jpeg"
        frame_path = output_dir / frame_name
        cv2.imwrite(str(frame_path), frame[0], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        manifest.append(
            {
                "file": frame_name,
                "index": i,
                "time_ms": int(t_ms),
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    metadata = {
        "mode": "validate_replay",
        "map_name": map_name,
        "width": width,
        "height": height,
        "fps": fps,
        "step_ms": step_ms,
        "running_speed": running_speed,
        "replay_speed": replay_speed,
        "sent_replay_command": send_replay_command,
        "used_startup_validatepath": validate_replay_path is not None,
        "total_frames": len(frames),
        "race_finished": race_finished,
        "final_time_ms": int(current_time),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metadata


def _signal_handler(sig, frame):
    print("\nReceived SIGINT, exiting.")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture frames via TMInterface Validate + replay mode.")
    parser.add_argument("--replay-path", type=Path, required=True, help="Path to *.replay.gbx")
    parser.add_argument("--track-id", type=str, default=None, help="Track ID (defaults to replay parent folder name)")
    parser.add_argument("--tracks-dir", type=Path, default=Path("maps/tracks"), help="Tracks dir for challenge lookup/extract")
    parser.add_argument("--output-dir", type=Path, default=Path("capture_validate_out"), help="Output directory")
    parser.add_argument("--width", type=int, default=None, help="Frame width")
    parser.add_argument("--height", type=int, default=None, help="Frame height")
    parser.add_argument("--fps", type=float, default=10.0, help="Capture FPS")
    parser.add_argument("--step-ms", type=int, default=10, help="Step period in ms")
    parser.add_argument("--running-speed", type=int, default=None, help="TM countdown speed")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="TMInterface replay_speed")
    parser.add_argument("--base-tmi-port", type=int, default=None, help="TMInterface port")
    parser.add_argument("--prepare-only", action="store_true", help="Only copy replay/map and print manual Validate steps")
    parser.add_argument("--no-replay-command", action="store_true", help="Do not send TMInterface `replay` command")
    parser.add_argument("--use-startup-validatepath", action="store_true", help="Enable startup /validatepath automation (mass-validation mode; may not keep live TMInterface session)")
    parser.add_argument("--config", type=Path, default=_default_yaml, help="Config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.config.exists():
        set_config(load_config(args.config))
    cfg = get_config()

    signal.signal(signal.SIGINT, _signal_handler)

    replay_path = args.replay_path.resolve()
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay not found: {replay_path}")

    if args.step_ms <= 0 or args.step_ms % 10 != 0:
        raise ValueError("--step-ms must be a positive multiple of 10.")

    game_base_path = Path(cfg.trackmania_base_path)
    track_id = args.track_id if args.track_id is not None else replay_path.parent.name
    track_id = str(track_id)

    copied_replays = _copy_replay_to_all_candidate_replays(replay_path, game_base_path)
    prepared_maps = _prepare_map_for_all_candidate_paths(
        replay_path=replay_path,
        track_id=track_id,
        tracks_dir=args.tracks_dir.resolve(),
        config_base_path=game_base_path,
    )
    if not prepared_maps:
        raise RuntimeError("Could not prepare map in any candidate user directory (TmForever/TrackMania).")
    dst_challenge = prepared_maps[0]

    if copied_replays:
        for p in copied_replays:
            log.info("Replay copied to: %s", p)
    else:
        log.warning("Replay copy failed for all candidate user dirs.")
    for p in prepared_maps:
        log.info("Map ready at: %s", p)

    if args.prepare_only:
        print("")
        print("Preparation done. Next steps in game:")
        print("  1) Replays -> Edit a replay -> choose copied replay")
        print("  2) Launch -> Validate")
        print("  3) Run this script again without --prepare-only")
        return

    width = args.width if args.width is not None else cfg.W_downsized
    height = args.height if args.height is not None else cfg.H_downsized
    if not args.use_startup_validatepath:
        base_tmi_port = args.base_tmi_port if args.base_tmi_port is not None else cfg.base_tmi_port
    else:
        # In this TMInterface build custom_port is not supported; use default.
        base_tmi_port = args.base_tmi_port if args.base_tmi_port is not None else 8478
    running_speed = args.running_speed if args.running_speed is not None else cfg.running_speed

    out_dir = args.output_dir.resolve() / track_id / replay_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = run_validate_replay_capture(
        map_name=dst_challenge.name,
        output_dir=out_dir,
        width=width,
        height=height,
        fps=args.fps,
        step_ms=args.step_ms,
        running_speed=running_speed,
        tmi_port=base_tmi_port,
        replay_speed=args.replay_speed,
        send_replay_command=not args.no_replay_command,
        validate_replay_path=copied_replays[0] if (args.use_startup_validatepath and copied_replays) else None,
    )
    if metadata is None:
        log.error("Capture failed.")
    else:
        log.info("Done. Saved %d frames to %s", metadata["total_frames"], out_dir)


if __name__ == "__main__":
    main()

