"""
Filter track_ids.txt: remove track IDs that have at least one replay with respawn.

Reads an input .txt (one track_id per line), checks maps/replays/<track_id>/*.replay.gbx
for respawn/enter control events. Writes only track_ids that have no such replays,
in the same order, to a new .txt file.

Not all tracks have replays downloaded: missing replays dir or empty dir is treated
as "no respawn" (track kept). Use --only-with-replays to output only tracks that
actually have at least one replay (and none have respawn).

Parse errors (corrupt/truncated .gbx, UTF-8 inside pygbx) come from the replay files
or pygbx, not from this script; such replays are treated as "no respawn".

Usage:
  python scripts/filter_track_ids_no_respawn.py
  python scripts/filter_track_ids_no_respawn.py --input maps/track_ids.txt --output maps/track_ids_no_respawn.txt
  python scripts/filter_track_ids_no_respawn.py --replays-dir maps/replays --only-with-replays

Then use the output with capture_replays_tmnf.py:
  python scripts/capture_replays_tmnf.py --track-ids maps/track_ids_no_respawn.txt --replays-dir maps/replays --output-dir out

Optional: pip install tqdm for a progress bar when checking replays.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from contextlib import redirect_stderr
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

log = logging.getLogger(__name__)
_PROGRESS_INTERVAL = 500


def replay_has_respawn(replay_path: Path, parse_error_out: list[int] | None = None) -> bool:
    """True if the replay contains any respawn or enter control event.
    Unreadable/corrupt replays (UTF-8 errors, truncated file in pygbx) → return False and optionally count in parse_error_out.
    """
    try:
        from pygbx import Gbx, GbxType
    except ImportError:
        log.error("pygbx is required. Install with: pip install pygbx")
        raise
    import logging as _logmod
    _saved = []
    for _name in ("pygbx", "pygbx.gbx"):
        _log = _logmod.getLogger(_name)
        _saved.append((_log, _log.level))
        _log.setLevel(_logmod.CRITICAL)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stderr(devnull):
                gbx = Gbx(str(replay_path))
                ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
                if not ghosts:
                    return False
                ghost = min(
                    ghosts,
                    key=lambda g: getattr(g, "cp_times", [0])[-1] if getattr(g, "cp_times", None) else 0,
                )
                control_entries = list(getattr(ghost, "control_entries", []) or [])
                for ce in control_entries:
                    name = (
                        str(getattr(ce, "event_name", getattr(ce, "EventName", "")) or "")
                    ).strip().lower()
                    if name in ("respawn", "enter"):
                        return True
                return False
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        if parse_error_out is not None:
            parse_error_out[0] += 1
        # Corrupt/truncated replay, UTF-8 errors in pygbx, struct.unpack, etc. → treat as no respawn
        return False
    finally:
        for _log, _lev in _saved:
            _log.setLevel(_lev)


def check_track(
    track_id: str,
    replays_dir: Path,
    parse_error_count: list[int],
) -> tuple[bool, bool]:
    """Returns (has_respawn, has_any_replay). Increments parse_error_count[0] when a replay file fails to parse."""
    track_dir = replays_dir / track_id
    if not track_dir.is_dir():
        return (False, False)
    replay_files = [
        p
        for p in track_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".gbx" and ".replay" in p.name.lower()
    ]
    if not replay_files:
        return (False, False)
    for p in replay_files:
        if replay_has_respawn(p, parse_error_count):
            return (True, True)
    return (False, True)


def _worker_check_one(item: tuple[str, str]) -> tuple[str, bool, int]:
    """Worker: (track_id, replays_dir_str) -> (track_id, has_respawn, parse_errors)."""
    track_id, replays_dir_str = item
    replays_dir = Path(replays_dir_str)
    parse_error_count: list[int] = [0]
    has_respawn, _ = check_track(track_id, replays_dir, parse_error_count)
    return (track_id, has_respawn, parse_error_count[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter track_ids.txt: remove tracks that have replays with respawn."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("maps/track_ids.txt"),
        help="Input .txt with one track_id per line (order preserved in output). Default: maps/track_ids.txt",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("maps/track_ids_no_respawn.txt"),
        help="Output .txt with filtered track_ids. Default: maps/track_ids_no_respawn.txt",
    )
    parser.add_argument(
        "--replays-dir",
        "-r",
        type=Path,
        default=Path("maps/replays"),
        help="Path to replays: track_id/*.replay.gbx. Default: maps/replays",
    )
    parser.add_argument(
        "--only-with-replays",
        action="store_true",
        help="Only output track IDs that have at least one replay (and none have respawn). Skip tracks with no replays downloaded.",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of parallel workers. Default: CPU count - 1.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log each excluded track.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    track_ids = [
        line.strip()
        for line in args.input.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    total_input = len(track_ids)
    log.info("Read %d track IDs from %s", total_input, args.input)

    if not args.replays_dir.exists():
        log.warning("Replays dir does not exist: %s", args.replays_dir)
        existing_ids = set()
    else:
        existing_ids = {d.name for d in args.replays_dir.iterdir() if d.is_dir()}
    log.info("Tracks with replays dir in %s: %d", args.replays_dir, len(existing_ids))

    to_check = [tid for tid in track_ids if tid in existing_ids]
    no_replays_ids = [tid for tid in track_ids if tid not in existing_ids]
    no_replays = len(no_replays_ids)
    to_process = len(to_check)
    log.info("Will check %d tracks (have replays); %d in input have no replays dir (skipped).", to_process, no_replays)

    if to_process == 0:
        if args.only_with_replays:
            kept = []
        else:
            kept = no_replays_ids[:]
        excluded = []
        total_parse_errors = 0
    else:
        work_items = [(tid, str(args.replays_dir)) for tid in to_check]
        result_by_id: dict[str, bool] = {}
        total_parse_errors = 0

        chunksize = max(1, to_process // (args.workers * 4)) if args.workers > 1 else 1

        if args.workers <= 1:
            iterator = (_worker_check_one(item) for item in work_items)
            if tqdm is not None:
                iterator = tqdm(
                    iterator,
                    total=to_process,
                    desc="Checking replays",
                    unit="track",
                    file=sys.stderr,
                    dynamic_ncols=True,
                )
            for i, r in enumerate(iterator):
                result_by_id[r[0]] = r[1]
                total_parse_errors += r[2]
                if tqdm is None and ((i + 1) % _PROGRESS_INTERVAL == 0 or (i + 1) == to_process):
                    pct = 100.0 * (i + 1) / to_process
                    sys.stderr.write(f"\rProgress: {i + 1} / {to_process} ({pct:.1f}%)  ")
                    sys.stderr.flush()
        else:
            with mp.Pool(args.workers) as pool:
                iterator = pool.imap_unordered(
                    _worker_check_one,
                    work_items,
                    chunksize=chunksize,
                )
                if tqdm is not None:
                    iterator = tqdm(
                        iterator,
                        total=to_process,
                        desc="Checking replays",
                        unit="track",
                        file=sys.stderr,
                        dynamic_ncols=True,
                    )
                for i, r in enumerate(iterator):
                    result_by_id[r[0]] = r[1]
                    total_parse_errors += r[2]
                    if tqdm is None and ((i + 1) % _PROGRESS_INTERVAL == 0 or (i + 1) == to_process):
                        pct = 100.0 * (i + 1) / to_process
                        sys.stderr.write(f"\rProgress: {i + 1} / {to_process} ({pct:.1f}%)  ")
                        sys.stderr.flush()
        if tqdm is None and to_process > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()

        excluded = [tid for tid in to_check if result_by_id[tid]]
        kept_set: set[str] = set()
        if not args.only_with_replays:
            kept_set.update(no_replays_ids)
        for tid in to_check:
            if not result_by_id[tid]:
                kept_set.add(tid)
        kept = [tid for tid in track_ids if tid in kept_set]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    log.info(
        "Wrote %d track IDs to %s",
        len(kept),
        args.output,
    )
    log.info(
        "Summary: kept=%d  excluded (has respawn)=%d  no replays (not in maps/replays)=%d  parse errors (corrupt replays)=%d",
        len(kept),
        len(excluded),
        no_replays,
        total_parse_errors,
    )
    if total_parse_errors > 0:
        log.info(
            "Parse errors are from pygbx reading corrupt/truncated .gbx or non-UTF-8 strings; those replays were treated as 'no respawn'.",
        )
    if excluded and not args.verbose:
        log.info("First 20 excluded: %s", excluded[:20])
    log.info("Use with capture: --track-ids %s", args.output)


if __name__ == "__main__":
    main()
