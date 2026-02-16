"""
Filter track_ids.txt: remove track IDs with custom MapType or long preview (MediaTracker intro).

Reads an input .txt (one track_id per line), checks Challenge.Gbx files for:
  - Non-standard environment (e.g., not Stadium/Speed/Alpine/Rally/Bay/Island/Coast/Desert)
  - Custom flags that indicate scripts or custom map behavior
  - [NOT YET IMPLEMENTED] MediaTracker intro duration > threshold (--max-preview-duration)
    Limitation: pygbx doesn't parse MediaTracker clips (CGameCtnMediaClip, class 0x03079000) yet.
    When implemented, will check intro clip duration and exclude maps with preview > N seconds.

Not all tracks have .Challenge.Gbx downloaded: missing files are treated as "standard"
(track kept). Use --only-with-maps to output only tracks that have a Challenge file.

Parse errors (corrupt/truncated .gbx, UTF-8 inside pygbx) come from the map files
or pygbx, not from this script; such maps are treated as "standard".

Usage:
  # Basic: filter by environment
  python scripts/filter_track_ids_custom_maptype.py

  # Custom input/output
  python scripts/filter_track_ids_custom_maptype.py --input maps/track_ids.txt --output maps/track_ids_standard.txt

  # Only tracks with Challenge.Gbx
  python scripts/filter_track_ids_custom_maptype.py --tracks-dir maps/tracks_tmnf --only-with-maps

  # (Future) Filter by preview duration
  python scripts/filter_track_ids_custom_maptype.py --max-preview-duration 15.0

Then use the output with capture_replays_tmnf.py:
  python scripts/capture_replays_tmnf.py --track-ids maps/track_ids_standard.txt --tracks-dir maps/tracks_tmnf --output-dir out

Optional: pip install tqdm for a progress bar when checking maps.
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

# Standard TMNF/TMUF environments
STANDARD_ENVIRONMENTS = {
    "Stadium",
    "Speed",
    "Alpine",
    "Rally",
    "Bay",
    "Island",
    "Coast",
    "Desert",
}


def map_is_custom(
    map_path: Path,
    max_preview_duration_s: float | None = None,
    parse_error_out: list[int] | None = None,
) -> bool:
    """True if the map has a custom MapType, non-standard environment, or long preview.
    
    Unreadable/corrupt maps (UTF-8 errors, truncated file in pygbx) → return False and optionally count in parse_error_out.
    
    Checks:
      - environment not in STANDARD_ENVIRONMENTS (case-insensitive)
      - flags field (future: check for script flags)
      - MediaTracker intro duration > max_preview_duration_s (NOT YET IMPLEMENTED: pygbx doesn't parse MediaTracker clips)
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
                gbx = Gbx(str(map_path))
                challenges = gbx.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
                if not challenges:
                    return False
                challenge = challenges[0]

                # Check environment
                env = getattr(challenge, "environment", None)
                if env:
                    env_str = str(env).strip()
                    if env_str and env_str not in STANDARD_ENVIRONMENTS:
                        # Non-standard environment (e.g., custom, stunts, etc.)
                        return True

                # Check flags (future: parse for script/custom behavior indicators)
                # flags = getattr(challenge, "flags", 0)
                # if flags & SOME_CUSTOM_FLAG_MASK:
                #     return True

                # Check MediaTracker intro duration (NOT YET IMPLEMENTED)
                # Currently pygbx doesn't parse MediaTracker clips (CGameCtnMediaClip, class 0x03079000),
                # so we can't check duration yet. When pygbx adds support or we implement manual parsing:
                #   1. Find CGameCtnMediaClip with intro=True or triggers at race start
                #   2. Calculate total duration of all intro clips
                #   3. if total_intro_duration_s > max_preview_duration_s: return True
                # For now, this check is skipped.
                if max_preview_duration_s is not None:
                    # TODO: Implement MediaTracker intro duration parsing
                    # Possible approach: manually parse chunk 0x03043054 (CGameCtnChallenge::ChallengeParameters)
                    # which might contain intro clip references, or chunk 0x03043049 (intro clip list)
                    pass

                return False
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        if parse_error_out is not None:
            parse_error_out[0] += 1
        # Corrupt/truncated map, UTF-8 errors in pygbx, struct.unpack, etc. → treat as standard
        return False
    finally:
        for _log, _lev in _saved:
            _log.setLevel(_lev)


def check_track(
    track_id: str,
    tracks_dir: Path,
    max_preview_duration_s: float | None,
    parse_error_count: list[int],
) -> tuple[bool, bool]:
    """Returns (is_custom, has_map_file). Increments parse_error_count[0] when a map file fails to parse."""
    # Search for Challenge.Gbx in:
    #   1. tracks_dir/<track_id>/*.Challenge.Gbx
    #   2. tracks_dir/<track_id>.Challenge.Gbx
    #   3. tracks_dir/*<track_id>*.Challenge.Gbx (fuzzy)
    track_subdir = tracks_dir / track_id
    if track_subdir.is_dir():
        for p in track_subdir.iterdir():
            if p.is_file() and p.suffix.lower() == ".gbx" and ".challenge" in p.name.lower():
                if map_is_custom(p, max_preview_duration_s, parse_error_count):
                    return (True, True)
                return (False, True)

    exact = tracks_dir / f"{track_id}.Challenge.Gbx"
    if exact.is_file():
        if map_is_custom(exact, max_preview_duration_s, parse_error_count):
            return (True, True)
        return (False, True)

    exact_lower = tracks_dir / f"{track_id}.challenge.gbx"
    if exact_lower.is_file():
        if map_is_custom(exact_lower, max_preview_duration_s, parse_error_count):
            return (True, True)
        return (False, True)

    # Fuzzy search
    if tracks_dir.is_dir():
        for p in list(tracks_dir.glob(f"*{track_id}*.Challenge.Gbx")) + list(tracks_dir.glob(f"*{track_id}*.challenge.gbx")):
            if p.is_file():
                if map_is_custom(p, max_preview_duration_s, parse_error_count):
                    return (True, True)
                return (False, True)

    return (False, False)


def _worker_check_one(item: tuple[str, str, float | None]) -> tuple[str, bool, int]:
    """Worker: (track_id, tracks_dir_str, max_preview_duration_s) -> (track_id, is_custom, parse_errors)."""
    track_id, tracks_dir_str, max_preview_duration_s = item
    tracks_dir = Path(tracks_dir_str)
    parse_error_count: list[int] = [0]
    is_custom, _ = check_track(track_id, tracks_dir, max_preview_duration_s, parse_error_count)
    return (track_id, is_custom, parse_error_count[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter track_ids.txt: remove tracks with custom MapType or long preview."
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
        default=Path("maps/track_ids_standard.txt"),
        help="Output .txt with filtered track_ids (only standard maps). Default: maps/track_ids_standard.txt",
    )
    parser.add_argument(
        "--tracks-dir",
        "-t",
        type=Path,
        default=Path("maps/tracks_tmnf"),
        help="Path to Challenge.Gbx files: <track_id>/*.Challenge.Gbx or <track_id>.Challenge.Gbx. Default: maps/tracks_tmnf",
    )
    parser.add_argument(
        "--only-with-maps",
        action="store_true",
        help="Only output track IDs that have a Challenge.Gbx file (and are standard). Skip tracks with no map downloaded.",
    )
    parser.add_argument(
        "--max-preview-duration",
        type=float,
        default=None,
        help="[NOT YET IMPLEMENTED] Maximum MediaTracker intro duration in seconds. Maps with longer previews are excluded. "
             "Example: --max-preview-duration 15.0 (requires pygbx MediaTracker parsing support)",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="Number of parallel jobs (default: cpu_count). Set to 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    input_path: Path = args.input
    output_path: Path = args.output
    tracks_dir: Path = args.tracks_dir
    only_with_maps: bool = args.only_with_maps
    max_preview_duration_s: float | None = args.max_preview_duration
    jobs: int | None = args.jobs

    if not input_path.is_file():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    if not tracks_dir.is_dir():
        log.error("Tracks directory not found: %s", tracks_dir)
        sys.exit(1)

    if max_preview_duration_s is not None:
        log.warning(
            "--max-preview-duration is not yet implemented (pygbx doesn't parse MediaTracker clips). "
            "Only environment-based filtering will be applied."
        )

    log.info("Reading track IDs from %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        track_ids = [line.strip() for line in f if line.strip()]
    log.info("Read %d track IDs", len(track_ids))

    log.info("Checking maps in %s (jobs=%s)", tracks_dir, jobs or "auto")
    tracks_dir_str = str(tracks_dir.resolve())

    use_multiprocessing = jobs != 1
    if use_multiprocessing:
        if jobs is None:
            jobs = mp.cpu_count()
        log.info("Using %d parallel jobs", jobs)
        with mp.Pool(jobs) as pool:
            items = [(tid, tracks_dir_str, max_preview_duration_s) for tid in track_ids]
            if tqdm:
                results = list(tqdm(pool.imap(_worker_check_one, items, chunksize=10), total=len(track_ids), desc="Checking maps", unit="map"))
            else:
                results = []
                for i, r in enumerate(pool.imap(_worker_check_one, items, chunksize=10), start=1):
                    results.append(r)
                    if i % _PROGRESS_INTERVAL == 0 or i == len(track_ids):
                        log.info("Progress: %d/%d maps checked", i, len(track_ids))
    else:
        log.info("Single-threaded mode")
        results = []
        for i, tid in enumerate(track_ids, start=1):
            results.append(_worker_check_one((tid, tracks_dir_str, max_preview_duration_s)))
            if not tqdm and (i % _PROGRESS_INTERVAL == 0 or i == len(track_ids)):
                log.info("Progress: %d/%d maps checked", i, len(track_ids))

    # Build output: preserve order, filter custom maps
    output_ids = []
    custom_count = 0
    no_map_count = 0
    parse_error_total = 0

    for tid in track_ids:
        # Find result for this track_id
        res = next((r for r in results if r[0] == tid), None)
        if res is None:
            # Should not happen
            log.warning("Missing result for track_id=%s (internal error)", tid)
            continue
        _, is_custom, parse_errors = res
        parse_error_total += parse_errors

        # Determine if track has a map file
        has_map = False
        parse_error_count_check: list[int] = [0]
        _, has_map = check_track(tid, tracks_dir, max_preview_duration_s, parse_error_count_check)

        if is_custom:
            custom_count += 1
            continue

        if not has_map:
            no_map_count += 1
            if only_with_maps:
                continue

        output_ids.append(tid)

    log.info("Total custom maps (excluded): %d", custom_count)
    log.info("Total maps without Challenge.Gbx: %d", no_map_count)
    log.info("Parse errors (corrupt/truncated .gbx): %d", parse_error_total)
    log.info("Writing %d track IDs to %s", len(output_ids), output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for tid in output_ids:
            f.write(f"{tid}\n")

    log.info("Done. Output: %s", output_path)
    log.info("Use with: python scripts/capture_replays_tmnf.py --track-ids %s --tracks-dir %s --output-dir out", output_path, tracks_dir)


if __name__ == "__main__":
    main()
