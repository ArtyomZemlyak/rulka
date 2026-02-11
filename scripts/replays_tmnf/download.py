"""
Download top TMNF replays: by track ID/name/file, or list popular then download.
Uses run_list_popular from list_popular when --list-popular.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .api import download_replay, extract_challenge_from_replay, get_track_replays, safe_filename, search_tracks
from .list_popular import run_list_popular
from .pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download top TMNF leaderboard replays from TMNF-X (ManiaExchange).")
    ap.add_argument("--track-id", type=int, help="Single track ID (from tmnf.exchange)")
    ap.add_argument("--track-name", type=str, help="Search tracks by name and use first match")
    ap.add_argument("--track-ids", type=Path, help="File with one track ID per line (ignored if --track-id/--track-name set)")
    ap.add_argument("--list-popular", action="store_true", help="Fetch popular track IDs (via list_popular), then optionally download")
    ap.add_argument("--output", type=Path, default=Path("popular_track_ids_tmnf.txt"), help="Output file for track IDs when --list-popular")
    ap.add_argument("--output-dir", type=Path, default=Path("replays_tmnf"), help="Output directory for replays")
    ap.add_argument("--top", type=int, default=50, help="Number of top replays per track")
    ap.add_argument("--offset", type=int, default=0, help="Offset in replay list (pagination)")
    ap.add_argument("--rate-delay", type=float, default=0.0, help="Seconds between API/download requests (default: 0)")
    ap.add_argument("--workers", type=int, default=8, help="Parallel download workers (default: 8, use 1 for sequential)")
    ap.add_argument("--api-workers", type=int, default=0, help="Parallel API workers for fetching replay lists (default: 0 = same as --workers; pipeline bottleneck is API, increase to speed up)")
    ap.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars (use only logging)")
    ap.add_argument("--dry-run", action="store_true", help="Only list what would be downloaded")
    # list-popular options (used when --list-popular or by list_popular_tracks_tmnf)
    ap.add_argument("--pages", type=int, default=9999, help="Max pages when --list-popular (stop on empty page or no more)")
    ap.add_argument("--per-page", type=int, default=100, help="Tracks per page when --list-popular (fewer requests)")
    ap.add_argument("--order", type=str, default="activity", choices=["all", "awards", "activity", "uploaded"], help="Order when --list-popular (default: activity = most active first)")
    ap.add_argument("--min-awards", type=int, default=0, help="When --list-popular: only tracks with at least this many awards")
    ap.add_argument("--min-replays", type=int, default=0, help="When --list-popular: only tracks with at least this many replays")
    ap.add_argument("--max-tracks", type=int, default=0, help="When --list-popular: stop after this many tracks (0 = no limit)")
    ap.add_argument("--download-replays", action="store_true", help="When --list-popular: after listing, download top replays per track")
    ap.add_argument("--replays-dir", type=Path, default=Path("replays_tmnf"), help="Directory for replays when --list-popular --download-replays")
    ap.add_argument("--replays-per-track", type=int, default=20, help="Max replays per track when --list-popular --download-replays")
    ap.add_argument("--download-tracks", action="store_true", help="When --list-popular: after listing, download track .gbx files")
    ap.add_argument("--tracks-dir", type=Path, default=Path("tracks_tmnf"), help="Directory for track .gbx when --list-popular --download-tracks or --extract-tracks-from-replays")
    ap.add_argument("--extract-tracks-from-replays", action="store_true", help="Extract embedded map from first replay per track into tracks-dir (with --list-popular or --track-ids; requires pygbx)")
    ap.add_argument("--save-json", type=Path, default=None, help="When --list-popular: also save track metadata to this JSON file")
    args = ap.parse_args()

    if args.list_popular:
        run_list_popular(args)
        return

    use_tqdm = TQDM_AVAILABLE and not args.no_tqdm
    track_ids: list[int] = []

    if args.track_id is not None:
        track_ids = [args.track_id]
        log.info("Single track: id=%s", args.track_id)
    elif args.track_name:
        log.info("Searching tracks by name: %s", args.track_name)
        results = search_tracks(args.track_name, limit=5)
        time.sleep(args.rate_delay)
        if not results:
            log.error("No tracks found for query: %s", args.track_name)
            return
        track_ids = [int(r["TrackId"]) for r in results]
        log.info("Found %d track(s): %s", len(track_ids), [r.get("TrackName", r.get("TrackId")) for r in results])
    elif args.track_ids and args.track_ids.exists():
        track_ids = []
        for line in args.track_ids.read_text().splitlines():
            line = line.strip()
            if line and line.isdigit():
                track_ids.append(int(line))
        log.info("Loaded %d track IDs from %s", len(track_ids), args.track_ids)

    if not track_ids:
        log.error("Provide --track-id, --track-name, or --track-ids file (or --list-popular).")
        return

    workers = max(1, int(args.workers))

    # Pipeline: --track-ids file + not dry-run => parallel pipeline with resume
    if args.track_ids and args.track_ids.exists() and not args.dry_run:
        api_workers = getattr(args, "api_workers", 0) or workers
        run_pipeline(
            track_ids=track_ids,
            replays_dir=args.replays_dir,
            tracks_dir=args.tracks_dir,
            replays_per_track=args.top,
            offset=args.offset,
            download_workers=workers,
            map_workers=workers,
            extract_maps=getattr(args, "extract_tracks_from_replays", False),
            rate_delay=args.rate_delay,
            use_tqdm=use_tqdm,
            api_workers=api_workers,
        )
        log.info("Done.")
        return

    def _fetch_replay_list(item):
        track_id, out_dir, limit, off = item
        replays = get_track_replays(track_id, limit=limit, offset=off)
        out_list = []
        seen = set()
        for i, rec in enumerate(replays):
            rid = rec.get("ReplayId")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            user = (rec.get("User") or {})
            name = user.get("Name", "unknown")
            time_ms = rec.get("ReplayTime")
            pos = i + 1 + off
            safe_name = safe_filename(f"pos{pos}_{name}_{time_ms}ms.replay.gbx") if time_ms is not None else safe_filename(f"pos{pos}_{name}.replay.gbx")
            out_path = out_dir / str(track_id) / safe_name
            out_list.append((rid, out_path))
        return out_list

    replay_tasks = []
    if args.dry_run:
        for track_id in track_ids:
            replays = get_track_replays(track_id, limit=args.top, offset=args.offset)
            if args.rate_delay > 0:
                time.sleep(args.rate_delay)
            for i, rec in enumerate(replays):
                rid = rec.get("ReplayId")
                if not rid:
                    continue
                if not use_tqdm:
                    log.info("  would download ReplayId=%s for track %s", rid, track_id)
    else:
        fetch_args = [(tid, args.output_dir, args.top, args.offset) for tid in track_ids]
        log.info("Downloading up to %d replays per track (top %d) to %s (workers=%d)", args.top, args.top, args.output_dir, workers)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                if use_tqdm:
                    for task_list in tqdm(ex.map(_fetch_replay_list, fetch_args), total=len(fetch_args), desc="Replay lists", unit="track"):
                        replay_tasks.extend(task_list)
                else:
                    for task_list in ex.map(_fetch_replay_list, fetch_args):
                        replay_tasks.extend(task_list)
        else:
            for track_id in track_ids:
                if not use_tqdm:
                    log.info("Track ID: %s", track_id)
                replay_tasks.extend(_fetch_replay_list((track_id, args.output_dir, args.top, args.offset)))
                if args.rate_delay > 0:
                    time.sleep(args.rate_delay)

    if workers > 1 and replay_tasks:
        def _download_one(item):
            rid, out_path = item
            return (download_replay(rid, out_path, log_fail_once=False), rid)
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            if use_tqdm:
                results = list(tqdm(ex.map(_download_one, replay_tasks), total=len(replay_tasks), desc="Replays", unit="replay"))
            else:
                results = list(ex.map(_download_one, replay_tasks))
            for ok, rid in results:
                if not ok:
                    failed.append(rid)
        if failed:
            log.warning("%d replay(s) failed to download (first ReplayId: %s).", len(failed), failed[0])
    elif replay_tasks:
        first_replay_attempt = True
        for rid, out_path in replay_tasks:
            log_first = first_replay_attempt
            first_replay_attempt = False
            if download_replay(rid, out_path, log_fail_once=log_first) and not use_tqdm:
                log.info("  -> %s", out_path)
            if args.rate_delay > 0:
                time.sleep(args.rate_delay)

    if getattr(args, "extract_tracks_from_replays", False) and not args.dry_run and track_ids:
        tracks_dir = Path(getattr(args, "tracks_dir", Path("tracks_tmnf"))).resolve()
        tracks_dir.mkdir(parents=True, exist_ok=True)
        log.info("Extracting embedded maps from first replay per track into: %s", tracks_dir)
        for track_id in track_ids:
            replay_dir = args.output_dir / str(track_id)
            if not replay_dir.is_dir():
                continue
            replays_list = sorted(replay_dir.glob("*.replay.gbx")) or list(replay_dir.glob("*.gbx"))
            if not replays_list:
                continue
            out_path = tracks_dir / f"{track_id}.Challenge.Gbx"
            if extract_challenge_from_replay(replays_list[0], out_path) and not use_tqdm:
                log.info("  track %s -> %s", track_id, out_path)
        log.info("Finished extracting tracks from replays.")

    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C)")
        sys.exit(130)
