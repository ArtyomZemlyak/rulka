"""
List popular TMNF-X tracks (paginated). Contains the listing function used by
list_popular_tracks_tmnf.py and by the download script when --list-popular.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .api import (
    download_replay,
    download_track,
    extract_challenge_from_replay,
    get_track_replays,
    safe_filename,
    search_tracks_paginated,
)

log = logging.getLogger(__name__)


def run_list_popular(args: argparse.Namespace) -> None:
    """
    Fetch popular track IDs from TMNF-X (paginated). Stops on first empty page.
    Optionally write to file, save JSON metadata, download tracks, download replays.
    """
    use_tqdm = TQDM_AVAILABLE and not getattr(args, "no_tqdm", False)
    orders_to_try = ["awards", "activity", "uploaded"] if args.order == "all" else [args.order]
    order1_map = {"awards": 8, "activity": 6, "uploaded": 2}  # enum order1 из доки MX

    all_track_ids: list[int] = []
    all_tracks_meta: list[dict] = []
    seen_ids: set[int] = set()

    for order_name in orders_to_try:
        order1 = order1_map[order_name]
        log.info("Fetching tracks from TMNF-X (order=%s, min_awards=%d, min_replays=%d)", order_name, args.min_awards, args.min_replays)
        after_track_id = None
        next_future = None
        with ThreadPoolExecutor(max_workers=1) as prefetch_exec:
            for page in range(args.pages):
                if next_future is not None:
                    try:
                        batch, has_more = next_future.result(timeout=120)
                    except Exception as e:
                        log.warning("Prefetched page failed: %s", e)
                        batch, has_more = [], False
                    next_future = None
                else:
                    batch, has_more = search_tracks_paginated(limit=args.per_page, after_track_id=after_track_id, order1=order1)
                    if not batch and page == 0 and order1 >= 0:
                        batch, has_more = search_tracks_paginated(limit=args.per_page, after_track_id=None, order1=-1)
                if args.rate_delay > 0:
                    time.sleep(args.rate_delay)
                if not batch:
                    log.info("  %s: empty page %d, stopping (no gaps assumed).", order_name, page + 1)
                    break
                if not has_more:
                    log.info("  %s: API reported no more results after page %d.", order_name, page + 1)
                total_before_page = len(all_track_ids)
                for t in batch:
                    tid = t.get("TrackId")
                    if tid is None:
                        continue
                    try:
                        tid = int(tid)
                    except (TypeError, ValueError):
                        continue
                    if tid in seen_ids:
                        continue
                    if args.min_awards > 0:
                        awards = t.get("Awards")
                        if awards is None or int(awards) < args.min_awards:
                            continue
                    if args.min_replays > 0:
                        replay_count = t.get("ReplayCount") or t.get("Replays")
                        if replay_count is not None:
                            if int(replay_count) < args.min_replays:
                                continue
                        else:
                            replays_check = get_track_replays(tid, limit=args.min_replays)
                            if args.rate_delay > 0:
                                time.sleep(args.rate_delay)
                            if len(replays_check) < args.min_replays:
                                continue
                    seen_ids.add(tid)
                    all_track_ids.append(tid)
                    all_tracks_meta.append(t)
                passed_this_page = len(all_track_ids) - total_before_page
                if not use_tqdm:
                    log.info("  %s page %d: %d from API, %d passed (total %d)", order_name, page + 1, len(batch), passed_this_page, len(all_track_ids))
                if args.max_tracks and len(all_track_ids) >= args.max_tracks:
                    all_track_ids = all_track_ids[: args.max_tracks]
                    all_tracks_meta = all_tracks_meta[: args.max_tracks]
                    break
                if passed_this_page == 0 and page > 0:
                    log.info("  %s: page %d returned no new tracks (API likely repeats first page), stopping.", order_name, page + 1)
                    break
                if len(batch) < args.per_page:
                    break
                if not has_more:
                    break
                last = batch[-1]
                tid_last = last.get("TrackId")
                try:
                    after_track_id = int(tid_last) if tid_last is not None else None
                except (TypeError, ValueError):
                    after_track_id = None
                if after_track_id is None:
                    break
                next_future = prefetch_exec.submit(
                    search_tracks_paginated,
                    args.per_page,
                    after_track_id,
                    order1,
                )
        if args.max_tracks and len(all_track_ids) >= args.max_tracks:
            break

    log.info("Total tracks collected: %d", len(all_track_ids))
    if len(all_track_ids) == 0 and args.min_replays > 0:
        log.info("Hint: TMNF-X often returns newest tracks first (many have 0 replays). Try --min-replays 0 or --min-awards 1.")

    if not args.dry_run and all_track_ids:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_abs = args.output.resolve()
        args.output.write_text("\n".join(str(t) for t in all_track_ids) + "\n")
        log.info("Track IDs file: %s (wrote %d IDs)", out_abs, len(all_track_ids))
    save_json = getattr(args, "save_json", None)
    if save_json and all_tracks_meta and not args.dry_run:
        save_json = Path(save_json)
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(all_tracks_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Metadata JSON: %s", save_json.resolve())

    if getattr(args, "download_tracks", False) and all_track_ids and not args.dry_run:
        tracks_dir = Path(getattr(args, "tracks_dir", Path("tracks_tmnf"))).resolve()
        tracks_dir.mkdir(parents=True, exist_ok=True)
        log.info("Tracks (maps) will be saved to: %s", tracks_dir)
        # TMNF-X не отдаёт карты по API (404). Пробуем первый трек; если 404 — один раз предупреждаем и не долбим остальные.
        first_ok = False
        for i, track_id in enumerate(all_track_ids):
            meta = all_tracks_meta[i] if i < len(all_tracks_meta) else {}
            name = meta.get("TrackName", str(track_id))
            safe_name = safe_filename(f"{track_id}_{name}.Challenge.gbx")[:180]
            out_path = tracks_dir / safe_name
            uid = meta.get("UId")
            if download_track(track_id, out_path, uid=uid):
                first_ok = True
                if not use_tqdm:
                    log.info("Track %d/%d: id=%s -> %s", i + 1, len(all_track_ids), track_id, out_path)
                break
            if i == 0:
                log.warning(
                    "TMNF-X does not provide track (map) download API (404). "
                    "Use --download-replays --extract-tracks-from-replays to get maps from replays."
                )
                break
        if first_ok:
            track_iter = enumerate(all_track_ids[1:], start=2)
            if use_tqdm:
                track_iter = tqdm(track_iter, total=len(all_track_ids) - 1, desc="Tracks (maps)", unit="track")
            for j, track_id in track_iter:
                i = j - 1
                meta = all_tracks_meta[i] if i < len(all_tracks_meta) else {}
                name = meta.get("TrackName", str(track_id))
                safe_name = safe_filename(f"{track_id}_{name}.Challenge.gbx")[:180]
                out_path = tracks_dir / safe_name
                uid = meta.get("UId")
                if not use_tqdm:
                    log.info("Track %d/%d: id=%s (map)", j, len(all_track_ids), track_id)
                if download_track(track_id, out_path, uid=uid) and not use_tqdm:
                    log.info("  -> %s", out_path)
                if args.rate_delay > 0:
                    time.sleep(args.rate_delay)
        log.info("Finished downloading tracks.")

    if getattr(args, "download_replays", False) and all_track_ids and not args.dry_run:
        replays_dir = Path(getattr(args, "replays_dir", Path("replays_tmnf"))).resolve()
        replays_dir.mkdir(parents=True, exist_ok=True)
        replays_per_track = getattr(args, "replays_per_track", 20)
        workers = max(1, int(getattr(args, "workers", 1)))
        log.info("Replays will be saved to: %s (subdir per track ID)", replays_dir)
        log.info("Downloading up to %d replays per track (workers=%d)...", replays_per_track, workers)

        def _fetch_replay_list(item):
            track_id, replays_dir, limit = item
            replays = get_track_replays(track_id, limit=limit)
            out_list = []
            for j, rec in enumerate(replays):
                rid = rec.get("ReplayId")
                if not rid:
                    continue
                user = (rec.get("User") or {})
                name = user.get("Name", "unknown")
                time_ms = rec.get("ReplayTime")
                pos = j + 1
                safe_name = safe_filename(f"pos{pos}_{name}_{time_ms}ms.replay.gbx") if time_ms is not None else safe_filename(f"pos{pos}_{name}.replay.gbx")
                out_path = replays_dir / str(track_id) / safe_name
                out_list.append((rid, out_path))
            return out_list

        # Параллельно получаем списки реплеев по трекам, затем собираем в один список
        replay_tasks = []
        fetch_args = [(tid, replays_dir, replays_per_track) for tid in all_track_ids]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                if use_tqdm:
                    for task_list in tqdm(
                        ex.map(_fetch_replay_list, fetch_args),
                        total=len(fetch_args),
                        desc="Replay lists",
                        unit="track",
                    ):
                        replay_tasks.extend(task_list)
                else:
                    for task_list in ex.map(_fetch_replay_list, fetch_args):
                        replay_tasks.extend(task_list)
        else:
            for i, item in enumerate(fetch_args):
                if not use_tqdm:
                    log.info("Track %d/%d: id=%s (fetching replay list)", i + 1, len(all_track_ids), item[0])
                replay_tasks.extend(_fetch_replay_list(item))
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
                download_replay(rid, out_path, log_fail_once=log_first)
                if args.rate_delay > 0:
                    time.sleep(args.rate_delay)
        log.info("Finished downloading replays.")

    if getattr(args, "extract_tracks_from_replays", False) and all_track_ids and not args.dry_run:
        replays_dir = Path(getattr(args, "replays_dir", Path("replays_tmnf"))).resolve()
        tracks_dir = Path(getattr(args, "tracks_dir", Path("tracks_tmnf"))).resolve()
        tracks_dir.mkdir(parents=True, exist_ok=True)
        log.info("Extracting embedded maps from first replay per track into: %s", tracks_dir)
        for i, track_id in enumerate(all_track_ids):
            track_dir = replays_dir / str(track_id)
            if not track_dir.is_dir():
                continue
            replays_list = sorted(track_dir.glob("*.replay.gbx")) or list(track_dir.glob("*.gbx"))
            if not replays_list:
                continue
            meta = all_tracks_meta[i] if i < len(all_tracks_meta) else {}
            name = meta.get("TrackName", str(track_id))
            safe_name = safe_filename(f"{track_id}_{name}.Challenge.Gbx")[:180]
            out_path = tracks_dir / safe_name
            if extract_challenge_from_replay(replays_list[0], out_path):
                if not (TQDM_AVAILABLE and not getattr(args, "no_tqdm", False)):
                    log.info("  track %s -> %s", track_id, out_path)
            else:
                log.debug("  track %s: no challenge in first replay", track_id)
        log.info("Finished extracting tracks from replays.")

    log.info("Done.")
