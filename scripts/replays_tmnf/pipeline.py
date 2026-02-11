"""
Пайплайн загрузки реплеев и карт по списку треков: три конвейера с очередями и возобновлением.

1. API workers (N потоков): по очереди берут индекс трека, один запрос к API (get_track_replays),
   кладут задачи в download_queue. Раньше был один producer — узкое место; теперь N параллельных запросов к API.
2. Download workers (N потоков): забирают реплеи из очереди и качают в replays_dir/track_id/.
   Одновременно заполняют map_queue для извлечения карт.
3. Map workers (M потоков): забирают (track_id, путь_реплея) из очереди, извлекают карту в tracks_dir.

Суммарная обработка: producer по одному треку кладёт в очередь до top реплеев; воркеры обрабатывают
параллельно. В прогресс-баре: треки — из producer, replays/maps — счётчики из воркеров (обновляются
раз в секунду в главном потоке). При включённом tqdm периодические логи по трекам отключены,
чтобы строка бара не затиралась.

Прогресс: replays_dir/.replay_progress — индекс следующего трека. При перезапуске продолжаем с него.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from .api import (
    download_replay,
    extract_challenge_from_replay,
    get_track_replays,
    safe_filename,
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

PROGRESS_FILENAME = ".replay_progress"
log = logging.getLogger(__name__)


def _load_progress(progress_path: Path) -> int:
    if not progress_path.exists():
        return 0
    try:
        return int(progress_path.read_text().strip())
    except (ValueError, OSError):
        return 0


def _save_progress(progress_path: Path, index: int) -> None:
    progress_path.write_text(str(index), encoding="utf-8")


def _track_has_any_gbx(track_dir: Path) -> bool:
    """Проверка «есть ли хотя бы один .gbx» без полного glob — один проход, стоп на первом."""
    try:
        for p in track_dir.iterdir():
            if p.name.endswith(".gbx"):
                return True
        return False
    except OSError:
        return False


def _count_replays_in_dir(replays_dir: Path) -> int:
    """Подсчёт уже скачанных реплеев (по .gbx в подкаталогах track_id)."""
    count = 0
    try:
        for track_dir in replays_dir.iterdir():
            if not track_dir.is_dir() or track_dir.name == PROGRESS_FILENAME:
                continue
            for p in track_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".gbx":
                    count += 1
    except OSError:
        pass
    return count


def _count_maps_in_dir(tracks_dir: Path) -> int:
    """Подсчёт уже извлечённых карт (.gbx в tracks_dir)."""
    count = 0
    try:
        for p in tracks_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".gbx":
                count += 1
    except OSError:
        pass
    return count


def run_pipeline(
    track_ids: list[int],
    replays_dir: Path,
    tracks_dir: Path,
    replays_per_track: int,
    offset: int,
    download_workers: int,
    map_workers: int,
    extract_maps: bool,
    rate_delay: float,
    use_tqdm: bool = True,
    api_workers: int = 1,
) -> None:
    replays_dir = replays_dir.resolve()
    tracks_dir = tracks_dir.resolve()
    replays_dir.mkdir(parents=True, exist_ok=True)
    if extract_maps:
        tracks_dir.mkdir(parents=True, exist_ok=True)
    progress_path = replays_dir / PROGRESS_FILENAME
    start_index = min(_load_progress(progress_path), len(track_ids))

    if not track_ids or start_index >= len(track_ids):
        log.info("Nothing to do (tracks=%d, start_index=%d).", len(track_ids), start_index)
        return

    use_tqdm = use_tqdm and TQDM_AVAILABLE
    shutdown_event = threading.Event()
    # Отдельные блокировки, чтобы download/map воркеры не конкурировали за один lock
    tracks_lock = threading.Lock()
    replays_lock = threading.Lock()
    maps_lock = threading.Lock()
    # При возобновлении подставляем уже скачанные реплеи и карты в счётчики
    initial_replays = _count_replays_in_dir(replays_dir) if start_index > 0 else 0
    initial_maps = _count_maps_in_dir(tracks_dir) if (start_index > 0 and extract_maps) else 0
    stats = {"tracks": start_index, "replays": initial_replays, "maps": initial_maps}

    api_workers = max(1, api_workers)
    log.info(
        "Pipeline started: %d tracks, replays_dir=%s, api_workers=%d, download_workers=%d, map_workers=%d, extract_maps=%s",
        len(track_ids), replays_dir, api_workers, download_workers, map_workers, extract_maps,
    )

    download_queue: Queue[tuple[int | None, Path | None, int | None]] = Queue(maxsize=10000)
    map_queue: Queue[tuple[int | None, Path | None]] = Queue(maxsize=5000) if extract_maps else None

    # Shared state for API workers
    next_index = start_index
    index_lock = threading.Lock()
    completed_indices: set[int] = set(range(0, start_index))
    progress_lock = threading.Lock()
    last_saved_progress = start_index

    pbar = None
    if use_tqdm and tqdm is not None:
        # stdout — бар виден в том же потоке, что и логи; disable=False чтобы показывать и при перенаправлении
        pbar = tqdm(
            total=len(track_ids),
            initial=start_index,
            desc="Pipeline",
            unit="track",
            dynamic_ncols=True,
            mininterval=0.3,
            file=sys.stdout,
            disable=False,
        )

    def api_worker() -> None:
        nonlocal next_index, last_saved_progress
        while True:
            if shutdown_event.is_set():
                break
            with index_lock:
                i = next_index
                if i >= len(track_ids):
                    break
                next_index = i + 1
            track_id = track_ids[i]
            track_dir = replays_dir / str(track_id)
            if track_dir.exists() and _track_has_any_gbx(track_dir):
                with tracks_lock:
                    stats["tracks"] = max(stats["tracks"], i + 1)
                continue
            replays = get_track_replays(track_id, limit=replays_per_track, offset=offset)
            if shutdown_event.is_set():
                break
            if rate_delay > 0:
                time.sleep(rate_delay)
            seen: set[int] = set()
            for j, rec in enumerate(replays):
                rid = rec.get("ReplayId")
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                user = rec.get("User") or {}
                name = user.get("Name", "unknown")
                time_ms = rec.get("ReplayTime")
                pos = j + 1 + offset
                safe_name = (
                    safe_filename(f"pos{pos}_{name}_{time_ms}ms.replay.gbx")
                    if time_ms is not None
                    else safe_filename(f"pos{pos}_{name}.replay.gbx")
                )
                out_path = track_dir / safe_name
                if not out_path.exists():
                    download_queue.put((rid, out_path, track_id))
            with tracks_lock:
                stats["tracks"] = max(stats["tracks"], i + 1)
            with progress_lock:
                completed_indices.add(i)
                while last_saved_progress < len(track_ids) and last_saved_progress in completed_indices:
                    last_saved_progress += 1
                if last_saved_progress % 50 == 0:
                    _save_progress(progress_path, last_saved_progress)

    def download_worker() -> None:
        while True:
            if shutdown_event.is_set():
                break
            try:
                item = download_queue.get(timeout=1)
            except Empty:
                continue
            rid, out_path, track_id = item
            if rid is None:
                break
            if out_path.exists():
                continue
            if download_replay(rid, out_path, log_fail_once=False):
                with replays_lock:
                    stats["replays"] += 1
                if map_queue is not None and track_id is not None:
                    map_queue.put((track_id, out_path))

    def map_worker() -> None:
        while True:
            if shutdown_event.is_set():
                break
            try:
                item = map_queue.get(timeout=1)
            except Empty:
                continue
            track_id, replay_path = item
            if track_id is None:
                break
            try:
                map_path = tracks_dir / f"{track_id}.Challenge.Gbx"
                if map_path.exists():
                    continue
                if extract_challenge_from_replay(replay_path, map_path):
                    with maps_lock:
                        stats["maps"] += 1
            except Exception:
                # pygbx и др. могут выбросить (например UTF-8 в реплее) — пропускаем этот реплей, воркер не падает
                continue

    if start_index > 0 and pbar is None:
        log.info("Resuming from track index %d (of %d)", start_index, len(track_ids))
    api_threads = [threading.Thread(target=api_worker, name=f"api-{k}") for k in range(api_workers)]
    for t in api_threads:
        t.start()
    down_threads = [threading.Thread(target=download_worker, name=f"download-{k}") for k in range(download_workers)]
    for t in down_threads:
        t.start()
    map_threads: list[threading.Thread] = []
    if extract_maps and map_queue is not None:
        map_threads = [threading.Thread(target=map_worker, name=f"map-{k}") for k in range(map_workers)]
        for t in map_threads:
            t.start()

    def api_still_running() -> bool:
        return any(t.is_alive() for t in api_threads)

    try:
        while api_still_running():
            for t in api_threads:
                t.join(timeout=0.5)
            with tracks_lock:
                t = stats["tracks"]
            with replays_lock:
                r = stats["replays"]
            with maps_lock:
                m = stats["maps"]
            if pbar is not None:
                pbar.n = t
                pbar.set_postfix(replays=r, maps=m, refresh=False)
                pbar.refresh()
            else:
                # Резерв: без tqdm выводим одну обновляемую строку
                sys.stdout.write(f"\rPipeline: {t}/{len(track_ids)} tracks | replays={r} maps={m}   ")
                sys.stdout.flush()
        with progress_lock:
            _save_progress(progress_path, last_saved_progress)
        if pbar is None:
            log.info("API workers finished. Sending shutdown to %d download workers", download_workers)
        for _ in range(download_workers):
            try:
                download_queue.put((None, None, None), timeout=5)
            except Exception:
                break
        for t in down_threads:
            t.join(timeout=2)
        while any(t.is_alive() for t in down_threads):
            for t in down_threads:
                t.join(timeout=1)
        if map_queue is not None:
            for _ in range(map_workers):
                try:
                    map_queue.put((None, None), timeout=2)
                except Exception:
                    break
        for t in map_threads:
            t.join(timeout=2)
        while any(t.is_alive() for t in map_threads):
            for t in map_threads:
                t.join(timeout=1)
        if pbar is None:
            sys.stdout.write("\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        log.info("Interrupt requested (Ctrl+C), shutting down...")
        shutdown_event.set()
        for _ in range(download_workers):
            try:
                download_queue.put((None, None, None), timeout=2)
            except Exception:
                break
        for t in api_threads:
            t.join(timeout=5)
        if api_still_running():
            log.warning("Some API workers still in request; download workers will exit when queue drains.")
        for t in down_threads:
            t.join(timeout=5)
        if map_queue is not None:
            for _ in range(map_workers):
                try:
                    map_queue.put((None, None), timeout=2)
                except Exception:
                    break
        for t in map_threads:
            t.join(timeout=5)
        if pbar is not None:
            pbar.close()
        else:
            sys.stdout.write("\n")
            sys.stdout.flush()
        raise

    if pbar is not None:
        pbar.close()
    else:
        sys.stdout.write("\n")
        sys.stdout.flush()
    log.info("Pipeline finished. Progress saved to %s", progress_path)
