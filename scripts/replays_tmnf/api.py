"""
TMNF-X (ManiaExchange) API: track search, replays, download.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "Rulka-TMNF-ReplayDownload/1.0 (TrackMania RL; +https://github.com/your-repo)"
TMNF_BASE = "https://tmnf.exchange"


def _req_json(url: str) -> dict | list:
    h = {"User-Agent": USER_AGENT}
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def search_tracks(query: str, limit: int = 20) -> list[dict]:
    """Search TMNF-X tracks by name. Returns list of {TrackId, TrackName, ...}."""
    fields = "TrackId,TrackName,UId,AuthorTime,Uploader.Name"
    params = urllib.parse.urlencode({"trackname": query, "limit": limit, "fields": fields})
    url = f"{TMNF_BASE}/api/tracks?{params}"
    data = _req_json(url)
    if isinstance(data, dict) and "Results" in data:
        return data["Results"]
    if isinstance(data, list):
        return data
    return []


# Search Tracks (api2.mania.exchange Method 43). Параметры из доки:
# fields (обязателен), order1 (enum порядка), count (кол-во, default 40),
# after (Int64) — результаты после указанного TrackId (курсорная пагинация).
def search_tracks_paginated(
    limit: int = 100,
    after_track_id: int | None = None,
    order1: int = 8,
    mode: int | None = None,
    fields: str = "TrackId,TrackName,UId,AuthorTime,Awards,Uploader.Name,TrackValue,Comments",
) -> tuple[list[dict], bool]:
    """Search TMNF-X tracks. Пагинация: after_track_id = последний TrackId с предыдущей страницы."""
    params = {"fields": fields, "count": limit}
    if order1 >= 0:
        params["order1"] = order1
    if mode is not None:
        params["mode"] = mode
    if after_track_id is not None:
        params["after"] = after_track_id
    url = f"{TMNF_BASE}/api/tracks?{urllib.parse.urlencode(params)}"
    data = _req_json(url)
    results = []
    if isinstance(data, dict) and "Results" in data:
        results = data["Results"]
    elif isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, list):
        results = data
    has_more = isinstance(data, dict) and data.get("More", False)
    return (results if isinstance(results, list) else [], has_more)


def download_track(track_id: int, out_path: Path, uid: str | None = None) -> bool:
    """Download track (map) .gbx from TMNF-X. В доке MX Download Map указан для TM2.MX/TMX/SMX; у TMNF-X публичного API для файла может не быть."""
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": f"{TMNF_BASE}/",
        "Accept": "application/x-gbx,application/octet-stream,*/*",
    }
    # Варианты URL: на TMNF-X кнопка «Download» на странице трека ведёт на /trackgbx/{id} (не /mapgbx как в доке MX для TMX/TM2).
    candidates = [
        f"{TMNF_BASE}/trackgbx/{track_id}",
        f"{TMNF_BASE}/mapgbx/{track_id}",
        f"{TMNF_BASE}/api/mapgbx/{track_id}",
        f"{TMNF_BASE}/mapgbx?id={track_id}",
        f"{TMNF_BASE}/track/download/{track_id}",
        f"{TMNF_BASE}/download/track/{track_id}",
        f"{TMNF_BASE}/tracks/download/{track_id}",
        f"{TMNF_BASE}/maps/download/{track_id}",
        f"{TMNF_BASE}/api/tracks/download/{track_id}",
        f"{TMNF_BASE}/api/tracks/download?id={track_id}",
    ]
    if uid:
        candidates.extend([
            f"{TMNF_BASE}/mapgbx?guid={urllib.parse.quote(uid)}",
            f"{TMNF_BASE}/maps/download/{uid}",
        ])
    tried = []  # (url, code) для лога
    last_body = ""
    for url in candidates:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                if len(data) < 200 and data.startswith(b"<"):
                    tried.append((url, "HTML"))
                    continue  # похоже на HTML (ошибка/логин), не сохраняем
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            try:
                last_body = e.read().decode("utf-8", errors="replace").strip()[:800]
            except Exception:
                pass
            tried.append((url, e.code))
            continue
    import logging
    log = logging.getLogger(__name__)
    parts = [f"{url.replace(TMNF_BASE, '')} -> {c}" for url, c in tried]
    log.warning("Track %s: download failed. Tried: %s", track_id, ", ".join(parts))
    if last_body and "404" not in last_body[:200]:
        log.warning("Response body: %s", last_body[:300])
    return False


def get_track_replays(track_id: int, limit: int = 100, offset: int = 0) -> list[dict]:
    """Get replays for a track (best times first)."""
    fields = "ReplayId,User.UserId,User.Name,ReplayTime,ReplayScore,Position,IsBest,IsLeaderboard"
    params = urllib.parse.urlencode({
        "trackid": track_id,
        "fields": fields,
        "limit": limit,
        "offset": offset,
    })
    url = f"{TMNF_BASE}/api/replays?{params}"
    data = _req_json(url)
    if isinstance(data, dict) and "Results" in data:
        return data["Results"]
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    return []


def download_replay(replay_id: int, out_path: Path, log_fail_once: bool = False) -> bool:
    """Download replay file from TMNF-X. Tries several URL patterns (API may not document replay download)."""
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": f"{TMNF_BASE}/",
        "Accept": "application/x-gbx,application/octet-stream,*/*",
    }
    # На сайте: кнопка скачивания реплея ведёт на /recordgbx/{id} (не /replaygbx/).
    candidates = [
        f"{TMNF_BASE}/recordgbx/{replay_id}",
        f"{TMNF_BASE}/replaygbx/{replay_id}",
        f"{TMNF_BASE}/replays/download/{replay_id}",
        f"{TMNF_BASE}/replay/download/{replay_id}",
        f"{TMNF_BASE}/api/replays/download/{replay_id}",
        f"{TMNF_BASE}/api/replay/download/{replay_id}",
        f"{TMNF_BASE}/replays/{replay_id}/download",
        f"{TMNF_BASE}/replay/show/{replay_id}/download",
    ]
    tried = []
    for url in candidates:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                if len(data) < 200 and data.startswith(b"<"):
                    tried.append((url.replace(TMNF_BASE, ""), "HTML"))
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            tried.append((url.replace(TMNF_BASE, ""), e.code))
    if log_fail_once:
        import logging
        log = logging.getLogger(__name__)
        parts = [f"{p} -> {c}" for p, c in tried]
        log.warning(
            "Replay %s: download failed (tried %s). TMNF-X may not offer public replay-download API.",
            replay_id, ", ".join(parts),
        )
    return False


def safe_filename(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", s)[:200]


# Chunk IDs in replay GBX that contain embedded challenge (map) as raw GBX: size (uint32) + data
_REPLAY_CHALLENGE_CHUNK_IDS = (0x03093002, 0x2403F002)


def extract_challenge_from_replay(replay_path: Path, out_path: Path) -> bool:
    """
    Extract embedded track (challenge) from a TMNF replay file using pygbx.
    Writes raw challenge GBX to out_path. Returns True if extraction succeeded.
    """
    try:
        from pygbx import Gbx
    except ImportError:
        return False
    if not replay_path.is_file():
        return False
    # pygbx при битых/не-UTF-8 строках в реплее логирует ERROR и бросает — приглушаем на время парсинга
    import logging
    _saved = []
    for _name in ("pygbx", "pygbx.gbx"):
        _log = logging.getLogger(_name)
        _saved.append((_log, _log.level))
        _log.setLevel(logging.CRITICAL)
    try:
        gbx = Gbx(str(replay_path))
    except Exception:
        return False
    finally:
        for _log, _lev in _saved:
            _log.setLevel(_lev)
    for chunk_id in _REPLAY_CHALLENGE_CHUNK_IDS:
        br = gbx.find_raw_chunk_id(chunk_id)
        if br is None:
            continue
        try:
            size = br.read_uint32()
            if size <= 0 or size > 50 * 1024 * 1024:
                continue
            data = br.read(size)
            if len(data) != size:
                continue
        except Exception:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return True
    return False
