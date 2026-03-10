"""
Convert rulka maps/img + replays + vcp to Hugging Face dataset format.

Output: Parquet shards (data/train-*.parquet, data/val-*.parquet),
replays/, vcp/, track_index.json, README.md.

Usage:
  python scripts/dataset/convert_to_hf_dataset.py --output-dir hf_dataset
  python scripts/dataset/convert_to_hf_dataset.py --data-dir maps/img --output-dir hf_dataset --no-vcp
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from trackmania_rl.pretrain.datasets import split_track_ids
from trackmania_rl.pretrain.preprocess import _load_replay_manifest_and_timeline

log = logging.getLogger(__name__)


def _default_workers() -> int:
    """Default number of workers: cpu_count - 1, at least 1."""
    n = multiprocessing.cpu_count() or 1
    return max(1, n - 1)

VCP_DISTANCE = "0.5"
VCP_SUFFIX = "cl"


def _scan_one_replay(
    replay_dir: Path,
    track_id: str,
    replay_name: str,
    require_action_idx: bool,
) -> tuple[Path, str, str, list[dict], dict] | None:
    """Process one replay dir. Return (replay_dir, track_id, replay_name, entries, metadata) or None if skipped."""
    manifest_path = replay_dir / "manifest.json"
    meta_path = replay_dir / "metadata.json"
    if not manifest_path.exists():
        return None

    entries, _, _, _ = _load_replay_manifest_and_timeline(replay_dir)
    if not entries:
        return None

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    metadata.setdefault("track_id", track_id)
    metadata.setdefault("replay_name", replay_name)

    if require_action_idx:
        entries = [e for e in entries if e.get("action_idx") is not None]
        if not entries:
            return None

    return (replay_dir, track_id, replay_name, entries, metadata)


def scan_data_dir(
    data_dir: Path,
    require_action_idx: bool = False,
    workers: int = 1,
) -> list[tuple[Path, str, str, list[dict], dict]]:
    """Scan data_dir for replay dirs, load manifest+metadata. Return list of (replay_dir, track_id, replay_name, entries, metadata)."""
    data_dir = Path(data_dir)
    tasks: list[tuple[Path, str, str]] = []

    for track_dir in sorted(data_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        track_id = track_dir.name
        for replay_dir in sorted(track_dir.iterdir()):
            if replay_dir.is_dir():
                tasks.append((replay_dir, track_id, replay_dir.name))

    if not tasks:
        return []

    result: list[tuple[Path, str, str, list[dict], dict]] = []
    if workers <= 1:
        for replay_dir, track_id, replay_name in tasks:
            item = _scan_one_replay(replay_dir, track_id, replay_name, require_action_idx)
            if item is not None:
                result.append(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_scan_one_replay, rd, tid, rn, require_action_idx): (rd, tid, rn)
                for rd, tid, rn in tasks
            }
            for future in as_completed(futures):
                item = future.result()
                if item is not None:
                    result.append(item)
    result.sort(key=lambda x: (x[1], x[2]))  # sort by track_id, replay_name
    return result


def build_flat_index(
    scan_result: list[tuple[Path, str, str, list[dict], dict]],
    train_ids: set[str],
    val_ids: set[str],
) -> tuple[list[tuple[Path, dict, dict]], list[tuple[Path, dict, dict]]]:
    """Build flat index (replay_dir, entry, metadata) for train and val."""
    train_index: list[tuple[Path, dict, dict]] = []
    val_index: list[tuple[Path, dict, dict]] = []

    for replay_dir, track_id, replay_name, entries, metadata in scan_result:
        for entry in entries:
            item = (replay_dir, entry, metadata)
            if track_id in train_ids:
                train_index.append(item)
            elif track_id in val_ids:
                val_index.append(item)

    return train_index, val_index


def row_generator(
    index: list[tuple[Path, dict, dict]],
) -> Iterator[dict[str, Any]]:
    """Yield rows for Dataset: image bytes + metadata."""
    for replay_dir, entry, metadata in index:
        fname = entry.get("file")
        if not fname:
            continue
        img_path = replay_dir / fname
        if not img_path.exists():
            log.warning("Missing frame: %s", img_path)
            continue

        try:
            img_bytes = img_path.read_bytes()
        except OSError as e:
            log.warning("Cannot read %s: %s", img_path, e)
            continue

        inputs = entry.get("inputs") or {}
        action_idx = entry.get("action_idx")
        if action_idx is not None:
            action_idx = int(action_idx)

        row: dict[str, Any] = {
            "image": {"bytes": img_bytes},
            "track_id": str(metadata.get("track_id", "")),
            "replay_name": str(metadata.get("replay_name", "")),
            "step": int(entry.get("step", 0)),
            "time_ms": int(entry.get("time_ms", 0)),
            "action_idx": action_idx,
            "accelerate": bool(inputs.get("accelerate", False)),
            "brake": bool(inputs.get("brake", False)),
            "left": bool(inputs.get("left", False)),
            "right": bool(inputs.get("right", False)),
        }
        fps = metadata.get("fps")
        if fps is not None:
            row["fps"] = float(fps)
        else:
            row["fps"] = None
        width = metadata.get("width")
        height = metadata.get("height")
        row["width"] = int(width) if width is not None else None
        row["height"] = int(height) if height is not None else None
        row["race_finished"] = bool(metadata.get("race_finished", False))
        yield row


def _copy_one_file(src: Path, dst: Path, use_symlink: bool) -> None:
    """Copy or symlink one file."""
    if use_symlink:
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_replays_and_vcp(
    index: list[tuple[Path, dict, dict]],
    replays_dir: Path,
    vcp_dir: Path | None,
    output_dir: Path,
    use_symlink: bool = False,
    workers: int = 1,
) -> dict[str, dict[str, Any]]:
    """Copy replays and vcp to output_dir. Return track_index."""
    output_replays = output_dir / "replays"
    output_vcp = output_dir / "vcp"

    seen_tracks: dict[str, set[str]] = {}
    for replay_dir, _entry, metadata in index:
        tid = metadata.get("track_id", replay_dir.parent.name)
        rname = metadata.get("replay_name", replay_dir.name)
        if tid not in seen_tracks:
            seen_tracks[tid] = set()
        seen_tracks[tid].add(rname)

    copy_tasks: list[tuple[Path, Path]] = []
    for track_id, replay_names in seen_tracks.items():
        src_track = replays_dir / track_id
        dst_track = output_replays / track_id
        dst_track.mkdir(parents=True, exist_ok=True)
        for replay_name in replay_names:
            src = src_track / f"{replay_name}.gbx"
            if src.exists():
                copy_tasks.append((src, dst_track / f"{replay_name}.gbx"))

    if workers <= 1:
        for src, dst in copy_tasks:
            _copy_one_file(src, dst, use_symlink)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda t: _copy_one_file(t[0], t[1], use_symlink), copy_tasks))

    vcp_suffix = f"{VCP_DISTANCE}m_{VCP_SUFFIX}"
    track_index: dict[str, dict[str, Any]] = {}
    vcp_tasks: list[tuple[Path, Path]] = []
    for track_id in seen_tracks:
        track_index[track_id] = {
            "replays": sorted(seen_tracks[track_id]),
            "has_vcp": False,
        }
        if vcp_dir:
            vcp_file = vcp_dir / f"{track_id}_{vcp_suffix}.npy"
            if vcp_file.exists():
                track_index[track_id]["has_vcp"] = True
                output_vcp.mkdir(parents=True, exist_ok=True)
                vcp_tasks.append((vcp_file, output_vcp / vcp_file.name))

    if vcp_tasks:
        if workers <= 1:
            for src, dst in vcp_tasks:
                _copy_one_file(src, dst, use_symlink)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda t: _copy_one_file(t[0], t[1], use_symlink), vcp_tasks))

    return track_index


def generate_readme(
    output_dir: Path,
    n_train: int,
    n_val: int,
    n_tracks: int,
    has_val_split: bool = True,
    repo_id: str = "username/rulka-tmnf-raw-v1",
) -> None:
    """Generate README.md Dataset Card for Hugging Face Hub."""
    data_files_yaml = "      - split: train\n        path: data/train-*\n"
    if has_val_split:
        data_files_yaml += "      - split: validation\n        path: data/val-*\n"
    readme = f"""---
license: cc-by-4.0
pretty_name: TrackMania Nations Forever Replay Frames
task_categories:
  - image-to-image
  - other
tags:
  - trackmania
  - racing
  - reinforcement-learning
  - behavioral-cloning
configs:
  - config_name: default
    data_files:
{data_files_yaml}
---

# TrackMania Nations Forever — Replay Frames Dataset

Frames captured from TMNF replays for RL/BC research. Each frame has associated
game inputs (accelerate, brake, steer left/right) and metadata.

## Dataset Summary

This dataset contains ~{n_train + n_val:,} frames extracted from {n_tracks:,} TrackMania Nations Forever
replays. Intended for behavioral cloning, reinforcement learning, and imitation learning research.
Source replays are from TMNF-X (ManiaExchange); frames were captured using rulka (TMInterface + capture_replays_tmnf).

## Data Source

- **Replays:** TMNF-X / ManiaExchange
- **Game:** TrackMania Nations Forever (Ubisoft/Nadeo)
- **Capture pipeline:** rulka (TMInterface + capture_replays_tmnf)

## Dataset Structure

- **data/train-*.parquet**, **data/val-*.parquet** — frames with embedded JPEG bytes + metadata
- **replays/<track_id>/*.gbx** — source replay files (one per captured replay)
- **vcp/<track_id>_0.5m_cl.npy** — waypoint trajectories (one per track)
- **track_index.json** — mapping track_id → replays, has_vcp

## Data Fields (per row)

| Column | Type | Description |
|--------|------|-------------|
| image | Image | JPEG bytes |
| track_id | string | TMNF-X track ID |
| replay_name | string | Replay filename stem (e.g. pos10_Dolby_42800ms.replay) |
| step | int | Frame index |
| time_ms | int | Simulation time (ms) |
| action_idx | int | Discrete action ID (0–8) |
| accelerate | bool | Gas pressed |
| brake | bool | Brake pressed |
| left | bool | Steer left |
| right | bool | Steer right |
| fps | float | Capture FPS |
| width, height | int | Frame resolution |
| race_finished | bool | Replay finished the race |

## Usage

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "{repo_id}"

ds = load_dataset(REPO_ID)
row = ds["train"][0]
# row["image"] is PIL Image
# row["track_id"], row["replay_name"], row["action_idx"], etc.

# Download replay for a sample
replay_path = hf_hub_download(
    REPO_ID,
    f"replays/{{row['track_id']}}/{{row['replay_name']}}.gbx",
    repo_type="dataset",
)

# Download VCP for a track
vcp_path = hf_hub_download(
    REPO_ID,
    f"vcp/{{row['track_id']}}_0.5m_cl.npy",
    repo_type="dataset",
)
```

## Considerations for Using the Data

- **Bias:** Replays reflect TMNF-X community distribution (popular tracks, competitive times). Training on this data may inherit these biases.
- **Limitations:** Frames are captured at varying FPS; action labels come from replay input parsing.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). See [LICENSE](LICENSE).

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{{rulka-tmnf-frames,
  title = {{TrackMania Nations Forever Replay Frames Dataset}},
  author = {{rulka contributors}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/datasets/{repo_id}}},
}}
```

## Statistics

| Split | Samples |
|-------|---------|
| train | {n_train:,} |
| validation | {n_val:,} |
| **Total tracks** | {n_tracks:,} |
"""

    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_license(output_dir: Path) -> None:
    """Write LICENSE file (CC-BY-4.0)."""
    license_text = """Creative Commons Attribution 4.0 International (CC BY 4.0)

This dataset is licensed under the Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit: https://creativecommons.org/licenses/by/4.0/

You are free to:
  - Share — copy and redistribute the material in any medium or format
  - Adapt — remix, transform, and build upon the material for any purpose

Under the following terms:
  - Attribution — You must give appropriate credit

Full legal text: https://creativecommons.org/licenses/by/4.0/legalcode
"""
    (output_dir / "LICENSE").write_text(license_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert rulka maps/img to Hugging Face dataset format (Parquet + replays + vcp).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("maps/img"),
        help="Root of track_id/replay_name/ frame tree",
    )
    parser.add_argument(
        "--replays-dir",
        type=Path,
        default=Path("maps/replays"),
        help="Replays directory (track_id/*.replay.gbx)",
    )
    parser.add_argument(
        "--vcp-dir",
        type=Path,
        default=Path("maps/vcp"),
        help="VCP directory",
    )
    parser.add_argument(
        "--no-vcp",
        action="store_true",
        help="Do not include VCP files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction (track-level split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--max-shard-size-mb",
        type=int,
        default=450,
        help="Target Parquet shard size (MB)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for scan, Dataset build, Parquet write, and copy. Default: cpu_count - 1",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying replays/vcp",
    )
    parser.add_argument(
        "--require-action-idx",
        action="store_true",
        help="Skip frames without action_idx",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="username/rulka-tmnf-raw-v1",
        help="Hugging Face repo id for README examples (e.g. username/dataset-name)",
    )
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else _default_workers()

    logging.basicConfig(level=getattr(logging, args.log_level))

    data_dir = args.data_dir.resolve()
    replays_dir = args.replays_dir.resolve()
    vcp_dir = None if args.no_vcp else args.vcp_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    # Phase 1: scan
    log.info("Scanning %s (workers=%d)...", data_dir, workers)
    scan_result = scan_data_dir(
        data_dir,
        require_action_idx=args.require_action_idx,
        workers=workers,
    )
    if not scan_result:
        log.error("No replay dirs found")
        raise SystemExit(1)

    total_frames = sum(len(entries) for _, _, _, entries, _ in scan_result)
    log.info("Found %d replays, %d total frames", len(scan_result), total_frames)

    # Split by track
    train_ids, val_ids = split_track_ids(data_dir, args.val_fraction, args.seed)
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    train_index, val_index = build_flat_index(scan_result, train_ids_set, val_ids_set)
    log.info("Train: %d frames, Val: %d frames", len(train_index), len(val_index))

    # Phase 2 & 3: Dataset + Parquet
    from datasets import Dataset, Features, Image, Value

    features = Features({
        "image": Image(),
        "track_id": Value("string"),
        "replay_name": Value("string"),
        "step": Value("int64"),
        "time_ms": Value("int64"),
        "action_idx": Value("int64"),
        "accelerate": Value("bool"),
        "brake": Value("bool"),
        "left": Value("bool"),
        "right": Value("bool"),
        "fps": Value("float32"),
        "width": Value("int64"),
        "height": Value("int64"),
        "race_finished": Value("bool"),
    })

    num_proc = workers if workers > 1 else None
    log.info("Building train dataset (num_proc=%s)...", num_proc or 1)
    train_ds = Dataset.from_generator(
        row_generator,
        features=features,
        gen_kwargs={"index": train_index},
        num_proc=num_proc,
    )
    log.info("Writing train Parquet shards (max %s MB each, workers=%d)...", args.max_shard_size_mb, workers)
    rows_per_shard = max(1000, (args.max_shard_size_mb * 1024 * 1024) // 25_000)
    num_train_shards = max(1, (len(train_ds) + rows_per_shard - 1) // rows_per_shard)
    data_dir = output_dir / "data"

    def _write_shard(ds: Any, n_shards: int, i: int, prefix: str) -> None:
        shard = ds.shard(num_shards=n_shards, index=i)
        path = data_dir / f"{prefix}-{i:05d}-of-{n_shards:05d}.parquet"
        shard.to_parquet(str(path))

    if workers <= 1:
        for i in range(num_train_shards):
            _write_shard(train_ds, num_train_shards, i, "train")
    else:
        with ThreadPoolExecutor(max_workers=min(workers, num_train_shards)) as ex:
            list(ex.map(
                lambda i: _write_shard(train_ds, num_train_shards, i, "train"),
                range(num_train_shards),
            ))

    if val_index:
        log.info("Building val dataset (num_proc=%s)...", num_proc or 1)
        val_ds = Dataset.from_generator(
            row_generator,
            features=features,
            gen_kwargs={"index": val_index},
            num_proc=num_proc,
        )
        log.info("Writing val Parquet shards...")
        num_val_shards = max(1, (len(val_ds) + rows_per_shard - 1) // rows_per_shard)
        if workers <= 1:
            for i in range(num_val_shards):
                _write_shard(val_ds, num_val_shards, i, "val")
        else:
            with ThreadPoolExecutor(max_workers=min(workers, num_val_shards)) as ex:
                list(ex.map(
                    lambda i: _write_shard(val_ds, num_val_shards, i, "val"),
                    range(num_val_shards),
                ))

    # Phase 4: replays + vcp
    log.info("Copying replays and vcp (workers=%d)...", workers)
    track_index = copy_replays_and_vcp(
        train_index + val_index,
        replays_dir,
        vcp_dir,
        output_dir,
        use_symlink=args.symlink,
        workers=workers,
    )

    # Track index
    (output_dir / "track_index.json").write_text(
        json.dumps(track_index, indent=2),
        encoding="utf-8",
    )

    # Phase 5: README + LICENSE
    generate_readme(
        output_dir,
        n_train=len(train_index),
        n_val=len(val_index),
        n_tracks=len(track_index),
        has_val_split=len(val_index) > 0,
        repo_id=args.repo_id,
    )
    write_license(output_dir)

    log.info("Done. Output: %s", output_dir)


if __name__ == "__main__":
    main()
