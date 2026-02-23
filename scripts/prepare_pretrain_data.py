"""
Preprocess raw TrackMania replay frames into a fast-read NumPy cache.

Reads raw frame images from ``--data-dir`` (``maps/img/<track>/<replay>/*.jpg``)
and writes a pre-processed cache to ``--output-dir``:

  output-dir/
    train.npy        — (N_train, n_stack, 1, H, W) float32, C-contiguous
    val.npy          — (N_val,   n_stack, 1, H, W) float32  (absent when --val-fraction 0)
    cache_meta.json  — parameters + source fingerprint for cache validation

The cache can then be used by setting ``preprocess_cache_dir`` in
``config_files/pretrain_config.yaml`` (or via ``--preprocess-cache-dir`` on
the pretrain script).

Training will **automatically** build the cache when ``preprocess_cache_dir``
is set and the cache is missing or stale.  Use this script when you want to
pre-warm the cache **before** training (e.g. on a machine with faster disk
access, or to separate the CPU-heavy preprocessing from GPU training).

Cache validation
----------------
A cache is reused when all of the following match the current training config:

  * source_data_dir  — absolute path to the raw frame directory
  * image_size       — square resolution in pixels
  * n_stack          — temporal stack depth
  * val_fraction     — fraction of tracks reserved for validation
  * seed             — RNG seed for the track split
  * source_signature — {n_tracks, n_replays, n_frame_files} of the raw data
                       (changes when replays are added or removed)

Any mismatch causes a rebuild.

Usage examples
--------------
# Minimal: defaults match pretrain_config.yaml defaults.
python scripts/prepare_pretrain_data.py \\
    --data-dir maps/img --output-dir cache/pretrain_64

# Custom resolution and stack:
python scripts/prepare_pretrain_data.py \\
    --data-dir maps/img --output-dir cache/pretrain_128_n4 \\
    --image-size 128 --n-stack 4 --val-fraction 0.1 --seed 42

# Skip val split:
python scripts/prepare_pretrain_data.py \\
    --data-dir maps/img --output-dir cache/pretrain_64_noval \\
    --val-fraction 0

# Parallel frame loading (SSD, many small files):
python scripts/prepare_pretrain_data.py \\
    --data-dir maps/img --output-dir cache/pretrain_64 \\
    --workers 4

# Force rebuild even if a valid cache already exists:
python scripts/prepare_pretrain_data.py \\
    --data-dir maps/img --output-dir cache/pretrain_64 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Preprocess raw replay frames into a fast-read NumPy cache "
            "for Level 0 visual pretraining."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "--data-dir", type=Path, required=True, dest="data_dir",
        metavar="PATH",
        help="Root of the raw frame tree  (maps/img/<track_id>/<replay_name>/).",
    )
    ap.add_argument(
        "--output-dir", type=Path, required=True, dest="output_dir",
        metavar="PATH",
        help="Directory to write cache files (train.npy, val.npy, cache_meta.json).",
    )
    ap.add_argument(
        "--image-size", type=int, default=64, dest="image_size",
        metavar="PX",
        help="Target square resolution in pixels.  Must match the IQN image_size.",
    )
    ap.add_argument(
        "--n-stack", type=int, default=1, dest="n_stack",
        metavar="N",
        help="Number of consecutive frames per temporal-stack sample.",
    )
    ap.add_argument(
        "--val-fraction", type=float, default=0.1, dest="val_fraction",
        metavar="F",
        help=(
            "Fraction of track IDs reserved for validation (track-level split). "
            "0 = no validation split (val.npy is not written)."
        ),
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the deterministic track-level train/val split.",
    )
    ap.add_argument(
        "--workers", type=int, default=0,
        help=(
            "Number of threads for parallel frame loading.  "
            "0 = single-threaded (safe default).  Increase on fast SSDs."
        ),
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rebuild the cache even if a valid one already exists.",
    )

    return ap


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from trackmania_rl.pretrain.preprocess import is_cache_valid, build_cache

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir

    if not data_dir.exists():
        log.error("data_dir does not exist: %s", data_dir)
        sys.exit(1)

    if not args.force and is_cache_valid(
        output_dir,
        data_dir,
        args.image_size,
        args.n_stack,
        args.val_fraction,
        args.seed,
    ):
        log.info(
            "Cache is already valid and up-to-date at %s  "
            "(use --force to rebuild).",
            output_dir,
        )
        return

    build_cache(
        data_dir=data_dir,
        cache_dir=output_dir,
        image_size=args.image_size,
        n_stack=args.n_stack,
        val_fraction=args.val_fraction,
        seed=args.seed,
        workers=args.workers,
    )

    log.info(
        "Done.  Set  preprocess_cache_dir: %s  in pretrain_config.yaml "
        "to use this cache for training.",
        output_dir,
    )


if __name__ == "__main__":
    main()
