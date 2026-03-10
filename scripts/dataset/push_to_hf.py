"""
Push a converted Hugging Face dataset (from convert_to_hf_dataset) to the Hub.

Usage:
  hf auth login   # if not already logged in
  python scripts/dataset/push_to_hf.py --local-path hf_dataset --repo-id username/trackmania-tmnf-frames
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path


def _default_workers() -> int:
    """Default workers for upload: cpu_count - 1, at least 1."""
    n = multiprocessing.cpu_count() or 1
    return max(1, n - 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push converted dataset to Hugging Face Hub.",
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        required=True,
        help="Path to dataset directory (output of convert_to_hf_dataset)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo id (e.g. username/dataset-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Upload workers (huggingface_hub upload_large_folder). Default: cpu_count - 1",
    )
    args = parser.parse_args()
    num_workers = args.num_workers if args.num_workers is not None else _default_workers()

    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    from huggingface_hub import HfApi

    api = HfApi()
    local_path = args.local_path.resolve()
    if not local_path.exists():
        raise SystemExit(f"Local path does not exist: {local_path}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Uploading {local_path} to {args.repo_id} (workers={num_workers})...")
    api.upload_large_folder(
        folder_path=str(local_path),
        repo_id=args.repo_id,
        repo_type="dataset",
        num_workers=num_workers,
    )
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
