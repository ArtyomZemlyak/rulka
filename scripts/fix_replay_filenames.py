"""
Rename replay files with non-ASCII characters in maps/replays/.

Replaces any non-ASCII character with '_' to avoid encoding issues
when TMInterface tries to load script files derived from replay names.

Usage:
    python scripts/fix_replay_filenames.py [--dry-run]

    --dry-run   Show what would be renamed without actually renaming.
"""

import os
import sys
from pathlib import Path


def is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def sanitize_filename(name: str) -> str:
    """Replace non-ASCII characters and spaces with '_'."""
    return "".join(c if (ord(c) < 128 and c != " ") else "_" for c in name)


def main():
    dry_run = "--dry-run" in sys.argv

    base = Path(__file__).resolve().parent.parent / "maps" / "replays"
    if not base.exists():
        print(f"ERROR: {base} does not exist")
        sys.exit(1)

    renamed = 0
    skipped = 0
    conflicts = 0

    for track_dir in sorted(base.iterdir()):
        if not track_dir.is_dir():
            continue
        for replay_file in sorted(track_dir.iterdir()):
            if not replay_file.is_file():
                continue
            old_name = replay_file.name
            if is_ascii(old_name) and " " not in old_name:
                continue

            new_name = sanitize_filename(old_name)
            new_path = replay_file.parent / new_name

            if new_path.exists() and new_path != replay_file:
                print(f"  CONFLICT: {track_dir.name}/{old_name} -> {new_name} (target exists, skipping)")
                conflicts += 1
                continue

            if dry_run:
                print(f"  WOULD RENAME: {track_dir.name}/{old_name} -> {new_name}")
            else:
                replay_file.rename(new_path)
                print(f"  RENAMED: {track_dir.name}/{old_name} -> {new_name}")
            renamed += 1

    action = "Would rename" if dry_run else "Renamed"
    print(f"\nDone. {action}: {renamed}, Conflicts: {conflicts}")
    if dry_run and renamed > 0:
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
