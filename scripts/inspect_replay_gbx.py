"""
Inspect .replay.gbx structure: sample_period, control_entries, control_names, etc.
Usage: python scripts/inspect_replay_gbx.py <path-to-replay.gbx>
       python scripts/inspect_replay_gbx.py maps/replays/1000074/*.replay.gbx
"""
from __future__ import annotations

import sys
from pathlib import Path

_script_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_script_root))

def inspect(replay_path: Path) -> None:
    from pygbx import Gbx, GbxType

    gbx = Gbx(str(replay_path))
    ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
    if not ghosts:
        print("No ghosts found")
        return

    for idx, ghost in enumerate(ghosts):
        print(f"\n=== Ghost {idx} ===")
        print(f"  race_time: {getattr(ghost, 'race_time', '?')}")
        print(f"  sample_period: {getattr(ghost, 'sample_period', '?')}")
        print(f"  uid: {getattr(ghost, 'uid', getattr(ghost, 'UID', '?'))}")
        print(f"  game_version: {getattr(ghost, 'game_version', '?')}")
        print(f"  control_names: {list(getattr(ghost, 'control_names', []) or [])}")

        control_entries = list(getattr(ghost, "control_entries", []) or [])
        print(f"  control_entries: {len(control_entries)} total")

        # Show time distribution
        times = [getattr(ce, "time", getattr(ce, "Time", None)) for ce in control_entries]
        times = [t for t in times if t is not None]
        if times:
            print(f"  time range: {min(times)} .. {max(times)}")
            # Check time deltas
            sorted_t = sorted(set(times))
            deltas = [sorted_t[i+1] - sorted_t[i] for i in range(len(sorted_t)-1)]
            if deltas:
                print(f"  min time delta: {min(deltas)}, max: {max(deltas)}")
                from collections import Counter
                delta_counts = Counter(deltas)
                print(f"  most common deltas: {delta_counts.most_common(5)}")

        # First 20 entries
        print("\n  First 20 control entries:")
        for i, ce in enumerate(control_entries[:20]):
            t = getattr(ce, "time", getattr(ce, "Time", "?"))
            en = getattr(ce, "event_name", getattr(ce, "EventName", "?"))
            enabled = getattr(ce, "enabled", getattr(ce, "Enabled", "?"))
            flags = getattr(ce, "flags", getattr(ce, "Flags", "?"))
            print(f"    [{i}] time={t} event_name={en!r} enabled={enabled} flags={flags}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_replay_gbx.py <replay.gbx> [replay2.gbx ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_file():
            print(f"\n{'='*60}\nFile: {p}")
            inspect(p)
        else:
            # support glob like maps/replays/1000074/*.replay.gbx
            candidates = list(p.parent.glob(p.name)) if "*" in arg else [p]
            for f in candidates:
                if f.exists() and f.is_file() and ".replay" in f.name.lower():
                    print(f"\n{'='*60}\nFile: {f}")
                    inspect(f)


if __name__ == "__main__":
    main()
