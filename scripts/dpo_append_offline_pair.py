"""Append one line to a DPO offline JSONL (chosen/rejected joblib paths).

Each joblib file must store ``tuple[rollout_results_dict, end_race_stats_dict]`` as produced by
saving the two objects from the learner/collector pipeline.

Example:
  python scripts/dpo_append_offline_pair.py path/chosen.joblib path/rejected.joblib data/dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("chosen_joblib", type=Path)
    p.add_argument("rejected_joblib", type=Path)
    p.add_argument("out_jsonl", type=Path)
    args = p.parse_args()
    line = json.dumps(
        {
            "chosen": str(args.chosen_joblib.resolve()),
            "rejected": str(args.rejected_joblib.resolve()),
        }
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"Appended 1 line to {args.out_jsonl}")


if __name__ == "__main__":
    main()
