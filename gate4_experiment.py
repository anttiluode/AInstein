#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ainstein.gate4 import (
    OperatorMemoryConfig,
    aggregate_operator_memory,
    run_operator_memory_world,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="AInstein Gate 4: contextual operator reinstatement and reflection."
    )
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("results/gate4.json"))
    p.add_argument("--assert-gate", action="store_true")
    args = p.parse_args()

    config = OperatorMemoryConfig()
    worlds = [
        run_operator_memory_world(seed, config)
        for seed in range(args.start_seed, args.start_seed + args.seeds)
    ]
    summary = aggregate_operator_memory(worlds)
    payload = {
        "gate": "Gate 4 — contextual operator reinstatement and reflection",
        "claim_scope": (
            "Synthetic operator-memory test. Distinct historical linear operators are "
            "learned and cached with context stamps; the current operator is also learned. "
            "Reflection is derived algebraically from two cached operators, not discovered "
            "as an open-ended cognitive mechanism."
        ),
        "config": {
            "eras": config.eras,
            "operator_dimension": config.d,
            "context_dimension": config.context_dim,
            "query_repeats_per_era": config.query_repeats,
            "few_shot_transfer_pairs": config.few_shot_pairs,
        },
        "worlds": worlds,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if args.assert_gate and not summary["gate_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
