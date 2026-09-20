#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ainstein.gate3 import GrowthConfig, aggregate_growth, run_growth_world


def main() -> int:
    p = argparse.ArgumentParser(
        description="AInstein Gate 3: topology-grown operator synthesis."
    )
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("results/gate3.json"))
    p.add_argument("--assert-gate", action="store_true")
    args = p.parse_args()

    config = GrowthConfig()
    worlds = [
        run_growth_world(seed, config)
        for seed in range(args.start_seed, args.start_seed + args.seeds)
    ]
    summary = aggregate_growth(worlds)
    payload = {
        "gate": "Gate 3 — topology-grown synthesis",
        "claim_scope": (
            "Synthetic program-growth test. The primitive language still supplies coordinate "
            "readout, multiplication and addition; the complete useful interaction programs "
            "are not listed to the search."
        ),
        "config": {
            "seeds": args.seeds,
            "residue_coordinates": config.d,
            "provenance_groups": 3,
            "operator_side": config.operator_side,
            "structural_budget": config.structural_budget,
            "hidden_orders": [2, 2, 3, 3, 4, 4],
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
