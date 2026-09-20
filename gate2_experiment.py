#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ainstein.gate2 import SelectionConfig, aggregate_selection, run_selection_world


def main() -> int:
    p = argparse.ArgumentParser(
        description="AInstein Gate 2: choose informative residue collisions under a strict budget."
    )
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=128)
    p.add_argument("--probe-budget", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("results/gate2.json"))
    p.add_argument("--assert-gate", action="store_true")
    args = p.parse_args()

    config = SelectionConfig(trials=args.trials, probe_budget=args.probe_budget)
    worlds = [
        run_selection_world(seed, config)
        for seed in range(args.start_seed, args.start_seed + args.seeds)
    ]
    summary = aggregate_selection(worlds)
    payload = {
        "gate": "Gate 2 — active collision selection",
        "claim_scope": (
            "Synthetic active-selection test. Gate 1's interaction law and a finite family "
            "of possible current needs are assumed known; the active need is hidden."
        ),
        "config": {
            "seeds": args.seeds,
            "trials_per_world": args.trials,
            "probe_budget": args.probe_budget,
            "candidate_pairs": worlds[0]["pair_count"] if worlds else None,
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
