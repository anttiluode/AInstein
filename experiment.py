#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ainstein import aggregate, run_world

def main() -> int:
    p = argparse.ArgumentParser(description="AInstein Gate 1: discovered constructive residue synthesis.")
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("results/gate1.json"))
    p.add_argument("--assert-gate", action="store_true")
    args = p.parse_args()

    worlds = [run_world(seed) for seed in range(args.start_seed, args.start_seed + args.seeds)]
    summary = aggregate(worlds)
    payload = {
        "gate": "Gate 1 — discovered constructive residue synthesis",
        "claim_scope": "Synthetic mechanism-discovery test only; not evidence of human or transformer invention.",
        "worlds": worlds,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.assert_gate and not summary["gate_pass"]:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
