from __future__ import annotations

import unittest

from ainstein.gate4 import (
    OperatorMemoryConfig,
    aggregate_operator_memory,
    run_operator_memory_world,
)


class AInsteinGate4Tests(unittest.TestCase):
    def test_small_receipt_reinstates_old_operator_after_current_changed(self):
        summary = aggregate_operator_memory([
            run_operator_memory_world(seed, OperatorMemoryConfig(query_repeats=4))
            for seed in range(4)
        ])
        self.assertTrue(summary["gate_pass"])
        metrics = summary["mean_metrics"]
        self.assertGreater(metrics["reinstated_old_task_r2"], 0.95)
        self.assertLess(summary["strongest_reinstatement_attacker_r2"], 0.20)

    def test_present_and_past_jointly_define_a_useful_relation_operator(self):
        summary = aggregate_operator_memory([
            run_operator_memory_world(seed, OperatorMemoryConfig(query_repeats=4))
            for seed in range(4)
        ])
        metrics = summary["mean_metrics"]
        self.assertGreater(metrics["reflection_transfer_r2"], 0.95)
        self.assertGreater(
            metrics["reflection_transfer_r2"]
            - summary["strongest_transfer_attacker_r2"],
            0.40,
        )

    def test_wrong_provenance_breaks_reflection(self):
        summary = aggregate_operator_memory([
            run_operator_memory_world(seed, OperatorMemoryConfig(query_repeats=4))
            for seed in range(4)
        ])
        self.assertLess(
            summary["mean_metrics"]["wrong_era_reflection_r2"],
            0.10,
        )

    def test_convex_operator_blend_is_not_operator_relation(self):
        summary = aggregate_operator_memory([
            run_operator_memory_world(seed, OperatorMemoryConfig(query_repeats=4))
            for seed in range(4)
        ])
        self.assertLess(
            summary["mean_metrics"]["best_convex_transfer_r2"],
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
