from __future__ import annotations

import unittest

from ainstein.gate3 import GrowthConfig, aggregate_growth, run_growth_world


class AInsteinGate3Tests(unittest.TestCase):
    def test_small_receipt_requires_depth_not_only_site_count(self):
        summary = aggregate_growth([
            run_growth_world(seed, GrowthConfig(random_wiring_repeats=4))
            for seed in range(4)
        ])
        self.assertTrue(summary["gate_pass"])
        metrics = summary["mean_metrics"]
        self.assertGreater(metrics["full_depth_operator_r2"], 0.97)
        self.assertLess(metrics["depth2_same_budget_r2"], 0.45)
        self.assertGreater(
            metrics["full_depth_operator_r2"] - metrics["depth2_same_budget_r2"],
            0.50,
        )

    def test_search_recovers_generated_programs_without_complete_formula_list(self):
        result = run_growth_world(0, GrowthConfig(random_wiring_repeats=4))
        metrics = result["metrics"]
        self.assertEqual(metrics["hidden_program_recovery_fraction"], 1.0)
        self.assertEqual(metrics["selected_program_count"], 6)
        self.assertEqual(metrics["used_structural_cost"], 12)

    def test_same_shape_wrong_wiring_fails(self):
        summary = aggregate_growth([
            run_growth_world(seed, GrowthConfig(random_wiring_repeats=4))
            for seed in range(4)
        ])
        metrics = summary["mean_metrics"]
        self.assertLess(metrics["same_shape_random_wiring_r2"], 0.10)
        self.assertLess(metrics["shuffled_address_wiring_r2"], 0.10)

    def test_gate_is_not_grammar_free_in_the_absolute_sense(self):
        # The scope fence is structural: complete programs are discovered, but
        # coordinate access / multiplication / addition remain supplied primitives.
        config = GrowthConfig()
        self.assertEqual(config.structural_budget, 12)


if __name__ == "__main__":
    unittest.main()
