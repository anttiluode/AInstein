from __future__ import annotations

import unittest

import numpy as np

from ainstein.gate2 import (
    SelectionConfig,
    aggregate_selection,
    make_selection_world,
    run_selection_world,
)


class AInsteinGate2Tests(unittest.TestCase):
    def test_world_has_many_candidate_collisions(self):
        world = make_selection_world(0)
        self.assertEqual(world["a_bank"].shape, (8, 4))
        self.assertEqual(world["b_bank"].shape, (8, 4))
        self.assertEqual(world["operators"].shape, (64, 16))
        self.assertEqual(world["signatures"].shape, (64, 3))
        self.assertEqual(world["needs"].shape, (8, 3))
        np.testing.assert_allclose(
            np.linalg.norm(world["signatures"], axis=1),
            np.ones(64),
            atol=1e-9,
        )

    def test_small_receipt_active_selection_beats_matched_budget_controls(self):
        config = SelectionConfig(trials=48)
        summary = aggregate_selection([
            run_selection_world(seed, config)
            for seed in range(4)
        ])
        self.assertTrue(summary["gate_pass"])
        self.assertGreater(summary["deltas"]["active_minus_random_utility"], 0.08)
        self.assertGreater(summary["deltas"]["active_minus_greedy_utility"], 0.10)

    def test_misbound_pair_addresses_create_confident_wrong_inference(self):
        config = SelectionConfig(trials=48)
        summary = aggregate_selection([
            run_selection_world(seed, config)
            for seed in range(4)
        ])
        active = summary["policies"]["active"]
        misbound = summary["policies"]["misbound"]
        self.assertLess(misbound["mean_posterior_entropy_bits"], 0.6)
        self.assertGreater(
            active["mean_selected_utility"] - misbound["mean_selected_utility"],
            0.30,
        )

    def test_budget_is_less_than_five_percent_of_pair_space(self):
        result = run_selection_world(0, SelectionConfig(trials=8))
        self.assertLessEqual(result["probe_budget"] / result["pair_count"], 0.05)


if __name__ == "__main__":
    unittest.main()
