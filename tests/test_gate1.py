from __future__ import annotations
import unittest
from ainstein.core import aggregate, make_world, run_world

class AInsteinGate1Tests(unittest.TestCase):
    def test_pair_split_is_by_pair_not_repeat(self):
        world = make_world(0)
        train = {x.pair for x in world.train}
        val = {x.pair for x in world.validation}
        test = {x.pair for x in world.test}
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_small_receipt_has_constructive_heldout_synthesis(self):
        summary = aggregate([run_world(seed) for seed in range(4)])
        self.assertTrue(summary["gate_pass"])
        self.assertGreater(summary["mean_metrics"]["joint_heldout_operator_r2"], 0.9)
        self.assertLess(summary["strongest_branch_convex_or_lookup_r2"], 0.1)

    def test_provenance_is_causal(self):
        summary = aggregate([run_world(seed) for seed in range(4)])
        self.assertGreater(summary["provenance_erased_drop"], 0.4)
        self.assertGreater(summary["mean_metrics"]["joint_span_residual"], 0.5)

    def test_attention_convexity_is_not_a_transformer_impossibility_claim(self):
        metrics = aggregate([run_world(seed) for seed in range(4)])["mean_metrics"]
        self.assertLess(metrics["convex_branch_mix_r2"], 0.2)
        self.assertGreater(metrics["postmix_mlp_control_r2"], 0.8)

if __name__ == "__main__":
    unittest.main()
