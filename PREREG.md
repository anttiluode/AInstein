# Gate 1 preregistration — discovered constructive residue synthesis

This document fixes the claim boundary and pass/fail rule for the 32-world receipt.

## Claim

Two individually insufficient provenance-stamped residues can support selection of a reusable cross-residue interaction from a generic candidate grammar, and that interaction can transfer to A/B combinations never seen together during fitting.

This is a **mechanism-discovery** claim, not an autonomous-invention claim.

## World

- 12 A items and 12 B items.
- Pair-level train/validation/test splits; repeats of one pair never cross splits.
- Latent dimension 4; target operator is 4×4.
- Target operator is an unknown orthogonal transform of the ordered bilinear latent feature.
- A and B residues are independently rotated into role-specific observed frames.
- Gaussian residue noise = 0.05.
- Three repeats per pair.
- Physical residue order is randomized.
- Provenance stamps identify A and B.

## Candidate grammar

The same ridge readout is fitted for each candidate and validation operator R² selects one:

1. sum
2. linear concatenation
3. Hadamard product
4. symmetric outer interaction
5. ordered outer interaction

Test data never participates in mechanism selection.

## Attackers

A-only, B-only, best validation-chosen convex interpolation of branch operator proposals, pair-ID lookup, provenance erased, provenance swapped, and branch-span projection.

## Transformer-edge sanity control

A separate control performs a convex equal-weight mix after role-specific value projections:

    0.5 [A, 0] + 0.5 [0, B] = [0.5 A, 0.5 B]

It then applies a nonlinear tanh feature layer plus learned ridge readout.

This control is expected to be capable of solving the interaction. Its purpose is to reject the false inference that because one attention retrieval is convex, a transformer block cannot synthesize nonlinear cross-token features.

## Gate rule

Across 32 worlds, all must hold:

- ordered outer selected in at least 90% of worlds
- mean held-out operator R² at least 0.90
- mean held-out operator-application R² at least 0.90
- maximum mean R² among branch A, branch B, convex branch mix, and pair lookup at most 0.10
- mean joint-operator residual outside the branch-only span at least 0.50
- held-out operator R² minus provenance-erased R² at least 0.40
- transformer-edge nonlinear post-mix control R² at least 0.80
- no pair leakage

The transformer control is a sanity requirement, not a superiority claim.

## Allowed interpretation if passed

Within this synthetic bilinear family, validation-based mechanism selection finds a reusable cross-residue law that generalizes to unseen pairings, and the resulting operator is not explained by either branch alone, convex interpolation, pair lookup, or provenance-free pooling.

Not allowed: "the system invented a new idea", "transformers cannot do this", or "human creativity works this way."
