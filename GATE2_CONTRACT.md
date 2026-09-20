# Gate 2 contract — active collision selection

Gate 2 assumes Gate 1's ordered interaction law is already available. It asks a different question:

> **When 64 legal residue collisions are available and only three can be tried, can the system choose collisions that reveal which operator is useful now?**

This is a synthetic active-experiment-design gate. It is intentionally closer to AnotherOddThing / WhatToLookAt than to Gate 1's representation-learning problem.

## World

Each world contains 8 A residues and 8 B residues, giving 64 legal pairings.

Most residues cluster around one common latent mode and a few occupy rarer modes. This creates redundancy without hard-coding a winning pair. Gate 1's ordered interaction is frozen and generates a candidate operator for every pair. A random diagnostic projection gives every candidate operator a normalized three-dimensional probe signature.

The current bottleneck is hidden. It is one of the eight sign corners in three dimensions.

A collision probe returns only one noisy bit:

    event = 1[ need dot signature(pair) + noise > 0 ]

The system knows the finite family of possible needs and its candidate response model, but not which need is active.

## Matched-budget policies

All policies get the same hidden need, the same Gaussian noise tape, and exactly three pair probes.

- **active information gain** — choose the pair with maximum expected posterior-entropy reduction;
- **random distinct** — choose three random pairs without replacement;
- **greedy exploitation** — choose the pair with highest current expected utility, without valuing information;
- **misbound provenance** — run the active policy with the correct response library attached to the wrong pair addresses.

After three probes, every policy chooses the pair with maximum posterior-expected utility.

Utility is normalized cosine alignment between the current need and the selected synthesized-operator signature. An oracle sees the hidden need and chooses the best of all 64 pairs.

## Fixed Gate 2 acceptance rule

The committed 32-world / 128-trial receipt is considered a pass only if all hold:

- active mean selected utility >= 0.93;
- active - random utility >= 0.08;
- active - greedy utility >= 0.10;
- active oracle-pair hit rate >= 0.80;
- active - random oracle-hit rate >= 0.50;
- active posterior entropy <= 0.50 bits;
- active - misbound utility >= 0.30;
- only <= 5% of the 64 candidate pairs are probed.

This contract is a falsifiable engineering/scientific gate, not a claim of prospective preregistration. The implementation geometry and thresholds were sanity-checked while the gate was being built.

## Scope fence

A pass means only that a correct finite candidate-response model can use active experiment design to allocate a tiny collision budget better than matched random or exploit-only policies in this structured residue bank.

It does **not** mean:

- the system invented the candidate operators;
- real cognition has an eight-state bottleneck;
- a finite hypothesis family is realistic for open-ended creativity;
- information gain automatically solves search over arbitrary residue sets.

The misbinding attacker is especially important: low posterior entropy is not enough. The source/address relation has to be correct.
