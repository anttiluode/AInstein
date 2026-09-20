# AInstein

> **Can two separately insufficient residues interact so that the combined state exposes a useful operator that neither branch had individually?**

AInstein starts from the constructive-residue question that fell out of OperatorTime: preserve counterfactual branch residues, preserve where they came from, collide them in a nonlinear synthesis step, and ask whether the result is a useful operator unavailable from either branch alone.

The core object is a stamped residue (R, Z). R is the state change left by a branch; Z is provenance: branch, event, role, perspective, or actual/simulated source.

The north-star claim is deliberately stricter than "mix ideas":

**new combination + new operator component + useful held-out behavior + provenance-sensitive composition + no branch-only/interpolation/lookup explanation + interaction discovered rather than scripted.**

## Important transformer correction

A single attention retrieval is a softmax-weighted sum of value vectors, so for fixed values that step is convex. That does **not** make a whole transformer convex-limited. Learned value projections, residual paths, and nonlinear MLPs can construct cross-token interactions after attention.

So AInstein is not built around "transformers can only interpolate." Gate 1 includes a transformer-style nonlinear post-mix control and requires it to work. The question is instead whether residue, provenance, operator novelty, and mechanism discovery can be made explicit and attacked directly.

## Gate 1 — discovered constructive residue synthesis

Each synthetic world contains 12 A items and 12 B items. Every item is individually present in training, but entire A/B pair identities are held out. The hidden useful operator is an unknown orthogonal transform of an ordered bilinear latent feature:

O_ij = Q (a_i tensor b_j)

Observed residues live in independent role-specific coordinate frames and receive noise. Their physical record order is randomized. Provenance stamps identify A versus B.

The synthesis engine is not directly handed the correct operation. It selects on validation data from five candidate families: sum, linear concatenation, Hadamard product, symmetric outer interaction, and ordered outer interaction. It is then frozen and tested on pairings never seen together.

### Attackers

Gate 1 tests branch A alone, branch B alone, the best convex mix of their operator proposals, pair-ID lookup, erased provenance, swapped provenance, and the distance of the synthesized operator from the span of branch-only operator proposals.

It also contains a deliberately strong transformer-style control: role-specific value projections are convexly mixed, followed by a nonlinear random-feature MLP-like stage and a learned readout. That control is allowed to succeed.

### 32-world receipt

| measurement | mean |
|---|---:|
| ordered outer selected | **32 / 32** |
| held-out operator R² | **0.9781** |
| held-out operator application R² | **0.9789** |
| branch A R² | -0.0388 |
| branch B R² | -0.0363 |
| best convex branch mix R² | **0.0116** |
| pair lookup R² | -0.0574 |
| provenance erased R² | **0.2651** |
| provenance shuffled R² | -0.5838 |
| synthesized operator outside branch span | **0.7457 relative residual** |
| transformer-style post-mix control R² | **0.9467** |

All predeclared Gate 1 checks pass. Erasing the A/B source relation drops held-out operator R² by about **0.713**.

The clean result is not "invention solved." It is narrower: in this synthetic family, validation-based mechanism selection finds a reusable cross-residue law that transfers to unseen pairings while branch-only, convex interpolation, and pair lookup controls fail. Provenance is causal, and the synthesized operator has a large component outside the branch-only span.

The successful transformer-style control matters just as much: nonlinear constructive composition is also available to transformer-like machinery. The interesting distinction here is explicit residue/provenance/discovery discipline, not a representational impossibility for transformers.

## What Gate 1 does not establish

This is **not autonomous invention**. The candidate grammar already contains an ordered outer interaction, and the world is bilinear. Gate 1 discovers which supplied mechanism family works; it does not invent a new algebra nobody provided.

It also does not establish a biological mechanism, human creativity, semantic novelty, or superiority over transformers.

## Next gates

**Gate 2 — which residues?** Give the system many preserved stamped residues and a limited compute budget. It must select the pair and interaction worth testing. This is where AnotherOddThing and WhatToLookAt enter.

**Gate 3 — escape the supplied grammar.** Search a small executable operator/program language or grow a compositional circuit, then test on held-out residue families.

**Gate 4 — operator time.** Replace simple A/B role stamps with event, time, perspective, branch, and actual/simulated provenance.

**Gate 5 — useful writeback.** Let the synthesized operator alter future resident state, then require the acquired capability to remain useful after the original residues are gone.

## Run

    python -m pip install -r requirements.txt
    python experiment.py --assert-gate --seeds 32
    python -m unittest discover -s tests -v

The interactive index.html that bootstrapped the repo is intentionally retained as the visual branching-residue sketch. The executable scientific gate now lives beside it.

## Lineage

AInstein is downstream of OperatorTime's Gate 3, GAx/ThirdWay branching, Sihti's residue framing, SelfAndOtherObjectsInTime provenance/event identity, and the active-selection line in AnotherOddThing / WhatToLookAt.

The working question is:

> **Can a system preserve branch-specific residues, keep their source relations intact, discover a useful interaction between them, and thereby make a genuinely new operator available for what happens next?**
