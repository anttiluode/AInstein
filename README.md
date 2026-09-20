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


## Gate 2 — which residues deserve the collision budget?

Gate 1 assumed that the relevant A/B pair had already been presented. Gate 2 makes pair choice itself part of the computation.

Each world now contains **8 A residues × 8 B residues = 64 legal collisions**. The Gate-1 ordered interaction is frozen. Every candidate pair has a predicted operator signature, but the current bottleneck/need is hidden. A real collision probe returns only one noisy bit about how that pair responds to the current need.

Every policy gets the same current need, the same noise tape and exactly **three** probes — only **4.6875%** of the pair space.

The active policy imports the central idea from AnotherOddThing: maintain a posterior over possible current needs and choose the residue pair whose collision has maximum expected information gain. The random control gets three distinct random pair probes. A greedy control spends its probes on the pair that currently looks most useful instead of asking what would reduce uncertainty.

There is also a provenance attacker. It gives the active algorithm the correct response library but attaches those predictions to the **wrong pair addresses**.

### 32-world / 128-trial receipt

| policy | selected utility | oracle-pair hit | posterior entropy |
|---|---:|---:|---:|
| **active information gain** | **0.9531** | **0.8999** | **0.255 bits** |
| random distinct | 0.8460 | 0.1990 | 1.330 bits |
| greedy exploit-only | 0.8041 | 0.0942 | 1.769 bits |
| provenance/address misbound | 0.4762 | 0.0225 | **0.247 bits** |

Active beats random utility by **+0.1070** and greedy by **+0.1489** while examining fewer than 5% of possible collisions. Its oracle-pair hit advantage over random is **+0.7009**.

The strangest result is the misbinding attacker. It becomes almost as *certain* as the correct active system — **0.247 bits** of posterior entropy — while choosing nearly useless pairs.

That gives Gate 2 a useful correction:

> **confidence is not provenance.**

A system can have a sharp posterior over a model whose source/address binding is wrong.

So the emerging chain is now:

    preserve residues
        -> preserve source relation
        -> predict candidate collisions
        -> spend a tiny budget on informative collisions
        -> infer the current bottleneck
        -> choose the useful synthesis

This is the first place where the AnotherOddThing and WhatToLookAt branches become operational parts of AInstein rather than genealogy arrows.

See `GATE2_CONTRACT.md` and `results/gate2.json`.


## Gate 3 — grow the interaction program, not just choose it

Gate 1 still had one major cheat: the correct complete interaction family was already sitting in a menu. Gate 3 removes that menu.

The synthesis substrate now exposes only three elementary capabilities:

    address one residue coordinate
    multiply a branch by one more addressed coordinate
    add discovered branch outputs through a learned operator readout

A useful interaction program therefore has to be assembled from primitive steps.

Each synthetic world contains **three provenance groups × four coordinates = 12 residue coordinates** and asks for a **3×3 effective operator**. The hidden operator is generated by six unknown cross-residue programs with orders:

    2, 2, 3, 3, 4, 4

The search is never told those coordinate tuples. It grows candidate branches by residual reduction per nonlinear composition node.

The important constraint is that every architecture receives the **same total budget of 12 nonlinear nodes**. A shallow system may spend all 12 on twelve pairwise interactions. A deeper system may spend the same 12 nodes building fewer longer interaction chains.

### 32-world receipt

| topology | held-out operator R² |
|---|---:|
| **grown programs, depth/order ≤ 4** | **0.9986** |
| same 12-node budget, depth/order ≤ 3 | **0.6459** |
| same 12-node budget, depth/order ≤ 2 | **0.3091** |
| same discovered branch-shape, random wiring | **-0.0118** |
| discovered topology with shuffled addresses | **-0.0030** |

Full growth reaches **0.9986 operator-application R²**, recovers all **6/6 hidden programs in every world**, and uses exactly the same 12-node budget as the shallow controls.

This gives AInstein a different kind of result from Gate 1:

> **the complete useful interaction no longer has to exist as a named formula in the tray.**

The system still has a primitive language — address, multiply, add — so this is not "grammar free" in an absolute sense. But the winning fourth-order interaction programs are now constructed by search rather than selected wholesale.

The wiring attacks matter as much as the depth result. Keeping branch counts and branch depths while changing **which residue addresses meet** destroys performance. Once again, provenance/address is computational rather than decorative.

### Why the dendrite paper is relevant — and why the matching numbers are not

Aizenbud et al. (PNAS 2026) report that modeled neuronal functional complexity is associated much more strongly with dendritic area and branch geometry than with branch count alone, and that nonlinear NMDA integration adds complexity under their biophysical assumptions.

Gate 3 takes only the architectural provocation:

    number of nonlinear sites is not enough
                +
    arrangement / depth of those sites matters

It does **not** map AInstein's structure onto real dendritic morphology. And the earlier AInstein branch-span residual **0.7457** versus Aizenbud's dendritic-area **R² = 0.74** is explicitly treated as a numerical coincidence: they are different quantities.

See `GATE3_CONTRACT.md` and `results/gate3.json`.


## Exploratory site — temporal foveation and reflection

`temporal_foveation.html` is a deliberately non-gated mechanism sketch built from the autobiographical-memory discussion that followed Gate 3.

The working idea is:

    unexpected change
        -> decayed stamped residue
        -> current context re-addresses it
        -> nonlinear retrieval knee
        -> partial past-state reinstatement
        -> present/past interference
        -> reflection operator

The page treats memory retrieval as **temporal foveation**: fine detail is not continuously instantiated, but a low-bandwidth event-state can sharpen when current context, relation and affect jointly cross a knee.

The important distinction is that the old state does not replace the current state. Both remain available:

    O_reflect = H(S_now, a * S_past)

so the interesting computation is a relation between states — "current self observing past self" — rather than replay alone.

The whorl-like field on the page is an explicit visual analogy to the older `ArtificialCortex/the_whorl` line. That work showed, in its own toy field, that structured circular coupling can make a spiral state easier to revisit/pin. Here the whorl is **not** asserted as a biological autobiographical-memory mechanism. It is only a visual reminder that topology can change which latent states are easy to revisit.

This sketch also sharpens a possible future gate: a stored residue should earn its keep by changing **what happens next** — prediction, sensing allocation, or operator choice — before the expected event occurs. Mere reconstruction of an old scene would not be enough.

## Next gates

**Gate 4 — operator time.** Replace simple A/B role stamps with event, time, perspective, branch, and actual/simulated provenance, and ask whether collision value depends on the trajectory that produced the residue.

**Gate 5 — beyond multiplicative grammar.** Gate 3 still supplies multiplication as the compositional primitive. The next synthesis gate should search across primitive families or grow a small executable circuit whose useful nonlinearity is not known in advance.

**Gate 6 — useful writeback.** Let the synthesized operator alter future resident state, then require the acquired capability to remain useful after the original residues are gone.

**Gate 7 — open-ended collision allocation.** Remove the finite eight-hypothesis need family used by Gate 2 and ask whether uncertainty can be represented and reduced without a closed candidate list.

## Run

    python -m pip install -r requirements.txt
    python experiment.py --assert-gate --seeds 32
    python gate2_experiment.py --assert-gate --seeds 32 --trials 128
    python gate3_experiment.py --assert-gate --seeds 32
    python -m unittest discover -s tests -v

The interactive index.html that bootstrapped the repo is intentionally retained as the visual branching-residue sketch. The executable scientific gate now lives beside it.

## Lineage

AInstein is downstream of OperatorTime's Gate 3, GAx/ThirdWay branching, Sihti's residue framing, SelfAndOtherObjectsInTime provenance/event identity, and the active-selection line in AnotherOddThing / WhatToLookAt.

The working question is:

> **Can a system preserve branch-specific residues, keep their source relations intact, discover a useful interaction between them, and thereby make a genuinely new operator available for what happens next?**
