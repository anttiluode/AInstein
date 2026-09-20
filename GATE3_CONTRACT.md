# Gate 3 contract — topology-grown synthesis

Gate 1 chose a complete interaction from a small hand-supplied menu. Gate 2 chose which residue collision deserved a scarce probe. Gate 3 removes the complete-formula menu.

The search receives only a tiny executable substrate:

1. address one coordinate from a provenance-stamped residue bank;
2. multiply the current branch value by one additional coordinate;
3. add discovered branch outputs through a learned linear operator readout.

A branch program is therefore grown from primitives rather than named in advance.

## Synthetic world

Each example has 12 scalar residue coordinates arranged as three provenance groups of four coordinates. The output is a 3 x 3 effective operator.

Every world contains six hidden cross-residue programs with orders:

    2, 2, 3, 3, 4, 4

For example, a hidden fourth-order branch may be equivalent to:

    (((x_1 * x_6) * x_9) * x_10)

but the search is never given that tuple.

Each hidden program contributes along an independent random operator direction. Input values are continuous and independently resampled, so pair/program identity lookup cannot solve the task.

## Growth

Candidate branches are generated recursively from the multiplication primitive up to a maximum allowed order. Search greedily adds the branch whose normalized correlation with the current held-out residual is greatest **per nonlinear composition node**.

A branch of order k costs k - 1 nonlinear nodes.

Every architecture gets the same total structural budget:

    12 nonlinear nodes

This makes the key attacker fair:

- max order 2 may spend all 12 nodes on 12 shallow pairwise branches;
- max order 3 may spend all 12 nodes on shallower/medium branches;
- max order 4 may spend the same 12 nodes on fewer deeper branches.

So success cannot be attributed only to "more nonlinear units."

## Wiring attackers

Two controls preserve gross structural capacity while breaking relation identity.

**Same-shape random wiring** keeps the discovered branch-order histogram but replaces each branch's coordinate addresses with random cross-provenance addresses, then refits the operator readout.

**Shuffled-address wiring** applies one random global coordinate permutation to the discovered topology and refits the readout.

If either succeeds, topology/address is not carrying the result.

## 32-world fixed receipt

| condition | mean held-out operator R² |
|---|---:|
| full growth, max order 4 | **0.9986** |
| same 12-node budget, max order 3 | **0.6459** |
| same 12-node budget, max order 2 | **0.3091** |
| same branch-shape, random wiring | **-0.0118** |
| shuffled discovered addresses | **-0.0030** |

Full growth also reaches **0.9986 application R²**, recovers **6/6 hidden programs in every world**, selects six branch programs, and spends exactly the 12-node structural budget.

## Acceptance rule

Across 32 worlds:

- full-depth operator R² >= 0.97;
- full-depth operator-application R² >= 0.97;
- same-budget depth-2 R² <= 0.45;
- same-budget depth-3 R² between 0.50 and 0.80;
- deep minus shallow R² >= 0.50;
- same-shape random-wiring R² <= 0.10;
- shuffled-address R² <= 0.10;
- hidden-program recovery >= 0.95;
- structural budget is never exceeded.

## What this establishes

Within this deliberately sparse polynomial family, useful operator interactions can be **assembled from elementary primitives** instead of selected as complete formulas. Under an equal nonlinear-node budget, deeper compositional topology unlocks operator components that arbitrarily many shallow pairwise sites cannot represent. Correct wiring/address is causal.

## What this does not establish

This is not grammar-free synthesis in the absolute sense. Multiplication, coordinate addressing and additive readout are still supplied primitives. The hidden worlds are sparse polynomials. Greedy residual search is also an engineered discovery algorithm.

It therefore does not establish human invention, biological dendritic growth, transformer inferiority, or open-ended program synthesis.

## Relation to dendritic morphology

Aizenbud et al. (PNAS, 2026, DOI 10.1073/pnas.2533168123) report that modeled single-neuron functional complexity is much better associated with dendritic area and branch geometry than with bifurcation count alone, and that nonlinear NMDA integration further increases complexity under their model assumptions.

Gate 3 uses that only as an architectural provocation:

> **capacity may depend on how nonlinear interaction sites are arranged, not merely how many exist.**

The numbers are not mapped. In particular AInstein's earlier 0.7457 branch-span residual and Aizenbud's R² = 0.74 dendritic-area correlation are different quantities; their numerical similarity is treated as coincidence, not evidence.

Gate 3 is not a dendrite simulation.
