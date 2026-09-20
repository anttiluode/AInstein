from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import erf, log2, sqrt

import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    n: int = 8
    d: int = 4
    descriptor_noise: float = 0.18
    response_scale: float = 1.6
    sigma: float = 0.35
    probe_budget: int = 3
    trials: int = 128


def _orth(rng: np.random.Generator, d: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    s = np.sign(np.diag(r))
    s[s == 0] = 1
    return q * s


def make_selection_world(seed: int, config: SelectionConfig = SelectionConfig()) -> dict[str, np.ndarray]:
    if config.n != 8 or config.d != 4:
        raise ValueError("Gate 2 currently fixes n=8 and d=4 so the preregistered residue-bank geometry is stable")

    rng = np.random.default_rng(seed)
    # Most residues live near one common mode; rare residues occupy other modes.
    # This creates a realistic allocation problem: many legal collisions are redundant.
    codes = np.array([0, 0, 0, 1, 0, 0, 2, 3], dtype=int)
    ua = _orth(rng, config.d)
    ub = _orth(rng, config.d)
    q = _orth(rng, config.d * config.d)

    a_bank = []
    b_bank = []
    for k in codes:
        x = ua[:, k] + config.descriptor_noise * rng.normal(size=config.d)
        a_bank.append(x / (np.linalg.norm(x) + 1e-12))
    for k in codes[::-1]:
        x = ub[:, k] + config.descriptor_noise * rng.normal(size=config.d)
        b_bank.append(x / (np.linalg.norm(x) + 1e-12))

    a_bank = np.asarray(a_bank)
    b_bank = np.asarray(b_bank)

    # Gate 1's ordered interaction is treated as frozen. Every A/B pair has a
    # predicted operator, but the current hidden bottleneck is not known.
    operators = np.asarray([
        q @ np.kron(a, b)
        for a in a_bank
        for b in b_bank
    ])

    # A cheap 3-D probe signature stands in for what the frozen synthesis law
    # predicts about each pair before we spend a real collision on the current need.
    diagnostic = rng.normal(size=(3, config.d * config.d))
    signatures = operators @ diagnostic.T
    signatures /= np.linalg.norm(signatures, axis=1, keepdims=True) + 1e-12

    needs = np.asarray(list(product([-1.0, 1.0], repeat=3)), dtype=float)
    needs /= np.linalg.norm(needs, axis=1, keepdims=True)

    return {
        "a_bank": a_bank,
        "b_bank": b_bank,
        "operators": operators,
        "signatures": signatures,
        "needs": needs,
    }


def entropy_bits(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(float(x) / sqrt(2.0)))


def response_model(world: dict[str, np.ndarray], config: SelectionConfig) -> tuple[np.ndarray, np.ndarray]:
    signatures = world["signatures"]
    needs = world["needs"]
    means = config.response_scale * (needs @ signatures.T)
    probabilities = np.vectorize(lambda x: _normal_cdf(x / config.sigma))(means)
    return means, probabilities


def posterior_update(prior: np.ndarray, p_on: np.ndarray, event: int) -> np.ndarray:
    likelihood = p_on if int(event) == 1 else (1.0 - p_on)
    posterior = np.asarray(prior, dtype=float) * likelihood
    mass = float(posterior.sum())
    if mass <= 0.0:
        raise ValueError("observation has zero probability under every candidate need")
    return posterior / mass


def expected_information_gain(prior: np.ndarray, p_on: np.ndarray) -> float:
    probability_on = float(np.asarray(prior) @ np.asarray(p_on))
    expected_entropy = 0.0
    if probability_on > 1e-15:
        expected_entropy += probability_on * entropy_bits(posterior_update(prior, p_on, 1))
    if probability_on < 1.0 - 1e-15:
        expected_entropy += (1.0 - probability_on) * entropy_bits(posterior_update(prior, p_on, 0))
    return entropy_bits(prior) - expected_entropy


def utility_table(world: dict[str, np.ndarray]) -> np.ndarray:
    # 1 means the pair's synthesized operator signature is perfectly aligned
    # with the current need; 0 means perfectly anti-aligned.
    return (world["needs"] @ world["signatures"].T + 1.0) / 2.0


def _choose_active(prior: np.ndarray, probabilities: np.ndarray, available: list[int], model_perm: np.ndarray) -> int:
    return max(
        available,
        key=lambda pair: (expected_information_gain(prior, probabilities[:, model_perm[pair]]), -pair),
    )


def _choose_greedy(
    prior: np.ndarray,
    utility: np.ndarray,
    available: list[int],
    model_perm: np.ndarray,
) -> int:
    return max(
        available,
        key=lambda pair: (float(prior @ utility[:, model_perm[pair]]), -pair),
    )


def run_policy(
    world: dict[str, np.ndarray],
    *,
    true_need: int,
    noise_tape: np.ndarray,
    config: SelectionConfig,
    policy: str,
    random_order: np.ndarray | None = None,
    model_perm: np.ndarray | None = None,
) -> dict[str, object]:
    means, probabilities = response_model(world, config)
    utility = utility_table(world)
    pair_count = world["signatures"].shape[0]
    if model_perm is None:
        model_perm = np.arange(pair_count)

    prior = np.full(world["needs"].shape[0], 1.0 / world["needs"].shape[0], dtype=float)
    available = list(range(pair_count))
    probes: list[int] = []

    for round_index in range(config.probe_budget):
        if policy == "active":
            pair = _choose_active(prior, probabilities, available, model_perm)
        elif policy == "random":
            if random_order is None:
                raise ValueError("random policy needs a matched random pair order")
            pair = int(random_order[round_index])
        elif policy == "greedy":
            pair = _choose_greedy(prior, utility, available, model_perm)
        else:
            raise ValueError(f"unknown policy: {policy}")

        analog = float(means[true_need, pair]) + config.sigma * float(noise_tape[round_index])
        event = int(analog > 0.0)
        prior = posterior_update(prior, probabilities[:, model_perm[pair]], event)
        available.remove(pair)
        probes.append(pair)

    expected_utility = np.asarray([
        prior @ utility[:, model_perm[pair]]
        for pair in range(pair_count)
    ])
    selected = int(np.argmax(expected_utility))
    oracle = int(np.argmax(utility[true_need]))

    return {
        "selected_pair": selected,
        "oracle_pair": oracle,
        "selected_utility": float(utility[true_need, selected]),
        "oracle_utility": float(utility[true_need, oracle]),
        "oracle_hit": int(selected == oracle),
        "posterior_entropy_bits": entropy_bits(prior),
        "probes": probes,
    }


def run_selection_world(seed: int, config: SelectionConfig = SelectionConfig()) -> dict[str, object]:
    world = make_selection_world(seed, config)
    pair_count = int(world["signatures"].shape[0])
    rng = np.random.default_rng(50_000 + seed)

    true_needs = rng.integers(0, world["needs"].shape[0], size=config.trials)
    noise_tapes = rng.standard_normal((config.trials, config.probe_budget))
    random_orders = np.asarray([
        rng.permutation(pair_count)[: config.probe_budget]
        for _ in range(config.trials)
    ])
    # Misbinding attacker: the candidate-response library is right, but attached
    # to the wrong pair addresses. This tests whether provenance/address integrity
    # is causal rather than decorative.
    misbound_perm = rng.permutation(pair_count)

    records: dict[str, list[dict[str, object]]] = {
        "active": [],
        "random": [],
        "greedy": [],
        "misbound": [],
    }

    for trial in range(config.trials):
        common = dict(
            world=world,
            true_need=int(true_needs[trial]),
            noise_tape=noise_tapes[trial],
            config=config,
        )
        records["active"].append(run_policy(**common, policy="active"))
        records["random"].append(
            run_policy(**common, policy="random", random_order=random_orders[trial])
        )
        records["greedy"].append(run_policy(**common, policy="greedy"))
        records["misbound"].append(
            run_policy(**common, policy="active", model_perm=misbound_perm)
        )

    result: dict[str, object] = {
        "seed": int(seed),
        "pair_count": pair_count,
        "probe_budget": int(config.probe_budget),
        "trial_count": int(config.trials),
    }
    for policy, items in records.items():
        result[policy] = {
            "mean_selected_utility": float(np.mean([x["selected_utility"] for x in items])),
            "mean_regret": float(np.mean([
                float(x["oracle_utility"]) - float(x["selected_utility"])
                for x in items
            ])),
            "oracle_hit_rate": float(np.mean([x["oracle_hit"] for x in items])),
            "mean_posterior_entropy_bits": float(np.mean([
                x["posterior_entropy_bits"] for x in items
            ])),
        }
    return result


def aggregate_selection(worlds: list[dict[str, object]]) -> dict[str, object]:
    policies = ("active", "random", "greedy", "misbound")
    mean: dict[str, dict[str, float]] = {}
    for policy in policies:
        metric_names = worlds[0][policy].keys()
        mean[policy] = {
            name: float(np.mean([world[policy][name] for world in worlds]))
            for name in metric_names
        }

    deltas = {
        "active_minus_random_utility":
            mean["active"]["mean_selected_utility"] - mean["random"]["mean_selected_utility"],
        "active_minus_greedy_utility":
            mean["active"]["mean_selected_utility"] - mean["greedy"]["mean_selected_utility"],
        "active_minus_misbound_utility":
            mean["active"]["mean_selected_utility"] - mean["misbound"]["mean_selected_utility"],
        "active_minus_random_oracle_hit":
            mean["active"]["oracle_hit_rate"] - mean["random"]["oracle_hit_rate"],
        "active_entropy_reduction_vs_random":
            mean["random"]["mean_posterior_entropy_bits"] - mean["active"]["mean_posterior_entropy_bits"],
        "probe_fraction":
            float(worlds[0]["probe_budget"]) / float(worlds[0]["pair_count"]),
    }

    checks = {
        "active_utility_ge_0_93":
            mean["active"]["mean_selected_utility"] >= 0.93,
        "active_minus_random_utility_ge_0_08":
            deltas["active_minus_random_utility"] >= 0.08,
        "active_minus_greedy_utility_ge_0_10":
            deltas["active_minus_greedy_utility"] >= 0.10,
        "active_oracle_hit_ge_0_80":
            mean["active"]["oracle_hit_rate"] >= 0.80,
        "active_minus_random_oracle_hit_ge_0_50":
            deltas["active_minus_random_oracle_hit"] >= 0.50,
        "active_entropy_le_0_50_bits":
            mean["active"]["mean_posterior_entropy_bits"] <= 0.50,
        "active_minus_misbound_utility_ge_0_30":
            deltas["active_minus_misbound_utility"] >= 0.30,
        "probe_fraction_le_0_05":
            deltas["probe_fraction"] <= 0.05,
    }

    return {
        "worlds": int(len(worlds)),
        "policies": mean,
        "deltas": deltas,
        "gate_checks": checks,
        "gate_pass": bool(all(checks.values())),
    }
