from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .core import fit, pred, r2


@dataclass(frozen=True)
class GrowthConfig:
    d: int = 12
    operator_side: int = 3
    train_n: int = 1024
    validation_n: int = 512
    test_n: int = 1024
    noise: float = 0.03
    structural_budget: int = 12
    random_wiring_repeats: int = 16


def _candidate_terms(groups: np.ndarray, max_order: int) -> list[tuple[int, ...]]:
    terms: list[tuple[int, ...]] = []
    for order in range(2, max_order + 1):
        for term in combinations(range(len(groups)), order):
            # Every nonlinear branch must combine at least two provenance groups.
            if len(set(groups[list(term)])) >= 2:
                terms.append(term)
    return terms


def _feature_matrix(x: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
    if not terms:
        return np.zeros((len(x), 0), dtype=float)
    return np.column_stack([np.prod(x[:, term], axis=1) for term in terms])


def _standardize_train_apply(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0) + 1e-12
    return ((train - mean) / scale, *[(x - mean) / scale for x in others])


def make_growth_world(seed: int, config: GrowthConfig = GrowthConfig()) -> dict[str, object]:
    if config.d != 12 or config.operator_side != 3:
        raise ValueError("Gate 3 currently fixes 12 residue coordinates and a 3x3 operator")

    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(3), 4)
    hidden_orders = (2, 2, 3, 3, 4, 4)
    by_order = {
        order: [t for t in _candidate_terms(groups, order) if len(t) == order]
        for order in (2, 3, 4)
    }

    hidden: list[tuple[int, ...]] = []
    for order in hidden_orders:
        choices = [term for term in by_order[order] if term not in hidden]
        hidden.append(choices[int(rng.integers(len(choices)))])

    out_dim = config.operator_side ** 2
    q, _ = np.linalg.qr(rng.normal(size=(out_dim, out_dim)))
    coefficient_rows = q[:, : len(hidden)].T

    total = config.train_n + config.validation_n + config.test_n
    x = rng.uniform(-1.0, 1.0, size=(total, config.d))
    hidden_raw = _feature_matrix(x, hidden)

    hidden_train = hidden_raw[: config.train_n]
    hidden_mean = hidden_train.mean(axis=0)
    hidden_scale = hidden_train.std(axis=0) + 1e-12
    hidden_features = (hidden_raw - hidden_mean) / hidden_scale

    y = hidden_features @ coefficient_rows
    y += config.noise * rng.normal(size=y.shape)

    a = config.train_n
    b = a + config.validation_n
    return {
        "groups": groups,
        "hidden_programs": tuple(hidden),
        "hidden_orders": hidden_orders,
        "train_x": x[:a],
        "train_y": y[:a],
        "validation_x": x[a:b],
        "validation_y": y[a:b],
        "test_x": x[b:],
        "test_y": y[b:],
    }


def grow_programs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    groups: np.ndarray,
    *,
    max_order: int,
    structural_budget: int,
) -> dict[str, object]:
    candidates = _candidate_terms(groups, max_order)
    train_raw = _feature_matrix(train_x, candidates)
    validation_raw = _feature_matrix(validation_x, candidates)
    train_features, validation_features = _standardize_train_apply(
        train_raw, validation_raw
    )

    selected_indices: list[int] = []
    selected_programs: list[tuple[int, ...]] = []
    used: set[int] = set()
    remaining = int(structural_budget)
    prediction = np.repeat(
        train_y.mean(axis=0, keepdims=True), len(train_y), axis=0
    )
    growth_curve: list[dict[str, object]] = []

    while True:
        residual = train_y - prediction
        correlation = train_features.T @ residual
        numerator = np.sum(correlation * correlation, axis=1)
        denominator = np.sum(train_features * train_features, axis=0) + 1e-12
        scores = numerator / denominator

        feasible: list[tuple[float, int]] = []
        for index, program in enumerate(candidates):
            cost = len(program) - 1
            if index not in used and cost <= remaining:
                # Reward predictive residual reduction per nonlinear composition node.
                feasible.append((float(scores[index]) / cost, index))

        if not feasible:
            break

        _, index = max(feasible)
        program = candidates[index]
        cost = len(program) - 1
        used.add(index)
        selected_indices.append(index)
        selected_programs.append(program)
        remaining -= cost

        current_train = train_features[:, selected_indices]
        current_validation = validation_features[:, selected_indices]
        weights = fit(current_train, train_y)
        prediction = pred(current_train, weights)
        growth_curve.append(
            {
                "structural_cost": structural_budget - remaining,
                "validation_r2": r2(validation_y, pred(current_validation, weights)),
                "program": list(program),
                "program_order": len(program),
            }
        )

    return {
        "programs": tuple(selected_programs),
        "growth_curve": growth_curve,
        "structural_cost": structural_budget - remaining,
    }


def _fit_programs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    programs: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
) -> np.ndarray:
    train_raw = _feature_matrix(train_x, list(programs))
    test_raw = _feature_matrix(test_x, list(programs))
    train_features, test_features = _standardize_train_apply(train_raw, test_raw)
    return pred(test_features, fit(train_features, train_y))


def _random_programs_with_orders(
    rng: np.random.Generator,
    groups: np.ndarray,
    orders: list[int],
    *,
    forbidden: set[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    selected: list[tuple[int, ...]] = []
    for order in orders:
        pool = [
            term
            for term in _candidate_terms(groups, order)
            if len(term) == order
            and term not in forbidden
            and term not in selected
        ]
        selected.append(pool[int(rng.integers(len(pool)))])
    return tuple(selected)


def run_growth_world(seed: int, config: GrowthConfig = GrowthConfig()) -> dict[str, object]:
    world = make_growth_world(seed, config)
    groups = np.asarray(world["groups"])
    train_x = np.asarray(world["train_x"])
    train_y = np.asarray(world["train_y"])
    validation_x = np.asarray(world["validation_x"])
    validation_y = np.asarray(world["validation_y"])
    test_x = np.asarray(world["test_x"])
    test_y = np.asarray(world["test_y"])
    hidden = tuple(world["hidden_programs"])

    grown: dict[int, dict[str, object]] = {}
    predictions: dict[int, np.ndarray] = {}
    for max_order in (2, 3, 4):
        result = grow_programs(
            train_x,
            train_y,
            validation_x,
            validation_y,
            groups,
            max_order=max_order,
            structural_budget=config.structural_budget,
        )
        grown[max_order] = result
        predictions[max_order] = _fit_programs(
            train_x, train_y, test_x, result["programs"]
        )

    full_programs = tuple(grown[4]["programs"])
    full_prediction = predictions[4]

    rng = np.random.default_rng(100_000 + seed)
    probes = rng.normal(size=(len(test_y), config.operator_side))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True) + 1e-12
    true_application = np.einsum(
        "nij,nj->ni",
        test_y.reshape(-1, config.operator_side, config.operator_side),
        probes,
    )
    predicted_application = np.einsum(
        "nij,nj->ni",
        full_prediction.reshape(-1, config.operator_side, config.operator_side),
        probes,
    )

    order_histogram = [len(program) for program in full_programs]
    random_scores: list[float] = []
    for _ in range(config.random_wiring_repeats):
        random_programs = _random_programs_with_orders(
            rng,
            groups,
            order_histogram,
            forbidden=set(hidden),
        )
        random_prediction = _fit_programs(
            train_x, train_y, test_x, random_programs
        )
        random_scores.append(r2(test_y, random_prediction))

    permutation = rng.permutation(config.d)
    shuffled_programs = tuple(
        tuple(sorted(int(permutation[i]) for i in program))
        for program in full_programs
    )
    shuffled_prediction = _fit_programs(
        train_x, train_y, test_x, shuffled_programs
    )

    hidden_recovery = len(set(hidden) & set(full_programs)) / len(hidden)

    metrics = {
        "full_depth_operator_r2": r2(test_y, full_prediction),
        "full_depth_application_r2": r2(
            true_application, predicted_application
        ),
        "depth2_same_budget_r2": r2(test_y, predictions[2]),
        "depth3_same_budget_r2": r2(test_y, predictions[3]),
        "same_shape_random_wiring_r2": float(np.mean(random_scores)),
        "shuffled_address_wiring_r2": r2(test_y, shuffled_prediction),
        "hidden_program_recovery_fraction": float(hidden_recovery),
        "selected_program_count": int(len(full_programs)),
        "used_structural_cost": int(grown[4]["structural_cost"]),
    }

    return {
        "seed": int(seed),
        "hidden_programs": [list(x) for x in hidden],
        "selected_programs": [list(x) for x in full_programs],
        "metrics": metrics,
        "growth_curve": grown[4]["growth_curve"],
    }


def aggregate_growth(worlds: list[dict[str, object]]) -> dict[str, object]:
    names = worlds[0]["metrics"].keys()
    mean_metrics = {
        name: float(np.mean([world["metrics"][name] for world in worlds]))
        for name in names
    }

    checks = {
        "full_depth_operator_r2_ge_0_97":
            mean_metrics["full_depth_operator_r2"] >= 0.97,
        "full_depth_application_r2_ge_0_97":
            mean_metrics["full_depth_application_r2"] >= 0.97,
        "depth2_same_budget_r2_le_0_45":
            mean_metrics["depth2_same_budget_r2"] <= 0.45,
        "depth3_same_budget_r2_between_0_50_and_0_80":
            0.50 <= mean_metrics["depth3_same_budget_r2"] <= 0.80,
        "deep_minus_shallow_r2_ge_0_50":
            (
                mean_metrics["full_depth_operator_r2"]
                - mean_metrics["depth2_same_budget_r2"]
            ) >= 0.50,
        "same_shape_random_wiring_r2_le_0_10":
            mean_metrics["same_shape_random_wiring_r2"] <= 0.10,
        "shuffled_address_wiring_r2_le_0_10":
            mean_metrics["shuffled_address_wiring_r2"] <= 0.10,
        "hidden_program_recovery_ge_0_95":
            mean_metrics["hidden_program_recovery_fraction"] >= 0.95,
        "structural_budget_respected":
            mean_metrics["used_structural_cost"] <= 12.0,
    }

    return {
        "worlds": len(worlds),
        "mean_metrics": mean_metrics,
        "gate_checks": checks,
        "gate_pass": bool(all(checks.values())),
    }
