from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatorMemoryConfig:
    eras: int = 8
    d: int = 6
    context_dim: int = 5
    train_n: int = 128
    test_n: int = 128
    query_repeats: int = 8
    context_noise: float = 0.12
    observation_noise: float = 0.02
    few_shot_pairs: int = 3


def _orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1.0
    return q * sign


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    sse = float(np.sum((target - prediction) ** 2))
    centered = target - target.mean(axis=0, keepdims=True)
    sst = float(np.sum(centered**2))
    return 1.0 - sse / max(sst, 1e-20)


def _learn_operator(
    rng: np.random.Generator,
    operator: np.ndarray,
    n: int,
    noise: float,
) -> np.ndarray:
    x = rng.normal(size=(n, operator.shape[0]))
    y = x @ operator.T + noise * rng.normal(size=x.shape)
    return np.linalg.lstsq(x, y, rcond=None)[0].T


def make_operator_memory_world(
    seed: int,
    config: OperatorMemoryConfig = OperatorMemoryConfig(),
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    past_true = tuple(_orthogonal(rng, config.d) for _ in range(config.eras))
    current_true = _orthogonal(rng, config.d)

    contexts = rng.normal(size=(config.eras, config.context_dim))
    contexts /= np.linalg.norm(contexts, axis=1, keepdims=True) + 1e-12

    past_learned = tuple(
        _learn_operator(
            rng,
            operator,
            config.train_n,
            config.observation_noise,
        )
        for operator in past_true
    )
    current_learned = _learn_operator(
        rng,
        current_true,
        config.train_n,
        config.observation_noise,
    )

    return {
        "past_true": past_true,
        "past_learned": past_learned,
        "current_true": current_true,
        "current_learned": current_learned,
        "contexts": contexts,
    }


def _retrieve(contexts: np.ndarray, query: np.ndarray) -> int:
    return int(np.argmax(contexts @ query))


def run_operator_memory_world(
    seed: int,
    config: OperatorMemoryConfig = OperatorMemoryConfig(),
) -> dict[str, object]:
    world = make_operator_memory_world(seed, config)
    rng = np.random.default_rng(200_000 + seed)

    past_true = world["past_true"]
    past_learned = world["past_learned"]
    current_true = np.asarray(world["current_true"])
    current_learned = np.asarray(world["current_learned"])
    contexts = np.asarray(world["contexts"])
    average_past = np.mean(np.asarray(past_learned), axis=0)

    records: list[dict[str, float]] = []

    for era, true_past in enumerate(past_true):
        for _ in range(config.query_repeats):
            query = contexts[era] + config.context_noise * rng.normal(
                size=config.context_dim
            )
            query /= np.linalg.norm(query) + 1e-12
            selected = _retrieve(contexts, query)
            selected_past = np.asarray(past_learned[selected])

            x = rng.normal(size=(config.test_n, config.d))
            y_past = x @ np.asarray(true_past).T
            y_now = x @ current_true.T

            # Reinstatement: context selects an archived learned operator and reuses it
            # after the active operator has changed.
            reuse = x @ selected_past.T
            current_on_old_task = x @ current_learned.T
            average_on_old_task = x @ average_past.T

            # Context-erased attacker: choose one archived operator without the stamp.
            erased_index = int(rng.integers(config.eras))
            erased_on_old_task = x @ np.asarray(past_learned[erased_index]).T

            # Reflection: if both operators coexist, their relation itself is an operator.
            # T maps a representation produced by the reinstated past operator into the
            # current representation. No paired transfer examples are used here.
            reflection = current_learned @ np.linalg.pinv(selected_past)
            reflected = y_past @ reflection.T

            wrong_index = (selected + 1) % config.eras
            wrong_reflection = current_learned @ np.linalg.pinv(
                np.asarray(past_learned[wrong_index])
            )
            wrong_reflected = y_past @ wrong_reflection.T

            current_direct = y_past @ current_learned.T
            past_direct = y_past @ selected_past.T

            # Stronger interpolation attacker: choose the best convex mixture with
            # access to the test target. Even this oracle interpolation is not allowed
            # to compose/invert the two operators.
            best_convex_r2 = -np.inf
            for alpha in np.linspace(0.0, 1.0, 41):
                mixed = alpha * current_learned + (1.0 - alpha) * selected_past
                best_convex_r2 = max(
                    best_convex_r2,
                    _r2(y_now, y_past @ mixed.T),
                )

            # Few-shot attacker: learn the past->present map from a tiny number of
            # paired examples instead of deriving it from the two learned operators.
            x_shot = rng.normal(size=(config.few_shot_pairs, config.d))
            old_shot = x_shot @ np.asarray(true_past).T
            new_shot = x_shot @ current_true.T
            few_shot_map = np.linalg.lstsq(old_shot, new_shot, rcond=None)[0].T
            few_shot = y_past @ few_shot_map.T

            records.append(
                {
                    "retrieval_hit": float(selected == era),
                    "reinstated_old_task_r2": _r2(y_past, reuse),
                    "current_operator_old_task_r2": _r2(
                        y_past, current_on_old_task
                    ),
                    "average_cache_old_task_r2": _r2(
                        y_past, average_on_old_task
                    ),
                    "context_erased_old_task_r2": _r2(
                        y_past, erased_on_old_task
                    ),
                    "reflection_transfer_r2": _r2(y_now, reflected),
                    "wrong_era_reflection_r2": _r2(y_now, wrong_reflected),
                    "current_direct_transfer_r2": _r2(y_now, current_direct),
                    "past_direct_transfer_r2": _r2(y_now, past_direct),
                    "best_convex_transfer_r2": float(best_convex_r2),
                    "few_shot_transfer_r2": _r2(y_now, few_shot),
                }
            )

    metric_names = records[0].keys()
    metrics = {
        name: float(np.mean([record[name] for record in records]))
        for name in metric_names
    }

    return {
        "seed": int(seed),
        "queries": len(records),
        "metrics": metrics,
    }


def aggregate_operator_memory(worlds: list[dict[str, object]]) -> dict[str, object]:
    metric_names = worlds[0]["metrics"].keys()
    mean_metrics = {
        name: float(np.mean([world["metrics"][name] for world in worlds]))
        for name in metric_names
    }

    strongest_reinstatement_attacker = max(
        mean_metrics["current_operator_old_task_r2"],
        mean_metrics["average_cache_old_task_r2"],
        mean_metrics["context_erased_old_task_r2"],
    )
    strongest_transfer_attacker = max(
        mean_metrics["wrong_era_reflection_r2"],
        mean_metrics["current_direct_transfer_r2"],
        mean_metrics["past_direct_transfer_r2"],
        mean_metrics["best_convex_transfer_r2"],
        mean_metrics["few_shot_transfer_r2"],
    )

    checks = {
        "retrieval_hit_rate_ge_0_97":
            mean_metrics["retrieval_hit"] >= 0.97,
        "reinstated_old_task_r2_ge_0_95":
            mean_metrics["reinstated_old_task_r2"] >= 0.95,
        "reinstatement_attackers_r2_le_0_20":
            strongest_reinstatement_attacker <= 0.20,
        "reflection_transfer_r2_ge_0_95":
            mean_metrics["reflection_transfer_r2"] >= 0.95,
        "reflection_minus_best_attacker_ge_0_40":
            (
                mean_metrics["reflection_transfer_r2"]
                - strongest_transfer_attacker
            ) >= 0.40,
        "wrong_era_reflection_r2_le_0_10":
            mean_metrics["wrong_era_reflection_r2"] <= 0.10,
        "best_convex_transfer_r2_le_0_10":
            mean_metrics["best_convex_transfer_r2"] <= 0.10,
        "few_shot_transfer_r2_le_0_70":
            mean_metrics["few_shot_transfer_r2"] <= 0.70,
    }

    return {
        "worlds": len(worlds),
        "mean_metrics": mean_metrics,
        "strongest_reinstatement_attacker_r2": float(
            strongest_reinstatement_attacker
        ),
        "strongest_transfer_attacker_r2": float(strongest_transfer_attacker),
        "gate_checks": checks,
        "gate_pass": bool(all(checks.values())),
    }
