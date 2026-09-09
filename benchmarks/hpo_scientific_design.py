"""Deterministic retrospective benchmark for scientific-design HPO.

Run from a source checkout with::

    python benchmarks/hpo_scientific_design.py

This is deliberately CPU-only and dependency-free.  It compares the mixed-kNN optimization-only
selector retained by LambdaForge with the joint performance/understanding policy at the same
candidate budget.  The benchmark is a regression instrument, not a claim about every objective.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

from lambdaforge.hpo.AdaptiveSampler import AdaptiveSampler, CandidateObservation
from lambdaforge.hpo.ScientificDesign import ExperimentalDesignPolicy, ScientificQuestionAnalyzer

Parameters = Mapping[str, Any]
Landscape = Callable[[Parameters], float]


def _pool() -> dict[int, dict[str, Any]]:
    return {
        index + 1: {"x": x / 8.0, "feature": feature}
        for index, (x, feature) in enumerate(
            (x, feature) for x in range(9) for feature in (False, True)
        )
    }


def _landscapes() -> dict[str, Landscape]:
    return {
        "smooth": lambda p: 1.0 - (float(p["x"]) - 0.75) ** 2,
        "interaction": lambda p: 1.0
        - (float(p["x"]) - (0.8 if bool(p["feature"]) else 0.3)) ** 2,
        "flat-feature": lambda p: 1.0 - (float(p["x"]) - 0.625) ** 2,
    }


def _candidates(
    selected: list[int],
    outcomes: Mapping[int, Mapping[int | None, float]],
    pool: Mapping[int, Parameters],
) -> list[dict[str, Any]]:
    return [
        {
            "trial": trial,
            "parameters": dict(pool[trial]),
            "runs": [
                {"seed": seed, "state": "succeeded", "final_objective": value}
                for seed, value in values.items()
            ],
        }
        for trial in selected
        if (values := outcomes.get(trial))
    ]


def _run_one(
    landscape: Landscape,
    *,
    scientific: bool,
    budget: int,
    fingerprint: str,
) -> dict[str, Any]:
    pool = _pool()
    selected = [1, len(pool)]
    outcomes: dict[int, dict[int | None, float]] = {
        trial: {1: landscape(pool[trial])} for trial in selected
    }
    while len(selected) < budget:
        if scientific:
            understanding = ScientificQuestionAnalyzer.analyze(
                _candidates(selected, outcomes, pool),
                {"metric": "utility", "mode": "max"},
                practical_margin=0.02,
                fingerprint=f"{fingerprint}:{len(selected)}",
                candidate_pool=pool,
            )
            ranked = ExperimentalDesignPolicy(
                pool, mode="max", practical_margin=0.02
            ).rank(outcomes, selected=selected, scientific_state=understanding)
            trial = ranked[0].trial
        else:
            observations = tuple(
                CandidateObservation(trial, next(iter(outcomes[trial].values())), 0.0)
                for trial in selected
            )
            proposal = AdaptiveSampler(pool, mode="max").propose(
                observations, selected=selected, count=1
            )
            if not proposal:
                break
            trial = proposal[0]
        selected.append(trial)
        outcomes[trial] = {1: landscape(pool[trial])}
    optimum = max(landscape(parameters) for parameters in pool.values())
    achieved = max(value for values in outcomes.values() for value in values.values())
    final = ScientificQuestionAnalyzer.analyze(
        _candidates(selected, outcomes, pool),
        {"metric": "utility", "mode": "max"},
        practical_margin=0.02,
        fingerprint=f"{fingerprint}:final",
        candidate_pool=pool,
        final=True,
    )
    return {
        "simple_regret": optimum - achieved,
        "best_utility": achieved,
        "questions_resolved": sum(
            question["conclusion_kind"] not in {"UNRESOLVED", "NO_CLEAR_PREFERENCE"}
            for question in (
                *final["parameter_questions"],
                *final["interaction_questions"],
            )
        ),
        "mean_confidence": sum(
            float(question["confidence"])
            for question in (
                *final["parameter_questions"],
                *final["interaction_questions"],
            )
        )
        / max(1, len(final["parameter_questions"]) + len(final["interaction_questions"])),
        "runs": len(selected),
    }


def run_benchmark(*, budget: int = 8) -> dict[str, Any]:
    """Return same-budget optimization and understanding regression metrics."""
    results: dict[str, Any] = {}
    for name, landscape in _landscapes().items():
        results[name] = {
            "optimization_only": _run_one(
                landscape,
                scientific=False,
                budget=budget,
                fingerprint=f"benchmark:{name}:old",
            ),
            "scientific_design": _run_one(
                landscape,
                scientific=True,
                budget=budget,
                fingerprint=f"benchmark:{name}:new",
            ),
        }
    old_regret = sum(value["optimization_only"]["simple_regret"] for value in results.values())
    new_regret = sum(value["scientific_design"]["simple_regret"] for value in results.values())
    practical_tolerance = 0.02
    problem_count = len(_landscapes())
    old_mean_regret = old_regret / problem_count
    new_mean_regret = new_regret / problem_count
    results["summary"] = {
        "budget_per_problem": budget,
        "optimization_only_mean_regret": old_mean_regret,
        "scientific_design_mean_regret": new_mean_regret,
        "practical_regret_tolerance": practical_tolerance,
        "optimization_quality_preserved": new_mean_regret
        <= old_mean_regret + practical_tolerance + math.ulp(max(1.0, old_mean_regret)),
    }
    return results


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
