"""Seed stability, scientific Pareto views and evidence-backed findings."""

from __future__ import annotations

import hashlib
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def seed_stability(
    candidates: Sequence[Mapping[str, Any]], *, mode: str, fingerprint: str, replicates: int = 2000
) -> dict[str, Any]:
    comparable = [value for value in candidates if int(value.get("n", 0)) > 0]
    if len(comparable) < 2:
        return {"status": "insufficient", "winner_rank_one_fraction": None, "replicates": 0}
    ranked = sorted(comparable, key=lambda value: float(value["mean"]), reverse=mode == "max")
    winner = ranked[0]
    rng = random.Random(int(hashlib.sha256(fingerprint.encode()).hexdigest()[:16], 16))
    rank_one = 0
    used = max(100, replicates)
    for _ in range(used):
        sampled: list[tuple[int, float]] = []
        for candidate in comparable:
            values = [
                float(run["final_objective"])
                for run in candidate.get("runs", ())
                if isinstance(run, Mapping)
                and run.get("phase") != "confirmation"
                and isinstance(run.get("final_objective"), int | float)
            ]
            sampled.append(
                (int(candidate["trial"]), statistics.fmean(rng.choice(values) for _ in values))
            )
        best = (max if mode == "max" else min)(sampled, key=lambda value: value[1])
        rank_one += best[0] == int(winner["trial"])
    return {
        "status": "available",
        "winner_trial": winner["trial"],
        "winner_rank_one_fraction": rank_one / used,
        "replicates": used,
        "interpretation": "bootstrap ranking stability, not a probability of the true best model",
    }


def resource_analysis(candidates: Sequence[Mapping[str, Any]], *, mode: str) -> dict[str, Any]:
    targets = ("duration_seconds", "gpu_seconds", "cpu_seconds", "peak_vram", "peak_ram")
    rows = []
    for candidate in candidates:
        if not isinstance(candidate.get("mean"), int | float):
            continue
        cost = candidate.get("resource_cost", {})
        rows.append(
            {
                "trial": candidate.get("trial"),
                "objective": candidate.get("mean"),
                **{name: cost.get(name) if isinstance(cost, Mapping) else None for name in targets},
            }
        )
    pareto: dict[str, list[int]] = {}
    for target in targets:
        valid = [row for row in rows if isinstance(row.get(target), int | float)]
        pareto[target] = [
            int(str(row["trial"]))
            for row in valid
            if not any(
                _dominates(other, row, target=target, mode=mode)
                for other in valid
                if other is not row
            )
        ]
    alternatives: list[dict[str, Any]] = []
    if rows:
        winner = (max if mode == "max" else min)(
            rows, key=lambda value: float(str(value["objective"]))
        )
        margin = max(abs(float(str(winner["objective"]))) * 0.02, 1e-12)
        for row in rows:
            if row is winner:
                continue
            signed_gap = (float(str(winner["objective"])) - float(str(row["objective"]))) * (
                1 if mode == "max" else -1
            )
            winner_gpu, row_gpu = winner.get("gpu_seconds"), row.get("gpu_seconds")
            if (
                signed_gap <= margin
                and isinstance(winner_gpu, int | float)
                and isinstance(row_gpu, int | float)
                and winner_gpu > 0
                and row_gpu < winner_gpu
            ):
                alternatives.append(
                    {
                        "trial": row["trial"],
                        "objective_gap": signed_gap,
                        "gpu_time_reduction": 1 - float(row_gpu) / float(winner_gpu),
                        "compared_with_trial": winner["trial"],
                    }
                )
    return {"targets": rows, "pareto": pareto, "efficient_alternatives": alternatives}


def findings(
    *,
    candidates: Sequence[Mapping[str, Any]],
    surrogate: Mapping[str, Any],
    effects: Mapping[str, Any],
    coverage: Mapping[str, Any],
    boundaries: Mapping[str, Any],
    pool: Mapping[str, Any],
    seeds: Mapping[str, Any],
    winner: Mapping[str, Any],
    resources: Mapping[str, Any],
    pruning: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    observations = int(surrogate.get("observations", 0))
    if observations < 8:
        output.append(
            _finding(
                "insufficient_evidence",
                "warning",
                "Study evidence is still limited",
                f"{observations} complete comparable candidates are available.",
                [f"complete candidates: {observations}"],
                "low",
                "Surrogate conclusions remain descriptive.",
                "Collect more full-fidelity candidates before making strong parameter claims.",
                support=observations,
            )
        )
    elif surrogate.get("quality") == "poor":
        output.append(
            _finding(
                "poor_surrogate",
                "warning",
                "The analysis surrogate has poor held-out accuracy",
                f"Candidate-level cross-validation RMSE is {surrogate.get('rmse')!r}.",
                [
                    f"CV MAE: {surrogate.get('mae')!r}",
                    f"rank correlation: {surrogate.get('spearman')!r}",
                ],
                "low",
                "Effect surfaces may not generalize.",
                "Use observed candidate tables and improve search coverage.",
                support=observations,
            )
        )
    global_importance = effects.get("global_importance", {})
    if isinstance(global_importance, Mapping):
        for name, detail in global_importance.items():
            if isinstance(detail, Mapping) and float(detail.get("importance", 0)) >= 0.2:
                output.append(
                    _finding(
                        "strong_parameter",
                        "info",
                        f"{name} explains substantial surrogate variation",
                        (
                            "Global functional importance is "
                            f"{float(detail['importance']):.3f} across "
                            f"{detail.get('support', 0)} candidates."
                        ),
                        [
                            "Predictive functional variance decomposition",
                            "Association is not causal",
                        ],
                        str(detail.get("reliability", "low")),
                        "This parameter differentiates predicted outcomes globally.",
                        "Inspect its response curve and interactions before changing the "
                        "search space.",
                        support=int(detail.get("support", 0)),
                    )
                )
    top_importance = effects.get("top_region_importance", {})
    if isinstance(top_importance, Mapping):
        for name, detail in top_importance.items():
            if isinstance(detail, Mapping) and float(detail.get("importance", 0)) >= 0.15:
                output.append(
                    _finding(
                        "top_region_parameter",
                        "info",
                        f"{name} is concentrated in the predicted top region",
                        f"Top-region divergence is {float(detail['importance']):.3f}.",
                        [f"support: {detail.get('support', observations)} candidates"],
                        str(detail.get("reliability", "low")),
                        "Promising candidates use this parameter differently from the full design.",
                        "Inspect its adjusted response and interactions before narrowing "
                        "the space.",
                        support=int(detail.get("support", observations)),
                    )
                )
    interactions = effects.get("interactions", ())
    if isinstance(interactions, Sequence):
        for detail in interactions[:3]:
            if not isinstance(detail, Mapping) or float(detail.get("importance", 0)) < 0.1:
                continue
            left, right = detail.get("left", "?"), detail.get("right", "?")
            output.append(
                _finding(
                    "important_interaction",
                    "info",
                    f"{left} and {right} show non-additive predictive structure",
                    f"Pair interaction importance is {float(detail['importance']):.3f}.",
                    [f"pair support: {detail.get('support', observations)} candidates"],
                    str(detail.get("reliability", "low")),
                    "Changing either parameter alone may not reproduce their joint response.",
                    "Inspect the pair surface and avoid one-dimensional causal interpretation.",
                    support=int(detail.get("support", observations)),
                )
            )
    for name, detail in boundaries.items():
        if isinstance(detail, Mapping) and detail.get("status") in {"possible", "likely"}:
            output.append(
                _finding(
                    "boundary_saturation",
                    "warning",
                    f"{name} may be truncated by the {detail.get('boundary')} boundary",
                    (
                        "Top concentration is "
                        f"{float(detail.get('top_boundary_concentration', 0)):.2f} with "
                        f"{float(detail.get('boundary_enrichment', 0)):.2f}x enrichment."
                    ),
                    [
                        f"local direction: {detail.get('local_response_direction')}",
                        f"boundary support: {detail.get('boundary_support')}",
                    ],
                    "high" if detail.get("status") == "likely" else "moderate",
                    "The optimum may lie beyond the authored range.",
                    "Consider extending this boundary in a new study; LambdaForge will not "
                    "expand it automatically.",
                    support=int(detail.get("boundary_support", 0)),
                )
            )
    joint = coverage.get("joint", {}) if isinstance(coverage, Mapping) else {}
    if isinstance(joint, Mapping) and joint.get("quality") == "poor":
        output.append(
            _finding(
                "poor_coverage",
                "warning",
                "Search-space coverage is poor",
                (
                    "Median reference distance is "
                    f"{joint.get('attempted', {}).get('median_nearest_observed_distance')!r}."
                ),
                [
                    "baseline percentile: "
                    f"{joint.get('coverage_percentile_vs_same_size_baseline')!r}"
                ],
                "moderate",
                "Parameter conclusions may rely on extrapolation.",
                "Allocate future candidates to uncovered regions.",
                support=len(candidates),
            )
        )
    if pool.get("resolution_limited"):
        output.append(
            _finding(
                "candidate_pool_resolution",
                "warning",
                "Candidate-pool resolution may limit local refinement",
                (
                    "Largest uncovered distance is "
                    f"{pool.get('largest_uncovered_region_approximation')!r}."
                ),
                [
                    f"pool size: {pool.get('candidate_pool_size')}",
                    f"median spacing: {pool.get('median_nearest_neighbour_spacing')!r}",
                ],
                "moderate",
                "The finite pool may be too coarse near the current optimum.",
                "Increase candidate-pool resolution in the next authored study.",
                support=int(pool.get("observed_candidates", 0)),
            )
        )
    stability = seeds.get("winner_rank_one_fraction")
    if isinstance(stability, int | float) and stability < 0.7:
        output.append(
            _finding(
                "unstable_winner",
                "warning",
                "The candidate ranking is seed-sensitive",
                f"The winner remains rank one in {float(stability):.1%} of seed bootstraps.",
                [f"bootstrap replicates: {seeds.get('replicates')}"],
                "moderate",
                "The apparent winner may change with additional seeds.",
                "Run confirmation seeds for the leading candidates.",
                support=int(seeds.get("replicates", 0)),
            )
        )
    if winner.get("confirmation_status") == "regressed":
        output.append(
            _finding(
                "confirmation_regression",
                "warning",
                "The screening winner regressed on fresh confirmation seeds",
                "Screening and confirmation estimates differ beyond the practical tolerance.",
                ["Fresh confirmation evidence is kept separate from screening"],
                "high",
                "The screening estimate was optimistic.",
                "Prefer the confirmed ranking or collect more fresh seeds.",
                support=1,
            )
        )
    false_prune_rate = pruning.get("false_prune_rate")
    regret = pruning.get("regret")
    if (
        isinstance(false_prune_rate, int | float)
        and float(false_prune_rate) > 0.1
    ) or (isinstance(regret, int | float) and float(regret) > 0):
        output.append(
            _finding(
                "pruning_miscalibration",
                "warning",
                "Retrospective pruning would discard competitive evidence",
                (
                    f"False-prune rate is {float(false_prune_rate or 0):.1%}; "
                    f"retrospective regret is {float(regret or 0):.6g}."
                ),
                [
                    f"calibration candidates: {pruning.get('calibration_candidates')}",
                    f"curve RMSE: {pruning.get('curve_rmse')!r}",
                ],
                "moderate",
                "The configured early-stop policy may be too aggressive for these curves.",
                "Increase evidence/confirmations or lower pruning aggressiveness in a new study.",
                support=int(pruning.get("calibration_candidates", 0) or 0),
            )
        )
    for name, detail in constraints.items():
        if not isinstance(detail, Mapping) or float(detail.get("bottleneck_fraction", 0)) < 0.25:
            continue
        output.append(
            _finding(
                "constraint_bottleneck",
                "warning",
                f"Constraint {name} is a frequent selection bottleneck",
                (
                    f"It fails for {int(detail.get('failed_candidates', 0))} of "
                    f"{int(detail.get('candidate_count', 0))} evaluated candidates."
                ),
                [f"failure fraction: {float(detail['bottleneck_fraction']):.1%}"],
                "high",
                "Objective quality alone cannot make these candidates selectable.",
                "Inspect the constraint metric and authored threshold before expanding search.",
                support=int(detail.get("candidate_count", 0)),
            )
        )
    for alternative in resources.get("efficient_alternatives", ()):
        output.append(
            _finding(
                "efficient_alternative",
                "info",
                f"Trial {alternative['trial']} is a compute-efficient alternative",
                (
                    "It is practically close to the winner while using "
                    f"{float(alternative['gpu_time_reduction']):.1%} less GPU time."
                ),
                [f"objective gap: {alternative['objective_gap']:.6g}"],
                "moderate",
                "Similar scientific quality may be available at lower cost.",
                "Review this candidate without automatically replacing the scientific winner.",
                support=1,
            )
        )
    return output


def _finding(
    kind: str,
    severity: str,
    title: str,
    statement: str,
    evidence: list[str],
    reliability: str,
    implication: str,
    recommendation: str,
    *,
    support: int,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "severity": severity,
        "title": title,
        "statement": statement,
        "evidence": evidence,
        "reliability": reliability,
        "reliability_components": {"support": support},
        "implication": implication,
        "recommendation": recommendation,
    }


def _dominates(
    left: Mapping[str, Any], right: Mapping[str, Any], *, target: str, mode: str
) -> bool:
    objective_better = (
        float(left["objective"]) >= float(right["objective"])
        if mode == "max"
        else float(left["objective"]) <= float(right["objective"])
    )
    cost_better = float(left[target]) <= float(right[target])
    strict = float(left["objective"]) != float(right["objective"]) or float(left[target]) != float(
        right[target]
    )
    return objective_better and cost_better and strict


__all__ = ["findings", "resource_analysis", "seed_stability"]
