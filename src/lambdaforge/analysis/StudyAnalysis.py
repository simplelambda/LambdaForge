"""Orchestrate and atomically persist one versioned Study Analysis document."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.analysis.Coverage import analyze_coverage
from lambdaforge.analysis.Diagnostics import findings, resource_analysis, seed_stability
from lambdaforge.analysis.Effects import analyze_effects, infer_space, validate_surrogate
from lambdaforge.analysis.Evidence import (
    candidate_statistics,
    compare_candidates,
    evidence_fingerprint,
    normalize_evidence,
    winner_summary,
)
from lambdaforge.hpo.ObjectiveUtility import ObjectiveUtility
from lambdaforge.hpo.ScientificDesign import ScientificQuestionAnalyzer
from lambdaforge.work.atomic import atomic_write_json

ANALYSIS_VERSION = 4


class StudyAnalysis:
    """Compute deterministic post-study evidence without changing HPO decisions."""

    @classmethod
    def compute(
        cls,
        source: Mapping[str, Any],
        *,
        objective: Mapping[str, Any] | None = None,
        authored_space: Mapping[str, Any] | None = None,
        status: str | None = None,
        provisional_bootstrap_replicates: int = 500,
    ) -> dict[str, Any]:
        normalized_objective = cls.objective(source, objective)
        mode = str(normalized_objective["mode"])
        candidates, runs = normalize_evidence(source, normalized_objective)
        stable_input = {
            "execution_id": source.get("execution_id"),
            "objective": normalized_objective,
            "candidates": candidates,
            "authored_space": dict(authored_space or {}),
        }
        fingerprint = evidence_fingerprint(stable_input)
        resolved_status = status or cls._status(source, runs)
        bootstrap_replicates = (
            2000 if resolved_status == "final" else provisional_bootstrap_replicates
        )
        policy = cls._search_policy(source)
        equivalence_margin = (
            float(normalized_objective["practical_margin"])
            if isinstance(normalized_objective.get("practical_margin"), int | float)
            else cls._equivalence_margin(policy)
        )
        aggregates = candidate_statistics(
            candidates,
            mode=mode,
            fingerprint=fingerprint,
            bootstrap_replicates=bootstrap_replicates,
        )
        winner = winner_summary(
            aggregates,
            mode=mode,
            equivalence_margin=equivalence_margin,
        )
        space = infer_space(aggregates, authored_space)
        surrogate = validate_surrogate(aggregates, space)
        coverage, pool, boundaries = analyze_coverage(
            aggregates,
            space=space,
            mode=mode,
            fingerprint=fingerprint,
            proposal_pool_size=cls._proposal_pool_size(source, policy),
        )
        effects = analyze_effects(
            aggregates,
            space=space,
            mode=mode,
            fingerprint=fingerprint,
            provisional=resolved_status == "provisional",
            surrogate_diagnostics=surrogate,
            coverage_quality=str(coverage.get("joint", {}).get("quality", "insufficient")),
        )
        seeds = seed_stability(
            aggregates,
            mode=mode,
            fingerprint=fingerprint,
            replicates=bootstrap_replicates,
        )
        resources = resource_analysis(
            aggregates,
            mode=mode,
            equivalence_margin=equivalence_margin,
        )
        comparisons = cls._comparisons(aggregates, winner=winner, mode=mode)
        constraint_summary = cls._constraints(aggregates)
        pareto = cls._component_pareto(aggregates, normalized_objective)
        pruning = cls._pruning(source)
        design = cls._study_design(source)
        sweep_analysis = (
            cls._sweep_analysis(
                aggregates,
                design=design,
                mode=mode,
                practical_margin=equivalence_margin,
                fingerprint=fingerprint,
            )
            if design.get("type") == "sweep"
            else None
        )
        scientific_understanding = ScientificQuestionAnalyzer.analyze(
            aggregates,
            normalized_objective,
            practical_margin=equivalence_margin,
            fingerprint=fingerprint,
            final=resolved_status == "final",
            parameter_space=authored_space,
        )
        generated_findings = findings(
            candidates=aggregates,
            surrogate=surrogate,
            effects=effects,
            coverage=coverage,
            boundaries=boundaries,
            pool=pool,
            seeds=seeds,
            winner=winner,
            resources=resources,
            pruning=pruning,
            constraints=constraint_summary,
        )
        observed_candidate_count = sum(candidate.get("n", 0) > 0 for candidate in aggregates)
        required_by_candidate = cls._required_by_candidate(design)
        evidence_complete_candidate_count = sum(
            bool(required_by_candidate.get(int(candidate.get("trial", 0))))
            and required_by_candidate[int(candidate.get("trial", 0))].issubset(
                {
                    run.get("seed")
                    for run in candidate.get("runs", ())
                    if isinstance(run, Mapping) and run.get("final_objective") is not None
                }
            )
            for candidate in aggregates
        )
        required_runs = sum(len(values) for values in required_by_candidate.values())
        required_attempted = sum(
            len(
                required_by_candidate.get(int(candidate.get("trial", 0)), set())
                & {
                    run.get("seed")
                    for run in candidate.get("runs", ())
                    if isinstance(run, Mapping)
                    and (
                        run.get("final_objective") is not None
                        or bool(run.get("censored"))
                    )
                }
            )
            for candidate in aggregates
        )
        required_completed = sum(
            len(
                required_by_candidate.get(int(candidate.get("trial", 0)), set())
                & {
                    run.get("seed")
                    for run in candidate.get("runs", ())
                    if isinstance(run, Mapping) and run.get("final_objective") is not None
                }
            )
            for candidate in aggregates
        )
        design_status = (
            str(source.get("design_status"))
            if source.get("design_status") in {"complete", "incomplete"}
            else "complete"
            if required_runs and required_attempted == required_runs
            else "incomplete"
            if required_runs
            else "unknown"
        )
        scientific_status = cls._scientific_status(scientific_understanding)
        if isinstance(sweep_analysis, Mapping):
            exact = sweep_analysis.get("exact_conclusion", {})
            exact_kind = exact.get("kind") if isinstance(exact, Mapping) else None
            scientific_status = (
                "unresolved"
                if exact_kind in {None, "UNRESOLVED", "NO_CLEAR_PREFERENCE"}
                else "resolved"
            )
        analysis = {
            "analysis_version": ANALYSIS_VERSION,
            "source": {
                "execution_id": source.get("execution_id"),
                "study_fingerprint": source.get("scientific_fingerprint"),
                "evidence_fingerprint": fingerprint,
                "status": resolved_status,
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "objective": normalized_objective,
            "search_space": space,
            "summary": {
                "candidate_count": len(aggregates),
                "observed_candidate_count": observed_candidate_count,
                "complete_candidate_count": (
                    evidence_complete_candidate_count
                    if required_by_candidate
                    else observed_candidate_count
                ),
                "evidence_complete_candidate_count": evidence_complete_candidate_count,
                "run_count": len(runs),
                "complete_run_count": sum(run.get("final_objective") is not None for run in runs),
                "censored_run_count": sum(bool(run.get("censored")) for run in runs),
                "failed_run_count": sum(run.get("state") == "failed" for run in runs),
                "required_run_count": required_runs,
                "required_completed": required_completed,
                "required_attempted": required_attempted,
                "required_censored": max(0, required_attempted - required_completed),
                "required_missing": max(0, required_runs - required_attempted),
                "evidence_completion_fraction": (
                    required_completed / required_runs if required_runs else None
                ),
                "attempted_completion_fraction": (
                    required_attempted / required_runs if required_runs else None
                ),
                "design_status": design_status,
                "scientific_status": scientific_status,
            },
            "study_design": design or None,
            "design_status": design_status,
            "scientific_status": scientific_status,
            "winner": winner,
            "candidates": aggregates,
            "candidate_comparisons": comparisons,
            "seed_analysis": seeds,
            "parameter_importance": effects.get("global_importance", {}),
            "top_region_importance": effects.get("top_region_importance", {}),
            "response_curves": effects.get("response_curves", {}),
            "interactions": {
                "pairs": effects.get("interactions", []),
                "matrix": effects.get("interaction_matrix", {}),
                "surfaces": effects.get("pairwise_surfaces", {}),
                "residual_higher_order_variance": effects.get("residual_higher_order_variance"),
            },
            "coverage": coverage,
            "candidate_pool_resolution": pool,
            "boundaries": boundaries,
            "surrogate": surrogate,
            "pruning": pruning,
            "scientific_understanding": scientific_understanding,
            "parameter_conclusions": scientific_understanding["parameter_questions"],
            "interaction_conclusions": scientific_understanding["interaction_questions"],
            "practical_optimal_region": scientific_understanding["practical_optimal_region"],
            "optimization_opportunity": scientific_understanding["optimization_opportunity"],
            "scientific_uncertainty": scientific_understanding["scientific_uncertainty"],
            "sweep_analysis": sweep_analysis,
            "resources": resources,
            "resource_conditioning": cls._resource_conditioning(source),
            "pareto": pareto,
            "constraints": constraint_summary,
            "findings": generated_findings,
            "methodology": {
                "bootstrap_replicates": bootstrap_replicates,
                "effects": "mixed-space k-NN functional predictions; descriptive, not causal",
                "candidate_cv_unit": "candidate",
                "fidelity_policy": "full-fidelity terminal Runs only for final ranking",
                "pruned_policy": (
                    "retained as censored observed evidence, excluded from final seed means"
                ),
                "top_region_source": "observed-candidate-ranking",
                "resource_cost_policy": (
                    "intrinsic median per comparable full-fidelity Run is separate from "
                    "controller total spend"
                ),
                "resource_sampling_policy": (
                    "candidate evidence is conditioned by recorded physical admission; resource "
                    "blocking is operational and is never interpreted as poor objective evidence"
                ),
            },
        }
        return _portable_analysis_value(analysis)

    @staticmethod
    def _study_design(source: Mapping[str, Any]) -> dict[str, Any]:
        direct = source.get("design")
        if isinstance(direct, Mapping):
            return dict(direct)
        summary = source.get("summary")
        if isinstance(summary, Mapping) and isinstance(summary.get("study_design"), Mapping):
            return dict(summary["study_design"])
        designs = source.get("study_designs")
        if isinstance(designs, Sequence) and not isinstance(designs, str | bytes):
            selected = next((value for value in designs if isinstance(value, Mapping)), None)
            if selected is not None:
                return dict(selected)
        return {}

    @staticmethod
    def _required_by_candidate(design: Mapping[str, Any]) -> dict[int, set[Any]]:
        evidence = design.get("evidence")
        requirements = evidence.get("requirements", ()) if isinstance(evidence, Mapping) else ()
        output: dict[int, set[Any]] = {}
        for value in requirements:
            if not isinstance(value, Mapping) or value.get("required") is not True:
                continue
            candidate = value.get("candidate")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                output.setdefault(candidate, set()).add(value.get("seed"))
        return output

    @staticmethod
    def _scientific_status(scientific: Mapping[str, Any]) -> str:
        questions = [
            value
            for value in scientific.get("parameter_questions", ())
            if isinstance(value, Mapping)
        ]
        if not questions or all(
            value.get("conclusion_kind") in {"UNRESOLVED", "NO_CLEAR_PREFERENCE"}
            for value in questions
        ):
            return "unresolved"
        if any(
            value.get("conclusion_kind") in {"UNRESOLVED", "NO_CLEAR_PREFERENCE"}
            for value in questions
        ):
            return "partially_resolved"
        return "resolved"

    @classmethod
    def _sweep_analysis(
        cls,
        candidates: Sequence[Mapping[str, Any]],
        *,
        design: Mapping[str, Any],
        mode: str,
        practical_margin: float | None,
        fingerprint: str,
    ) -> dict[str, Any]:
        """Describe fixed shared-seed evidence without imputing missing cells."""
        required = cls._required_by_candidate(design)
        sign = 1.0 if mode == "max" else -1.0
        rows: list[dict[str, Any]] = []
        seed_values: dict[int, dict[Any, float]] = {}
        evidence_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            trial = int(candidate.get("trial", 0))
            candidate_runs = [
                run for run in candidate.get("runs", ()) if isinstance(run, Mapping)
            ]
            values = {
                run.get("seed"): float(run["final_objective"])
                for run in candidate_runs
                if run.get("phase") != "confirmation"
                and isinstance(run.get("final_objective"), int | float)
                and not isinstance(run.get("final_objective"), bool)
            }
            seed_values[trial] = values
            observed = list(values.values())
            needed = required.get(trial, set())
            standard_deviation = statistics.stdev(observed) if len(observed) >= 2 else None
            rows.append(
                {
                    "trial": trial,
                    "parameters": dict(candidate.get("parameters", {})),
                    "completed_seeds": sorted(values, key=str),
                    "required_seeds": sorted(needed, key=str),
                    "completed_seed_count": len(set(values) & needed) if needed else len(values),
                    "required_seed_count": len(needed),
                    "mean": statistics.fmean(observed) if observed else None,
                    "median": statistics.median(observed) if observed else None,
                    "standard_deviation": standard_deviation,
                    "standard_error": (
                        standard_deviation / math.sqrt(len(observed))
                        if standard_deviation is not None
                        else None
                    ),
                    "uncertainty_interval": candidate.get("ci95"),
                    "paired_support": None,
                    "missing_cells": sorted(needed - set(values), key=str),
                    "unpaired_descriptive_seed_count": len(set(values) - needed),
                }
            )
            state_by_seed = {
                run.get("seed"): (
                    "completed"
                    if run.get("final_objective") is not None
                    else "pruned"
                    if bool(run.get("censored"))
                    else "failed"
                    if run.get("state") in {"failed", "infeasible"}
                    else str(run.get("state", "missing"))
                )
                for run in candidate_runs
                if run.get("phase") != "confirmation"
            }
            evidence_rows.append(
                {
                    "trial": trial,
                    "parameters": dict(candidate.get("parameters", {})),
                    "cells": [
                        {"seed": seed, "state": state_by_seed.get(seed, "missing")}
                        for seed in sorted(needed, key=str)
                    ],
                }
            )
        for row in rows:
            trial = int(row["trial"])
            row["paired_support"] = max(
                (
                    len(set(seed_values[trial]) & set(other))
                    for other_trial, other in seed_values.items()
                    if other_trial != trial
                ),
                default=0,
            )
        reference_rule = design.get("reference")
        reference_trial = next(
            (
                int(candidate.get("trial", 0))
                for candidate in candidates
                if isinstance(reference_rule, Mapping)
                and all(
                    candidate.get("parameters", {}).get(name) == value
                    for name, value in reference_rule.items()
                )
            ),
            None,
        )

        def comparison(left: int, right: int) -> dict[str, Any]:
            shared = sorted(
                set(seed_values.get(left, {})) & set(seed_values.get(right, {})),
                key=str,
            )
            differences = [
                sign * (seed_values[left][seed] - seed_values[right][seed]) for seed in shared
            ]
            mean = statistics.fmean(differences) if differences else None
            interval = cls._paired_interval(
                differences,
                fingerprint=f"{fingerprint}:{left}:{right}",
            )
            return {
                "left_trial": left,
                "right_trial": right,
                "paired_seeds": shared,
                "paired_support": len(shared),
                "missing_seed_cells": sorted(
                    (required.get(left, set()) | required.get(right, set())) - set(shared),
                    key=str,
                ),
                "paired_mean_difference": mean,
                "paired_uncertainty_interval": interval,
                "probability_of_superiority": (
                    (
                        sum(value > 0 for value in differences)
                        + 0.5 * sum(value == 0 for value in differences)
                    )
                    / len(differences)
                    if differences
                    else None
                ),
                "sign_consistency": (
                    max(
                        sum(value >= 0 for value in differences),
                        sum(value <= 0 for value in differences),
                    )
                    / len(differences)
                    if differences
                    else None
                ),
                "probability_of_practical_equivalence": (
                    sum(abs(value) <= practical_margin for value in differences) / len(differences)
                    if differences and practical_margin is not None
                    else None
                ),
                "method": "paired-shared-seed-block-bootstrap",
            }

        trials = sorted(seed_values)
        pairs = [
            comparison(trial, reference_trial)
            for trial in trials
            if reference_trial is not None and trial != reference_trial
        ] if reference_trial is not None else [
            comparison(left, right)
            for index, left in enumerate(trials)
            for right in trials[index + 1 :]
        ][:100]
        common_seeds = sorted(
            set.intersection(*(set(values) for values in seed_values.values()))
            if seed_values and all(seed_values.values())
            else set(),
            key=str,
        )
        point_means = {
            trial: statistics.fmean(values.values())
            for trial, values in seed_values.items()
            if values
        }
        point_leader = (
            (max if mode == "max" else min)(point_means, key=lambda trial: point_means[trial])
            if point_means
            else None
        )
        winner_counts = {trial: 0 for trial in trials}
        equivalent_count = 0
        unresolved_count = 0
        resamples = 2000 if common_seeds else 0
        rng = random.Random(int(fingerprint.replace("sha256:", "")[:16], 16))
        for _ in range(resamples):
            sampled = [rng.choice(common_seeds) for _seed in common_seeds]
            means = {
                trial: statistics.fmean(seed_values[trial][seed] for seed in sampled)
                for trial in trials
            }
            if practical_margin is not None and (
                max(sign * value for value in means.values())
                - min(sign * value for value in means.values())
                <= practical_margin
            ):
                equivalent_count += 1
                continue
            ordered_scores = sorted((sign * value, trial) for trial, value in means.items())
            if len(ordered_scores) >= 2 and math.isclose(
                ordered_scores[-1][0],
                ordered_scores[-2][0],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                unresolved_count += 1
            else:
                winner_counts[ordered_scores[-1][1]] += 1
        best_probability = {
            str(trial): count / resamples if resamples else None
            for trial, count in winner_counts.items()
        }
        equivalent_probability = equivalent_count / resamples if resamples else None
        modal_trial = max(winner_counts, key=lambda trial: winner_counts[trial]) if trials else None
        modal_probability = (
            winner_counts[modal_trial] / resamples
            if modal_trial is not None and resamples
            else None
        )
        if equivalent_probability is not None and equivalent_probability > max(
            0.5, modal_probability or 0.0
        ):
            conclusion = {
                "kind": "PRACTICALLY_EQUIVALENT",
                "trials": trials,
                "confidence": equivalent_probability,
            }
        elif modal_trial is not None and modal_probability is not None and modal_probability > 0.5:
            conclusion = {
                "kind": "PREFERRED",
                "trials": [modal_trial],
                "confidence": modal_probability,
            }
        else:
            conclusion = {
                "kind": "NO_CLEAR_PREFERENCE" if common_seeds else "UNRESOLVED",
                "trials": [],
                "confidence": 1.0 - (modal_probability or 0.0),
            }
        return {
            "analysis_version": 1,
            "design": "fixed-shared-seed-sweep",
            "reference": dict(reference_rule) if isinstance(reference_rule, Mapping) else None,
            "reference_trial": reference_trial,
            "cells": rows,
            "evidence_matrix": {
                "seeds": sorted(
                    {seed for values in required.values() for seed in values}, key=str
                ),
                "rows": evidence_rows,
            },
            "comparisons": pairs,
            "common_complete_seeds": common_seeds,
            "best_value_probability": best_probability,
            "point_estimate_leader": point_leader,
            "exact_conclusion": conclusion,
            "conclusion_distribution": {
                **{
                    f"PREFERRED:{trial}": probability
                    for trial, probability in best_probability.items()
                    if probability is not None
                },
                **(
                    {"PRACTICALLY_EQUIVALENT": equivalent_probability}
                    if equivalent_probability is not None
                    else {}
                ),
                **(
                    {"NO_CLEAR_PREFERENCE": unresolved_count / resamples}
                    if resamples and unresolved_count
                    else {}
                ),
            },
            "pairwise_matrix_bounded": len(pairs) >= 100,
            "missing_cells_are_imputed": False,
            "primary_unit": "paired seed difference",
            "practical_margin": practical_margin,
        }

    @staticmethod
    def _paired_interval(values: Sequence[float], *, fingerprint: str) -> list[float] | None:
        if len(values) < 2:
            return None
        rng = random.Random(int(fingerprint.replace("sha256:", "")[:16], 16))
        samples = sorted(
            statistics.fmean(rng.choice(values) for _ in values) for _ in range(1000)
        )
        return [samples[24], samples[974]]

    @staticmethod
    def _resource_conditioning(source: Mapping[str, Any]) -> dict[str, Any]:
        value = source.get("resource_conditioning")
        return dict(value) if isinstance(value, Mapping) else {"available": False}

    @classmethod
    def persist(
        cls,
        source: Mapping[str, Any],
        path: str | Path,
        *,
        objective: Mapping[str, Any] | None = None,
        authored_space: Mapping[str, Any] | None = None,
        status: str | None = None,
        recompute: bool = False,
    ) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        computed = cls.compute(
            source,
            objective=objective,
            authored_space=authored_space,
            status=status,
        )
        if target.is_file() and not target.is_symlink() and not recompute:
            try:
                cached = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                cached = None
            if (
                isinstance(cached, dict)
                and cached.get("analysis_version") == ANALYSIS_VERSION
                and cached.get("source", {}).get("evidence_fingerprint")
                == computed["source"]["evidence_fingerprint"]
            ):
                return cached
        atomic_write_json(target, computed)
        return computed

    @staticmethod
    def objective(
        source: Mapping[str, Any], supplied: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        raw = supplied or source.get("objective")
        if not isinstance(raw, Mapping):
            summary = source.get("summary")
            raw = summary.get("objective") if isinstance(summary, Mapping) else None
        if not isinstance(raw, Mapping):
            for candidate in source.get("candidates", ()):
                if not isinstance(candidate, Mapping):
                    continue
                for run in candidate.get("runs", ()):
                    observation = (
                        run.get("objective_observation") if isinstance(run, Mapping) else None
                    )
                    if isinstance(observation, Mapping) and observation.get("metric"):
                        raw = {
                            "metric": observation["metric"],
                            "mode": observation.get("mode", "max"),
                        }
                        break
        if not isinstance(raw, Mapping):
            for run in source.get("runs", ()):
                observation = run.get("objective_observation") if isinstance(run, Mapping) else None
                if isinstance(observation, Mapping) and observation.get("metric"):
                    raw = {"metric": observation["metric"], "mode": observation.get("mode", "max")}
                    break
        if not isinstance(raw, Mapping):
            raise ValueError("Study Analysis requires persisted objective metadata.")
        return ObjectiveUtility.normalize(raw)

    @staticmethod
    def authored_space(configuration: Mapping[str, Any]) -> dict[str, Any]:
        search = configuration.get("sweep", configuration.get("search"))
        if isinstance(search, Mapping):
            nested = search.get("space")
            if isinstance(nested, Mapping):
                return {
                    str(name): (
                        dict(value)
                        if isinstance(value, Mapping)
                        else {"values": list(value)}
                        if isinstance(value, Sequence) and not isinstance(value, str | bytes)
                        else {"values": [value]}
                    )
                    for name, value in nested.items()
                }
            return {
                str(name): dict(value)
                for name, value in search.items()
                if isinstance(value, Mapping) and ("range" in value or "values" in value)
            }
        for level in configuration.get("steps", ()):
            if not isinstance(level, Mapping):
                continue
            definitions = level.get("parallel", (level,))
            if not isinstance(definitions, Sequence) or isinstance(definitions, str | bytes):
                continue
            for definition in definitions:
                if not isinstance(definition, Mapping):
                    continue
                discovered = StudyAnalysis.authored_space(definition)
                if discovered:
                    return discovered
        return {}

    @staticmethod
    def _status(source: Mapping[str, Any], runs: list[dict[str, Any]]) -> str:
        active = {"running", "scheduled", "retrying", "preparing", "staging", "queued"}
        if source.get("finished") is True:
            return "final"
        if source.get("status") in {"succeeded", "failed", "cancelled"} and not any(
            run.get("state") in active for run in runs
        ):
            return "final"
        return "provisional"

    @staticmethod
    def _comparisons(
        candidates: list[dict[str, Any]], *, winner: Mapping[str, Any], mode: str
    ) -> list[dict[str, Any]]:
        reference = winner.get("screening_winner")
        if not isinstance(reference, Mapping):
            return []
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("trial") == reference.get("trial")
            ),
            None,
        )
        if selected is None:
            return []
        return [
            compare_candidates(selected, candidate, mode=mode)
            for candidate in candidates
            if candidate is not selected and candidate.get("mean") is not None
        ]

    @staticmethod
    def _constraints(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for candidate in candidates:
            for name, detail in candidate.get("constraints", {}).items():
                if isinstance(detail, Mapping):
                    grouped.setdefault(str(name), []).append(detail)
        return {
            name: {
                "candidate_count": len(values),
                "failed_candidates": sum(not bool(value.get("pass")) for value in values),
                "bottleneck_fraction": sum(not bool(value.get("pass")) for value in values)
                / len(values),
            }
            for name, values in grouped.items()
        }

    @staticmethod
    def _component_pareto(
        candidates: list[dict[str, Any]], objective: Mapping[str, Any]
    ) -> dict[str, Any]:
        metrics = objective.get("metrics")
        if not isinstance(metrics, Mapping) or len(metrics) < 2:
            return {"objective_components": [], "front": []}
        comparable = [
            candidate
            for candidate in candidates
            if all(name in candidate.get("objective_components", {}) for name in metrics)
        ]
        front = []
        for candidate in comparable:
            dominated = False
            for other in comparable:
                if other is candidate:
                    continue
                no_worse = True
                strict = False
                for name, rule in metrics.items():
                    mode = str(rule.get("mode", "max")) if isinstance(rule, Mapping) else "max"
                    left = float(other["objective_components"][name])
                    right = float(candidate["objective_components"][name])
                    no_worse &= left >= right if mode == "max" else left <= right
                    strict |= left != right
                if no_worse and strict:
                    dominated = True
                    break
            if not dominated:
                front.append(candidate["trial"])
        return {"objective_components": list(metrics), "front": front}

    @staticmethod
    def _pruning(source: Mapping[str, Any]) -> dict[str, Any]:
        for key in ("pruning_audit", "pruning"):
            value = source.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        controller = source.get("controller")
        if isinstance(controller, Mapping):
            latest = controller.get("last")
            if isinstance(latest, Mapping) and isinstance(latest.get("pruning_audit"), Mapping):
                return dict(latest["pruning_audit"])
        hpo = source.get("hpo_analysis")
        if isinstance(hpo, Mapping):
            return {
                "status": "descriptive-live-evidence",
                "censored_pruned_candidates": hpo.get("censored_pruned_candidates"),
                "note": "Retrospective CurveEvidence audit was not persisted for this study.",
            }
        return {"status": "unavailable"}

    @staticmethod
    def _search_policy(source: Mapping[str, Any]) -> dict[str, Any]:
        controller = source.get("controller")
        if not isinstance(controller, Mapping):
            return {}
        recent = controller.get("recent", ())
        if isinstance(recent, Sequence) and not isinstance(recent, str | bytes):
            for event in reversed(recent):
                if (
                    isinstance(event, Mapping)
                    and event.get("action") == "INITIALIZE"
                    and isinstance(event.get("policy"), Mapping)
                ):
                    return dict(event["policy"])
        latest = controller.get("last")
        if isinstance(latest, Mapping) and isinstance(latest.get("policy"), Mapping):
            return dict(latest["policy"])
        return {}

    @staticmethod
    def _equivalence_margin(policy: Mapping[str, Any]) -> float | None:
        practical = policy.get("practical_equivalence_margin")
        if isinstance(practical, int | float) and not isinstance(practical, bool):
            return float(practical)
        if "practical_equivalence_margin" in policy:
            return None
        seed_racing = policy.get("seed_racing")
        if isinstance(seed_racing, Mapping) and isinstance(
            seed_racing.get("equivalence_margin"), int | float
        ):
            return float(seed_racing["equivalence_margin"])
        value = policy.get("equivalence_margin")
        return float(value) if isinstance(value, int | float) else None

    @staticmethod
    def _proposal_pool_size(source: Mapping[str, Any], policy: Mapping[str, Any]) -> int | None:
        value = policy.get("proposal_pool_size")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        controller = source.get("controller")
        recent = controller.get("recent", ()) if isinstance(controller, Mapping) else ()
        if isinstance(recent, Sequence) and not isinstance(recent, str | bytes):
            for event in reversed(recent):
                if isinstance(event, Mapping) and isinstance(event.get("proposal_pool_size"), int):
                    return int(event["proposal_pool_size"])
        return None


def _portable_analysis_value(value: Any) -> Any:
    """Keep the analysis document strict JSON when scientific parameters contain paths."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _portable_analysis_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_portable_analysis_value(item) for item in value]
    return value


__all__ = ["ANALYSIS_VERSION", "StudyAnalysis"]
