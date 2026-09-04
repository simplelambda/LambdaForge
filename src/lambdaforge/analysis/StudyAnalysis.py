"""Orchestrate and atomically persist one versioned Study Analysis document."""

from __future__ import annotations

import json
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
from lambdaforge.work.atomic import atomic_write_json

ANALYSIS_VERSION = 2


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
        equivalence_margin = cls._equivalence_margin(policy)
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
        return {
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
                "complete_candidate_count": sum(
                    candidate.get("n", 0) > 0 for candidate in aggregates
                ),
                "run_count": len(runs),
                "complete_run_count": sum(run.get("final_objective") is not None for run in runs),
                "censored_run_count": sum(bool(run.get("censored")) for run in runs),
                "failed_run_count": sum(run.get("state") == "failed" for run in runs),
            },
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
            "resources": resources,
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
            },
        }

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
        search = configuration.get("search")
        if isinstance(search, Mapping):
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


__all__ = ["ANALYSIS_VERSION", "StudyAnalysis"]
