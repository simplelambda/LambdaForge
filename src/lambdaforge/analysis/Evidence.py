"""Normalize persisted Run evidence and compute seed-aware candidate statistics."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

from lambdaforge.analysis.Ordering import best, signed_improvement
from lambdaforge.hpo.ObjectiveUtility import ObjectiveUtility


def evidence_fingerprint(value: Mapping[str, Any]) -> str:
    """Return a stable fingerprint excluding volatile analysis timestamps."""
    import json

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def normalize_evidence(
    source: Mapping[str, Any], objective: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return candidate and Run rows from 0.13 live summaries or Execution envelopes."""
    evaluator = ObjectiveUtility(objective)
    raw_candidates = source.get("candidates")
    if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, str | bytes):
        candidates = [dict(item) for item in raw_candidates if isinstance(item, Mapping)]
    else:
        grouped: dict[int, dict[str, Any]] = {}
        for raw in source.get("runs", ()):
            if not isinstance(raw, Mapping):
                continue
            trial = raw.get("trial")
            trial = trial if isinstance(trial, Mapping) else {}
            index = int(trial.get("index", 0))
            candidate = grouped.setdefault(
                index,
                {
                    "trial": index,
                    "parameters": dict(trial.get("parameters", {})),
                    "runs": [],
                },
            )
            candidate["runs"].append(dict(raw))
        candidates = [grouped[key] for key in sorted(grouped)]

    normalized_runs: list[dict[str, Any]] = []
    output_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        normalized: list[dict[str, Any]] = []
        for raw in candidate.get("runs", ()):
            if not isinstance(raw, Mapping):
                continue
            run = _normalize_run(raw, evaluator)
            normalized.append(run)
            normalized_runs.append(run)
        output_candidates.append(
            {
                "trial": int(candidate.get("trial", 0)),
                "parameters": dict(candidate.get("parameters", {})),
                "runs": normalized,
                "state": candidate.get("state"),
            }
        )
    return output_candidates, normalized_runs


def candidate_statistics(
    candidates: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    fingerprint: str,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    """Aggregate full-fidelity seeds while retaining censored evidence separately."""
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        runs = [run for run in candidate.get("runs", ()) if isinstance(run, Mapping)]
        screening = [
            run
            for run in runs
            if run.get("phase") != "confirmation" and _finite(run.get("final_objective"))
        ]
        confirmation = [
            run
            for run in runs
            if run.get("phase") == "confirmation" and _finite(run.get("final_objective"))
        ]
        partial = [run for run in runs if _finite(run.get("best_observed_objective"))]
        base = _seed_summary(
            screening,
            fingerprint=f"{fingerprint}:{candidate.get('trial')}:screening",
            bootstrap_replicates=bootstrap_replicates,
        )
        confirmation_summary = _seed_summary(
            confirmation,
            fingerprint=f"{fingerprint}:{candidate.get('trial')}:confirmation",
            bootstrap_replicates=bootstrap_replicates,
        )
        best_partial = _best([float(run["best_observed_objective"]) for run in partial], mode=mode)
        output.append(
            {
                "trial": int(candidate.get("trial", 0)),
                "parameters": dict(candidate.get("parameters", {})),
                "state": candidate.get("state"),
                **base,
                "screening": base,
                "confirmation": confirmation_summary,
                "best_observed_objective": best_partial,
                "censored_observations": sum(bool(run.get("censored")) for run in runs),
                "failed_runs": sum(run.get("state") == "failed" for run in runs),
                "resource_cost": _resource_summary(runs),
                "objective_components": _component_summary(screening),
                "constraints": _constraint_summary(screening),
                "runs": [dict(run) for run in runs],
            }
        )
    return output


def winner_summary(
    candidates: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    equivalence_margin: float | None = None,
) -> dict[str, Any]:
    comparable = [candidate for candidate in candidates if _finite(candidate.get("mean"))]
    ranked = sorted(
        comparable,
        key=lambda value: float(value["mean"]),
        reverse=mode == "max",
    )
    screening = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    confirmed = [
        candidate for candidate in candidates if _finite(_nested(candidate, "confirmation", "mean"))
    ]
    confirmed.sort(
        key=lambda value: float(_nested(value, "confirmation", "mean")),
        reverse=mode == "max",
    )
    confirmed_winner = confirmed[0] if confirmed else None
    status = "not_configured"
    confirmation_detail: dict[str, Any] | None = None
    if screening is not None and screening.get("confirmation", {}).get("n", 0):
        screen_mean = float(screening["mean"])
        confirm_mean = float(screening["confirmation"]["mean"])
        margin_source = "authored-equivalence-margin"
        tolerance = equivalence_margin
        if tolerance is None:
            tolerance = max(abs(screen_mean) * 0.02, 1e-12)
            margin_source = "fallback-relative-2-percent"
        signed = signed_improvement(confirm_mean, screen_mean, mode)
        screen_se = screening.get("standard_error")
        confirm_se = screening["confirmation"].get("standard_error")
        difference_se = (
            math.sqrt(float(screen_se) ** 2 + float(confirm_se) ** 2)
            if _finite(screen_se) and _finite(confirm_se)
            else None
        )
        uncertainty = 1.96 * difference_se if difference_se is not None else 0.0
        if signed + uncertainty < -tolerance:
            status = "regressed"
        elif signed - uncertainty > tolerance:
            status = "improved"
        elif abs(signed) + uncertainty <= tolerance:
            status = "confirmed_equivalent"
        else:
            status = "inconclusive"
        confirmation_detail = {
            "screening_estimate": screen_mean,
            "confirmation_estimate": confirm_mean,
            "signed_improvement": signed,
            "standard_error": difference_se,
            "equivalence_margin": tolerance,
            "equivalence_margin_source": margin_source,
            "status": status,
        }
    elif confirmed:
        status = "inconclusive"
    elif any(candidate.get("confirmation", {}).get("planned", False) for candidate in candidates):
        status = "confirmation_incomplete"
    return {
        "screening_winner": _candidate_ref(screening),
        "confirmed_winner": _candidate_ref(confirmed_winner),
        "runner_up": _candidate_ref(runner_up),
        "confirmation_status": status,
        "confirmation": confirmation_detail,
    }


def compare_candidates(
    left: Mapping[str, Any], right: Mapping[str, Any], *, mode: str
) -> dict[str, Any]:
    """Compare candidates with paired seeds when possible, otherwise propagated SE."""
    sign = 1.0 if mode == "max" else -1.0
    left_seeds = _seed_values(left)
    right_seeds = _seed_values(right)
    shared = sorted(set(left_seeds) & set(right_seeds), key=str)
    if len(shared) >= 2:
        differences = [sign * (left_seeds[seed] - right_seeds[seed]) for seed in shared]
        mean = statistics.fmean(differences)
        se = statistics.stdev(differences) / math.sqrt(len(differences))
        method = "paired-seed-normal-approximation"
    else:
        mean = sign * (float(left["mean"]) - float(right["mean"]))
        se = math.sqrt(
            float(left.get("standard_error") or 0) ** 2
            + float(right.get("standard_error") or 0) ** 2
        )
        method = "independent-normal-approximation"
    probability = (
        1.0
        if se <= 0 and mean >= 0
        else 0.0
        if se <= 0
        else statistics.NormalDist(mean, se).cdf(0) * -1 + 1
    )
    return {
        "left_trial": left.get("trial"),
        "right_trial": right.get("trial"),
        "mean_signed_difference": mean,
        "standard_error": se,
        "probability_superior_approximation": probability,
        "shared_seed_count": len(shared),
        "method": method,
    }


def _normalize_run(raw: Mapping[str, Any], evaluator: ObjectiveUtility) -> dict[str, Any]:
    state = str(raw.get("state", raw.get("status", "unknown")))
    observation = raw.get("objective_observation")
    observation = observation if isinstance(observation, Mapping) else {}
    current = raw.get("current_observed_objective", observation.get("current"))
    best = raw.get("best_observed_objective", observation.get("best", raw.get("best_objective")))
    status = raw.get("objective_status")
    if not isinstance(status, Mapping):
        missing = [] if _finite(current) else list(evaluator.required_metrics)
        status = {
            "status": "complete" if _finite(current) else "incomplete",
            "missing_components": missing,
            "latest_step": raw.get("latest_step", observation.get("current_step")),
            "latest_complete_step": observation.get("current_step") if _finite(current) else None,
        }
    status_mapping = dict(status) if isinstance(status, Mapping) else {}
    fidelity = raw.get("fidelity")
    fidelity = dict(fidelity) if isinstance(fidelity, Mapping) else {}
    full = not fidelity or (
        isinstance(fidelity.get("target"), int)
        and isinstance(fidelity.get("maximum"), int)
        and int(fidelity["target"]) >= int(fidelity["maximum"])
    )
    final = raw.get("final_objective")
    if (
        final is None
        and state == "succeeded"
        and full
        and raw.get("termination_type", "completed") == "completed"
        and status_mapping.get("status") == "complete"
    ):
        final = best
    phase = str(raw.get("phase", raw.get("study_phase", "search")))
    raw_metrics = raw.get("metrics")
    metrics: Mapping[str, Any] = raw_metrics if isinstance(raw_metrics, Mapping) else {}
    return {
        "key": raw.get("key", raw.get("run_id")),
        "seed": raw.get("seed"),
        "phase": phase,
        "state": state,
        "fidelity": fidelity,
        "current_observed_objective": float(current) if _finite(current) else None,
        "best_observed_objective": float(best) if _finite(best) else None,
        "final_objective": float(final) if _finite(final) else None,
        "current_step": observation.get("current_step", raw.get("latest_step")),
        "best_step": observation.get("best_step", raw.get("best_step")),
        "objective_status": status_mapping,
        "censored": state == "pruned" or raw.get("termination_type") == "performance_pruned",
        "censoring_reason": "performance-pruned" if state == "pruned" else None,
        "metrics": dict(metrics),
        "components": dict(observation.get("components", {}))
        if isinstance(observation.get("components"), Mapping)
        else {},
        "constraints": dict(observation.get("constraints", {}))
        if isinstance(observation.get("constraints"), Mapping)
        else {},
        "duration_seconds": raw.get("duration_seconds"),
        "gpu_seconds": raw.get(
            "gpu_seconds", raw.get("duration_seconds") if raw.get("gpu_index") is not None else None
        ),
        "peak_vram": _resource_value(raw, "peak_vram"),
        "peak_ram": _resource_value(raw, "peak_ram"),
        "cpu_seconds": _resource_value(raw, "cpu_seconds"),
        "failure": dict(raw.get("failure", {}))
        if isinstance(raw.get("failure"), Mapping)
        else None,
    }


def _seed_summary(
    runs: Sequence[Mapping[str, Any]], *, fingerprint: str, bootstrap_replicates: int
) -> dict[str, Any]:
    values = [float(run["final_objective"]) for run in runs]
    seeds = [run.get("seed") for run in runs]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "standard_deviation": None,
            "standard_error": None,
            "ci95": None,
            "seeds": [],
        }
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else None
    se = sd / math.sqrt(len(values)) if sd is not None else None
    ci = (
        _bootstrap_ci(values, fingerprint=fingerprint, replicates=bootstrap_replicates)
        if len(values) >= 3
        else None
    )
    return {
        "n": len(values),
        "mean": mean,
        "standard_deviation": sd,
        "standard_error": se,
        "ci95": ci,
        "seeds": seeds,
        "uncertainty": "empirical"
        if len(values) >= 3
        else "weak"
        if len(values) == 2
        else "unavailable",
    }


def _bootstrap_ci(values: Sequence[float], *, fingerprint: str, replicates: int) -> list[float]:
    seed = int(hashlib.sha256(fingerprint.encode()).hexdigest()[:16], 16)
    randomizer = random.Random(seed)
    samples = sorted(
        statistics.fmean(randomizer.choice(values) for _ in values)
        for _ in range(max(100, replicates))
    )
    return [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]


def _seed_values(candidate: Mapping[str, Any]) -> dict[Any, float]:
    return {
        run.get("seed"): float(run["final_objective"])
        for run in candidate.get("runs", ())
        if isinstance(run, Mapping)
        and run.get("phase") != "confirmation"
        and run.get("seed") is not None
        and _finite(run.get("final_objective"))
    }


def _resource_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = ("duration_seconds", "gpu_seconds", "cpu_seconds", "peak_vram", "peak_ram")
    comparable = [
        run
        for run in runs
        if run.get("phase") != "confirmation" and _finite(run.get("final_objective"))
    ]
    per_run = {
        name: statistics.median([float(run[name]) for run in comparable if _finite(run.get(name))])
        if any(_finite(run.get(name)) for run in comparable)
        else None
        for name in names
    }
    spend = {
        name: sum(float(run[name]) for run in runs if _finite(run.get(name)))
        if name not in {"peak_vram", "peak_ram"}
        else max((float(run[name]) for run in runs if _finite(run.get(name))), default=None)
        for name in names
    }
    spend.update(
        {
            "runs_started": sum(
                _finite(run.get("duration_seconds")) or run.get("state") != "scheduled"
                for run in runs
            ),
            "seeds_purchased": len(
                {run.get("seed") for run in runs if run.get("seed") is not None}
            ),
            "promotions": sum(
                bool(run.get("fidelity"))
                and run.get("fidelity", {}).get("target") != run.get("fidelity", {}).get("maximum")
                for run in runs
            ),
        }
    )
    return {
        "intrinsic_per_comparable_run": {
            **per_run,
            "matched_run_count": len(comparable),
            "aggregation": "median-full-fidelity-screening-run",
        },
        "controller_spend": spend,
    }


def _component_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for run in runs:
        for name, detail in run.get("components", {}).items():
            if isinstance(detail, Mapping) and _finite(detail.get("raw")):
                grouped.setdefault(str(name), []).append(float(detail["raw"]))
    return {name: statistics.fmean(values) for name, values in grouped.items()}


def _constraint_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        for name, detail in run.get("constraints", {}).items():
            if isinstance(detail, Mapping):
                grouped.setdefault(str(name), []).append(detail)
    return {
        name: {
            "pass": all(bool(value.get("satisfied")) for value in values),
            "values": [value.get("value") for value in values],
            "failed_seeds": sum(not bool(value.get("satisfied")) for value in values),
        }
        for name, values in grouped.items()
    }


def _resource_value(raw: Mapping[str, Any], name: str) -> Any:
    resources = raw.get("resources")
    if isinstance(resources, Mapping):
        usage = resources.get("usage")
        if isinstance(usage, Mapping) and name in usage:
            return usage[name]
    return raw.get(name)


def _candidate_ref(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "trial": candidate.get("trial"),
        "mean": candidate.get("mean"),
        "standard_error": candidate.get("standard_error"),
        "n": candidate.get("n"),
        "parameters": dict(candidate.get("parameters", {})),
    }


def _nested(value: Mapping[str, Any], first: str, second: str) -> Any:
    nested = value.get(first)
    return nested.get(second) if isinstance(nested, Mapping) else None


def _best(values: Sequence[float], *, mode: str) -> float | None:
    return best(values, key=float, mode=mode) if values else None


def _finite(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


__all__ = [
    "candidate_statistics",
    "compare_candidates",
    "evidence_fingerprint",
    "normalize_evidence",
    "winner_summary",
]
