"""Seal a native Study selection and its persisted scientific conclusions, never rerun HPO."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from lambdaforge.analysis.StudyAnalysis import ANALYSIS_VERSION, StudyAnalysis
from lambdaforge.products.models import ProductContract, StudyProduct
from lambdaforge.products.registry import _encoded


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"StudyDecision requires authoritative {label} metadata.")
    return dict(value)


def build_study_decision(
    source: Mapping[str, Any],
    *,
    name: str,
    contract: str,
    analysis: Mapping[str, Any] | None = None,
    authored_space: Mapping[str, Any] | None = None,
) -> StudyProduct:
    """Preserve the native selected candidate, independently of an auxiliary operational failure.

    A selected point estimate is not a resolved scientific conclusion. Incomplete confirmation
    prevents selection; absent/stale analysis is never guessed or regenerated. An explicitly
    supplied final analysis must match the current native evidence identity. No model/scalar
    files are read and no scheduler/controller state is changed.
    """
    if (
        type(source.get("execution_result_version")) is not int
        or source["execution_result_version"] != 1
        or source.get("status")
        not in {
            "succeeded",
            "failed",
            "completed_with_failures",
            "cancelled",
            "interrupted",
        }
        or not isinstance(source.get("execution_id"), str)
        or not source["execution_id"]
    ):
        raise ValueError(
            "StudyDecision requires a finalized native Execution result, not live telemetry."
        )
    summary = _mapping(source.get("summary"), "summary")
    objective = _mapping(summary.get("objective"), "objective")
    design = _mapping(summary.get("study_design"), "StudyDesign")
    if summary.get("evidence_by_work"):
        raise ValueError("Select one Study; a composed Execution has no single decision authority.")
    if summary.get("objective_error"):
        raise ValueError(f"Study objective evidence is incomplete: {summary['objective_error']}")
    rows = source.get("runs")
    if not isinstance(rows, list | tuple) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("StudyDecision requires authoritative Run/Attempt evidence.")
    confirmation = summary.get("confirmation")
    confirmation = _mapping(confirmation, "confirmation") if confirmation is not None else None
    incomplete_confirmation = bool(
        confirmation
        and (
            confirmation.get("status") != "complete"
            or confirmation.get("confirmation_incomplete") is True
        )
    )
    selected = summary.get("best")
    if selected is not None:
        selected = _mapping(selected, "selected candidate")
        candidates = summary.get("candidates")
        if not isinstance(candidates, list | tuple):
            raise ValueError("Study selected candidate has no authoritative candidate inventory.")
        matches = [
            row
            for row in candidates
            if isinstance(row, Mapping) and row.get("trial") == selected.get("trial")
        ]
        if len(matches) != 1 or any(
            _encoded(matches[0].get(key)) != _encoded(value)
            for key, value in selected.items()
            if key not in {"selection_score", "selection_basis"}
        ):
            raise ValueError("Study selected candidate differs from its persisted evidence.")
        if (
            selected.get("partially_censored") is True
            or _mapping(selected.get("feasibility"), "candidate feasibility").get("feasible")
            is not True
        ):
            raise ValueError(
                "A censored or infeasible candidate cannot become a StudyDecision selection."
            )
        if selected.get("selection_basis") not in {
            "fresh-confirmation-mean",
            "conservative-search-bound",
        }:
            raise ValueError("Study selected candidate has an unknown native selection method.")
    evidence_identity = StudyAnalysis.evidence_identity(
        source, objective=objective, authored_space=authored_space
    )
    scientific: Mapping[str, Any] = {}
    sweep: Mapping[str, Any] = {}
    if analysis is not None:
        header = _mapping(analysis.get("source"), "Analysis provenance")
        if (
            type(analysis.get("analysis_version")) is not int
            or analysis["analysis_version"] != ANALYSIS_VERSION
            or header.get("execution_id") != source["execution_id"]
            or header.get("status") != "final"
            or header.get("evidence_fingerprint") != evidence_identity
        ):
            raise ValueError(
                "Persisted Analysis does not match this Study evidence; "
                "run lf results analyze EXECUTION --recompute before publishing its conclusions."
            )
        scientific = _mapping(analysis.get("scientific_understanding"), "scientific conclusions")
        if analysis.get("sweep_analysis") is not None:
            sweep = _mapping(analysis["sweep_analysis"], "sweep conclusions")
    inputs: dict[bytes, dict[str, Any]] = {}
    run_evidence = []
    selected_rows: list[Mapping[str, Any]] = []
    if selected is not None:
        ids = selected.get("runs")
        if not isinstance(ids, list | tuple) or not ids or len(set(ids)) != len(ids):
            raise ValueError("Selected candidate must name unique logical Runs.")
        for run_id in ids:
            candidates = [row for row in rows if row.get("run_id") == run_id]
            if not candidates or any(
                type(row.get("attempt_number")) is not int for row in candidates
            ):
                raise ValueError(
                    "Selected candidate is missing authoritative Run/Attempt evidence."
                )
            latest = max(candidates, key=lambda row: row["attempt_number"])
            if sum(row["attempt_number"] == latest["attempt_number"] for row in candidates) != 1:
                raise ValueError("Selected Run has conflicting latest Attempts.")
            fidelity = _mapping(latest["fidelity"], "fidelity") if latest.get("fidelity") else {}
            if (
                latest.get("status") != "succeeded"
                or latest.get("pruned")
                or latest.get("termination_type", "completed") != "completed"
                or (fidelity and fidelity.get("target", -1) < fidelity.get("maximum", 0))
            ):
                raise ValueError(
                    "Selected candidate points to incomplete/censored latest evidence."
                )
            if (
                selected["selection_basis"] == "fresh-confirmation-mean"
                and selected.get("confirmation_complete") is not True
            ):
                raise ValueError("A partial confirmation mean cannot become a selected decision.")
            selected_rows.append(latest)
            run_evidence.append(
                {
                    key: latest.get(key)
                    for key in (
                        "run_id",
                        "attempt_id",
                        "attempt_number",
                        "seed",
                        "study_phase",
                        "fidelity",
                    )
                }
            )
            for raw in latest.get("inputs", ()):
                meaning = {
                    key: raw.get(key)
                    for key in (
                        "name",
                        "kind",
                        "content_id",
                        "sha256",
                        "size_bytes",
                    )
                }
                if not meaning["content_id"] and not meaning["sha256"]:
                    raise ValueError("Selected decision input lacks exact content identity.")
                inputs[_encoded(meaning)] = meaning
    available = selected is not None and not incomplete_confirmation
    preferred = selected if available else None
    unresolved = list(scientific.get("unresolved_questions", ()))
    if analysis is None:
        unresolved.append(
            {"question": "scientific-conclusions", "reason": "final analysis not persisted"}
        )
    if incomplete_confirmation:
        unresolved.append(
            {"question": "confirmation", "reason": "required fresh-seed evidence incomplete"}
        )
    if not available:
        unresolved.append(
            {"question": "selection", "reason": "no eligible complete native selection"}
        )
    payload = {
        "decision_version": 1,
        "status": "selection_available" if available else "selection_unavailable",
        "preferred_candidate": preferred["trial"] if preferred is not None else None,
        "preferred_parameters": dict(preferred["parameters"]) if preferred is not None else None,
        "selection_score": preferred["selection_score"] if preferred is not None else None,
        "selection_method": preferred["selection_basis"] if preferred is not None else None,
        "evidence": run_evidence,
        "uncertainty": (
            {
                "screening": selected.get("search_uncertainty"),
                "confirmation": selected.get("confirmation_uncertainty"),
            }
            if selected is not None
            else None
        ),
        "confirmation": confirmation,
        "scientific_status": analysis.get("scientific_status")
        if analysis is not None
        else "unresolved",
        "scientific_evidence": {
            key: scientific.get(key)
            for key in (
                "evidence",
                "practical_optimal_region",
                "parameter_questions",
                "interaction_questions",
            )
        },
        "sweep_conclusion": {
            key: sweep.get(key)
            for key in (
                "exact_conclusion",
                "formal_sequential_evidence",
                "sequential_decision",
            )
        }
        if sweep
        else None,
        "unresolved_questions": unresolved,
        "evidence_fingerprint": evidence_identity,
        "point_estimate_is_scientific_resolution": False,
    }
    invocation = sorted(
        {
            _encoded({"work_class": row.get("work_class"), "parameters": row.get("parameters", {})})
            for row in selected_rows
        }
    )
    import json

    return StudyProduct(
        name,
        "StudyDecision",
        ProductContract(contract, ("inputs", "objective", "design", "selection", "invocations")),
        payload,
        {
            "inputs": [inputs[key] for key in sorted(inputs)],
            "objective": objective,
            "design": design,
            "selection": {
                key: payload[key]
                for key in (
                    "status",
                    "preferred_parameters",
                    "selection_method",
                    "confirmation",
                )
            },
            "invocations": [json.loads(value) for value in invocation],
        },
        {
            "execution_id": source["execution_id"],
            "evidence_fingerprint": evidence_identity,
            "config_fingerprint": source.get("scientific_fingerprint"),
            "source_status": source["status"],
            "native_result_sha256": hashlib.sha256(
                _encoded({key: value for key, value in source.items() if not key.startswith("_")})
            ).hexdigest(),
        },
    )
