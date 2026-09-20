"""Small, transport-safe projections of live Study telemetry.

The authoritative ``summary.json`` is intentionally rich.  Interactive clients must not
download it merely to draw a trial table: large studies can contain many paths, failures and
resource diagnostics per Run.  These helpers keep one stable, bounded read model next to the
authoritative evidence without becoming another result store.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

_CANDIDATE_FIELDS = (
    "trial",
    "parameters",
    "state",
    "selection_objective",
    "selection_seed_count",
    "selection_standard_error",
    "current_objective",
    "best_objective",
    "partially_censored",
    "pareto_optimal",
    "latest_metrics",
    "cost",
    "feasibility",
    "confirmation_status",
)

_RUN_FIELDS = (
    "key",
    "seed",
    "phase",
    "purpose",
    "target_questions",
    "fidelity",
    "state",
    "current_observed_objective",
    "best_observed_objective",
    "final_objective",
    "objective_status",
    "objective_censoring",
    "latest_metrics",
    "latest_step",
    "best_step",
    "best_objective",
    "duration_seconds",
    "gpu_index",
    "gpu_token",
    "termination_type",
    "prune_reason",
    "scientific_continuation",
)


def interactive_study(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project telemetry onto the complete but compact interactive Study index."""
    if value.get("detail_level") == "interactive":
        return copy.deepcopy(dict(value))
    candidates: list[dict[str, Any]] = []
    raw_candidates = value.get("candidates", ())
    if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, str | bytes):
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, Mapping):
                continue
            candidate = {
                field: copy.deepcopy(raw_candidate[field])
                for field in _CANDIDATE_FIELDS
                if field in raw_candidate
            }
            runs: list[dict[str, Any]] = []
            raw_runs = raw_candidate.get("runs", ())
            if isinstance(raw_runs, Sequence) and not isinstance(raw_runs, str | bytes):
                for raw_run in raw_runs:
                    if isinstance(raw_run, Mapping):
                        runs.append(
                            {
                                field: copy.deepcopy(raw_run[field])
                                for field in _RUN_FIELDS
                                if field in raw_run
                            }
                        )
            candidate["runs"] = runs
            candidates.append(candidate)

    controller = value.get("controller")
    controller = controller if isinstance(controller, Mapping) else {}
    admission = value.get("admission")
    admission = admission if isinstance(admission, Mapping) else {}
    projected_controller = {
        field: copy.deepcopy(controller[field])
        for field in ("last", "recent", "history_count", "surrogate_belief", "scheduler")
        if field in controller
    }
    return {
        field: copy.deepcopy(value[field])
        for field in (
            "study_telemetry_version",
            "name",
            "execution_id",
            "strategy",
            "objective",
            "planned_runs",
            "planned_candidates",
            "counts",
            "cost",
            "initial_design",
            "coverage_state",
            "hpo_analysis",
            "surrogate_belief",
            "finished",
            "created_at_utc",
            "updated_at_utc",
        )
        if field in value
    } | {
        "detail_level": "interactive",
        "candidates": candidates,
        "controller": projected_controller,
        # Admission history is diagnostic history.  The Study workspace needs only the latest
        # device/readiness explanation; the durable resource trace remains authoritative.
        "admission": {
            "current": copy.deepcopy(admission.get("current")),
            "updated_at_utc": admission.get("updated_at_utc"),
        },
    }


__all__ = ["interactive_study"]
