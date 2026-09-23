"""Derived research-centric view over durable low-level jobs."""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lambdaforge.controlplane.jobs import JobRecord


@dataclass(frozen=True, slots=True)
class ResearchAttempt:
    """Human-numbered scheduler attempt linked to its machine Job identifier."""

    number: int
    job_id: str
    state: str
    cluster: str
    scheduler: str
    scheduler_id: str | None
    job_type: str
    created_at_utc: str
    updated_at_utc: str
    retry_of: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a display-first record while retaining the Job ID for automation."""
        return {
            "number": self.number,
            "label": f"Attempt {self.number}",
            "job_id": self.job_id,
            "state": self.state,
            "cluster": self.cluster,
            "scheduler": self.scheduler,
            "scheduler_id": self.scheduler_id,
            "job_type": self.job_type,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "retry_of": self.retry_of,
        }


@dataclass(frozen=True, slots=True)
class ResearchWork:
    """Group attempts of one scientific revision on one target without owning new state."""

    work_id: str
    name: str
    kind: str
    cluster: str
    state: str
    scientific_identity: str | None
    scientific_revision: str | None
    completed_units: int | None
    planned_units: int | None
    unit: str
    attempts: int
    primary_job_id: str
    job_ids: tuple[str, ...]
    attempt_history: tuple[ResearchAttempt, ...]
    created_at_utc: str
    updated_at_utc: str
    study: Mapping[str, Any] | None = None
    study_expected: bool = False
    study_job_id: str | None = None

    def to_dict(self, *, study_detail: str = "full") -> dict[str, Any]:
        """Return the stable read model at the requested Study detail level.

        Collection views use ``overview`` so a list of Work never becomes an
        accidental transfer of every candidate and Run.  The explicit Study
        endpoint remains the authority for the bounded candidate index.
        """
        if study_detail not in {"full", "overview"}:
            raise ValueError("study_detail must be 'full' or 'overview'.")
        study = (
            study_overview(self.study)
            if self.study is not None and study_detail == "overview"
            else copy.deepcopy(self.study)
            if self.study is not None
            else None
        )
        return {
            "work_id": self.work_id,
            "name": self.name,
            "kind": self.kind,
            "cluster": self.cluster,
            "state": self.state,
            "scientific_identity": self.scientific_identity,
            "scientific_revision": self.scientific_revision,
            "progress": {
                "completed": self.completed_units,
                "total": self.planned_units,
                "unit": self.unit,
            },
            "attempts": self.attempts,
            "primary_job_id": self.primary_job_id,
            "job_ids": list(self.job_ids),
            "attempt_history": [attempt.to_dict() for attempt in self.attempt_history],
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "study": study,
            "study_expected": self.study_expected,
            "study_job_id": self.study_job_id,
        }


def study_overview(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project full Study telemetry onto the fields required by collection views."""
    if value.get("detail_level") == "overview":
        return copy.deepcopy(dict(value))
    objective = value.get("objective")
    objective = dict(objective) if isinstance(objective, Mapping) else {}
    candidates = value.get("candidates", ())
    candidates = (
        [item for item in candidates if isinstance(item, Mapping)]
        if isinstance(candidates, Sequence) and not isinstance(candidates, str | bytes)
        else []
    )
    mode = str(objective.get("mode", "max"))

    def choose(field: str, *, partial_only: bool = False) -> dict[str, Any] | None:
        comparable = [
            item
            for item in candidates
            if isinstance(item.get(field), int | float)
            and not isinstance(item.get(field), bool)
            and (not partial_only or item.get("selection_objective") is None)
        ]
        if not comparable:
            return None
        selected = sorted(
            comparable,
            key=lambda item: float(item[field]),
            reverse=mode == "max",
        )[0]
        return {
            "trial": selected.get("trial"),
            "value": selected.get(field),
            "field": field,
        }

    counts = value.get("counts")
    cost = value.get("cost")
    admission = value.get("admission")
    current_admission = admission.get("current") if isinstance(admission, Mapping) else None
    return {
        "study_telemetry_version": value.get("study_telemetry_version"),
        "detail_level": "overview",
        "name": value.get("name"),
        "execution_id": value.get("execution_id"),
        "strategy": value.get("strategy"),
        "objective": copy.deepcopy(objective),
        "planned_runs": value.get("planned_runs"),
        "planned_candidates": value.get("planned_candidates"),
        "status": value.get("status"),
        "design_status": value.get("design_status"),
        "scientific_status": value.get("scientific_status"),
        "finish_reason": value.get("finish_reason"),
        "required_runs": value.get("required_runs"),
        "required_completed": value.get("required_completed"),
        "required_pruned": value.get("required_pruned"),
        "required_failed": value.get("required_failed"),
        "required_missing": value.get("required_missing"),
        "evidence_completion_fraction": value.get("evidence_completion_fraction"),
        "counts": copy.deepcopy(dict(counts)) if isinstance(counts, Mapping) else {},
        "cost": copy.deepcopy(dict(cost)) if isinstance(cost, Mapping) else {},
        "admission": {
            "current": copy.deepcopy(dict(current_admission))
            if isinstance(current_admission, Mapping)
            else None
        },
        "leader": choose("selection_objective"),
        "partial_leader": choose("best_objective", partial_only=True),
        "finished": value.get("finished"),
        "updated_at_utc": value.get("updated_at_utc"),
    }


def _is_parameter_study_summary(value: Mapping[str, Any] | None) -> bool:
    """Distinguish comparable Runs from legacy one-Run telemetry."""
    if value is None:
        return False
    planned = value.get("planned_runs")
    if isinstance(planned, int) and not isinstance(planned, bool) and planned > 1:
        return True
    candidates = value.get("candidates", ())
    if not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes):
        return False
    mapped = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    return len(mapped) > 1 or any(
        isinstance(candidate.get("runs"), Sequence)
        and not isinstance(candidate.get("runs"), str | bytes)
        and len(candidate.get("runs", ())) > 1
        for candidate in mapped
    )


def aggregate_research_work(records: Sequence[JobRecord]) -> tuple[ResearchWork, ...]:
    """Derive semantic work groups while leaving JobStore as the sole authority."""
    grouped: dict[tuple[str, str, str], list[JobRecord]] = defaultdict(list)
    for record in records:
        identity_key = str(record.metadata.get("scientific_identity") or record.job_id)
        name = str(record.metadata.get("name") or record.config_path or record.job_id)
        grouped[(identity_key, record.cluster, name)].append(record)
    output = []
    for (identity_key, cluster, name), attempts in grouped.items():
        ordered = sorted(attempts, key=lambda value: value.created_at_utc)
        active = [record for record in ordered if not record.state.terminal]
        primary = active[-1] if active else ordered[-1]
        identity_value = primary.metadata.get("scientific_identity")
        identity = str(identity_value) if identity_value else None
        revision = (
            str(primary.metadata.get("scientific_revision"))
            if primary.metadata.get("scientific_revision")
            else identity.removeprefix("sha256:")[:12]
            if identity
            else None
        )
        remote = primary.metadata.get("remote_state", {})
        remote = remote if isinstance(remote, dict) else {}
        observed_study: dict[str, Any] | None = None
        study_job_id: str | None = None
        # Terminal provider reconciliation may contain less detail than an earlier
        # live observation.  Study identity and telemetry are durable history, so use
        # the newest Attempt that actually observed them instead of making a failed
        # Study disappear into the generic Work list.
        for candidate_record in reversed(ordered):
            candidate_remote = candidate_record.metadata.get("remote_state", {})
            if not isinstance(candidate_remote, Mapping):
                continue
            study_value = candidate_remote.get("study")
            if isinstance(study_value, Mapping):
                observed_study = dict(study_value)
                study_job_id = candidate_record.job_id
                break
        declared_studies = [
            value.metadata.get("study_expected")
            for value in ordered
            if isinstance(value.metadata.get("study_expected"), bool)
        ]
        study_expected = (
            any(declared_studies)
            if declared_studies
            else _is_parameter_study_summary(observed_study)
        )
        study = observed_study if study_expected else None
        progress = remote.get("progress", {})
        progress = progress if isinstance(progress, dict) else {}
        planned_raw = progress.get("total", primary.metadata.get("planned_units"))
        planned = int(planned_raw) if isinstance(planned_raw, int) else None
        completed_raw = progress.get("completed", primary.metadata.get("completed_units"))
        completed = int(completed_raw) if isinstance(completed_raw, int) else None
        if completed is None and primary.state.value == "succeeded":
            completed = planned
        semantic_kind = "work"
        digest = hashlib.sha256(f"{identity_key}\0{cluster}\0{name}".encode()).hexdigest()[:16]
        output.append(
            ResearchWork(
                f"work-{digest}",
                name,
                semantic_kind,
                cluster,
                primary.state.value,
                identity,
                revision,
                completed,
                planned,
                "items" if progress else str(primary.metadata.get("unit") or "runs"),
                len(ordered),
                primary.job_id,
                tuple(record.job_id for record in ordered),
                tuple(
                    ResearchAttempt(
                        number,
                        record.job_id,
                        record.state.value,
                        record.cluster,
                        record.scheduler,
                        record.scheduler_id,
                        record.job_type,
                        record.created_at_utc,
                        record.updated_at_utc,
                        record.retry_of,
                    )
                    for number, record in enumerate(ordered, 1)
                ),
                ordered[0].created_at_utc,
                max(record.updated_at_utc for record in ordered),
                study,
                study_expected,
                study_job_id,
            )
        )
    return tuple(sorted(output, key=lambda value: value.updated_at_utc, reverse=True))
