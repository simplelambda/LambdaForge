"""Normalized scientific design, evidence obligations and execution policy.

These objects deliberately stop at the scientific/operational boundary.  They say which logical
Run identities exist and which are mandatory; the existing adaptive resource dispatcher remains
the sole authority for physical placement, packing and recovery.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lambdaforge.execution.ResourceRequest import ResourceRequest


class EvidenceKind(str, Enum):
    """Scientific purpose assigned to one planned Run identity."""

    SWEEP_REQUIRED = "SWEEP_REQUIRED"
    MINIMUM_REPLICATION = "MINIMUM_REPLICATION"
    STARTUP_COVERAGE = "STARTUP_COVERAGE"
    ADAPTIVE_OPTIONAL = "ADAPTIVE_OPTIONAL"
    SHARED_SEED = "SHARED_SEED"
    CONFIRMATION = "CONFIRMATION"
    SCIENTIFIC_CONTINUATION = "SCIENTIFIC_CONTINUATION"


class EvidenceState(str, Enum):
    """Lifecycle state of one evidence identity, independent of its Attempts."""

    PLANNED = "PLANNED"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PRUNED = "PRUNED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INFEASIBLE = "INFEASIBLE"


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """One stable logical scientific identity, independent of physical Attempts."""

    candidate: int
    seed: int | None
    kind: EvidenceKind
    required: bool
    phase: str = "search"
    fidelity: int | None = None
    state: EvidenceState = EvidenceState.PLANNED

    @property
    def key(self) -> str:
        seed = "none" if self.seed is None else str(self.seed)
        fidelity = "none" if self.fidelity is None else str(self.fidelity)
        return f"candidate-{self.candidate}:seed-{seed}:{self.phase}:fidelity-{fidelity}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "candidate": self.candidate,
            "seed": self.seed,
            "fidelity": self.fidelity,
            "phase": self.phase,
            "kind": self.kind.value,
            "required": self.required,
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class EvidencePlan:
    """Finite auditable evidence obligations known before physical dispatch."""

    requirements: tuple[EvidenceRequirement, ...]

    @property
    def required(self) -> tuple[EvidenceRequirement, ...]:
        return tuple(value for value in self.requirements if value.required)

    def to_dict(self) -> dict[str, Any]:
        required = self.required
        return {
            "evidence_plan_version": 1,
            "required_run_count": len(required),
            "optional_run_count": len(self.requirements) - len(required),
            "requirements": [value.to_dict() for value in self.requirements],
        }

    @classmethod
    def fixed(
        cls,
        *,
        candidates: int,
        seeds: Sequence[int | None],
        kind: EvidenceKind = EvidenceKind.SWEEP_REQUIRED,
    ) -> EvidencePlan:
        return cls(
            tuple(
                EvidenceRequirement(candidate, seed, kind, True)
                for candidate in range(1, candidates + 1)
                for seed in seeds
            )
        )

    @classmethod
    def adaptive(
        cls,
        *,
        candidates: int,
        seeds: Sequence[int | None],
        minimum: int,
    ) -> EvidencePlan:
        requirements: list[EvidenceRequirement] = []
        for candidate in range(1, candidates + 1):
            for index, seed in enumerate(seeds):
                required = index < minimum
                requirements.append(
                    EvidenceRequirement(
                        candidate,
                        seed,
                        EvidenceKind.MINIMUM_REPLICATION
                        if required
                        else EvidenceKind.ADAPTIVE_OPTIONAL,
                        required,
                    )
                )
        return cls(tuple(requirements))


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Normalized physical limits shared by fixed and adaptive scientific designs."""

    runs_per_gpu: int | None = None
    max_parallel: int | None = None
    failure_retries: int = 1
    max_runs: int | None = None
    max_time_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("runs_per_gpu", "max_parallel", "max_runs"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"execution.{name} must be a positive integer or auto.")
        if (
            isinstance(self.failure_retries, bool)
            or not isinstance(self.failure_retries, int)
            or not 0 <= self.failure_retries <= 3
        ):
            raise ValueError("execution.failure_retries must be an integer from 0 to 3.")
        if self.max_time_seconds is not None and (
            not math.isfinite(self.max_time_seconds) or self.max_time_seconds <= 0
        ):
            raise ValueError("execution.max_time must be positive.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ExecutionPolicy:
        raw = dict(value or {})
        unknown = set(raw) - {
            "runs_per_gpu",
            "max_parallel",
            "failure_retries",
            "max_runs",
            "max_time",
        }
        if unknown:
            raise ValueError(f"Unknown execution field(s): {sorted(unknown)}.")

        def concurrency(name: str, default: int | None) -> int | None:
            selected = raw.get(name, default)
            if selected is None or (isinstance(selected, str) and selected.lower() == "auto"):
                return None
            if isinstance(selected, bool) or not isinstance(selected, int):
                raise ValueError(f"execution.{name} must be a positive integer or auto.")
            return selected

        raw_retries = raw.get("failure_retries", 1)
        if isinstance(raw_retries, bool) or not isinstance(raw_retries, int):
            raise ValueError("execution.failure_retries must be an integer from 0 to 3.")
        raw_max_runs = raw.get("max_runs")
        if raw_max_runs is not None and (
            isinstance(raw_max_runs, bool) or not isinstance(raw_max_runs, int)
        ):
            raise ValueError("execution.max_runs must be a positive integer.")

        raw_time = raw.get("max_time")
        max_time = (
            ResourceRequest.from_mapping({"time": raw_time}).runtime_seconds
            if raw_time is not None
            else None
        )
        return cls(
            runs_per_gpu=concurrency("runs_per_gpu", None),
            max_parallel=concurrency("max_parallel", None),
            failure_retries=raw_retries,
            max_runs=raw_max_runs,
            max_time_seconds=max_time,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs_per_gpu": self.runs_per_gpu if self.runs_per_gpu is not None else "auto",
            "max_parallel": self.max_parallel if self.max_parallel is not None else "auto",
            "failure_retries": self.failure_retries,
            "max_runs": self.max_runs,
            "max_time": self.max_time_seconds,
        }


@dataclass(frozen=True, slots=True)
class StudyDesign:
    """One normalized scientific protocol, regardless of authored YAML spelling."""

    kind: str
    space: Mapping[str, Any]
    reference: Mapping[str, Any] | None
    evidence: EvidencePlan
    goal: str = "balanced"
    replication: str = "fixed"
    seed_source: Mapping[str, Any] = field(default_factory=dict)
    policy_version: str = "study-design-v2"

    def __post_init__(self) -> None:
        if self.kind not in {"adaptive", "sweep", "repeated"}:
            raise ValueError(f"Unknown Study design kind: {self.kind!r}.")
        if self.goal not in {"optimize", "balanced", "understand"}:
            raise ValueError("Study goal must be optimize, balanced or understand.")
        if self.replication not in {"fixed", "adaptive", "auto-blocks"}:
            raise ValueError("Study replication must be fixed, adaptive or auto-blocks.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_design_version": 1,
            "type": self.kind,
            "space": {str(name): dict(value) for name, value in self.space.items()},
            "reference": dict(self.reference) if self.reference is not None else None,
            "evidence": self.evidence.to_dict(),
            "goal": self.goal,
            "replication": self.replication,
            "replication_policy": (
                {
                    "unit": "complete-shared-seed-block",
                    "inference": "paired-hoeffding-confidence-sequence",
                    "policy_version": "paired-hoeffding-cs-v1",
                    "hpo_pruning": False,
                }
                if self.replication == "auto-blocks"
                else None
            ),
            "seed_source": dict(self.seed_source),
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True, slots=True)
class StudyConvergenceState:
    """Auditable automatic-stop state, separate from hard budget exhaustion."""

    goal: str
    optimization_stable: bool
    contenders_stable: bool
    questions_resolved: bool
    useful_action_available: bool
    confirmation_complete: bool
    evidence_events: int
    version: str = "scientific-convergence-v1"

    @property
    def converged(self) -> bool:
        if self.useful_action_available or not self.confirmation_complete:
            return False
        if self.goal == "optimize":
            return self.optimization_stable and self.contenders_stable
        if self.goal == "understand":
            return self.optimization_stable and self.questions_resolved
        return self.optimization_stable and self.contenders_stable and self.questions_resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "goal": self.goal,
            "converged": self.converged,
            "optimization_stable": self.optimization_stable,
            "contenders_stable": self.contenders_stable,
            "questions_resolved": self.questions_resolved,
            "useful_action_available": self.useful_action_available,
            "confirmation_complete": self.confirmation_complete,
            "evidence_events": self.evidence_events,
        }


@dataclass(frozen=True, slots=True)
class ResolvedStudyConfiguration:
    """Complete versioned authority consumed and persisted by the runtime."""

    name: str
    study_type: str
    design: Mapping[str, Any]
    execution: Mapping[str, Any]
    objective: Mapping[str, Any]
    resources: Mapping[str, Any]
    search: Mapping[str, Any] | None = None
    version: str = "resolved-study-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_configuration_version": self.version,
            "name": self.name,
            "study_type": self.study_type,
            "design": dict(self.design),
            "search": dict(self.search) if self.search is not None else None,
            "execution": dict(self.execution),
            "objective": dict(self.objective),
            "resources": dict(self.resources),
            "policy_versions": {
                "seed_stream": "project-sha256-v1",
                "parameter_space": "parameter-space-v1",
                "convergence": "scientific-convergence-v1",
                "resource": "ari-v3.1",
            },
        }


__all__ = [
    "EvidenceKind",
    "EvidencePlan",
    "EvidenceRequirement",
    "EvidenceState",
    "ExecutionPolicy",
    "ResolvedStudyConfiguration",
    "StudyConvergenceState",
    "StudyDesign",
]
