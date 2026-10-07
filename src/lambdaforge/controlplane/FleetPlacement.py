"""Placement of already-authored Run identities, never scientific proposal generation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from lambdaforge.controlplane.Fleet import ClusterHealth, FleetMember
from lambdaforge.ImmutableJson import FrozenJsonMapping
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity


@dataclass(frozen=True, slots=True)
class ExecutionEquivalence:
    """Fail-closed initial stratum: exact code, environment, input, numerics and hardware."""

    code: str
    environment: str
    inputs: str
    numerics: str
    hardware: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise ValueError("Execution equivalence requires all five verified identity fields.")

    @property
    def key(self) -> str:
        return ScientificIdentity.from_payload(asdict(self)).digest

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GlobalRun:
    """Concrete scientific work accepted from one planner; placement is not in its identity."""

    study_identity: str
    candidate: str
    seed: int | None
    phase: str
    fidelity: Mapping[str, Any]
    parameters: Mapping[str, Any]
    equivalence: ExecutionEquivalence
    gpu_memory_bytes: int = 0
    requires_gpu: bool = True
    priority_class: str = "required"
    proposal: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_locations: tuple[str, ...] = ()
    storage_bytes: int = 0

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.study_identity, self.candidate, self.phase)
        ):
            raise ValueError("A global Run requires Study, candidate and phase identities.")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("Global Run seed must be an integer or null.")
        if (
            isinstance(self.gpu_memory_bytes, bool)
            or not isinstance(self.gpu_memory_bytes, int)
            or self.gpu_memory_bytes < 0
        ):
            raise ValueError("Global Run memory requirement cannot be negative.")
        if not isinstance(self.requires_gpu, bool):
            raise TypeError("Global Run requires_gpu must be a boolean.")
        if (
            isinstance(self.storage_bytes, bool)
            or not isinstance(self.storage_bytes, int)
            or self.storage_bytes < 0
        ):
            raise ValueError("Global Run storage requirement must be nonnegative bytes.")
        if self.priority_class not in {
            "required",
            "decision",
            "confirmation",
            "primary",
            "predictive",
        }:
            raise ValueError("Unknown scientific priority class.")
        if self.priority_class == "predictive":
            required = {
                "planner_revision",
                "evidence_revision",
                "model_revision",
                "reason",
                "policy",
            }
            if not required <= self.proposal.keys():
                raise ValueError(
                    "Predictive Runs require versioned scientific proposal provenance."
                )
            for name in ("planner_revision", "evidence_revision", "model_revision"):
                revision = self.proposal[name]
                if (
                    isinstance(revision, bool)
                    or not isinstance(revision, (int, str))
                    or isinstance(revision, int)
                    and revision < 0
                    or isinstance(revision, str)
                    and not revision.strip()
                ):
                    raise ValueError(f"Predictive {name} must identify a real revision.")
            if any(
                not isinstance(self.proposal[name], str) or not self.proposal[name].strip()
                for name in ("reason", "policy")
            ):
                raise ValueError(
                    "Predictive proposals require a nonempty scientific reason/policy."
                )
        object.__setattr__(self, "parameters", FrozenJsonMapping(self.parameters))
        object.__setattr__(self, "fidelity", FrozenJsonMapping(self.fidelity))
        object.__setattr__(self, "proposal", FrozenJsonMapping(self.proposal))
        # Reject non-finite/unserializable science before it reaches a durable transaction.
        # FrozenJsonMapping alone is immutable, but deliberately not a JSON schema validator.
        json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @property
    def key(self) -> str:
        return ScientificIdentity.from_payload(
            {
                "study": self.study_identity,
                "candidate": self.candidate,
                "seed": self.seed,
                "phase": self.phase,
                "fidelity": dict(self.fidelity),
            }
        ).digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_identity": self.study_identity,
            "run_key": self.key,
            "candidate": self.candidate,
            "seed": self.seed,
            "phase": self.phase,
            "fidelity": dict(self.fidelity),
            "parameters": dict(self.parameters),
            "equivalence": self.equivalence.to_dict(),
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "storage_bytes": self.storage_bytes,
            "requires_gpu": self.requires_gpu,
            "priority_class": self.priority_class,
            "proposal": dict(self.proposal),
            "checkpoint_locations": list(self.checkpoint_locations),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GlobalRun:
        run = cls(
            study_identity=value["study_identity"],
            candidate=value["candidate"],
            seed=value["seed"],
            phase=value["phase"],
            fidelity=value["fidelity"],
            parameters=value["parameters"],
            equivalence=ExecutionEquivalence(**value["equivalence"]),
            gpu_memory_bytes=value.get("gpu_memory_bytes", 0),
            requires_gpu=value.get("requires_gpu", True),
            priority_class=value.get("priority_class", "required"),
            proposal=value.get("proposal", {}),
            checkpoint_locations=tuple(value.get("checkpoint_locations", ())),
            storage_bytes=value.get("storage_bytes", 0),
        )
        if value.get("run_key", run.key) != run.key:
            raise ValueError("Global Run identity does not match its definition.")
        return run


@dataclass(frozen=True, slots=True)
class ClusterOffer:
    """Fresh local admission attestation, NOT a permission inferred from physical GPU usage.

    Slots count additional Runs that the local executor will consider; the executor must still
    revalidate ARI/allocation/grants at actual dispatch. Unknown facts remain null/not-ready.
    """

    cluster: str
    health: ClusterHealth
    scheduler: str
    gpu_access: str
    observed_at: float
    valid_until: float
    slots: int = 0
    admissible_gpus: int = 0
    free_memory_bytes: int | None = None
    equivalence: ExecutionEquivalence | None = None
    environment_ready: bool = False
    inputs_ready: bool = False
    local_admission_verified: bool = False
    startup_seconds: float | None = None
    expected_run_seconds: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    acknowledged_leases: tuple[str, ...] = ()
    admissible_storage_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("slots", "admissible_gpus"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Cluster offer {name} must be a nonnegative integer.")
        if not math.isfinite(self.observed_at) or not math.isfinite(self.valid_until):
            raise ValueError("Offer timestamps must be finite.")
        if self.valid_until <= self.observed_at:
            raise ValueError("Cluster offers require a positive freshness interval.")
        if not isinstance(self.health, ClusterHealth):
            raise TypeError("Offer health must be a ClusterHealth value.")
        for name in ("environment_ready", "inputs_ready", "local_admission_verified"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Cluster offer {name} must be an explicit boolean.")
        if self.free_memory_bytes is not None and (
            isinstance(self.free_memory_bytes, bool) or not isinstance(self.free_memory_bytes, int)
        ):
            raise TypeError("Cluster offer free memory must be an integer byte count or null.")
        if self.admissible_storage_bytes is not None and (
            isinstance(self.admissible_storage_bytes, bool)
            or not isinstance(self.admissible_storage_bytes, int)
            or self.admissible_storage_bytes < 0
        ):
            raise ValueError("Cluster offer storage must be nonnegative bytes or unknown.")
        if self.equivalence is not None and not isinstance(self.equivalence, ExecutionEquivalence):
            raise TypeError("Offer equivalence must be a verified ExecutionEquivalence or null.")
        for name in ("free_memory_bytes", "startup_seconds", "expected_run_seconds"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Cluster offer {name} must be finite and nonnegative or null.")
        object.__setattr__(self, "diagnostics", FrozenJsonMapping(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                value.value
                if isinstance(value, ClusterHealth)
                else value.to_dict()
                if isinstance(value, ExecutionEquivalence)
                else dict(value)
                if isinstance(value, Mapping)
                else value
            )
            for name, value in (
                (field_name, getattr(self, field_name)) for field_name in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class Placement:
    """Chosen operational target and an inspectable non-scientific placement audit."""

    cluster: str
    score: tuple[float, int, str]
    reason: Mapping[str, Any]


class GlobalPlacementBroker:
    """Hard compatibility filters, then estimated completion/load/locality ordering."""

    def exclusions(
        self,
        run: GlobalRun,
        offer: ClusterOffer,
        member: FleetMember,
        *,
        now: float,
        active_runs: int = 0,
        inflight_jobs: int = 0,
    ) -> tuple[str, ...]:
        reasons = []
        if member.state != ClusterHealth.ONLINE:
            reasons.append(member.state.value)
        if offer.health not in {ClusterHealth.ONLINE, ClusterHealth.DEGRADED}:
            reasons.append(offer.health.value)
        if offer.cluster != member.cluster:
            reasons.append("offer-target-mismatch")
        if offer.observed_at > now or offer.valid_until <= now:
            reasons.append("offer-stale")
        if not offer.local_admission_verified:
            reasons.append("local-admission-unverified")
        if not offer.environment_ready or not offer.inputs_ready:
            reasons.append("environment-or-inputs-unavailable")
        if offer.equivalence != run.equivalence:
            reasons.append("execution-equivalence-mismatch")
        if offer.slots < 1:
            reasons.append("no-admissible-slot")
        if member.max_runs is not None and active_runs >= member.max_runs:
            reasons.append("member-run-cap")
        if member.max_inflight_jobs is not None and inflight_jobs >= member.max_inflight_jobs:
            reasons.append("member-job-cap")
        if run.requires_gpu:
            if offer.admissible_gpus < 1:
                reasons.append("no-granted-gpu-capacity")
            if member.max_gpus is not None and offer.admissible_gpus > member.max_gpus:
                reasons.append("local-executor-has-not-applied-gpu-cap")
            if offer.free_memory_bytes is None or offer.free_memory_bytes < run.gpu_memory_bytes:
                reasons.append("memory-fit-unknown-or-insufficient")
        if run.checkpoint_locations and offer.cluster not in run.checkpoint_locations:
            reasons.append("checkpoint-transfer-required")
        if run.storage_bytes and (
            offer.admissible_storage_bytes is None
            or offer.admissible_storage_bytes < run.storage_bytes
        ):
            reasons.append("storage-fit-unknown-or-insufficient")
        return tuple(reasons)

    def place(
        self,
        run: GlobalRun,
        offers: Sequence[ClusterOffer],
        members: Mapping[str, FleetMember],
        *,
        now: float,
        active: Mapping[str, int] | None = None,
        jobs: Mapping[str, int] | None = None,
    ) -> tuple[Placement | None, dict[str, tuple[str, ...]]]:
        active, jobs = active or {}, jobs or {}
        excluded: dict[str, tuple[str, ...]] = {}
        candidates = []
        for offer in offers:
            member = members.get(offer.cluster)
            if member is None:
                excluded[offer.cluster] = ("not-a-fleet-member",)
                continue
            reasons = self.exclusions(
                run,
                offer,
                member,
                now=now,
                active_runs=active.get(offer.cluster, 0),
                inflight_jobs=jobs.get(offer.cluster, 0),
            )
            if reasons:
                excluded[offer.cluster] = reasons
                continue
            # Unknown queue/duration is explicitly unknown, not a fabricated prediction.
            estimated = (
                offer.startup_seconds + offer.expected_run_seconds
                if offer.startup_seconds is not None and offer.expected_run_seconds is not None
                else math.inf
            )
            load = active.get(offer.cluster, 0)
            score = (estimated * (1 + load / max(1, offer.slots)), load, offer.cluster)
            candidates.append(
                Placement(
                    offer.cluster,
                    score,
                    {
                        "policy": "fleet-completion-locality-v1",
                        "compatible": True,
                        "expected_start_seconds": offer.startup_seconds,
                        "expected_run_seconds": offer.expected_run_seconds,
                        "headroom_bytes": offer.free_memory_bytes,
                        "equivalence_stratum": run.equivalence.key,
                        "checkpoint_local": offer.cluster in run.checkpoint_locations,
                        "local_admission_recheck_required": True,
                    },
                )
            )
        return (min(candidates, key=lambda value: value.score) if candidates else None, excluded)
