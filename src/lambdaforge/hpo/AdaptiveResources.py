"""Persistent, candidate-specific resource prediction and GPU placement.

The scientific controller owns *what is valuable*.  This module owns only when and where a
scientifically ranked action can run.  It deliberately has no dependency on CUDA, schedulers or
the Work runtime so its safety rules can be exercised with deterministic CPU-only simulations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from lambdaforge.hpo.ParameterSpace import ParameterSpace
from lambdaforge.work.atomic import atomic_write_json

ResourceRunState = Literal["completed", "oom", "pruned", "failed", "cancelled"]
AdmissionMode = Literal["SAFE_ADMISSION", "EXPLORATORY_ADMISSION"]
ResourceEvidenceQuality = Literal[
    "terminal-high-quality",
    "sampled-physical",
    "nvml-process-exact",
    "allocator-process",
    "allocator-peak-supported",
    "active-provisional",
    "aggregate-inferred",
    "lower-bound-only",
    "unavailable",
]
PlacementState = Literal[
    "ADMITTED",
    "RESOURCE_BLOCKED",
    "RESOURCE_INFEASIBLE_ON_DEVICE_TYPE",
]

# These bounds cap implementation cost and persisted diagnostic size; they are not scientific
# policy knobs.  Placement results are invariant once all relevant frontier actions fit here.
_HISTORY_LIMIT = 2048
_FRONTIER_LIMIT = 16
_TRAJECTORY_LIMIT = 192
_BOOTSTRAP_POINTS = 31
_OOM_ALLOCATION = re.compile(
    r"(?:tried to allocate|attempted to allocate)\s+([0-9]+(?:\.[0-9]+)?)\s*"
    r"(bytes?|kib|mib|gib|kb|mb|gb)",
    re.IGNORECASE,
)


def resource_identity(
    specification: Mapping[str, Any],
    *,
    code_fingerprint: str,
    environment_fingerprint: str,
    hardware: str,
) -> tuple[str, str]:
    """Return candidate and compatibility identities without requiring user configuration."""
    definition = specification.get("definition", {})
    definition = definition if isinstance(definition, Mapping) else {}
    parameters = dict(specification.get("parameters", {}))
    varied = dict(specification.get("trial_parameters", {}))
    fixed = {key: value for key, value in parameters.items() if key not in varied}
    common = {
        "work_class": definition.get("work_class"),
        "fixed_arguments": _portable(fixed),
        "fidelity": _portable(specification.get("hpo_fidelity", {})),
        "resource_shape": {
            key: _portable(parameters.get(key))
            for key in parameters
            if any(
                marker in key.lower()
                for marker in (
                    "batch",
                    "precision",
                    "sequence",
                    "nodes",
                    "edges",
                    "points",
                    "gradient_accumulation",
                )
            )
        },
        "code": code_fingerprint,
        "environment": environment_fingerprint,
        "hardware": hardware,
    }
    compatibility = _digest(common)
    candidate = _digest({**common, "parameters": _portable(varied)})
    return candidate, compatibility


@dataclass(frozen=True, slots=True)
class ResourceTrajectorySample:
    """One bounded physical/allocator observation for a Run."""

    elapsed_seconds: float
    physical_bytes: int
    device_free_bytes: int | None = None
    external_bytes: int | None = None
    cuda_allocated_bytes: int | None = None
    cuda_reserved_bytes: int | None = None
    cuda_max_allocated_bytes: int | None = None
    utilization: float | None = None
    throughput: float | None = None
    step: int | None = None
    phase: str | None = None
    checkpoint_step: int | None = None
    checkpoint_duration_seconds: float | None = None
    phases_seen: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResourceTrajectoryStatistics:
    """Versioned incremental sufficient statistics independent of monitor cadence.

    The bounded sketches retain robust scale information; progress and phase counters—not raw
    sample count—drive survival evidence.  This object is scheduling state, whereas
    :class:`BoundedResourceTrajectory` remains a compact diagnostic representation.
    """

    statistics_version: int = 1
    observations_seen: int = 0
    progress_cycles_seen: int = 0
    distinct_steps_seen: int = 0
    optimizer_cycles_seen: int = 0
    phase_visit_counts: dict[str, int] = field(default_factory=dict)
    phases_seen: list[str] = field(default_factory=list)
    material_peak_count: int = 0
    last_material_peak_time: float | None = None
    last_material_peak_step: int | None = None
    last_material_peak_phase: str | None = None
    cycles_since_material_peak: int = 0
    elapsed_since_material_peak: float = 0.0
    running_peak_bytes: int = 0
    current_memory_bytes: int = 0
    growth_distribution: list[int] = field(default_factory=list)
    noise_distribution: list[int] = field(default_factory=list)
    allocator_residual_distribution: list[int] = field(default_factory=list)
    checkpoint_times: list[float] = field(default_factory=list)
    checkpoint_durations: list[float] = field(default_factory=list)
    phase_maxima: dict[str, int] = field(default_factory=dict)
    phase_growth: dict[str, list[int]] = field(default_factory=dict)
    meaningful_event_times: list[float] = field(default_factory=list)
    last_elapsed_seconds: float = 0.0
    last_step: int | None = None
    last_phase: str | None = None
    last_checkpoint_step: int | None = None
    last_physical_bytes: int | None = None
    last_reserved_bytes: int | None = None
    last_allocated_bytes: int | None = None
    signed_changes: list[int] = field(default_factory=list)
    positive_streak: int = 0
    monotonic_growth_bytes: int = 0
    reconstructed_from_bounded: bool = False

    @classmethod
    def from_samples(
        cls,
        values: Sequence[ResourceTrajectorySample],
        *,
        reconstructed: bool = False,
    ) -> ResourceTrajectoryStatistics:
        result = cls(reconstructed_from_bounded=reconstructed)
        for value in values:
            result.observe(value)
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResourceTrajectoryStatistics:
        allowed = cls.__dataclass_fields__
        payload = {key: item for key, item in value.items() if key in allowed}
        for key in (
            "phase_visit_counts",
            "phase_maxima",
            "phase_growth",
        ):
            payload[key] = dict(payload.get(key, {}))
        for key in (
            "phases_seen",
            "growth_distribution",
            "noise_distribution",
            "allocator_residual_distribution",
            "checkpoint_times",
            "checkpoint_durations",
            "meaningful_event_times",
            "signed_changes",
        ):
            payload[key] = list(payload.get(key, ()))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def observe(self, sample: ResourceTrajectorySample) -> None:
        """Merge one poll while counting only genuine progress/allocation evidence."""
        self.observations_seen += 1
        phase = _resource_phase_class(sample.phase)
        announced = tuple(
            dict.fromkeys(_resource_phase_class(value) for value in sample.phases_seen)
        )
        new_phases = [value for value in (*announced, phase) if value not in self.phases_seen]
        for value in new_phases:
            self.phases_seen.append(value)

        first_step = self.last_step is None and sample.step is not None
        step_changed = (
            self.last_step is not None
            and sample.step is not None
            and sample.step != self.last_step
        )
        phase_changed = self.last_phase is not None and phase != self.last_phase
        if self.last_phase is None or phase_changed:
            self.phase_visit_counts[phase] = self.phase_visit_counts.get(phase, 0) + 1
        meaningful = step_changed or phase_changed or bool(new_phases)
        if first_step:
            self.distinct_steps_seen = 1
        elif step_changed:
            self.progress_cycles_seen += 1
            self.distinct_steps_seen += 1
        entered_optimizer = phase == "optimizer" and (
            self.last_phase != "optimizer" or step_changed
        )
        if entered_optimizer:
            self.optimizer_cycles_seen += 1
        if meaningful:
            _bounded_append(self.meaningful_event_times, float(sample.elapsed_seconds))
            if self.last_material_peak_time is not None:
                self.cycles_since_material_peak += 1

        current = max(0, int(sample.physical_bytes))
        delta = current - self.last_physical_bytes if self.last_physical_bytes is not None else 0
        reserved_delta = (
            sample.cuda_reserved_bytes - self.last_reserved_bytes
            if sample.cuda_reserved_bytes is not None and self.last_reserved_bytes is not None
            else None
        )
        allocated_delta = (
            sample.cuda_allocated_bytes - self.last_allocated_bytes
            if sample.cuda_allocated_bytes is not None and self.last_allocated_bytes is not None
            else None
        )
        if self.last_physical_bytes is not None:
            _bounded_append(self.signed_changes, delta)
            prior_sign = next((value for value in reversed(self.signed_changes[:-1]) if value), 0)
            sign_reversal = bool(delta and prior_sign and (delta > 0) != (prior_sign > 0))
            allocator_aligned = any(
                value is not None and value > 0 and delta > 0
                for value in (reserved_delta, allocated_delta)
            )
            residuals = [
                abs(delta - value)
                for value in (reserved_delta, allocated_delta)
                if value is not None
            ]
            for residual in residuals:
                if residual:
                    _bounded_append(self.allocator_residual_distribution, residual)
            if delta > 0:
                self.positive_streak += 1
                self.monotonic_growth_bytes += delta
            elif delta < 0:
                self.positive_streak = 0
                self.monotonic_growth_bytes = 0
            if sign_reversal or (
                delta and not allocator_aligned and current <= self.running_peak_bytes
            ):
                _bounded_append(self.noise_distribution, abs(delta))

            noise = _robust_upper_scale(
                (*self.noise_distribution, *self.allocator_residual_distribution)
            )
            new_peak = current > self.running_peak_bytes
            persistent_trend = self.positive_streak >= 2 and self.monotonic_growth_bytes > noise
            abrupt = new_peak and delta > noise
            material = new_peak and (abrupt or (persistent_trend and allocator_aligned))
            if material:
                gain = current - self.running_peak_bytes
                _bounded_append(self.growth_distribution, gain)
                _bounded_append(self.phase_growth.setdefault(phase, []), gain)
                self.material_peak_count += 1
                self.last_material_peak_time = float(sample.elapsed_seconds)
                self.last_material_peak_step = sample.step
                self.last_material_peak_phase = phase
                self.cycles_since_material_peak = 0
        elif current > 0:
            self.material_peak_count = 1
            self.last_material_peak_time = float(sample.elapsed_seconds)
            self.last_material_peak_step = sample.step
            self.last_material_peak_phase = phase

        self.running_peak_bytes = max(self.running_peak_bytes, current)
        self.current_memory_bytes = current
        self.phase_maxima[phase] = max(self.phase_maxima.get(phase, 0), current)
        if self.last_material_peak_time is not None:
            self.elapsed_since_material_peak = max(
                0.0, float(sample.elapsed_seconds) - self.last_material_peak_time
            )
        if (
            sample.checkpoint_step is not None
            and sample.checkpoint_step != self.last_checkpoint_step
        ):
            _bounded_append(self.checkpoint_times, float(sample.elapsed_seconds))
            if (
                sample.checkpoint_duration_seconds is not None
                and sample.checkpoint_duration_seconds >= 0
            ):
                _bounded_append(
                    self.checkpoint_durations,
                    float(sample.checkpoint_duration_seconds),
                )
        self.last_elapsed_seconds = max(self.last_elapsed_seconds, float(sample.elapsed_seconds))
        self.last_step = sample.step if sample.step is not None else self.last_step
        self.last_phase = phase
        self.last_checkpoint_step = (
            sample.checkpoint_step
            if sample.checkpoint_step is not None
            else self.last_checkpoint_step
        )
        self.last_physical_bytes = current
        self.last_reserved_bytes = (
            sample.cuda_reserved_bytes
            if sample.cuda_reserved_bytes is not None
            else self.last_reserved_bytes
        )
        self.last_allocated_bytes = (
            sample.cuda_allocated_bytes
            if sample.cuda_allocated_bytes is not None
            else self.last_allocated_bytes
        )


@dataclass(frozen=True, slots=True)
class ResourcePhaseModel:
    """Observed allocation-relevant phases for one compatible Work family."""

    expected_phases: tuple[str, ...] = ()
    peak_probability: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_observations(
        cls, observations: Sequence[ResourceProfileObservation]
    ) -> ResourcePhaseModel:
        visits: dict[str, int] = {}
        peaks: dict[str, int] = {}
        for observation in observations:
            for phase in observation.phases_seen:
                normalized = _resource_phase_class(phase)
                visits[normalized] = visits.get(normalized, 0) + 1
            if observation.peak_phase:
                normalized = _resource_phase_class(observation.peak_phase)
                peaks[normalized] = peaks.get(normalized, 0) + 1
        # Completeness tracks historically material phases, not a universal lifecycle checklist.
        # A repeatedly observed phase with no peak remains represented by the general residual
        # tail but does not prevent an early scientific prune from yielding a complete resource
        # profile. Conversely, one real late peak is enough to retain that phase explicitly.
        expected = tuple(sorted(phase for phase, count in peaks.items() if count > 0))
        probabilities = {
            phase: (peaks.get(phase, 0) + 0.5) / (max(1, visits.get(phase, 0)) + 1.0)
            for phase in expected
        }
        return cls(expected, probabilities)

    def completeness(self, phases_seen: Sequence[str]) -> tuple[bool, tuple[str, ...]]:
        observed = {_resource_phase_class(value) for value in phases_seen}
        missing = tuple(
            phase
            for phase in self.expected_phases
            if phase not in observed and self.peak_probability.get(phase, 0.0) > 0
        )
        return not missing, missing


class BoundedResourceTrajectory:
    """Downsample a long resource stream while preserving extrema and transitions."""

    def __init__(self, values: Sequence[ResourceTrajectorySample] = ()) -> None:
        self._values: list[ResourceTrajectorySample] = []
        self.statistics = ResourceTrajectoryStatistics()
        for value in values:
            self.append(value)

    @property
    def values(self) -> tuple[ResourceTrajectorySample, ...]:
        return tuple(self._values)

    def append(self, value: ResourceTrajectorySample) -> None:
        self.statistics.observe(value)
        self._values.append(value)
        if len(self._values) <= _TRAJECTORY_LIMIT:
            return
        # Extrema and phase boundaries are scientific/resource evidence.  Step boundaries can be
        # far more numerous than the bounded read model, so retain a representative subset rather
        # than allowing them to overflow the budget and accidentally evict an old true peak.
        mandatory = {0, len(self._values) - 1}
        for selector in (
            lambda sample: sample.physical_bytes,
            lambda sample: -sample.physical_bytes,
            lambda sample: -(sample.device_free_bytes or 0),
            lambda sample: sample.device_free_bytes or 0,
            lambda sample: sample.cuda_allocated_bytes or 0,
            lambda sample: sample.cuda_reserved_bytes or 0,
        ):
            mandatory.add(
                max(range(len(self._values)), key=lambda index: selector(self._values[index]))
            )
        phase_boundaries: set[int] = set()
        step_boundaries: set[int] = set()
        for index in range(1, len(self._values)):
            before, current = self._values[index - 1], self._values[index]
            if before.phase != current.phase:
                phase_boundaries.update((index - 1, index))
            elif before.step != current.step:
                step_boundaries.update((index - 1, index))
        mandatory.update(
            _representative_indices(
                sorted(phase_boundaries), max(0, _TRAJECTORY_LIMIT - len(mandatory))
            )
        )
        mandatory.update(
            _representative_indices(
                sorted(step_boundaries), max(0, _TRAJECTORY_LIMIT - len(mandatory)) // 2
            )
        )
        available = max(0, _TRAJECTORY_LIMIT - len(mandatory))
        optional = [index for index in range(len(self._values)) if index not in mandatory]
        mandatory.update(_representative_indices(optional, available))
        self._values = [self._values[index] for index in sorted(mandatory)]


@dataclass(frozen=True, slots=True)
class OOMResourceEvidence:
    """Censored lower-bound evidence derived from one allocation failure."""

    candidate_key: str
    compatibility_key: str
    hardware: str
    total_bytes: int
    lower_bound_bytes: int
    lower_bound_source: str
    headroom_bytes: int
    attempted_allocation_bytes: int | None = None
    candidate_resident_bytes: int | None = None
    physical_free_bytes: int | None = None
    external_bytes: int | None = None
    active_co_runs: int = 0
    phase: str | None = None
    step: int | None = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlacementOOMEvidence:
    """One failed co-location, kept separate from intrinsic candidate demand."""

    candidate_key: str
    compatibility_key: str
    hardware: str
    resident_candidates: tuple[str, ...]
    headroom_bytes: int
    physical_free_bytes: int | None
    aggregate_used_bytes: int | None
    attempted_allocation_bytes: int | None
    candidate_resident_bytes: int | None
    evidence_quality: ResourceEvidenceQuality
    experiment_signature: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PlacementOOMEvidence:
        return cls(
            candidate_key=str(value.get("candidate_key", "")),
            compatibility_key=str(value.get("compatibility_key", "")),
            hardware=str(value.get("hardware", "")),
            resident_candidates=tuple(str(item) for item in value.get("resident_candidates", ())),
            headroom_bytes=max(0, int(value.get("headroom_bytes", 0) or 0)),
            physical_free_bytes=_optional_int(value.get("physical_free_bytes")),
            aggregate_used_bytes=_optional_int(value.get("aggregate_used_bytes")),
            attempted_allocation_bytes=_optional_int(value.get("attempted_allocation_bytes")),
            candidate_resident_bytes=_optional_int(value.get("candidate_resident_bytes")),
            evidence_quality=str(value.get("evidence_quality", "lower-bound-only")),  # type: ignore[arg-type]
            experiment_signature=str(value.get("experiment_signature", "")),
            timestamp_utc=str(value.get("timestamp_utc", "")),
        )


@dataclass(frozen=True, slots=True)
class ResourceTrajectoryAnalysis:
    """Low-cost deterministic interpretation of one live memory trajectory."""

    state: str
    running_peak_bytes: int
    current_bytes: int
    last_peak_step: int | None
    completed_cycles_since_peak: int
    residual_samples: tuple[int, ...]
    phase: str | None
    reason: str
    residual_weights: tuple[float, ...] = ()
    growth_hazard: float = 1.0
    expected_residual_growth_bytes: int = 0
    measurement_noise_bytes: int = 0
    critical_phases_seen: tuple[str, ...] = ()
    next_decision_seconds: float | None = None
    next_decision_uncertainty_seconds: float | None = None
    evidence_cycles: int = 0
    phase_hazards: Mapping[str, float] = field(default_factory=dict)
    tail_probability: float = 1.0
    last_material_peak_time: float | None = None
    checkpoint_cadence_seconds: float | None = None


class ResourceTrajectoryAnalyzer:
    """Detect ramps and provisional plateaus from change events and progress cycles."""

    @staticmethod
    def analyze(
        values: Sequence[ResourceTrajectorySample],
        *,
        historical_residuals: Sequence[int] = (),
        total_bytes: int = 0,
        trajectory_statistics: ResourceTrajectoryStatistics | None = None,
        phase_model: ResourcePhaseModel | None = None,
    ) -> ResourceTrajectoryAnalysis:
        if not values and (
            trajectory_statistics is None or trajectory_statistics.observations_seen == 0
        ):
            return ResourceTrajectoryAnalysis(
                "STARTING",
                0,
                0,
                None,
                0,
                (max(0, total_bytes),),
                None,
                "no live samples",
                (1.0,),
            )
        sufficient = trajectory_statistics or ResourceTrajectoryStatistics.from_samples(
            values, reconstructed=True
        )
        return _analysis_from_statistics(
            sufficient,
            historical_residuals=historical_residuals,
            total_bytes=total_bytes,
            phase_model=phase_model,
        )

    @staticmethod
    def update(
        statistics_state: ResourceTrajectoryStatistics,
        sample: ResourceTrajectorySample,
        *,
        historical_residuals: Sequence[int] = (),
        total_bytes: int = 0,
        phase_model: ResourcePhaseModel | None = None,
    ) -> ResourceTrajectoryAnalysis:
        """Update scheduling evidence in O(1) amortized time for one monitor observation."""
        statistics_state.observe(sample)
        return _analysis_from_statistics(
            statistics_state,
            historical_residuals=historical_residuals,
            total_bytes=total_bytes,
            phase_model=phase_model,
        )


@dataclass(frozen=True, slots=True)
class ActiveResourceEvidence:
    """Provisional, right-censored resource evidence published by an active Run."""

    candidate_key: str
    compatibility_key: str
    parameters: Mapping[str, Any]
    hardware: str
    total_bytes: int
    current_bytes: int
    running_peak_bytes: int
    future_peak_samples: tuple[int, ...]
    resource_state: str
    elapsed_seconds: float
    phase: str | None = None
    step: int | None = None
    checkpoint_step: int | None = None
    checkpoint_elapsed_seconds: float | None = None
    checkpoint_resumable: bool = False
    throughput: float | None = None
    co_runners: int = 0
    measurement_provenance: ResourceEvidenceQuality = "active-provisional"
    trajectory: tuple[ResourceTrajectorySample, ...] = ()
    future_peak_weights: tuple[float, ...] = ()
    growth_hazard: float = -1.0
    expected_residual_growth_bytes: int = 0
    next_decision_seconds: float | None = None
    next_decision_uncertainty_seconds: float | None = None
    target_step: int | None = None
    trajectory_statistics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the bounded portable right-censored snapshot."""
        value = asdict(self)
        value["parameters"] = _portable(self.parameters)
        value["trajectory"] = [item.to_dict() for item in self.trajectory]
        value["trajectory_statistics"] = _portable(self.trajectory_statistics)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ActiveResourceEvidence:
        """Restore a previous controller snapshot as provisional evidence, never a live Run."""
        payload = {
            key: item
            for key, item in value.items()
            if key in cls.__dataclass_fields__ and key != "trajectory"
        }
        payload["parameters"] = dict(payload.get("parameters", {}))
        payload["future_peak_samples"] = tuple(
            int(item) for item in payload.get("future_peak_samples", ())
        )
        payload["future_peak_weights"] = tuple(
            float(item) for item in payload.get("future_peak_weights", ())
        )
        payload["trajectory"] = tuple(
            ResourceTrajectorySample(
                **{
                    key: item
                    for key, item in sample.items()
                    if key in ResourceTrajectorySample.__dataclass_fields__
                    and key != "phases_seen"
                },
                phases_seen=tuple(
                    str(item) for item in sample.get("phases_seen", ())
                    if isinstance(item, str)
                ),
            )
            for sample in value.get("trajectory", ())
            if isinstance(sample, Mapping)
        )
        payload["trajectory_statistics"] = dict(payload.get("trajectory_statistics", {}))
        return cls(**payload)

    @property
    def progress_since_checkpoint_seconds(self) -> float:
        if self.checkpoint_elapsed_seconds is None:
            return self.elapsed_seconds
        return max(0.0, self.elapsed_seconds - self.checkpoint_elapsed_seconds)


def oom_evidence(
    message: str,
    *,
    candidate_key: str,
    compatibility_key: str,
    hardware: str,
    total_bytes: int,
    headroom_bytes: int,
    resident_bytes: int | None,
    physical_free_bytes: int | None,
    external_bytes: int | None,
    active_co_runs: int,
    phase: str | None = None,
    step: int | None = None,
) -> OOMResourceEvidence:
    """Extract only defensible OOM information; never invent allocator precision."""
    attempted = _allocation_bytes(message)
    if attempted is not None and resident_bytes is not None:
        lower = resident_bytes + attempted
        source = "candidate-resident-plus-attempted-allocation"
    elif resident_bytes is not None:
        lower = resident_bytes
        source = "candidate-resident-observed"
    else:
        # Device headroom proves only that this *packing* failed. Without candidate attribution it
        # is not an intrinsic lower bound and must not poison heavy+small placements.
        lower = 0
        source = "placement-only-no-candidate-attribution"
    return OOMResourceEvidence(
        candidate_key=candidate_key,
        compatibility_key=compatibility_key,
        hardware=hardware,
        total_bytes=total_bytes,
        lower_bound_bytes=max(0, lower),
        lower_bound_source=source,
        headroom_bytes=max(0, headroom_bytes),
        attempted_allocation_bytes=attempted,
        candidate_resident_bytes=resident_bytes,
        physical_free_bytes=physical_free_bytes,
        external_bytes=external_bytes,
        active_co_runs=active_co_runs,
        phase=phase,
        step=step,
    )


@dataclass(frozen=True, slots=True)
class ResourceProfileObservation:
    """Portable exact or right-censored resource evidence for one Run."""

    candidate_key: str
    compatibility_key: str
    work_class: str
    parameters: Mapping[str, Any]
    fixed_arguments: Mapping[str, Any]
    fidelity: Mapping[str, int]
    hardware: str
    total_bytes: int
    state: ResourceRunState
    observed_peak_bytes: int
    peak_is_exact: bool
    duration_seconds: float
    trial: int | None = None
    seed: int | None = None
    run_id: str | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    time_to_peak_seconds: float | None = None
    time_to_stable_seconds: float | None = None
    peak_phase: str | None = None
    peak_step: int | None = None
    allocated_peak_bytes: int | None = None
    reserved_peak_bytes: int | None = None
    lower_bound_bytes: int = 0
    lower_bound_source: str | None = None
    batch_size: Any = None
    precision: Any = None
    dataset_signature: str | None = None
    code_fingerprint: str = ""
    environment_fingerprint: str = ""
    co_runners: int = 0
    throughput: float | None = None
    trajectory: tuple[Mapping[str, Any], ...] = ()
    measurement_quality: ResourceEvidenceQuality = "sampled-physical"
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    scientific_termination: str | None = None
    resource_profile_quality: str = "CENSORED"
    phases_seen: tuple[str, ...] = ()
    predicted_fit_probability: float | None = None
    placement_succeeded: bool | None = None
    admission_mode: AdmissionMode | None = None
    predicted_peak_bytes: int | None = None
    predicted_time_to_envelope_seconds: float | None = None
    checkpoint_duration_seconds: float | None = None
    trajectory_statistics: Mapping[str, Any] = field(default_factory=dict)
    observation_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["parameters"] = _portable(self.parameters)
        value["fixed_arguments"] = _portable(self.fixed_arguments)
        value["fidelity"] = dict(self.fidelity)
        value["trajectory"] = [dict(item) for item in self.trajectory]
        value["phases_seen"] = list(self.phases_seen)
        value["trajectory_statistics"] = _portable(self.trajectory_statistics)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResourceProfileObservation:
        allowed = cls.__dataclass_fields__
        payload = {key: item for key, item in value.items() if key in allowed}
        payload["parameters"] = dict(payload.get("parameters", {}))
        payload["fixed_arguments"] = dict(payload.get("fixed_arguments", {}))
        payload["fidelity"] = dict(payload.get("fidelity", {}))
        payload["trajectory"] = tuple(
            dict(item) for item in payload.get("trajectory", ()) if isinstance(item, Mapping)
        )
        payload["phases_seen"] = tuple(str(item) for item in payload.get("phases_seen", ()))
        payload["trajectory_statistics"] = dict(payload.get("trajectory_statistics", {}))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ResourcePrediction:
    """Calibrated future-envelope and duration prediction for one candidate."""

    candidate_key: str
    predicted_peak_bytes: int
    lower_bytes: int
    upper_bytes: int
    known_lower_bound_bytes: int
    predicted_time_to_envelope_seconds: float | None
    predicted_duration_seconds: float | None
    compatible_history_count: int
    exact_history_count: int
    support: str
    calibration: str
    backend: str
    samples: tuple[int, ...]
    provisional_lower_bound_bytes: int = 0
    support_score: float = 0.0
    nonconformity: float = 1.0
    future_residual_samples: tuple[int, ...] = ()
    measurement_quality: ResourceEvidenceQuality = "unavailable"
    is_provisional: bool = False
    sample_weights: tuple[float, ...] = ()
    duration_samples: tuple[float, ...] = ()
    duration_source: str = "unknown"

    @property
    def commitment_bytes(self) -> int:
        return max(self.upper_bytes, self.known_lower_bound_bytes)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["commitment_bytes"] = self.commitment_bytes
        return value


class ResourceDemandModel:
    """Small-data mixed-space bootstrap model with censored-bound support."""

    def __init__(
        self,
        observations: Sequence[ResourceProfileObservation] = (),
        *,
        active_evidence: Sequence[ActiveResourceEvidence] = (),
        placement_failures: Sequence[PlacementOOMEvidence] = (),
        parameter_schema: Mapping[str, Any] | None = None,
    ) -> None:
        self.observations = tuple(observations)
        self.active_evidence = tuple(active_evidence)
        self.placement_failures = tuple(placement_failures)
        self.parameter_schema = dict(parameter_schema or {})
        points = tuple(value.parameters for value in self.observations) + tuple(
            value.parameters for value in self.active_evidence
        )
        self.parameter_space = ParameterSpace.from_schema(self.parameter_schema, points)
        self._prediction_cache: dict[str, ResourcePrediction] = {}

    def phase_model(self, compatibility_key: str, hardware: str) -> ResourcePhaseModel:
        """Return learned phase expectations for one compatible Work/device family."""
        return ResourcePhaseModel.from_observations(
            tuple(
                value
                for value in self.observations
                if value.compatibility_key == compatibility_key and value.hardware == hardware
            )
        )

    def calibrate_fit_probability(self, probability: float) -> float:
        """Apply bounded empirical calibration without touching scientific outcomes."""
        probability = min(1.0, max(0.0, probability))
        comparable = [
            value
            for value in self.observations
            if value.predicted_fit_probability is not None
            and value.placement_succeeded is not None
            and int(float(value.predicted_fit_probability) * 10)
            == min(9, int(probability * 10))
        ]
        if not comparable:
            return probability
        successes = sum(bool(value.placement_succeeded) for value in comparable)
        # Beta prior centred on the model probability; evidence widens/corrects naturally.
        return (successes + 2.0 * probability) / (len(comparable) + 2.0)

    def checkpoint_duration(
        self, compatibility_key: str, hardware: str
    ) -> float | None:
        """Return a compatible observed checkpoint cost, or unknown when absent."""
        values = [
            float(value.checkpoint_duration_seconds)
            for value in self.observations
            if value.compatibility_key == compatibility_key
            and value.hardware == hardware
            and value.checkpoint_duration_seconds is not None
            and value.checkpoint_duration_seconds >= 0
        ]
        return statistics.median(values) if values else None

    def predict(
        self,
        *,
        candidate_key: str,
        compatibility_key: str,
        parameters: Mapping[str, Any],
        hardware: str,
        total_bytes: int,
        user_minimum_bytes: int = 0,
        observed_prefix_peak_bytes: int = 0,
        phase: str | None = None,
    ) -> ResourcePrediction:
        cache_key = _digest(
            {
                "candidate": candidate_key,
                "compatibility": compatibility_key,
                "parameters": _portable(parameters),
                "hardware": hardware,
                "total": total_bytes,
                "minimum": user_minimum_bytes,
                "prefix_peak": observed_prefix_peak_bytes,
                "phase": phase,
            }
        )
        cached = self._prediction_cache.get(cache_key)
        if cached is not None:
            return cached

        def remember(value: ResourcePrediction) -> ResourcePrediction:
            if len(self._prediction_cache) >= _HISTORY_LIMIT:
                self._prediction_cache.clear()
            self._prediction_cache[cache_key] = value
            return value

        compatible = [
            value
            for value in self.observations
            if value.compatibility_key == compatibility_key and value.hardware == hardware
        ]
        candidate_matches = [value for value in compatible if value.candidate_key == candidate_key]
        evidence = (
            candidate_matches
            or sorted(
                compatible,
                key=lambda value: (
                    _mixed_distance(parameters, value.parameters, self.parameter_space),
                    value.timestamp_utc,
                ),
            )[: min(24, len(compatible))]
        )
        active_compatible = [
            value
            for value in self.active_evidence
            if value.compatibility_key == compatibility_key and value.hardware == hardware
        ]
        active_matches = [
            value for value in active_compatible if value.candidate_key == candidate_key
        ]
        active_distances = sorted(
            (
                _mixed_distance(parameters, value.parameters, self.parameter_space),
                value,
            )
            for value in active_compatible
        )
        support_scale = _active_support_scale(active_distances, self.parameter_space)
        active_support = max(
            (
                math.exp(-distance / max(support_scale, 1e-12))
                for distance, _value in active_distances
            ),
            default=0.0,
        )
        active_near = active_matches or [
            value
            for distance, value in active_distances[: min(12, len(active_distances))]
            if math.exp(-distance / max(support_scale, 1e-12)) > 1e-6
        ]
        bounds = [
            value.lower_bound_bytes
            for value in compatible
            if value.candidate_key == candidate_key and value.lower_bound_bytes > 0
        ]
        # A neighbour informs the distribution but cannot create an intrinsic lower bound for a
        # different candidate.  Only the exact resource identity is logically binding.
        provisional_bounds = [value.running_peak_bytes for value in active_matches]
        known_lower = max([user_minimum_bytes, observed_prefix_peak_bytes, *bounds], default=0)
        provisional_lower = max([known_lower, *provisional_bounds], default=known_lower)
        exact_peaks = [value.observed_peak_bytes for value in evidence if value.peak_is_exact]
        censored = [
            max(value.observed_peak_bytes, value.lower_bound_bytes)
            for value in evidence
            if not value.peak_is_exact
        ]
        exact_durations = [
            value.duration_seconds
            for value in candidate_matches
            if value.duration_seconds > 0
        ]
        nearby_durations = [
            value.duration_seconds for value in evidence if value.duration_seconds > 0
        ]
        compatible_durations = [
            value.duration_seconds for value in compatible if value.duration_seconds > 0
        ]
        active_duration_estimates = [
            value.elapsed_seconds * value.target_step / value.step
            for value in active_near
            if value.target_step is not None
            and value.target_step > 0
            and value.step is not None
            and value.step > 0
            and value.elapsed_seconds > 0
        ]
        if exact_durations:
            durations, duration_source = exact_durations, "same-candidate-history"
        elif nearby_durations:
            durations, duration_source = nearby_durations, "nearby-compatible-history"
        elif active_duration_estimates:
            durations, duration_source = active_duration_estimates, "live-step-rate"
        elif compatible_durations:
            durations, duration_source = compatible_durations, "compatible-history"
        else:
            durations, duration_source = [], "unknown"
        peak_times = [
            value.time_to_peak_seconds
            for value in evidence
            if value.time_to_peak_seconds is not None and value.time_to_peak_seconds >= 0
        ]
        if not exact_peaks:
            predicted = max(provisional_lower, max(censored, default=0), user_minimum_bytes)
            if active_near:
                active_draws: list[int] = []
                active_weights: list[float] = []
                for value in active_near:
                    distance = _mixed_distance(parameters, value.parameters, self.parameter_space)
                    proximity = 1.0 if value in active_matches else math.exp(
                        -distance / max(support_scale, 1e-12)
                    )
                    weights = _normalized_weights(
                        value.future_peak_weights, len(value.future_peak_samples)
                    )
                    for sample, weight in zip(value.future_peak_samples, weights, strict=True):
                        active_draws.append(max(provisional_lower, sample))
                        active_weights.append(weight * proximity)
                samples, sample_weights = _coalesce_weighted_samples(active_draws, active_weights)
                if not samples:
                    samples = (provisional_lower, total_bytes)
                    sample_weights = (0.5, 0.5)
                upper = max(samples)
                support = "active-same-candidate" if active_matches else "active-near-compatible"
                quality: ResourceEvidenceQuality = "active-provisional"
            else:
                upper = max(predicted, total_bytes)
                samples = tuple(sorted({known_lower, predicted, upper}))
                sample_weights = _normalized_weights((), len(samples))
                support = "censored-only" if evidence else "cold-start"
                quality = "lower-bound-only" if evidence else "unavailable"
            return remember(
                ResourcePrediction(
                    candidate_key,
                    predicted,
                    known_lower,
                    upper,
                    known_lower,
                    statistics.median(peak_times) if peak_times else None,
                    statistics.median(durations) if durations else None,
                    len(compatible),
                    0,
                    support,
                    "poor",
                    "mixed-live-survival-v3",
                    samples,
                    provisional_lower,
                    1.0 if active_matches else active_support if active_near else 0.0,
                    0.0 if active_matches else 1.0 - active_support if active_near else 1.0,
                    tuple(max(0, value - provisional_lower) for value in samples),
                    quality,
                    bool(active_near),
                    sample_weights,
                    tuple(float(value) for value in durations),
                    duration_source,
                )
            )
        if not candidate_matches and len(exact_peaks) < 2:
            # One neighbouring configuration supplies a location hint but no empirical evidence
            # about how resource demand varies over the authored mixed space.  Keep cold-start
            # upper uncertainty at device capacity until at least one between-candidate residual
            # can exist; this is a model-identifiability condition, not a user threshold.
            predicted = max(known_lower, exact_peaks[0])
            distance = min(
                (
                    _mixed_distance(parameters, value.parameters, self.parameter_space)
                    for value in evidence
                ),
                default=1.0,
            )
            samples = tuple(
                sorted(
                    {
                        known_lower,
                        predicted,
                        int(predicted + distance * (total_bytes - predicted)),
                        total_bytes,
                    }
                )
            )
            return remember(
                ResourcePrediction(
                    candidate_key,
                    predicted,
                    known_lower,
                    max(predicted, total_bytes),
                    known_lower,
                    statistics.median(peak_times) if peak_times else None,
                    statistics.median(durations) if durations else None,
                    len(compatible),
                    0,
                    "sparse-near-compatible",
                    "poor",
                    "schema-support-bootstrap-v3",
                    samples,
                    provisional_lower,
                    max(0.0, 1.0 - distance),
                    distance,
                    tuple(max(0, value - provisional_lower) for value in samples),
                    "sampled-physical",
                    False,
                    _normalized_weights((), len(samples)),
                    tuple(float(value) for value in durations),
                    duration_source,
                )
            )
        weighted = _weighted_peaks(parameters, evidence, self.parameter_space)
        samples = _deterministic_bootstrap(weighted)
        residuals = _loo_underprediction_residuals(evidence, self.parameter_space)
        calibrated = tuple(
            max(known_lower, sample + residuals[index % len(residuals)])
            for index, sample in enumerate(samples)
        )
        # Nearby censored/OOM observations are constraints on the predictive tail, not fabricated
        # exact peaks. Their influence fades continuously with authored-space distance.
        censored_tail = [
            int(
                max(value.observed_peak_bytes, value.lower_bound_bytes)
                * max(
                    0.0,
                    1.0
                    - _mixed_distance(parameters, value.parameters, self.parameter_space),
                )
            )
            for value in evidence
            if not value.peak_is_exact
            and max(value.observed_peak_bytes, value.lower_bound_bytes) > 0
        ]
        if censored_tail:
            calibrated = tuple(sorted((*calibrated, max(censored_tail))))
        predicted = int(statistics.median(calibrated))
        lower = max(known_lower, min(calibrated))
        upper = max(lower, max(calibrated))
        nearest = min(
            (
                _mixed_distance(parameters, value.parameters, self.parameter_space)
                for value in evidence
            ),
            default=1.0,
        )
        loo_support = _empirical_support_distance(evidence, self.parameter_space)
        nonconformity = nearest / max(loo_support, 1e-12) if loo_support > 0 else nearest
        if nonconformity > 1.0:
            tail = int(upper + min(1.0, nonconformity - 1.0) * (total_bytes - upper))
            calibrated = tuple(sorted((*calibrated, max(upper, tail))))
            upper = max(calibrated)
        # A known later phase peak remains in the future envelope even if a live prefix currently
        # appears stable in training.
        if phase and phase != "validation":
            validation = [
                value.observed_peak_bytes
                for value in evidence
                if value.peak_is_exact and value.peak_phase == "validation"
            ]
            upper = max(upper, max(validation, default=0))
        return remember(
            ResourcePrediction(
                candidate_key,
                max(predicted, known_lower),
                lower,
                upper,
                known_lower,
                statistics.median(peak_times) if peak_times else None,
                statistics.median(durations) if durations else None,
                len(compatible),
                sum(value.peak_is_exact for value in candidate_matches),
                "exact-candidate" if candidate_matches else "near-compatible",
                "good"
                if len(exact_peaks) >= 8
                else "moderate"
                if len(exact_peaks) >= 3
                else "poor",
                "schema-loo-bootstrap-v3",
                tuple(sorted(calibrated)),
                provisional_lower,
                1.0 / (1.0 + nonconformity),
                nonconformity,
                tuple(max(0, value - provisional_lower) for value in calibrated),
                (
                    "terminal-high-quality"
                    if all(
                        value.measurement_quality == "terminal-high-quality"
                        for value in evidence
                        if value.peak_is_exact
                    )
                    else "sampled-physical"
                ),
                False,
                _normalized_weights((), len(calibrated)),
                tuple(float(value) for value in durations),
                duration_source,
            )
        )

    def aggregate_throughput(self, compatibility_key: str, concurrency: int) -> float | None:
        values = [
            value.throughput
            for value in self.observations
            if value.compatibility_key == compatibility_key
            and value.co_runners + 1 == concurrency
            and value.throughput is not None
            and value.throughput > 0
        ]
        values.extend(
            value.throughput
            for value in self.active_evidence
            if value.compatibility_key == compatibility_key
            and value.co_runners + 1 == concurrency
            and value.throughput is not None
            and value.throughput > 0
        )
        return statistics.median(values) * concurrency if values else None

    def adjusted_duration(
        self,
        compatibility_key: str,
        duration_seconds: float | None,
        concurrency: int,
    ) -> float | None:
        """Condition a candidate duration on observed compatible co-location.

        Duration evidence is preferred because it already includes every source of slowdown.
        Throughput supplies the equivalent ratio when comparable durations are not available.
        No synthetic slowdown factor is introduced when either side lacks support.
        """
        if duration_seconds is None or concurrency <= 1:
            return duration_seconds
        solo_durations = [
            value.duration_seconds
            for value in self.observations
            if value.compatibility_key == compatibility_key
            and value.co_runners == 0
            and value.duration_seconds > 0
        ]
        packed_durations = [
            value.duration_seconds
            for value in self.observations
            if value.compatibility_key == compatibility_key
            and value.co_runners + 1 == concurrency
            and value.duration_seconds > 0
        ]
        if solo_durations and packed_durations:
            ratio = statistics.median(packed_durations) / statistics.median(solo_durations)
            return duration_seconds * ratio
        solo_rates = [
            value.throughput
            for value in self.observations
            if value.compatibility_key == compatibility_key
            and value.co_runners == 0
            and value.throughput is not None
            and value.throughput > 0
        ]
        packed_rates = [
            value.throughput
            for value in self.observations
            if value.compatibility_key == compatibility_key
            and value.co_runners + 1 == concurrency
            and value.throughput is not None
            and value.throughput > 0
        ]
        if solo_rates and packed_rates:
            return (
                duration_seconds * statistics.median(solo_rates) / statistics.median(packed_rates)
            )
        return duration_seconds


@dataclass(frozen=True, slots=True)
class ActiveResourceCommitment:
    """Future-memory commitment and rollback context for one active Run."""

    candidate_key: str
    commitment_bytes: int
    current_bytes: int = 0
    running_peak_bytes: int = 0
    future_peak_samples: tuple[int, ...] = ()
    remaining_seconds: float | None = None
    resource_state: str = "RAMPING"
    phase: str | None = None
    step: int | None = None
    checkpoint_step: int | None = None
    checkpoint_elapsed_seconds: float | None = None
    checkpoint_resumable: bool = False
    elapsed_seconds: float = 0.0
    measurement_provenance: ResourceEvidenceQuality = "unavailable"
    admission_mode: AdmissionMode = "SAFE_ADMISSION"
    exploration_signature: str | None = None
    trial: int | None = None
    seed: int | None = None
    future_peak_weights: tuple[float, ...] = ()
    growth_hazard: float = -1.0
    expected_residual_growth_bytes: int = 0
    next_decision_seconds: float | None = None
    next_decision_uncertainty_seconds: float | None = None
    checkpoint_request_path: str | None = None
    evidence_cycles: int = 0
    phase_hazards: Mapping[str, float] = field(default_factory=dict)
    tail_probability: float = 1.0
    last_material_peak_time: float | None = None
    checkpoint_cadence_seconds: float | None = None
    checkpoint_duration_seconds: float | None = None
    throughput: float | None = None


@dataclass(frozen=True, slots=True)
class GPUResourceState:
    """One legally visible GPU reconciled with physical and controller-owned usage."""

    index: int
    token: str
    hardware: str
    total_bytes: int
    free_bytes: int
    external_bytes: int
    active: tuple[ActiveResourceCommitment, ...] = ()
    run_cap: int = 1

    @property
    def future_committed_bytes(self) -> int:
        return sum(value.commitment_bytes for value in self.active)

    @property
    def predicted_headroom_bytes(self) -> int:
        return max(0, self.total_bytes - self.external_bytes - self.future_committed_bytes)

    @property
    def provisional_committed_bytes(self) -> int:
        return sum(
            _weighted_quantile(value.future_peak_samples, value.future_peak_weights, 0.5)
            if value.future_peak_samples
            else max(value.current_bytes, value.running_peak_bytes)
            for value in self.active
        )

    @property
    def provisional_headroom_bytes(self) -> int:
        return max(0, self.total_bytes - self.external_bytes - self.provisional_committed_bytes)

    @property
    def admission_headroom_bytes(self) -> int:
        """Return the strictest physical/current or future-envelope capacity bound."""
        return min(self.free_bytes, self.predicted_headroom_bytes)


@dataclass(frozen=True, slots=True)
class ScientificActionValue:
    """Controller-to-resource contract on a bounded scheduling scale."""

    rank: int
    normalized_value: float | None = None
    uncertainty: float | None = None
    expected_cost_seconds: float | None = None
    evidence_kind: str = "rank-only"
    resource_blocked_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateResourceAction:
    """Scientifically ranked action with hardware-conditioned resource predictions."""

    key: str
    specification: Mapping[str, Any]
    scientific_value: float
    prediction: ResourcePrediction
    device_predictions: Mapping[int, ResourcePrediction] = field(default_factory=dict)
    value_contract: ScientificActionValue | None = None

    def prediction_for(self, device: GPUResourceState) -> ResourcePrediction:
        """Return the hardware-conditioned prediction for one legally visible device."""
        return self.device_predictions.get(device.index, self.prediction)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Auditable safe, exploratory, blocked or infeasible placement decision."""

    candidate_key: str
    trial: int | None
    state: PlacementState
    target_gpu: int | None
    reason: str
    scientific_value: float
    prediction: Mapping[str, Any]
    devices: tuple[Mapping[str, Any], ...]
    fit_probability: float
    expected_cost_seconds: float | None
    earliest_opportunity_seconds: float | None = None
    backfill: bool = False
    displaced_candidate: str | None = None
    admission_mode: AdmissionMode = "SAFE_ADMISSION"
    rollback_cost_seconds: float = 0.0
    resource_information_value: float = 0.0
    exploration_signature: str | None = None
    exploration_evaluation: Mapping[str, Any] | None = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExplorationEvaluation:
    """Dimensionally coherent comparison of WAIT and one bounded ladder increment."""

    candidate: str
    gpu: int
    physical_free_bytes: int
    provisional_headroom_bytes: int
    resource_states: tuple[str, ...]
    peak_hazard: float
    fit_probability: float
    scientific_value: float
    normalized_scientific_value: float
    scientific_value_rate: float | None
    resource_information_value: float
    rollback_seconds: float
    rollback_value: float
    interference_cost: float
    wait_regret: float
    final_delta_value: float
    accepted: bool
    rejection_reason: str
    plan: str
    experiment_signature: str | None = None
    evidence_cycles: int = 0
    last_material_peak_time: float | None = None
    phase_hazards: Mapping[str, float] = field(default_factory=dict)
    tail_probability: float = 0.0
    checkpoint_cost_seconds: float | None = None
    rollback_after_checkpoint_seconds: float | None = None
    wait_regret_details: Mapping[str, Any] = field(default_factory=dict)
    fit_uncertainty_sources: tuple[str, ...] = ()
    evaluation_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WaitRegretTracker:
    """Integrate useful idle GPU opportunity without a timeout or age bonus."""

    def __init__(self, persisted: Mapping[str, Any] | None = None) -> None:
        self._states: dict[int, dict[str, Any]] = {}
        if isinstance(persisted, Mapping):
            raw_states = persisted.get("states", {})
            if isinstance(raw_states, Mapping):
                for key, raw in raw_states.items():
                    if isinstance(raw, Mapping):
                        self._states[int(key)] = {
                            "time": None,
                            "regret": max(0.0, float(raw.get("regret", 0.0))),
                            "signature": str(raw.get("signature", "restored")),
                            "rate": max(0.0, float(raw.get("rate", 0.0))),
                            "usable": max(0.0, min(1.0, float(raw.get("usable", 0.0)))),
                            "reset_reason": raw.get("reset_reason"),
                        }

    def update(
        self,
        device: GPUResourceState,
        actions: Sequence[CandidateResourceAction],
        normalized_values: Mapping[str, float],
        *,
        now: float,
    ) -> float:
        rates = [
            normalized_values[action.key] / duration
            for action in actions
            if (duration := _candidate_duration(action.prediction_for(device), device)) is not None
            and duration > 0
            and action.prediction_for(device).known_lower_bound_bytes
            <= device.provisional_headroom_bytes
        ]
        best_rate = max(rates, default=0.0)
        usable = (
            min(device.free_bytes, device.provisional_headroom_bytes) / max(1, device.total_bytes)
            if actions and len(device.active) < device.run_cap
            else 0.0
        )
        signature = _digest(
            {
                "active": [
                    {
                        "candidate": value.candidate_key,
                        "state": value.resource_state,
                        "hazard_band": round(_commitment_growth_hazard(value), 1),
                        "checkpoint": value.checkpoint_step,
                    }
                    for value in device.active
                ],
                "pending": [value.key for value in actions],
                "usable_band": round(usable, 1),
            }
        )
        previous = self._states.get(device.index)
        regret = float(previous.get("regret", 0.0)) if previous is not None else 0.0
        if previous is not None and previous.get("time") is not None:
            # Close the previous physical integration segment at its previous rate. A changing
            # frontier/hazard/checkpoint starts a new segment but never erases sunk idle loss.
            regret += max(0.0, now - float(previous["time"])) * float(
                previous.get("usable", 0.0)
            ) * float(previous.get("rate", 0.0))
        reset_reason = None
        if not actions:
            regret, reset_reason = 0.0, "no-pending-scientific-work"
        elif usable <= 0:
            regret, reset_reason = 0.0, "no-usable-idle-capacity"
        elif best_rate <= 0:
            regret, reset_reason = 0.0, "no-physically-feasible-opportunity"
        self._states[device.index] = {
            "time": now,
            "regret": regret,
            "signature": signature,
            "rate": best_rate,
            "usable": usable,
            "reset_reason": reset_reason,
        }
        return regret

    def reset(self, gpu: int, *, reason: str = "capacity-filled") -> None:
        self._states[gpu] = {
            "time": None,
            "regret": 0.0,
            "signature": "reset",
            "rate": 0.0,
            "usable": 0.0,
            "reset_reason": reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "wait_regret_version": 1,
            "states": {
                str(gpu): {
                    key: value
                    for key, value in state.items()
                    if key != "time"
                }
                for gpu, state in self._states.items()
            },
        }

    def diagnostics(self, gpu: int) -> Mapping[str, Any]:
        state = self._states.get(gpu, {})
        return {
            "accumulated_opportunity_loss": float(state.get("regret", 0.0)),
            "current_idle_usable_fraction": float(state.get("usable", 0.0)),
            "scientific_opportunity_rate": float(state.get("rate", 0.0)),
            "segment_started_monotonic": state.get("time"),
            "reset_reason": state.get("reset_reason"),
        }


class GPUPlacementPlanner:
    """Deterministic best-fit placement over a bounded scientific frontier."""

    def __init__(
        self,
        model: ResourceDemandModel,
        *,
        wait_regret: WaitRegretTracker | None = None,
    ) -> None:
        self.model = model
        self.wait_regret = wait_regret or WaitRegretTracker()
        self.last_exploration_evaluations: tuple[ExplorationEvaluation, ...] = ()

    def update_model(self, model: ResourceDemandModel) -> None:
        self.model = model

    def place(
        self,
        actions: Sequence[CandidateResourceAction],
        devices: Sequence[GPUResourceState],
        *,
        max_launches: int,
        now: float | None = None,
    ) -> tuple[tuple[AdmissionDecision, ...], tuple[AdmissionDecision, ...]]:
        ranked = sorted(
            actions[:_FRONTIER_LIMIT],
            key=lambda value: (
                value.value_contract.rank if value.value_contract is not None else math.inf,
                -value.scientific_value,
                value.key,
            ),
        )
        values = _normalized_scientific_values(ranked)
        decision_time = float(now) if now is not None else 0.0
        regrets = {
            device.index: self.wait_regret.update(
                device,
                tuple(
                    action
                    for action in ranked
                    if action.prediction_for(device).known_lower_bound_bytes
                    <= device.provisional_headroom_bytes
                    and not self._hurts_throughput(action, device)
                ),
                values,
                now=decision_time,
            )
            for device in devices
        }
        mutable = list(devices)
        admitted: list[AdmissionDecision] = []
        blocked: list[AdmissionDecision] = []
        reservation = self._reservation(ranked, mutable)
        for action in ranked:
            if len(admitted) >= max_launches:
                break
            choices: list[tuple[int, float, float, int]] = []
            device_notes: list[dict[str, Any]] = []
            for position, device in enumerate(mutable):
                prediction = action.prediction_for(device)
                state, reason = self._fit_state(action, device)
                probability = _fit_probability(prediction, device.admission_headroom_bytes)
                adjusted_duration = self.model.adjusted_duration(
                        _action_compatibility(action, device),
                        prediction.predicted_duration_seconds,
                        len(device.active) + 1,
                    )
                duration = adjusted_duration if adjusted_duration is not None else math.inf
                slack = device.admission_headroom_bytes - prediction.commitment_bytes
                if state == "ADMITTED" and self._hurts_throughput(action, device):
                    state, reason = (
                        "RESOURCE_BLOCKED",
                        "predicted-aggregate-throughput-would-not-improve",
                    )
                if state == "ADMITTED" and reservation is not None:
                    reserved_key, reserved_gpu, window = reservation
                    if action.key != reserved_key and device.index == reserved_gpu:
                        if window is not None and duration > window:
                            state, reason = (
                                "RESOURCE_BLOCKED",
                                "reserved-window-for-higher-value-heavy-action",
                            )
                device_notes.append(
                    {
                        "gpu": device.index,
                        "token": device.token,
                        "hardware": device.hardware,
                        "total_bytes": device.total_bytes,
                        "physical_free_bytes": device.free_bytes,
                        "external_bytes": device.external_bytes,
                        "future_committed_bytes": device.future_committed_bytes,
                        "predicted_headroom_bytes": device.predicted_headroom_bytes,
                        "admission_headroom_bytes": device.admission_headroom_bytes,
                        "active_runs": len(device.active),
                        "fit_probability": probability,
                        "prediction": prediction.to_dict(),
                        "state": state,
                        "reason": reason,
                    }
                )
                if state == "ADMITTED":
                    # Least non-negative slack is best fit. Risk and duration distinguish almost
                    # equivalent packings without changing scientific ranking.
                    choices.append((slack, -probability, duration, position))
            if not choices:
                infeasible = bool(device_notes) and all(
                    note["state"] == "RESOURCE_INFEASIBLE_ON_DEVICE_TYPE" for note in device_notes
                )
                earliest_values = [
                    _device_fit_time(
                        action.prediction_for(device).commitment_bytes,
                        device,
                    )
                    for device in mutable
                ]
                earliest = min(
                    (value for value in earliest_values if value is not None),
                    default=None,
                )
                blocked.append(
                    AdmissionDecision(
                        action.key,
                        _trial(action.specification),
                        "RESOURCE_INFEASIBLE_ON_DEVICE_TYPE" if infeasible else "RESOURCE_BLOCKED",
                        None,
                        "known lower bound exceeds every compatible device type"
                        if infeasible
                        else "predicted future GPU-memory envelope does not fit now",
                        action.scientific_value,
                        action.prediction.to_dict(),
                        tuple(device_notes),
                        max((float(note["fit_probability"]) for note in device_notes), default=0.0),
                        action.prediction.predicted_duration_seconds,
                        earliest,
                    )
                )
                continue
            _slack, negative_probability, _duration, selected = min(choices)
            device = mutable[selected]
            selected_prediction = action.prediction_for(device)
            commitment = ActiveResourceCommitment(
                action.key,
                selected_prediction.commitment_bytes,
                remaining_seconds=self.model.adjusted_duration(
                    _action_compatibility(action, device),
                    selected_prediction.predicted_duration_seconds,
                    len(device.active) + 1,
                ),
                trial=_trial(action.specification),
                seed=_optional_int(action.specification.get("seed")),
            )
            mutable[selected] = GPUResourceState(
                device.index,
                device.token,
                device.hardware,
                device.total_bytes,
                device.free_bytes,
                device.external_bytes,
                (*device.active, commitment),
                device.run_cap,
            )
            higher = next(
                (item for item in blocked if item.scientific_value > action.scientific_value), None
            )
            admitted.append(
                AdmissionDecision(
                    action.key,
                    _trial(action.specification),
                    "ADMITTED",
                    device.index,
                    "highest-value feasible action"
                    if higher is None
                    else "safe backfill while a higher-value action is resource-blocked",
                    action.scientific_value,
                    selected_prediction.to_dict(),
                    tuple(device_notes),
                    -negative_probability,
                    self.model.adjusted_duration(
                        _action_compatibility(action, device),
                        selected_prediction.predicted_duration_seconds,
                        len(device.active) + 1,
                    ),
                    backfill=higher is not None,
                    displaced_candidate=higher.candidate_key if higher is not None else None,
                    admission_mode="SAFE_ADMISSION",
                )
            )
            self.wait_regret.reset(device.index, reason="safe-work-admitted")
        # Conservative upper envelopes are deliberately broad at cold start.  Once live evidence
        # exists, evaluate one incremental co-location experiment rather than waiting for a long
        # Run to terminate.  This evolves the same planner: there is no compatibility scheduler.
        remaining_launches = max(0, max_launches - len(admitted))
        evaluations: list[ExplorationEvaluation] = []
        if remaining_launches and blocked:
            by_key = {value.key: value for value in ranked}
            experiments: list[
                tuple[
                    float,
                    float,
                    int,
                    str,
                    CandidateResourceAction,
                    GPUResourceState,
                    str,
                    float,
                    float,
                ]
            ] = []
            for blocked_decision in blocked:
                uncertain_action = by_key.get(blocked_decision.candidate_key)
                if uncertain_action is None:
                    continue
                for device in mutable:
                    policy_reason = next(
                        (
                            str(note.get("reason"))
                            for note in blocked_decision.devices
                            if note.get("gpu") == device.index
                        ),
                        "",
                    )
                    evaluated = self._exploration_value(
                        uncertain_action,
                        device,
                        mutable,
                        normalized_scientific_value=values.get(uncertain_action.key, 0.0),
                        wait_regret=regrets.get(device.index, 0.0),
                        policy_reason=policy_reason,
                    )
                    evaluations.append(evaluated)
                    if not evaluated.accepted:
                        continue
                    experiments.append(
                        (
                            -evaluated.final_delta_value,
                            -uncertain_action.scientific_value,
                            device.index,
                            uncertain_action.key,
                            uncertain_action,
                            device,
                            str(evaluated.experiment_signature),
                            evaluated.rollback_seconds,
                            evaluated.resource_information_value,
                        )
                    )
            if experiments:
                (
                    _negative_value,
                    _negative_science,
                    _index,
                    _candidate_sort_key,
                    uncertain_action,
                    selected_device,
                    signature,
                    rollback,
                    information,
                ) = min(experiments)
                selected_prediction = uncertain_action.prediction_for(selected_device)
                probability = _joint_fit_probability(selected_prediction, selected_device)
                admitted.append(
                    AdmissionDecision(
                        uncertain_action.key,
                        _trial(uncertain_action.specification),
                        "ADMITTED",
                        selected_device.index,
                        "checkpoint-aware incremental resource experiment; live trajectory "
                        "makes learning/using idle capacity more valuable than waiting",
                        uncertain_action.scientific_value,
                        selected_prediction.to_dict(),
                        next(
                            value.devices
                            for value in blocked
                            if value.candidate_key == uncertain_action.key
                        ),
                        probability,
                        self.model.adjusted_duration(
                            _action_compatibility(uncertain_action, selected_device),
                            selected_prediction.predicted_duration_seconds,
                            len(selected_device.active) + 1,
                        ),
                        admission_mode="EXPLORATORY_ADMISSION",
                        rollback_cost_seconds=rollback,
                        resource_information_value=information,
                        exploration_signature=signature,
                        exploration_evaluation=next(
                            value.to_dict()
                            for value in evaluations
                            if value.candidate == uncertain_action.key
                            and value.gpu == selected_device.index
                        ),
                    )
                )
                self.wait_regret.reset(selected_device.index)
                blocked = [
                    value for value in blocked if value.candidate_key != uncertain_action.key
                ]
        self.last_exploration_evaluations = tuple(evaluations)
        return tuple(admitted), tuple(blocked)

    def _exploration_value(
        self,
        action: CandidateResourceAction,
        device: GPUResourceState,
        all_devices: Sequence[GPUResourceState],
        *,
        normalized_scientific_value: float,
        wait_regret: float,
        policy_reason: str = "",
    ) -> ExplorationEvaluation:
        """Compare EXPLORE with WAIT until the next evidence-bearing event."""
        prediction = action.prediction_for(device)
        signature = _resource_experiment_signature(action, device)

        def rejected(reason: str, *, probability: float = 0.0) -> ExplorationEvaluation:
            return ExplorationEvaluation(
                action.key,
                device.index,
                device.free_bytes,
                device.provisional_headroom_bytes,
                tuple(value.resource_state for value in device.active),
                max((_commitment_growth_hazard(value) for value in device.active), default=0.0),
                probability,
                action.scientific_value,
                normalized_scientific_value,
                None,
                0.0,
                0.0,
                0.0,
                0.0,
                wait_regret,
                -math.inf,
                False,
                reason,
                "WAIT",
                signature,
            )

        if len(device.active) >= device.run_cap:
            return rejected("RUN_CAP")
        if device.active and not any(value.current_bytes > 0 for value in device.active):
            return rejected("AWAITING_LIVE_TELEMETRY")
        if policy_reason == "reserved-window-for-higher-value-heavy-action":
            return rejected("PROTECTED_HEAVY_WINDOW")
        if self._hurts_throughput(action, device):
            return rejected("THROUGHPUT_REGRESSION")
        if prediction.known_lower_bound_bytes > device.provisional_headroom_bytes:
            return rejected("HARD_LOWER_BOUND")
        if any(value.admission_mode == "EXPLORATORY_ADMISSION" for value in device.active):
            return rejected("UNVALIDATED_LADDER_STEP")
        # Equivalent GPUs share live evidence. Never duplicate the same unvalidated ladder step,
        # but allow different resource questions to be explored in parallel on larger grants.
        # One sibling always remains outside an unvalidated packing as the protected progress
        # lane; this is a structural invariant rather than a configurable GPU count.
        interchangeable = tuple(
            other for other in all_devices if other.hardware == device.hardware
        )
        exploring = tuple(
            value
            for other in interchangeable
            for value in other.active
            if value.admission_mode == "EXPLORATORY_ADMISSION"
        )
        if any(value.exploration_signature == signature for value in exploring):
            return rejected("DUPLICATE_EXPERIMENT")
        exploration_lanes = sum(
            any(value.admission_mode == "EXPLORATORY_ADMISSION" for value in other.active)
            for other in interchangeable
        )
        if len(interchangeable) >= 2 and exploration_lanes >= len(interchangeable) - 1:
            return rejected("PROTECTED_LANE")
        if any(
            failure.hardware == device.hardware
            and failure.candidate_key == prediction.candidate_key
            and failure.resident_candidates
            == tuple(sorted(value.candidate_key for value in device.active))
            and device.provisional_headroom_bytes <= failure.headroom_bytes
            for failure in self.model.placement_failures
        ):
            return rejected("KNOWN_FAILED_PLACEMENT")
        if any(
            failure.experiment_signature == signature for failure in self.model.placement_failures
        ):
            return rejected("DUPLICATE_FAILED_EXPERIMENT")
        probability = self.model.calibrate_fit_probability(
            _joint_fit_probability(prediction, device)
        )
        if probability <= 0:
            return rejected("HARD_PHYSICAL_CAPACITY", probability=probability)

        candidate_duration = _candidate_duration(prediction, device)
        horizon = _next_decision_horizon(prediction, device)
        scientific_rate = (
            normalized_scientific_value / candidate_duration
            if candidate_duration is not None and candidate_duration > 0
            else None
        )
        opportunity_rates = [
            1.0 / value.remaining_seconds
            for value in device.active
            if value.remaining_seconds is not None and value.remaining_seconds > 0
        ]
        if scientific_rate is not None:
            opportunity_rates.append(scientific_rate)
        opportunity_rate = max(opportunity_rates, default=0.0)

        peak_hazard = max(
            (_commitment_growth_hazard(value) for value in device.active), default=0.0
        )
        expected_growth = sum(value.expected_residual_growth_bytes for value in device.active)
        physical_margin = max(
            0,
            device.free_bytes
            - prediction.known_lower_bound_bytes
            - expected_growth,
        )
        pressure = 1.0 - physical_margin / max(1, device.free_bytes)
        probability *= max(0.0, 1.0 - peak_hazard * max(0.0, pressure))
        if probability <= 0:
            return rejected("HIGH_GROWTH_HAZARD", probability=probability)

        resident_rollback_by_run = [
            value.elapsed_seconds
            if not value.checkpoint_resumable
            else max(0.0, value.elapsed_seconds - (value.checkpoint_elapsed_seconds or 0.0))
            for value in device.active
        ]
        resident_rollback = sum(resident_rollback_by_run)
        failed_new_progress = (horizon / 2.0) if horizon is not None else 0.0
        rollback = resident_rollback * peak_hazard + failed_new_progress
        rollback_value = (1.0 - probability) * rollback * opportunity_rate

        entropy = _binary_entropy(probability)
        transfer = sum(other.hardware == device.hardware for other in all_devices) / max(
            1, len(all_devices)
        )
        # With no wall-time evidence, keep duration explicitly unknown. One normalized future
        # scheduling decision is still a coherent value-of-information unit; this replaces the
        # former and highly distorting fiction that every unknown Run lasts one second.
        scheduling_window_value = (
            opportunity_rate * horizon
            if opportunity_rate > 0 and horizon is not None
            else 1.0
        )
        information = (
            entropy
            * transfer
            * scheduling_window_value
            * max(0.0, 1.0 - peak_hazard)
            * (device.free_bytes / max(1, device.total_bytes))
        )
        science_completed = (
            probability
            * scientific_rate
            * min(candidate_duration, horizon)
            if scientific_rate is not None
            and candidate_duration is not None
            and horizon is not None
            else 0.0
        )
        before = self.model.aggregate_throughput(
            _action_compatibility(action, device), len(device.active)
        )
        after = self.model.aggregate_throughput(
            _action_compatibility(action, device), len(device.active) + 1
        )
        interference = (
            max(0.0, (before - after) / before)
            if before is not None and after is not None and before > 0
            else 0.0
        )
        interference_value = interference * opportunity_rate * (horizon or 0.0)
        expected_value = (
            science_completed + information + wait_regret - rollback_value - interference_value
        )
        checkpointable = [
            bool(
                value.checkpoint_request_path
                and (
                    value.checkpoint_elapsed_seconds is None
                    or value.elapsed_seconds > value.checkpoint_elapsed_seconds
                )
            )
            for value in device.active
        ]
        checkpoint_durations = [
            value.checkpoint_duration_seconds
            for value in device.active
            if value.checkpoint_duration_seconds is not None
            and value.checkpoint_duration_seconds >= 0
        ]
        checkpoint_cost = (
            statistics.median(checkpoint_durations)
            if checkpoint_durations
            else self.model.checkpoint_duration(
                _action_compatibility(action, device), device.hardware
            )
        )
        rollback_after_checkpoint = sum(
            0.0 if can_checkpoint else rollback_now
            for rollback_now, can_checkpoint in zip(
                resident_rollback_by_run, checkpointable, strict=True
            )
        )
        checkpoint_delta = (
            science_completed
            + information
            + wait_regret
            - (1.0 - probability)
            * (rollback_after_checkpoint + failed_new_progress)
            * opportunity_rate
            - (checkpoint_cost or 0.0) * opportunity_rate
            - interference_value
        )
        request_in_progress = any(
            value.checkpoint_request_path and Path(value.checkpoint_request_path).is_file()
            for value in device.active
        )
        if request_in_progress:
            return rejected("CHECKPOINT_IN_PROGRESS", probability=probability)
        if (
            expected_value <= 0 < checkpoint_delta
            and any(checkpointable)
            and checkpoint_cost is not None
        ):
            return ExplorationEvaluation(
                action.key,
                device.index,
                device.free_bytes,
                device.provisional_headroom_bytes,
                tuple(value.resource_state for value in device.active),
                peak_hazard,
                probability,
                action.scientific_value,
                normalized_scientific_value,
                scientific_rate,
                information,
                rollback,
                rollback_value,
                interference_value,
                wait_regret,
                checkpoint_delta,
                False,
                "CHECKPOINT_REQUESTED",
                "CHECKPOINT_THEN_EXPLORE",
                signature,
                max((value.evidence_cycles for value in device.active), default=0),
                max(
                    (
                        value.last_material_peak_time
                        for value in device.active
                        if value.last_material_peak_time is not None
                    ),
                    default=None,
                ),
                _combined_phase_hazards(device.active),
                max((value.tail_probability for value in device.active), default=0.0),
                checkpoint_cost,
                rollback_after_checkpoint,
                self.wait_regret.diagnostics(device.index),
                _fit_uncertainty_sources(prediction, device),
            )
        reason = (
            "THROUGHPUT_REGRESSION"
            if interference > 0 and expected_value <= 0
            else "ROLLBACK_DOMINATES"
            if rollback_value > science_completed + information + wait_regret
            else "WAIT_CURRENTLY_BETTER"
            if expected_value <= 0
            else "IDLE_OPPORTUNITY_DOMINATES"
        )
        return ExplorationEvaluation(
            action.key,
            device.index,
            device.free_bytes,
            device.provisional_headroom_bytes,
            tuple(value.resource_state for value in device.active),
            peak_hazard,
            probability,
            action.scientific_value,
            normalized_scientific_value,
            scientific_rate,
            information,
            rollback,
            rollback_value,
            interference_value,
            wait_regret,
            expected_value,
            expected_value > 0,
            reason,
            "EXPLORE" if expected_value > 0 else "WAIT",
            signature,
            max((value.evidence_cycles for value in device.active), default=0),
            max(
                (
                    value.last_material_peak_time
                    for value in device.active
                    if value.last_material_peak_time is not None
                ),
                default=None,
            ),
            _combined_phase_hazards(device.active),
            max((value.tail_probability for value in device.active), default=0.0),
            checkpoint_cost,
            rollback_after_checkpoint,
            self.wait_regret.diagnostics(device.index),
            _fit_uncertainty_sources(prediction, device),
        )

    def _fit_state(
        self, action: CandidateResourceAction, device: GPUResourceState
    ) -> tuple[PlacementState, str]:
        prediction = action.prediction_for(device)
        if len(device.active) >= device.run_cap:
            return "RESOURCE_BLOCKED", "runs_per_gpu-hard-cap"
        if prediction.known_lower_bound_bytes > device.total_bytes:
            return "RESOURCE_INFEASIBLE_ON_DEVICE_TYPE", "known-lower-bound-exceeds-device"
        residents = tuple(sorted(value.candidate_key for value in device.active))
        if any(
            failure.hardware == device.hardware
            and failure.candidate_key == prediction.candidate_key
            and failure.resident_candidates == residents
            and device.provisional_headroom_bytes <= failure.headroom_bytes
            for failure in self.model.placement_failures
        ):
            return "RESOURCE_BLOCKED", "known-failed-placement-dominates-current-condition"
        if prediction.known_lower_bound_bytes > device.admission_headroom_bytes:
            return "RESOURCE_BLOCKED", "dominated-by-known-oom-lower-bound"
        if prediction.commitment_bytes > device.admission_headroom_bytes:
            return "RESOURCE_BLOCKED", "future-envelope-exceeds-predicted-headroom"
        return "ADMITTED", "fits-conservative-future-envelope"

    def _hurts_throughput(self, action: CandidateResourceAction, device: GPUResourceState) -> bool:
        current = len(device.active)
        if current == 0:
            return False
        before = self.model.aggregate_throughput(_action_compatibility(action, device), current)
        after = self.model.aggregate_throughput(_action_compatibility(action, device), current + 1)
        previous = self.model.aggregate_throughput(
            _action_compatibility(action, device), max(1, current - 1)
        )
        already_saturated = (
            current > 1
            and previous is not None
            and before is not None
            and before <= previous
        )
        predicted_regression = before is not None and after is not None and after <= before
        return already_saturated or predicted_regression

    @staticmethod
    def _reservation(
        actions: Sequence[CandidateResourceAction], devices: Sequence[GPUResourceState]
    ) -> tuple[str, int, float | None] | None:
        for action in actions:
            if any(
                action.prediction_for(device).commitment_bytes <= device.admission_headroom_bytes
                and len(device.active) < device.run_cap
                for device in devices
            ):
                continue
            candidates = [
                device
                for device in devices
                if action.prediction_for(device).commitment_bytes
                <= device.total_bytes - device.external_bytes
            ]
            if not candidates:
                continue
            selected = min(
                candidates,
                key=lambda device: (
                    _device_fit_time(action.prediction_for(device).commitment_bytes, device)
                    or math.inf,
                    device.index,
                ),
            )
            return (
                action.key,
                selected.index,
                _device_fit_time(action.prediction_for(selected).commitment_bytes, selected),
            )
        return None


class ResourceHistoryStore:
    """Atomic bounded per-Study ledger plus compatible project-scoped warm history."""

    def __init__(self, study_root: Path | None, shared_root: Path | None = None) -> None:
        self.study_root = study_root
        self.shared_root = shared_root
        self._memory: list[ResourceProfileObservation] = []
        self._last_decision_signature: str | None = None
        self._last_ledger_signature: str | None = None
        self._last_exploration_signature: str | None = None
        self._last_trace_signature: str | None = None
        if self.study_root is not None:
            self.study_root.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[ResourceProfileObservation, ...]:
        records = list(self._memory)
        if self.study_root is not None:
            records.extend(self._read(self.study_root / "observations.jsonl"))
        if self.shared_root is not None:
            records.extend(self._read(self.shared_root / "observations.jsonl"))
        unique: dict[tuple[str, str], ResourceProfileObservation] = {}
        for value in records:
            unique[(value.candidate_key, value.timestamp_utc)] = value
        return tuple(unique.values())

    def append(self, observation: ResourceProfileObservation) -> None:
        self._memory.append(observation)
        if self.study_root is not None:
            self._append(self.study_root / "observations.jsonl", observation)
        if self.shared_root is not None:
            self._append(self.shared_root / "observations.jsonl", observation)

    def load_placement_failures(self) -> tuple[PlacementOOMEvidence, ...]:
        output: list[PlacementOOMEvidence] = []
        for root in (self.study_root, self.shared_root):
            path = root / "placement-oom.jsonl" if root is not None else None
            if path is None or not path.is_file() or path.is_symlink():
                continue
            try:
                for line in path.read_text(encoding="utf-8").splitlines()[-_HISTORY_LIMIT:]:
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        output.append(PlacementOOMEvidence.from_mapping(value))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return tuple({value.experiment_signature: value for value in output}.values())

    def append_placement_failure(self, evidence: PlacementOOMEvidence) -> None:
        if self.study_root is not None:
            self._append_mapping(self.study_root / "placement-oom.jsonl", evidence.to_dict())
        if self.shared_root is not None:
            self._append_mapping(self.shared_root / "placement-oom.jsonl", evidence.to_dict())

    def load_active(self) -> tuple[ActiveResourceEvidence, ...]:
        """Load crash-surviving provisional evidence without claiming its processes are alive."""
        if self.study_root is None:
            return ()
        path = self.study_root / "active-evidence.json"
        if not path.is_file() or path.is_symlink():
            return ()
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
            items = (
                decoded.get("items", ())
                if isinstance(decoded, Mapping)
                else decoded
            )
            if not isinstance(items, list):
                return ()
            return tuple(
                ActiveResourceEvidence.from_mapping(value)
                for value in items
                if isinstance(value, Mapping)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return ()

    def persist_active(self, evidence: Sequence[ActiveResourceEvidence]) -> None:
        """Atomically replace the current bounded provisional snapshot."""
        if self.study_root is None:
            return
        atomic_write_json(
            self.study_root / "active-evidence.json",
            {
                "active_evidence_version": 2,
                "items": [value.to_dict() for value in evidence],
            },
        )

    def load_wait_regret(self) -> Mapping[str, Any]:
        """Load bounded scheduling debt from a previous controller process."""
        if self.study_root is None:
            return {}
        path = self.study_root / "wait-regret.json"
        if not path.is_file() or path.is_symlink():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, Mapping) else {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def persist_wait_regret(self, tracker: WaitRegretTracker) -> None:
        if self.study_root is not None:
            atomic_write_json(self.study_root / "wait-regret.json", tracker.to_dict())

    def record_event(self, event: str, details: Mapping[str, Any]) -> None:
        """Append one meaningful resource transition; callers suppress sample noise."""
        if self.study_root is None:
            return
        self._append_mapping(
            self.study_root / "resource-events.jsonl",
            {
                "event": event,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **_portable(details),
            },
        )

    def record_decisions(self, decisions: Sequence[AdmissionDecision]) -> None:
        if self.study_root is None:
            return
        signature = _digest(
            {
                "decisions": [
                    {
                        key: value
                        for key, value in decision.to_dict().items()
                        if key != "timestamp_utc"
                    }
                    for decision in decisions
                ]
            }
        )
        if signature == self._last_decision_signature:
            return
        self._last_decision_signature = signature
        for decision in decisions:
            self._append_mapping(self.study_root / "admission-decisions.jsonl", decision.to_dict())

    def record_exploration_evaluations(
        self, evaluations: Sequence[ExplorationEvaluation]
    ) -> None:
        """Persist changed WHY-WAIT/WHY-EXPLORE evidence without poll-cycle spam."""
        if self.study_root is None or not evaluations:
            return
        bounded = tuple(evaluations[:_FRONTIER_LIMIT])
        signature = _digest(
            {
                "evaluations": [
                    {
                        "candidate": value.candidate,
                        "gpu": value.gpu,
                        "accepted": value.accepted,
                        "reason": value.rejection_reason,
                        "plan": value.plan,
                        "fit_band": round(value.fit_probability, 1),
                        "hazard_band": round(value.peak_hazard, 1),
                        "wait_regret_order": (
                            int(math.floor(math.log10(1.0 + value.wait_regret)))
                            if value.wait_regret > 0
                            else 0
                        ),
                        "delta_sign": value.final_delta_value > 0,
                    }
                    for value in bounded
                ]
            }
        )
        if signature == self._last_exploration_signature:
            return
        self._last_exploration_signature = signature
        for value in bounded:
            self._append_mapping(
                self.study_root / "exploration-evaluations.jsonl", value.to_dict()
            )
        if not any(value.accepted for value in bounded):
            leading = max(bounded, key=lambda value: value.final_delta_value)
            self.record_event("RESOURCE_WAIT", leading.to_dict())

    def persist_ledger(self, devices: Sequence[GPUResourceState]) -> None:
        if self.study_root is None:
            return
        values = [
            {
                **asdict(device),
                "future_committed_bytes": device.future_committed_bytes,
                "predicted_headroom_bytes": device.predicted_headroom_bytes,
                "admission_headroom_bytes": device.admission_headroom_bytes,
            }
            for device in devices
        ]
        signature = _digest({"devices": values})
        if signature == self._last_ledger_signature:
            return
        self._last_ledger_signature = signature
        atomic_write_json(
            self.study_root / "commitment-ledger.json",
            {
                "ledger_version": 1,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "devices": values,
            },
        )

    def record_trace(
        self,
        *,
        elapsed_seconds: float,
        devices: Sequence[GPUResourceState],
        actions: Sequence[CandidateResourceAction],
        decisions: Sequence[AdmissionDecision],
        exploration_evaluations: Sequence[ExplorationEvaluation] = (),
    ) -> None:
        """Append one changed canonical scheduling snapshot for deterministic replay."""
        if self.study_root is None:
            return
        payload = {
            "trace_version": 1,
            "elapsed_seconds": max(0.0, float(elapsed_seconds)),
            "devices": [
                {
                    "gpu": value.index,
                    "token": value.token,
                    "hardware": value.hardware,
                    "total_bytes": value.total_bytes,
                    "physical_free_bytes": value.free_bytes,
                    "external_bytes": value.external_bytes,
                    "active_runs": len(value.active),
                    "run_cap": value.run_cap,
                    "resource_states": [item.resource_state for item in value.active],
                    "active_candidates": [item.candidate_key for item in value.active],
                    "active_current_bytes": [item.current_bytes for item in value.active],
                    "active_remaining_seconds": [
                        item.remaining_seconds for item in value.active
                    ],
                    "active_throughputs": [
                        item.throughput for item in value.active
                    ],
                    "growth_hazards": [
                        _commitment_growth_hazard(item) for item in value.active
                    ],
                }
                for value in devices
            ],
            "frontier": [
                {
                    "candidate": value.key,
                    "scientific_value": value.scientific_value,
                    "value_contract": (
                        asdict(value.value_contract) if value.value_contract is not None else None
                    ),
                    "predictions": {
                        str(device.index): value.prediction_for(device).to_dict()
                        for device in devices
                    },
                }
                for value in actions[:_FRONTIER_LIMIT]
            ],
            "decisions": [value.to_dict() for value in decisions],
            "exploration": [value.to_dict() for value in exploration_evaluations],
        }
        signature = _digest(
            {key: value for key, value in payload.items() if key != "elapsed_seconds"}
        )
        if signature == self._last_trace_signature:
            return
        self._last_trace_signature = signature
        self._append_mapping(self.study_root / "scheduler-trace.jsonl", payload)

    @staticmethod
    def _read(path: Path) -> list[ResourceProfileObservation]:
        if not path.is_file() or path.is_symlink():
            return []
        output: list[ResourceProfileObservation] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[-_HISTORY_LIMIT:]:
                value = json.loads(line)
                if isinstance(value, Mapping):
                    output.append(ResourceProfileObservation.from_mapping(value))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []
        return output

    @classmethod
    def _append(cls, cls_path: Path, value: ResourceProfileObservation) -> None:
        cls._append_mapping(cls_path, value.to_dict())

    @staticmethod
    def _append_mapping(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            os.write(
                descriptor,
                (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode(),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def resource_state(
    prediction: ResourcePrediction,
    *,
    observed_peak_bytes: int,
    phase: str | None,
    trajectory: Sequence[ResourceTrajectorySample] = (),
) -> str:
    """Expose the trajectory posterior state; placement consumes the distribution itself."""
    if observed_peak_bytes <= 0:
        return "STARTING"
    if trajectory:
        analysis = ResourceTrajectoryAnalyzer.analyze(
            trajectory,
            historical_residuals=prediction.future_residual_samples,
            total_bytes=max(prediction.upper_bytes, observed_peak_bytes),
        )
        if analysis.state in {"STARTING", "RAMPING", "PLATEAU_UNCONFIRMED"}:
            return analysis.state
    possible_later_phase = (
        phase not in {None, "validation", "terminal"}
        and prediction.upper_bytes > observed_peak_bytes
    )
    uncertainty = prediction.upper_bytes - prediction.lower_bytes
    decision_material = uncertainty > max(0, prediction.upper_bytes - observed_peak_bytes)
    if possible_later_phase or decision_material:
        return "PROVISIONALLY_STABLE" if trajectory else "RAMPING"
    return "RESOURCE_STABLE"


def _analysis_from_statistics(
    value: ResourceTrajectoryStatistics,
    *,
    historical_residuals: Sequence[int],
    total_bytes: int,
    phase_model: ResourcePhaseModel | None = None,
) -> ResourceTrajectoryAnalysis:
    """Derive one posterior from mergeable progress evidence, never poll count."""
    peak = value.running_peak_bytes
    spare = max(0, total_bytes - peak)
    noise = _robust_upper_scale(
        (*value.noise_distribution, *value.allocator_residual_distribution)
    )
    growth_history = [item for item in value.growth_distribution if item > 0]
    midpoint = max(1, len(growth_history) // 2)
    converging = (
        len(growth_history) >= 4
        and statistics.median(growth_history[midpoint:])
        < statistics.median(growth_history[:midpoint])
    )
    converged_tail = (
        sum(
            item <= statistics.median(growth_history[midpoint:])
            for item in reversed(growth_history)
        )
        if converging
        else 0
    )
    meaningful = max(
        0,
        value.cycles_since_material_peak,
        value.progress_cycles_seen if converging else converged_tail,
    )
    latest_material = (
        value.material_peak_count > 0
        and value.cycles_since_material_peak == 0
        and not converging
    )
    # Jeffreys prior over a material peak in the next progress interval. Repeated identical NVML
    # observations do not enter either term.
    hazard = (0.5 + (1.0 if latest_material else 0.0)) / (meaningful + 1.0)
    observed = tuple(dict.fromkeys(value.phases_seen))
    known = phase_model or ResourcePhaseModel(observed, {})
    complete, missing = known.completeness(observed)
    phase_hazards = {
        phase: (
            min(1.0, hazard)
            if phase in observed
            else max(hazard, float(known.peak_probability.get(phase, hazard)))
        )
        for phase in dict.fromkeys((*known.expected_phases, *observed))
    }
    if missing:
        hazard = max(hazard, max(phase_hazards[phase] for phase in missing))
    hazard = min(1.0, max(0.005, hazard))

    growth = growth_history
    expected = (
        int(statistics.median(growth[-max(1, math.isqrt(len(growth))) :]))
        if growth
        else noise
    )
    expected = min(spare, int(expected * hazard))
    empirical = [max(0, min(spare, int(item))) for item in historical_residuals]
    support = [0, expected, *empirical, spare]
    tail_probability = max(
        0.001,
        hazard * (0.35 if missing else 0.1),
    )
    body_mass = max(0.0, 1.0 - tail_probability)
    empirical_share = body_mass * hazard * 0.25 if empirical else 0.0
    weights = [
        body_mass * (1.0 - hazard),
        body_mass * hazard - empirical_share,
    ]
    weights.extend(
        (empirical_share / len(empirical) for _ in empirical)
        if empirical
        else ()
    )
    # Normalize below: exact tail mass is retained regardless of support resolution.
    weights.append(tail_probability)
    residuals, residual_weights = _coalesce_weighted_samples(support, weights)

    progress_intervals = [
        right - left
        for left, right in zip(
            value.meaningful_event_times, value.meaningful_event_times[1:], strict=False
        )
        if right > left
    ]
    progress_cadence = statistics.median(progress_intervals) if progress_intervals else None
    progress_uncertainty = _median_absolute_deviation(progress_intervals)
    checkpoint_intervals = [
        right - left
        for left, right in zip(value.checkpoint_times, value.checkpoint_times[1:], strict=False)
        if right > left
    ]
    checkpoint_cadence = (
        statistics.median(checkpoint_intervals) if checkpoint_intervals else None
    )
    checkpoint_uncertainty = _median_absolute_deviation(checkpoint_intervals)
    until_checkpoint = None
    if checkpoint_cadence is not None and value.checkpoint_times:
        until_checkpoint = max(
            0.0,
            checkpoint_cadence
            - max(0.0, value.last_elapsed_seconds - value.checkpoint_times[-1]),
        )
    next_events = [
        (estimate, uncertainty)
        for estimate, uncertainty in (
            (progress_cadence, progress_uncertainty),
            (until_checkpoint, checkpoint_uncertainty),
        )
        if estimate is not None and estimate > 0
    ]
    next_event = min(next_events, key=lambda item: item[0]) if next_events else None
    if value.observations_seen <= 1 or peak <= 0:
        state, reason = "STARTING", "the first allocation cycle is not characterized"
    elif latest_material or (
        not converging and value.positive_streak >= 2 and value.monotonic_growth_bytes > noise
    ):
        state, reason = "RAMPING", "a persistent material allocation trend is active"
    elif meaningful >= 2 and complete:
        state, reason = (
            "PROVISIONALLY_STABLE",
            "meaningful progress cycles survived without a material peak",
        )
    else:
        state, reason = (
            "PLATEAU_UNCONFIRMED",
            "progress or allocation-relevant phase evidence remains incomplete",
        )
    return ResourceTrajectoryAnalysis(
        state=state,
        running_peak_bytes=peak,
        current_bytes=value.current_memory_bytes,
        last_peak_step=value.last_material_peak_step,
        completed_cycles_since_peak=meaningful,
        residual_samples=residuals,
        phase=value.last_phase,
        reason=reason,
        residual_weights=residual_weights,
        growth_hazard=hazard,
        expected_residual_growth_bytes=expected,
        measurement_noise_bytes=noise,
        critical_phases_seen=observed,
        next_decision_seconds=next_event[0] if next_event is not None else None,
        next_decision_uncertainty_seconds=(
            next_event[1] if next_event is not None else None
        ),
        evidence_cycles=value.progress_cycles_seen,
        phase_hazards=phase_hazards,
        tail_probability=tail_probability,
        last_material_peak_time=value.last_material_peak_time,
        checkpoint_cadence_seconds=checkpoint_cadence,
    )


def _bounded_append(values: list[Any], value: Any, limit: int = 63) -> None:
    values.append(value)
    if len(values) > limit:
        del values[: len(values) - limit]


def _median_absolute_deviation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _robust_upper_scale(values: Sequence[int]) -> int:
    positive = sorted(abs(int(value)) for value in values if value)
    if not positive:
        return 0
    centre = statistics.median(positive)
    mad = statistics.median(abs(value - centre) for value in positive)
    return max(1, int(centre + 3.0 * 1.4826 * mad))


def _representative_indices(indices: Sequence[int], count: int) -> set[int]:
    """Select deterministic coverage across an ordered index set without exceeding ``count``."""
    if count <= 0 or not indices:
        return set()
    if len(indices) <= count:
        return set(indices)
    if count == 1:
        return {indices[len(indices) // 2]}
    last = len(indices) - 1
    return {indices[round(position * last / (count - 1))] for position in range(count)}


def _weighted_peaks(
    parameters: Mapping[str, Any],
    observations: Sequence[ResourceProfileObservation],
    schema: Mapping[str, Any] | ParameterSpace | None = None,
) -> list[int]:
    values: list[int] = []
    for observation in observations:
        if not observation.peak_is_exact:
            continue
        distance = _mixed_distance(parameters, observation.parameters, schema)
        repeats = max(1, int(round(4 / (1 + 4 * distance))))
        values.extend([observation.observed_peak_bytes] * repeats)
    return values


def _deterministic_bootstrap(values: Sequence[int]) -> tuple[int, ...]:
    ordered = sorted(values)
    if not ordered:
        return (0,)
    return tuple(
        ordered[min(len(ordered) - 1, int(index * len(ordered) / _BOOTSTRAP_POINTS))]
        for index in range(_BOOTSTRAP_POINTS)
    )


def _loo_underprediction_residuals(
    observations: Sequence[ResourceProfileObservation],
    schema: Mapping[str, Any] | ParameterSpace | None = None,
) -> tuple[int, ...]:
    """Calibrate false-safe error using the same local mixed-space model leave-one-out."""
    exact = [value for value in observations if value.peak_is_exact]
    if len(exact) < 2:
        peaks = [value.observed_peak_bytes for value in exact]
        return (0, max(peaks, default=0) - min(peaks, default=0))
    residuals: list[int] = []
    for index, observation in enumerate(exact):
        peers = [value for offset, value in enumerate(exact) if offset != index]
        local = _weighted_peaks(observation.parameters, peers, schema)
        predicted = int(statistics.median(local)) if local else 0
        residuals.append(max(0, observation.observed_peak_bytes - predicted))
    return tuple(sorted(residuals)) or (0,)


def _mixed_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    schema: Mapping[str, Any] | ParameterSpace | None = None,
) -> float:
    space = (
        schema
        if isinstance(schema, ParameterSpace)
        else ParameterSpace.from_schema(schema, (left, right))
    )
    return space.distance(left, right)


def _empirical_support_distance(
    observations: Sequence[ResourceProfileObservation],
    schema: Mapping[str, Any] | ParameterSpace,
) -> float:
    """Median nearest-neighbour distance of the evidence itself (LOO support scale)."""
    if len(observations) < 2:
        return 0.0
    distances = []
    for index, observation in enumerate(observations):
        distances.append(
            min(
                _mixed_distance(observation.parameters, other.parameters, schema)
                for offset, other in enumerate(observations)
                if offset != index
            )
        )
    return statistics.median(distances)


def _resource_phase_class(value: str | None) -> str:
    """Normalize worker narration into allocation-relevant phase classes."""
    normalized = str(value or "").lower().replace("_", "-")
    if "backward" in normalized:
        return "backward"
    if "optimizer" in normalized:
        return "optimizer"
    if "validation" in normalized or "evaluation" in normalized:
        return "validation"
    if "checkpoint" in normalized:
        return "checkpoint"
    if "forward" in normalized or normalized == "training":
        return "forward"
    if normalized == "terminal":
        return "terminal"
    return normalized or "unknown"


def _normalized_weights(weights: Sequence[float], count: int) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if len(weights) != count or any(value < 0 or not math.isfinite(value) for value in weights):
        return tuple(1.0 / count for _ in range(count))
    total = sum(weights)
    if total <= 0:
        return tuple(1.0 / count for _ in range(count))
    return tuple(float(value) / total for value in weights)


def _coalesce_weighted_samples(
    samples: Sequence[int], weights: Sequence[float]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Combine repeated support points without turning a rare tail into an equal draw."""
    if not samples:
        return (), ()
    normalized = _normalized_weights(weights, len(samples))
    combined: dict[int, float] = {}
    for sample, weight in zip(samples, normalized, strict=True):
        combined[max(0, int(sample))] = combined.get(max(0, int(sample)), 0.0) + weight
    ordered = tuple(sorted(combined))
    return ordered, _normalized_weights(tuple(combined[value] for value in ordered), len(ordered))


def _weighted_cdf(samples: Sequence[int], weights: Sequence[float], limit: int) -> float:
    normalized = _normalized_weights(weights, len(samples))
    return sum(
        weight
        for sample, weight in zip(samples, normalized, strict=True)
        if sample <= limit
    )


def _weighted_quantile(
    samples: Sequence[int], weights: Sequence[float], quantile: float
) -> int:
    if not samples:
        return 0
    pairs = sorted(zip(samples, _normalized_weights(weights, len(samples)), strict=True))
    target = min(1.0, max(0.0, quantile))
    cumulative = 0.0
    for sample, weight in pairs:
        cumulative += weight
        if cumulative >= target:
            return int(sample)
    return int(pairs[-1][0])


def _active_support_scale(
    distances: Sequence[tuple[float, ActiveResourceEvidence]],
    schema: Mapping[str, Any] | ParameterSpace,
) -> float:
    if not distances:
        return 1.0
    evidence = [value for _distance, value in distances]
    pairwise = [
        _mixed_distance(left.parameters, right.parameters, schema)
        for index, left in enumerate(evidence)
        for right in evidence[index + 1 :]
    ]
    if pairwise:
        return max(statistics.median(pairwise), 1e-6)
    nearest = distances[0][0]
    # With one active neighbour, the remaining normalized domain distance is a weak support
    # bandwidth: identical candidates share fully, boundary-to-boundary extrapolation vanishes.
    return max(1e-6, 1.0 - nearest)


def _fit_probability(prediction: ResourcePrediction, headroom: int) -> float:
    if prediction.known_lower_bound_bytes > headroom:
        return 0.0
    if not prediction.samples:
        return 0.0
    return _weighted_cdf(prediction.samples, prediction.sample_weights, headroom)


def _combined_phase_hazards(
    commitments: Sequence[ActiveResourceCommitment],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for value in commitments:
        for phase, hazard in value.phase_hazards.items():
            output[str(phase)] = max(output.get(str(phase), 0.0), float(hazard))
    return output


def _fit_uncertainty_sources(
    prediction: ResourcePrediction, device: GPUResourceState
) -> tuple[str, ...]:
    sources: list[str] = []
    if len(prediction.samples) > 1:
        sources.append("candidate-tail")
    for value in device.active:
        if len(value.future_peak_samples) > 1:
            sources.append(f"resident:{value.candidate_key}:tail")
        unseen = [phase for phase, hazard in value.phase_hazards.items() if hazard > 0.25]
        sources.extend(f"resident:{value.candidate_key}:phase:{phase}" for phase in unseen)
    if device.external_bytes:
        sources.append("external-physical-pressure")
    if prediction.known_lower_bound_bytes:
        sources.append("known-lower-bound")
    return tuple(dict.fromkeys(sources))


def _joint_fit_probability(prediction: ResourcePrediction, device: GPUResourceState) -> float:
    """Tail-aware small-ensemble fit probability with exact discrete integration.

    One member integrates run-specific residuals independently.  The other uses a shared latent
    resource-model quantile.  Their equal model ensemble is explicit, unlike index-aligned Monte
    Carlo, and every discrete mass boundary—including a 0.5% tail—is represented.
    """
    distributions = [
        (
            prediction.samples or (prediction.predicted_peak_bytes,),
            prediction.sample_weights,
        ),
        *[
            (
                value.future_peak_samples
                or (max(value.current_bytes, value.running_peak_bytes, value.commitment_bytes),),
                value.future_peak_weights,
            )
            for value in device.active
        ],
    ]
    candidate_samples, candidate_weights = distributions[0]
    candidate_physical_fit = _weighted_cdf(
        candidate_samples, candidate_weights, device.free_bytes
    )
    if candidate_physical_fit <= 0:
        return 0.0

    summed: dict[int, float] = {device.external_bytes: 1.0}
    for samples, weights in distributions:
        pairs = tuple(zip(samples, _normalized_weights(weights, len(samples)), strict=True))
        combined: dict[int, float] = {}
        for current, current_weight in summed.items():
            for sample, weight in pairs:
                total = current + int(sample)
                combined[total] = combined.get(total, 0.0) + current_weight * weight
        summed = _compress_weighted_support(combined)
    independent = sum(weight for total, weight in summed.items() if total <= device.total_bytes)

    breakpoints = {0.0, 1.0}
    for samples, weights in distributions:
        cumulative = 0.0
        for _sample, weight in sorted(
            zip(samples, _normalized_weights(weights, len(samples)), strict=True)
        ):
            cumulative += weight
            breakpoints.add(min(1.0, cumulative))
    ordered = sorted(breakpoints)
    shared = 0.0
    for left, right in zip(ordered, ordered[1:], strict=False):
        if right <= left:
            continue
        quantile = (left + right) / 2.0
        candidate_value = _weighted_quantile(
            candidate_samples, candidate_weights, quantile
        )
        used = device.external_bytes + sum(
            _weighted_quantile(samples, weights, quantile)
            for samples, weights in distributions
        )
        if used <= device.total_bytes and candidate_value <= device.free_bytes:
            shared += right - left
    return min(candidate_physical_fit, max(0.0, min(1.0, (independent + shared) / 2.0)))


def _compress_weighted_support(values: Mapping[int, float]) -> dict[int, float]:
    """Bound convolution cost while preserving exact small supports and both tails."""
    if len(values) <= 4096:
        return dict(values)
    ordered = sorted(values.items())
    total = sum(weight for _sample, weight in ordered)
    output: dict[int, float] = {ordered[0][0]: ordered[0][1], ordered[-1][0]: ordered[-1][1]}
    target = total / 4094
    mass = 0.0
    weighted_sum = 0.0
    for sample, weight in ordered[1:-1]:
        mass += weight
        weighted_sum += sample * weight
        if mass >= target:
            representative = int(round(weighted_sum / mass))
            output[representative] = output.get(representative, 0.0) + mass
            mass = weighted_sum = 0.0
    if mass:
        representative = int(round(weighted_sum / mass))
        output[representative] = output.get(representative, 0.0) + mass
    normalizer = sum(output.values())
    return {sample: weight / normalizer for sample, weight in output.items()}


def _normalized_scientific_values(
    actions: Sequence[CandidateResourceAction],
) -> dict[str, float]:
    """Translate controller ordering, not its arbitrary numeric scale, into planner utility."""
    count = len(actions)
    return {
        action.key: (
            min(1.0, max(0.0, float(action.value_contract.normalized_value)))
            if action.value_contract is not None
            and action.value_contract.normalized_value is not None
            else (count - index) / max(1, count)
        )
        for index, action in enumerate(actions)
    }


def _commitment_growth_hazard(value: ActiveResourceCommitment) -> float:
    if value.growth_hazard >= 0:
        return min(1.0, value.growth_hazard)
    return {
        "STARTING": 0.8,
        "RAMPING": 1.0,
        "PLATEAU_UNCONFIRMED": 0.45,
        "PROVISIONALLY_STABLE": 0.15,
        "RESOURCE_STABLE": 0.05,
    }.get(value.resource_state, 0.5)


def _candidate_duration(
    prediction: ResourcePrediction, device: GPUResourceState
) -> float | None:
    if prediction.predicted_duration_seconds is not None:
        return prediction.predicted_duration_seconds
    if prediction.duration_samples:
        return statistics.median(prediction.duration_samples)
    # Active progress is a legitimate current-Study estimate. It is explicitly uncertain and
    # preferable to the former fabricated one-second duration.
    resident = [
        value.remaining_seconds
        for value in device.active
        if value.remaining_seconds is not None and value.remaining_seconds > 0
    ]
    return statistics.median(resident) if resident else None


def _next_decision_horizon(
    prediction: ResourcePrediction, device: GPUResourceState
) -> float | None:
    candidates = [
        value
        for commitment in device.active
        for value in (commitment.next_decision_seconds, commitment.remaining_seconds)
        if value is not None and value > 0
    ]
    candidates.extend(
        max(
            0.0,
            float(value.checkpoint_cadence_seconds)
            - (value.elapsed_seconds - float(value.checkpoint_elapsed_seconds)),
        )
        for value in device.active
        if value.checkpoint_cadence_seconds is not None
        and value.checkpoint_cadence_seconds > 0
        and value.checkpoint_elapsed_seconds is not None
        and value.elapsed_seconds >= value.checkpoint_elapsed_seconds
    )
    if prediction.predicted_time_to_envelope_seconds is not None:
        candidates.append(prediction.predicted_time_to_envelope_seconds)
    duration = _candidate_duration(prediction, device)
    if duration is not None:
        candidates.append(duration)
    future = [value for value in candidates if value > 0]
    return min(future) if future else None


def _binary_entropy(probability: float) -> float:
    value = min(1.0 - 1e-12, max(1e-12, probability))
    return -(value * math.log2(value) + (1.0 - value) * math.log2(1.0 - value))


def _resource_experiment_signature(
    action: CandidateResourceAction, device: GPUResourceState
) -> str:
    prediction = action.prediction_for(device)
    candidate_samples = prediction.samples or (prediction.predicted_peak_bytes,)
    return _digest(
        {
            "prediction_version": "mixed-live-survival-v3",
            "hardware": device.hardware,
            "candidate": prediction.candidate_key,
            "residents": sorted(value.candidate_key for value in device.active),
            "resident_states": sorted(value.resource_state for value in device.active),
            "checkpointed": sorted(value.checkpoint_resumable for value in device.active),
            # Encode the headroom's relation to the current posterior instead of raw bytes. Tiny
            # allocator fluctuations therefore cannot manufacture a "new" experiment, while a
            # genuinely different fit outcome changes the signature immediately.
            "candidate_fit_samples": sum(
                sample <= device.provisional_headroom_bytes for sample in candidate_samples
            ),
            "candidate_sample_count": len(candidate_samples),
            "joint_fit_probability": _joint_fit_probability(prediction, device),
        }
    )


def _earliest_fit(commitment: int, devices: Sequence[GPUResourceState]) -> float | None:
    values = [_device_fit_time(commitment, device) for device in devices]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


def _device_fit_time(commitment: int, device: GPUResourceState) -> float | None:
    if commitment <= device.admission_headroom_bytes:
        return 0.0
    known_current = sum(value.current_bytes for value in device.active)
    unattributed_owned = max(
        0,
        device.total_bytes - device.free_bytes - device.external_bytes - known_current,
    )
    commitment_total = sum(value.commitment_bytes for value in device.active)
    releases = sorted(
        (
            value.remaining_seconds,
            value.commitment_bytes,
            value.current_bytes
            + (
                int(unattributed_owned * value.commitment_bytes / commitment_total)
                if commitment_total
                else 0
            ),
        )
        for value in device.active
        if value.remaining_seconds is not None
    )
    future_headroom = device.predicted_headroom_bytes
    physical_headroom = device.free_bytes
    for seconds, future_released, current_released in releases:
        future_headroom += future_released
        physical_headroom += current_released
        if commitment <= min(future_headroom, physical_headroom):
            return seconds
    return None


def _allocation_bytes(message: str) -> int | None:
    match = _OOM_ALLOCATION.search(message)
    if match is None:
        return None
    amount, unit = match.groups()
    multiplier = {
        "byte": 1,
        "bytes": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }[unit.lower()]
    return int(float(amount) * multiplier)


def _trial(specification: Mapping[str, Any]) -> int | None:
    value = specification.get("trial_index")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _action_compatibility(action: CandidateResourceAction, device: GPUResourceState) -> str:
    """Resolve the compatibility identity belonging to the actual target hardware."""
    identities = action.specification.get("resource_device_identities", {})
    if isinstance(identities, Mapping):
        identity = identities.get(device.index, identities.get(str(device.index)))
        if (
            isinstance(identity, Sequence)
            and not isinstance(identity, str | bytes | bytearray)
            and len(identity) == 2
        ):
            return str(identity[1])
    return str(action.specification.get("resource_compatibility_key", ""))


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _portable(value), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _portable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _portable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_portable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


__all__ = [
    "ActiveResourceEvidence",
    "ActiveResourceCommitment",
    "AdmissionMode",
    "AdmissionDecision",
    "BoundedResourceTrajectory",
    "CandidateResourceAction",
    "ExplorationEvaluation",
    "GPUPlacementPlanner",
    "GPUResourceState",
    "OOMResourceEvidence",
    "PlacementOOMEvidence",
    "ResourceDemandModel",
    "ResourceHistoryStore",
    "ResourcePrediction",
    "ResourceProfileObservation",
    "ResourceTrajectorySample",
    "ResourceTrajectoryStatistics",
    "ResourceTrajectoryAnalysis",
    "ResourceTrajectoryAnalyzer",
    "ResourcePhaseModel",
    "ScientificActionValue",
    "WaitRegretTracker",
    "oom_evidence",
    "resource_identity",
    "resource_state",
]
