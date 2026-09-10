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

from lambdaforge.work.atomic import atomic_write_json

ResourceRunState = Literal["completed", "oom", "pruned", "failed", "cancelled"]
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BoundedResourceTrajectory:
    """Downsample a long resource stream while preserving extrema and transitions."""

    def __init__(self, values: Sequence[ResourceTrajectorySample] = ()) -> None:
        self._values = list(values)

    @property
    def values(self) -> tuple[ResourceTrajectorySample, ...]:
        return tuple(self._values)

    def append(self, value: ResourceTrajectorySample) -> None:
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
        lower = max(resident_bytes, headroom_bytes)
        source = "candidate-resident-and-effective-headroom"
    else:
        lower = headroom_bytes
        source = "effective-headroom-at-failure"
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
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["parameters"] = _portable(self.parameters)
        value["fixed_arguments"] = _portable(self.fixed_arguments)
        value["fidelity"] = dict(self.fidelity)
        value["trajectory"] = [dict(item) for item in self.trajectory]
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

    @property
    def commitment_bytes(self) -> int:
        return max(self.upper_bytes, self.known_lower_bound_bytes)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["commitment_bytes"] = self.commitment_bytes
        return value


class ResourceDemandModel:
    """Small-data mixed-space bootstrap model with censored-bound support."""

    def __init__(self, observations: Sequence[ResourceProfileObservation] = ()) -> None:
        self.observations = tuple(observations)
        self._prediction_cache: dict[str, ResourcePrediction] = {}

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
        candidate_matches = [
            value for value in compatible if value.candidate_key == candidate_key
        ]
        evidence = (
            candidate_matches
            or sorted(
                compatible,
                key=lambda value: (
                    _mixed_distance(parameters, value.parameters),
                    value.timestamp_utc,
                ),
            )[: min(24, len(compatible))]
        )
        bounds = [
            value.lower_bound_bytes
            for value in compatible
            if value.candidate_key == candidate_key and value.lower_bound_bytes > 0
        ]
        known_lower = max([user_minimum_bytes, observed_prefix_peak_bytes, *bounds], default=0)
        exact_peaks = [value.observed_peak_bytes for value in evidence if value.peak_is_exact]
        censored = [
            max(value.observed_peak_bytes, value.lower_bound_bytes)
            for value in evidence
            if not value.peak_is_exact
        ]
        durations = [value.duration_seconds for value in evidence if value.duration_seconds > 0]
        peak_times = [
            value.time_to_peak_seconds
            for value in evidence
            if value.time_to_peak_seconds is not None and value.time_to_peak_seconds >= 0
        ]
        if not exact_peaks:
            # Extreme uncertainty intentionally yields one-Run-per-GPU cold start.  A censored OOM
            # still strengthens the lower bound but never masquerades as an exact target.
            predicted = max(known_lower, max(censored, default=0), user_minimum_bytes)
            upper = max(predicted, total_bytes)
            samples = tuple(sorted({known_lower, predicted, upper}))
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
                    "censored-only" if evidence else "cold-start",
                    "poor",
                    "mixed-bootstrap-v1",
                    samples,
                )
            )
        if not candidate_matches and len(exact_peaks) < 2:
            # One neighbouring configuration supplies a location hint but no empirical evidence
            # about how resource demand varies over the authored mixed space.  Keep cold-start
            # upper uncertainty at device capacity until at least one between-candidate residual
            # can exist; this is a model-identifiability condition, not a user threshold.
            predicted = max(known_lower, exact_peaks[0])
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
                    "mixed-bootstrap-v1",
                    tuple(sorted({known_lower, predicted, total_bytes})),
                )
            )
        weighted = _weighted_peaks(parameters, evidence)
        samples = _deterministic_bootstrap(weighted)
        residuals = _underprediction_residuals(weighted)
        calibrated = tuple(
            max(known_lower, sample + residuals[index % len(residuals)])
            for index, sample in enumerate(samples)
        )
        predicted = int(statistics.median(calibrated))
        lower = max(known_lower, min(calibrated))
        upper = max(lower, max(calibrated))
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
                "mixed-bootstrap-v1",
                tuple(sorted(calibrated)),
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
            return duration_seconds * statistics.median(solo_rates) / statistics.median(
                packed_rates
            )
        return duration_seconds


@dataclass(frozen=True, slots=True)
class ActiveResourceCommitment:
    candidate_key: str
    commitment_bytes: int
    current_bytes: int = 0
    remaining_seconds: float | None = None
    resource_state: str = "RAMPING"
    trial: int | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class GPUResourceState:
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
    def admission_headroom_bytes(self) -> int:
        """Return the strictest physical/current or future-envelope capacity bound."""
        return min(self.free_bytes, self.predicted_headroom_bytes)


@dataclass(frozen=True, slots=True)
class CandidateResourceAction:
    key: str
    specification: Mapping[str, Any]
    scientific_value: float
    prediction: ResourcePrediction
    device_predictions: Mapping[int, ResourcePrediction] = field(default_factory=dict)

    def prediction_for(self, device: GPUResourceState) -> ResourcePrediction:
        """Return the hardware-conditioned prediction for one legally visible device."""
        return self.device_predictions.get(device.index, self.prediction)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
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
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUPlacementPlanner:
    """Deterministic best-fit placement over a bounded scientific frontier."""

    def __init__(self, model: ResourceDemandModel) -> None:
        self.model = model

    def place(
        self,
        actions: Sequence[CandidateResourceAction],
        devices: Sequence[GPUResourceState],
        *,
        max_launches: int,
    ) -> tuple[tuple[AdmissionDecision, ...], tuple[AdmissionDecision, ...]]:
        ranked = sorted(
            actions[:_FRONTIER_LIMIT],
            key=lambda value: (-value.scientific_value, value.key),
        )
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
                duration = self.model.adjusted_duration(
                    _action_compatibility(action, device),
                    prediction.predicted_duration_seconds,
                    len(device.active) + 1,
                ) or 1.0
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
                )
            )
        # If conservative envelopes leave every device idle, a strictly improved condition may
        # justify one bounded probe.  This is opportunity-cost reasoning, not a fit threshold: it
        # is never used when safer useful work was admitted, and deterministic lower bounds still
        # cannot be violated.
        if not admitted and max_launches > 0 and blocked:
            by_key = {value.key: value for value in ranked}
            candidate = blocked[0]
            uncertain_action = by_key.get(candidate.candidate_key)
            if uncertain_action is not None:
                risky = [
                    device
                    for device in devices
                    if not device.active
                    and uncertain_action.prediction_for(device).known_lower_bound_bytes
                    < device.admission_headroom_bytes
                    and _fit_probability(
                        uncertain_action.prediction_for(device), device.admission_headroom_bytes
                    )
                    > 0
                    and len(device.active) < device.run_cap
                ]
                if risky:
                    selected_device = min(
                        risky,
                        key=lambda device: (
                            max(
                                0,
                                device.admission_headroom_bytes
                                - uncertain_action.prediction_for(device).predicted_peak_bytes,
                            ),
                            device.index,
                        ),
                    )
                    selected_prediction = uncertain_action.prediction_for(selected_device)
                    admitted.append(
                        AdmissionDecision(
                            uncertain_action.key,
                            _trial(uncertain_action.specification),
                            "ADMITTED",
                            selected_device.index,
                            "bounded uncertain admission because every compatible GPU "
                            "would otherwise idle",
                            uncertain_action.scientific_value,
                            selected_prediction.to_dict(),
                            candidate.devices,
                            _fit_probability(
                                selected_prediction,
                                selected_device.admission_headroom_bytes,
                            ),
                            self.model.adjusted_duration(
                                _action_compatibility(uncertain_action, selected_device),
                                selected_prediction.predicted_duration_seconds,
                                1,
                            ),
                        )
                    )
                    blocked = [
                        value for value in blocked if value.candidate_key != uncertain_action.key
                    ]
        return tuple(admitted), tuple(blocked)

    @staticmethod
    def _fit_state(
        action: CandidateResourceAction, device: GPUResourceState
    ) -> tuple[PlacementState, str]:
        prediction = action.prediction_for(device)
        if len(device.active) >= device.run_cap:
            return "RESOURCE_BLOCKED", "runs_per_gpu-hard-cap"
        if prediction.known_lower_bound_bytes > device.total_bytes:
            return "RESOURCE_INFEASIBLE_ON_DEVICE_TYPE", "known-lower-bound-exceeds-device"
        if prediction.known_lower_bound_bytes > device.admission_headroom_bytes:
            return "RESOURCE_BLOCKED", "dominated-by-known-oom-lower-bound"
        if prediction.commitment_bytes > device.admission_headroom_bytes:
            return "RESOURCE_BLOCKED", "future-envelope-exceeds-predicted-headroom"
        return "ADMITTED", "fits-conservative-future-envelope"

    def _hurts_throughput(self, action: CandidateResourceAction, device: GPUResourceState) -> bool:
        current = len(device.active)
        if current == 0:
            return False
        before = self.model.aggregate_throughput(
            _action_compatibility(action, device), current
        )
        after = self.model.aggregate_throughput(
            _action_compatibility(action, device), current + 1
        )
        return before is not None and after is not None and after <= before

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
) -> str:
    """Infer stability from evidence, not elapsed-time magic."""
    if prediction.compatible_history_count == 0:
        return "RAMPING"
    if observed_peak_bytes <= 0:
        return "STARTING"
    possible_later_phase = (
        phase not in {None, "validation", "terminal"}
        and prediction.upper_bytes > observed_peak_bytes
    )
    uncertainty = prediction.upper_bytes - prediction.lower_bytes
    decision_material = uncertainty > max(0, prediction.upper_bytes - observed_peak_bytes)
    return "RAMPING" if possible_later_phase or decision_material else "RESOURCE_STABLE"


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
    parameters: Mapping[str, Any], observations: Sequence[ResourceProfileObservation]
) -> list[int]:
    values: list[int] = []
    for observation in observations:
        if not observation.peak_is_exact:
            continue
        distance = _mixed_distance(parameters, observation.parameters)
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


def _underprediction_residuals(values: Sequence[int]) -> tuple[int, ...]:
    if len(values) < 2:
        return (0, max(values, default=0) - min(values, default=0))
    residuals = []
    for index, value in enumerate(values):
        peers = [item for offset, item in enumerate(values) if offset != index]
        residuals.append(max(0, value - int(statistics.median(peers))))
    return tuple(sorted(residuals)) or (0,)


def _mixed_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        a, b = left.get(key), right.get(key)
        if (
            isinstance(a, bool)
            or isinstance(b, bool)
            or not isinstance(a, int | float)
            or not isinstance(b, int | float)
        ):
            total += 0.0 if a == b else 1.0
            continue
        scale = max(abs(float(a)), abs(float(b)), 1.0)
        total += min(1.0, abs(float(a) - float(b)) / scale)
    return total / len(keys)


def _fit_probability(prediction: ResourcePrediction, headroom: int) -> float:
    if prediction.known_lower_bound_bytes > headroom:
        return 0.0
    if not prediction.samples:
        return 0.0
    return sum(sample <= headroom for sample in prediction.samples) / len(prediction.samples)


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


def _action_compatibility(
    action: CandidateResourceAction, device: GPUResourceState
) -> str:
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
    "ActiveResourceCommitment",
    "AdmissionDecision",
    "BoundedResourceTrajectory",
    "CandidateResourceAction",
    "GPUPlacementPlanner",
    "GPUResourceState",
    "OOMResourceEvidence",
    "ResourceDemandModel",
    "ResourceHistoryStore",
    "ResourcePrediction",
    "ResourceProfileObservation",
    "ResourceTrajectorySample",
    "oom_evidence",
    "resource_identity",
    "resource_state",
]
