"""Trace-based resource scheduler replay with explicit counterfactual boundaries."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ReplayPolicyName = Literal["recorded", "ari-v2-compat", "ari-v3-compat", "ari-v3.1"]


@dataclass(frozen=True, slots=True)
class ResourceTraceEvent:
    """One bounded scheduler snapshot; it never embeds scientific artifacts."""

    elapsed_seconds: float
    devices: tuple[Mapping[str, Any], ...]
    frontier: tuple[Mapping[str, Any], ...]
    decisions: tuple[Mapping[str, Any], ...]
    exploration: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResourceTraceEvent:
        def items(name: str) -> tuple[Mapping[str, Any], ...]:
            raw = value.get(name, ())
            return tuple(item for item in raw if isinstance(item, Mapping))

        return cls(
            float(value.get("elapsed_seconds", 0.0)),
            items("devices"),
            items("frontier"),
            items("decisions"),
            items("exploration"),
        )


class ResourceSchedulerReplay:
    """Replay factual scheduling until an alternative first disagrees.

    Compatibility policies are intentionally replay-only approximations, not parallel production
    schedulers. After divergence, metrics are trace-conditioned simulations and are labelled as
    such: the actual alternate future is unknowable without running it.
    """

    def __init__(
        self,
        events: Sequence[ResourceTraceEvent],
        *,
        observations: Sequence[Mapping[str, Any]] = (),
        placement_failures: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.events = tuple(sorted(events, key=lambda item: item.elapsed_seconds))
        self.observations = tuple(observations)
        self.placement_failures = tuple(placement_failures)

    @classmethod
    def from_execution(cls, execution_dir: Path) -> ResourceSchedulerReplay:
        root = execution_dir / "hpo-control" / "resources"
        trace = root / "scheduler-trace.jsonl"
        if not trace.is_file() or trace.is_symlink():
            raise FileNotFoundError(
                "This Study has no structured resource replay trace; older Studies remain "
                "readable but cannot be replayed counterfactually."
            )
        return cls(
            tuple(
                ResourceTraceEvent.from_mapping(item)
                for item in _read_jsonl(trace)
                if int(item.get("trace_version", 0)) == 1
            ),
            observations=_read_jsonl(root / "observations.jsonl"),
            placement_failures=_read_jsonl(root / "placement-oom.jsonl"),
        )

    def replay(self, policy: ReplayPolicyName = "ari-v3.1") -> dict[str, Any]:
        """Return auditable factual/simulated segments and research-relevant metrics."""
        if policy not in {"recorded", "ari-v2-compat", "ari-v3-compat", "ari-v3.1"}:
            raise ValueError(f"Unknown resource replay policy {policy!r}.")
        divergence: int | None = None
        suppressed: set[str] = set()
        admitted_candidates: dict[str, Mapping[str, Any]] = {}
        time_to: dict[str, dict[str, float]] = {}
        concurrency_area = idle_vram_time = active_area = throughput_area = 0.0
        blocked_time = starvation_time = 0.0
        rollback = 0.0
        checkpoint_costs: dict[str, float] = {}
        first_time = self.events[0].elapsed_seconds if self.events else 0.0
        prior_time = first_time
        prior_devices: tuple[Mapping[str, Any], ...] = ()
        prior_blocked = prior_starved = False

        for index, event in enumerate(self.events):
            recorded = _recorded_admissions(event)
            chosen = _policy_admissions(event, policy)
            if divergence is None and chosen != recorded:
                divergence = index
            suppressed.update(candidate for candidate, _gpu in recorded - chosen)
            suppressed.difference_update(candidate for candidate, _gpu in chosen)

            elapsed = max(0.0, event.elapsed_seconds - prior_time)
            for device in prior_devices:
                candidates = tuple(map(str, device.get("active_candidates", ())))
                active_indices = tuple(
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate not in suppressed
                )
                active = len(active_indices) if candidates else int(
                    device.get("active_runs", 0) or 0
                )
                total = int(device.get("total_bytes", 0) or 0)
                free = min(
                    total,
                    int(device.get("physical_free_bytes", 0) or 0)
                    + _suppressed_resident_bytes(device, suppressed),
                )
                concurrency_area += elapsed * active
                active_area += elapsed * int(active > 0)
                idle_vram_time += elapsed * free / max(1, total)
                throughput_area += elapsed * _active_throughput(
                    device, active_indices, candidates
                )
            blocked_time += elapsed * int(prior_blocked)
            starvation_time += elapsed * int(prior_starved)

            frontier = {str(item.get("candidate")): item for item in event.frontier}
            for candidate, gpu in chosen:
                admitted_candidates.setdefault(candidate, frontier.get(candidate, {}))
                matching_decisions = [
                    item
                    for item in event.decisions
                    if str(item.get("candidate_key")) == candidate
                    and str(item.get("target_gpu")) == gpu
                ]
                rollback += sum(
                    float(item.get("rollback_cost_seconds", 0.0) or 0.0)
                    for item in matching_decisions
                )
                for item in event.exploration:
                    if str(item.get("candidate")) != candidate or item.get("plan") != (
                        "CHECKPOINT_THEN_EXPLORE"
                    ):
                        continue
                    signature = str(item.get("experiment_signature", candidate))
                    checkpoint_costs[signature] = float(
                        item.get("checkpoint_cost_seconds", 0.0) or 0.0
                    )
                device = next(
                    (item for item in event.devices if str(item.get("gpu")) == gpu), {}
                )
                device_candidates = tuple(map(str, device.get("active_candidates", ())))
                active = (
                    sum(value not in suppressed for value in device_candidates)
                    if device_candidates
                    else int(device.get("active_runs", 0) or 0)
                ) + 1
                levels = time_to.setdefault(gpu, {})
                for level in range(2, active + 1):
                    levels.setdefault(str(level), event.elapsed_seconds)

            blocked = tuple(
                item for item in event.decisions if item.get("state") == "RESOURCE_BLOCKED"
            )
            admitted_values = [
                float(item.get("scientific_value", 0.0) or 0.0)
                for item in event.decisions
                if item.get("state") == "ADMITTED"
            ]
            prior_blocked = bool(blocked)
            prior_starved = bool(blocked) and (
                not admitted_values
                or max(float(item.get("scientific_value", 0.0) or 0.0) for item in blocked)
                > max(admitted_values)
            )
            prior_time = event.elapsed_seconds
            prior_devices = event.devices

        duration = max(0.0, prior_time - first_time)
        gpu_count = max(1, max((len(item.devices) for item in self.events), default=1))
        completed = [item for item in self.observations if item.get("state") == "completed"]
        pruned = [item for item in self.observations if item.get("state") == "pruned"]
        ooms = [item for item in self.observations if item.get("state") == "oom"]
        exploratory_ooms = [
            item for item in ooms if item.get("admission_mode") == "EXPLORATORY_ADMISSION"
        ]
        exploratory_candidates = [
            str(item.get("candidate_key", "")) for item in exploratory_ooms
        ]
        informative_ooms = len(set(exploratory_candidates))
        hours = duration / 3600.0
        peak_errors = [
            abs(int(item["predicted_peak_bytes"]) - int(item["observed_peak_bytes"]))
            for item in self.observations
            if isinstance(item.get("predicted_peak_bytes"), int | float)
            and isinstance(item.get("observed_peak_bytes"), int | float)
        ]
        envelope_errors = [
            abs(
                float(item["predicted_time_to_envelope_seconds"])
                - float(item["time_to_peak_seconds"])
            )
            for item in self.observations
            if isinstance(item.get("predicted_time_to_envelope_seconds"), int | float)
            and isinstance(item.get("time_to_peak_seconds"), int | float)
        ]
        normalized_value = sum(
            _frontier_value(item, rank)
            for rank, item in enumerate(admitted_candidates.values(), 1)
        )
        return {
            "replay_version": 1,
            "policy": policy,
            "events": len(self.events),
            "counterfactual_divergence": (
                {
                    "event_index": divergence,
                    "elapsed_seconds": self.events[divergence].elapsed_seconds,
                }
                if divergence is not None
                else None
            ),
            "segments": {
                "factual_events": divergence if divergence is not None else len(self.events),
                "counterfactual_simulated_events": (
                    len(self.events) - divergence if divergence is not None else 0
                ),
                "post_divergence_is_observed": False,
                "simulation_model": "trace-conditioned observed-envelope compatibility replay",
                "terminal_outcomes_source": (
                    "recorded observations"
                    if divergence is None
                    else "unavailable after counterfactual divergence"
                ),
                "model_diagnostics_source": "recorded observations",
            },
            "policy_assumptions": _policy_assumptions(policy),
            "metrics": {
                "time_to_concurrency": time_to,
                "mean_active_runs_per_gpu": concurrency_area / max(1e-12, duration * gpu_count),
                "idle_vram_time_gpu_seconds": idle_vram_time,
                "idle_compute_time_gpu_seconds": None,
                "gpu_active_fraction": active_area / max(1e-12, duration * gpu_count),
                "scientific_value_per_hour": normalized_value / max(1e-12, hours),
                "useful_actions_per_hour": len(admitted_candidates) / max(1e-12, hours),
                "aggregate_throughput": throughput_area / max(1e-12, duration),
                "completed_runs_per_hour": (
                    len(completed) / max(1e-12, hours) if divergence is None else None
                ),
                "pruned_informative_runs_per_hour": (
                    len(pruned) / max(1e-12, hours) if divergence is None else None
                ),
                "exploratory_oom_count": len(exploratory_ooms) if divergence is None else None,
                "informative_oom_count": informative_ooms if divergence is None else None,
                "redundant_oom_count": (
                    max(0, len(exploratory_candidates) - informative_ooms)
                    if divergence is None
                    else None
                ),
                "avoidable_oom_count": (
                    _avoidable_oom_count(self.observations) if divergence is None else None
                ),
                "known_dominated_oom_count": (
                    _known_dominated_oom_count(self.placement_failures)
                    if divergence is None
                    else None
                ),
                "rollback_gpu_seconds": rollback,
                "checkpoint_overhead_seconds": sum(checkpoint_costs.values()),
                "heavy_candidate_starvation_seconds": (
                    starvation_time if divergence is None else None
                ),
                "resource_blocked_seconds": blocked_time if divergence is None else None,
                "false_safe_admission_count": _false_safe_count(self.observations),
                "false_safe_admission_rate": _false_safe_rate(self.observations),
                # A counterfactual wait error cannot be inferred from occupancy alone. A future
                # simulator may populate it; until then the factual blocked duration remains
                # separate and this value is deliberately unknown.
                "false_conservative_wait_seconds": None,
                "false_conservative_wait_rate": None,
                "fit_probability_calibration": _fit_calibration(self.observations),
                "peak_prediction_mae_bytes": statistics.fmean(peak_errors) if peak_errors else None,
                "time_to_envelope_mae_seconds": (
                    statistics.fmean(envelope_errors) if envelope_errors else None
                ),
                "duration_seconds": duration,
            },
        }


def _recorded_admissions(event: ResourceTraceEvent) -> set[tuple[str, str]]:
    return {
        (str(item.get("candidate_key")), str(item.get("target_gpu")))
        for item in event.decisions
        if item.get("state") == "ADMITTED" and item.get("target_gpu") is not None
    }


def _policy_admissions(
    event: ResourceTraceEvent, policy: ReplayPolicyName
) -> set[tuple[str, str]]:
    recorded = _recorded_admissions(event)
    if policy in {"recorded", "ari-v3.1"}:
        return recorded
    if policy == "ari-v3-compat":
        # Approximate pre-v3.1 behaviour: safe admissions remain identical, while an
        # exploratory admission loses the accumulated WaitRegret contribution that v3.1
        # now carries exactly across frontier/signature changes.
        exploration_by_candidate = {
            str(item.get("candidate")): item for item in event.exploration
        }
        return {
            value
            for value in recorded
            if not any(
                item.get("candidate_key") == value[0]
                and item.get("admission_mode") == "EXPLORATORY_ADMISSION"
                for item in event.decisions
            )
            or (
                isinstance(
                    exploration_by_candidate.get(value[0], {}).get("final_delta_value"),
                    int | float,
                )
                and float(
                    exploration_by_candidate[value[0]].get("final_delta_value", 0.0)
                )
                - float(exploration_by_candidate[value[0]].get("wait_regret", 0.0) or 0.0)
                > 0
            )
        }
    ramping = {
        str(device.get("gpu"))
        for device in event.devices
        if "RAMPING" in set(map(str, device.get("resource_states", ())))
    }
    return {
        value
        for value in recorded
        if value[1] not in ramping
        and any(
            item.get("candidate_key") == value[0]
            and item.get("admission_mode") in {"BASELINE_ADMISSION", "SAFE_ADMISSION"}
            for item in event.decisions
        )
    }


def _suppressed_resident_bytes(
    device: Mapping[str, Any], suppressed: set[str]
) -> int:
    candidates = tuple(map(str, device.get("active_candidates", ())))
    current = tuple(device.get("active_current_bytes", ()))
    if len(candidates) != len(current):
        return 0
    return sum(
        max(0, int(value))
        for candidate, value in zip(candidates, current, strict=True)
        if candidate in suppressed and isinstance(value, int | float)
    )


def _active_throughput(
    device: Mapping[str, Any], active_indices: Sequence[int], candidates: Sequence[str]
) -> float:
    values = tuple(device.get("active_throughputs", ()))
    if candidates and len(values) == len(candidates):
        return sum(
            float(values[index])
            for index in active_indices
            if isinstance(values[index], int | float)
        )
    return sum(float(value) for value in values if isinstance(value, int | float))


def _policy_assumptions(policy: ReplayPolicyName) -> Mapping[str, Any]:
    if policy == "ari-v2-compat":
        return {
            "kind": "simplified compatibility approximation",
            "rule": "admit only recorded safe decisions outside RAMPING",
        }
    if policy == "ari-v3-compat":
        return {
            "kind": "simplified compatibility approximation",
            "rule": "remove exact accumulated WaitRegret from recorded exploratory decisions",
        }
    return {
        "kind": "recorded production decisions",
        "rule": "use the scheduler decisions persisted by the execution",
    }


def _frontier_value(value: Mapping[str, Any], rank: int) -> float:
    contract = value.get("value_contract")
    if isinstance(contract, Mapping) and isinstance(contract.get("normalized_value"), int | float):
        return min(1.0, max(0.0, float(contract["normalized_value"])))
    return 1.0 / max(1, rank)


def _false_safe_count(observations: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        item.get("state") == "oom"
        and item.get("admission_mode") == "SAFE_ADMISSION"
        for item in observations
    )


def _false_safe_rate(observations: Sequence[Mapping[str, Any]]) -> float | None:
    safe = [item for item in observations if item.get("admission_mode") == "SAFE_ADMISSION"]
    return _false_safe_count(observations) / len(safe) if safe else None


def _avoidable_oom_count(observations: Sequence[Mapping[str, Any]]) -> int:
    """Count OOMs admitted as safe rather than bounded information experiments."""
    return sum(
        item.get("state") == "oom"
        and item.get("admission_mode") == "SAFE_ADMISSION"
        for item in observations
    )


def _known_dominated_oom_count(failures: Sequence[Mapping[str, Any]]) -> int:
    """Count exact repeated placement failures already dominated by earlier evidence."""
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    repeated = 0
    for item in failures:
        identity = (
            str(item.get("candidate_key", "")),
            str(item.get("hardware", "")),
            tuple(sorted(map(str, item.get("resident_candidates", ())))),
        )
        if identity in seen:
            repeated += 1
        seen.add(identity)
    return repeated


def _fit_calibration(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = [
        (float(item["predicted_fit_probability"]), bool(item["placement_succeeded"]))
        for item in observations
        if isinstance(item.get("predicted_fit_probability"), int | float)
        and isinstance(item.get("placement_succeeded"), bool)
    ]
    bins: dict[str, list[tuple[float, bool]]] = {}
    for probability, success in pairs:
        index = min(9, max(0, int(probability * 10)))
        bins.setdefault(f"{index / 10:.1f}-{(index + 1) / 10:.1f}", []).append(
            (probability, success)
        )
    return {
        "observations": len(pairs),
        "brier_score": (
            statistics.fmean((probability - float(success)) ** 2 for probability, success in pairs)
            if pairs
            else None
        ),
        "bins": {
            key: {
                "count": len(values),
                "mean_predicted": statistics.fmean(item[0] for item in values),
                "observed_fit_rate": statistics.fmean(float(item[1]) for item in values),
            }
            for key, values in bins.items()
        },
    }


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file() or path.is_symlink():
        return ()
    output: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            output.append(value)
    return tuple(output)


__all__ = ["ReplayPolicyName", "ResourceSchedulerReplay", "ResourceTraceEvent"]
