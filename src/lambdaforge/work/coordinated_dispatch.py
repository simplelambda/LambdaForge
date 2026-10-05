"""Execution-only Study dispatcher connecting native planning to exact Fleet leases.

No optimizer or sweep analyzer lives here. Native adaptive/fixed/sequential planners retain seed,
pruning, callback and stopping authority. Ordered metrics and exact stop requests cross provider
owners; checkpoint continuation and recovery remain explicitly gated.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.FleetPlacement import GlobalRun
from lambdaforge.controlplane.PreparedShardExecutor import PreparedShardExecutor
from lambdaforge.controlplane.StudyCoordinator import StudyCoordinator
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.hpo.AdaptiveSearch import AdaptiveSearchPolicy
from lambdaforge.work.atomic import atomic_write_json
from lambdaforge.work.dispatcher import FrontierCallback, InfeasibleCallback, ResultCallback
from lambdaforge.work.fleet_stream import ScalarStream
from lambdaforge.work.models import WorkResult
from lambdaforge.work.study import StudyTelemetry


class CoordinatedDispatcher:
    """Retain one central scientific authority while prepared provider Jobs execute Runs.

    Construct with an initialized coordinator and verified executors. Invocations persist before
    enqueue, so an owner can reconnect without reconstructing operational paths from its cwd.
    Exceptions never cancel healthy provider Jobs. FleetStudyService owns public preparation.
    """

    def __init__(
        self,
        coordinator: StudyCoordinator,
        executors: Mapping[str, PreparedShardExecutor],
        *,
        poll_seconds: float = 0.1,
    ) -> None:
        if not 0 < poll_seconds <= 60 or not executors:
            raise ValueError("Coordinated CPU dispatch requires executors and bounded polling.")
        if len({executor.equivalence for executor in executors.values()}) != 1:
            raise ValueError("All prepared executors must share one verified execution stratum.")
        self.coordinator = coordinator
        self.executors = dict(executors)
        self.poll_seconds = poll_seconds

    def invocation(self, key: str) -> Mapping[str, Any]:
        """Read the exact durable invocation; executor placement never authors science."""
        path = self.coordinator.root / "invocations" / (key.removeprefix("sha256:") + ".json")
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Prepared invocation is symlinked.")
        return dict(json.loads(path.read_text()))

    def __call__(
        self,
        specifications: Sequence[dict[str, Any]],
        *,
        resources: ResourceRequest,
        policy: AdaptiveSearchPolicy,
        objective_metric: str,
        objective_mode: str,
        objective: Mapping[str, Any],
        historical_results: Sequence[WorkResult],
        parallelism: int,
        telemetry: StudyTelemetry | None = None,
        on_result: ResultCallback | None = None,
        on_resource_blocked: FrontierCallback | None = None,
        on_resource_infeasible: InfeasibleCallback | None = None,
        on_queued_cancel: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> tuple[WorkResult, ...]:
        """Execute required CPU evidence and feed every accepted result to its original planner."""
        del on_resource_infeasible
        if (
            resources.gpu_count
            and any(executor.allocation_id is None for executor in self.executors.values())
        ) or (
            policy.early_stopping
            and any(executor.allocation_id is None for executor in self.executors.values())
        ):
            raise ValueError(
                "Distributed GPU/pruning execution requires prepared allocation owners."
            )
        if not specifications:
            return ()
        if parallelism < 1:
            raise ValueError("Distributed parallelism must be positive.")
        equivalence = next(iter(self.executors.values())).equivalence
        snapshot = self.coordinator.snapshot()
        queued: dict[str, dict[str, Any]] = {}
        results: list[WorkResult] = []
        streams: dict[str, ScalarStream] = {}
        complete_streams: set[str] = set()
        pruning_acknowledged: set[str] = set()
        frontier_revision: tuple[str, ...] | None = None
        adaptive = any(value["definition"].get("search_policy") for value in specifications)

        def global_run(value: Mapping[str, Any]) -> GlobalRun:
            return GlobalRun(
                snapshot["study_identity"],
                str(value["trial_index"]),
                value.get("seed"),
                value.get("hpo_phase", "search"),
                {},
                value["parameters"],
                equivalence,
                requires_gpu=bool(resources.gpu_count),
                gpu_memory_bytes=resources.gpu_memory_bytes,
                priority_class=(
                    "confirmation"
                    if value.get("hpo_phase") == "confirmation"
                    else "required"
                    if value.get("evidence_required") or not adaptive
                    else "primary"
                ),
            )

        def enqueue(values: Sequence[Mapping[str, Any]]) -> None:
            runs = []
            for specification in values:
                # Only paths become strings; arbitrary objects never become portable by repr.
                def portable(value: Any) -> Any:
                    if isinstance(value, Path):
                        return str(value)
                    raise TypeError(f"Nonportable invocation value: {type(value).__name__}")

                value = json.loads(
                    json.dumps(dict(specification), default=portable, allow_nan=False)
                )
                # Workers need a compact invocation, never the whole fixed evidence design or
                # scientific policy. The original planner retains both and owns all callbacks.
                definition = dict(value["definition"])
                for name in ("study_design", "search_policy", "execution_policy"):
                    definition.pop(name, None)
                value["definition"] = definition
                if value["execution_id"] != snapshot["execution_id"]:
                    raise ValueError(
                        "Dispatcher cannot change its exact native Execution identity."
                    )
                if value.get("study_recovery") or value.get("restart") or value.get("hpo_fidelity"):
                    raise ValueError(
                        "Checkpoint/adoption binding is not enabled for prepared CPU dispatch."
                    )
                run = global_run(value)
                path = (
                    self.coordinator.root
                    / "invocations"
                    / (run.key.removeprefix("sha256:") + ".json")
                )
                if path.is_symlink() or path.parent.is_symlink():
                    raise ValueError("Prepared invocation is symlinked.")
                existing = next(
                    (item for item in self.coordinator.run_records() if item["run_key"] == run.key),
                    None,
                )
                if path.exists() and json.loads(path.read_text()) != value:
                    previous = json.loads(path.read_text())
                    mutable = {
                        "hpo_scheduler_action",
                        "hpo_scheduler_priority",
                        "hpo_scientific_value",
                        "hpo_opportunistic",
                        "hpo_probe_purpose",
                        "hpo_target_questions",
                        "evidence_required",
                        "evidence_requirement",
                        "hpo_pruner_calibration",
                    }
                    if {name: child for name, child in previous.items() if name not in mutable} != {
                        name: child for name, child in value.items() if name not in mutable
                    }:
                        raise ValueError(
                            "One global Run cannot acquire another scientific invocation."
                        )
                    if existing is None or existing["attempts"]:
                        raise ValueError("A leased invocation cannot be replanned.")
                    audit = path.parent / "revisions" / run.key.removeprefix("sha256:")
                    from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity

                    revision = ScientificIdentity.from_payload(previous).digest.removeprefix(
                        "sha256:"
                    )
                    atomic_write_json(audit / (revision + ".json"), previous)
                    self.coordinator.replan_unstarted(run)
                    atomic_write_json(path, value)
                elif existing is not None and existing["state"] == "cancelled":
                    self.coordinator.replan_unstarted(run)
                if not path.exists():
                    atomic_write_json(path, value)
                queued[run.key] = value
                runs.append(run)
            self.coordinator.enqueue(runs)

        delivered: set[str] = set()
        with self.coordinator.leadership():
            for executor in self.executors.values():
                if executor.allocation_id is not None:
                    executor.start_allocation(
                        specifications[0]["source"], snapshot["study_identity"]
                    )
            enqueue(specifications)
            while queued:
                for cluster, executor in self.executors.items():
                    self.coordinator.reconcile(cluster, executor)
                records = self.coordinator.run_records()
                # The exact terminal native metrics must be acknowledged before the planner
                # consumes a result. A terminal response can span several bounded reads.
                groups: dict[tuple[str, str], dict[str, ScalarStream]] = {}
                for record in records:
                    key = record["run_key"]
                    if key not in queued or not record["attempts"]:
                        continue
                    attempt = record["attempts"][-1]
                    executor = self.executors[attempt["cluster"]]
                    if executor.allocation_id is None:
                        continue
                    if key not in streams:
                        streams[key] = ScalarStream(
                            self.coordinator.root
                            / "scalars"
                            / key.removeprefix("sha256:")
                            / str(attempt["attempt"]),
                            {
                                "study_identity": snapshot["study_identity"],
                                "run_key": key,
                                **{
                                    name: attempt[name]
                                    for name in ("attempt", "lease_id", "cluster", "shard_id")
                                },
                            },
                        )
                    if key not in complete_streams:
                        groups.setdefault((attempt["cluster"], attempt["shard_id"]), {})[key] = (
                            streams[key]
                        )
                for (cluster, shard_id), receivers in groups.items():
                    try:
                        completed = self.executors[cluster].pull_scalars(shard_id, receivers)
                    except (OSError, ConnectionError, TimeoutError):
                        # Never guess a terminal metric stream or issue a replacement lease.
                        continue
                    complete_streams.update(key for key, complete in completed.items() if complete)
                for record in records:
                    key = record["run_key"]
                    if key in queued and telemetry is not None and record["attempts"]:
                        attempt = record["attempts"][-1]
                        live = self.executors[attempt["cluster"]].run_observation(key)
                        if live is not None:
                            telemetry.run_placement(
                                queued[key],
                                cluster=attempt["cluster"],
                                job_id=attempt["job_id"],
                                shard_id=attempt["shard_id"],
                                lease_id=attempt["lease_id"],
                                attempt=attempt["attempt"],
                            )
                            telemetry.remote_run_observed(queued[key], live)
                    if key in queued and record["state"] == "failed" and not record["accepted"]:
                        raise RuntimeError(
                            f"Shard rejected before scientific execution: {key}; "
                            f"{record['attempts'][-1].get('failure', {})}"
                        )
                    if key not in queued or not record["accepted"] or key in delivered:
                        continue
                    if key in streams and key not in complete_streams:
                        continue
                    envelope = self.coordinator.result(key)
                    assert envelope is not None
                    from lambdaforge.work.runner import _work_result_from_mapping

                    result = _work_result_from_mapping(envelope["result"])
                    if key in streams:
                        result = replace(
                            result,
                            scalar_mirror_paths=(
                                streams[key].path("metrics"),
                                streams[key].path("training-metrics"),
                            ),
                        )
                    if (
                        result.seed != queued[key].get("seed")
                        or result.trial is None
                        or int(result.trial["index"]) != int(queued[key]["trial_index"])
                    ):
                        raise ValueError("Native result differs from its exact planner invocation.")
                    delivered.add(key)
                    specification = queued.pop(key)
                    results.append(result)
                    if telemetry is not None:
                        telemetry.run_finished(specification, result)
                        attempt = record["attempts"][-1]
                        telemetry.run_placement(
                            specification,
                            cluster=attempt["cluster"],
                            job_id=attempt["job_id"],
                            shard_id=attempt["shard_id"],
                            lease_id=attempt["lease_id"],
                            attempt=attempt["attempt"],
                        )
                    if on_result is not None:
                        pending_keys = {
                            item["run_key"] for item in records if item["state"] != "planned"
                        }
                        # Terminal envelopes not yet delivered in this batch still belong to
                        # pending, not the revocable frontier. A callback must never discard them.
                        planned = [
                            value for item, value in queued.items() if item not in pending_keys
                        ]
                        pending = [value for item, value in queued.items() if item in pending_keys]
                        refill = on_result(result, planned, pending)
                        retained = {global_run(value).key for value in refill}
                        if retained & (pending_keys | delivered):
                            raise ValueError("Planner refill repeats a pending or completed Run.")
                        withdrawn = [
                            item
                            for item in queued
                            if item not in pending_keys and item not in retained
                        ]
                        if withdrawn:
                            self.coordinator.withdraw_planned(
                                withdrawn, reason="native-scientific-planner-frontier-replacement"
                            )
                        for item in withdrawn:
                            queued.pop(item)
                        enqueue(refill)
                    if telemetry is not None:
                        telemetry.refresh()
                if policy.early_stopping:
                    from lambdaforge.work.runner import _request_early_stops

                    live_specifications = []
                    for record in records:
                        key = record["run_key"]
                        if key not in queued or key not in streams or record["state"] != "running":
                            continue
                        stop = streams[key].root / "central.stop"
                        live_specifications.append(
                            {
                                **queued[key],
                                "hpo_metrics_path": str(streams[key].path("metrics")),
                                "hpo_training_metrics_path": str(
                                    streams[key].path("training-metrics")
                                ),
                                "hpo_stop_path": str(stop),
                            }
                        )
                    _request_early_stops(
                        live_specifications,
                        metric=objective_metric,
                        mode=objective_mode,
                        min_step=policy.early_stopping_min_step,
                        confirmations=policy.early_stopping_confirmations,
                        probability_threshold=policy.early_stopping_probability_threshold,
                        margin=policy.early_stopping_equivalence_margin,
                        historical_results=(*historical_results, *results),
                        objective=objective,
                    )
                    for value in live_specifications:
                        key = global_run(value).key
                        stop = Path(value["hpo_stop_path"])
                        if key in pruning_acknowledged or not stop.exists():
                            continue
                        outbox = streams[key].root / "pruning.json"
                        if not outbox.exists():
                            atomic_write_json(
                                outbox,
                                {
                                    "identity": streams[key].identity,
                                    "run_key": key,
                                    "reason": stop.read_text(),
                                    "evidence": json.loads(
                                        stop.with_name(stop.name + ".evidence.json").read_text()
                                    ),
                                },
                            )
                        identity = streams[key].identity
                        try:
                            acknowledgement = self.executors[identity["cluster"]].request_pruning(
                                identity["shard_id"], json.loads(outbox.read_text())
                            )
                        except (OSError, ConnectionError, TimeoutError):
                            continue
                        if acknowledgement != "not-running":
                            pruning_acknowledged.add(key)
                    from lambdaforge.work.runner import _cancel_queued_pruned_candidates

                    planned_keys = {
                        item["run_key"] for item in records if item["state"] == "planned"
                    }
                    unstarted = deque(queued[key] for key in queued if key in planned_keys)
                    _cancel_queued_pruned_candidates(
                        unstarted,
                        live_specifications,
                        telemetry=telemetry,
                        on_cancel=on_queued_cancel,
                    )
                    retained = {global_run(value).key for value in unstarted}
                    withdrawn = [
                        key for key in queued if key in planned_keys and key not in retained
                    ]
                    if withdrawn:
                        self.coordinator.withdraw_planned(
                            withdrawn, reason="native-candidate-performance-prune"
                        )
                        for key in withdrawn:
                            queued.pop(key)
                if not queued:
                    break
                status = self.coordinator.snapshot()
                if status["state"] in {"paused", "resuming"}:
                    raise RuntimeError("Study is paused; exact pending state remains durable.")
                records = self.coordinator.run_records()
                active = sum(
                    item["state"] in {"leased", "queued", "running", "unknown_remote"}
                    for item in records
                )
                if status["remaining_seconds"] == 0 and not any(
                    item["state"] in {"queued", "running", "unknown_remote"} for item in records
                ):
                    raise RuntimeError(
                        "Study time budget exhausted with unsubmitted required evidence."
                    )
                offers = []
                room = max(0, parallelism - active)
                for executor in self.executors.values():
                    offer = executor.offer(records)
                    offers.append(offer)
                planned_keys = {item["run_key"] for item in records if item["state"] == "planned"}
                revision = tuple(sorted(queued))
                if (
                    on_resource_blocked is not None
                    and revision != frontier_revision
                    and room > len(planned_keys)
                    and sum(offer.slots for offer in offers) > len(planned_keys)
                ):
                    frontier_revision = revision
                    extension = on_resource_blocked(
                        [value for key, value in queued.items() if key in planned_keys],
                        [value for key, value in queued.items() if key not in planned_keys],
                    )
                    if any(
                        global_run(value).key in queued or global_run(value).key in delivered
                        for value in extension
                    ):
                        raise ValueError("Native frontier extension repeats existing evidence.")
                    enqueue(extension)
                for shard in self.coordinator.unsubmitted_shards():
                    self.coordinator.submit_shard(shard, self.executors[shard.cluster])
                for shard in self.coordinator.plan_shards(offers, maximum_new_runs=room):
                    self.coordinator.submit_shard(shard, self.executors[shard.cluster])
                if telemetry is not None:
                    telemetry.refresh()
                if active == 0 and (
                    status["remaining_runs"] == 0 or status["remaining_seconds"] == 0
                ):
                    raise RuntimeError(
                        "Study budget exhausted with required evidence still pending."
                    )
                time.sleep(self.poll_seconds)
            # WorkRunner may invoke this boundary again for confirmation or another native
            # evidence phase. The parent Fleet service owns allocation lifetime, not a batch.
        return tuple(results)
