"""Execution-only fixed CPU dispatcher connecting native Study planning to exact Fleet leases.

No optimizer or sweep analyzer lives here. The existing fixed/sequential planner keeps its seed
blocks, callback and stopping authority. Remote/GPU/adaptive streaming remain explicitly gated.
"""

from __future__ import annotations

import json
import time
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
from lambdaforge.work.models import WorkResult
from lambdaforge.work.study import StudyTelemetry


class CoordinatedDispatcher:
    """Retain one central scientific authority while prepared direct CPU Jobs run independently.

    Construct with an initialized coordinator and verified executors. Invocations persist before
    enqueue, so an owner can reconnect without reconstructing operational paths from its cwd.
    Exceptions never cancel healthy provider Jobs. This boundary is not a public Fleet launcher.
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
        del objective_metric, objective_mode, objective, historical_results
        del on_resource_blocked, on_resource_infeasible, on_queued_cancel
        if (
            resources.gpu_count
            or policy.early_stopping
            or any(value["definition"].get("search_policy") for value in specifications)
        ):
            raise ValueError(
                "Distributed GPU/adaptive execution requires provider and metric integration."
            )
        if not specifications:
            return ()
        if parallelism < 1:
            raise ValueError("Distributed parallelism must be positive.")
        equivalence = next(iter(self.executors.values())).equivalence
        snapshot = self.coordinator.snapshot()
        queued: dict[str, dict[str, Any]] = {}
        results: list[WorkResult] = []

        def global_run(value: Mapping[str, Any]) -> GlobalRun:
            return GlobalRun(
                snapshot["study_identity"],
                str(value["trial_index"]),
                value.get("seed"),
                value.get("hpo_phase", "search"),
                {},
                value["parameters"],
                equivalence,
                requires_gpu=False,
                priority_class="required",
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
                if path.exists() and json.loads(path.read_text()) != value:
                    raise ValueError("One global Run cannot acquire another invocation.")
                if not path.exists():
                    atomic_write_json(path, value)
                queued[run.key] = value
                runs.append(run)
            self.coordinator.enqueue(runs)

        delivered: set[str] = set()
        with self.coordinator.leadership():
            enqueue(specifications)
            while queued:
                for cluster, executor in self.executors.items():
                    self.coordinator.reconcile(cluster, executor)
                records = self.coordinator.run_records()
                for record in records:
                    key = record["run_key"]
                    if key in queued and record["state"] == "failed" and not record["accepted"]:
                        raise RuntimeError(
                            f"Shard rejected before scientific execution: {key}; "
                            f"{record['attempts'][-1].get('failure', {})}"
                        )
                    if key not in queued or not record["accepted"] or key in delivered:
                        continue
                    envelope = self.coordinator.result(key)
                    assert envelope is not None
                    from lambdaforge.work.runner import _work_result_from_mapping

                    result = _work_result_from_mapping(envelope["result"])
                    if (
                        result.seed != queued[key].get("seed")
                        or result.trial is None
                        or int(result.trial["index"]) != int(queued[key]["trial_index"])
                    ):
                        raise ValueError("Native result differs from its exact planner invocation.")
                    delivered.add(key)
                    queued.pop(key)
                    results.append(result)
                    if on_result is not None:
                        pending_keys = {
                            item["run_key"]
                            for item in records
                            if item["state"] != "planned"
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
                    slots = min(offer.slots, room)
                    offers.append(replace(offer, slots=slots))
                    room -= slots
                for shard in self.coordinator.unsubmitted_shards():
                    self.coordinator.submit_shard(shard, self.executors[shard.cluster])
                for shard in self.coordinator.plan_shards(offers):
                    self.coordinator.submit_shard(shard, self.executors[shard.cluster])
                if active == 0 and (
                    status["remaining_runs"] == 0 or status["remaining_seconds"] == 0
                ):
                    raise RuntimeError(
                        "Study budget exhausted with required evidence still pending."
                    )
                time.sleep(self.poll_seconds)
        return tuple(results)
