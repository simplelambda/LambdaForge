"""Fail-closed owner/inbox/offer regressions; no physical inventory grants permission."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.Fleet import FleetMember
from lambdaforge.controlplane.FleetPlacement import ExecutionEquivalence
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.PreparedShardExecutor import PreparedShardExecutor
from lambdaforge.controlplane.StudyCoordinator import StudyShard
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.work.atomic import atomic_write_json
from lambdaforge.work.member import (
    allocation_manifest,
    enqueue,
    observe,
    request_pruning,
    scientific_stream,
)
from tests.work.test_concrete_shard import prepared_shard


def allocation(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    _control, shard, invocations = prepared_shard(tmp_path)
    source = tmp_path / "science.yaml"
    source.write_text(
        "name: science\nrun: tests.work_cases.ConfirmationFailureWork\n"
        "with: {quality: 0.1}\nseeds: [101, 102]\nresources: {cpu: 4}\n"
    )
    root = tmp_path / "member"
    job_id = "job-fleet-allocation-" + "a" * 64
    atomic_write_json(
        root / "allocation.json",
        {
            "allocation_version": 1,
            "root": str(root),
            "job_id": job_id,
            "study_identity": shard.study_identity,
            "member": shard.member.to_dict(),
            "source": str(source),
            "resources": {"cpu": 4},
            "equivalence": shard.leases[0].run.equivalence.to_dict(),
            "input_bindings": [],
        },
    )
    wave = {
        "allocation_version": 1,
        "job_id": job_id,
        "shard": shard.to_dict(),
        "invocations": invocations,
        "equivalence": shard.leases[0].run.equivalence.to_dict(),
    }
    return root, wave


def test_member_inbox_is_immutable_idempotent_and_drain_safe(tmp_path: Path) -> None:
    root, wave = allocation(tmp_path)
    assert enqueue(root, wave) == wave["job_id"]
    assert enqueue(root, wave) == wave["job_id"]
    payload = observe(root, shard_id=wave["shard"]["shard_id"])
    assert payload["accepted"] and payload["observation"]["results"] == {}
    changed = json.loads(json.dumps(wave))
    key = next(iter(changed["invocations"]))
    changed["invocations"][key]["seed_metadata"] = {"changed": True}
    with pytest.raises(ValueError, match="cannot change"):
        enqueue(root, changed)
    atomic_write_json(root / "shutdown.json", {"job_id": wave["job_id"]})
    with pytest.raises(ValueError, match="draining"):
        enqueue(root, wave)


def native_stream_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], Path]:
    from lambdaforge.work.study import StudyTelemetry, study_run_key

    root, wave = allocation(tmp_path)
    enqueue(root, wave)
    worker = root / "waves" / wave["shard"]["shard_id"] / "worker"
    invocations = list(wave["invocations"].values())
    atomic_write_json(
        worker / "worker.json",
        {"manifest": {"shard": wave["shard"], "invocations": invocations}, "results": {}},
    )
    telemetry = StudyTelemetry(worker / "study")
    for lease in wave["shard"]["leases"]:
        value = wave["invocations"][lease["run"]["run_key"]]
        directory = worker / "execution" / str(value["trial_index"]) / str(value["seed"])
        directory.mkdir(parents=True)
        (directory / "metrics.jsonl").write_text('{"name":"score","value":0.1,"step":3}\n')
        telemetry._write_run(
            study_run_key(value),
            {
                "attempt_id": "attempt-0001",
                "trial": value["trial_index"],
                "seed": value["seed"],
                "phase": "search",
                "state": "running",
                "run_dir": str(directory),
                "metrics_path": str(directory / "metrics.jsonl"),
                "training_metrics_path": str(directory / "training-metrics.jsonl"),
            },
        )
    return root, wave, worker


def test_exact_member_scalar_stream_and_replayed_directed_pruning(tmp_path: Path) -> None:
    from lambdaforge.work.fleet_stream import CHANNELS, ScalarStream

    root, wave, worker = native_stream_fixture(tmp_path)
    lease = wave["shard"]["leases"][0]
    key = lease["run"]["run_key"]
    cursors = {key: dict.fromkeys(CHANNELS, 0)}
    response = scientific_stream(root, wave["shard"]["shard_id"], cursors)
    stream = response["streams"][key]
    receiver = ScalarStream(tmp_path / "receiver", stream["identity"])
    assert not receiver.ingest(stream)
    assert not receiver.ingest(stream)
    assert json.loads(receiver.path("metrics").read_text())["step"] == 3
    command = {
        "identity": stream["identity"],
        "run_key": key,
        "reason": "native calibrated prune",
        "evidence": {"common_step": 3, "probability_competitive": 0.01},
    }
    assert request_pruning(root, wave["shard"]["shard_id"], command)["status"] == "requested"
    assert request_pruning(root, wave["shard"]["shard_id"], command)["status"] == "requested"
    stops = list((worker / "execution").rglob("*.stop"))
    assert len(stops) == 1
    assert stops[0].read_text() == command["reason"]
    assert (
        json.loads(stops[0].with_name(stops[0].name + ".evidence.json").read_text())
        == command["evidence"]
    )
    assert read_worker_state(worker)["results"] == {}  # a request is not terminal evidence
    with pytest.raises(ValueError, match="cannot change"):
        request_pruning(root, wave["shard"]["shard_id"], {**command, "reason": "different"})
    with pytest.raises(ValueError, match="exact live lease"):
        request_pruning(
            root,
            wave["shard"]["shard_id"],
            {**command, "identity": {**stream["identity"], "attempt": 2}},
        )


def read_worker_state(worker: Path) -> dict[str, Any]:
    return dict(json.loads((worker / "worker.json").read_text()))


@pytest.mark.parametrize("protected", ["evidence_required", "hpo_startup_anchor", "confirmation"])
def test_member_never_prunes_protected_native_evidence(tmp_path: Path, protected: str) -> None:
    root, wave, worker = native_stream_fixture(tmp_path)
    lease = wave["shard"]["leases"][0]
    key = lease["run"]["run_key"]
    value = wave["invocations"][key]
    value["hpo_phase" if protected == "confirmation" else protected] = (
        "confirmation" if protected == "confirmation" else True
    )
    atomic_write_json(root / "inbox" / (wave["shard"]["shard_id"] + ".json"), wave)
    command = {
        "run_key": key,
        "identity": {
            "study_identity": wave["shard"]["study_identity"],
            "run_key": key,
            "attempt": lease["attempt"],
            "lease_id": lease["lease_id"],
            "cluster": "A",
            "shard_id": wave["shard"]["shard_id"],
        },
        "reason": "stop",
        "evidence": {},
    }
    with pytest.raises(ValueError, match="protected"):
        request_pruning(root, wave["shard"]["shard_id"], command)
    assert not list(worker.rglob("*.stop"))


def test_native_reprioritization_only_changes_unleased_planning(tmp_path: Path) -> None:
    from lambdaforge.controlplane.Fleet import ClusterHealth
    from lambdaforge.controlplane.FleetPlacement import ClusterOffer

    control, shard, _invocations = prepared_shard(tmp_path)
    run = replace(shard.leases[0].run, candidate="9", priority_class="primary")
    control.enqueue([run])
    revised = replace(run, priority_class="required")
    control.replan_unstarted(revised)
    record = next(value for value in control.run_records() if value["run_key"] == run.key)
    assert record["definition"]["priority_class"] == "required"
    assert len(record["planning_history"]) == 1
    control.replan_unstarted(revised)
    assert (
        len(
            next(value for value in control.run_records() if value["run_key"] == run.key)[
                "planning_history"
            ]
        )
        == 1
    )
    with pytest.raises(ValueError, match="scientific identity"):
        control.replan_unstarted(replace(revised, parameters={"quality": 999}))
    # Use another exact member with no previously leased Runs, retaining the same stratum.
    from lambdaforge.controlplane.Fleet import Fleet, FleetMember

    control.update_fleet(Fleet("cpu", (shard.member, FleetMember("B", max_runs=1))))
    control.plan_shards(
        [
            ClusterOffer(
                "B",
                ClusterHealth.ONLINE,
                "local",
                "auto",
                0,
                100,
                slots=1,
                equivalence=run.equivalence,
                environment_ready=True,
                inputs_ready=True,
                local_admission_verified=True,
            )
        ]
    )
    with pytest.raises(ValueError, match="unleased"):
        control.replan_unstarted(run)


@pytest.mark.parametrize("remote,code", [(True, 255), (True, 1), (False, 255)])
def test_member_transport_failure_is_unknown_not_a_failed_scientific_run(
    tmp_path: Path,
    remote: bool,
    code: int,
) -> None:
    root, wave = allocation(tmp_path)
    profile = ClusterProfile(
        "A",
        transport="ssh" if remote else "local",
        host="synthetic.invalid" if remote else None,
        environment="managed",
        workspace=str(tmp_path / "provider"),
    )
    plane = Mock()
    plane.catalog.get.return_value = profile
    plane.factory.transport.return_value.run.return_value.returncode = code
    plane.factory.transport.return_value.run.return_value.stderr = "synthetic failure"
    executor = PreparedShardExecutor(
        profile,
        FleetMember.from_mapping(wave["shard"]["member"]),
        root=root,
        resources=ResourceRequest(cpu_cores=4),
        equivalence=ExecutionEquivalence(**wave["equivalence"]),
        invocation=lambda key: wave["invocations"][key],
        control_plane=plane,
        allocation_id="a" * 64,
    )
    executor._member_receipt = Mock(return_value={"root": str(root), "python": "python"})  # type: ignore[method-assign]
    with pytest.raises(ConnectionError if remote and code == 255 else RuntimeError):
        executor._member_call("--observe")


def test_member_rejects_foreign_owner_class_and_duplicate_run_waves(tmp_path: Path) -> None:
    root, wave = allocation(tmp_path)
    with pytest.raises(ValueError, match="identities"):
        enqueue(root, {**wave, "job_id": "another-owner"})
    changed = json.loads(json.dumps(wave))
    next(iter(changed["invocations"].values()))["definition"]["work_class"] = (
        "tests.work_cases.CudaWork"
    )
    with pytest.raises(ValueError, match="Work class"):
        enqueue(root, changed)
    enqueue(root, wave)
    duplicate = {**wave, "shard": {**wave["shard"], "shard_id": "b" * 64}}
    with pytest.raises(ValueError, match="second member wave"):
        enqueue(root, duplicate)


def test_member_root_and_metadata_symlinks_fail_closed(tmp_path: Path) -> None:
    root, _wave = allocation(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="regular owned"):
        allocation_manifest(alias)


def test_live_remote_view_is_not_local_file_or_completed_scientific_evidence(
    tmp_path: Path,
) -> None:
    from lambdaforge.work.study import StudyTelemetry, study_run_key

    _control, _shard, invocations = prepared_shard(tmp_path)
    specification = next(iter(invocations.values()))
    telemetry = StudyTelemetry(tmp_path / "central-view")
    telemetry.initialize(
        name="science",
        execution_id="execution",
        strategy="exhaustive",
        objective={"metric": "score", "mode": "max"},
        specifications=[specification],
    )
    telemetry.schedule([specification])
    # A local file at the same absolute path must not contaminate a remote observation.
    unrelated = tmp_path / "unrelated-metrics.jsonl"
    unrelated.write_text('{"name":"score","value":99.0,"step":99}\n')
    observation = {
        "state": "running",
        "metrics": {"score": 0.2},
        "latest_step": 7,
        "best_step": 7,
        "best_objective": 0.2,
        "metrics_path": str(unrelated),
        "parameters": {"quality": 999},
        "seed": 999,
    }
    with pytest.raises(ValueError, match="placement"):
        telemetry.remote_run_observed(specification, observation)
    telemetry.run_placement(
        specification,
        cluster="A",
        job_id="job-owned",
        shard_id="a" * 64,
        lease_id="lease-owned",
        attempt=1,
    )
    telemetry.remote_run_observed(specification, observation)
    snapshot = telemetry.refresh()
    view = snapshot["candidates"][0]
    run = view["runs"][0]
    assert run["latest_step"] == 7
    assert run["latest_metrics"]["score"] == 0.2
    assert run["final_objective"] is None
    assert snapshot["counts"]["completed_runs"] == 0
    assert run["seed"] == specification["seed"]
    assert telemetry._run_state(study_run_key(specification)).get("parameters") != {"quality": 999}


@pytest.mark.parametrize("field", ["seed", "trial", "attempt", "path", "traversal"])
def test_live_run_observation_requires_exact_identity_and_owner_path(
    tmp_path: Path, field: str
) -> None:
    from lambdaforge.controlplane.Fleet import FleetMember

    root, wave = allocation(tmp_path)
    shard = StudyShard.from_mapping(wave["shard"])
    profile = ClusterProfile("A", environment="managed", workspace=str(tmp_path / "provider"))
    plane = Mock()
    plane.catalog.get.return_value = profile
    plane.jobs.get.return_value.state = JobState.RUNNING
    executor = PreparedShardExecutor(
        profile,
        FleetMember.from_mapping(wave["shard"]["member"]),
        root=root,
        resources=ResourceRequest(cpu_cores=4),
        equivalence=ExecutionEquivalence(**wave["equivalence"]),
        invocation=lambda key: wave["invocations"][key],
        control_plane=plane,
        allocation_id="a" * 64,
    )
    executor._member_receipt = Mock(return_value={"root": str(root)})  # type: ignore[method-assign]
    lease = shard.leases[0]
    native = root / "waves" / shard.shard_id / "worker" / "execution" / "attempt-0001"
    run = {
        "trial": int(lease.run.candidate),
        "seed": lease.run.seed,
        "phase": lease.run.phase,
        "attempt_id": "attempt-0001",
        "state": "running",
        "run_dir": str(native),
        "log_path": str(native / "work.log"),
        "metrics_path": str(native / "metrics.jsonl"),
        "training_metrics_path": str(native / "training-metrics.jsonl"),
    }
    if field == "seed":
        run["seed"] = 999
    elif field == "trial":
        run["trial"] = 999
    elif field == "attempt":
        run["attempt_id"] = "attempt-0002"
    elif field == "path":
        run["run_dir"] = "/unrelated/work"
    else:
        run["log_path"] = str(native / ".." / ".." / "unrelated.log")
    executor._member_call = Mock(
        return_value={  # type: ignore[method-assign]
            "accepted": True,
            "observation": {"shard_id": shard.shard_id, "runs": {lease.run.key: run}},
        }
    )
    with pytest.raises(ValueError, match="Live Run metadata"):
        executor.observe(shard, executor.allocation_job_id)


@pytest.mark.parametrize("invalid", ["stale", "foreign", "equivalence", "unknown", "terminal"])
def test_only_live_exact_fresh_allocation_can_offer_capacity(
    tmp_path: Path,
    invalid: str,
) -> None:
    root, wave = allocation(tmp_path)
    from lambdaforge.controlplane.Fleet import FleetMember

    profile = ClusterProfile("A", environment="managed", workspace=str(tmp_path / "provider"))
    plane = Mock()
    plane.catalog.get.return_value = profile
    plane.jobs.get.return_value.state = JobState.RUNNING
    executor = PreparedShardExecutor(
        profile,
        FleetMember.from_mapping(wave["shard"]["member"]),
        root=root,
        resources=ResourceRequest(cpu_cores=4),
        equivalence=ExecutionEquivalence(**wave["equivalence"]),
        invocation=lambda key: wave["invocations"][key],
        control_plane=plane,
        allocation_id="a" * 64,
    )
    now = time.time()
    value: dict[str, Any] = {
        "offer_version": 1,
        "job_id": wave["job_id"],
        "observed_at": now - 1,
        "valid_until": now + 10,
        "equivalence": wave["equivalence"],
        "slots": 4,
        "admissible_gpus": 0,
        "free_memory_bytes": None,
    }
    executor._member_call = Mock(return_value={"offer": value})  # type: ignore[method-assign]
    assert executor.offer(()).slots == 4
    if invalid == "stale":
        value["valid_until"] = now - 1
    elif invalid == "foreign":
        value["job_id"] = "foreign"
    elif invalid == "equivalence":
        value["equivalence"] = replace(executor.equivalence, hardware="other").to_dict()
    elif invalid == "unknown":
        plane.jobs.get.return_value.state = JobState.UNKNOWN
    else:
        plane.jobs.get.return_value.state = JobState.SUCCEEDED
    assert executor.offer(()).slots == 0
