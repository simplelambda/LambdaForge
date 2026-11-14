"""CPU-starting Runs must not inherit device reserve/foreign VRAM; no actual GPU required."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import psutil
import pytest

from lambdaforge.hpo.AdaptiveResources import ActiveResourceCommitment, ResourceDemandModel
from lambdaforge.runtime.GPUObservation import GPUObservationCache, collect_nvml, collect_smi
from lambdaforge.work import runner


@pytest.fixture
def observation(monkeypatch):
    pid = os.getpid()
    identity = {"pid": pid, "create_time": psutil.Process(pid).create_time(), "phase": "startup"}
    row = {
        "uuid": "GPU-granted",
        "quality": "measured",
        "processes": [],
        "compute_process_count": 0,
        "process_inventory_complete": True,
    }
    value = {"quality": "measured", "observed_at": time.time(), "devices": [row]}
    monkeypatch.setattr(GPUObservationCache, "__init__", lambda self: None)
    monkeypatch.setattr(GPUObservationCache, "read", lambda self: value)
    monkeypatch.setattr(runner, "_descendant_pids", lambda p: {p})
    return identity, value, row


def test_complete_empty_inventory_is_exact_zero_not_missing(observation):
    identity, _value, _row = observation
    assert runner._gpu_process_memory((), {"run": identity}) == ({"run": 0}, "nvml-process-exact")


@pytest.mark.parametrize(
    "kind",
    ["legacy", "partial", "unavailable", "reused", "allocator", "before-birth", "missing-time"],
)
def test_incomplete_unknown_or_conflicting_evidence_never_invents_zero(observation, kind):
    identity, value, row = observation
    if kind == "legacy":
        row.pop("process_inventory_complete")
    elif kind == "partial":
        row["process_inventory_complete"] = False
    elif kind == "unavailable":
        value["quality"] = "unavailable"
    elif kind == "reused":
        identity["create_time"] -= 1
    elif kind == "before-birth":
        value["observed_at"] = identity["create_time"] - 1
    elif kind == "missing-time":
        value.pop("observed_at")
    else:
        identity["cuda_reserved_bytes"] = 100
    assert runner._gpu_process_memory((), {"run": identity}) == ({}, "unavailable")


def test_foreign_device_memory_is_not_cpu_worker_memory(observation):
    identity, _value, row = observation
    row["compute_process_count"] = 1
    row["processes"] = [{"pid": 987654321, "create_time": 1, "used_bytes": 70 * 1024**3}]
    assert runner._gpu_process_memory((), {"run": identity}) == ({"run": 0}, "nvml-process-exact")


@pytest.mark.parametrize("used", [None, "N/A"])
def test_unknown_owned_usage_is_not_zero_even_if_another_device_has_bytes(observation, used):
    identity, value, row = observation
    process = {"pid": identity["pid"], "create_time": identity["create_time"], "used_bytes": used}
    row["processes"] = [process]
    row["compute_process_count"] = 1
    value["devices"].append({**row, "uuid": "other", "processes": [{**process, "used_bytes": 20}]})
    assert runner._gpu_process_memory((), {"run": identity}) == ({}, "unavailable")


def test_actual_live_update_preserves_reserved_headroom_but_not_as_cpu_run_usage(
    observation, monkeypatch
):
    identity, _value, _row = observation
    run = object()
    spec = {
        "trial_parameters": {},
        "definition": {"resources": {"gpu_memory_bytes": 0}},
        "hpo_dispatched_monotonic": 0.0,
    }
    monkeypatch.setattr(runner, "_read_live_resource_snapshot", lambda _: identity)
    commitments = {run: ActiveResourceCommitment("candidate", 50 * 1024**3)}
    evidence = runner._update_active_resource_commitments(
        ((139 * 1024**3, 140 * 1024**3),),
        pending={run: (spec, 0, object())},
        commitments=commitments,
        trajectories={},
        metadata={
            run: {"candidate_key": "candidate", "compatibility_key": "work", "hardware": "H200"}
        },
        model=ResourceDemandModel(),
        now=60.0,
    )
    assert commitments[run].current_bytes == 0
    assert evidence[0].current_bytes == 0
    assert evidence[0].measurement_provenance == "nvml-process-exact"
    assert evidence[0].trajectory[-1].device_free_bytes == 139 * 1024**3
    assert evidence[0].trajectory[-1].external_bytes == 1024**3
    # No learning that CPU initialization has validated a low CUDA training envelope.
    assert commitments[run].resource_state not in {"RESOURCE_STABLE", "PROVISIONALLY_STABLE"}


def test_collectors_distinguish_empty_process_inventory_from_query_failure(monkeypatch):
    class NVML:
        nvmlInit = staticmethod(lambda: None)
        nvmlShutdown = staticmethod(lambda: None)
        nvmlDeviceGetCount = staticmethod(lambda: 1)
        nvmlDeviceGetHandleByIndex = staticmethod(lambda _: 0)
        nvmlDeviceGetUUID = staticmethod(lambda _: "GPU-granted")
        nvmlDeviceGetName = staticmethod(lambda _: "test")
        nvmlDeviceGetMemoryInfo = staticmethod(lambda _: SimpleNamespace(free=139, total=140))
        nvmlDeviceGetComputeRunningProcesses = staticmethod(lambda _: [])

    assert collect_nvml(NVML())[0]["process_inventory_complete"] is True
    monkeypatch.setattr(
        NVML,
        "nvmlDeviceGetComputeRunningProcesses",
        staticmethod(lambda _: (_ for _ in ()).throw(OSError("denied"))),
    )
    assert collect_nvml(NVML())[0]["process_inventory_complete"] is False
    monkeypatch.setattr(
        NVML,
        "nvmlDeviceGetComputeRunningProcesses",
        staticmethod(lambda _: [SimpleNamespace(pid=os.getpid())]),
    )
    assert collect_nvml(NVML())[0]["process_inventory_complete"] is False
    monkeypatch.setattr(
        "lambdaforge.runtime.GPUObservation._smi_query",
        lambda q: "" if q.startswith("--query-compute") else "0,GPU-granted,H200,139,140,0\n",
    )
    assert collect_smi()[0]["process_inventory_complete"] is True


def test_smi_failed_lifetime_inspection_is_not_a_complete_inventory(monkeypatch):
    monkeypatch.setattr(
        "lambdaforge.runtime.GPUObservation._smi_query",
        lambda q: (
            "GPU-granted,987654321,10\n"
            if q.startswith("--query-compute")
            else "0,GPU-granted,H200,139,140,0\n"
        ),
    )
    assert collect_smi()[0]["process_inventory_complete"] is False
