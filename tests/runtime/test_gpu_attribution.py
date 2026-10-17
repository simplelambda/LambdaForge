"""MIG parent/instance rows and reused lifetimes must not fabricate owned memory."""

import os

import psutil
import pytest

from lambdaforge.runtime.GPUObservation import GPUObservationCache
from lambdaforge.work import runner


def test_nvml_parent_and_mig_instance_are_not_double_counted_for_one_worker(monkeypatch):
    pid, birth = os.getpid(), psutil.Process().create_time()
    process = {"pid": pid, "create_time": birth, "used_bytes": 20}
    sample = {
        "quality": "measured",
        "devices": [
            {"uuid": "GPU-one", "parent_uuid": None, "processes": [process]},
            {"uuid": "MIG-one", "parent_uuid": "GPU-one", "processes": [process]},
        ],
    }
    monkeypatch.setattr(GPUObservationCache, "__init__", lambda self: None)
    monkeypatch.setattr(GPUObservationCache, "read", lambda self: sample)
    monkeypatch.setattr(runner, "_descendant_pids", lambda pid: {pid})
    values, quality = runner._gpu_process_memory((), {"run": {"pid": pid, "create_time": birth}})
    assert values == {"run": 20} and quality == "nvml-process-exact"
    values, quality = runner._gpu_process_memory(
        (), {"run": {"pid": pid, "create_time": birth - 1}}
    )
    assert not values and quality == "unavailable"


def test_resource_charts_do_not_sum_mig_capacity_or_turn_missing_memory_into_zero():
    pytest.importorskip("textual")
    from lambdaforge.tui.widgets import ResourceDashboard

    host = {
        "gpus": [
            {"parent_uuid": None, "memory_total_bytes": 80, "memory_used_bytes": 40},
            {"parent_uuid": "GPU-one", "memory_total_bytes": 20, "memory_used_bytes": 5},
        ]
    }
    assert ResourceDashboard._metric_values(host, {})["gpu"][0] == 50
    host["gpus"][0]["memory_used_bytes"] = None
    assert ResourceDashboard._metric_values(host, {})["gpu"][0] is None


def test_legacy_chart_handles_missing_utilization_without_exception():
    from lambdaforge.cli.LiveJobMonitor import _gpu, _gpu_memory

    value = {
        "gpus": [{"utilization_percent": None, "memory_used_bytes": None, "memory_total_bytes": 80}]
    }
    assert _gpu(value) is None and _gpu_memory(value) is None
