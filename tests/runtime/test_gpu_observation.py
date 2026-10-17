"""GPU observation contracts are verified without loading a GPU driver or consumer model."""

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from lambdaforge.runtime.GPUDeviceIdentity import visible_devices
from lambdaforge.runtime.GPUObservation import (
    GPUObservationCache,
    GPUObservationPolicy,
    collect_nvml,
)


def row(uuid="GPU-one", free=60, total=100):
    return {
        "uuid": uuid,
        "index": 0,
        "parent_uuid": None,
        "name": "test",
        "free_bytes": free,
        "total_bytes": total,
        "quality": "measured",
    }


def test_shared_single_flight_and_distinct_admission_freshness(tmp_path):
    now = [100.0]
    calls = []

    def probe():
        calls.append(1)
        return {"quality": "measured", "observed_at": now[0], "devices": [row()]}

    cache = GPUObservationCache(tmp_path / "cache", clock=lambda: now[0], probe=probe)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: cache.read(), range(20)))
    assert len(calls) == 1
    assert values[-1]["devices"][0]["free_bytes"] == 60
    now[0] += 1
    assert cache.read()["statistics"]["real_queries"] == 1
    assert cache.read(admission=True)["statistics"]["real_queries"] == 2


def test_failure_backoff_never_blesses_stale_or_zero_memory(tmp_path):
    now = [100.0]
    calls = []

    def probe():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("driver unavailable")
        return {"quality": "measured", "observed_at": now[0], "devices": [row()]}

    cache = GPUObservationCache(tmp_path / "cache", clock=lambda: now[0], probe=probe)
    assert cache.read()["quality"] == "measured"
    now[0] += 6
    failed = cache.read(admission=True)
    assert failed["quality"] == "unavailable" and failed["devices"][0]["free_bytes"] == 60
    for _ in range(5):
        assert cache.read(admission=True)["quality"] == "unavailable"
    assert len(calls) == 2
    now[0] += 3
    cache.read()
    assert len(calls) == 3


@pytest.mark.parametrize("free,total", [(101, 100), (-1, 100), (0, 0), (True, 100), (None, 100)])
def test_incoherent_memory_is_not_admission_evidence(tmp_path, free, total):
    cache = GPUObservationCache(
        tmp_path / "cache",
        probe=lambda: {
            "devices": [row(free=free, total=total)],
            "quality": "measured",
            "observed_at": 0,
        },
    )
    assert cache.read(admission=True)["quality"] == "unavailable"


@pytest.mark.parametrize("visibility", ["", "-1", "none", "void"])
def test_disabled_grant_never_reads_a_physical_gpu(monkeypatch, visibility):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    with pytest.raises(RuntimeError, match="empty/disabled"):
        visible_devices([row()], 1)


def test_uuid_and_mig_grants_never_expand_or_use_parent_capacity(monkeypatch):
    devices = [
        row(),
        row("GPU-two"),
        {
            **row("MIG-instance", free=20, total=30),
            "parent_uuid": "GPU-one",
            "aliases": ["MIG-GPU-one/1/2"],
        },
    ]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-GPU-one/1/2,GPU-two")
    selected = visible_devices(devices, 2)
    assert [r["uuid"] for r in selected] == ["MIG-instance", "GPU-two"]
    assert selected[0]["total_bytes"] == 30
    with pytest.raises(RuntimeError, match="broaden"):
        visible_devices(devices, 3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-")
    with pytest.raises(RuntimeError, match="ambiguous"):
        visible_devices(devices, 1)


def test_numeric_cuda_mapping_is_pinned_not_guessed_from_nvml_index(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("LAMBDAFORGE_HOST_COORDINATION_ROOT", str(tmp_path))
    calls = []

    def identities(self):
        calls.append(1)
        return ["GPU-two", "GPU-one"]

    monkeypatch.setattr(GPUObservationCache, "visible_identity", identities)
    devices = [row(), {**row("GPU-two"), "index": 1}]
    assert [r["uuid"] for r in visible_devices(devices, 2)] == ["GPU-two", "GPU-one"]
    visible_devices(devices, 2)
    assert calls == [1]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,0"


def test_nvml_reports_missing_memory_without_zero_and_preserves_healthy_sibling():
    class NVML:
        nvmlInit = staticmethod(lambda: None)
        nvmlShutdown = staticmethod(lambda: None)
        nvmlDeviceGetCount = staticmethod(lambda: 2)
        nvmlDeviceGetHandleByIndex = staticmethod(lambda i: i)
        nvmlDeviceGetUUID = staticmethod(lambda i: f"GPU-{i}")
        nvmlDeviceGetName = staticmethod(lambda i: "fake")

        def nvmlDeviceGetMemoryInfo(self, i):
            if i == 0:
                raise RuntimeError("lost device")
            return SimpleNamespace(free=40, total=80)

    observed = collect_nvml(NVML())
    assert observed[0]["free_bytes"] is None and observed[0]["quality"] == "unavailable"
    assert observed[1]["free_bytes"] == 40 and observed[1]["quality"] == "measured"


def test_unsafe_cache_root_or_symlink_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        GPUObservationCache(link)
    if os.name == "posix":
        public = tmp_path / "public"
        public.mkdir(mode=0o755)
        with pytest.raises(ValueError, match="private"):
            GPUObservationCache(public)


def test_standard_measurement_source_never_imports_torch_or_cuda_runtime():
    import inspect

    from lambdaforge.runtime.GPUObservation import collect
    from lambdaforge.work.runner import _gpu_memory_inventory

    assert "import torch" not in inspect.getsource(collect)
    assert "mem_get_info" not in inspect.getsource(_gpu_memory_inventory)


def test_memory_inventory_uses_fresh_native_observation(monkeypatch):
    from lambdaforge.work.runner import _gpu_memory_inventory

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one,GPU-two")
    calls = []

    def read(self, *, admission=False):
        calls.append(admission)
        return {"quality": "measured", "devices": [row(), row("GPU-two", 30, 100)]}

    monkeypatch.setattr(GPUObservationCache, "__init__", lambda self: None)
    monkeypatch.setattr(GPUObservationCache, "read", read)
    assert _gpu_memory_inventory(2) == ((60, 100), (30, 100))
    assert calls == [True]


@pytest.mark.parametrize("kwargs", [{"timeout": 0}, {"admission_ttl": 0.1}, {"telemetry_ttl": 0.2}])
def test_bad_observation_policies_rejected(kwargs):
    with pytest.raises(ValueError):
        GPUObservationPolicy(**kwargs)


def test_blocked_helper_has_real_timeout_reaping_and_backoff(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time

    original = subprocess.Popen
    children = []

    def stalled(command, **kwargs):
        child = original([sys.executable, "-c", "import time;time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr("lambdaforge.runtime.GPUObservation.subprocess.Popen", stalled)
    cache = GPUObservationCache(tmp_path / "cache", policy=GPUObservationPolicy(timeout=0.1))
    started = time.monotonic()
    result = cache.read(admission=True)
    assert time.monotonic() - started < 2
    assert result["quality"] == "unavailable" and result["statistics"]["timeouts"] == 1
    assert children[0].poll() is not None
    cache.read(admission=True)
    assert len(children) == 1


@pytest.mark.parametrize(
    "output,expected",
    [
        ('export CUDA_VISIBLE_DEVICES="0,2"\n', ("0", "2")),
        ("CUDA_VISIBLE_DEVICES='MIG-GPU-one/1/2,GPU-two'\n", ("MIG-GPU-one/1/2", "GPU-two")),
        ('export CUDA_VISIBLE_DEVICES=""', ()),
        ("0,2\n", ("0", "2")),
        ("Reservation unavailable", None),
        ("", None),
        ('export CUDA_VISIBLE_DEVICES="$(touch bad)"', None),
        ("CUDA_VISIBLE_DEVICES=0\nCUDA_VISIBLE_DEVICES=1", None),
    ],
)
def test_authoritative_visibility_output_is_parsed_not_executed(output, expected):
    from lambdaforge.runtime.GPUGrant import parse_visibility

    assert parse_visibility(output) == expected


def test_partial_device_loss_preserves_other_admission_without_remapping(monkeypatch):
    import importlib

    runner = importlib.import_module("lambdaforge.work.runner")
    monkeypatch.setattr(runner, "_GPU_BOUND_IDENTITIES", {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one,GPU-two")
    snapshot = {"quality": "measured", "devices": [row(), row("GPU-two", 30, 100)]}
    monkeypatch.setattr(GPUObservationCache, "__init__", lambda self: None)
    monkeypatch.setattr(GPUObservationCache, "read", lambda self, **kw: snapshot)
    original = runner._gpu_memory_inventory(2)
    snapshot["devices"] = [row("GPU-two", 25, 100), row("GPU-new-ungranted", 100, 100)]
    memory, slots = runner._refresh_gpu_memory_inventory(2, original)
    assert slots == {1}
    assert memory == ((60, 100), (25, 100))  # Slot 0 is last-known, NOT admission evidence.
    assert runner._gpu_observation_slot(snapshot, 2, 1)["uuid"] == "GPU-two"
    with pytest.raises(RuntimeError, match="disappeared"):
        runner._gpu_observation_slot(snapshot, 2, 0)


def test_grant_comparison_uses_pinned_uuid_not_nvml_index():
    from lambdaforge.runtime.GPUGrant import intersect_visibility

    assert intersect_visibility(
        ["0", "1"], ["GPU-two"], uuid_by_token={"0": "GPU-one", "1": "GPU-two"}
    ) == frozenset({"1"})
    assert intersect_visibility(["0", "1"], ["GPU-two"]) is None
    assert intersect_visibility(["GPU-one"], ["0"]) is None
    assert intersect_visibility(["GPU-one", "GPU-other"], ["GPU-o"]) is None
    assert intersect_visibility(["GPU-one"], ["GPU-one-ungranted-suffix"]) == frozenset()
    assert intersect_visibility(
        ["MIG-instance"], ["MIG-GPU-one/1/2"], aliases={"MIG-instance": ["MIG-GPU-one/1/2"]}
    ) == frozenset({"MIG-instance"})


def test_observer_import_is_light_and_nvml_optional_in_a_fresh_process():
    import subprocess
    import sys

    code = (
        "import sys;import lambdaforge.runtime.GPUObservation;"
        "import lambdaforge.runtime.GPUDeviceIdentity;"
        "assert 'torch' not in sys.modules;assert 'pynvml' not in sys.modules"
    )
    assert subprocess.run([sys.executable, "-c", code], timeout=10, check=False).returncode == 0


def test_smi_fallback_batches_unknown_processes_without_inventing_empty_devices(monkeypatch):
    from lambdaforge.runtime.GPUObservation import collect_smi

    calls = []

    def query(fields):
        calls.append(fields)
        if fields.startswith("--query-compute"):
            raise OSError("process observation unavailable")
        return "0,GPU-one,Fake,40,80,0\n"

    monkeypatch.setattr("lambdaforge.runtime.GPUObservation._smi_query", query)
    value = collect_smi()[0]
    assert value["free_bytes"] == 40 * 1024**2
    assert value["compute_process_count"] is None
    assert len(calls) == 2


def test_smi_fallback_knows_empty_process_inventory(monkeypatch):
    from lambdaforge.runtime.GPUObservation import collect_smi

    monkeypatch.setattr(
        "lambdaforge.runtime.GPUObservation._smi_query",
        lambda fields: "" if fields.startswith("--query-compute") else "0,GPU-one,Fake,40,80,0\n",
    )
    assert collect_smi()[0]["compute_process_count"] == 0
