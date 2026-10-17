"""Synthetic physical/cgroup and lifetime evidence, without modifying real cgroups."""

import json
import os
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor
from lambdaforge.runtime.HostObservation import observe_host, pending_ram_bytes, process_tree_memory


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(total=1000, available=800))
    monkeypatch.setattr("lambdaforge.runtime.HostObservation._cgroup_directory", lambda: None)
    return observe_host


def write_group(path, maximum, current):
    path.mkdir(parents=True, exist_ok=True)
    (path / "memory.max").write_text(str(maximum))
    (path / "memory.current").write_text(str(current))


def test_cgroup_bounds_physical_capacity_and_observes_pressure(tmp_path, host):
    write_group(tmp_path, 600, 400)
    child = tmp_path / "job"
    write_group(child, "max", 100)
    (child / "memory.pressure").write_text("some avg10=0.10 avg60=0.01 avg300=0.0 total=50\n")
    (child / "memory.events").write_text("high 3\noom 1\noom_kill 0\n")
    value = host(cgroup=(child, tmp_path))
    assert value["effective_total_bytes"] == 600
    assert value["effective_available_bytes"] == 200
    assert value["cgroup"]["pressure"]["some"]["avg10"] == 0.1
    assert value["cgroup"]["events"]["oom"] == 1


def test_missing_cgroup_is_not_fictitious_enforcement(host):
    value = host()
    assert value["effective_available_bytes"] == 800
    assert value["cgroup"]["available"] is False


def test_cpu_quota_and_cpuset_never_broaden_inherited_affinity(tmp_path, host, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {1, 3, 5, 7})
    write_group(tmp_path, "max", 10)
    (tmp_path / "cpu.max").write_text("150000 100000")
    (tmp_path / "cpuset.cpus.effective").write_text("0-3,6-7")
    value = host(cgroup=(tmp_path, tmp_path))
    assert value["cpu_affinity"] == [1, 3, 7]
    assert value["effective_cpu_cores"] == 2
    assert value["cgroup"]["cpu_quota_cores"] == 1.5


def test_corrupt_cgroup_current_never_becomes_zero(tmp_path, host):
    write_group(tmp_path, 600, "broken")
    value = host(cgroup=(tmp_path, tmp_path))
    assert value["quality"] == "unavailable"
    assert value["effective_available_bytes"] is None


def test_pss_reconciles_physical_free_without_rss_double_count():
    assert pending_ram_bytes(600, {"pss_bytes": 400, "rss_bytes": 900}) == 200
    assert pending_ram_bytes(600, {"pss_bytes": None, "rss_bytes": 900}) == 600
    assert pending_ram_bytes(600, {"pss_bytes": 700}) == 0


def test_reused_pid_cannot_attribute_memory():
    import psutil

    value = process_tree_memory(os.getpid(), psutil.Process().create_time() - 1)
    assert value["quality"] == "unavailable"
    assert value["rss_bytes"] is None


def test_capacity_uses_pending_not_resident_ram_and_retains_exec_owner(tmp_path, monkeypatch, host):
    from lambdaforge.runtime.HostCoordination import boot_identity, host_identity

    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1, 2})
    owner = ProcessIdentity.create(os.getpid(), os.getpgrp(), ("old-argv-before-exec",), "job-old")
    (tmp_path / "job-old.json").write_text(
        json.dumps(
            {
                "supervisor_identity": owner.to_dict(),
                "cpus": [0],
                "ram_bytes": 600,
                "host": host_identity(),
                "boot": boot_identity(),
            }
        )
    )
    monkeypatch.setattr(
        "lambdaforge.runtime.HostObservation.process_tree_memory", lambda *args: {"pss_bytes": 500}
    )
    current = ProcessIdentity.create(os.getpid(), os.getpgrp(), (), "job-new")
    # 800 physically available minus 100 not-yet-resident, rather than minus 600 again.
    assert ProcessSupervisor._acquire_capacity(tmp_path, current, 1, 650) == (1,)
    assert (tmp_path / "job-old.json").exists()  # argv mismatch is not a dead owner.


def test_unresolved_owner_protects_cpu_and_future_ram(tmp_path, monkeypatch, host):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1})
    owner = ProcessIdentity.create(os.getpid(), os.getpgrp(), (), "job-old")
    (tmp_path / "job-old.json").write_text(
        json.dumps(
            {
                "supervisor_identity": owner.to_dict(),
                "cpus": [0],
                "ram_bytes": 600,
            }
        )
    )
    monkeypatch.setattr(ProcessIdentity, "storage_status", lambda self: "unresolved")
    monkeypatch.setattr(
        "lambdaforge.runtime.HostObservation.process_tree_memory", lambda *args: {"pss_bytes": None}
    )
    current = ProcessIdentity.create(os.getpid(), os.getpgrp(), (), "job-new")
    assert ProcessSupervisor._acquire_capacity(tmp_path, current, 1, 300) is None
    assert (tmp_path / "job-old.json").exists()


def test_zero_authored_ram_is_labelled_unknown(tmp_path, monkeypatch, host):
    from lambdaforge.runtime.HostLeaseStore import lease_directory

    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0})
    owner = ProcessIdentity.create(os.getpid(), os.getpgrp(), (), "job-new")
    assert ProcessSupervisor._acquire_capacity(tmp_path, owner, 1, 0) == (0,)
    assert (
        json.loads((lease_directory(tmp_path) / "job-new.json").read_text())["ram_demand_known"]
        is False
    )


def test_impossible_cgroup_request_fails_before_launch(tmp_path, monkeypatch, host):
    monkeypatch.setattr(
        "lambdaforge.runtime.HostObservation.observe_host",
        lambda: {
            "effective_total_bytes": 100,
            "effective_available_bytes": 80,
            "effective_cpu_cores": 2,
        },
    )
    owner = ProcessIdentity.create(os.getpid(), os.getpgrp(), (), "job-new")
    with pytest.raises(RuntimeError, match="physical/cgroup"):
        ProcessSupervisor._acquire_capacity(tmp_path, owner, 1, 200)


def test_corrupt_lease_is_not_deleted(tmp_path, host):
    path = tmp_path / "job-missing.json"
    path.write_text("{}")
    owner = ProcessIdentity.create(os.getpid(), os.getpgrp(), (), "job-new")
    with pytest.raises(ValueError, match="ownership"):
        ProcessSupervisor._acquire_capacity(tmp_path, owner, 1, 0)
    assert path.read_text() == "{}"


def test_new_gpu_run_checks_aggregate_study_ram_before_creating_worker(monkeypatch, host):
    from lambdaforge.execution import ResourceRequest
    from lambdaforge.hpo.AdaptiveResources import ResourceDemandModel
    from lambdaforge.work import runner

    monkeypatch.delenv("LAMBDAFORGE_GPU_ADMISSION_ROOT", raising=False)
    monkeypatch.setattr(
        "lambdaforge.runtime.HostObservation.process_tree_memory",
        lambda *args: {"pss_bytes": 75, "rss_bytes": 150},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("RAM-blocked dispatch must not create a GPU worker")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", forbidden)
    value = {"definition": {"resources": {"ram_bytes": 33}}}
    assert (
        runner._launch_gpu_run(
            value,
            token="GPU-one",
            slot=0,
            resources=ResourceRequest(gpu_count=1, ram_bytes=100),
            action=None,
            device=None,
            model=ResourceDemandModel(),
        )
        is None
    )
    assert value["resource_launch_block"] == "study_ram_allocation_headroom_or_unknown"


def test_native_first_run_reconciles_helper_ram_instead_of_stalling(monkeypatch, host):
    from lambdaforge.execution import ResourceRequest
    from lambdaforge.hpo.AdaptiveResources import ResourceDemandModel
    from lambdaforge.hpo.HostResources import HostResourceLease
    from lambdaforge.work import runner

    monkeypatch.delenv("LAMBDAFORGE_GPU_ADMISSION_ROOT", raising=False)
    monkeypatch.setattr(
        "lambdaforge.runtime.HostObservation.process_tree_memory",
        lambda *args: {"pss_bytes": 24, "rss_bytes": 45},
    )
    future = object()

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, *args):
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", Pool)
    resources = ResourceRequest(cpu_cores=4, gpu_count=1, ram_bytes=100)
    lease = HostResourceLease.allocate(
        resources,
        active_runs=1,
        hard_concurrency_ceiling=4,
        controller_memory_bytes=20,
    )
    value = {
        "definition": {"resources": lease.resources(resources).to_dict()},
        "host_resource_lease": lease.to_dict(),
    }
    result = runner._launch_gpu_run(
        value,
        token="GPU-one",
        slot=0,
        resources=resources,
        action=None,
        device=None,
        model=ResourceDemandModel(),
    )
    assert result is not None and result[0] is future
    assert value["definition"]["resources"]["ram_bytes"] == 76
    assert value["host_resource_lease"]["observed_resident_memory_bytes"] == 24
    # Reconciliation is not permission to exceed a physically spent aggregate allocation.
    monkeypatch.setattr(
        "lambdaforge.runtime.HostObservation.process_tree_memory",
        lambda *args: {"pss_bytes": 100, "rss_bytes": 150},
    )
    assert (
        runner._launch_gpu_run(
            value,
            token="GPU-one",
            slot=0,
            resources=resources,
            action=None,
            device=None,
            model=ResourceDemandModel(),
        )
        is None
    )
    assert value["resource_launch_block"] == "study_ram_allocation_headroom_or_unknown"


def test_cpu_only_heartbeat_does_not_start_any_gpu_observer(tmp_path, monkeypatch, host):
    import psutil

    state = {"state": "running", "requested_resources": {"gpu_count": 0}}
    (tmp_path / "state.json").write_text(json.dumps(state))

    def forbidden(*args, **kwargs):
        raise AssertionError("A CPU-only Job must not query the GPU driver")

    monkeypatch.setattr(
        "lambdaforge.runtime.GPUObservation.GPUObservationCache.__init__", forbidden
    )
    identity = ProcessIdentity.create(
        os.getpid(), os.getpgrp(), psutil.Process().cmdline(), "job-cpu"
    )
    ProcessSupervisor._heartbeat(tmp_path, identity, "2026-10-13T10:00:00Z")
    result = json.loads((tmp_path / "state.json").read_text())
    assert result["observed_usage"]["gpu_observation"]["quality"] == "not-requested"
    assert result["observed_usage"]["process_count"] > 0


def test_reused_heartbeat_root_cannot_attribute_processes_or_probe_gpu(tmp_path, monkeypatch):
    from lambdaforge.runtime.GPUObservation import GPUObservationCache

    identity = ProcessIdentity.create(os.getpid(), os.getpgrp(), (), "job-reused")
    identity = ProcessIdentity(
        identity.pid,
        identity.process_group,
        identity.create_time - 10,
        identity.command_sha256,
        identity.job_id,
    )
    monkeypatch.setattr(GPUObservationCache, "read", lambda self, **kw: pytest.fail("queried GPU"))
    ProcessSupervisor._heartbeat(tmp_path, identity, "2026-10-13T10:00:00Z")
    usage = json.loads((tmp_path / "state.json").read_text())["observed_usage"]
    assert usage["process_observation"] == "unavailable"
    assert "rss_bytes" not in usage and "gpu_memory_bytes" not in usage


def test_heartbeat_rejects_cached_gpu_process_with_different_birth(tmp_path, monkeypatch, host):
    from lambdaforge.runtime.GPUObservation import GPUObservationCache

    identity = ProcessIdentity.create(os.getpid(), os.getpgrp(), (), "job-current")
    row = {
        "uuid": "GPU-a",
        "quality": "measured",
        "compute_process_count": 1,
        "processes": [
            {"pid": identity.pid, "create_time": identity.create_time - 10, "used_bytes": 999999}
        ],
    }
    monkeypatch.setattr(
        GPUObservationCache, "read", lambda self, **kw: {"quality": "measured", "devices": [row]}
    )
    ProcessSupervisor._heartbeat(tmp_path, identity, "2026-10-13T10:00:00Z")
    usage = json.loads((tmp_path / "state.json").read_text())["observed_usage"]
    assert "gpu_memory_bytes" not in usage  # A stale lifetime is unknown, not false zero.
    assert usage["gpu_observation"]["process_attribution"] == "unavailable"
