"""Host and stable-device lease isolation without GPUs, providers or persisted Studies."""

import json
import os

import pytest

from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor
from lambdaforge.runtime.HostCoordination import boot_identity, host_identity
from lambdaforge.runtime.HostLeaseStore import gpu_lease_name, lease_directory, read_lease


def owner(name):
    return ProcessIdentity.create(os.getpid(), os.getpgrp(), (), name)


def device(uuid):
    return {
        "uuid": uuid,
        "quality": "measured",
        "compute_process_count": 0,
        "total_bytes": 100,
        "free_bytes": 100,
        "utilization_percent": 0,
    }


def test_remapped_ordinals_coordinate_same_uuid_and_release_without_observation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ProcessSupervisor, "_direct_gpu_devices", lambda: {0: device("GPU-a")})
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-a"), 1) == (0,)
    monkeypatch.setattr(ProcessSupervisor, "_direct_gpu_devices", lambda: {7: device("GPU-a")})
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-b"), 1) is None
    path = lease_directory(tmp_path) / (gpu_lease_name("GPU-a") + ".json")
    assert read_lease(path)["device_uuid"] == "GPU-a"
    monkeypatch.setattr(
        ProcessSupervisor, "_direct_gpu_devices", lambda: pytest.fail("release re-probed")
    )
    assert ProcessSupervisor._leased_gpu_visibility(tmp_path, "job-a", (0,)) == "GPU-a"
    ProcessSupervisor._release_gpus(tmp_path, "job-a", (0,))
    assert not path.exists()


def test_same_ordinal_different_uuid_and_mig_instances_are_independent(tmp_path, monkeypatch):
    for index, uuid in enumerate(("GPU-a", "GPU-b", "MIG-c", "MIG-d")):
        monkeypatch.setattr(
            ProcessSupervisor, "_direct_gpu_devices", lambda uuid=uuid: {0: device(uuid)}
        )
        assert ProcessSupervisor._acquire_gpus(tmp_path, owner(f"job-{index}"), 1) == (0,)
    assert len(list(lease_directory(tmp_path).glob("gpu-*.json"))) == 4


def test_parent_exclusive_lease_protects_mig_but_sibling_instances_do_not_conflict(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ProcessSupervisor,
        "_direct_gpu_devices",
        lambda: {0: device("MIG-a") | {"parent_uuid": "GPU-parent"}},
    )
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-a"), 1) == (0,)
    monkeypatch.setattr(
        ProcessSupervisor,
        "_direct_gpu_devices",
        lambda: {0: device("MIG-b") | {"parent_uuid": "GPU-parent"}},
    )
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-b"), 1) == (0,)
    monkeypatch.setattr(ProcessSupervisor, "_direct_gpu_devices", lambda: {0: device("GPU-parent")})
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-parent"), 1) is None
    ProcessSupervisor._release_gpus(tmp_path, "job-a", (0,))
    ProcessSupervisor._release_gpus(tmp_path, "job-b", (0,))
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-parent"), 1) == (0,)
    monkeypatch.setattr(
        ProcessSupervisor,
        "_direct_gpu_devices",
        lambda: {0: device("MIG-a") | {"parent_uuid": "GPU-parent"}},
    )
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-a"), 1) is None


def test_direct_uuid_grant_requires_no_driver_enumeration(monkeypatch):
    from lambdaforge.runtime.GPUObservation import GPUObservationCache

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-b")
    monkeypatch.setattr(GPUObservationCache, "__init__", lambda self: None)
    monkeypatch.setattr(
        GPUObservationCache,
        "read",
        lambda self, **kwargs: {
            "quality": "measured",
            "devices": [device("GPU-other"), device("MIG-b") | {"parent_uuid": "GPU-parent"}],
        },
    )
    monkeypatch.setattr(
        GPUObservationCache, "visible_identity", lambda self: pytest.fail("CUDA enumeration")
    )
    assert ProcessSupervisor._direct_gpu_devices()[0]["uuid"] == "MIG-b"


def test_shared_home_foreign_host_records_never_use_local_pid_status(tmp_path, monkeypatch):
    (tmp_path / "gpu-0.json").write_text(
        json.dumps(
            {
                "host": "other-host",
                "boot": "other-boot",
                "supervisor_identity": owner("job-other").to_dict(),
            }
        )
    )
    monkeypatch.setattr(
        ProcessIdentity, "storage_status", lambda self: pytest.fail("foreign PID lookup")
    )
    monkeypatch.setattr(ProcessSupervisor, "_direct_gpu_devices", lambda: {0: device("GPU-a")})
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-new"), 1) == (0,)
    assert (tmp_path / "gpu-0.json").exists()


def test_unscoped_legacy_ordinal_protects_all_devices_even_if_pid_looks_dead(tmp_path, monkeypatch):
    path = tmp_path / "gpu-0.json"
    path.write_text(json.dumps({"supervisor_identity": owner("job-old").to_dict()}))
    monkeypatch.setattr(
        ProcessIdentity, "storage_status", lambda self: pytest.fail("ambiguous PID lookup")
    )
    monkeypatch.setattr(ProcessSupervisor, "_direct_gpu_devices", lambda: {3: device("GPU-a")})
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-new"), 1) is None
    assert "shared-home host" in ProcessSupervisor._lease_wait_reason
    assert path.exists()


def test_attested_previous_boot_lease_is_reclaimed_without_pid_lookup(tmp_path, monkeypatch):
    scoped = lease_directory(tmp_path)
    path = scoped / (gpu_lease_name("GPU-a") + ".json")
    path.write_text(
        json.dumps(
            {
                "host": host_identity(),
                "boot": "previous-boot",
                "device_uuid": "GPU-a",
                "supervisor_identity": owner("job-old").to_dict(),
            }
        )
    )
    monkeypatch.setattr(
        ProcessIdentity, "storage_status", lambda self: pytest.fail("old boot PID lookup")
    )
    monkeypatch.setattr(ProcessSupervisor, "_direct_gpu_devices", lambda: {0: device("GPU-a")})
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-new"), 1) == (0,)
    assert read_lease(path)["boot"] == boot_identity()


def test_two_shared_home_hosts_do_not_share_cpu_or_gpu_lease_namespace(tmp_path, monkeypatch):
    import lambdaforge.runtime.HostCoordination as coordination
    import lambdaforge.runtime.HostLeaseStore as store

    monkeypatch.setattr(ProcessSupervisor, "_direct_gpu_devices", lambda: {0: device("GPU-a")})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0})
    monkeypatch.setattr(
        "lambdaforge.runtime.HostObservation.observe_host",
        lambda: {
            "effective_total_bytes": 100,
            "effective_available_bytes": 100,
            "effective_cpu_cores": 1,
            "cpu_affinity": [0],
        },
    )
    for host in ("host-a", "host-b"):
        monkeypatch.setattr(coordination, "host_identity", lambda host=host: host)
        monkeypatch.setattr(store, "host_identity", lambda host=host: host)
        assert ProcessSupervisor._acquire_gpus(tmp_path / "gpus", owner(f"job-{host}"), 1) == (0,)
        assert ProcessSupervisor._acquire_capacity(
            tmp_path / "cpus", owner(f"job-{host}"), 1, 10
        ) == (0,)


def test_release_rejects_other_lifetime_even_with_matching_job_label(tmp_path, monkeypatch):
    monkeypatch.setattr(ProcessSupervisor, "_direct_gpu_devices", lambda: {0: device("GPU-a")})
    assert ProcessSupervisor._acquire_gpus(tmp_path, owner("job-a"), 1) == (0,)
    path = lease_directory(tmp_path) / (gpu_lease_name("GPU-a") + ".json")
    value = read_lease(path)
    value["supervisor_identity"]["create_time"] -= 10
    path.write_text(json.dumps(value))
    ProcessSupervisor._release_gpus(tmp_path, "job-a", (0,))
    assert path.exists()


def test_unsafe_or_corrupt_lease_metadata_is_not_deleted(tmp_path):
    protected = tmp_path / "scientific-evidence"
    protected.write_text("preserved")
    linked = tmp_path / "gpu-linked.json"
    linked.symlink_to(protected)
    with pytest.raises(OSError):
        read_lease(linked)
    corrupt = tmp_path / "gpu-corrupt.json"
    corrupt.write_text("not-json")
    with pytest.raises(ValueError):
        read_lease(corrupt)
    fifo = tmp_path / "gpu-fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="Unsafe"):
        read_lease(fifo)
    assert protected.read_text() == "preserved" and corrupt.exists() and fifo.exists()


def test_native_thread_limits_preserve_smaller_authored_values_and_do_not_alter_grants():
    original = {
        "OMP_NUM_THREADS": "999",
        "MKL_NUM_THREADS": "2",
        "CUDA_VISIBLE_DEVICES": "GPU-owned",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "USER_SETTING": "unchanged",
    }
    value = ProcessSupervisor._limit_thread_environment(original, 4)
    assert value["OMP_NUM_THREADS"] == "4" and value["MKL_NUM_THREADS"] == "2"
    assert (
        value["CUDA_VISIBLE_DEVICES"] == "GPU-owned" and value["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    )
    assert value["USER_SETTING"] == "unchanged" and original["OMP_NUM_THREADS"] == "999"


def test_linked_lease_root_is_rejected_before_observation(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        ProcessSupervisor, "_direct_gpu_devices", lambda: pytest.fail("queried GPU")
    )
    with pytest.raises(ValueError, match="symbolic"):
        ProcessSupervisor._acquire_gpus(linked, owner("job-new"), 1)
    assert list(real.iterdir()) == []
