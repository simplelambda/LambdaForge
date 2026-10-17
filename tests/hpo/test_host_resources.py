from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.hpo.HostResources import HostResourceLease


def test_first_dispatch_reserves_controller_ram_without_losing_aggregate_identity():
    resources = ResourceRequest(cpu_cores=4, ram_bytes=100)
    lease = HostResourceLease.allocate(
        resources, active_runs=1, hard_concurrency_ceiling=4, controller_memory_bytes=20
    )
    assert lease.aggregate_ram_bytes == 100
    assert lease.ram_commitment_bytes == 80
    assert lease.controller_memory_bytes == 20
    assert lease.resources(resources).ram_bytes == 80


def test_host_lease_uses_actual_residents_not_theoretical_parallelism() -> None:
    resources = ResourceRequest(cpu_cores=36, ram_bytes=96 * 1024**3, gpu_count=3, processes=8)

    lease = HostResourceLease.allocate(resources, active_runs=3, hard_concurrency_ceiling=36)

    assert lease.cpu_soft_share == 12
    assert lease.ram_commitment_bytes == 32 * 1024**3
    assert lease.processes == 8
    assert lease.resources(resources).cpu_cores == 12


def test_host_lease_contracts_as_real_concurrency_grows() -> None:
    resources = ResourceRequest(cpu_cores=36, ram_bytes=96 * 1024**3, processes=12)

    one = HostResourceLease.allocate(resources, active_runs=1, hard_concurrency_ceiling=36)
    six = HostResourceLease.allocate(resources, active_runs=6, hard_concurrency_ceiling=36)

    assert one.cpu_soft_share == 36
    assert six.cpu_soft_share == 6
    assert six.ram_commitment_bytes == 16 * 1024**3


def test_fresh_whole_tree_headroom_bounds_native_fair_share():
    resources = ResourceRequest(cpu_cores=4, ram_bytes=100)
    lease = HostResourceLease.allocate(
        resources,
        active_runs=1,
        hard_concurrency_ceiling=4,
        controller_memory_bytes=20,
        resident_memory_bytes=24,
    )
    assert lease.ram_commitment_bytes == 76
    assert lease.ram_commitment_bytes + lease.observed_resident_memory_bytes == 100
    assert lease.to_dict()["observed_resident_memory_bytes"] == 24
    full = HostResourceLease.allocate(
        resources,
        active_runs=2,
        hard_concurrency_ceiling=4,
        resident_memory_bytes=100,
    )
    assert full.ram_commitment_bytes == 0
