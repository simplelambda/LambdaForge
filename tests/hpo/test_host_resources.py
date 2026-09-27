from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.hpo.HostResources import HostResourceLease


def test_host_lease_uses_actual_residents_not_theoretical_parallelism() -> None:
    resources = ResourceRequest(cpu_cores=36, ram_bytes=96 * 1024**3, gpu_count=3, processes=8)

    lease = HostResourceLease.allocate(
        resources, active_runs=3, hard_concurrency_ceiling=36
    )

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
