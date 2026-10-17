"""Work-conserving host-resource leases for isolated Study Runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lambdaforge.execution.ResourceRequest import ResourceRequest


@dataclass(frozen=True, slots=True)
class HostResourceLease:
    """One dispatch-time share of an enclosing Study's aggregate host allocation.

    CPU is a soft, elastic share backed by affinity rebalancing; RAM/storage are admission-time
    commitments rather than a claim that every possible theoretical Run is resident.  The lease
    is provenance carried by the Run and is not a second scheduler.
    """

    aggregate_cpu: int
    aggregate_ram_bytes: int
    aggregate_storage_bytes: int
    active_runs: int
    hard_concurrency_ceiling: int
    cpu_floor: int
    cpu_soft_share: int
    ram_commitment_bytes: int
    storage_commitment_bytes: int
    processes: int
    controller_memory_bytes: int = 0
    policy_version: str = "elastic-host-lease-v1"
    observed_resident_memory_bytes: int | None = None

    @classmethod
    def allocate(
        cls,
        resources: ResourceRequest,
        *,
        active_runs: int,
        hard_concurrency_ceiling: int,
        controller_memory_bytes: int = 0,
        resident_memory_bytes: int | None = None,
    ) -> HostResourceLease:
        residents = max(1, int(active_runs))
        ceiling = max(residents, int(hard_concurrency_ceiling))
        cpu_share = max(1, resources.cpu_cores // residents)
        # RAM and scratch are committed against current residents. They are re-evaluated before
        # every dispatch instead of being permanently divided by a hypothetical maximum.
        overhead = max(0, int(controller_memory_bytes))
        ram = max(0, resources.ram_bytes - overhead) // residents if resources.ram_bytes else 0
        # The final launch observes the whole tree, including resource trackers and existing
        # workers. A stale fair share is a ceiling, not an additional fixed RAM requirement.
        # Reconcile with actual aggregate headroom rather than blocking a lone Run forever
        # because helper memory was absent from the earlier controller-only measurement.
        observed = max(0, int(resident_memory_bytes)) if resident_memory_bytes is not None else None
        if resources.ram_bytes and observed is not None:
            ram = min(ram, max(0, resources.ram_bytes - observed))
        storage = resources.storage_bytes // residents if resources.storage_bytes else 0
        return cls(
            aggregate_cpu=resources.cpu_cores,
            aggregate_ram_bytes=resources.ram_bytes,
            aggregate_storage_bytes=resources.storage_bytes,
            active_runs=residents,
            hard_concurrency_ceiling=ceiling,
            cpu_floor=1,
            cpu_soft_share=cpu_share,
            ram_commitment_bytes=ram,
            storage_commitment_bytes=storage,
            processes=min(cpu_share, resources.processes),
            controller_memory_bytes=overhead,
            observed_resident_memory_bytes=observed,
        )

    def resources(self, aggregate: ResourceRequest) -> ResourceRequest:
        return ResourceRequest(
            cpu_cores=self.cpu_soft_share,
            ram_bytes=self.ram_commitment_bytes,
            gpu_count=1 if aggregate.gpu_count else 0,
            gpu_memory_bytes=aggregate.gpu_memory_bytes,
            runtime_seconds=aggregate.runtime_seconds,
            storage_bytes=self.storage_commitment_bytes,
            processes=self.processes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "aggregate": {
                "cpu": self.aggregate_cpu,
                "ram_bytes": self.aggregate_ram_bytes,
                "storage_bytes": self.aggregate_storage_bytes,
            },
            "active_runs_at_dispatch": self.active_runs,
            "hard_concurrency_ceiling": self.hard_concurrency_ceiling,
            "cpu_floor": self.cpu_floor,
            "cpu_soft_share": self.cpu_soft_share,
            "ram_commitment_bytes": self.ram_commitment_bytes,
            "controller_memory_bytes": self.controller_memory_bytes,
            "observed_resident_memory_bytes": self.observed_resident_memory_bytes,
            "storage_commitment_bytes": self.storage_commitment_bytes,
            "processes": self.processes,
            "provenance": "dispatch-time-aggregate-host-capacity",
        }


__all__ = ["HostResourceLease"]
