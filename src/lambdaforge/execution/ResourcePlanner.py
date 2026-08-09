"""Deterministic heterogeneous resource packing."""

from __future__ import annotations

from collections.abc import Mapping

from lambdaforge.execution.ResourcePlan import ResourcePlan
from lambdaforge.execution.ResourceRequest import ResourceRequest


class ResourcePlanner:
    """Pack requests into capacity-safe first-fit waves without runtime guessing."""

    def plan(
        self,
        requests: Mapping[str, ResourceRequest],
        *,
        capacity: ResourceRequest,
        manual_waves: tuple[tuple[str, ...], ...] | None = None,
    ) -> ResourcePlan:
        """Validate optional manual waves or create deterministic first-fit waves."""
        for name, request in requests.items():
            if not self._fits(request, capacity):
                raise ValueError(f"Resource request {name!r} exceeds host capacity.")
        waves = manual_waves or self._pack(requests, capacity)
        flattened = [name for wave in waves for name in wave]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(requests):
            raise ValueError("Manual resource waves must contain every request exactly once.")
        for wave in waves:
            total = self._sum(tuple(requests[name] for name in wave))
            if not self._fits(total, capacity):
                raise ValueError(f"Resource wave {wave!r} exceeds host capacity.")
        totals = [self._sum(tuple(requests[name] for name in wave)) for wave in waves]
        runtimes = [
            max((requests[name].runtime_seconds or 0 for name in wave), default=0) for wave in waves
        ]
        has_runtime = all(request.runtime_seconds is not None for request in requests.values())
        return ResourcePlan(
            waves=waves,
            peak_cpu_cores=max((value.cpu_cores for value in totals), default=0),
            peak_ram_bytes=max((value.ram_bytes for value in totals), default=0),
            peak_gpu_count=max((value.gpu_count for value in totals), default=0),
            storage_bytes=sum(request.storage_bytes for request in requests.values()),
            estimated_runtime_seconds=sum(runtimes) if has_runtime else None,
        )

    def _pack(
        self, requests: Mapping[str, ResourceRequest], capacity: ResourceRequest
    ) -> tuple[tuple[str, ...], ...]:
        waves: list[list[str]] = []
        for name in sorted(
            requests,
            key=lambda key: (
                -requests[key].gpu_count,
                -requests[key].ram_bytes,
                -requests[key].cpu_cores,
                key,
            ),
        ):
            for wave in waves:
                if self._fits(self._sum(tuple(requests[item] for item in (*wave, name))), capacity):
                    wave.append(name)
                    break
            else:
                waves.append([name])
        return tuple(tuple(wave) for wave in waves)

    @staticmethod
    def _fits(request: ResourceRequest, capacity: ResourceRequest) -> bool:
        return (
            request.cpu_cores <= capacity.cpu_cores
            and request.ram_bytes <= capacity.ram_bytes
            and request.gpu_count <= capacity.gpu_count
            and request.gpu_memory_bytes <= capacity.gpu_memory_bytes
        )

    @staticmethod
    def _sum(requests: tuple[ResourceRequest, ...]) -> ResourceRequest:
        return ResourceRequest(
            cpu_cores=sum(value.cpu_cores for value in requests),
            ram_bytes=sum(value.ram_bytes for value in requests),
            gpu_count=sum(value.gpu_count for value in requests),
            gpu_memory_bytes=sum(value.gpu_memory_bytes for value in requests),
            storage_bytes=sum(value.storage_bytes for value in requests),
        )
