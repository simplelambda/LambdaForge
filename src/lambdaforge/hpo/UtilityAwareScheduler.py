"""Deterministic utility-aware online resource packing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveAssignment import AdaptiveAssignment
from lambdaforge.hpo.AdaptiveResource import AdaptiveResource
from lambdaforge.hpo.MemoryCapacityKind import MemoryCapacityKind


class UtilityAwareScheduler:
    """Pack highest-utility actions with CPU, job-count and conservative VRAM limits."""

    def pack(
        self,
        actions: Sequence[AdaptiveAction],
        resources: Sequence[AdaptiveResource],
        *,
        active_reservations: Mapping[str, Sequence[int]] | None = None,
    ) -> tuple[AdaptiveAssignment, ...]:
        """Return a reproducible greedy utility-aware best-fit assignment."""
        reservations = {
            resource.name: list((active_reservations or {}).get(resource.name, ()))
            for resource in resources
        }
        assignments: list[AdaptiveAssignment] = []
        for action in sorted(actions, key=lambda value: (-value.utility, value.action_id)):
            feasible = [
                resource
                for resource in resources
                if len(reservations[resource.name]) < resource.max_jobs
                and (
                    resource.memory_capacity.kind is MemoryCapacityKind.UNBOUNDED
                    or (
                        resource.memory_capacity.kind is MemoryCapacityKind.KNOWN
                        and resource.memory_capacity.bytes is not None
                        and sum(reservations[resource.name]) + action.memory_reservation_bytes
                        <= resource.memory_capacity.bytes
                    )
                )
            ]
            if not feasible:
                continue
            selected = min(
                feasible,
                key=lambda resource: (
                    (
                        (resource.memory_capacity.bytes or 0)
                        - sum(reservations[resource.name])
                        - action.memory_reservation_bytes
                    )
                    if resource.memory_capacity.kind is MemoryCapacityKind.KNOWN
                    else 0,
                    resource.name,
                ),
            )
            reservations[selected.name].append(action.memory_reservation_bytes)
            assignments.append(AdaptiveAssignment(action, selected))
        return tuple(assignments)
