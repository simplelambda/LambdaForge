"""Deterministic utility-aware online resource packing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveAssignment import AdaptiveAssignment
from lambdaforge.hpo.AdaptiveResource import AdaptiveResource


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
                    resource.memory_capacity_bytes <= 0
                    or sum(reservations[resource.name]) + action.memory_reservation_bytes
                    <= resource.memory_capacity_bytes
                )
            ]
            if not feasible:
                continue
            selected = min(
                feasible,
                key=lambda resource: (
                    (
                        resource.memory_capacity_bytes
                        - sum(reservations[resource.name])
                        - action.memory_reservation_bytes
                    )
                    if resource.memory_capacity_bytes
                    else 0,
                    resource.name,
                ),
            )
            reservations[selected.name].append(action.memory_reservation_bytes)
            assignments.append(AdaptiveAssignment(action, selected))
        return tuple(assignments)
