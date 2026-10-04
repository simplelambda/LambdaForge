"""Fleet preflight observations reuse ResourceService; physical facts do not grant GPUs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.Fleet import Fleet
from lambdaforge.controlplane.ResourceService import ResourceService


class FleetResourceService:
    """Operational discovery, not another monitor or a local GPU admission implementation."""

    def __init__(self, catalog: ClusterCatalog, resources: ResourceService | None = None) -> None:
        self.catalog = catalog
        self.resources = resources or ResourceService(catalog)

    def inspect(self, fleet: Fleet) -> dict[str, Any]:
        def observe(cluster: str) -> dict[str, Any]:
            profile = self.catalog.get(cluster)
            member = fleet.member(cluster)
            snapshot = self.resources.get(cluster)
            # Existing ResourceService probes report physical observations, not allocation or
            # site grants. Only a shard executor may later attest currently admissible slots.
            return {
                "cluster": cluster,
                "operator_state": member.state.value,
                "health": "degraded" if snapshot.online else "unreachable",
                "required": member.required,
                "scheduler": profile.scheduler,
                "gpu_access": profile.gpu_access.effective_mode(profile.scheduler),
                "caps": member.to_dict(),
                "resource_snapshot": snapshot.to_dict(),
                "admissible_capacity": None,
                "environment_ready": None,
                "inputs_ready": None,
                "explanation": (
                    "Physical observations are not an allocation. Local executor "
                    "readiness and admission have not been attested."
                ),
            }

        with ThreadPoolExecutor(max_workers=min(4, len(fleet.members))) as pool:
            members = list(pool.map(observe, (member.cluster for member in fleet.members)))
        return {
            "fleet": fleet.to_dict(),
            "members": members,
            "required_members_reachable": all(
                not item["required"] or item["health"] != "unreachable" for item in members
            ),
            "dispatch_ready": False,
        }

    def set_member_state(self, name: str, cluster: str, state: str, *, path: str | Path) -> None:
        """Persist operator controls; never signal active/foreign processes."""
        from dataclasses import replace

        from lambdaforge.controlplane.Fleet import ClusterHealth

        fleet = self.catalog.fleet(name)
        fleet.member(cluster)
        changed = replace(
            fleet,
            members=tuple(
                replace(member, state=ClusterHealth(state)) if member.cluster == cluster else member
                for member in fleet.members
            ),
        )
        ClusterCatalog.add_fleet(path, changed)
