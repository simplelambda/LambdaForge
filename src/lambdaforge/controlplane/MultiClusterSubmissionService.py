"""Explicit independent submission of one config to several clusters."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.JobGroupStore import JobGroupStore
from lambdaforge.controlplane.jobs import JobGroup
from lambdaforge.controlplane.SubmissionService import SubmissionService
from lambdaforge.execution.ResourceRequest import ResourceRequest


class MultiClusterSubmissionService:
    """Submit independent replicas and label them honestly as a group."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        *,
        groups: JobGroupStore | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.control = ControlPlane(self.catalog)
        self.groups = groups or JobGroupStore()

    def submit(
        self,
        config: str | Path,
        clusters: tuple[str, ...],
        *,
        resources: ResourceRequest | None = None,
        dry_run: bool = False,
        independent_hpo: bool = False,
        wait_for_submit: bool = True,
    ) -> JobGroup:
        if len(set(clusters)) != len(clusters) or not clusters:
            raise ValueError("Multi-cluster submission requires unique cluster names.")
        materialized = AuthoringConfig.from_yaml(config).materialize()
        hpo = materialized.values.get("hpo", {})
        if isinstance(hpo, dict) and hpo.get("enabled") and not independent_hpo:
            raise ValueError(
                "Multi-cluster HPO is not coordinated. Use --independent-hpo to run "
                "separate studies with no shared optimizer state."
            )
        name = ControlPlane._configuration_name(config)
        group_id = f"group-{uuid4().hex[:16]}"
        group = JobGroup(
            group_id,
            name,
            (),
            tuple(clusters),
            datetime.now(timezone.utc).isoformat(),
        )
        if not dry_run:
            # Persist before the first provider call so accepted jobs remain discoverable
            # if a later cluster rejects the submission.
            self.groups.put(group)
        handles = []
        for cluster in clusters:
            if not dry_run and not wait_for_submit:
                handle = SubmissionService(self.catalog).enqueue(
                    config,
                    cluster=cluster,
                    resources=resources,
                    group_id=group_id,
                )
            else:
                handle, _ = self.control.submit(
                    config,
                    cluster=cluster,
                    resources=resources,
                    dry_run=dry_run,
                    group_id=group_id,
                )
            handles.append(handle)
            group = JobGroup(
                group_id,
                name,
                tuple(value.job_id for value in handles),
                tuple(clusters),
                group.created_at_utc,
            )
            if not dry_run:
                self.groups.put(group)
        return group
