"""Thin dataset CLI adapter; dataset behavior remains in data services."""

from __future__ import annotations

import argparse
import json
from typing import Any

import yaml

from lambdaforge.configuration.ProjectConfigService import ProjectConfigService
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.data.DatasetBuildService import DatasetBuildService
from lambdaforge.data.DatasetService import DatasetService
from lambdaforge.data.recipe_config import DatasetRecipeConfig


class DatasetCommands:
    """Parse/render dataset commands while delegating lifecycle decisions to services."""

    @classmethod
    def run(cls, arguments: argparse.Namespace) -> int:
        service = DatasetService(clusters=ClusterCatalog.load(arguments.clusters))
        operation = arguments.dataset_command
        if operation in {"plan", "build"}:
            recipe_path = ProjectConfigService().resolve(arguments.dataset, kind="dataset")
            recipe = DatasetRecipeConfig.from_yaml(recipe_path)
            builds = DatasetBuildService(
                service.registry,
                JobService(service.clusters, factory=service.factory),
            )
            if operation == "plan":
                payload: Any = builds.plan(
                    recipe,
                    cluster=arguments.on,
                    force=arguments.force,
                    force_stages=arguments.force_stage,
                ).to_dict()
            else:
                handle = builds.submit(
                    recipe,
                    cluster=arguments.on,
                    force=arguments.force,
                    force_stages=arguments.force_stage,
                    dry_run=arguments.dry_run,
                    wait_for_submit=getattr(arguments, "wait_for_submit", False),
                )
                payload = handle.to_dict()
        elif operation == "list":
            payload = [
                value.to_dict()
                for value in service.list(cluster=arguments.on, all_clusters=arguments.all)
            ]
        elif operation == "show":
            payload = service.show(arguments.dataset).to_dict()
        elif operation == "add":
            payload = service.add(
                arguments.manifest, cluster=arguments.on, root=arguments.root
            ).to_dict()
        elif operation == "locations":
            payload = [value.to_dict() for value in service.locations(arguments.dataset)]
        elif operation == "stats":
            schema = (
                yaml.safe_load(arguments.schema.read_text(encoding="utf-8"))
                if arguments.schema
                else None
            )
            payload = service.stats(arguments.dataset, cluster=arguments.on, schema=schema)
        elif operation == "members":
            partitions = dict(cls._partition(value) for value in arguments.partition)
            payload = service.members(
                arguments.dataset,
                cluster=arguments.on,
                partitions=partitions,
                offset=arguments.offset,
                limit=arguments.limit,
            )
        elif operation == "member":
            payload = service.member(arguments.dataset, arguments.member_id, cluster=arguments.on)
        elif operation == "diff":
            payload = service.diff(arguments.left, arguments.right, cluster=arguments.on)
        elif operation == "verify":
            payload = service.verify(arguments.dataset, cluster=arguments.on)
        elif operation == "lineage":
            payload = service.lineage(arguments.dataset)
        elif operation == "remove":
            removed = service.remove(arguments.dataset, cluster=arguments.on)
            payload = {"removed": True, "remaining": removed.to_dict() if removed else None}
        elif operation == "delete":
            payload = service.delete(
                arguments.dataset, cluster=arguments.on, apply=arguments.apply
            ).to_dict()
        elif operation == "materialize":
            payload = service.materialize(
                arguments.dataset,
                cluster=arguments.on,
                strategy=arguments.strategy,
                apply=arguments.apply,
            ).to_dict()
        else:
            payload = service.replicate(
                arguments.dataset,
                source=arguments.source,
                destination=arguments.destination,
                apply=arguments.apply,
            ).to_dict()
        if getattr(arguments, "json", False):
            if isinstance(payload, dict) and getattr(arguments, "default_cluster_source", None):
                payload.setdefault("target_source", arguments.default_cluster_source)
            print(json.dumps(payload, indent=2))
        else:
            cls._human(
                operation,
                payload,
                default_source=getattr(arguments, "default_cluster_source", None),
                verbose=getattr(arguments, "verbose", False),
            )
        return 0

    @staticmethod
    def _partition(value: str) -> tuple[str, str]:
        key, separator, selected = value.partition("=")
        if not separator or not key or not selected:
            raise ValueError("--partition requires NAME=VALUE.")
        return key, selected

    @staticmethod
    def _human(
        operation: str,
        payload: Any,
        *,
        default_source: str | None = None,
        verbose: bool = False,
    ) -> None:
        if operation == "list":
            print("DATASET  MEMBERS  PLACEMENTS  CONTENT")
            for record in payload:
                print(
                    f"{record['name']}@{record['version']}  {record['sample_count']}  "
                    f"{','.join(item['cluster'] for item in record['placements']) or '-'}  "
                    f"{record['dataset_id']}"
                )
            return
        if operation == "plan":
            source = f" ({default_source})" if default_source else ""
            print(f"Dataset: {payload['dataset']}  Target: {payload['target_cluster']}{source}")
            resources = payload["resources"]
            print(
                "Reservation: "
                f"CPU={resources['cpu_cores']} RAM={resources['ram_bytes']} "
                f"GPU={resources['gpu_count']} processes={resources['processes']} "
                f"time={resources['runtime_seconds']}s"
            )
            print("STAGE  ACTION" + ("  REASON" if verbose else ""))
            for stage in payload["stages"]:
                suffix = f"  {stage['reason']}" if verbose else ""
                print(f"{stage['stage']}  {stage['action']}{suffix}")
            publish = payload["publish"]
            suffix = f"  {publish['reason']}" if verbose else ""
            print(f"publish  {publish['action']}{suffix}")
            return
        if operation == "members":
            print("MEMBER  PARTITIONS  TARGETS  ASSETS")
            for member in payload["members"]:
                print(
                    f"{member['id']}  {member['partitions']}  {member['targets']}  "
                    f"{','.join(member['assets']) or '-'}"
                )
            print(f"Returned {payload['returned']} (offset={payload['offset']}).")
            return
        if operation == "show":
            print(f"Dataset: {payload['name']}@{payload['version']}")
            print(f"Content: {payload['dataset_id']}")
            print(f"Build: {payload.get('build_id') or 'unknown'}")
            print(f"Members: {payload['sample_count']}")
            print(f"Partitions: {payload.get('partitions', {})}")
            print(f"Placements: {', '.join(p['cluster'] for p in payload['placements']) or 'none'}")
            return
        if operation == "build":
            source = f" ({default_source})" if default_source else ""
            print(
                f"Dataset build job {payload['job_id']}: {payload['state']} "
                f"on {payload['cluster']}{source}"
            )
            return
        print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).rstrip())
