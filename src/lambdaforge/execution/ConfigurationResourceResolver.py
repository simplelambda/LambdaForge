"""Resolve scheduler resources from every runnable configuration family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.data.recipe_config import DatasetRecipeConfig
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.workflows.WorkflowConfig import WorkflowConfig


class ConfigurationResourceResolver:
    """Turn authored resources into the fixed allocation submitted to a scheduler.

    Tasks and experiments map directly. Workflows and dataset recipes run inside one allocation,
    so their request is either the explicit top-level mapping or a conservative aggregation of
    node/stage requests across the dependency DAG and ``max_parallel`` limit.
    """

    _ADDITIVE_FIELDS = (
        "cpu_cores",
        "ram_bytes",
        "gpu_count",
        "storage_bytes",
        "processes",
    )

    @classmethod
    def resolve(cls, source: str | Path) -> ResourceRequest:
        """Resolve one composed YAML without constructing consumer objects."""
        path = Path(source).expanduser().resolve()
        materialized = AuthoringConfig.from_yaml(path).materialize()
        return cls._resolve(materialized.values, materialized.kind, source=path)

    @classmethod
    def resolve_dataset(cls, config: DatasetRecipeConfig) -> ResourceRequest:
        """Resolve an already validated dataset recipe without requiring a source path."""
        return cls._dataset(config)

    @classmethod
    def _resolve(
        cls,
        values: Mapping[str, Any],
        kind: ConfigurationKind,
        *,
        source: Path | None,
    ) -> ResourceRequest:
        if kind is ConfigurationKind.TASK:
            return TaskConfig(values, source=source).resources
        if kind is ConfigurationKind.EXPERIMENT:
            return ExperimentConfig(values, source=source).resources
        if kind is ConfigurationKind.WORKFLOW:
            return cls._workflow(WorkflowConfig(values, source=source))
        return cls._dataset(DatasetRecipeConfig(values, source=source))

    @classmethod
    def _workflow(cls, config: WorkflowConfig) -> ResourceRequest:
        if config.resource_override is not None:
            return ResourceRequest.from_mapping(config.resource_override)
        requests: dict[str, ResourceRequest] = {}
        dependencies: dict[str, tuple[str, ...]] = {}
        for node in config.nodes:
            dependencies[node.name] = node.needs
            if node.resources:
                requests[node.name] = ResourceRequest.from_mapping(node.resources)
                continue
            values, source, _ = node.materialize()
            materialized = AuthoringConfig(values, source=source).materialize()
            requests[node.name] = cls._resolve(
                materialized.values,
                materialized.kind,
                source=source,
            )
        return cls._aggregate(requests, cls._levels(dependencies), config.max_parallel)

    @classmethod
    def _dataset(cls, config: DatasetRecipeConfig) -> ResourceRequest:
        if config.resource_override is not None:
            return ResourceRequest.from_mapping(config.resource_override)
        requests: dict[str, ResourceRequest] = {}
        dependencies: dict[str, tuple[str, ...]] = {}
        for stage in config.stages:
            dependencies[stage.name] = stage.needs
            if isinstance(stage.task, Path):
                requests[stage.name] = cls.resolve(stage.task)
                continue
            source = config.source_dir / ".lambdaforge-embedded-dataset-stage.yaml"
            materialized = AuthoringConfig(stage.task, source=source).materialize()
            requests[stage.name] = cls._resolve(
                materialized.values,
                materialized.kind,
                source=source,
            )
        return cls._aggregate(requests, cls._levels(dependencies), config.max_parallel)

    @staticmethod
    def _levels(dependencies: Mapping[str, Sequence[str]]) -> tuple[tuple[str, ...], ...]:
        remaining = {name: set(needs) for name, needs in dependencies.items()}
        completed: set[str] = set()
        levels: list[tuple[str, ...]] = []
        while remaining:
            ready = tuple(sorted(name for name, needs in remaining.items() if needs <= completed))
            if not ready:
                raise ValueError("Resource aggregation found a dependency cycle.")
            levels.append(ready)
            completed.update(ready)
            for name in ready:
                del remaining[name]
        return tuple(levels)

    @classmethod
    def _aggregate(
        cls,
        requests: Mapping[str, ResourceRequest],
        levels: Sequence[Sequence[str]],
        max_parallel: int,
    ) -> ResourceRequest:
        """Return a safe fixed allocation for a bounded in-process DAG executor."""
        capacity: dict[str, int] = {field: 0 for field in cls._ADDITIVE_FIELDS}
        for level in levels:
            concurrent = min(max_parallel, len(level))
            for field in cls._ADDITIVE_FIELDS:
                values = sorted(
                    (int(getattr(requests[name], field)) for name in level), reverse=True
                )
                capacity[field] = max(capacity[field], sum(values[:concurrent]))
        all_requests = [requests[name] for level in levels for name in level]
        runtime_values = [request.runtime_seconds for request in all_requests]
        runtime = (
            sum(value for value in runtime_values if value is not None)
            if runtime_values and None not in runtime_values
            else None
        )
        return ResourceRequest(
            **capacity,
            gpu_memory_bytes=max((request.gpu_memory_bytes for request in all_requests), default=0),
            runtime_seconds=runtime,
        )
