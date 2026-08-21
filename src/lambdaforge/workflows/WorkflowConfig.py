"""Validated workflow YAML configuration."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.configuration.AuthoringConfigNormalizer import AuthoringConfigNormalizer
from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.workflows.WorkflowNode import WorkflowNode
from lambdaforge.workflows.WorkflowSchemaCatalog import WorkflowSchemaCatalog


class WorkflowConfig:
    """Own a versioned DAG configuration and stable workflow identity."""

    CURRENT_VERSION = "1.0"

    def __init__(self, data: Mapping[str, Any], *, source: str | Path | None = None) -> None:
        materialized = AuthoringConfigNormalizer().normalize(data, source=source)
        if materialized.kind is not ConfigurationKind.WORKFLOW:
            raise ValueError("Configuration does not describe a workflow.")
        data = materialized.values
        schema_errors = WorkflowSchemaCatalog().validation_errors(data)
        if schema_errors:
            raise ValueError("Invalid workflow configuration:\n- " + "\n- ".join(schema_errors))
        if data.get("kind") != "workflow":
            raise ValueError("Workflow documents require 'kind: workflow'.")
        if data.get("schema_version") != self.CURRENT_VERSION:
            raise ValueError(f"Workflow schema_version must be {self.CURRENT_VERSION!r}.")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Workflow name must be a non-empty string.")
        raw_nodes = data.get("nodes")
        if not isinstance(raw_nodes, Mapping) or not raw_nodes:
            raise ValueError("Workflow nodes must be a non-empty mapping.")
        self.source = Path(source).resolve() if source else None
        source_dir = self.source.parent if self.source else Path.cwd().resolve()
        self.name = name
        self.max_parallel = self._positive_int(data.get("max_parallel", 1), "max_parallel")
        raw_resources = data.get("resources")
        if raw_resources is not None and not isinstance(raw_resources, Mapping):
            raise TypeError("Workflow resources must be a mapping.")
        self.resource_override = (
            copy.deepcopy(dict(raw_resources)) if isinstance(raw_resources, Mapping) else None
        )
        output = Path(str(data.get("output_root", "runs/workflows")))
        self.output_root = (source_dir / output).resolve() if not output.is_absolute() else output
        self.nodes = tuple(
            WorkflowNode.from_mapping(str(node_name), node, source_dir)
            for node_name, node in raw_nodes.items()
            if isinstance(node, Mapping)
        )
        if len(self.nodes) != len(raw_nodes):
            raise TypeError("Every workflow node must be a mapping.")
        self._validate_graph()

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkflowConfig:
        """Load a composed workflow, preserving the leaf document as its path base."""
        resolved = ConfigurationComposer().resolve(path)
        if resolved.contains_secrets:
            raise ValueError(
                "Workflow structure cannot contain secrets because materialized node snapshots "
                "are persisted. Read secrets inside consumer code from the runtime environment."
            )
        return cls(resolved.materialized(reveal_secrets=True), source=path)

    @property
    def run_dir(self) -> Path:
        """Return the workflow directory keyed by its complete DAG definition."""
        payload = {
            "name": self.name,
            "nodes": [
                {
                    "name": node.name,
                    "config": str(node.config),
                    "needs": node.needs,
                    "bindings": dict(node.bindings),
                    "resources": dict(node.resources),
                    "on": node.cluster,
                }
                for node in self.nodes
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        return self.output_root / self.name / digest[:16]

    def _validate_graph(self) -> None:
        names = {node.name for node in self.nodes}
        for node in self.nodes:
            missing = set(node.needs) - names
            if missing:
                raise ValueError(
                    f"Workflow node {node.name!r} has unknown dependencies: {sorted(missing)}"
                )
        visiting: set[str] = set()
        visited: set[str] = set()
        dependencies = {node.name: node.needs for node in self.nodes}

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"Workflow dependency cycle includes {name!r}.")
            if name in visited:
                return
            visiting.add(name)
            for dependency in dependencies[name]:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in names:
            visit(name)

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Workflow {label} must be a positive integer.")
        return value
