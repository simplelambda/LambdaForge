"""Typed `kind: dataset` recipe configuration."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
from lambdaforge.data.DatasetRecipeSchemaCatalog import DatasetRecipeSchemaCatalog
from lambdaforge.data.DatasetRecipeStage import DatasetRecipeStage
from lambdaforge.workflows.WorkflowConfig import WorkflowConfig


class DatasetRecipeConfig:
    """Describe how stages build and atomically publish one immutable dataset version."""

    CURRENT_VERSION = "1.0"

    def __init__(self, data: Mapping[str, Any], *, source: str | Path | None = None) -> None:
        normalized = copy.deepcopy(dict(data))
        normalized.setdefault("schema_version", self.CURRENT_VERSION)
        schema_errors = DatasetRecipeSchemaCatalog().validation_errors(normalized)
        if schema_errors:
            raise ValueError("Invalid dataset recipe:\n- " + "\n- ".join(schema_errors))
        data = normalized
        if data.get("kind") != "dataset":
            raise ValueError("Dataset recipes require kind: dataset.")
        if str(data.get("schema_version", self.CURRENT_VERSION)) != self.CURRENT_VERSION:
            raise ValueError(f"Dataset recipe schema_version must be {self.CURRENT_VERSION!r}.")
        descriptor = data.get("dataset")
        if not isinstance(descriptor, Mapping):
            raise TypeError("Dataset recipes require a dataset mapping.")
        name = descriptor.get("name")
        version = descriptor.get("version")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset recipe name must be non-empty.")
        if version is None or not str(version).strip():
            raise ValueError("Dataset recipe version must be explicit and non-empty.")
        raw_stages = data.get("stages")
        if not isinstance(raw_stages, Mapping) or not raw_stages:
            raise ValueError("Dataset recipes require at least one stage.")
        publish = data.get("publish")
        if not isinstance(publish, Mapping):
            raise TypeError("Dataset recipes require a publish mapping.")
        if publish.get("from") not in raw_stages:
            raise ValueError("Dataset publish.from must name an existing stage.")
        if not isinstance(publish.get("index"), str) or not str(publish["index"]).strip():
            raise ValueError("Dataset publish.index must be a run-relative JSONL path.")
        self.source = Path(source).resolve() if source is not None else None
        self.source_dir = self.source.parent if self.source else Path.cwd().resolve()
        self.name = name
        self.version = str(version)
        self.dataset = copy.deepcopy(dict(descriptor))
        self.publish = copy.deepcopy(dict(publish))
        self.max_parallel = int(data.get("max_parallel", 1))
        configured_root = os.environ.get("LAMBDAFORGE_DATASET_BUILD_ROOT") or data.get(
            "output_root", "runs/datasets"
        )
        self.output_root = self._path(configured_root)
        publication_root = os.environ.get("LAMBDAFORGE_DATASET_ROOT")
        self.publication_root = (
            self._path(publication_root)
            if publication_root is not None
            else self.output_root / "published"
        )
        self.stages = tuple(
            DatasetRecipeStage.from_mapping(str(stage_name), stage, self.source_dir)
            for stage_name, stage in raw_stages.items()
            if isinstance(stage, Mapping)
        )
        if len(self.stages) != len(raw_stages):
            raise TypeError("Every dataset recipe stage must be a mapping.")
        self._validate_graph()

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetRecipeConfig:
        """Load a composed recipe while retaining source-relative stage paths."""
        resolution = ConfigurationComposer().resolve(path)
        if resolution.contains_secrets:
            raise ValueError("Dataset recipe structure cannot persist composed secrets.")
        return cls(resolution.materialized(reveal_secrets=True), source=path)

    @property
    def selector(self) -> str:
        """Return the immutable human alias produced by this recipe."""
        return f"{self.name}@{self.version}"

    @property
    def fingerprint(self) -> str:
        """Identify recipe structure and referenced task source bytes."""
        stage_descriptors = []
        for stage in self.stages:
            if isinstance(stage.task, Path):
                digest = hashlib.sha256(stage.task.read_bytes()).hexdigest()
                task: Any = {"sha256": digest}
            else:
                task = copy.deepcopy(dict(stage.task))
            stage_descriptors.append(
                {
                    "name": stage.name,
                    "task": task,
                    "needs": stage.needs,
                    "bindings": copy.deepcopy(stage.bindings),
                    "required": stage.required,
                    "reuse": stage.reuse,
                }
            )
        payload = {
            "recipe_identity_version": 1,
            "dataset": self.dataset,
            "publish": self.publish,
            "stages": stage_descriptors,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def workflow(self) -> WorkflowConfig:
        """Compile stages to the established Workflow DAG engine."""
        return WorkflowConfig(
            {
                "kind": "workflow",
                "schema_version": "1.0",
                "name": f"dataset-{self.name}-{self.version}",
                "output_root": str(self.output_root / "builds"),
                "max_parallel": self.max_parallel,
                "nodes": {stage.name: stage.to_workflow_node() for stage in self.stages},
            },
            source=self.source,
        )

    def downstream(self, names: set[str]) -> set[str]:
        """Return forced stages plus every transitive scientific dependent."""
        selected = set(names)
        changed = True
        while changed:
            changed = False
            for stage in self.stages:
                if stage.name not in selected and set(stage.needs) & selected:
                    selected.add(stage.name)
                    changed = True
        return selected

    def _path(self, value: Any) -> Path:
        path = Path(str(value))
        return path.resolve() if path.is_absolute() else (self.source_dir / path).resolve()

    def _validate_graph(self) -> None:
        names = {stage.name for stage in self.stages}
        for stage in self.stages:
            missing = set(stage.needs) - names
            if missing:
                raise ValueError(
                    f"Dataset stage {stage.name!r} has unknown dependencies: {sorted(missing)}."
                )
        self.workflow()
