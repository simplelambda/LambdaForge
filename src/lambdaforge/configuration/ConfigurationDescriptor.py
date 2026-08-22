"""One authoritative, side-effect-free description of executable configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskFingerprint import TaskFingerprint


@dataclass(frozen=True, slots=True)
class ConfigurationDescriptor:
    """Describe research identity and operational shape without constructing user objects."""

    source: Path
    kind: ConfigurationKind
    name: str
    scientific_identity: str
    datasets: tuple[str, ...]
    job_type: str
    planned_units: int | None
    unit: str
    materialized: dict[str, Any]

    @classmethod
    def from_path(
        cls, path: str | Path, *, resolve_task_code_identity: bool = True
    ) -> ConfigurationDescriptor:
        """Materialize one document and derive the metadata shared by every execution path."""
        source = Path(path).expanduser().resolve()
        document = AuthoringConfig.from_yaml(source).materialize()
        values = document.to_dict()
        name = cls._name(values, source)
        job_type = document.kind.value
        planned_units: int | None = 1
        unit = "task"
        if document.kind is ConfigurationKind.TASK:
            scientific_identity = (
                TaskConfig(values, source=source).fingerprint
                if resolve_task_code_identity
                else TaskFingerprint.digest(values)
            )
            if "preprocess" in values:
                job_type = "preprocessing"
            if cls._is_simple_work(values):
                job_type = "work"
        elif document.kind is ConfigurationKind.EXPERIMENT:
            config = ExperimentConfig(values, source=source)
            runs = config.expand()
            scientific_identity = RunFingerprint.digest(
                {"runs": [RunFingerprint.payload(run) for run in runs]}
            )
            hpo = values.get("hpo", {})
            adaptive = isinstance(hpo, dict) and bool(hpo.get("enabled"))
            job_type = "hpo" if adaptive else "experiment"
            planned_units = None if adaptive else len(runs)
            unit = "trials" if adaptive else "runs"
        else:
            scientific_identity = RunFingerprint.digest(document.to_dict())
            collection_key = "stages" if document.kind is ConfigurationKind.DATASET else "nodes"
            collection = values.get(collection_key)
            planned_units = len(collection) if isinstance(collection, dict) else 1
            if document.kind is ConfigurationKind.DATASET:
                job_type, unit = "dataset-build", "stages"
            else:
                unit = "nodes"
                if cls._is_simple_work(values):
                    metadata = values.get("metadata", {})
                    if (
                        isinstance(metadata, Mapping)
                        and int(metadata.get("search_variants", 1)) > 1
                    ):
                        job_type, unit = "hpo", "trials"
                    else:
                        job_type, unit = "work", "runs"
        return cls(
            source,
            document.kind,
            name,
            scientific_identity,
            cls.dataset_references(values),
            job_type,
            planned_units,
            unit,
            values,
        )

    @staticmethod
    def _name(values: dict[str, Any], source: Path) -> str:
        experiment = values.get("experiment", {})
        if isinstance(experiment, dict) and experiment.get("name"):
            return str(experiment["name"])
        dataset = values.get("dataset", {})
        if isinstance(dataset, dict) and dataset.get("name"):
            version = dataset.get("version")
            return f"{dataset['name']}@{version}" if version is not None else str(dataset["name"])
        return str(values.get("name", source.stem))

    @staticmethod
    def dataset_references(values: Mapping[str, Any]) -> tuple[str, ...]:
        """Return every explicit logical dataset selector in one materialized document."""
        found: set[str] = set()

        def visit(item: object) -> None:
            if isinstance(item, str) and item.startswith("dataset:"):
                found.add(item.removeprefix("dataset:").split("/", 1)[0])
            elif isinstance(item, Mapping):
                if set(item).issubset({"dataset", "version", "content_id", "subpath"}) and item.get(
                    "dataset"
                ):
                    name = str(item["dataset"])
                    version = item.get("version")
                    found.add(f"{name}@{version}" if version is not None else name)
                for nested in item.values():
                    visit(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(values)
        return tuple(sorted(found))

    @staticmethod
    def _is_simple_work(values: Mapping[str, Any]) -> bool:
        extensions = values.get("extensions", {})
        authoring = extensions.get("authoring", {}) if isinstance(extensions, Mapping) else {}
        metadata = values.get("metadata", {})
        return bool(
            isinstance(authoring, Mapping)
            and authoring.get("simple_work")
            or isinstance(metadata, Mapping)
            and metadata.get("authoring") == "simple-work"
        )

    @staticmethod
    def dataset_ids(values: Mapping[str, Any]) -> tuple[str, ...]:
        """Return immutable managed content identities captured during materialization."""
        found: set[str] = set()

        def visit(item: object) -> None:
            if isinstance(item, Mapping):
                identity = item.get("identity")
                if isinstance(identity, Mapping):
                    dataset_id = identity.get("dataset_id", identity.get("content_id"))
                    if isinstance(dataset_id, str) and dataset_id.startswith("sha256:"):
                        found.add(dataset_id)
                for nested in item.values():
                    visit(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(values)
        return tuple(sorted(found))

    @property
    def revision(self) -> str:
        """Return a concise human-facing scientific revision."""
        return self.scientific_identity.removeprefix("sha256:")[:12]

    def metadata(self) -> dict[str, Any]:
        """Return stable metadata persisted on the low-level job record."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "scientific_identity": self.scientific_identity,
            "scientific_revision": self.revision,
            "source_config_path": str(self.source),
            "datasets": list(self.datasets),
            "dataset_ids": list(self.dataset_ids(self.materialized)),
            "planned_units": self.planned_units,
            "unit": self.unit,
        }
