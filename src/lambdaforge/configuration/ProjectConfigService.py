"""Project-local discovery and name resolution for runnable YAML sources."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.configuration.ProjectConfigRecord import ProjectConfigRecord
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.results.ResultService import ResultService


class ProjectConfigService:
    """Index bounded project YAML files and keep every operation tied to its source."""

    EXCLUDED_PARTS = {
        ".git",
        ".lambdaforge",
        ".venv",
        "build",
        "dist",
        "node_modules",
        "runs",
    }

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        jobs: JobService | None = None,
    ) -> None:
        self.root = self._root(Path(root or Path.cwd()).resolve())
        self.jobs = jobs or JobService(ClusterCatalog.load())

    def list(self, *, kind: str | None = None) -> tuple[ProjectConfigRecord, ...]:
        active = tuple(
            record for record in self.jobs.list(refresh=False) if not record.state.terminal
        )
        values = []
        for path in self._paths():
            if not self._looks_runnable(path):
                continue
            try:
                materialized = AuthoringConfig.from_yaml(path).materialize()
                payload = materialized.to_dict()
                name = self._name(payload, path)
                current_kind = materialized.kind.value
                if kind is not None and current_kind != kind:
                    continue
                related = tuple(
                    job.job_id
                    for job in active
                    if job.metadata.get("name") == name
                    or Path(str(job.config_path)).name == path.name
                )
                related_clusters = tuple(
                    sorted({job.cluster for job in active if job.job_id in related})
                )
                values.append(
                    ProjectConfigRecord(
                        name,
                        current_kind,
                        path,
                        datasets=self.datasets(payload),
                        resources=self._resources(payload),
                        hpo_enabled=self._hpo(payload),
                        active_jobs=related,
                        active_clusters=related_clusters,
                        last_result=self._last_result(payload, path, name),
                    )
                )
            except Exception as error:
                # Non-LambdaForge YAML is not a broken LambdaForge configuration.
                if self._looks_like_lambdaforge(path):
                    values.append(
                        ProjectConfigRecord(
                            path.stem,
                            "unknown",
                            path,
                            valid=False,
                            error=f"{error.__class__.__name__}: {error}",
                        )
                    )
        return tuple(sorted(values, key=lambda value: (value.kind, value.name, str(value.path))))

    def resolve(self, selector: str | Path, *, kind: str | None = None) -> Path:
        path = Path(selector)
        if path.exists():
            return path.resolve()
        matches = tuple(
            record
            for record in self.list(kind=kind)
            if record.valid and (record.name == str(selector) or record.path.stem == str(selector))
        )
        if not matches:
            available = tuple(record.name for record in self.list(kind=kind) if record.valid)
            raise KeyError(f"Unknown configuration {str(selector)!r}; available: {available}.")
        if len(matches) > 1:
            paths = tuple(str(record.path.relative_to(self.root)) for record in matches)
            raise ValueError(f"Configuration name {str(selector)!r} is ambiguous: {paths}.")
        return matches[0].path

    def show(self, selector: str | Path) -> ProjectConfigRecord:
        path = self.resolve(selector)
        return next(record for record in self.list() if record.path == path)

    def _paths(self) -> tuple[Path, ...]:
        values = []
        for suffix in ("*.yaml", "*.yml"):
            for path in self.root.rglob(suffix):
                relative = path.relative_to(self.root)
                if not any(
                    part in self.EXCLUDED_PARTS or part.startswith(".") for part in relative.parts
                ):
                    values.append(path.resolve())
        return tuple(sorted(set(values)))

    @staticmethod
    def _root(start: Path) -> Path:
        return next(
            (value for value in (start, *start.parents) if (value / "pyproject.toml").is_file()),
            start,
        )

    @staticmethod
    def _name(payload: Mapping[str, Any], path: Path) -> str:
        experiment = payload.get("experiment")
        if isinstance(experiment, Mapping) and experiment.get("name"):
            return str(experiment["name"])
        dataset = payload.get("dataset")
        if isinstance(dataset, Mapping) and dataset.get("name"):
            return str(dataset["name"])
        return str(payload.get("name", path.stem))

    @classmethod
    def datasets(cls, value: object) -> tuple[str, ...]:
        """Return logical dataset names referenced by a materialized document."""
        found: set[str] = set()

        def visit(item: object) -> None:
            if isinstance(item, str) and item.startswith("dataset:"):
                found.add(item.removeprefix("dataset:").split("/", 1)[0])
            elif isinstance(item, Mapping):
                if set(item).issubset(
                    {"dataset", "version", "content_id", "subpath"}
                ) and item.get("dataset"):
                    found.add(str(item["dataset"]))
                for nested in item.values():
                    visit(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(value)
        return tuple(sorted(found))

    @staticmethod
    def _resources(payload: Mapping[str, Any]) -> dict[str, Any]:
        value = payload.get("resources", payload.get("execution", {}))
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _hpo(payload: Mapping[str, Any]) -> bool:
        value = payload.get("hpo", {})
        return isinstance(value, Mapping) and bool(value.get("enabled", False))

    @staticmethod
    def _last_result(payload: Mapping[str, Any], source: Path, name: str) -> dict[str, Any] | None:
        experiment = payload.get("experiment", {})
        configured = (
            experiment.get("output_root")
            if isinstance(experiment, Mapping)
            else payload.get("output_root")
        )
        root = Path(str(configured or "runs"))
        if not root.is_absolute():
            root = source.parent / root
        try:
            matches = tuple(
                record
                for record in ResultService((root,)).records()
                if record.result.name == name or record.result.name.startswith(f"{name}__")
            )
        except Exception:
            return None
        if not matches:
            return None
        latest = max(
            matches,
            key=lambda record: (
                record.result.finished_at_utc or record.result.started_at_utc or record.attempt_id
            ),
        )
        return latest.to_dict()

    @staticmethod
    def _looks_like_lambdaforge(path: Path) -> bool:
        try:
            prefix = path.read_text(encoding="utf-8")[:8192]
        except OSError:
            return False
        return any(
            token in prefix
            for token in (
                "schema_version:",
                "kind: task",
                "kind: workflow",
                "kind: dataset",
                "experiment:",
            )
        )

    @staticmethod
    def _looks_runnable(path: Path) -> bool:
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return ProjectConfigService._looks_like_lambdaforge(path)
        if not isinstance(value, Mapping):
            return False
        if value.get("kind") in {"task", "workflow", "dataset"} or "experiment" in value:
            return True
        if "extends" in value or "include" in value:
            return True
        return "preprocess" in value or (
            "model" in value and ("trainer" in value or "loss" in value or "losses" in value)
        )
