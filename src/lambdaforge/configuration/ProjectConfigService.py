"""Project-local discovery and name resolution for runnable YAML sources."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from lambdaforge.configuration.ConfigurationDescriptor import ConfigurationDescriptor
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
        jobs = self.jobs.list(refresh=False)
        values = []
        for path in self._paths():
            if not self._looks_runnable(path):
                continue
            try:
                descriptor = ConfigurationDescriptor.from_path(path)
                payload = descriptor.materialized
                name = descriptor.name
                current_kind = descriptor.kind.value
                if kind is not None and current_kind != kind:
                    continue
                named_history = tuple(
                    job
                    for job in jobs
                    if job.metadata.get("name") == name
                    or Path(str(job.config_path)).name == path.name
                )
                history = tuple(
                    job
                    for job in named_history
                    if job.metadata.get("scientific_identity")
                    == descriptor.scientific_identity
                    or not job.metadata.get("scientific_identity")
                    )
                active = tuple(job for job in history if not job.state.terminal)
                related = tuple(job.job_id for job in active)
                related_clusters = tuple(
                    sorted({job.cluster for job in active})
                )
                results = self._results(path)
                completed = len(
                    {
                        record.config_fingerprint or record.attempt_id
                        for record in results
                        if record.status == "ok"
                    }
                )
                last_result = self._last_result_from(results)
                state = self._state(active, results)
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
                        last_result=last_result,
                        scientific_identity=descriptor.scientific_identity,
                        scientific_revision=descriptor.revision,
                        state=state,
                        planned_runs=descriptor.planned_units,
                        completed_runs=completed,
                        unit=descriptor.unit,
                        attempt_count=max(len(named_history), len(results)),
                        executions=tuple(self._execution(job) for job in named_history),
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
            if record.valid
            and (
                record.name == str(selector)
                or record.path.stem == str(selector)
                or (
                    record.kind == "dataset"
                    and record.name.partition("@")[0] == str(selector)
                )
            )
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
    def datasets(value: object) -> tuple[str, ...]:
        """Return logical dataset names referenced by a materialized document."""
        return (
            ConfigurationDescriptor.dataset_references(value)
            if isinstance(value, Mapping)
            else ()
        )

    @staticmethod
    def _execution(job: Any) -> dict[str, Any]:
        identity = job.metadata.get("scientific_identity")
        return {
            "execution_id": job.job_id,
            "job_id": job.job_id,
            "cluster": job.cluster,
            "state": job.state.value,
            "attempt": int(job.metadata.get("attempt", 1)),
            "retry_of": job.retry_of,
            "created_at_utc": job.created_at_utc,
            "updated_at_utc": job.updated_at_utc,
            "scheduler_id": job.scheduler_id,
            "scientific_identity": identity,
            "scientific_revision": (
                str(identity).removeprefix("sha256:")[:12] if identity else None
            ),
            "resources": dict(job.resources),
        }

    @staticmethod
    def _results(source: Path) -> tuple[Any, ...]:
        try:
            return ResultService().resolve(source)
        except Exception:
            return ()

    @staticmethod
    def _last_result_from(records: tuple[Any, ...]) -> dict[str, Any] | None:
        if not records:
            return None
        latest = max(
            records,
            key=lambda record: (
                record.result.finished_at_utc
                or record.result.started_at_utc
                or record.attempt_id
            ),
        )
        return latest.to_dict()

    @staticmethod
    def _state(active: tuple[Any, ...], results: tuple[Any, ...]) -> str:
        priorities = ("running", "queued", "staging", "preparing", "paused", "unknown")
        active_states = {job.state.value for job in active}
        for state in priorities:
            if state in active_states:
                return state
        statuses = {record.status for record in results}
        if "ok" in statuses:
            return "succeeded"
        if "failed" in statuses:
            return "failed"
        if "interrupted" in statuses:
            return "interrupted"
        return "not_run"

    @staticmethod
    def _resources(payload: Mapping[str, Any]) -> dict[str, Any]:
        value = payload.get("resources", payload.get("execution", {}))
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _hpo(payload: Mapping[str, Any]) -> bool:
        value = payload.get("hpo", {})
        return isinstance(value, Mapping) and bool(value.get("enabled", False))

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
