"""Domain-service facade shared by Research Console screens, never CLI subprocesses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from lambdaforge.analysis.StudyAnalysis import StudyAnalysis
from lambdaforge.controlplane.ClusterAuthentication import ClusterAuthentication
from lambdaforge.controlplane.ClusterBootstrapResult import ClusterBootstrapResult
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterService import ClusterService
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.CredentialService import CredentialService
from lambdaforge.controlplane.Doctor import Doctor
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.OverviewService import OverviewService
from lambdaforge.controlplane.ResourceService import ResourceService
from lambdaforge.controlplane.StorageService import StorageService
from lambdaforge.controlplane.StudyExportService import StudyExportService
from lambdaforge.controlplane.SubmissionService import SubmissionService
from lambdaforge.controlplane.WorkService import WorkService
from lambdaforge.data.DatasetService import DatasetService
from lambdaforge.tui.RecentWorkStore import RecentWorkStore
from lambdaforge.work.config import WorkConfig, WorkYamlError
from lambdaforge.work.ResultStore import ResultStore


class ConsoleServices:
    """Construct reusable domain services once for one console session."""

    def __init__(self, catalog: ClusterCatalog | None = None) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        # One console session owns one provider factory.  In particular this keeps a
        # password SSH session / OpenSSH ControlMaster reusable across resource probes,
        # bootstrap and dataset operations instead of opening parallel transports for
        # every visual panel.
        self.factory = ControlPlaneFactory()
        self.overview = OverviewService(self.catalog, self.factory)
        self.jobs = JobService(self.catalog, factory=self.factory)
        self.storage = StorageService(self.catalog, self.factory, jobs=self.jobs)
        self.works = WorkService(self.catalog, jobs=self.jobs, storage=self.storage)
        self.exports = StudyExportService(
            self.catalog,
            jobs=self.jobs,
            works=self.works,
            factory=self.factory,
        )
        self.resources = ResourceService(self.catalog, self.factory)
        self.cluster_service = ClusterService(self.catalog, self.factory)
        self.credentials = CredentialService()
        self.datasets = DatasetService(clusters=self.catalog, factory=self.factory)
        self.results = ResultStore()
        self.recent_work = RecentWorkStore()
        self._last_overview: dict[str, Any] | None = None

    def cluster_names(self) -> tuple[str, ...]:
        return self.catalog.names()

    def validate_work(self, config: Path) -> dict[str, Any]:
        return WorkConfig.validate_file(config).to_dict()

    def explain_work(self, config: Path) -> dict[str, Any]:
        return WorkConfig.from_yaml(config).explanation()

    def submit_work(self, config: Path, cluster: str) -> dict[str, Any]:
        return SubmissionService(self.catalog, self.jobs).enqueue(config, cluster=cluster).to_dict()

    def recent_work_configs(self, *, limit: int = 12) -> tuple[dict[str, str], ...]:
        """Return recent valid local YAML paths from MRU state and existing Job history."""
        choices = {item["path"]: item for item in self.recent_work.items(limit=20)}
        for record in self.jobs.list(refresh=False):
            source = record.metadata.get("source_config_path") or record.config_path
            if not isinstance(source, str):
                continue
            path = Path(source).expanduser()
            if not path.is_absolute() or path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            path = path.resolve()
            if not path.is_file():
                continue
            candidate = {
                "path": str(path),
                "name": str(record.metadata.get("name") or path.stem),
                "last_used_utc": record.created_at_utc,
            }
            previous = choices.get(str(path))
            if previous is None or candidate["last_used_utc"] > previous["last_used_utc"]:
                choices[str(path)] = candidate
        ordered = sorted(
            choices.values(), key=lambda item: item.get("last_used_utc", ""), reverse=True
        )
        return tuple(ordered[: max(0, min(int(limit), 20))])

    def remember_work_config(self, config: Path, *, name: str | None = None) -> None:
        """Remember one successfully validated YAML without copying its contents."""
        self.recent_work.remember(config, name=name)

    def overview_snapshot(self) -> dict[str, Any]:
        value = self.overview.snapshot()
        self._last_overview = value
        return value

    def research_snapshot(self) -> dict[str, Any]:
        """Load only collection-level Work/Study data for those root screens."""
        return self.overview.research_snapshot()

    def cluster_rows(self) -> list[dict[str, Any]]:
        clusters = (self._last_overview or {}).get("clusters", ())
        snapshots = {
            str(value.get("cluster")): value for value in clusters if isinstance(value, dict)
        }
        return [
            {**self.catalog.inspect(name), "resources": snapshots.get(name, {})}
            for name in self.catalog.names()
        ]

    def cluster_resources(self, name: str) -> dict[str, Any]:
        """Return one redacted live resource snapshot for visual refresh."""
        return self.resources.get(name).to_dict()

    def dataset_rows(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.datasets.list()]

    def result_rows(self) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in record.items() if key != "_manifest_path"}
            for record in self.results.list()
        ]

    def analyze(self, selector: str, *, recompute: bool = False) -> dict[str, Any]:
        return self.results.analysis(selector, recompute=recompute)

    def report(self, selector: str, output: Path) -> Path:
        return self.results.report(selector, output)

    def export_result(self, selector: str, destination: Path) -> dict[str, Any]:
        """Export one local persisted Execution through the shared package format."""
        return self.results.export(selector, destination)

    def export_work(self, selector: str, destination: Path) -> dict[str, Any]:
        """Download and export one successful local or remote Study/Work."""
        return self.exports.export(selector, destination)

    def study_run(
        self, job_id: str, run_key: str, *, tail: int = 2_000, curve_points: int = 200
    ) -> dict[str, Any]:
        """Load one bounded local/remote Run dashboard through ``JobService``."""
        return self.jobs.study_run(job_id, run_key, tail=tail, curve_points=curve_points)

    def study(self, job_id: str) -> dict[str, Any]:
        """Refresh one bounded study snapshot without rebuilding the global overview."""
        value = self.jobs.study(job_id)
        if value is None:
            raise KeyError(f"Job {job_id!r} has no Study telemetry.")
        return value

    def study_actions(self, job_id: str) -> tuple[dict[str, Any], ...]:
        """Load complete HPO decisions only when a Study workspace requests them."""
        return self.jobs.study_actions(job_id)

    def result_analysis(self, selector: str) -> dict[str, Any]:
        """Return the persisted Study Analysis shared by CLI, TUI and HTML report."""
        return self.results.analysis(selector)

    def live_study_analysis(
        self, study: Mapping[str, Any], *, job_id: str | None = None
    ) -> dict[str, Any]:
        """Analyze the current bounded Study snapshot without requiring a final Execution."""
        authored: Mapping[str, Any] = {}
        if job_id:
            try:
                record = self.jobs.get(job_id, refresh=False)
                configured = record.metadata.get("source_config_path") or record.config_path
                source = Path(str(configured)).expanduser().resolve() if configured else None
                if source is not None and source.is_file() and not source.is_symlink():
                    authored = StudyAnalysis.authored_space(WorkConfig.from_yaml(source).raw)
            except (OSError, TypeError, ValueError, WorkYamlError):
                # Observed candidate values still provide a truthful provisional domain.
                authored = {}
        objective = study.get("objective")
        objective = objective if isinstance(objective, Mapping) else None
        analysis = StudyAnalysis.compute(
            study,
            objective=objective,
            authored_space=authored,
            status="provisional",
            provisional_bootstrap_replicates=200,
        )
        analysis["search_space_source"] = (
            "authored configuration" if authored else "observed candidates"
        )
        live = study.get("hpo_analysis")
        if isinstance(live, Mapping):
            analysis["live_hpo"] = dict(live)
        return analysis

    def work_logs(self, job_id: str, *, tail: int = 2_000) -> dict[str, Any]:
        """Load structured Work logs without exposing provider paths to widgets."""
        return self.jobs.log_report(job_id, tail=tail, include_traceback=False)

    def cancel_work(self, selector: str) -> dict[str, Any]:
        return self.works.cancel(selector)

    def delete_work(self, selector: str, *, apply: bool = False) -> dict[str, Any]:
        return self.works.delete(selector, apply=apply)

    def retry_job(self, job_id: str) -> dict[str, Any]:
        return self.jobs.retry(job_id).to_dict()

    def dataset_describe(self, selector: str) -> dict[str, Any]:
        return self.datasets.describe(selector)

    def dataset_members(self, selector: str, *, limit: int = 200) -> dict[str, Any]:
        return self.datasets.members(selector, limit=limit)

    def dataset_stats(self, selector: str) -> dict[str, Any]:
        return self.datasets.stats(selector)

    def dataset_summary(self, selector: str) -> dict[str, Any]:
        """Read logical split/target counts without walking heavy asset trees."""
        return self.datasets.summary(selector)

    def dataset_verify(self, selector: str) -> dict[str, Any]:
        return self.datasets.verify(selector)

    def delete_dataset(self, selector: str, *, apply: bool = False) -> dict[str, Any]:
        """Preview or remove an entire DatasetVersion through its domain service."""
        return self.datasets.delete_version(selector, apply=apply)

    def delete_result(self, selector: str, *, apply: bool = False) -> dict[str, Any]:
        return self.results.delete(selector, apply=apply)

    def compare_results(self, selectors: tuple[str, ...]) -> dict[str, Any]:
        return self.results.compare(selectors)

    def cluster_detail(self, name: str) -> dict[str, Any]:
        """Compose redacted profile, current resources and storage status."""
        return {
            **self.catalog.inspect(name),
            "resources": self.resources.get(name).to_dict(),
            "storage": self.storage.status(name).to_dict(),
        }

    def doctor(self, name: str) -> dict[str, Any]:
        """Run the same read-only diagnostic service used by the CLI."""
        return Doctor(self.catalog).check(name).to_dict()

    def bootstrap(
        self,
        name: str,
        *,
        project: Path | None = None,
        dry_run: bool = False,
        progress: Any = None,
    ) -> dict[str, Any]:
        """Plan or apply managed bootstrap directly through the control plane."""
        result: ClusterBootstrapResult = self.cluster_service.bootstrap(
            name,
            project=project,
            dry_run=dry_run,
            progress=progress,
        )
        return result.to_dict()

    def set_cluster_credential(self, name: str, secret: str) -> dict[str, Any]:
        """Store a password in the OS keyring and persist only its reference."""
        profile = self.catalog.definition(name)
        source = self.catalog.source(name)
        if source is None or name == "local":
            raise ValueError("The selected cluster has no writable credential profile.")
        reference = profile.auth.credential
        if not reference or not reference.startswith("keyring:"):
            endpoint = f"{profile.user or 'default'}@{profile.host or 'local'}"
            reference = f"keyring:cluster/{name}/{endpoint}"
        self.credentials.store(reference, secret)
        ClusterCatalog.add(
            source,
            replace(profile, auth=ClusterAuthentication("password", reference)),
        )
        return {"cluster": name, "status": "stored", "credential_reference": reference}

    def delete_cluster_credential(self, name: str) -> dict[str, Any]:
        """Delete a keyring secret and leave password authentication interactive."""
        profile = self.catalog.definition(name)
        source = self.catalog.source(name)
        reference = profile.auth.credential
        if source is None or not reference or not reference.startswith("keyring:"):
            raise ValueError("The selected cluster has no stored keyring credential.")
        self.credentials.delete(reference)
        ClusterCatalog.add(
            source,
            replace(profile, auth=ClusterAuthentication("password", None)),
        )
        return {"cluster": name, "status": "deleted", "credential_reference": reference}

    def remove_cluster(self, name: str, *, apply: bool = False) -> dict[str, Any]:
        """Preview or remove one exact writable cluster profile."""
        source = self.catalog.source(name)
        if source is None:
            raise ValueError("The built-in local cluster cannot be removed.")
        preview = {
            "cluster": name,
            "catalog": str(source),
            "will_remove": ["cluster profile and non-secret credential reference"],
            "will_preserve": ["OS keyring secret", "jobs", "datasets", "remote files"],
            "applied": apply,
        }
        if apply:
            ClusterCatalog.remove(source, name)
        return preview


__all__ = ["ConsoleServices"]
