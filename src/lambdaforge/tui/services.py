"""Domain-service facade shared by Research Console screens, never CLI subprocesses."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
from lambdaforge.execution.ResourceRequest import ResourceRequest
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

    def dependency_choices(self, *, offset: int = 0) -> list[dict[str, Any]]:
        """Metadata-only local/imported catalog and independent published product choices."""
        if type(offset) is not int or offset < 0:
            raise ValueError("Invalid dependency catalog page.")
        rows = [
            {
                "kind": "imported result" if row.get("imported") else "result",
                "label": row["display_name"],
                "state": row["status"],
                "identity": row["execution_id"],
            }
            for row in self.results.catalog()
        ]
        products = []
        for row in self.product_rows(offset=offset):
            products.append(
                {
                    "kind": "product",
                    "label": row["name"],
                    "state": row["contract"]["identifier"],
                    "identity": row["content_id"],
                    "contract": row["contract"]["identifier"],
                }
            )
        return rows[offset : offset + 100] + products

    def dependency_details(self, choice: Mapping[str, Any]) -> dict[str, Any]:
        """Selected source metadata only, never artifact bytes or all Run curves."""
        if choice["kind"] == "product":
            return self.product_detail(str(choice["identity"]))
        return self.results.view(str(choice["identity"]), view="overview")

    def dependency_reference(
        self,
        choice: Mapping[str, Any],
        *,
        run: str | None = None,
        attempt: int | None = None,
        artifact: str | None = None,
    ) -> dict[str, Any]:
        if choice["kind"] == "product":
            from lambdaforge.products.dependency import ProductRequirement, resolve_product_input

            requirement = ProductRequirement(
                str(choice["identity"]), str(choice["contract"]), artifact=artifact
            )
            resolved, _ = resolve_product_input(requirement.to_dict(), Path.cwd())
            return {"product": {**requirement.to_dict(), "name": resolved.content_id}}
        return self.results.reference(str(choice["identity"]), run=run, attempt=attempt)

    def materialize_dependency(
        self,
        selector: str,
        cluster: str,
        *,
        kind: str,
        apply: bool = False,
        expected_evidence_id: str | None = None,
    ) -> dict[str, Any]:
        from lambdaforge.controlplane.DependencyMaterialization import DependencyMaterialization

        return DependencyMaterialization(self.catalog, self.factory).materialize(
            selector,
            cluster=cluster,
            kind=kind,
            apply=apply,
            expected_evidence_id=expected_evidence_id,
        )

    def fleet_targets(self) -> tuple[str, ...]:
        """Operational targets, separate from real cluster profile names."""
        return tuple("fleet:" + name for name in self.catalog.fleet_names())

    def validate_work(
        self, config: Path, *, input_bindings: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if input_bindings:
            parsed = WorkConfig.from_yaml(config, input_bindings=input_bindings)
            errors = parsed.validation_errors(check_inputs=True)
            return {"valid": not errors, "errors": list(errors), "name": parsed.name}
        return WorkConfig.validate_file(config).to_dict()

    def explain_work(
        self, config: Path, *, input_bindings: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return WorkConfig.from_yaml(config, input_bindings=input_bindings).explanation()

    def submit_work(
        self, config: Path, cluster: str, *, input_bindings: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if cluster.startswith("fleet:"):
            if input_bindings:
                raise ValueError("Declare Fleet dependency inputs in the authored Study YAML.")
            fleet = self.catalog.fleet(cluster.removeprefix("fleet:"))
            return (
                SubmissionService(self.catalog, self.jobs)
                .enqueue(
                    config,
                    cluster=fleet.coordinator,
                    fleet=fleet.name,
                    resources=ResourceRequest(cpu_cores=1),
                )
                .to_dict()
            )
        return (
            SubmissionService(self.catalog, self.jobs)
            .enqueue(config, cluster=cluster, input_bindings=input_bindings)
            .to_dict()
        )

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
        self._include_imports(value)
        self._last_overview = value
        return value

    def research_snapshot(self) -> dict[str, Any]:
        """Load only collection-level Work/Study data for those root screens."""
        value = self.overview.research_snapshot()
        self._include_imports(value)
        return value

    def _include_imports(self, value: dict[str, Any]) -> None:
        from lambdaforge.work.ImportedStudy import ImportedStudy

        items = value.setdefault("work", {}).setdefault("items", [])
        existing = {item.get("work_id") for item in items}
        items.extend(
            row
            for row in ImportedStudy.rows(self.results.root, include_works=True)
            if row["work_id"] not in existing
        )

    def _imported(self, selector: str) -> Any:
        from lambdaforge.work.ImportedStudy import ImportedStudy

        return (
            ImportedStudy(self.results.root, selector) if selector.startswith("import:") else None
        )

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
        records = self.datasets.list(all_clusters=True)
        if not records and self.datasets.discovery_failures:
            from lambdaforge.data.errors import DatasetResolutionError

            raise DatasetResolutionError(
                "Dataset discovery is incomplete; absence cannot be established. "
                + "; ".join(self.datasets.discovery_failures)
            )
        identities: dict[str, set[str]] = {}
        references = {record.key: record.dataset_id for record in self.datasets.registry.records()}
        for record in records:
            identities.setdefault(record.key, set()).add(record.dataset_id)
        return [
            record.to_dict()
            | {
                "inventory_conflict": len(identities[record.key]) > 1,
                "project_reference": references.get(record.key) == record.dataset_id,
                "discovery_warnings": list(self.datasets.discovery_warnings),
            }
            for record in records
        ]

    def result_rows(self) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in record.items() if key != "_manifest_path"}
            for record in self.results.catalog()
        ]

    def product_rows(self, *, offset: int = 0) -> list[dict[str, Any]]:
        """One bounded catalog page; never touch model bytes or scientific fitting."""
        from lambdaforge.products import ProductRegistry

        return [
            {
                "name": value.name,
                "kind": value.kind,
                "contract": value.contract.to_dict(),
                "content_id": value.content_id,
                "scientific_id": value.scientific_id,
                "artifact_count": len(value.artifacts),
                "size_bytes": sum(item.size_bytes for item in value.artifacts),
            }
            for value in ProductRegistry().list(offset=offset, limit=100)
        ]

    def product_detail(self, selector: str, *, offset: int = 0) -> dict[str, Any]:
        from lambdaforge.products import ProductRegistry

        registry = ProductRegistry()
        return {
            "product": registry.show(selector).to_dict(),
            "provenance": registry.provenance(selector, offset=offset, limit=20),
            "consumers": registry.consumers(selector, offset=offset, limit=20),
            "offset": offset,
        }

    def verify_product(self, selector: str) -> dict[str, Any]:
        from lambdaforge.products import ProductRegistry

        return ProductRegistry().verify(selector)

    def export_product(self, selector: str, destination: Path) -> dict[str, Any]:
        from lambdaforge.products import ProductBundle, ProductRegistry

        return ProductBundle.export(ProductRegistry(), selector, destination, apply=True)

    def import_study(self, source: Path, *, apply: bool = False) -> dict[str, Any]:
        return self.results.import_export(source, apply=apply)

    def analyze(self, selector: str, *, recompute: bool = False) -> dict[str, Any]:
        return self.results.analysis(selector, recompute=recompute)

    def report(self, selector: str, output: Path) -> Path:
        return self.results.report(selector, output)

    def work_result_view(self, work: Mapping[str, Any], **options: Any) -> dict[str, Any]:
        """Shared Console route for native/provider/imported evidence, selected on demand."""
        job = work.get("primary_job_id")
        if job and not work.get("imported"):
            return self.jobs.work_result_view(str(job), **options)
        return self.results.view(str(work["execution_id"]), **options)

    def work_report(self, work: Mapping[str, Any], output: Path) -> Path:
        """Explicit provider/local report collection, never a bulk artifact export."""
        from lambdaforge.analysis.WorkReport import write_work_html

        overview = self.work_result_view(work)
        attempts, sections = [], []
        for row in overview["attempts"]:
            options = {"run_id": row["run_id"], "attempt": row["attempt_number"]}
            attempts.append(self.work_result_view(work, view="attempt", **options))
            sections.extend(self.work_result_view(work, view="html", **options).get("sections", ()))
        return write_work_html(overview, attempts, output, sections=sections)

    def work_output_preview(
        self, work: Mapping[str, Any], name: str, **options: Any
    ) -> dict[str, Any]:
        """An explicit 64 KiB text preview; never download arbitrary model/HTML bytes."""
        job = work.get("primary_job_id")
        if job and not work.get("imported"):
            return self.jobs.work_result_view(
                str(job), view="outputs", artifact_preview=name, **options
            )
        return self.results.output_preview(str(work["execution_id"]), name, **options)

    def historical_result(self, execution_id: str) -> dict[str, Any]:
        """Navigate an exact known result using compact catalog metadata, never a name alias."""
        matches = [row for row in self.results.catalog() if row["execution_id"] == execution_id]
        if len(matches) != 1:
            raise ValueError(
                "Exact historical Execution is not available in this catalog; "
                "explicitly export/import its evidence before opening it here."
            )
        return {key: value for key, value in matches[0].items() if key != "_manifest_path"}

    def export_result(self, selector: str, destination: Path) -> dict[str, Any]:
        """Export one local persisted Execution through the shared package format."""
        return self.results.export(selector, destination)

    def export_work(
        self,
        selector: str,
        destination: Path,
        *,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Download and export the newest state of one local or remote Study/Work."""
        return self.exports.export(selector, destination, progress=progress)

    def study_run(
        self, job_id: str, run_key: str, *, tail: int = 2_000, curve_points: int = 200
    ) -> dict[str, Any]:
        """Load one bounded local/remote Run dashboard through ``JobService``."""
        imported = self._imported(job_id)
        if imported:
            return imported.run(run_key, tail=tail, curve_points=curve_points)
        return self.jobs.study_run(job_id, run_key, tail=tail, curve_points=curve_points)

    def study(self, job_id: str) -> dict[str, Any]:
        """Refresh one bounded study snapshot without rebuilding the global overview."""
        imported = self._imported(job_id)
        value = imported.view("interactive") if imported else self.jobs.study(job_id)
        if value is None:
            raise KeyError(f"Job {job_id!r} has no Study telemetry.")
        return value

    def study_workspace(self, job_id: str, *, latest_job_id: str) -> dict[str, Any]:
        if self._imported(job_id):
            return self.study(job_id)
        value = self.jobs.study_workspace(job_id, latest_job_id=latest_job_id)
        if value is None:
            raise KeyError(f"Job {latest_job_id!r} has no Study telemetry.")
        return value

    def study_trial(self, job_id: str, trial: int) -> dict[str, Any]:
        imported = self._imported(job_id)
        if imported:
            return imported.view("trial", trial=trial)
        return self.jobs.study_trial(job_id, trial)

    def study_trials(self, job_id: str, *, query: str) -> dict[str, Any]:
        imported = self._imported(job_id)
        if imported:
            return imported.view("interactive", query=query)
        return self.jobs.study_trials(job_id, query=query)

    def study_panel(self, job_id: str, view: str) -> dict[str, Any]:
        imported = self._imported(job_id)
        if imported:
            return imported.view(view)
        return self.jobs.study_panel(job_id, view)

    def study_analysis(self, job_id: str, *, full: bool = False) -> dict[str, Any]:
        """Read final analysis, or compute/cache it on its execution host on explicit demand."""
        imported = self._imported(job_id)
        if imported:
            return imported.analysis(full=full)
        return self.jobs.study_panel(job_id, "analysis-report" if full else "analysis")

    def study_report_sections(self, job_id: str) -> list[dict[str, str]]:
        """Explicit report generation only; ordinary Study views never fetch project HTML."""
        if self._imported(job_id):
            return []  # Original project HTML remains in the portable archive.
        return self.jobs.study_panel(job_id, "report-sections").get("sections", [])

    def study_hpo(self, job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Present existing live HPO conclusions, without refitting analysis on the controller."""
        from lambdaforge.analysis.Effects import infer_space

        panel = self.study_panel(job_id, "hpo")
        hpo = panel.get("hpo_analysis") or {}
        initialization = (panel.get("controller") or {}).get("initialization") or {}
        policy = initialization.get("policy") or {}
        analysis = {
            "status": "provisional",
            "live_hpo": hpo,
            "search_space": infer_space((), policy.get("parameter_space") or {}),
            "coverage": {
                "marginal": {
                    str(detail["parameter"]): {
                        "observed_range": detail.get("observed_range"),
                        "observed_levels": [g.get("value") for g in detail.get("groups", ())],
                    }
                    for detail in hpo.get("parameters", ())
                    if isinstance(detail, Mapping) and detail.get("parameter") is not None
                }
            },
            "scientific_understanding": hpo.get("scientific_understanding") or {},
        }
        return panel, analysis

    def study_actions(self, job_id: str) -> tuple[dict[str, Any], ...]:
        """Load complete HPO decisions only when a Study workspace requests them."""
        imported = self._imported(job_id)
        if imported:
            return imported.actions()
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
        imported = self._imported(job_id)
        if imported:
            return self.results.log_report(job_id.removeprefix("import:"), tail=tail)
        return self.jobs.log_report(job_id, tail=tail, include_traceback=False)

    def cancel_work(self, selector: str) -> dict[str, Any]:
        return self.works.cancel(selector)

    def delete_work(self, selector: str, *, apply: bool = False) -> dict[str, Any]:
        return self.works.delete(selector, apply=apply)

    def retry_preview(self, job_id: str) -> dict[str, Any]:
        return self.jobs.retry_preview(job_id)

    def retry_job(self, job_id: str, *, accept_code_change: bool = False) -> dict[str, Any]:
        return self.jobs.retry(job_id, accept_code_change=accept_code_change).to_dict()

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

    def replicate_dataset(
        self,
        selector: str,
        *,
        source: str,
        destination: str,
        expected_content_id: str,
        apply: bool = False,
        route: str = "auto",
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Use the native preview/apply placement operation, never a console-only copy."""
        result = self.datasets.replicate(
            selector,
            source=source,
            destination=destination,
            apply=apply,
            expected_content_id=expected_content_id,
            route=route,
            progress=progress,
        ).to_dict()
        if apply:
            result["placements"] = [
                item.to_dict() for item in self.datasets.registry.get(selector).placements
            ]
        return result

    def delete_dataset(self, selector: str, *, apply: bool = False) -> dict[str, Any]:
        """Preview or remove an entire DatasetVersion through its domain service."""
        return self.datasets.delete_version(selector, apply=apply)

    def manage_dataset_copy(
        self,
        selector: str,
        *,
        operation: str,
        cluster: str,
        content_id: str,
        apply: bool = False,
        expected_root: str | None = None,
        expected_controller_id: str | None = None,
    ) -> dict[str, Any]:
        """Same exact-reference/placement preview and apply as the public CLI."""
        if operation == "adopt":
            return self.datasets.adopt(
                selector,
                cluster=cluster,
                content_id=content_id,
                apply=apply,
                expected_root=expected_root,
                expected_controller_id=expected_controller_id,
            )
        if operation == "remove":
            return self.datasets.retire(
                selector,
                cluster=cluster,
                content_id=content_id,
                apply=apply,
                expected_root=expected_root,
            )
        if operation == "delete":
            return self.datasets.delete(
                selector,
                cluster=cluster,
                content_id=content_id,
                apply=apply,
                expected_root=expected_root,
            ).to_dict()
        raise ValueError("Unknown Dataset copy operation.")

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

    def clean_storage(self, name: str, *, apply: bool = False) -> dict[str, Any]:
        """Use exactly the same safe preview/apply authority as native lf clean."""
        return self.storage.gc(name, apply=apply).to_dict()

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
