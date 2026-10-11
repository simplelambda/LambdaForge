"""Public project-pinned facade for agents and external Python applications.

Only metadata/explicit native planning is exposed initially. No shell execution, CLI subprocess,
widget dependencies, hidden artifact download or alternate scientific state exists here.
"""

from __future__ import annotations

import ast
import inspect
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml

from lambdaforge._version import VERSION
from lambdaforge.agents.contracts import Operation, Permission, operation, page
from lambdaforge.agents.knowledge import KINDS, KnowledgeService
from lambdaforge.agents.scope import ProjectScope


class AgentServices:
    """Bind one project for the lifetime of a GUI or MCP session, independent of cwd."""

    def __init__(
        self,
        root: str | Path,
        *,
        allow_project_code: bool = False,
        toolsets: tuple[str, ...] | None = None,
        permissions: tuple[Permission, ...] = ("READ", "PLAN"),
    ) -> None:
        self.scope = ProjectScope(root)
        self.allow_project_code = allow_project_code
        self.toolsets = toolsets
        if set(permissions) - {"READ", "PLAN", "WRITE_PROJECT", "SUBMIT", "OPERATE", "DESTRUCTIVE"}:
            raise ValueError("Unknown session permission.")
        self.permissions = tuple(dict.fromkeys(permissions))
        self.knowledge = KnowledgeService(self.scope)
        self._catalog: Any = None
        self._jobs: Any = None

    def check_operation(self, spec: Operation) -> None:
        self.scope.assert_current()
        if spec.permission not in self.permissions:
            raise PermissionError("Operation permission is not enabled by the session operator.")
        if (
            self.toolsets is not None
            and spec.toolset != "core"
            and spec.toolset not in self.toolsets
        ):
            raise PermissionError("Operation is outside this session's selected toolsets.")
        if spec.project_code and not self.allow_project_code:
            raise PermissionError(
                "Native validation imports trusted project Python. Enable --allow-project-code "
                "explicitly in the server's operator configuration to allow this operation."
            )

    def _clusters(self) -> Any:
        if self._catalog is None:
            from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog

            self._catalog = ClusterCatalog.load(project=self.scope.project)
        return self._catalog

    def _job_service(self) -> Any:
        if self._jobs is None:
            from lambdaforge.controlplane.JobService import JobService

            self._jobs = JobService(self._clusters())
        return self._jobs

    def _results(self) -> Any:
        from lambdaforge.work.ResultStore import ResultStore

        return ResultStore(self.scope.path(".lambdaforge/runs", internal=True))

    def _products(self) -> Any:
        from lambdaforge.products import ProductRegistry

        return ProductRegistry(self.scope.path(".lambdaforge/products", internal=True))

    def _datasets(self) -> Any:
        from lambdaforge.data.DatasetRegistry import DatasetRegistry
        from lambdaforge.data.DatasetService import DatasetService

        return DatasetService(
            DatasetRegistry(self.scope.path(".lambdaforge/datasets.json", internal=True)),
            self._clusters(),
        )

    def _config(self, config: str) -> Any:
        from lambdaforge.work.config import WorkConfig

        self.scope.file(config, limit=512 * 1024)
        if Path(config).suffix not in {".yaml", ".yml"}:
            raise ValueError("Work configuration requires a project YAML file.")
        return WorkConfig.from_yaml(self.scope.path(config))

    def _dataset_record(self, content_id: str) -> Any:
        if not content_id.startswith("sha256:"):
            raise ValueError("Use an exact registered dataset content ID, not a name.")
        matches = [
            record
            for record in self._datasets().registry.records()
            if record.dataset_id == content_id
        ]
        if len(matches) != 1:
            raise ValueError("Dataset identity is missing or ambiguous in this project's registry.")
        return matches[0]

    @operation("project.info", "core", "READ", "ProjectContext", "docs/MANUAL.md")
    def project_info(self) -> dict[str, Any]:
        """Identify the pinned project and installed framework without importing project code."""
        return {
            **self.scope.project.to_dict(),
            "framework_version": VERSION,
            "permissions": list(self.permissions),
            "allow_project_code": self.allow_project_code,
            "knowledge": self.knowledge.status(),
            "http_available": True,
            "http_scope": "opt-in paired loopback READ/PLAN; no network deployment",
            "mutations_available": "WRITE_PROJECT" in self.permissions,
            "mutation_scope": (
                "locally approved knowledge edits/ZIP export; no scientific operations"
            ),
        }

    @operation("agent.actions", "core", "READ", "ApprovalService", "docs/ai/INTEGRATION.md")
    def agent_actions(self) -> dict[str, Any]:
        """Discover the closed native action set; unavailable permissions never become approval."""
        from lambdaforge.agents.approvals import ACTIONS

        return {
            "items": [
                {
                    "action": name,
                    "permission": spec.permission,
                    "service": spec.service,
                    "session_enabled": spec.permission in self.permissions
                    and (self.toolsets is None or "knowledge" in self.toolsets),
                    "arguments": sorted(spec.arguments),
                    "required_arguments": sorted(spec.required),
                    "human_approval_required": True,
                }
                for name, spec in ACTIONS.items()
            ],
            "submit_operate_destructive_available": False,
        }

    @operation(
        "agent.action.preview", "knowledge", "PLAN", "ApprovalService", "docs/ai/INTEGRATION.md"
    )
    def preview_agent_action(self, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Pure exact proposal; consumer content is data, never an approval or instruction."""
        from lambdaforge.agents.approvals import ApprovalService

        return ApprovalService(self.scope).preview(action, arguments)

    @operation(
        "agent.action.request",
        "knowledge",
        "WRITE_PROJECT",
        "ApprovalService",
        "docs/ai/INTEGRATION.md",
        side_effects=("persist expiring unapproved local proposal and audit metadata",),
    )
    def request_agent_action(
        self, action: str, arguments: dict[str, Any], ttl_seconds: int = 900
    ) -> dict[str, Any]:
        """Queue a proposal for a human local reviewer; this never approves or executes it."""
        from lambdaforge.agents.approvals import ApprovalService

        return ApprovalService(self.scope).request(action, arguments, ttl_seconds=ttl_seconds)

    @operation(
        "agent.action.status", "knowledge", "READ", "ApprovalService", "docs/ai/INTEGRATION.md"
    )
    def agent_action_status(self, request_id: str) -> dict[str, Any]:
        """Exact signed approval status without note bodies, credentials or signing key."""
        from lambdaforge.agents.approvals import ApprovalService

        return ApprovalService(self.scope).inspect(request_id)

    @operation(
        "agent.action.apply",
        "knowledge",
        "WRITE_PROJECT",
        "ApprovalService",
        "docs/ai/INTEGRATION.md",
        side_effects=("consume trusted approval once and edit exact interpretation Markdown",),
        idempotent=False,
    )
    def apply_agent_action(self, request_id: str, plan_hash: str) -> dict[str, Any]:
        """Apply only a still-valid locally approved plan; replay/stale/self-approval is refused."""
        from lambdaforge.agents.approvals import ApprovalService

        return ApprovalService(self.scope).apply(
            request_id, plan_hash=plan_hash, permissions=self.permissions
        )

    @operation("framework.version", "core", "READ", "LambdaForgeVersion", "README.md")
    def framework_version(self) -> dict[str, Any]:
        """Return installed framework and agent contract versions, not a guessed release."""
        return {"version": VERSION, "agent_schema_version": 1}

    @operation(
        "agent.dependencies.check", "core", "READ", "DependencyService", "docs/ai/INTEGRATION.md"
    )
    def check_dependencies(self, extra: str = "agents") -> dict[str, Any]:
        """Check the declared optional group without import, install or network access."""
        from lambdaforge.agents.dependencies import DependencyService

        return DependencyService(self.scope).check(extra)

    @operation("agent.http", "core", "READ", "Official MCP SDK", "docs/ai/INTEGRATION.md")
    def describe_http_transport(self) -> dict[str, Any]:
        """Describe paired local transport, without revealing keys or starting a server."""
        from lambdaforge.agents.mcp_server import dependency_status

        return {
            "available": dependency_status()["available"],
            "enabled_by_default": False,
            "bind_address": "127.0.0.1",
            "permissions": ["READ", "PLAN"],
            "authentication": "explicit private local capability-token pairing, not OAuth",
            "network_deployment_supported": False,
            "browser_origins": "explicit exact allowlist",
        }

    @operation(
        "framework.skills.list", "core", "READ", "Installed assets", "docs/ai/INTEGRATION.md"
    )
    def list_agent_skills(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Discover official installed Skills without loading their entire procedures."""
        from lambdaforge.agents.integration import skill_metadata, skill_names

        return page(
            [
                {k: v for k, v in skill_metadata(name).items() if k != "document"}
                for name in skill_names()
            ],
            offset,
            limit,
        )

    @operation(
        "framework.skills.describe", "core", "READ", "Installed assets", "docs/ai/INTEGRATION.md"
    )
    def get_agent_skill_metadata(self, name: str) -> dict[str, Any]:
        """Read one installed specialized Skill, never an arbitrary filesystem path."""
        from lambdaforge.agents.integration import skill_metadata

        return skill_metadata(name)

    @operation(
        "framework.capabilities", "core", "READ", "CapabilityCatalog", "docs/ai/INTEGRATION.md"
    )
    def framework_capabilities(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Page actual operations, disabled permissions and native limitations."""
        from lambdaforge.agents.catalog import capabilities

        return page(capabilities(self), offset, limit)

    @operation(
        "framework.capability.describe",
        "core",
        "READ",
        "CapabilityCatalog",
        "docs/ai/INTEGRATION.md",
    )
    def capability_describe(self, identifier: str) -> dict[str, Any]:
        """Describe one exact capability, including schemas, service and session availability."""
        from lambdaforge.agents.catalog import capabilities

        matches = [row for row in capabilities(self) if row["identifier"] == identifier]
        if len(matches) != 1:
            raise KeyError("Unknown capability identifier.")
        return matches[0]

    @operation(
        "framework.capability.search", "core", "READ", "CapabilityCatalog", "docs/ai/INTEGRATION.md"
    )
    def capability_search(self, query: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Search descriptions/identifiers without sending every tool to the model's context."""
        from lambdaforge.agents.catalog import capabilities

        if len(query) > 500:
            raise ValueError("Capability query is too long.")
        terms = query.casefold().split()
        return page(
            [
                row
                for row in capabilities(self)
                if all(
                    term in (row["identifier"] + " " + row["description"]).casefold()
                    for term in terms
                )
            ],
            offset,
            limit,
        )

    @operation("project.structure", "core", "READ", "ProjectScope", "docs/MANUAL.md")
    def project_structure(self, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Page source/config/document paths without secrets, environments or nested projects."""
        return page(
            list(self.scope.files(suffixes=(".py", ".yaml", ".yml", ".md", ".toml"))), offset, limit
        )

    @operation("project.work_classes", "work", "READ", "Python AST", "docs/MANUAL.md")
    def find_work_classes(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Static class candidates; native validation must confirm inheritance/importability."""
        classes = []
        for relative in self.scope.files(suffixes=(".py",)):
            source = self.scope.file(relative, limit=256 * 1024).decode("utf-8")
            try:
                tree = ast.parse(source, filename=relative)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and any(
                    (isinstance(base, ast.Name) and base.id == "Work")
                    or (isinstance(base, ast.Attribute) and base.attr == "Work")
                    for base in node.bases
                ):
                    classes.append(
                        {
                            "file": relative,
                            "class": node.name,
                            "line": node.lineno,
                            "validated": False,
                            "documentation": ast.get_docstring(node),
                        }
                    )
        return page(classes, offset, limit)

    @operation("project.configs", "work", "READ", "ProjectScope", "docs/MANUAL.md")
    def list_project_configs(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """List YAML paths without executing Work imports or treating every YAML as validated."""
        return page(list(self.scope.files(suffixes=(".yaml", ".yml"))), offset, limit)

    @operation("work.config.inspect", "work", "READ", "YAML safe loader", "docs/MANUAL.md")
    def inspect_config(self, config: str) -> dict[str, Any]:
        """Read authored YAML as untrusted data only; this is not scientific validation."""
        if Path(config).suffix not in {".yaml", ".yml"}:
            raise ValueError("Configuration inspection accepts YAML only.")
        value = yaml.safe_load(self.scope.file(config, limit=256 * 1024))
        if not isinstance(value, dict):
            raise ValueError("Configuration must contain a mapping.")
        return {"config": config, "authored": value, "validated": False}

    @operation(
        "work.validate",
        "work",
        "PLAN",
        "WorkConfig.validate_file",
        "docs/MANUAL.md",
        project_code=True,
    )
    def validate_work(self, config: str) -> dict[str, Any]:
        """Native validation imports trusted project Python, but launches no Work."""
        from lambdaforge.work.config import WorkConfig

        self.scope.file(config, limit=512 * 1024)
        return WorkConfig.validate_file(self.scope.path(config)).to_dict()

    @operation(
        "work.explain",
        "work",
        "PLAN",
        "WorkConfig.explanation",
        "docs/MANUAL.md",
        project_code=True,
    )
    def explain_work(self, config: str) -> dict[str, Any]:
        """Explain native parameters/defaults/resources with the same WorkConfig as CLI."""
        return self._config(config).explanation()

    @operation(
        "work.resolve",
        "work",
        "PLAN",
        "WorkConfig.resolved_configuration",
        "docs/MANUAL.md",
        project_code=True,
    )
    def resolve_work_config(self, config: str) -> dict[str, Any]:
        """Resolve native scientific policies without submitting, reserving or editing YAML."""
        return self._config(config).resolved_configuration()

    @operation(
        "execution.preview",
        "work",
        "PLAN",
        "WorkConfig.preflight",
        "docs/MANUAL.md",
        project_code=True,
    )
    def preview_execution(self, config: str) -> dict[str, Any]:
        """Local scientific/resource preflight, not remote staging or admission approval."""
        return self._config(config).preflight()

    @operation("framework.api.work", "core", "READ", "Work public signature", "docs/MANUAL.md")
    def api_reference(self) -> dict[str, Any]:
        """Installed Work methods/properties/docstrings; no consumer import or guessed APIs."""
        from lambdaforge.work.Work import Work

        entries = []
        for name, member in inspect.getmembers(Work):
            if name.startswith("_"):
                continue
            entries.append(
                {
                    "name": name,
                    "signature": str(inspect.signature(member)) if callable(member) else None,
                    "documentation": inspect.getdoc(member),
                }
            )
        return {"api": "lambdaforge.Work", "version": VERSION, "members": entries}

    @operation(
        "framework.api.study", "core", "READ", "Native Study design contracts", "docs/MANUAL.md"
    )
    def study_api_reference(self) -> dict[str, Any]:
        """Installed normalization contracts/signatures; not executable alternate Study objects."""
        from lambdaforge.work.design import EvidencePlan, ExecutionPolicy, StudyDesign

        return {
            "version": VERSION,
            "execution_model": "Work.run; search/sweep/seeds normalize into these native contracts",
            "contracts": [
                {
                    "name": cls.__name__,
                    "signature": str(inspect.signature(cls)),
                    "documentation": inspect.getdoc(cls),
                }
                for cls in (StudyDesign, EvidencePlan, ExecutionPolicy)
            ],
        }

    @operation("framework.api.models", "core", "READ", "Installed model exports", "docs/MANUAL.md")
    def list_model_catalog(
        self, query: str = "", offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """Page installed neural exports statically, without importing Torch/optional backends."""
        if len(query) > 500:
            raise ValueError("Model catalog query is too long.")
        source = files("lambdaforge").joinpath("nn/models/__init__.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        exports = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                exports.update(ast.literal_eval(node.value))
        entries = [
            {
                "name": item.asname or item.name,
                "module": node.module,
                "public_api": "lambdaforge.nn.models." + (item.asname or item.name),
                "signature_loaded": False,
                "tensor_contract": "inspect the installed component",
            }
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for item in node.names
            if (item.asname or item.name) in exports
            and query.casefold() in (item.asname or item.name).casefold()
        ]
        return page(sorted(entries, key=lambda row: row["name"]), offset, limit)

    @operation(
        "framework.api.model", "core", "READ", "Installed model declarations", "docs/MANUAL.md"
    )
    def get_model_contract(self, name: str) -> dict[str, Any]:
        """Exact constructor/forward/docstrings and declared tensor semantics, without Torch."""
        from lambdaforge.agents.model_catalog import model_contract

        return model_contract(name)

    @operation("framework.recipes", "work", "READ", "Installed examples", "docs/ai/INTEGRATION.md")
    def get_recipe(self, name: str) -> dict[str, Any]:
        """One packaged native example, never arbitrary project files or executable construction."""
        from lambdaforge.agents.integration import example_assets

        examples = example_assets()
        if name not in examples:
            raise ValueError("Unknown recipe; inspect list_recipes.")
        return {
            "name": name,
            "source": "installed example",
            "content": examples[name],
            "executed": False,
        }

    @operation(
        "framework.recipes.list", "work", "READ", "Installed examples", "docs/ai/INTEGRATION.md"
    )
    def list_recipes(self) -> dict[str, Any]:
        """Discover packaged native Work/Study/input examples without writing them."""
        from lambdaforge.agents.integration import example_assets

        return {
            "items": [
                {"name": name, "bytes": len(value.encode("utf-8"))}
                for name, value in sorted(example_assets().items())
            ]
        }

    @operation(
        "jobs.list", "operations", "READ", "JobService.list", "docs/OPERATIONAL_RELIABILITY.md"
    )
    def list_jobs(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Persisted project Job summaries, with native observation age; no provider refresh."""
        from lambdaforge.controlplane.OverviewService import OverviewService

        return page(
            [
                OverviewService._job_overview(record)
                for record in self._job_service().list(refresh=False)
            ],
            offset,
            limit,
        )

    @operation("works.list", "study", "READ", "aggregate_research_work", "docs/MANUAL.md")
    def list_works(
        self, offset: int = 0, limit: int = 20, studies_only: bool = False
    ) -> dict[str, Any]:
        """Compact semantic Works/Studies, never all epochs; imported evidence remains read-only."""
        from lambdaforge.controlplane.ResearchWork import aggregate_research_work
        from lambdaforge.work.ImportedStudy import ImportedStudy

        values = [
            record.to_dict(study_detail="overview")
            for record in aggregate_research_work(self._job_service().list(refresh=False))
        ]
        values.extend(ImportedStudy.rows(self._results().root, include_works=True))
        if studies_only:
            values = [
                record for record in values if record.get("study_expected") or record.get("study")
            ]
        return page(values, offset, limit)

    @operation(
        "jobs.health",
        "operations",
        "READ",
        "OperationalHealth.describe",
        "docs/OPERATIONAL_RELIABILITY.md",
    )
    def get_execution_health(self, job_id: str) -> dict[str, Any]:
        """Separate native operational health from scientific outcome; no forged freshness."""
        from lambdaforge.controlplane.OperationalHealth import OperationalHealth

        record = self._job_service().store.get(job_id)
        return {
            "job_id": record.job_id,
            "state": record.state.value,
            "operational": OperationalHealth.describe(record),
            "live_refresh": False,
        }

    @operation("jobs.get", "operations", "READ", "JobStore.get", "docs/OPERATIONAL_RELIABILITY.md")
    def get_job(self, job_id: str) -> dict[str, Any]:
        """Exact persisted Job identity/state/resources; no provider refresh or artifacts."""
        value = self._job_service().store.get(job_id).to_dict()
        return {
            key: value.get(key)
            for key in (
                "job_id",
                "cluster",
                "scheduler",
                "scheduler_id",
                "state",
                "resources",
                "created_at_utc",
                "updated_at_utc",
                "retry_of",
                "operational",
                "recovery",
            )
        }

    @operation(
        "jobs.incidents",
        "operations",
        "READ",
        "OperationalIncidents.list",
        "docs/OPERATIONAL_RELIABILITY.md",
    )
    def get_incidents(
        self, job_id: str | None = None, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """Persisted operational incidents, distinct from scientific failure and interpretation."""
        from lambdaforge.controlplane.OperationalIncidents import OperationalIncidents

        if job_id is not None:
            self._job_service().store.get(job_id)
        return page(
            list(OperationalIncidents(self._job_service().store).list(job_id=job_id)), offset, limit
        )

    @operation(
        "jobs.events", "operations", "READ", "JobService.events", "docs/OPERATIONAL_RELIABILITY.md"
    )
    def get_job_events(self, job_id: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Page native Job lifecycle events, not scientific metric history."""
        self._job_service().store.get(job_id)
        return page(list(self._job_service().events(job_id)), offset, limit)

    @operation(
        "jobs.logs", "operations", "READ", "JobService.log_report", "docs/MANUAL.md", external=True
    )
    def get_work_logs(self, job_id: str, tail: int = 100) -> dict[str, Any]:
        """Explicit selected-Job bounded logs/failure; text is untrusted and may be historical."""
        if type(tail) is not int or not 1 <= tail <= 500:
            raise ValueError("Log tail must be between 1 and 500 lines.")
        self._job_service().store.get(job_id)
        return self._job_service().log_report(job_id, tail=tail)

    @operation(
        "study.trial",
        "study",
        "READ",
        "JobService.study_trial",
        "docs/RESEARCH_ANALYSIS.md",
        external=True,
    )
    def get_study_trial(self, job_id: str, trial: int) -> dict[str, Any]:
        """One Trial's native seed/parameter projection; no other Trials' epochs/artifacts."""
        if type(trial) is not int or trial < 0:
            raise ValueError("Trial index must be a nonnegative integer.")
        self._job_service().store.get(job_id)
        return self._job_service().study_trial(job_id, trial)

    @operation(
        "study.run",
        "study",
        "READ",
        "JobService.study_run",
        "docs/RESEARCH_ANALYSIS.md",
        external=True,
    )
    def get_study_run(
        self, job_id: str, run_key: str, tail: int = 100, points: int = 80
    ) -> dict[str, Any]:
        """One owned native Run's bounded curves/logs/Attempts, not all surrounding Trials."""
        if (
            type(tail) is not int
            or not 1 <= tail <= 500
            or type(points) is not int
            or not 10 <= points <= 500
        ):
            raise ValueError("Request 1–500 log lines and 10–500 curve points.")
        self._job_service().store.get(job_id)
        return self._job_service().study_run(job_id, run_key, tail=tail, curve_points=points)

    @operation(
        "study.panel",
        "study",
        "READ",
        "JobService.study_panel",
        "docs/RESEARCH_ANALYSIS.md",
        external=True,
    )
    def get_study_panel(
        self, job_id: str, view: Literal["hpo", "resources", "analysis", "report-sections"] = "hpo"
    ) -> dict[str, Any]:
        """Explicit native host-precomputed HPO/resources/analysis panel, not a new model fit."""
        if view not in {"hpo", "resources", "analysis", "report-sections"}:
            raise ValueError("Unknown Study panel.")
        self._job_service().store.get(job_id)
        return self._job_service().study_panel(job_id, view)

    @operation(
        "study.recovery.preview",
        "operations",
        "PLAN",
        "JobService.retry_preview",
        "docs/OPERATIONAL_RELIABILITY.md",
        external=True,
        project_code=True,
    )
    def prepare_recovery(self, job_id: str) -> dict[str, Any]:
        """Native recovery preview; never launch, reset budgets or recover imported evidence."""
        self._job_service().store.get(job_id)
        return self._job_service().retry_preview(job_id)

    @operation(
        "jobs.recovery.history",
        "operations",
        "READ",
        "JobService.recovery_history",
        "docs/OPERATIONAL_RELIABILITY.md",
    )
    def get_recovery_history(self, job_id: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Paged native recovery audit; no provider calls or checkpoint payloads."""
        self._job_service().store.get(job_id)
        return self._job_service().recovery_history(job_id, offset=offset, limit=limit)

    @operation("results.list", "results", "READ", "ResultStore.catalog", "docs/WORK_RESULTS.md")
    def list_results(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Compact local/imported result catalog, without artifact bytes or consumer imports."""
        values = [
            {k: v for k, v in row.items() if k != "_manifest_path"}
            for row in self._results().catalog()
        ]
        return page(values, offset, limit)

    @operation("results.view", "results", "READ", "ResultStore.view", "docs/WORK_RESULTS.md")
    def get_result(
        self,
        execution_id: str,
        view: Literal["overview", "metrics", "outputs", "logs", "provenance"] = "overview",
        run_id: str | None = None,
        attempt: int | None = None,
        points: int = 100,
    ) -> dict[str, Any]:
        """Exact Execution/Run/Attempt evidence; missing values stay absent, never zero."""
        if not execution_id.startswith("execution-") or not 1 <= points <= 500:
            raise ValueError("Use an exact Execution ID and 1–500 curve points.")
        if view not in {"overview", "metrics", "outputs", "logs", "provenance"}:
            raise ValueError("Unknown result view.")
        return self._results().view(
            execution_id, view=view, run_id=run_id, attempt=attempt, points=points, tail=100
        )

    @operation(
        "results.reference", "results", "PLAN", "ResultStore.reference", "docs/WORK_RESULTS.md"
    )
    def get_historical_reference(
        self, execution_id: str, run_id: str | None = None, attempt: int | None = None
    ) -> dict[str, Any]:
        """Pin native historical evidence; an import never becomes an executable recovery."""
        if not execution_id.startswith("execution-"):
            raise ValueError("Use an exact Execution ID.")
        return self._results().reference(execution_id, run=run_id, attempt=attempt)

    @operation(
        "results.artifact.preview",
        "results",
        "READ",
        "ResultStore.output_preview",
        "docs/WORK_RESULTS.md",
    )
    def preview_result_artifact(
        self, execution_id: str, name: str, run_id: str | None = None, attempt: int | None = None
    ) -> dict[str, Any]:
        """Explicit verified UTF-8 preview of one registered owned artifact, at most 64 KiB."""
        if not execution_id.startswith("execution-"):
            raise ValueError("Use an exact Execution ID.")
        return self._results().output_preview(execution_id, name, run_id=run_id, attempt=attempt)

    @operation("results.compare", "results", "READ", "ResultStore.compare", "docs/WORK_RESULTS.md")
    def compare_results(
        self,
        execution_ids: list[str],
        metric: str | None = None,
        mode: Literal["min", "max"] = "max",
    ) -> dict[str, Any]:
        """Compare 2–8 exact local/imported Executions using native Run evidence semantics."""
        if not 2 <= len(execution_ids) <= 8 or any(
            not item.startswith("execution-") for item in execution_ids
        ):
            raise ValueError("Compare 2–8 exact Execution IDs.")
        return self._results().compare(tuple(execution_ids), metric=metric, mode=mode)

    @operation("products.list", "products", "READ", "ProductRegistry.list", "docs/PRODUCTS.md")
    def list_products(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Sealed product metadata only; do not read/download model members."""
        page([], offset, limit)
        values = self._products().list(offset=offset, limit=limit + 1)
        return {
            "items": [
                {
                    "name": row.name,
                    "kind": row.kind,
                    "contract": row.contract.to_dict(),
                    "content_id": row.content_id,
                    "scientific_id": row.scientific_id,
                    "artifact_count": len(row.artifacts),
                }
                for row in values[:limit]
            ],
            "next_offset": offset + limit if len(values) > limit else None,
        }

    @operation("products.get", "products", "READ", "ProductRegistry.show", "docs/PRODUCTS.md")
    def get_product(self, content_id: str) -> dict[str, Any]:
        """Exact sealed product identity/contract and artifact metadata, not arbitrary paths."""
        if not content_id.startswith("sha256:"):
            raise ValueError("Use an exact sealed product content ID.")
        return {"product": self._products().show(content_id).to_dict()}

    @operation("products.evidence", "products", "READ", "ProductRegistry", "docs/PRODUCTS.md")
    def get_product_evidence(
        self,
        content_id: str,
        view: Literal["contract", "members", "provenance", "consumers"] = "contract",
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Sealed contract/members or paged actual provenance/consumption, without model bytes."""
        if not content_id.startswith("sha256:"):
            raise ValueError("Use an exact sealed product content ID.")
        page([], offset, limit)
        registry = self._products()
        product = registry.show(content_id)
        if view == "contract":
            return {"content_id": product.content_id, "contract": product.contract.to_dict()}
        if view == "members":
            return page([member.to_dict() for member in product.artifacts], offset, limit)
        if view not in {"provenance", "consumers"}:
            raise ValueError("Unknown product evidence view.")
        rows = getattr(registry, view)(content_id, offset=offset, limit=limit + 1)
        return {
            "items": list(rows[:limit]),
            "offset": offset,
            "next_offset": offset + limit if len(rows) > limit else None,
        }

    @operation(
        "datasets.list", "datasets", "READ", "DatasetService.list", "docs/DATASET_RECONSTRUCTION.md"
    )
    def list_datasets(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Local registry summaries/placements, without asset scans or remote transfers."""
        return page(
            [
                {
                    "name": row.name,
                    "version": row.version,
                    "content_id": row.dataset_id,
                    "sample_count": row.sample_count,
                    "splits": row.splits,
                    "placements": [item.to_dict() for item in row.placements],
                }
                for row in self._datasets().list()
            ],
            offset,
            limit,
        )

    @operation(
        "datasets.get",
        "datasets",
        "READ",
        "DatasetRegistry.records",
        "docs/DATASET_RECONSTRUCTION.md",
    )
    def get_dataset(self, content_id: str) -> dict[str, Any]:
        """Exact cached dataset counts/partitions/contract; no member index or asset scan."""
        record = self._dataset_record(content_id)
        return {
            "name": record.name,
            "version": record.version,
            "content_id": record.dataset_id,
            "sample_count": record.sample_count,
            "splits": record.splits,
            "partitions": record.partitions,
            "target_schema": record.target_schema,
            "scientific_contract": record.metadata.get("lambdaforge_science"),
            "observation": "controller registry, not fresh physical verification",
        }

    @operation(
        "datasets.placements",
        "datasets",
        "READ",
        "DatasetRegistry.records",
        "docs/DATASET_RECONSTRUCTION.md",
    )
    def get_dataset_placements(
        self, content_id: str, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """Known placements of the exact content ID, without inferring physical availability."""
        page([], offset, limit)
        return page(
            [placement.to_dict() for placement in self._dataset_record(content_id).placements],
            offset,
            limit,
        )

    @operation(
        "datasets.members",
        "datasets",
        "READ",
        "DatasetService.members",
        "docs/DATASET_RECONSTRUCTION.md",
        external=True,
    )
    def get_dataset_members(
        self, content_id: str, cluster: str, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """Read a selected-placement member page, without asset bytes or guessed identity."""
        page([], offset, limit)
        record = self._dataset_record(content_id)
        return self._datasets().members(record.key, cluster=cluster, offset=offset, limit=limit)

    @operation("clusters.list", "clusters", "READ", "ClusterCatalog", "docs/MANUAL.md")
    def list_clusters(self) -> dict[str, Any]:
        """Configured names/providers only; no resources/credential lookup/provider contact."""
        catalog = self._clusters()
        return {
            "items": [
                {
                    "name": name,
                    "scheduler": catalog.definition(name).scheduler,
                    "transport": catalog.definition(name).transport,
                }
                for name in catalog.names()
            ]
        }

    @operation("clusters.get", "clusters", "READ", "ClusterCatalog.inspect", "docs/MANUAL.md")
    def get_cluster(self, name: str) -> dict[str, Any]:
        """Redacted configured/effective profile, not credential values or live GPU grants."""
        return self._clusters().inspect(name)

    @operation("fleet.get", "clusters", "READ", "ClusterCatalog.fleet", "docs/MANUAL.md")
    def get_fleet_info(self, name: str) -> dict[str, Any]:
        """Authored Fleet/member limits only; does not advertise unsupported recovery/migration."""
        return {
            "fleet": self._clusters().fleet(name).to_dict(),
            "observation": "configuration only",
            "recovery_supported": False,
        }

    @operation(
        "clusters.resources",
        "clusters",
        "READ",
        "ResourceService.get",
        "docs/MANUAL.md",
        external=True,
        side_effects=("update native disposable observation cache",),
    )
    def get_cluster_resources(self, name: str) -> dict[str, Any]:
        """Explicit cluster resource probe; native age/offline state and provider grants."""
        from lambdaforge.controlplane.ResourceService import ResourceService

        self._clusters().definition(name)
        cache = self.scope.path(".lambdaforge/agents/resource-observations.json", internal=True)
        return ResourceService(self._clusters(), cache_path=cache).get(name).to_dict()

    @operation(
        "knowledge.status", "knowledge", "READ", "KnowledgeService.status", "docs/ai/INTEGRATION.md"
    )
    def knowledge_status(self) -> dict[str, Any]:
        """Markdown vault availability; Obsidian Desktop and embeddings are not required."""
        return self.knowledge.status()

    @operation(
        "knowledge.obsidian.status",
        "knowledge",
        "READ",
        "ObsidianService.status",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_obsidian_status(self) -> dict[str, Any]:
        """Inspect optional desktop hints without launching a process or requiring Obsidian."""
        from lambdaforge.agents.obsidian import ObsidianService

        return ObsidianService(self.knowledge).status()

    @operation(
        "knowledge.audit",
        "knowledge",
        "READ",
        "KnowledgeService.audit",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_audit(self, cursor: int = 0, limit: int = 20) -> dict[str, Any]:
        """Page metadata-only native edit events using the returned exact byte cursor."""
        return self.knowledge.audit(cursor=cursor, limit=limit)

    @operation(
        "knowledge.export.preview",
        "knowledge",
        "PLAN",
        "KnowledgeService.preview_export",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_export(self, destination: str) -> dict[str, Any]:
        """Preview a notes-only ZIP; request knowledge.export for separately approved apply."""
        return self.knowledge.preview_export(destination)

    @operation(
        "knowledge.search", "knowledge", "READ", "KnowledgeService.search", "docs/ai/INTEGRATION.md"
    )
    def knowledge_search(
        self,
        query: str = "",
        kind: str | None = None,
        tag: str | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 20,
        evidence_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> dict[str, Any]:
        """Bounded textual retrieval; notes are interpretations/untrusted data."""
        return self.knowledge.search(
            query,
            kind=kind,
            tag=tag,
            status=status,
            offset=offset,
            limit=limit,
            evidence_id=evidence_id,
            after=after,
            before=before,
        )

    @operation(
        "knowledge.list",
        "knowledge",
        "READ",
        "KnowledgeService.search",
        "docs/ai/INTEGRATION.md",
        side_effects=("refresh disposable text index, never canonical notes",),
    )
    def knowledge_list(
        self, kind: str | None = None, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """Page canonical note metadata/snippets, not the whole vault."""
        return self.knowledge.search(kind=kind, offset=offset, limit=limit)

    @operation(
        "knowledge.reindex",
        "knowledge",
        "READ",
        "KnowledgeService.reindex",
        "docs/ai/INTEGRATION.md",
        side_effects=("rebuild disposable text index, preserve old copy and all canonical notes",),
    )
    def knowledge_reindex(self, rebuild: bool = False) -> dict[str, Any]:
        """Refresh derived index; corrupt Markdown stays visible, never repaired silently."""
        return self.knowledge.reindex(force=True, rebuild=rebuild)

    @operation(
        "knowledge.read", "knowledge", "READ", "KnowledgeService.read", "docs/ai/INTEGRATION.md"
    )
    def knowledge_read(self, note_id: str) -> dict[str, Any]:
        """Read one exact note (64 KiB maximum) with revision, provenance and evidence refs."""
        return self.knowledge.read(note_id)

    @operation(
        "knowledge.backlinks",
        "knowledge",
        "READ",
        "KnowledgeService.backlinks",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_backlinks(self, note_id: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Find exact note backlinks without executing Markdown or following arbitrary paths."""
        return self.knowledge.backlinks(note_id, offset=offset, limit=limit)

    @operation(
        "knowledge.related",
        "knowledge",
        "READ",
        "KnowledgeService.related",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_related(self, note_id: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Find explicit links/tags/exact evidence, not semantic or scientific confidence."""
        return self.knowledge.related(note_id, offset=offset, limit=limit)

    @operation(
        "knowledge.references.verify",
        "knowledge",
        "READ",
        "KnowledgeService.verify_references",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_verify_references(self, note_id: str) -> dict[str, Any]:
        """Check local exact references, never the interpretation's scientific validity."""

        def resolve(reference: dict[str, Any]) -> dict[str, Any]:
            kind = reference["kind"]
            if kind == "product":
                product = self._products().show(reference["id"])
                return {
                    "status": "registered",
                    "content_id": product.content_id,
                    "bytes_verified": False,
                }
            if kind == "dataset":
                matches = [
                    record
                    for record in self._datasets().registry.records()
                    if record.dataset_id == reference["id"]
                ]
                if not matches:
                    raise KeyError("Dataset is not registered locally.")
                return {
                    "status": "registered",
                    "content_id": reference["id"],
                    "placements": len(matches[0].placements),
                    "bytes_verified": False,
                }
            self._results().reference(
                reference["id"], run=reference.get("run"), attempt=reference.get("attempt")
            )
            return {
                "status": "resolved",
                "source": "native local ResultStore",
                "bytes_verified": False,
            }

        return self.knowledge.verify_references(note_id, resolve)

    @operation(
        "knowledge.hypotheses",
        "knowledge",
        "READ",
        "KnowledgeService.search",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_list_hypotheses(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Page proposed hypotheses without mistaking them for verified scientific findings."""
        return self.knowledge.search(kind="hypothesis", offset=offset, limit=limit)

    @operation(
        "knowledge.decisions",
        "knowledge",
        "READ",
        "KnowledgeService.search",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_list_decisions(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Page interpretations/decisions, distinct from sealed StudyDecision products."""
        return self.knowledge.search(kind="decision", offset=offset, limit=limit)

    @operation(
        "knowledge.questions",
        "knowledge",
        "READ",
        "KnowledgeService.search",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_list_open_questions(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Page draft/proposed/open research questions, not closed interpretations."""
        return self.knowledge.search(
            kind="research-question",
            statuses=["draft", "proposed", "open"],
            offset=offset,
            limit=limit,
        )

    @operation(
        "knowledge.templates",
        "knowledge",
        "READ",
        "KnowledgeService.template",
        "docs/ai/INTEGRATION.md",
    )
    def knowledge_templates(self, kind: str | None = None) -> dict[str, Any]:
        """List supported note types or return a purpose-specific Markdown template."""
        return (
            {"kinds": list(KINDS)}
            if kind is None
            else {"kind": kind, "body": self.knowledge.template(kind)}
        )


__all__ = ["AgentServices"]
