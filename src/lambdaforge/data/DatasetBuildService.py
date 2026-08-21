"""Plan and execute dataset recipes over the existing Task/Workflow lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.jobs import JobHandle
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.data.build_models import DatasetBuildPlan, DatasetBuildResult, DatasetStagePlan
from lambdaforge.data.DatasetPublisher import DatasetPublisher
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.recipe_config import DatasetRecipeConfig
from lambdaforge.execution.ConfigurationResourceResolver import ConfigurationResourceResolver
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskExecutionPlan import TaskPlanAction
from lambdaforge.tasks.TaskRun import TaskRun
from lambdaforge.workflows.WorkflowRunner import WorkflowRunner


class DatasetBuildService:
    """Reuse verified stage tasks, retain failed-build evidence and publish exactly once."""

    _REFERENCE = re.compile(r"^\$\{nodes\.([^.]+)\.(outputs|metrics|artifacts)\.([^{}]+)\}$")

    def __init__(
        self,
        registry: DatasetRegistry | None = None,
        jobs: JobService | None = None,
    ) -> None:
        self.registry = registry or DatasetRegistry()
        self.jobs = jobs or JobService()

    def plan(
        self,
        recipe: DatasetRecipeConfig,
        *,
        cluster: str = "local",
        force: bool = False,
        force_stages: Sequence[str] = (),
    ) -> DatasetBuildPlan:
        """Plan REUSE/EXECUTE from Task fingerprints and propagate forced invalidation."""
        names = {stage.name for stage in recipe.stages}
        unknown = set(force_stages) - names
        if unknown:
            raise KeyError(f"Unknown forced dataset stages: {sorted(unknown)}.")
        forced = recipe.downstream(names if force else set(force_stages))
        remote_cache_unobserved = (
            cluster != "local" and os.environ.get("LAMBDAFORGE_CLUSTER") != cluster
        )
        previous = self._previous(recipe)
        previous_nodes = previous.get("stages", {}) if isinstance(previous, Mapping) else {}
        decisions: list[DatasetStagePlan] = []
        for stage in recipe.stages:
            if stage.name in forced or stage.reuse == "never":
                reason = (
                    "forced with downstream invalidation" if stage.name in forced else "reuse=never"
                )
                decisions.append(
                    DatasetStagePlan(stage.name, "EXECUTE", reason, required=stage.required)
                )
                continue
            if remote_cache_unobserved:
                decisions.append(
                    DatasetStagePlan(
                        stage.name,
                        "MISSING",
                        "remote stage-cache state was not mutated or assumed during this "
                        "controller-side plan; the durable target worker rechecks exact "
                        "Task fingerprints before execution",
                        required=stage.required,
                    )
                )
                continue
            try:
                data, source, resolution = self._materialize_stage(
                    stage, previous_nodes, recipe.source_dir
                )
                data["output_root"] = str(self._stage_cache(recipe) / stage.name)
                task = TaskConfig(data, source=source, resolution=resolution)
                task_plan = TaskRun(task).inspect()
                action = "REUSE" if task_plan.action is TaskPlanAction.SKIP else "EXECUTE"
                decisions.append(
                    DatasetStagePlan(
                        stage.name,
                        action,
                        task_plan.reason,
                        task.fingerprint,
                        stage.required,
                    )
                )
            except Exception as error:
                decisions.append(
                    DatasetStagePlan(
                        stage.name,
                        "EXECUTE",
                        "stage inputs require execution/rebinding: "
                        f"{error.__class__.__name__}: {error}",
                        required=stage.required,
                    )
                )
        try:
            record = self.registry.get(recipe.selector)
        except KeyError:
            record = None
        no_stage_work = all(stage.action == "REUSE" for stage in decisions)
        publish_action = "NOOP" if record is not None and no_stage_work else "PUBLISH"
        publish_reason = (
            "immutable version and every stage are already verified"
            if publish_action == "NOOP"
            else "validate selected stage output and atomically publish the immutable version"
        )
        return DatasetBuildPlan(
            recipe.selector,
            recipe.fingerprint,
            cluster,
            tuple(decisions),
            publish_action,
            publish_reason,
            resources=ConfigurationResourceResolver.resolve_dataset(recipe).to_dict(),
        )

    def build(
        self,
        recipe: DatasetRecipeConfig,
        *,
        force: bool = False,
        force_stages: Sequence[str] = (),
    ) -> DatasetBuildResult:
        """Execute locally; completed Task stages remain independently reusable after failure."""
        plan = self.plan(recipe, force=force, force_stages=force_stages)
        forced = frozenset(
            stage.stage
            for stage in plan.stages
            if stage.action == "EXECUTE"
            and (
                force
                or stage.stage in recipe.downstream(set(force_stages))
                or next(item for item in recipe.stages if item.name == stage.stage).reuse == "never"
            )
        )
        stage_cache = self._stage_cache(recipe)
        workflow_result = WorkflowRunner().run(
            recipe.workflow(),
            force_nodes=forced,
            node_output_root=stage_cache,
        )
        if not hasattr(workflow_result, "nodes"):
            raise RuntimeError("Dataset build unexpectedly returned a workflow plan.")
        nodes = workflow_result.nodes
        required_incomplete = [
            stage.name
            for stage in recipe.stages
            if stage.required and nodes.get(stage.name, {}).get("status") != "ok"
        ]
        provisional_id = self._build_id(recipe, nodes)
        if required_incomplete:
            failed = [
                stage.name
                for stage in recipe.stages
                if nodes.get(stage.name, {}).get("status") == "failed"
            ]
            blocked = [
                stage.name
                for stage in recipe.stages
                if nodes.get(stage.name, {}).get("status") == "blocked"
            ]
            root = failed[0] if failed else required_incomplete[0]
            stage_error = nodes.get(root, {}).get("error", {})
            cause = (
                str(stage_error.get("message", "stage failed"))
                if isinstance(stage_error, Mapping)
                else str(stage_error or "stage failed")
            )
            error = f"Dataset stage {root!r} failed: {cause}"
            if blocked:
                error += f" Downstream blocked: {', '.join(blocked)}."
            result = DatasetBuildResult(
                provisional_id,
                recipe.selector,
                "failed",
                nodes,
                error=error,
            )
            self._write_result(recipe, result)
            return result
        publish_stage = str(recipe.publish["from"])
        if nodes.get(publish_stage, {}).get("status") != "ok":
            result = DatasetBuildResult(
                provisional_id,
                recipe.selector,
                "failed",
                nodes,
                error=f"Publish stage {publish_stage!r} did not complete successfully.",
            )
            self._write_result(recipe, result)
            return result
        stage_provenance = {
            name: {
                "status": value.get("status"),
                "config_fingerprint": value.get("result", {}).get("config_fingerprint")
                if isinstance(value.get("result"), Mapping)
                else None,
            }
            for name, value in nodes.items()
        }
        identity_provenance = {
            "recipe_fingerprint": recipe.fingerprint,
            "stages": stage_provenance,
            "inputs": list(recipe.dataset.get("inputs", ())),
        }
        provenance = {
            "recipe": str(recipe.source) if recipe.source is not None else None,
            "config": str(recipe.source) if recipe.source is not None else None,
            **identity_provenance,
            "identity": identity_provenance,
        }
        record = DatasetPublisher(self.registry).publish(
            recipe,
            nodes[publish_stage],
            build_provenance=provenance,
            cluster=os.environ.get("LAMBDAFORGE_CLUSTER", "local"),
        )
        result = DatasetBuildResult(
            record.build_id or provisional_id,
            recipe.selector,
            "ok",
            nodes,
            record,
        )
        self._write_result(recipe, result)
        return result

    def submit(
        self,
        recipe: DatasetRecipeConfig,
        *,
        cluster: str,
        force: bool = False,
        force_stages: Sequence[str] = (),
        dry_run: bool = False,
        wait_for_submit: bool = True,
        resources: ResourceRequest | None = None,
    ) -> JobHandle:
        """Submit a durable dataset-build unit; the worker reuses the same service locally."""
        if recipe.source is None:
            raise ValueError("Durable dataset builds require a recipe YAML path.")
        arguments: list[str] = []
        if force:
            arguments.append("--force")
        for stage in force_stages:
            arguments.extend(("--force-stage", stage))
        request = resources or ConfigurationResourceResolver.resolve_dataset(recipe)
        if cluster != "local" and not dry_run and not wait_for_submit:
            from lambdaforge.controlplane.SubmissionService import SubmissionService

            return SubmissionService(self.jobs.catalog, self.jobs).enqueue(
                recipe.source,
                cluster=cluster,
                resources=request,
                run_arguments=arguments,
            )

        from lambdaforge.controlplane.ControlPlane import ControlPlane

        handle, _ = ControlPlane(self.jobs.catalog, jobs=self.jobs).submit(
            recipe.source,
            cluster=cluster,
            resources=request,
            dry_run=dry_run,
            run_arguments=arguments,
        )
        return handle

    def _previous(self, recipe: DatasetRecipeConfig) -> dict[str, Any]:
        path = self._result_path(recipe)
        if not path.is_file():
            build_root = recipe.output_root / "builds" / recipe.name / recipe.version
            candidates = sorted(
                build_root.glob("*/build-result.json"),
                key=lambda candidate: candidate.stat().st_mtime_ns,
                reverse=True,
            )
            path = candidates[0] if candidates else path
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _materialize_stage(
        self,
        stage: Any,
        outcomes: Mapping[str, Any],
        source_dir: Path,
    ) -> tuple[Any, Any, Any]:
        from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
        from lambdaforge.configuration.ResolvedConfiguration import ResolvedConfiguration

        if isinstance(stage.task, Path):
            resolution = ConfigurationComposer().resolve(stage.task)
            data = copy.deepcopy(dict(resolution.values))
            source = stage.task
        else:
            resolution = None
            data = dict(stage.task)
            source = source_dir / ".lambdaforge-embedded-dataset-stage.yaml"
        for path, value in stage.bindings.items():
            ExperimentConfig.set_value(data, str(path), self._resolve(value, outcomes))
        if resolution is not None and stage.bindings:
            resolution = ResolvedConfiguration(
                data,
                {
                    **dict(resolution.provenance),
                    **{str(path): f"dataset binding:{stage.name}" for path in stage.bindings},
                },
                resolution.sources,
            )
        return data, source, resolution

    @staticmethod
    def _stage_cache(recipe: DatasetRecipeConfig) -> Path:
        return Path(
            os.environ.get(
                "LAMBDAFORGE_STAGE_CACHE_ROOT",
                str(recipe.output_root / ".stage-cache"),
            )
        ).resolve()

    def _resolve(self, value: Any, outcomes: Mapping[str, Any]) -> Any:
        if isinstance(value, Mapping):
            return {key: self._resolve(item, outcomes) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve(item, outcomes) for item in value]
        if not isinstance(value, str):
            return value
        match = self._REFERENCE.fullmatch(value)
        if match is None:
            return value
        node, section, path = match.groups()
        current: Any = outcomes[node].get(section)
        if section == "artifacts":
            for artifact in current or ():
                if artifact.get("path") == path:
                    return str(Path(outcomes[node]["result"]["run_dir"]) / path)
            raise KeyError(f"Unknown stage artifact {path!r} from {node!r}.")
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise KeyError(f"Unknown dataset stage reference {value!r}.")
            current = current[part]
        return current

    @staticmethod
    def _build_id(recipe: DatasetRecipeConfig, nodes: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            {"recipe": recipe.fingerprint, "stages": nodes},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def _result_path(self, recipe: DatasetRecipeConfig) -> Path:
        return (
            recipe.output_root
            / "builds"
            / recipe.name
            / recipe.version
            / recipe.fingerprint.removeprefix("sha256:")[:16]
            / "build-result.json"
        )

    def _write_result(self, recipe: DatasetRecipeConfig, result: DatasetBuildResult) -> None:
        path = self._result_path(recipe)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
