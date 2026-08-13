"""Dependency-aware local workflow execution."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.configuration.AuthoringConfigNormalizer import AuthoringConfigNormalizer
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.configuration.ResolvedConfiguration import ResolvedConfiguration
from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskRun import TaskRun
from lambdaforge.workflows.WorkflowConfig import WorkflowConfig
from lambdaforge.workflows.WorkflowNode import WorkflowNode
from lambdaforge.workflows.WorkflowPlan import WorkflowPlan
from lambdaforge.workflows.WorkflowResult import WorkflowResult


class WorkflowRunner:
    """Run ready DAG nodes with bounded concurrency and branch isolation."""

    _REFERENCE = re.compile(r"^\$\{nodes\.([^.]+)\.(outputs|metrics|artifacts)\.([^{}]+)\}$")

    def plan(self, config: WorkflowConfig) -> WorkflowPlan:
        """Return deterministic topological levels."""
        remaining = {node.name: set(node.needs) for node in config.nodes}
        levels: list[tuple[str, ...]] = []
        completed: set[str] = set()
        while remaining:
            ready = tuple(sorted(name for name, needs in remaining.items() if needs <= completed))
            if not ready:
                raise ValueError("Workflow contains a dependency cycle.")
            levels.append(ready)
            completed.update(ready)
            for name in ready:
                del remaining[name]
        return WorkflowPlan(
            config.name,
            config.run_dir,
            tuple(levels),
            config.max_parallel,
            {node.name: node.cluster for node in config.nodes},
        )

    def run(
        self, config: WorkflowConfig, *, dry_run: bool = False
    ) -> WorkflowResult | WorkflowPlan:
        """Execute ready nodes; failed branches block only their descendants."""
        plan = self.plan(config)
        if dry_run:
            return plan
        remote = [node.name for node in config.nodes if node.cluster != "local"]
        if remote:
            raise ValueError(
                "Remote workflow coordination is not enabled in the in-process DAG runner. "
                f"Remote nodes: {remote}. Submit their configs with 'lambdaforge run --on' "
                "and keep data transfer explicit."
            )
        node_by_name = {node.name: node for node in config.nodes}
        outcomes: dict[str, dict[str, Any]] = {}
        for level in plan.levels:
            runnable: list[WorkflowNode] = []
            for name in level:
                node = node_by_name[name]
                failed = [
                    dependency
                    for dependency in node.needs
                    if outcomes[dependency]["status"] != "ok"
                ]
                if failed and not node.continue_on_failure:
                    outcomes[name] = {"status": "blocked", "blocked_by": failed}
                else:
                    runnable.append(node)
            with ProcessPoolExecutor(
                max_workers=min(config.max_parallel, len(runnable) or 1),
                mp_context=mp.get_context("spawn"),
            ) as pool:
                futures = {
                    pool.submit(self._execute, node, outcomes): node.name for node in runnable
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        outcomes[name] = future.result()
                    except Exception as error:
                        outcomes[name] = {
                            "status": "failed",
                            "error": {"type": type(error).__name__, "message": str(error)},
                        }
        status = (
            "ok"
            if outcomes and all(value["status"] == "ok" for value in outcomes.values())
            else "failed"
        )
        result = WorkflowResult(config.name, config.run_dir, status, outcomes)
        self._write_result(config.run_dir / "workflow-result.json", result.to_dict())
        return result

    def _execute(self, node: WorkflowNode, outcomes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        data, source, resolution = node.materialize()
        for path, value in node.bindings.items():
            ExperimentConfig.set_value(data, str(path), self._resolve(value, outcomes))
        if AuthoringConfigNormalizer().detect(data) is ConfigurationKind.TASK:
            if resolution is not None:
                resolution = ResolvedConfiguration(
                    data,
                    {
                        **dict(resolution.provenance),
                        **{str(path): f"workflow binding:{node.name}" for path in node.bindings},
                    },
                    resolution.sources,
                )
            result = TaskRun(TaskConfig(data, source=source, resolution=resolution)).run()
            payload = result.to_dict()
            return {
                "status": payload["status"],
                "outputs": payload.get("outputs", {}),
                "metrics": payload.get("metrics", {}),
                "artifacts": payload.get("artifacts", []),
                "result": payload,
            }
        if resolution is not None and resolution.contains_secrets:
            raise ValueError(
                "Experiment workflow nodes cannot persist composed secrets; read credentials "
                "from the runtime environment instead."
            )
        result_list = Experiment(ExperimentConfig(data, source=source)).run()
        if not isinstance(result_list, list):
            adaptive = result_list.to_dict()
            summary = adaptive.get("summary", {})
            status = "ok" if summary.get("status") == "ok" else "failed"
            return {
                "status": status,
                "outputs": {
                    "best_configuration": summary.get("best_configuration"),
                    "study_dir": adaptive.get("study_dir"),
                },
                "metrics": summary,
                "artifacts": [
                    adaptive.get("state_path"),
                    adaptive.get("event_log_path"),
                ],
                "result": adaptive,
            }
        payloads = [result.to_dict() for result in result_list]
        status = (
            "ok" if payloads and all(item.get("status") == "ok" for item in payloads) else "failed"
        )
        metrics = payloads[-1].get("final_metrics", {}) if payloads else {}
        artifacts = [
            artifact
            for payload in payloads
            for artifact in payload.get("artifacts", [])
            if isinstance(artifact, dict)
        ]
        return {
            "status": status,
            "outputs": {},
            "metrics": metrics,
            "artifacts": artifacts,
            "results": payloads,
        }

    def _resolve(self, value: Any, outcomes: dict[str, dict[str, Any]]) -> Any:
        if isinstance(value, dict):
            return {key: self._resolve(item, outcomes) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve(item, outcomes) for item in value]
        if not isinstance(value, str):
            return value
        match = self._REFERENCE.fullmatch(value)
        if match is None:
            return value
        node, section, path = match.groups()
        if node not in outcomes:
            raise KeyError(f"Workflow output references unfinished node {node!r}.")
        current: Any = outcomes[node].get(section)
        if section == "artifacts":
            for artifact in current or ():
                if artifact.get("path") == path:
                    return str(Path(outcomes[node]["result"]["run_dir"]) / path)
            raise KeyError(f"Unknown artifact {path!r} from node {node!r}.")
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Unknown workflow reference {value!r}.")
            current = current[part]
        return current

    @staticmethod
    def _write_result(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
