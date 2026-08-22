"""Reproducible local runner for one generic task configuration."""

from __future__ import annotations

import inspect
import json
import os
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from lambdaforge.EnvironmentManifest import EnvironmentManifest
from lambdaforge.execution.FailureClassifier import FailureClassifier
from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.experiments.StdIOCapture import StdIOCapture
from lambdaforge.observability.EventLogger import EventLogger
from lambdaforge.plugins.PluginRegistry import PluginRegistry
from lambdaforge.runtime.api import RunContext, RuntimeCapture, activate
from lambdaforge.tasks.artifacts import ArtifactDeclaration, TaskArtifact
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskContext import TaskContext
from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan, TaskPlanAction
from lambdaforge.tasks.TaskOutput import TaskOutput
from lambdaforge.tasks.TaskResult import TaskResult, TaskStatus
from lambdaforge.tasks.TaskValidator import TaskValidator


class TaskRunner:
    """Plan, execute and persist a generic local task without training assumptions."""

    def plan(self, config: TaskConfig) -> TaskExecutionPlan:
        """Resolve a non-mutating execution plan without importing task code."""
        existing = self._read_result(config.run_dir / "result.json")
        complete = existing is not None and self._is_complete(existing, config)
        if complete and not config.rerun_completed:
            action = TaskPlanAction.SKIP
            reason = "a matching successful result and all declared artifacts already exist"
        else:
            action = TaskPlanAction.RUN
            if existing is None:
                reason = "no previous result exists for this task fingerprint"
            elif config.rerun_completed:
                reason = "rerun_completed is enabled"
            else:
                reason = "the previous result is incomplete or failed verification"
        spec = config["task"]
        if "target" in spec:
            target = str(spec["target"])
        else:
            plugin = spec["plugin"]
            target = f"plugin:{plugin['kind']}:{plugin['name']}"
        spec_params = spec.get("params", {})
        callable_path = None
        parameters: Mapping[str, Any] = {}
        if target == "lambdaforge.runtime.CallableTask" and isinstance(spec_params, Mapping):
            callable_path = str(spec_params.get("callable_path") or spec_params.get("class_path"))
            raw_parameters = spec_params.get("parameters", {})
            parameters = raw_parameters if isinstance(raw_parameters, Mapping) else {}
        return TaskExecutionPlan(
            name=config.name,
            run_dir=config.run_dir,
            suite_dir=config.suite_dir,
            config_fingerprint=config.fingerprint,
            task_target=target,
            action=action,
            reason=reason,
            required_artifacts=config.required_artifacts,
            inputs=[value.to_dict() for value in config.resolved_inputs],
            execution=config.get("execution", {"mode": "sequential"}),
            callable_path=callable_path,
            parameters=parameters,
            resources=config.resources.to_dict(),
        )

    def run(
        self,
        config: TaskConfig,
        *,
        dry_run: bool = False,
        stop_event: Any = None,
    ) -> TaskResult | TaskExecutionPlan:
        """Validate and execute one task, or return its side-effect-free dry-run plan."""
        report = TaskValidator().validate(config)
        if not report.is_valid:
            raise ValueError(report.summary())
        plan = self.plan(config)
        if dry_run:
            return plan
        if plan.action is TaskPlanAction.SKIP:
            existing = TaskResult.read_json(config.run_dir / "result.json")
            return existing.with_updates(skipped_existing=True)

        run_dir = config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._validate_internal_paths(run_dir)
        self._archive_previous_result(run_dir)
        self._write_config(config, run_dir / "config.yaml")
        started = datetime.now(timezone.utc)
        attempt_id = self._attempt_id(started, config.fingerprint)
        environment = EnvironmentManifest.capture(config.source_dir)
        environment.write(run_dir / "environment.json")
        plugins = PluginRegistry.default()
        started_clock = time.perf_counter()
        result: TaskResult
        events = EventLogger(run_dir / "events.jsonl")
        events.write(
            "task_started",
            {
                "name": config.name,
                "attempt_id": attempt_id,
                "config_fingerprint": config.fingerprint,
            },
        )

        with StdIOCapture(run_dir / "task.log", echo=True):
            with plugins.usage_session() as usage:
                try:
                    task = ObjectFactory.build(config["task"], plugins=plugins)
                    method = getattr(task, "run", None)
                    if not callable(method):
                        raise TypeError(
                            "The configured task object must expose a callable run method."
                        )
                    context = TaskContext(
                        name=config.name,
                        run_dir=run_dir.resolve(),
                        source_dir=config.source_dir,
                        attempt_id=attempt_id,
                        config_fingerprint=config.fingerprint,
                        resume=config.resume,
                        metadata=config.get("metadata", {}),
                        inputs=tuple(value.to_dict() for value in config.resolved_inputs),
                        outputs=config.outputs,
                        stop_event=stop_event,
                    )
                    if context.stop_requested:
                        raise KeyboardInterrupt("Task execution was cancelled before start.")
                    runtime = RuntimeCapture(
                        RunContext(
                            context.name,
                            context.run_dir,
                            context.source_dir,
                            context.attempt_id,
                            context.config_fingerprint,
                            getattr(task, "seed", None),
                            (
                                task.runtime_parameters(context)
                                if callable(getattr(task, "runtime_parameters", None))
                                else {}
                            ),
                        )
                    )
                    with activate(runtime):
                        returned = self._invoke(method, context)
                    output = self._merge_runtime(TaskOutput.from_value(returned), runtime)
                    artifacts = self._materialize_artifacts(output, config)
                    finished = datetime.now(timezone.utc)
                    result = TaskResult(
                        name=config.name,
                        run_dir=run_dir,
                        status=TaskStatus.OK,
                        seconds=time.perf_counter() - started_clock,
                        outputs=output.outputs,
                        metrics=output.metrics,
                        artifacts=artifacts,
                        metadata={
                            **dict(output.metadata),
                            "inputs": [value.to_dict() for value in config.resolved_inputs],
                        },
                        attempt_id=attempt_id,
                        config_fingerprint=config.fingerprint,
                        started_at_utc=started.isoformat(),
                        finished_at_utc=finished.isoformat(),
                    )
                except KeyboardInterrupt as error:
                    finished = datetime.now(timezone.utc)
                    result = self._failure_result(
                        config,
                        TaskStatus.INTERRUPTED,
                        error,
                        started,
                        finished,
                        attempt_id,
                        started_clock,
                    )
                except Exception as error:
                    traceback.print_exc()
                    finished = datetime.now(timezone.utc)
                    result = self._failure_result(
                        config,
                        TaskStatus.FAILED,
                        error,
                        started,
                        finished,
                        attempt_id,
                        started_clock,
                    )
                finally:
                    environment.with_plugins(usage.descriptors()).write(
                        run_dir / "environment.json"
                    )
        result.write_json(run_dir / "result.json")
        if result.status is TaskStatus.OK:
            self._register_dataset(config, run_dir, events)
        event_fields: dict[str, Any] = {
            "name": config.name,
            "attempt_id": attempt_id,
            "status": result.status.value,
            "seconds": result.seconds,
        }
        if result.error is not None:
            event_fields["failure_category"] = (
                FailureClassifier().classify(str(result.error.get("message", ""))).value
            )
        events.write("task_finished", event_fields)
        return result

    @staticmethod
    def _merge_runtime(output: TaskOutput, capture: RuntimeCapture) -> TaskOutput:
        artifacts = list(output.artifacts)
        if capture.metric_count:
            artifacts.append(
                ArtifactDeclaration(
                    "metrics.jsonl",
                    name="metrics",
                    kind="metrics",
                    media_type="application/x-ndjson",
                )
            )
        artifacts.extend(capture.artifacts)
        return TaskOutput(
            outputs=output.outputs,
            metrics={**dict(capture.metrics), **dict(output.metrics)},
            artifacts=artifacts,
            metadata={
                **dict(output.metadata),
                "metric_observations": capture.metric_count,
                "published_datasets": capture.datasets,
            },
        )

    @staticmethod
    def _register_dataset(config: TaskConfig, run_dir: Path, events: EventLogger) -> None:
        """Publish preprocessing manifests to the reconciliable dataset registry."""
        manifest = run_dir / "dataset-artifact.json"
        if not manifest.is_file() or manifest.is_symlink():
            return
        try:
            from lambdaforge.data.DatasetRegistry import DatasetRegistry

            registry_path = os.environ.get("LAMBDAFORGE_DATASET_REGISTRY") or (
                DatasetRegistry.project_path(config.source_dir)
            )
            record = DatasetRegistry(registry_path).register_artifact(
                manifest,
                cluster=os.environ.get("LAMBDAFORGE_CLUSTER", "local"),
                root=run_dir,
                producer={
                    "config": str(config.source) if config.source is not None else None,
                    "task_fingerprint": config.fingerprint,
                },
            )
            events.write("dataset_registered", {"dataset": record.key})
        except Exception as error:
            # The manifest/result remain authoritative; an auxiliary index can be reconciled.
            events.write(
                "dataset_registration_failed",
                {"error_type": error.__class__.__name__, "message": str(error)},
            )

    @staticmethod
    def _invoke(method: Any, context: TaskContext) -> Any:
        """Call an explicit context method or the documented zero-argument duck form."""
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(context)
        try:
            signature.bind(context)
        except TypeError as context_error:
            try:
                signature.bind()
            except TypeError as no_argument_error:
                raise TypeError(
                    "Task.run must accept either one TaskContext or no arguments. "
                    f"Context form: {context_error}; zero-argument form: {no_argument_error}."
                ) from no_argument_error
            return method()
        return method(context)

    def _materialize_artifacts(
        self,
        output: TaskOutput,
        config: TaskConfig,
    ) -> tuple[TaskArtifact, ...]:
        declarations = list(output.artifacts)
        declared_paths = {str(declaration.path) for declaration in declarations}
        for required in config.required_artifacts:
            if required not in declared_paths:
                declarations.append(ArtifactDeclaration(required))
        artifacts = tuple(
            TaskArtifact.materialize(declaration, config.run_dir) for declaration in declarations
        )
        paths = [artifact.path for artifact in artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("A task cannot declare the same artifact path more than once.")
        return artifacts

    def _is_complete(self, result: TaskResult, config: TaskConfig) -> bool:
        if result.status is not TaskStatus.OK or result.config_fingerprint != config.fingerprint:
            return False
        artifacts = {artifact.path: artifact for artifact in result.artifacts}
        for artifact in result.artifacts:
            try:
                current = TaskArtifact.materialize(
                    ArtifactDeclaration(
                        path=artifact.path,
                        kind=artifact.kind,
                        media_type=artifact.media_type,
                        metadata=artifact.metadata,
                    ),
                    config.run_dir,
                )
            except (OSError, TypeError, ValueError):
                return False
            if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
                return False
        for required in config.required_artifacts:
            if required not in artifacts:
                try:
                    TaskArtifact.materialize(ArtifactDeclaration(required), config.run_dir)
                except (OSError, TypeError, ValueError):
                    return False
        return True

    @staticmethod
    def _failure_result(
        config: TaskConfig,
        status: TaskStatus,
        error: BaseException,
        started: datetime,
        finished: datetime,
        attempt_id: str,
        started_clock: float,
    ) -> TaskResult:
        return TaskResult(
            name=config.name,
            run_dir=config.run_dir,
            status=status,
            seconds=time.perf_counter() - started_clock,
            error={
                "type": error.__class__.__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            attempt_id=attempt_id,
            config_fingerprint=config.fingerprint,
            started_at_utc=started.isoformat(),
            finished_at_utc=finished.isoformat(),
        )

    @staticmethod
    def _read_result(path: Path) -> TaskResult | None:
        if not path.exists() or path.is_symlink():
            return None
        try:
            return TaskResult.read_json(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _archive_previous_result(run_dir: Path) -> None:
        source = run_dir / "result.json"
        if not source.exists():
            return
        previous = TaskResult.read_json(source)
        attempts = run_dir / ".lambdaforge" / "attempts"
        if attempts.exists() and attempts.is_symlink():
            raise ValueError(f"Unsafe task attempt archive directory: {attempts}")
        attempts.mkdir(parents=True, exist_ok=True)
        attempt_id = previous.attempt_id or f"legacy-{uuid4().hex}"
        destination = attempts / f"result-{attempt_id}.json"
        if destination.exists():
            raise FileExistsError(f"Task attempt history already exists: {destination}")
        previous.write_json(destination)

    @staticmethod
    def _validate_internal_paths(run_dir: Path) -> None:
        """Reject task-owned metadata paths that could redirect writes through links."""
        paths = (
            run_dir / "config.yaml",
            run_dir / "environment.json",
            run_dir / "task.log",
            run_dir / "events.jsonl",
            run_dir / "events.jsonl.lock",
            run_dir / "result.json",
            run_dir / ".lambdaforge",
            run_dir / ".lambdaforge" / "attempts",
        )
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Unsafe symbolic link at task-owned path: {path}")

    @staticmethod
    def _write_config(config: TaskConfig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config.redacted_dict(), handle, sort_keys=False, allow_unicode=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _attempt_id(started: datetime, fingerprint: str) -> str:
        digest = fingerprint.removeprefix("sha256:")[:12]
        return f"{started.strftime('%Y%m%dT%H%M%S%fZ')}-{digest}-{uuid4().hex[:8]}"
