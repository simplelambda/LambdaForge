"""Sequential and scheduled subprocess execution for materialized runs."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from lambdaforge.experiments.ExecutionConfig import ExecutionConfig
from lambdaforge.experiments.ExecutionMode import ExecutionMode
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.ExperimentWorker import ExperimentWorker
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus
from lambdaforge.training.orchestration.TrainingJob import TrainingJob
from lambdaforge.training.orchestration.TrainingOrchestrator import TrainingOrchestrator


class ExperimentExecutor:
    """Run expanded experiments while preserving per-run isolation and logs."""

    def run(
        self,
        run_configs: Sequence[dict[str, Any]],
        execution: ExecutionConfig,
        on_run_finished: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    ) -> list[RunResult]:
        """Execute every materialized config using the selected mode."""
        if execution.mode is ExecutionMode.SEQUENTIAL:
            return self._run_sequential(run_configs, on_run_finished)
        return self._run_parallel(run_configs, execution, on_run_finished)

    def _run_sequential(
        self,
        run_configs: Sequence[dict[str, Any]],
        on_run_finished: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None,
    ) -> list[RunResult]:
        from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
        from lambdaforge.experiments.StdIOCapture import StdIOCapture

        runner = ExperimentRunner()
        results: list[RunResult] = []
        for config in run_configs:
            run_dir = runner.experiment_run_dir(config)
            run_dir.mkdir(parents=True, exist_ok=True)
            with StdIOCapture(run_dir / "train.log", echo=True):
                try:
                    result = runner._run_single_experiment_unlocked(config)
                except KeyboardInterrupt:
                    traceback.print_exc()
                    self._write_interrupted(config, "KeyboardInterrupt: execution interrupted.")
                    raise
                except Exception:
                    traceback.print_exc()
                    self._write_failure(config, traceback.format_exc().splitlines()[-1])
                    result = self._collect_result(config, 1)
            results.append(result)
            self._notify(on_run_finished, config, result, self._job_name(config))
        return results

    def _run_parallel(
        self,
        run_configs: Sequence[dict[str, Any]],
        execution: ExecutionConfig,
        on_run_finished: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None,
    ) -> list[RunResult]:
        slots = execution.slots()
        patched = [execution.patch_run(config) for config in run_configs]
        jobs = [
            TrainingJob(name=self._job_name(config), run=ExperimentWorker(config))
            for config in patched
        ]
        config_by_job = {job.name: config for job, config in zip(jobs, patched, strict=True)}
        print(
            f"Execution: mode={execution.mode.value}, gpus={execution.gpus}, "
            f"slots={len(slots)}, runs={len(jobs)}, "
            f"cpu_jobs={execution.cpu_jobs}, "
            f"cpu_threads_per_job={execution.cpu_threads_per_job}, "
            f"cpu_interop_threads_per_job={execution.cpu_interop_threads_per_job}, "
            f"cpu_cores_per_job={execution.cpu_cores_per_job}, "
            f"dataloader_num_workers_per_job={execution.dataloader_num_workers_per_job}",
            flush=True,
        )
        orchestrator = TrainingOrchestrator(
            grace_seconds=execution.grace_seconds,
            cpu_threads_per_job=execution.cpu_threads_per_job,
            cpu_interop_threads_per_job=execution.cpu_interop_threads_per_job,
            cpu_cores_per_job=execution.cpu_cores_per_job,
        )

        def finished(name: str, exit_code: int | None) -> None:
            config = config_by_job[name]
            self._notify(
                on_run_finished,
                config,
                self._collect_result(config, exit_code),
                name,
            )

        exit_codes = orchestrator.run_scheduled(jobs, slots, on_job_finished=finished)
        return [
            self._collect_result(config, exit_codes.get(self._job_name(config)))
            for config in patched
        ]

    def _collect_result(
        self,
        config: Mapping[str, Any],
        exit_code: int | None,
    ) -> RunResult:
        from lambdaforge.experiments.ExperimentRunner import ExperimentRunner

        run_dir = ExperimentRunner().experiment_run_dir(config)
        result_path = run_dir / "result.json"
        if result_path.exists():
            try:
                persisted = RunResult.read_json(result_path)
                if exit_code == 0:
                    return persisted
                if exit_code is not None and persisted.status is not RunStatus.OK:
                    return persisted
            except (OSError, ValueError, TypeError):
                pass
        status = RunStatus.INTERRUPTED if exit_code is None else RunStatus.FAILED
        attempt_id, fingerprint, started_at, finished_at = self._attempt_fields(config)
        return RunResult(
            result_version=2,
            name=ExperimentConfig.get_value(config, "experiment.name", "experiment"),
            run_dir=run_dir,
            variant=ExperimentConfig.get_value(config, "experiment.variant"),
            seed=ExperimentConfig.get_value(config, "experiment.seed"),
            status=status,
            exit_code=exit_code,
            attempt_id=attempt_id,
            config_fingerprint=fingerprint,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
        )

    def _write_failure(self, config: Mapping[str, Any], error: str) -> None:
        from lambdaforge.experiments.ExperimentRunner import ExperimentRunner

        run_dir = ExperimentRunner().experiment_run_dir(config)
        attempt_id, fingerprint, started_at, finished_at = self._attempt_fields(config)
        RunResult(
            result_version=2,
            name=ExperimentConfig.get_value(config, "experiment.name", "experiment"),
            run_dir=run_dir,
            variant=ExperimentConfig.get_value(config, "experiment.variant"),
            seed=ExperimentConfig.get_value(config, "experiment.seed"),
            status=RunStatus.FAILED,
            error=error,
            attempt_id=attempt_id,
            config_fingerprint=fingerprint,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
        ).write_json(run_dir / "result.json")

    def _write_interrupted(self, config: Mapping[str, Any], error: str) -> None:
        """Persist a resumable sequential interruption before propagating SIGINT."""
        from lambdaforge.experiments.ExperimentRunner import ExperimentRunner

        run_dir = ExperimentRunner().experiment_run_dir(config)
        attempt_id, fingerprint, started_at, finished_at = self._attempt_fields(config)
        RunResult(
            result_version=2,
            name=ExperimentConfig.get_value(config, "experiment.name", "experiment"),
            run_dir=run_dir,
            variant=ExperimentConfig.get_value(config, "experiment.variant"),
            seed=ExperimentConfig.get_value(config, "experiment.seed"),
            status=RunStatus.INTERRUPTED,
            error=error,
            attempt_id=attempt_id,
            config_fingerprint=fingerprint,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
        ).write_json(run_dir / "result.json")

    @staticmethod
    def _attempt_fields(config: Mapping[str, Any]) -> tuple[str, str, str, str]:
        """Create provenance fields for failures that occur outside the runner."""
        timestamp = datetime.now(timezone.utc)
        fingerprint = RunFingerprint.digest(config)
        return (
            (
                f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-"
                f"{fingerprint.removeprefix('sha256:')[:12]}-{uuid4().hex[:8]}"
            ),
            fingerprint,
            timestamp.isoformat(),
            timestamp.isoformat(),
        )

    @staticmethod
    def _job_name(config: Mapping[str, Any]) -> str:
        from lambdaforge.experiments.ExperimentRunner import ExperimentRunner

        variant = ExperimentConfig.get_value(config, "experiment.variant", "base")
        return f"{variant}__{ExperimentRunner().seed_segment(config)}"

    @staticmethod
    def _notify(
        callback: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None,
        config: Mapping[str, Any],
        result: Mapping[str, Any],
        name: str,
    ) -> None:
        if callback is None:
            return
        try:
            callback(config, result)
        except Exception as error:
            print(f"WARNING: post-run callback failed for {name}: {error}", flush=True)
