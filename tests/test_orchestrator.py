"""Spawn scheduler and process-lifecycle contract tests."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest
import torch.multiprocessing as mp

from lambdaforge.training import TrainingJob, TrainingOrchestrator
from lambdaforge.training.orchestration.ProcessGuard import (
    ORPHAN_EXIT_CODE,
    ProcessGuard,
)
from tests.fixtures.ExpectedParentGuardProbe import ExpectedParentGuardProbe
from tests.fixtures.FileWritingJob import FileWritingJob
from tests.fixtures.StopAfterFirstLaunchOrchestrator import (
    StopAfterFirstLaunchOrchestrator,
)


class TestTrainingOrchestrator:
    """Verify bounded scheduling with pickle-safe callable objects."""

    def test_runs_jobs_through_one_reusable_slot(self, tmp_path) -> None:
        paths = [tmp_path / "one.txt", tmp_path / "two.txt"]
        jobs = [TrainingJob(path.stem, FileWritingJob(path)) for path in paths]
        callbacks: list[tuple[str, int | None]] = []
        orchestrator = TrainingOrchestrator(
            poll_seconds=0.02,
            cpu_threads_per_job=1,
            cpu_interop_threads_per_job=1,
            cpu_cores_per_job=None,
        )
        exit_codes = orchestrator.run_scheduled(
            jobs,
            slots=[None],
            on_job_finished=lambda name, code: callbacks.append((name, code)),
        )
        assert exit_codes == {"one": 0, "two": 0}
        assert callbacks == [("one", 0), ("two", 0)]
        assert [path.read_text(encoding="utf-8") for path in paths] == ["ok", "ok"]

    def test_training_job_freezes_a_snapshot_of_device_assignment(self, tmp_path) -> None:
        requested_devices = [0, 2]
        job = TrainingJob(
            "immutable",
            FileWritingJob(tmp_path / "immutable.txt"),
            devices=requested_devices,
        )
        requested_devices.append(4)

        assert job.devices == (0, 2)
        with pytest.raises(FrozenInstanceError):
            job.devices = (1,)  # type: ignore[misc]

        cpu_job = TrainingJob(
            "cpu",
            FileWritingJob(tmp_path / "cpu.txt"),
            devices=[],
        )
        assert cpu_job.devices == ()
        orchestrator = TrainingOrchestrator(cpu_cores_per_job=None)
        assert orchestrator._resolve_cuda_visible_devices(cpu_job.devices) == ""

    @pytest.mark.parametrize(
        "devices, expected_error",
        [
            ([True], TypeError),
            ([0, 0], ValueError),
            ([-1], ValueError),
            ("0", TypeError),
        ],
    )
    def test_training_job_rejects_invalid_device_assignments(
        self,
        tmp_path,
        devices,
        expected_error,
    ) -> None:
        with pytest.raises(expected_error):
            TrainingJob(
                "invalid",
                FileWritingJob(tmp_path / "invalid.txt"),
                devices=devices,
            )

    @pytest.mark.parametrize(
        "arguments, expected_error",
        [
            ({"grace_seconds": True}, TypeError),
            ({"grace_seconds": float("nan")}, ValueError),
            ({"grace_seconds": float("inf")}, ValueError),
            ({"grace_seconds": -0.01}, ValueError),
            ({"poll_seconds": "0.1"}, TypeError),
            ({"poll_seconds": 0.0}, ValueError),
            ({"poll_seconds": float("-inf")}, ValueError),
        ],
    )
    def test_orchestrator_rejects_invalid_time_configuration(
        self,
        arguments,
        expected_error,
    ) -> None:
        with pytest.raises(expected_error):
            TrainingOrchestrator(**arguments)

    def test_process_guard_rejects_non_finite_and_boolean_times(self) -> None:
        guard = ProcessGuard()
        with pytest.raises(TypeError):
            guard.install_parent_death_guard(poll_seconds=True)
        with pytest.raises(ValueError):
            guard.terminate_process_tree(None, grace_seconds=float("nan"))
        with pytest.raises(ValueError):
            guard.install_parent_death_guard(expected_parent_pid=True)

    def test_manage_signals_false_runs_safely_from_secondary_thread(self) -> None:
        orchestrator = TrainingOrchestrator(
            manage_signals=False,
            cpu_cores_per_job=None,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(orchestrator.run, []).result(timeout=5.0)
        assert result == {}

    def test_manage_signals_true_fails_cleanly_from_secondary_thread(self) -> None:
        orchestrator = TrainingOrchestrator(
            manage_signals=True,
            cpu_cores_per_job=None,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(orchestrator.run, [])
            with pytest.raises(RuntimeError, match="main thread"):
                future.result(timeout=5.0)
        assert orchestrator._windows_job is None
        assert orchestrator._previous_signal_handlers == {}

    def test_run_stops_launching_after_stop_is_requested(self, tmp_path) -> None:
        paths = [tmp_path / f"job-{index}.txt" for index in range(3)]
        jobs = [TrainingJob(path.stem, FileWritingJob(path)) for path in paths]
        orchestrator = StopAfterFirstLaunchOrchestrator(
            grace_seconds=5.0,
            poll_seconds=0.02,
            cpu_cores_per_job=None,
        )

        exit_codes = orchestrator.run(jobs)

        assert exit_codes["job-0"] == 0
        assert exit_codes["job-1"] is None
        assert exit_codes["job-2"] is None
        assert not paths[1].exists()
        assert not paths[2].exists()
        assert all(not process.is_alive() for _, process in orchestrator.processes)

    def test_scheduled_stop_notifies_launched_job_once(self, tmp_path) -> None:
        paths = [tmp_path / f"scheduled-{index}.txt" for index in range(3)]
        jobs = [TrainingJob(path.stem, FileWritingJob(path)) for path in paths]
        callbacks: list[tuple[str, int | None]] = []
        orchestrator = StopAfterFirstLaunchOrchestrator(
            grace_seconds=5.0,
            poll_seconds=0.02,
            cpu_cores_per_job=None,
        )

        exit_codes = orchestrator.run_scheduled(
            jobs,
            slots=[None, None],
            on_job_finished=lambda name, code: callbacks.append((name, code)),
        )

        assert exit_codes == {
            "scheduled-0": 0,
            "scheduled-1": None,
            "scheduled-2": None,
        }
        assert callbacks == [("scheduled-0", 0)]
        assert all(not process.is_alive() for _, process in orchestrator.processes)

    @pytest.mark.skipif(os.name != "posix", reason="Parent PID guard is POSIX-only.")
    def test_expected_parent_pid_mismatch_exits_immediately(self) -> None:
        context = mp.get_context("spawn")
        process = context.Process(
            target=ExpectedParentGuardProbe(os.getpid() + 1),
            name="expected-parent-guard-probe",
        )
        process.start()
        try:
            process.join(timeout=5.0)
            assert not process.is_alive()
            assert process.exitcode == ORPHAN_EXIT_CODE
        finally:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
