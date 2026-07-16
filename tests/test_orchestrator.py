"""Spawn scheduler smoke tests."""

from lambdaforge.training import TrainingJob, TrainingOrchestrator
from tests.fixtures.FileWritingJob import FileWritingJob


class TestTrainingOrchestrator:
    """Verify bounded scheduling with pickle-safe callable objects."""

    def test_runs_jobs_through_one_reusable_slot(self, tmp_path) -> None:
        paths = [tmp_path / "one.txt", tmp_path / "two.txt"]
        jobs = [TrainingJob(path.stem, FileWritingJob(path)) for path in paths]
        orchestrator = TrainingOrchestrator(
            poll_seconds=0.02,
            cpu_threads_per_job=1,
            cpu_interop_threads_per_job=1,
            cpu_cores_per_job=None,
        )
        exit_codes = orchestrator.run_scheduled(jobs, slots=[None])
        assert exit_codes == {"one": 0, "two": 0}
        assert [path.read_text(encoding="utf-8") for path in paths] == ["ok", "ok"]
