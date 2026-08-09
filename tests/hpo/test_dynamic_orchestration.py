"""Dynamic scheduler integration contract."""

from __future__ import annotations

from lambdaforge.training import TrainingJob, TrainingOrchestrator
from tests.fixtures.FileWritingJob import FileWritingJob


def test_dynamic_supplier_observes_each_result_before_supplying_next(tmp_path) -> None:
    pending = [tmp_path / "one.txt", tmp_path / "two.txt"]
    supplied: list[str] = []
    finished: list[str] = []

    def next_job(slot_index: int, slot: tuple[int, ...] | None) -> TrainingJob | None:
        assert slot_index == 0
        assert slot == ()
        if not pending:
            return None
        path = pending.pop(0)
        supplied.append(path.stem)
        if path.stem == "two":
            assert finished == ["one"]
        return TrainingJob(path.stem, FileWritingJob(path))

    orchestrator = TrainingOrchestrator(poll_seconds=0.02, cpu_cores_per_job=None)
    codes = orchestrator.run_dynamic(
        [[]],
        next_job,
        lambda name, code, slot: finished.append(name),
    )
    assert codes == {"one": 0, "two": 0}
    assert supplied == finished == ["one", "two"]
