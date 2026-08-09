"""Dynamic scheduler integration contract."""

from __future__ import annotations

from lambdaforge.training import TrainingJob, TrainingOrchestrator
from tests.fixtures.FileWritingJob import FileWritingJob
from tests.fixtures.TimedWritingJob import TimedWritingJob


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


def test_dynamic_scheduler_dispatches_when_first_of_unequal_jobs_finishes(tmp_path) -> None:
    initial = {
        0: ("a", 0.05),
        1: ("b", 0.30),
        2: ("c", 0.40),
    }
    supplied_slots: set[int] = set()
    dispatched_follow_up = False
    finished: list[str] = []

    def next_job(slot_index: int, slot: tuple[int, ...] | None) -> TrainingJob | None:
        nonlocal dispatched_follow_up
        assert slot == ()
        if slot_index not in supplied_slots:
            supplied_slots.add(slot_index)
            name, duration = initial[slot_index]
            return TrainingJob(name, TimedWritingJob(tmp_path / f"{name}.txt", duration))
        if slot_index == 0 and not dispatched_follow_up:
            assert finished == ["a"]
            dispatched_follow_up = True
            return TrainingJob("d", TimedWritingJob(tmp_path / "d.txt", 0.01))
        return None

    orchestrator = TrainingOrchestrator(poll_seconds=0.01, cpu_cores_per_job=None)
    codes = orchestrator.run_dynamic(
        [[], [], []],
        next_job,
        lambda name, code, slot: finished.append(name),
    )

    assert codes == {"a": 0, "d": 0, "b": 0, "c": 0}
    assert dispatched_follow_up
