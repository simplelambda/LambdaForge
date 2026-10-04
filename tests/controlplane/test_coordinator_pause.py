"""Pause preserves exact ownership/evidence and never resets the Study clock."""

import json
from pathlib import Path

import pytest

from lambdaforge.controlplane.StudyCoordinator import StudyCoordinator
from tests.controlplane.test_coordinated_study import (
    FakeExecutor,
    coordinator,
    envelope,
    offer,
    run,
)


def test_pause_drains_unknown_owner_ingests_and_resumes_without_reset(tmp_path: Path) -> None:
    control = coordinator(tmp_path, max_runs=10, max_time_seconds=100)
    control.enqueue([run(1), run(2)])
    executor = FakeExecutor()
    shard = control.plan_shards([offer("A", 1)])[0]
    control.submit_shard(shard, executor)
    before = control.run_records()
    assert control.request_pause() == "pausing"
    with pytest.raises(ValueError, match="new scientific proposals"):
        control.enqueue([run(3)])
    control.enqueue([run(1), run(2)])  # Idempotent reconnect is not new science.
    assert not control.plan_shards([offer("B", 1)])
    executor.offline = True
    control.reconcile("A", executor)
    assert control.snapshot()["state"] == "pausing"
    with pytest.raises(ValueError, match="paused"):
        control.begin_resume()
    executor.offline = False
    executor.finish(shard)
    control.reconcile("A", executor)
    assert control.snapshot()["state"] == "paused"
    evidence = control.result(shard.leases[0].run.key)
    fresh = StudyCoordinator(control.root, clock=lambda: 40)
    fresh.begin_resume()
    assert fresh.snapshot()["state"] == "resuming"
    assert not fresh.plan_shards([offer("B", 1)])
    fresh.reconcile("A", executor)
    fresh.finish_resume()
    assert fresh.snapshot()["created_at"] == 10
    assert fresh.snapshot()["remaining_seconds"] == 70
    assert fresh.snapshot()["remaining_runs"] == 9
    assert fresh.result(shard.leases[0].run.key) == evidence
    assert fresh.run_records()[0]["definition"] == before[0]["definition"]
    next_shard = fresh.plan_shards([offer("B", 1)])[0]
    assert next_shard.leases[0].run.key != shard.leases[0].run.key


def test_pause_between_lease_and_submit_keeps_same_unspent_lease(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    executor = FakeExecutor()
    assert control.request_pause() == "paused"
    assert control.submit_shard(shard, executor) is None
    assert executor.submissions == 0
    assert control.unsubmitted_shards() == (shard,)
    executor.offline = True
    assert control.reconcile("A", executor) == {}
    control.mark_unreachable("A", "no contact before submission")
    assert control.snapshot()["state"] == "paused"
    assert control.unsubmitted_shards() == (shard,)
    executor.offline = False
    control.begin_resume()
    control.finish_resume()
    assert control.submit_shard(shard, executor)
    assert executor.submissions == 1
    assert control.snapshot()["attempts_reserved"] == 1
    assert not control.unsubmitted_shards()


def test_original_clock_can_be_adopted_once_but_not_replaced(tmp_path: Path) -> None:
    from lambdaforge.controlplane.Fleet import Fleet, FleetMember

    fleet = Fleet("fleet", (FleetMember("A"),))
    control = StudyCoordinator(tmp_path / "adopted", clock=lambda: 100)
    control.initialize("science", "execution", fleet, max_time_seconds=80, created_at=40)
    assert control.snapshot()["remaining_seconds"] == 20
    with pytest.raises(ValueError, match="clock"):
        control.initialize("science", "execution", fleet, max_time_seconds=80, created_at=100)


def test_v1_state_migrates_without_changing_evidence_or_clock(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    control.ingest(envelope(shard.leases[0], "A"))
    evidence = control.result(shard.leases[0].run.key)
    state = json.loads(control.state_path.read_text())
    state["coordinator_version"] = 1
    state.pop("lifecycle")
    state.pop("lifecycle_history")
    control.state_path.write_text(json.dumps(state))
    assert control.snapshot()["coordinator_version"] == 2
    assert control.request_pause() == "paused"
    migrated = json.loads(control.state_path.read_text())
    assert migrated["coordinator_version"] == 2
    assert migrated["created_at"] == state["created_at"]
    assert migrated["runs"] == state["runs"]
    assert control.result(shard.leases[0].run.key) == evidence


def test_v2_missing_pause_state_fails_closed(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    state = json.loads(control.state_path.read_text())
    state.pop("lifecycle")
    control.state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="lifecycle"):
        control.snapshot()


def test_planner_frontier_withdrawal_is_atomic_and_never_revokes_workers(tmp_path: Path) -> None:
    control = coordinator(tmp_path, max_runs=10)
    first, second, third = (run(index, priority_class="required") for index in (1, 2, 3))
    control.enqueue([first, second, third])
    shard = control.plan_shards([offer("A", 1)])[0]
    active = shard.leases[0].run.key
    waiting = [item.key for item in (first, second, third) if item.key != active]
    before = control.run_records()
    with pytest.raises(ValueError, match="leased or terminal"):
        control.withdraw_planned([waiting[0], active], reason="planner time budget")
    assert control.run_records() == before
    control.withdraw_planned(waiting, reason="native planner ended the block")
    after = {item["run_key"]: item for item in control.run_records()}
    assert after[active]["state"] == "leased"
    assert all(after[key]["state"] == "cancelled" for key in waiting)
    assert all(not after[key]["attempts"] and not after[key]["accepted"] for key in waiting)
    assert control.snapshot()["remaining_runs"] == 9


def test_pause_and_resume_cannot_send_a_lease_after_original_time_budget(tmp_path: Path) -> None:
    control = coordinator(tmp_path, max_time_seconds=20)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    assert control.request_pause() == "paused"
    restarted = StudyCoordinator(control.root, clock=lambda: 31)
    restarted.begin_resume()
    restarted.finish_resume()
    executor = FakeExecutor()
    assert restarted.submit_shard(shard, executor) is None
    assert executor.submissions == 0
    assert restarted.snapshot()["remaining_seconds"] == 0
    assert restarted.unsubmitted_shards() == (shard,)


@pytest.mark.parametrize(
    "field,value",
    [("created_at", float("nan")), ("created_at", -1), ("max_time_seconds", float("nan"))],
)
def test_corrupt_original_clock_or_budget_cannot_authorize_resume(
    tmp_path: Path, field: str, value: float
) -> None:
    control = coordinator(tmp_path)
    state = json.loads(control.state_path.read_text())
    state[field] = value
    control.state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="refusing restart"):
        control.begin_resume()
