"""Durable multi-target ownership without a second scientific optimizer or real clusters."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.controlplane.Fleet import ClusterHealth, Fleet, FleetMember
from lambdaforge.controlplane.FleetPlacement import (
    ClusterOffer,
    ExecutionEquivalence,
    GlobalPlacementBroker,
    GlobalRun,
)
from lambdaforge.controlplane.StudyCoordinator import (
    ContradictoryEvidenceError,
    RemoteObservation,
    RunLease,
    ShardRejectedError,
    StudyCoordinator,
    StudyShard,
)

EQUIVALENCE = ExecutionEquivalence("code-v1", "env-v1", "data-v1", "fp32", "H200-sm90")


def run(candidate: int, seed: int = 0, **options: Any) -> GlobalRun:
    if options.get("priority_class") == "predictive":
        options.setdefault(
            "proposal",
            {
                "planner_revision": 1,
                "evidence_revision": 2,
                "model_revision": 1,
                "reason": "parameter support debt",
                "policy": "test-lookahead-v1",
            },
        )
    return GlobalRun(
        "science", str(candidate), seed, "search", {}, {"x": candidate}, EQUIVALENCE, **options
    )


def offer(cluster: str, slots: int, **options: Any) -> ClusterOffer:
    return ClusterOffer(
        cluster,
        ClusterHealth.ONLINE,
        "local",
        "shared",
        0,
        1000,
        slots=slots,
        admissible_gpus=slots,
        free_memory_bytes=80 * 1024**3,
        equivalence=EQUIVALENCE,
        environment_ready=True,
        inputs_ready=True,
        local_admission_verified=True,
        **options,
    )


def envelope(lease: RunLease, cluster: str) -> dict[str, Any]:
    return {
        "study_identity": lease.run.study_identity,
        "run_key": lease.run.key,
        "attempt": lease.attempt,
        "lease_id": lease.lease_id,
        "cluster": cluster,
        "parameters": dict(lease.run.parameters),
        "equivalence": lease.run.equivalence.to_dict(),
        "state": "completed",
        "result": {"objective": int(lease.run.candidate) / 100},
    }


class FakeExecutor:
    """A durable local authority; transport failures do not interrupt its resident work."""

    def __init__(self) -> None:
        self.shards: dict[str, StudyShard] = {}
        self.observations: dict[str, tuple[RemoteObservation, ...]] = {}
        self.offline = False
        self.drop_acknowledgement = False
        self.submissions = 0

    def submit(self, shard: StudyShard) -> str:
        self.submissions += 1
        self.shards.setdefault(shard.shard_id, shard)
        if self.drop_acknowledgement:
            raise TimeoutError("Acknowledgement lost after remote acceptance")
        return "job-" + shard.shard_id

    def finish(self, shard: StudyShard) -> None:
        self.observations[shard.shard_id] = tuple(
            RemoteObservation(
                lease.run.key,
                lease.attempt,
                lease.lease_id,
                "completed",
                result=envelope(lease, shard.cluster),
            )
            for lease in shard.leases
        )

    def observe(self, shard: StudyShard, job_id: str | None) -> tuple[RemoteObservation, ...]:
        if self.offline:
            raise ConnectionError("Network partition")
        return self.observations.get(
            shard.shard_id,
            tuple(
                RemoteObservation(lease.run.key, lease.attempt, lease.lease_id, "running")
                for lease in shard.leases
            ),
        )


def coordinator(tmp_path: Path, **budgets: Any) -> StudyCoordinator:
    result = StudyCoordinator(tmp_path / "owned", clock=lambda: 10.0)
    result.initialize(
        "science",
        "execution",
        Fleet("fleet", tuple(FleetMember(name) for name in "ABC")),
        **budgets,
    )
    return result


def test_fixed_224_unique_runs_across_three_clusters(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    required = [run(candidate, seed) for candidate in range(56) for seed in range(4)]
    control.enqueue(required)
    control.enqueue(required)  # Re-delivery is not a second evidence design.
    offers = [offer("A", 2), offer("B", 3), offer("C", 1)]
    executors = {name: FakeExecutor() for name in "ABC"}
    placements: Counter[str] = Counter()
    attempts = set()
    while control.snapshot()["counts"].get("planned", 0):
        shards = control.plan_shards(offers)
        assert shards
        for shard in shards:
            placements[shard.cluster] += len(shard.leases)
            for lease in shard.leases:
                assert (lease.run.key, lease.attempt) not in attempts
                attempts.add((lease.run.key, lease.attempt))
            executor = executors[shard.cluster]
            control.submit_shard(shard, executor)
            executor.finish(shard)
            control.reconcile(shard.cluster, executor)
    assert set(placements) == set("ABC")
    assert len(attempts) == 224
    assert control.snapshot()["counts"] == {"completed": 224}
    assert control.snapshot()["study_identity"] == "science"
    assert {item.key: control.result(item.key)["result"]["objective"] for item in required} == {
        item.key: int(item.candidate) / 100 for item in required
    }


def test_partition_restart_reconciles_running_finished_lost_without_duplicates(
    tmp_path: Path,
) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(index) for index in range(6)])
    shard = control.plan_shards([offer("B", 3)])[0]
    executor = FakeExecutor()
    control.submit_shard(shard, executor)
    executor.offline = True
    assert control.reconcile("B", executor) == {"unreachable": 1}
    assert control.snapshot()["counts"]["unknown_remote"] == 3
    assert control.snapshot()["members"]["B"]["health"] == "unreachable"
    assert not control.retry_lost(shard.leases[0].run.key)
    # A different driver process/object loads only durable state.
    restarted = StudyCoordinator(control.root, clock=lambda: 20.0)
    restarted.initialize(
        "science", "execution", Fleet("fleet", tuple(FleetMember(name) for name in "ABC"))
    )
    other = restarted.plan_shards([offer("A", 2), offer("C", 1)])
    assert not {lease.run.key for lease in shard.leases} & {
        lease.run.key for assigned in other for lease in assigned.leases
    }
    first, second, third = shard.leases
    executor.offline = False
    executor.observations[shard.shard_id] = (
        RemoteObservation(first.run.key, first.attempt, first.lease_id, "running"),
        RemoteObservation(
            second.run.key, second.attempt, second.lease_id, "completed", envelope(second, "B")
        ),
        RemoteObservation(
            third.run.key,
            third.attempt,
            third.lease_id,
            "lost",
            loss_proof={
                "authority": "owned-shard-store",
                "no_live_owner": True,
                "lease_id": third.lease_id,
            },
        ),
    )
    restarted.reconcile("B", executor)
    assert restarted.result(second.run.key) is not None
    assert not restarted.retry_lost(first.run.key)
    assert restarted.retry_lost(third.run.key)
    replacement = restarted.plan_shards([offer("B", 1, acknowledged_leases=(first.lease_id,))])[0]
    assert replacement.leases[0].run.key == third.run.key
    assert replacement.leases[0].attempt == 2
    assert restarted.snapshot()["members"]["B"]["health"] == "online"


def test_idempotent_ingestion_quarantines_conflicting_or_wrong_environment(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    evidence = envelope(shard.leases[0], "A")
    assert control.ingest(evidence)
    assert not control.ingest(evidence)
    contradictory = {**evidence, "result": {"objective": 99}}
    with pytest.raises(ContradictoryEvidenceError, match="quarantined"):
        control.ingest(contradictory)
    assert control.result(shard.leases[0].run.key) == evidence
    assert control.snapshot()["quarantined"] == 1
    assert len(list((control.root / "quarantine").glob("*.json"))) == 1


def test_unacknowledged_leases_do_not_overbook_repeated_offer(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(index) for index in range(4)])
    shard = control.plan_shards([offer("A", 2)])[0]
    assert len(shard.leases) == 2
    assert not control.plan_shards([offer("A", 2)])


def test_ambiguous_submission_is_reconciled_not_resubmitted(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    executor = FakeExecutor()
    executor.drop_acknowledgement = True
    assert control.submit_shard(shard, executor) is None
    with pytest.raises(RuntimeError, match="must be reconciled"):
        control.submit_shard(shard, executor)
    executor.finish(shard)
    control.reconcile("A", executor)
    assert executor.submissions == 1
    assert control.snapshot()["counts"] == {"completed": 1}


def test_drain_budgets_and_predictive_withdrawal(tmp_path: Path) -> None:
    control = coordinator(tmp_path, max_runs=2)
    control.enqueue([run(1), run(2, priority_class="predictive"), run(3)])
    assert control.cancel_planned(run(2).key, reason="new-evidence-no-longer-useful")
    control.set_member_state("B", ClusterHealth.DRAINING)
    shards = control.plan_shards([offer("A", 1), offer("B", 3), offer("C", 1)])
    assert {shard.cluster for shard in shards} == {"A", "C"}
    assert control.snapshot()["attempts_reserved"] == 2
    for shard in shards:
        assert not control.cancel_planned(shard.leases[0].run.key, reason="must-continue")
    control.enqueue([run(4)])
    assert not control.plan_shards([offer("A", 5)])
    assert "run-budget-exhausted" in str(control.snapshot()["idle_reasons"])


@pytest.mark.parametrize(
    "changed,reason",
    [
        ({"equivalence": replace(EQUIVALENCE, hardware="A100")}, "execution-equivalence-mismatch"),
        ({"local_admission_verified": False}, "local-admission-unverified"),
        ({"free_memory_bytes": 1}, "memory-fit-unknown-or-insufficient"),
        ({"valid_until": 5}, "offer-stale"),
        ({"inputs_ready": False}, "environment-or-inputs-unavailable"),
    ],
)
def test_hard_filters_never_infer_permission(changed: dict[str, Any], reason: str) -> None:
    broker = GlobalPlacementBroker()
    chosen, excluded = broker.place(
        run(1, gpu_memory_bytes=20),
        [replace(offer("A", 2), **changed)],
        {"A": FleetMember("A")},
        now=10,
    )
    assert chosen is None
    assert reason in excluded["A"]


@pytest.mark.parametrize("available", [None, 99, 100])
def test_placement_requires_real_storage_fit(available: int | None) -> None:
    requested = run(1, storage_bytes=100)
    assert GlobalRun.from_mapping(requested.to_dict()) == requested
    assert requested.key == run(1, storage_bytes=200).key  # Operational, not scientific identity.
    selected, excluded = GlobalPlacementBroker().place(
        requested,
        [offer("A", 1, admissible_storage_bytes=available)],
        {"A": FleetMember("A")},
        now=10,
    )
    assert (selected is not None) is (available == 100)
    if selected is None:
        assert "storage-fit-unknown-or-insufficient" in excluded["A"]


def test_caps_command_slurm_and_checkpoint_locality() -> None:
    broker = GlobalPlacementBroker()
    requested = run(1, checkpoint_locations=("B",))
    direct = offer("A", 2)
    command = replace(offer("B", 1), gpu_access="command")
    scheduler = replace(offer("C", 1), scheduler="slurm", gpu_access="scheduler")
    chosen, excluded = broker.place(
        requested, [direct, command, scheduler], {name: FleetMember(name) for name in "ABC"}, now=10
    )
    assert chosen and chosen.cluster == "B"
    assert excluded["A"] == ("checkpoint-transfer-required",)
    assert chosen.reason["local_admission_recheck_required"]
    no_cap, reasons = broker.place(run(2), [direct], {"A": FleetMember("A", max_gpus=1)}, now=10)
    assert no_cap is None
    assert "gpu-cap" in str(reasons)
    assert requested.to_dict()["parameters"] == {"x": 1}


def test_time_budget_and_corrupt_restart_fail_closed(tmp_path: Path) -> None:
    control = coordinator(tmp_path, max_time_seconds=1)
    control.enqueue([run(1)])
    control.clock = lambda: 20
    assert not control.plan_shards([offer("A", 2)])
    assert "time-budget-exhausted" in str(control.snapshot()["idle_reasons"])
    control.state_path.write_text(json.dumps({"coordinator_version": 99}))
    with pytest.raises(ValueError, match="unsupported"):
        control.snapshot()


def test_global_key_ignores_placement_but_preserves_seed_phase_fidelity() -> None:
    item = run(1)
    assert replace(item, checkpoint_locations=("elsewhere",)).key == item.key
    assert replace(item, seed=3).key != item.key
    assert replace(item, phase="confirmation").key != item.key
    assert replace(item, fidelity={"target": 10}).key != item.key
    with pytest.raises(TypeError):
        item.parameters["x"] = 2


def test_competing_drivers_never_reserve_same_attempt(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(index) for index in range(10)])

    def dispatch() -> tuple[StudyShard, ...]:
        return StudyCoordinator(control.root, clock=lambda: 10).plan_shards([offer("A", 2)])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: dispatch(), range(2)))
    assert sum(len(shard.leases) for shards in results for shard in shards) == 2


def test_missing_observation_stays_unknown_not_lost(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    executor = FakeExecutor()
    control.submit_shard(shard, executor)
    executor.observations[shard.shard_id] = ()
    assert control.reconcile("A", executor) == {"unknown_remote": 1}
    assert control.snapshot()["counts"] == {"unknown_remote": 1}
    assert not control.retry_lost(shard.leases[0].run.key)


def test_wrong_environment_is_quarantined_then_correct_result_can_arrive(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    correct = envelope(shard.leases[0], "A")
    wrong = {**correct, "equivalence": replace(EQUIVALENCE, environment="different").to_dict()}
    with pytest.raises(ContradictoryEvidenceError):
        control.ingest(wrong)
    assert control.result(shard.leases[0].run.key) is None
    assert control.ingest(correct)


def test_lost_requires_owned_attempt_proof(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    lease = shard.leases[0]
    executor = FakeExecutor()
    control.submit_shard(shard, executor)
    executor.observations[shard.shard_id] = (
        RemoteObservation(
            lease.run.key,
            lease.attempt,
            lease.lease_id,
            "lost",
            loss_proof={"authority": "scheduler"},
        ),
    )
    with pytest.raises(ValueError, match="not proof"):
        control.reconcile("A", executor)
    assert not control.retry_lost(lease.run.key)


def test_predictive_provenance_is_required() -> None:
    with pytest.raises(ValueError, match="provenance"):
        replace(run(1), priority_class="predictive")


def test_disabled_between_reservation_and_submission_does_not_launch(tmp_path: Path) -> None:
    control = coordinator(tmp_path, max_runs=1)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    control.set_member_state("A", ClusterHealth.DISABLED)
    executor = FakeExecutor()
    assert control.submit_shard(shard, executor) is None
    assert executor.submissions == 0
    assert control.snapshot()["attempts_reserved"] == 0
    replacement = control.plan_shards([offer("B", 1)])[0]
    assert replacement.leases[0].attempt == 2


def test_removed_members_drain_existing_leases_without_identity_changes(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("B", 1)])[0]
    executor = FakeExecutor()
    control.submit_shard(shard, executor)
    control.update_fleet(Fleet("fleet", (FleetMember("A"), FleetMember("C"))))
    assert control.snapshot()["members"]["B"]["state"] == "draining"
    assert control.snapshot()["study_identity"] == "science"
    executor.finish(shard)
    control.reconcile("B", executor)
    assert control.result(shard.leases[0].run.key)


def test_scheduler_rejection_is_not_connectivity_or_scientific_evidence(tmp_path: Path) -> None:
    control = coordinator(tmp_path, max_runs=1)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]

    class RejectedExecutor(FakeExecutor):
        def submit(self, shard: StudyShard) -> str:
            raise ShardRejectedError("Scheduler rejected the allocation")

    assert control.submit_shard(shard, RejectedExecutor()) is None
    assert control.snapshot()["members"]["A"]["health"] == "degraded"
    assert control.snapshot()["counts"] == {"failed": 1}
    assert control.snapshot()["attempts_reserved"] == 0
    assert control.result(shard.leases[0].run.key) is None
    assert not control.retry_lost(shard.leases[0].run.key)


@pytest.mark.parametrize("observed_state", ["running", "completed"])
def test_delayed_submission_ack_never_regresses_observed_evidence(
    tmp_path: Path, observed_state: str
) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    lease = shard.leases[0]

    class DelayedAcknowledgement(FakeExecutor):
        def submit(self, shard: StudyShard) -> str:
            job_id = super().submit(shard)
            if observed_state == "completed":
                self.finish(shard)
            control.reconcile("A", self)
            return job_id

    assert control.submit_shard(shard, DelayedAcknowledgement())
    assert control.snapshot()["counts"] == {observed_state: 1}
    if observed_state == "completed":
        assert control.result(lease.run.key) == envelope(lease, "A")
    else:
        executor = FakeExecutor()
        executor.observations[shard.shard_id] = (
            RemoteObservation(lease.run.key, lease.attempt, lease.lease_id, "queued"),
        )
        control.reconcile("A", executor)
        assert control.snapshot()["counts"] == {"running": 1}


def test_old_ack_does_not_modify_a_positively_lost_attempt_replacement(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    control.enqueue([run(1)])
    shard = control.plan_shards([offer("A", 1)])[0]
    lease = shard.leases[0]

    class LostDuringAcknowledgement(FakeExecutor):
        def submit(self, shard: StudyShard) -> str:
            job_id = super().submit(shard)
            self.observations[shard.shard_id] = (
                RemoteObservation(
                    lease.run.key,
                    lease.attempt,
                    lease.lease_id,
                    "lost",
                    loss_proof={
                        "authority": "scheduler",
                        "lease_id": lease.lease_id,
                        "no_live_owner": True,
                    },
                ),
            )
            control.reconcile("A", self)
            assert control.retry_lost(lease.run.key)
            replacement = control.plan_shards([offer("B", 1)])[0]
            assert replacement.leases[0].attempt == 2
            return job_id

    old_job = control.submit_shard(shard, LostDuringAcknowledgement())
    value = json.loads(control.state_path.read_text())
    attempts = value["runs"][lease.run.key]["attempts"]
    assert attempts[0]["state"] == "lost"
    assert attempts[0]["job_id"] == old_job
    assert attempts[1]["state"] == "leased"
    assert attempts[1]["job_id"] is None
    assert control.snapshot()["counts"] == {"leased": 1}


def test_restart_rejects_a_different_coordinator_authority(tmp_path: Path) -> None:
    control = coordinator(tmp_path)
    with pytest.raises(ValueError, match="Restart cannot change"):
        control.initialize(
            "science",
            "execution",
            Fleet("fleet", (FleetMember("A"),), coordinator="B"),
        )


@pytest.mark.parametrize("field", ["max_runs", "quarantine", "members", "created_at"])
def test_incomplete_state_never_silently_reinitializes(tmp_path: Path, field: str) -> None:
    control = coordinator(tmp_path)
    value = json.loads(control.state_path.read_text())
    del value[field]
    control.state_path.write_text(json.dumps(value))
    before = control.state_path.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        control.snapshot()
    assert control.state_path.read_bytes() == before


def test_shard_rejects_duplicate_or_foreign_scientific_leases() -> None:
    lease = RunLease(run(1), 1, "lease-1")
    with pytest.raises(ValueError, match="unique"):
        StudyShard("shard", "science", "A", (lease, lease), FleetMember("A"))
    with pytest.raises(ValueError, match="another Study"):
        StudyShard("shard", "different-science", "A", (lease,), FleetMember("A"))
    with pytest.raises(ValueError, match="Fleet member"):
        StudyShard("shard", "science", "B", (lease,), FleetMember("A"))


@pytest.mark.parametrize("parameters", [{"x": float("nan")}, {"x": object()}])
def test_scientific_payloads_are_strict_json_before_enqueue(parameters: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(run(1), parameters=parameters)


@pytest.mark.parametrize("field", ["environment_ready", "inputs_ready", "local_admission_verified"])
def test_readiness_requires_a_boolean_attestation(field: str) -> None:
    with pytest.raises(TypeError, match="explicit boolean"):
        replace(offer("A", 1), **{field: "false"})
