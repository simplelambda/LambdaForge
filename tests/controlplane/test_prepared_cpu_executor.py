"""Prepared-provider preflight/fencing tests never infer acceptance or physical grants."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.PreparedCpuShardExecutor import PreparedCpuShardExecutor
from lambdaforge.controlplane.SchedulerSubmission import SchedulerSubmission
from lambdaforge.controlplane.StudyCoordinator import ShardRejectedError
from lambdaforge.execution.ResourceRequest import ResourceRequest
from tests.work.test_concrete_shard import prepared_shard


class Scheduler:
    """Only model the provider ACK boundary; these are not end-to-end execution tests."""

    def __init__(self) -> None:
        self.submissions = 0
        self.drop_ack = False

    def submit(self, command: Any, resources: Any, **options: Any) -> SchedulerSubmission:
        self.submissions += 1
        if self.drop_ack:
            raise TimeoutError("accepted but acknowledgement lost")
        return SchedulerSubmission(options["job_id"], JobState.STAGING)

    def state(self, job_id: str) -> JobState:
        return JobState.UNKNOWN


class Factory:
    """Never open a provider connection."""

    def __init__(self, provider: Scheduler) -> None:
        self.provider = provider

    def transport(self, profile: Any) -> None:
        return None

    def scheduler(self, profile: Any, transport: Any) -> Scheduler:
        return self.provider


def prepared(tmp_path: Path) -> tuple[Any, Any, Any, Any, Scheduler]:
    control, shard, invocations = prepared_shard(tmp_path)
    provider = Scheduler()
    executor = PreparedCpuShardExecutor(
        ClusterProfile("A", python=sys.executable, workspace=str(tmp_path / "host")),
        shard.member,
        root=tmp_path / "executor",
        resources=ResourceRequest(cpu_cores=4),
        equivalence=shard.leases[0].run.equivalence,
        invocation=lambda key: invocations[key],
        factory=Factory(provider),  # type: ignore[arg-type]
    )
    return control, shard, invocations, executor, provider


def test_invalid_invocation_is_rejected_before_submit_or_storage(tmp_path: Path) -> None:
    _control, shard, invocations, executor, provider = prepared(tmp_path)
    invocations[shard.leases[0].run.key]["seed"] = 999
    with pytest.raises(ShardRejectedError, match="preflight"):
        executor.submit(shard)
    assert provider.submissions == 0
    assert not executor.root.exists()


def test_ambiguous_provider_ack_never_repeats_submission(tmp_path: Path) -> None:
    control, shard, _invocations, executor, provider = prepared(tmp_path)
    before = executor.offer(control.run_records())
    assert not before.acknowledged_leases
    provider.drop_ack = True
    with pytest.raises(TimeoutError):
        executor.submit(shard)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        executor.submit(shard)
    assert provider.submissions == 1
    assert not executor.offer(control.run_records()).acknowledged_leases
    assert all(item.state == "unknown_remote" for item in executor.observe(shard, None))


def test_ack_receipt_is_durable_and_repeated_submit_is_idempotent(tmp_path: Path) -> None:
    control, shard, _invocations, executor, provider = prepared(tmp_path)
    job_id = executor.submit(shard)
    assert executor.submit(shard) == job_id
    assert provider.submissions == 1
    offer = executor.offer(control.run_records())
    assert set(offer.acknowledged_leases) == {lease.lease_id for lease in shard.leases}
    assert offer.slots == 0
    assert offer.admissible_gpus == 0
    receipt = json.loads((executor.root / shard.shard_id / "submission.json").read_text())
    assert receipt == {"job_id": job_id, "acknowledged": True}


def test_remote_or_gpu_profiles_never_get_prepared_cpu_attestation(tmp_path: Path) -> None:
    _control, shard, invocations, _executor, provider = prepared(tmp_path)
    for profile, resources, message in (
        (
            ClusterProfile("A", transport="ssh", host="not-contacted", workspace="/remote"),
            ResourceRequest(),
            "local direct",
        ),
        (
            ClusterProfile("A", python=sys.executable),
            ResourceRequest(gpu_count=1),
            "cannot claim GPU",
        ),
        (ClusterProfile("A", python="unverified-python"), ResourceRequest(), "verified current"),
    ):
        with pytest.raises(ValueError, match=message):
            PreparedCpuShardExecutor(
                profile,
                shard.member,
                root=tmp_path / "refused",
                resources=resources,
                equivalence=shard.leases[0].run.equivalence,
                invocation=lambda key: invocations[key],
                factory=Factory(provider),  # type: ignore[arg-type]
            )
    assert provider.submissions == 0
    assert not (tmp_path / "refused").exists()
