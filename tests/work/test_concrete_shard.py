"""Concrete shards reuse real isolated CPU Work execution, never a worker-side optimizer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.controlplane.Fleet import ClusterHealth, Fleet, FleetMember
from lambdaforge.controlplane.FleetPlacement import ClusterOffer, ExecutionEquivalence, GlobalRun
from lambdaforge.controlplane.StudyCoordinator import StudyCoordinator, StudyShard
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.work.shard import execute_concrete_shard


def prepared_shard(tmp_path: Path) -> tuple[StudyCoordinator, StudyShard, dict[str, Any]]:
    equivalence = ExecutionEquivalence("code", "env", "inputs", "fp32", "synthetic-cpu")
    control = StudyCoordinator(tmp_path / "coordinator", clock=lambda: 10)
    control.initialize("science", "execution", Fleet("cpu", (FleetMember("A", max_runs=4),)))
    runs = [
        GlobalRun(
            "science",
            str(candidate),
            seed,
            "search",
            {},
            {"quality": candidate / 10},
            equivalence,
            requires_gpu=False,
        )
        for candidate in (1, 2)
        for seed in (101, 102)
    ]
    control.enqueue(runs)
    shard = control.plan_shards(
        [
            ClusterOffer(
                "A",
                ClusterHealth.ONLINE,
                "local",
                "auto",
                0,
                100,
                slots=4,
                equivalence=equivalence,
                environment_ready=True,
                inputs_ready=True,
                local_admission_verified=True,
            )
        ]
    )[0]
    # A member's concurrent Run cap is not permission to add native invocations to its shard.
    assert len(shard.leases) == 4
    source = tmp_path / "science.yaml"
    source.write_text("name: science\nrun: tests.work_cases.ConfirmationFailureWork\n")
    invocations = {
        lease.run.key: {
            "definition": {
                "name": "science",
                "work_class": "tests.work_cases.ConfirmationFailureWork",
                "resources": ResourceRequest(cpu_cores=2).to_dict(),
                "objective": {"metric": "score", "mode": "max"},
            },
            "parameters": dict(lease.run.parameters),
            "trial_parameters": dict(lease.run.parameters),
            "seed": lease.run.seed,
            "trial_index": int(lease.run.candidate),
            "execution_id": "execution",
            "execution_dir": str(tmp_path / "not-worker-owned"),
            "source": str(source),
            "restart": False,
        }
        for lease in shard.leases
    }
    return control, shard, invocations


def test_real_cpu_children_keep_seed_identity_results_and_scientific_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, shard, invocations = prepared_shard(tmp_path)
    # Put one failing and one healthy seed in this concrete queue irrespective of key ordering.
    assert {lease.run.seed for lease in shard.leases} == {101, 102}
    root = tmp_path / "worker"
    from lambdaforge.work import runner

    def forbidden_planner(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A shard worker must not invoke a scientific planner")

    monkeypatch.setattr(runner, "_execute_adaptive_group", forbidden_planner)
    monkeypatch.setattr(runner, "_execute_fixed_evidence_group", forbidden_planner)
    arguments = dict(
        root=root,
        resources=ResourceRequest(cpu_cores=2),
        verified_equivalence=shard.leases[0].run.equivalence,
        parallelism=2,
    )
    results = execute_concrete_shard(shard, invocations, **arguments)
    assert {item["state"] for item in results} == {"completed", "failed"}
    for envelope in results:
        assert control.ingest(envelope)
        result = envelope["result"]
        assert result["seed"] == next(
            lease.run.seed for lease in shard.leases if lease.run.key == envelope["run_key"]
        )
        assert Path(result["run_dir"]).is_relative_to(root / "execution")
        assert (Path(result["run_dir"]) / "work.log").is_file()
        if envelope["state"] == "failed":
            assert result["failure"]["message"] == "synthetic confirmation failure"
        else:
            assert result["metrics"]["score"] == envelope["parameters"]["quality"]
    assert not list(root.rglob("decisions.jsonl"))
    assert not (tmp_path / "not-worker-owned").exists()
    # Acknowledgement loss/re-delivery does not execute consumer code a second time.
    retained_results = list(root.rglob("result.json"))
    assert execute_concrete_shard(shard, invocations, **arguments) == results
    assert list(root.rglob("result.json")) == retained_results
    state_path = root / "worker.json"
    state = json.loads(state_path.read_text())
    state["state"] = "running"
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="must be reconciled"):
        execute_concrete_shard(shard, invocations, **arguments)
    assert list(root.rglob("result.json")) == retained_results


def test_all_science_is_validated_before_any_child_launch(tmp_path: Path) -> None:
    _control, shard, invocations = prepared_shard(tmp_path)
    key = next(iter(invocations))
    invocations[key]["seed"] = 999
    root = tmp_path / "worker"
    with pytest.raises(ValueError, match="scientific lease"):
        execute_concrete_shard(
            shard,
            invocations,
            root=root,
            resources=ResourceRequest(cpu_cores=2),
            verified_equivalence=shard.leases[0].run.equivalence,
            parallelism=2,
        )
    assert not root.exists()


def test_gpu_execution_without_provider_grants_fails_closed(tmp_path: Path) -> None:
    _control, shard, invocations = prepared_shard(tmp_path)
    with pytest.raises(ValueError, match="granted provider"):
        execute_concrete_shard(
            shard,
            invocations,
            root=tmp_path / "worker",
            resources=ResourceRequest(cpu_cores=2, gpu_count=1),
            verified_equivalence=shard.leases[0].run.equivalence,
            parallelism=2,
        )


def test_an_interrupted_worker_is_never_blindly_reexecuted(tmp_path: Path) -> None:
    _control, shard, invocations = prepared_shard(tmp_path)
    # Positively reject an unknown recovery lease rather than silently starting Attempt 1.
    recovered = replace(shard, leases=(replace(shard.leases[0], attempt=2),))
    with pytest.raises(ValueError, match="Attempt binding"):
        execute_concrete_shard(
            recovered,
            {recovered.leases[0].run.key: invocations[recovered.leases[0].run.key]},
            root=tmp_path / "worker",
            resources=ResourceRequest(cpu_cores=2),
            verified_equivalence=shard.leases[0].run.equivalence,
            parallelism=1,
        )
    assert not (tmp_path / "worker").exists()


def test_foreign_execution_stratum_is_rejected_before_writing(tmp_path: Path) -> None:
    _control, shard, invocations = prepared_shard(tmp_path)
    with pytest.raises(ValueError, match="stratum"):
        execute_concrete_shard(
            shard,
            invocations,
            root=tmp_path / "worker",
            resources=ResourceRequest(cpu_cores=2),
            verified_equivalence=replace(shard.leases[0].run.equivalence, environment="changed"),
            parallelism=2,
        )
    assert not (tmp_path / "worker").exists()


def test_missing_worker_state_does_not_reset_an_existing_owned_root(tmp_path: Path) -> None:
    _control, shard, invocations = prepared_shard(tmp_path)
    root = tmp_path / "worker"
    root.mkdir()
    retained = root / "execution"
    retained.mkdir()
    with pytest.raises(ValueError, match="refusing reset"):
        execute_concrete_shard(
            shard,
            invocations,
            root=root,
            resources=ResourceRequest(cpu_cores=2),
            verified_equivalence=shard.leases[0].run.equivalence,
            parallelism=2,
        )
    assert retained.is_dir()
    assert not (root / "worker.json").exists()


def test_scientific_queue_cannot_inject_a_foreign_runtime_path(tmp_path: Path) -> None:
    _control, shard, invocations = prepared_shard(tmp_path)
    invocations[next(iter(invocations))]["hpo_stop_path"] = "/unowned/scientific-stop"
    with pytest.raises(ValueError, match="operational runtime fields"):
        execute_concrete_shard(
            shard,
            invocations,
            root=tmp_path / "worker",
            resources=ResourceRequest(cpu_cores=2),
            verified_equivalence=shard.leases[0].run.equivalence,
            parallelism=2,
        )
    assert not (tmp_path / "worker").exists()
