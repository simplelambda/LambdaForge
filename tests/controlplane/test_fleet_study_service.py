"""Public Fleet preparation/dispatch contract with real loopback direct providers.

Installation is a fixture, not an assertion of SSH/SLURM production readiness.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from lambdaforge.cli.parser import build_parser
from lambdaforge.configuration.ConfigurationDescriptor import ConfigurationDescriptor
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.Fleet import Fleet, FleetMember
from lambdaforge.controlplane.FleetStudyService import execute_fleet, fleet_preflight
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.ResearchWork import aggregate_research_work
from lambdaforge.controlplane.StudyExportService import StudyExportService
from lambdaforge.controlplane.SubmissionService import SubmissionService
from lambdaforge.controlplane.WorkService import WorkService
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.work import WorkConfig
from lambdaforge.work.atomic import atomic_write_json
from tests.controlplane.test_prepared_provider_dispatch import (
    BundleFixture,
    LoopbackFactory,
    RuntimeFixture,
    TorchFixture,
)


def setup_fleet(tmp_path: Path, *, observable: bool = False) -> tuple[Path, Fleet, ControlPlane]:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="fleet-product-test"\nversion="1"\n'
        '[tool.lambdaforge]\nproject_id="fleet-product-test"\n'
    )
    source = tmp_path / "science.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "name": "fleet-science",
                "run": (
                    "tests.work_cases.FleetObservableWork"
                    if observable
                    else "tests.work_cases.AdaptiveScoreWork"
                ),
                "with": {"release": str(tmp_path / "release")} if observable else {},
                "sweep": {"space": {"quality": {"values": [0.2, 0.8]}}},
                "seeds": [4, 7],
                "resources": {"cpu": 2},
                "execution": {"max_parallel": 2},
                "objective": {"metric": "score", "mode": "max", "range": [0, 1]},
            }
        )
    )
    import_root = str(Path(__file__).resolve().parents[2])
    profiles = {
        name: ClusterProfile(
            name,
            python=sys.executable,
            environment="managed",
            workspace=str(tmp_path / ("provider-" + name)),
            command_prefix=("env", "PYTHONPATH=" + import_root),
        )
        for name in ("A", "B")
    }
    profiles["local"] = ClusterProfile(
        "local", python=sys.executable, workspace=str(tmp_path / "coordinator-provider")
    )
    fleet = Fleet("test", (FleetMember("A", max_runs=2), FleetMember("B", max_runs=2)))
    catalog = ClusterCatalog(profiles, fleets={fleet.name: fleet})
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "config.yaml").write_bytes(source.read_bytes())
    atomic_write_json(
        directory / "manifest.json",
        {
            "code_identity": {"provider": "explicit", "revision": "test-code"},
        },
    )
    bundle = ExecutionBundle(
        "bundle-test",
        directory,
        directory / "config.yaml",
        directory / "manifest.json",
        100,
        "env-test",
    )
    factory = LoopbackFactory()
    jobs = JobService(catalog, JobStore(tmp_path / "jobs"), factory)
    plane = ControlPlane(
        catalog, jobs, BundleFixture(bundle), factory, TorchFixture(), RuntimeFixture()
    )  # type: ignore[arg-type]
    return source, fleet, plane


@pytest.mark.skipif(os.name != "posix", reason="Detached direct provider requires POSIX")
def test_product_service_one_science_and_analysis_across_real_member_allocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, fleet, plane = setup_fleet(tmp_path, observable=True)
    parent = "job-test-fleet-coordinator"
    descriptor = ConfigurationDescriptor.from_path(source)
    plane.jobs.reserve(
        cluster="local",
        resources=ResourceRequest(cpu_cores=1),
        config_path=source,
        job_id=parent,
        metadata={**descriptor.metadata(), "fleet": "test"},
    )
    # Native supervisor publishes Study telemetry beside the parent's work directory.
    record = plane.jobs.store.get(parent)
    owner_root = tmp_path / "coordinator-provider" / "jobs" / parent
    record = record.with_updates(
        work_dir=str(owner_root / "work"),
        metadata={**dict(record.metadata), "local_storage": {"run_root": str(owner_root.parent)}},
    )
    plane.jobs.store.write(record)
    monkeypatch.setenv("LAMBDAFORGE_STUDY_PATH", str(Path(record.work_dir).parent / "study"))
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", parent)
    from lambdaforge.work.study import StudyTelemetry, study_run_key

    live_reads: list[dict[str, Any]] = []
    original_observation = StudyTelemetry.remote_run_observed

    def observed(telemetry: StudyTelemetry, specification: Any, observation: Any) -> None:
        original_observation(telemetry, specification, observation)
        if (
            live_reads
            or observation.get("state") != "running"
            or observation.get("latest_step") != 7
        ):
            return
        telemetry.refresh()
        detail = plane.jobs.study_run(parent, study_run_key(specification))
        assert detail["state"] == "running"
        assert detail["current_step"] == 7
        assert "live Fleet scientific output" in detail["log"]
        assert detail["latest_metrics"]["score"] == specification["parameters"]["quality"]
        live_reads.append(detail)
        (tmp_path / "release").write_text("read the live owner")

    monkeypatch.setattr(StudyTelemetry, "remote_run_observed", observed)
    try:
        result = execute_fleet(source, fleet, plane, parent_job_id=parent)
        assert live_reads, "The Run view must be available before its terminal result"
        assert result["status"] == "succeeded"
        assert len(result["runs"]) == 4
        roots = list((plane.jobs.store.root / "fleets").glob("*/coordinator/coordinator.json"))
        assert len(roots) == 1
        state = json.loads(roots[0].read_text())
        assert len(state["runs"]) == 4
        assert {item["attempts"][-1]["cluster"] for item in state["runs"].values()} == {"A", "B"}
        assert all(item["accepted"] for item in state["runs"].values())
        assert len(plane.jobs.store.records()) == 3
        assert len(aggregate_research_work(plane.jobs.store.records())) == 1
        assert len(list((plane.jobs.store.root / "fleets").rglob("analysis.json"))) == 1
        assert not list((tmp_path / "provider-A").rglob("analysis.json"))
        assert not list((tmp_path / "provider-B").rglob("analysis.json"))
        detail = plane.jobs.study_run(parent, "trial-00001-seed-4")
        assert detail["placement"]["cluster"] in {"A", "B"}
        assert detail["cluster"] == detail["placement"]["cluster"]
        assert detail["state"] == "succeeded"
        assert detail["latest_metrics"]
        assert all(
            item["job_id"] == detail["placement"]["job_id"]
            for item in state["shards"].values()
            if item["cluster"] == detail["placement"]["cluster"]
        )
        # A member is never replayed as an independent Work or removed while its parent
        # references it. The semantic cancel targets every exact owned allocation.
        with pytest.raises(ValueError, match="references|retained"):
            plane.jobs.delete(detail["placement"]["job_id"])
        stopped = WorkService(plane.catalog, jobs=plane.jobs).cancel(parent)
        assert stopped["status"] in {"cancelled", "reconciled"}
    finally:
        for record in plane.jobs.store.records():
            if not record.state.terminal:
                plane.jobs.cancel(record.job_id)


def test_public_fleet_grammar_and_read_only_preflight(tmp_path: Path) -> None:
    source, fleet, plane = setup_fleet(tmp_path)
    args = build_parser().parse_args(["run", str(source), "--on-fleet", "test", "--dry-run"])
    assert args.on_fleet == "test"
    preview = fleet_preflight(WorkConfig.from_yaml(source), plane.catalog, fleet)
    assert preview["dispatch"] is False
    assert len(preview["members"]) == 2
    assert plane.jobs.store.records() == ()
    with pytest.raises(ValueError, match="not allowed"):
        build_parser().parse_args(["run", str(source), "--on-fleet", "test", "--on", "A"])


def test_detached_fleet_request_captures_profiles_not_secrets_or_manual_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, fleet, plane = setup_fleet(tmp_path)
    launches = []

    def launch(*args: Any, **kwargs: Any) -> None:
        launches.append((args, kwargs))

    module = importlib.import_module("lambdaforge.controlplane.SubmissionService")
    monkeypatch.setattr(
        module, "subprocess", SimpleNamespace(Popen=launch, DEVNULL=module.subprocess.DEVNULL)
    )
    handle = SubmissionService(plane.catalog, plane.jobs).enqueue(
        source,
        cluster="local",
        fleet=fleet.name,
    )
    request = json.loads(
        (plane.jobs.store.root / "submissions" / handle.job_id / "request.json").read_text()
    )
    assert request["fleet"]["name"] == "test"
    assert set(request["member_profiles"]) == {"A", "B", "local"}
    assert launches[0][1]["start_new_session"] is True
    assert plane.jobs.store.get(handle.job_id).metadata["execution_target"] == "fleet:test"
    assert aggregate_research_work(plane.jobs.store.records())[0].cluster == "fleet:test"
    assert ResourceRequest.from_mapping(request["resources"]).gpu_count == 0


def test_member_preparation_failure_drains_every_created_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, fleet, plane = setup_fleet(tmp_path)
    drained: list[str] = []

    class FailedPreparation:
        def __init__(self, profile: Any, member: Any, **kwargs: Any) -> None:
            self.member = member

        def start_allocation(self, *args: Any, **kwargs: Any) -> None:
            if self.member.cluster == "B":
                raise RuntimeError("Provider preparation failed")

        def drain_allocation(self) -> None:
            drained.append(self.member.cluster)
            if self.member.cluster == "A":
                raise ValueError("Corrupt owner receipt cannot stop other drain requests")

    module = importlib.import_module("lambdaforge.controlplane.FleetStudyService")
    monkeypatch.setattr(module, "PreparedShardExecutor", FailedPreparation)
    with pytest.raises(RuntimeError, match="Provider preparation failed"):
        execute_fleet(source, fleet, plane, parent_job_id="job-test-fleet-coordinator")
    assert drained == ["A", "B"]


def test_unsupported_fleet_recovery_and_export_do_not_fall_back_to_one_cluster(
    tmp_path: Path,
) -> None:
    from lambdaforge.controlplane.jobs import JobState

    source, fleet, plane = setup_fleet(tmp_path)
    descriptor = ConfigurationDescriptor.from_path(source)
    handle = plane.jobs.reserve(
        cluster="local",
        resources=ResourceRequest(cpu_cores=1),
        config_path=source,
        metadata={**descriptor.metadata(), "fleet": fleet.name, "execution_target": "fleet:test"},
    )
    record = plane.jobs.store.get(handle.job_id).with_updates(state=JobState.FAILED)
    plane.jobs.store.write(record)
    with pytest.raises(ValueError, match="Fleet recovery"):
        plane.jobs.retry(handle.job_id)
    with pytest.raises(ValueError, match="omit member evidence"):
        StudyExportService(plane.catalog, jobs=plane.jobs).export(
            handle.job_id, tmp_path / "export"
        )
    assert len(plane.jobs.store.records()) == 1
    assert not (tmp_path / "export").exists()


def test_public_fidelity_fleet_fails_before_any_owned_allocation(tmp_path: Path) -> None:
    source, fleet, plane = setup_fleet(tmp_path)
    authored = yaml.safe_load(source.read_text())
    authored.pop("sweep")
    authored["search"] = {"trials": 3, "quality": {"values": [0.2, 0.5, 0.8]}}
    authored["search"]["fidelity"] = {"min": 3, "max": 9}
    source.write_text(yaml.safe_dump(authored))
    with pytest.raises(ValueError, match="checkpoint continuation"):
        fleet_preflight(WorkConfig.from_yaml(source), plane.catalog, fleet)
    assert plane.jobs.store.records() == ()


@pytest.mark.skipif(os.name != "posix", reason="Detached direct provider requires POSIX")
@pytest.mark.parametrize("confirmation", [False, True])
def test_public_adaptive_fleet_uses_one_native_optimizer_and_complete_scalar_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    confirmation: bool,
) -> None:
    source, fleet, plane = setup_fleet(tmp_path)
    authored = yaml.safe_load(source.read_text())
    authored.pop("sweep")
    authored["search"] = {
        "trials": 3,
        "startup_trials": 2,
        "confirmation_seeds": [101, 102] if confirmation else [],
        "quality": {"values": [0.2, 0.5, 0.8]},
    }
    source.write_text(yaml.safe_dump(authored))
    # Prepared bundle fixture must reflect the same authored source used by preflight.
    (tmp_path / "bundle" / "config.yaml").write_bytes(source.read_bytes())
    parent = "job-adaptive-fleet-coordinator"
    plane.jobs.reserve(
        cluster="local",
        resources=ResourceRequest(cpu_cores=1),
        config_path=source,
        job_id=parent,
        metadata={"fleet": "test"},
    )
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", parent)
    from lambdaforge.controlplane.PreparedShardExecutor import PreparedShardExecutor

    original_call = PreparedShardExecutor._member_call
    disconnected = False

    def flaky_stream(executor: PreparedShardExecutor, *arguments: str) -> dict[str, Any]:
        nonlocal disconnected
        if "--stream" in arguments and not disconnected:
            disconnected = True
            raise ConnectionError("synthetic metric-stream partition; owner still running")
        return original_call(executor, *arguments)

    monkeypatch.setattr(PreparedShardExecutor, "_member_call", flaky_stream)
    result = execute_fleet(source, fleet, plane, parent_job_id=parent)
    assert disconnected
    assert result["status"] == "succeeded"
    runs = result["runs"]
    assert runs and all(value["scalar_mirror_paths"] for value in runs)
    assert all(Path(value["scalar_mirror_paths"][0]).read_text() for value in runs)
    assert len(list((plane.jobs.store.root / "fleets").rglob("hpo-control/state.json"))) == 1
    assert not list((tmp_path / "provider-A").rglob("hpo-control/state.json"))
    assert not list((tmp_path / "provider-B").rglob("hpo-control/state.json"))
    assert len(list((plane.jobs.store.root / "fleets").rglob("analysis.json"))) == 1
    assert len({(value["trial"]["index"], value["seed"]) for value in runs}) == len(runs)
    confirmed = [value for value in runs if value["study_phase"] == "confirmation"]
    assert bool(confirmed) is confirmation
    if confirmation:
        assert {value["seed"] for value in confirmed} == {101, 102}
        assert all(not value["pruned"] for value in confirmed)
