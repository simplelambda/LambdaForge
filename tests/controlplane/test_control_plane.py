"""Provider-neutral control-plane and job persistence tests."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from lambdaforge.controlplane import (
    ClusterCatalog,
    ClusterProfile,
    CommandResult,
    ControlPlane,
    ExecutionBundleBuilder,
    JobService,
    JobState,
    JobStore,
    Scheduler,
    SchedulerSubmission,
    Transport,
)
from lambdaforge.execution import ResourceRequest


class FakeTransport(Transport):
    """Deterministic command transport with no external side effects."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.puts: list[tuple[str, str]] = []
        self.remote_manifests: set[str] = set()

    def run(self, command: Sequence[str], *, cwd: str | Path | None = None) -> CommandResult:
        del cwd
        self.commands.append(tuple(command))
        if command[:2] == ("test", "-f"):
            return CommandResult(0 if command[2] in self.remote_manifests else 1)
        return CommandResult(0, "ok\n")

    def put(self, source: str | Path, destination: str | Path) -> None:
        destination_value = str(destination)
        self.puts.append((str(source), destination_value))
        self.remote_manifests.add(f"{destination_value}/manifest.json")


class FakeScheduler(Scheduler):
    """Persist state outside the service to model scheduler reconnection."""

    def __init__(self) -> None:
        self.states: dict[str, JobState] = {}

    def submit(
        self,
        command: Sequence[str],
        resources: ResourceRequest,
        *,
        work_dir: str | Path,
        dry_run: bool = False,
    ) -> SchedulerSubmission:
        del command, resources, work_dir
        if dry_run:
            return SchedulerSubmission(None, JobState.CREATED)
        scheduler_id = str(len(self.states) + 1)
        self.states[scheduler_id] = JobState.QUEUED
        return SchedulerSubmission(scheduler_id, JobState.QUEUED)

    def state(self, scheduler_id: str) -> JobState:
        return self.states[scheduler_id]

    def logs(self, scheduler_id: str, *, tail: int | None = None) -> str:
        del tail
        return f"log {scheduler_id}"

    def cancel(self, scheduler_id: str) -> None:
        self.states[scheduler_id] = JobState.CANCELLED


class FakeFactory:
    """Inject shared fake providers into every service operation."""

    def __init__(self) -> None:
        self.transport_instance = FakeTransport()
        self.scheduler_instance = FakeScheduler()

    def transport(self, profile: ClusterProfile) -> Transport:
        del profile
        return self.transport_instance

    def scheduler(self, profile: ClusterProfile, transport: Transport) -> Scheduler:
        del profile, transport
        return self.scheduler_instance


class TestControlPlane:
    """Exercise persistence, refresh, cancellation and retry through fake providers."""

    def test_job_reconnect_cancel_and_retry(self, tmp_path: Path) -> None:
        catalog = ClusterCatalog({"fake": ClusterProfile("fake", scheduler="slurm")})
        store = JobStore(tmp_path / "jobs")
        factory = FakeFactory()
        service = JobService(catalog, store, factory)  # type: ignore[arg-type]
        handle = service.submit(
            ("python", "train.py"),
            cluster="fake",
            resources=ResourceRequest(cpu_cores=2),
            work_dir=tmp_path,
        )
        assert handle.state is JobState.QUEUED
        factory.scheduler_instance.states[handle.scheduler_id or ""] = JobState.RUNNING
        reconnected = JobService(catalog, store, factory)  # type: ignore[arg-type]
        assert reconnected.get(handle.job_id).state is JobState.RUNNING
        cancelled = reconnected.cancel(handle.job_id)
        assert cancelled.state is JobState.CANCELLED
        retried = reconnected.retry(handle.job_id)
        assert retried.job_id != handle.job_id
        assert store.get(retried.job_id).retry_of == handle.job_id

    def test_resource_request_parses_portable_units(self) -> None:
        request = ResourceRequest.from_mapping(
            {"cpus": 8, "memory": "32GiB", "gpus": 2, "time": "4h", "processes": 2}
        )
        assert request.cpu_cores == 8
        assert request.ram_bytes == 32 * 1024**3
        assert request.gpu_count == 2
        assert request.runtime_seconds == 4 * 3600
        assert request.processes == 2

    def test_remote_bundle_stages_small_inputs_once(self, tmp_path: Path) -> None:
        """Content-addressed bundles must be cached instead of nesting repeated scp copies."""
        input_path = tmp_path / "records.jsonl"
        input_path.write_text('{"id": "a"}\n', encoding="utf-8")
        config_path = tmp_path / "preprocessing.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "name": "remote-preprocessing",
                    "inputs": {"raw": "records.jsonl"},
                    "outputs": {"processed": "processed"},
                    "preprocess": {"function": "builtins.dict"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        profile = ClusterProfile(
            "fake",
            transport="ssh",
            scheduler="slurm",
            host="fake-host",
            workspace="/remote/lambdaforge",
        )
        catalog = ClusterCatalog({"fake": profile})
        factory = FakeFactory()
        store = JobStore(tmp_path / "jobs")
        jobs = JobService(catalog, store, factory)  # type: ignore[arg-type]
        control = ControlPlane(
            catalog,
            jobs,
            ExecutionBundleBuilder(tmp_path / "bundles"),
            factory,  # type: ignore[arg-type]
        )

        first, first_bundle = control.submit(
            config_path, cluster="fake", run_arguments=("--force",)
        )
        second, second_bundle = control.submit(config_path, cluster="fake")

        assert first.job_id != second.job_id
        assert first_bundle.bundle_id == second_bundle.bundle_id
        assert len(factory.transport_instance.puts) == 1
        assert store.get(first.job_id).command[-1] == "--force"
        strict = yaml.safe_load(first_bundle.config_path.read_text(encoding="utf-8"))
        assert strict["inputs"][0]["path"].startswith("inputs/")
        assert (first_bundle.directory / strict["inputs"][0]["path"]).is_file()

    def test_data_environment_alias_resolves_one_cluster(self) -> None:
        """A physical data environment need not duplicate the cluster profile name."""
        profile = ClusterProfile("atlas", data_environment="project-storage")
        catalog = ClusterCatalog({"atlas": profile})
        assert catalog.for_data_environment("project-storage") is profile

    def test_default_local_profile_uses_active_python(self) -> None:
        """Doctor and local submission must inspect the environment invoking LambdaForge."""
        assert ClusterCatalog.load().get("local").python == sys.executable
