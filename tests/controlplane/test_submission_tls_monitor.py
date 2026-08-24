"""Local regressions for 0.9 asynchronous submission, TLS trust and monitoring."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.cli.LiveJobMonitor import MonitorRenderer
from lambdaforge.controlplane import (
    ClusterCatalog,
    ClusterProfile,
    CommandResult,
    ControlPlane,
    Doctor,
    ExecutionBundle,
    JobService,
    JobState,
    JobStore,
    Scheduler,
    SchedulerSubmission,
    SubmissionService,
    TlsTrust,
    TlsTrustResolver,
    Transport,
)
from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
from lambdaforge.controlplane.PreparedEnvironment import PreparedEnvironment
from lambdaforge.controlplane.python_runtime import PythonRuntime
from lambdaforge.controlplane.TorchInstallationPlan import TorchInstallationPlan
from lambdaforge.execution import ResourceRequest
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.reproducibility import CodeIdentity


class TrustTransport(Transport):
    """Return one configurable system-CA discovery result without network access."""

    def __init__(self, ca_file: str | None) -> None:
        self.ca_file = ca_file
        self.commands: list[tuple[str, ...]] = []

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del cwd, timeout
        self.commands.append(tuple(command))
        return CommandResult(
            0,
            json.dumps(
                {
                    "ca_file": self.ca_file,
                    "candidates": [self.ca_file] if self.ca_file else [],
                }
            )
            + "\n",
        )

    def put(self, source: str | Path, destination: str | Path) -> None:
        raise AssertionError(f"Trust discovery must not transfer {source} to {destination}.")


def remote_profile() -> ClusterProfile:
    return ClusterProfile.from_mapping(
        "remote",
        {
            "transport": "ssh",
            "host": "gpu.invalid",
            "workspace": "/work/user",
            "environment": "managed",
            "python": {"strategy": "auto", "executable": "python3"},
        },
    )


def task_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "queued-work",
                "run": "tests.work_cases.Producer",
                "resources": {"cpu": 2, "memory": "1GiB"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_host_ca_bundle_is_validated_and_propagated_as_argv_environment() -> None:
    transport = TrustTransport("/etc/ssl/certs/ca-certificates.crt")

    trust = TlsTrustResolver().resolve(remote_profile(), transport)

    assert trust.ca_file == "/etc/ssl/certs/ca-certificates.crt"
    assert trust.prefix()[0] == "env"
    assert "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in trust.prefix()
    assert "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" in trust.prefix()
    assert not any("unverified" in " ".join(command).lower() for command in transport.commands)


def test_missing_host_ca_bundle_fails_closed_without_disabling_verification() -> None:
    with pytest.raises(RuntimeError, match="no readable PEM CA bundle"):
        TlsTrustResolver().resolve(remote_profile(), TrustTransport(None))


def test_existing_runtime_has_no_injected_trust_policy() -> None:
    from lambdaforge.controlplane.python_runtime import PythonRuntime

    runtime = PythonRuntime(
        "existing",
        "/usr/bin/python3",
        "3.12.1",
        "CPython",
        "Linux",
        "x86_64",
        "existing",
        None,
        False,
        True,
        "reuse",
    )

    assert runtime.tls_trust is None
    assert runtime.to_dict()["tls_trust"] is None


def test_doctor_reports_legacy_managed_runtime_without_tls_trust() -> None:
    profile = remote_profile()
    assert profile.storage is not None
    runtime = PythonRuntime(
        "legacy-managed",
        "/managed/runtime/bin/python",
        "3.12.1",
        "CPython",
        "Linux",
        "x86_64",
        "micromamba",
        "2.0",
        True,
        True,
        "reuse",
    )

    class DoctorTransport(Transport):
        def run(
            self,
            command: Sequence[str],
            *,
            cwd: str | Path | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            del cwd, timeout
            source = " ".join(command)
            if command and command[0] == "cat":
                if command[1].endswith("active-environment"):
                    return CommandResult(0, "/managed/environment/bin/python\n")
                if command[1].endswith("active-python-runtime.json"):
                    return CommandResult(0, json.dumps(runtime.to_dict()))
                return CommandResult(1)
            if command and command[0] == "nvidia-smi":
                return CommandResult(1, stderr="no GPU")
            if command and command[-1] == "--version":
                return CommandResult(0, "Python 3.12.1\n")
            if "get_ca_certs" in source:
                return CommandResult(0, '{"ca_count": 100}\n')
            if "import lambdaforge" in source:
                return CommandResult(0, f"{LambdaForgeVersion.CURRENT}\n")
            if "import torch" in source or "torch.cuda" in source:
                return CommandResult(0, "2.7.0\n")
            return CommandResult(0)

        def put(self, source: str | Path, destination: str | Path) -> None:
            del source, destination

    class Factory:
        def __init__(self) -> None:
            self.transport_instance = DoctorTransport()

        def transport(self, selected: ClusterProfile) -> DoctorTransport:
            del selected
            return self.transport_instance

    report = Doctor(
        ClusterCatalog({"remote": profile}),
        Factory(),  # type: ignore[arg-type]
    ).check("remote")

    managed_tls = next(check for check in report.checks if check.name == "managed-python-tls")
    assert not managed_tls.ok
    assert "no recorded host CA trust policy" in managed_tls.message
    assert managed_tls.command == "lf clusters bootstrap remote"


def test_remote_enqueue_returns_preparing_before_any_remote_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = task_config(tmp_path / "task.yaml")
    profile = remote_profile()
    catalog = ClusterCatalog({"remote": profile})
    jobs = JobService(catalog, JobStore(tmp_path / "jobs"))
    launches: list[tuple[str, ...]] = []

    class Process:
        pid = 12345

    def launch(command: Sequence[str], **kwargs: Any) -> Process:
        assert kwargs["start_new_session"] is True
        launches.append(tuple(command))
        return Process()

    monkeypatch.setattr("subprocess.Popen", launch)
    monkeypatch.setattr(
        CodeIdentity,
        "capture",
        classmethod(lambda cls, source_dir: cls("test", "submission-fixture")),
    )

    handle = SubmissionService(catalog, jobs).enqueue(config, cluster="remote")

    assert handle.state is JobState.PREPARING
    assert jobs.get(handle.job_id, refresh=False).metadata["submission_phase"] == "queued-locally"
    assert launches and "lambdaforge.controlplane.SubmissionWorker" in launches[0]
    request = jobs.store.root / "submissions" / handle.job_id / "request.json"
    assert request.is_file()
    assert os.stat(request).st_mode & 0o777 == 0o600
    cancelled = jobs.cancel(handle.job_id)
    assert cancelled.state is JobState.CANCELLED


def test_local_enqueue_uses_the_same_non_blocking_durable_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = task_config(tmp_path / "task.yaml")
    profile = ClusterProfile(
        "local",
        python=sys.executable,
        storage={
            "state_root": str(tmp_path / "state"),
            "cache_root": str(tmp_path / "cache"),
            "run_root": str(tmp_path / "runtime-jobs"),
        },
    )
    catalog = ClusterCatalog({"local": profile})
    jobs = JobService(catalog, JobStore(tmp_path / "jobs"))
    launches: list[tuple[str, ...]] = []

    class Process:
        pid = 12345

    def launch(command: Sequence[str], **kwargs: Any) -> Process:
        assert kwargs["start_new_session"] is True
        launches.append(tuple(command))
        return Process()

    monkeypatch.setattr("subprocess.Popen", launch)
    monkeypatch.setattr(
        CodeIdentity,
        "capture",
        classmethod(lambda cls, source_dir: cls("test", "local-submission-fixture")),
    )

    handle = SubmissionService(catalog, jobs).enqueue(config, cluster="local")

    assert handle.state is JobState.PREPARING
    assert launches and "lambdaforge.controlplane.SubmissionWorker" in launches[0]


def test_top_level_local_run_enqueues_instead_of_running_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = task_config(tmp_path / "task.yaml")
    calls: list[str] = []

    class EnqueueOnly:
        def __init__(self, catalog: ClusterCatalog) -> None:
            del catalog

        def enqueue(self, source: Path, *, cluster: str, **kwargs: Any) -> Any:
            del source, kwargs
            calls.append(cluster)
            return SimpleNamespace(to_dict=lambda: {"job_id": "job-local-background"})

    monkeypatch.delenv("LAMBDAFORGE_JOB_ID", raising=False)
    monkeypatch.delenv("LAMBDAFORGE_EXECUTION_MODE", raising=False)
    monkeypatch.setitem(CommandLineInterface._run.__globals__, "SubmissionService", EnqueueOnly)
    monkeypatch.setattr(
        CommandLineInterface._run.__globals__["WorkRunner"],
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run inline")),
    )
    arguments = SimpleNamespace(
        config=config,
        on="local",
        clusters=None,
        dry_run=False,
        wait_for_submit=False,
        rerun=False,
        restart=False,
        allow_duplicate=False,
        json=False,
    )

    assert CommandLineInterface._run(arguments) == 0
    assert calls == ["local"]
    assert "Submitted job-local-background" in capsys.readouterr().out


def test_supervised_local_child_runs_inline_without_recursive_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = task_config(tmp_path / "task.yaml")
    calls: list[str] = []

    class NeverEnqueue:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise AssertionError("a supervised child must not enqueue another Job")

    class Outcome:
        @staticmethod
        def to_dict() -> dict[str, str]:
            return {"name": "queued-work", "status": "succeeded", "execution_id": "execution-1"}

    def run(*args: Any, **kwargs: Any) -> Outcome:
        del args, kwargs
        calls.append("inline")
        return Outcome()

    monkeypatch.setenv("LAMBDAFORGE_EXECUTION_MODE", "worker")
    monkeypatch.setitem(CommandLineInterface._run.__globals__, "SubmissionService", NeverEnqueue)
    monkeypatch.setattr(CommandLineInterface._run.__globals__["WorkRunner"], "run", run)
    arguments = SimpleNamespace(
        config=config,
        on="local",
        clusters=None,
        dry_run=False,
        wait_for_submit=False,
        rerun=False,
        restart=False,
        allow_duplicate=False,
        json=False,
    )

    assert CommandLineInterface._run(arguments) == 0
    assert calls == ["inline"]


def test_submission_worker_persists_pre_scheduler_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = task_config(tmp_path / "task.yaml")
    catalog = ClusterCatalog({"remote": remote_profile()})
    jobs = JobService(catalog, JobStore(tmp_path / "jobs"))

    class Process:
        pid = 12345

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(
        CodeIdentity,
        "capture",
        classmethod(lambda cls, source_dir: cls("test", "submission-fixture")),
    )
    handle = SubmissionService(catalog, jobs).enqueue(config, cluster="remote")
    request = jobs.store.root / "submissions" / handle.job_id / "request.json"

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("synthetic staging failure")

    monkeypatch.setattr(
        "lambdaforge.controlplane.SubmissionWorker.ControlPlane.submit",
        fail,
    )
    from lambdaforge.controlplane.SubmissionWorker import serve

    assert serve(request) == 1
    failed = jobs.get(handle.job_id, refresh=False)
    assert failed.state is JobState.FAILED
    assert failed.metadata["failure_phase"] == "preparation"
    assert "synthetic staging failure" in jobs.logs(handle.job_id)


def test_tls_environment_reaches_the_scientific_scheduler_command(tmp_path: Path) -> None:
    """TLS trust must extend beyond bootstrap into Work processes."""
    config = task_config(tmp_path / "task.yaml")
    profile = remote_profile()
    catalog = ClusterCatalog({"remote": profile})
    store = JobStore(tmp_path / "jobs")

    class RecordingTransport(Transport):
        def __init__(self) -> None:
            self.puts: list[tuple[str, str]] = []

        def run(
            self,
            command: Sequence[str],
            *,
            cwd: str | Path | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            del cwd, timeout
            return CommandResult(1 if command[:2] == ("test", "-f") else 0)

        def put(self, source: str | Path, destination: str | Path) -> None:
            self.puts.append((str(source), str(destination)))

    class RecordingScheduler(Scheduler):
        command: tuple[str, ...] = ()

        def submit(
            self,
            command: Sequence[str],
            resources: ResourceRequest,
            *,
            work_dir: str | Path,
            dry_run: bool = False,
        ) -> SchedulerSubmission:
            del resources, work_dir, dry_run
            self.command = tuple(command)
            return SchedulerSubmission("provider-1", JobState.QUEUED)

        def state(self, scheduler_id: str) -> JobState:
            del scheduler_id
            return JobState.QUEUED

        def logs(self, scheduler_id: str, *, tail: int | None = None) -> str:
            del scheduler_id, tail
            return ""

        def cancel(self, scheduler_id: str) -> None:
            del scheduler_id

    class Environment:
        def prepare(self, *args: Any, **kwargs: Any) -> PreparedEnvironment:
            return PreparedEnvironment("env-test", "/managed/env/bin/python", False)

    class Factory:
        def __init__(self) -> None:
            self.remote = RecordingTransport()
            self.provider = RecordingScheduler()

        def transport(self, selected: ClusterProfile) -> RecordingTransport:
            del selected
            return self.remote

        def scheduler(self, selected: ClusterProfile, transport: Transport) -> RecordingScheduler:
            del selected, transport
            return self.provider

        def environment_provider(self, selected: ClusterProfile) -> Environment:
            del selected
            return Environment()

    trust = TlsTrust("/etc/ssl/certs/ca-certificates.crt")

    class RuntimeResolver:
        def resolve(self, *args: Any, **kwargs: Any) -> PythonRuntime:
            return PythonRuntime(
                "managed-runtime",
                "/managed/runtime/bin/python",
                "3.12.1",
                "CPython",
                "Linux",
                "x86_64",
                "micromamba",
                "2.0",
                True,
                True,
                "reuse",
                "packages",
                trust,
            )

        def activate(self, *args: Any, **kwargs: Any) -> None:
            return None

    class CudaResolver:
        def resolve(self, *args: Any, **kwargs: Any) -> TorchInstallationPlan:
            return TorchInstallationPlan(
                "cpu", "2.7.0", "https://download.pytorch.org/whl/cpu", "cpu"
            )

    class Bundles:
        def build(self, *args: Any, **kwargs: Any) -> ExecutionBundle:
            directory = tmp_path / "bundle"
            directory.mkdir(exist_ok=True)
            materialized = directory / "config.yaml"
            materialized.write_bytes(config.read_bytes())
            manifest = directory / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            return ExecutionBundle(
                "bundle-test",
                directory,
                materialized,
                manifest,
                2,
                "env-test",
                ("lambdaforge.whl",),
            )

    factory = Factory()
    jobs = JobService(catalog, store, factory)  # type: ignore[arg-type]
    ControlPlane(
        catalog,
        jobs,
        Bundles(),  # type: ignore[arg-type]
        factory,  # type: ignore[arg-type]
        CudaResolver(),  # type: ignore[arg-type]
        RuntimeResolver(),  # type: ignore[arg-type]
    ).submit(config, cluster="remote")

    command = factory.provider.command
    assert command[0] == "env"
    assert "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in command
    assert "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" in command
    assert "LAMBDAFORGE_CACHE_ROOT=/work/user/.lambdaforge/cache" in command
    python_index = command.index("/managed/env/bin/python")
    assert command[python_index : python_index + 4] == (
        "/managed/env/bin/python",
        "-m",
        "lambdaforge",
        "run",
    )


def test_monitor_renderer_uses_the_same_machine_readable_job_items() -> None:
    payload = {
        "generated_at_utc": "2026-08-20T12:00:00+00:00",
        "clusters": [
            {
                "cluster": "gpu",
                "online": True,
                "observed": {
                    "cpu_load": 25,
                    "ram_total_bytes": 100,
                    "ram_available_bytes": 40,
                    "gpus": [{"utilization_percent": 80}],
                },
            }
        ],
        "jobs": {
            "total": 1,
            "by_state": {"preparing": 1},
            "items": [
                {
                    "job_id": "job-20260820120000-12345678",
                    "scheduler_id": None,
                    "state": "preparing",
                    "cluster": "gpu",
                    "job_type": "dataset-build",
                    "created_at_utc": "2026-08-20T11:59:00+00:00",
                    "resources": {},
                    "metadata": {"name": "dna", "submission_phase": "bundle"},
                }
            ],
        },
        "work": {
            "items": [
                {
                    "work_id": "work-dna",
                    "name": "dna",
                    "kind": "work",
                    "state": "preparing",
                    "cluster": "gpu",
                    "scientific_revision": "abc123",
                    "progress": {"completed": None, "total": None, "unit": "runs"},
                    "attempts": 1,
                    "primary_job_id": "job-20260820120000-12345678",
                    "job_ids": ["job-20260820120000-12345678"],
                    "created_at_utc": "2026-08-20T11:59:00+00:00",
                }
            ]
        },
    }

    rendered = MonitorRenderer.render(payload, width=160)

    assert "dna" in rendered
    assert "job-20260820120000-12345678" not in rendered
    assert "attempts=1" in rendered
    assert "80%" in rendered


def test_bundle_staging_rejects_nested_input_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    (source / "escape").symlink_to(outside)

    with pytest.raises(ValueError, match="contains a symbolic link"):
        ExecutionBundleBuilder._size(source)
