"""Focused coverage for secure credentials, catalog scopes and SLURM dialects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lambdaforge.controlplane.ClusterAuthentication import ClusterAuthentication
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.CredentialService import CredentialService
from lambdaforge.controlplane.EnvironmentCredentialProvider import EnvironmentCredentialProvider
from lambdaforge.controlplane.InteractiveCredentialProvider import InteractiveCredentialProvider
from lambdaforge.controlplane.PasswordSshTransport import PasswordSshTransport
from lambdaforge.controlplane.SlurmProfile import SlurmProfile
from lambdaforge.controlplane.SlurmScheduler import SlurmScheduler
from lambdaforge.controlplane.SystemKeyringCredentialProvider import (
    SystemKeyringCredentialProvider,
)
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.execution.ResourceRequest import ResourceRequest


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, key: str) -> str | None:
        return self.values.get((service, key))

    def set_password(self, service: str, key: str, value: str) -> None:
        self.values[(service, key)] = value

    def delete_password(self, service: str, key: str) -> None:
        del self.values[(service, key)]


class FakeTransport(Transport):
    def __init__(self, responses: list[CommandResult] | None = None) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.responses = list(responses or [])
        self.uploads: list[tuple[Path, str]] = []

    def run(self, command: tuple[str, ...], *, cwd: str | Path | None = None) -> CommandResult:
        del cwd
        self.commands.append(tuple(command))
        return self.responses.pop(0) if self.responses else CommandResult(0, "", "")

    def put(self, source: str | Path, destination: str | Path) -> None:
        self.uploads.append((Path(source), str(destination)))


class FakeChannel:
    def recv_exit_status(self) -> int:
        return 0


class FakeStream:
    def __init__(self, value: bytes) -> None:
        self.value = value
        self.channel = FakeChannel()

    def read(self) -> bytes:
        return self.value


class FakeSftp:
    def __init__(self) -> None:
        self.directories = {"/", "/remote"}
        self.puts: list[tuple[str, str]] = []
        self.gets: list[tuple[str, str]] = []

    def stat(self, value: str) -> SimpleNamespace:
        if value not in self.directories and value != "/remote/value.txt":
            raise OSError(value)
        return SimpleNamespace(st_mode=0o040755 if value in self.directories else 0o100644)

    def mkdir(self, value: str) -> None:
        self.directories.add(value)

    def put(self, source: str, destination: str) -> None:
        self.puts.append((source, destination))

    def get(self, source: str, destination: str) -> None:
        self.gets.append((source, destination))
        Path(destination).write_text("downloaded", encoding="utf-8")

    def close(self) -> None:
        pass


class FakeSshClient:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.policy: object | None = None
        self.connected: dict[str, object] = {}
        self.sftp = FakeSftp()

    def load_system_host_keys(self) -> None:
        pass

    def load_host_keys(self, path: str) -> None:
        assert Path(path).is_file()

    def set_missing_host_key_policy(self, policy: object) -> None:
        self.policy = policy

    def connect(self, **values: object) -> None:
        self.connected = values
        if self.failure is not None:
            raise self.failure

    def exec_command(self, command: str, *, timeout: float) -> tuple[None, FakeStream, FakeStream]:
        assert command == "cd /work && exec python train.py"
        assert timeout == 3
        return None, FakeStream(b"ok\n"), FakeStream(b"")

    def open_sftp(self) -> FakeSftp:
        return self.sftp

    def close(self) -> None:
        pass


class FakeParamiko:
    class RejectPolicy:
        pass

    def __init__(self, client: FakeSshClient) -> None:
        self.client = client

    def SSHClient(self) -> FakeSshClient:
        return self.client


def test_credentials_resolve_without_serializing_values() -> None:
    backend = FakeKeyring()
    keyring = SystemKeyringCredentialProvider(backend)
    keyring.set("keyring:cluster/atlas/me@login", "keyring-secret")
    profile = ClusterProfile(
        "atlas",
        transport="ssh",
        host="login",
        workspace="/work",
        auth=ClusterAuthentication("password", "keyring:cluster/atlas/me@login"),
    )
    service = CredentialService(
        interactive=InteractiveCredentialProvider(lambda _: "interactive-secret"),
        environment=EnvironmentCredentialProvider({"SSH_PASSWORD": "environment-secret"}),
        keyring=keyring,
    )
    assert service.resolve(profile) == "keyring-secret"
    environment_profile = ClusterProfile(
        "env",
        transport="ssh",
        host="login",
        workspace="/work",
        auth=ClusterAuthentication("password", "env:SSH_PASSWORD"),
    )
    assert service.resolve(environment_profile) == "environment-secret"
    interactive_profile = ClusterProfile(
        "interactive",
        transport="ssh",
        host="login",
        workspace="/work",
        auth=ClusterAuthentication("password"),
    )
    assert service.resolve(interactive_profile) == "interactive-secret"
    serialized = yaml.safe_dump(profile.to_dict())
    assert "keyring-secret" not in serialized
    assert "environment-secret" not in serialized
    assert "interactive-secret" not in serialized


def test_missing_keyring_entry_is_actionable() -> None:
    provider = SystemKeyringCredentialProvider(FakeKeyring())
    with pytest.raises(RuntimeError, match="credentials set"):
        provider.get("keyring:cluster/missing", prompt="ignored")


def test_password_ssh_rejects_unknown_keys_redacts_and_transfers(tmp_path: Path) -> None:
    client = FakeSshClient()
    module = FakeParamiko(client)
    transport = PasswordSshTransport(
        "login",
        user="me",
        password_provider=lambda: "top-secret",
        timeout=3,
        paramiko_module=module,
    )
    result = transport.run(("python", "train.py"), cwd="/work")
    assert result.stdout == "ok\n"
    assert isinstance(client.policy, FakeParamiko.RejectPolicy)
    assert client.connected["look_for_keys"] is False
    assert client.connected["allow_agent"] is False
    assert client.connected["timeout"] == 3
    source = tmp_path / "value.txt"
    source.write_text("value", encoding="utf-8")
    transport.put(source, "/remote/value.txt")
    destination = tmp_path / "downloaded.txt"
    transport.get("/remote/value.txt", destination)
    assert client.sftp.puts[-1][1] == "/remote/value.txt"
    assert destination.read_text(encoding="utf-8") == "downloaded"
    assert "top-secret" not in repr(transport)

    failing = PasswordSshTransport(
        "login",
        password_provider=lambda: "top-secret",
        paramiko_module=FakeParamiko(FakeSshClient(failure=RuntimeError("bad top-secret"))),
    )
    with pytest.raises(RuntimeError) as captured:
        failing.run(("true",))
    assert "top-secret" not in str(captured.value)
    assert "***" in str(captured.value)


def test_catalog_merges_user_project_and_explicit_with_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.chdir(project)
    user = ClusterCatalog.user_path()
    user.parent.mkdir(parents=True)
    user.write_text("clusters:\n  atlas:\n    workspace: user\n", encoding="utf-8")
    ClusterCatalog.project_path().write_text(
        "clusters:\n  atlas:\n    workspace: project\n  project-only: {}\n", encoding="utf-8"
    )
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "clusters:\n  atlas:\n    workspace: explicit\n  explicit-only: {}\n", encoding="utf-8"
    )
    catalog = ClusterCatalog.load(explicit)
    assert catalog.get("atlas").workspace == "explicit"
    assert catalog.get("project-only").workspace == ".lambdaforge/remote"
    assert catalog.get("explicit-only").workspace == ".lambdaforge/remote"
    inspected = catalog.inspect("atlas")
    assert inspected["source"] == str(explicit.resolve())
    assert inspected["shadowed_sources"] == [str(user), str(ClusterCatalog.project_path())]


def test_slurm_resource_dialects_commands_flags_and_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    profile = SlurmProfile.from_mapping(
        {
            "resource_mapping": {
                "gpu": {"option": "gres", "value": "gpu:a100:{gpus}"},
                "memory": {"option": "mem", "value": "{memory_gib}G"},
            },
            "scheduler_directives": {
                "partition": "accelerated",
                "exclusive": True,
                "constraint": ["nvlink", "ssd"],
            },
            "scheduler_commands": {
                "submit": {
                    "command": "site-submit",
                    "args": ["--machine", "{work_dir}", "{script}"],
                    "job_id_pattern": "^JOB-(\\d+)$",
                },
                "queue": ["site-queue", "{job_id}"],
                "cancel": ["site-cancel", "{job_id}"],
            },
            "job_script": {
                "shell": "/bin/zsh",
                "prologue": ["module load cuda"],
                "epilogue": ["echo complete"],
            },
        }
    )
    scheduler = SlurmScheduler(FakeTransport(), profile=profile)
    submission = scheduler.submit(
        ("python", "train.py"),
        ResourceRequest.from_mapping(
            {"cpus": 8, "memory": "10GiB", "gpus": 2, "time": "2h", "processes": 2}
        ),
        work_dir="/scratch/job",
        dry_run=True,
    )
    assert "--gres=gpu:a100:2" in submission.directives
    assert "--mem=10G" in submission.directives
    assert "--exclusive" in submission.directives
    assert submission.directives.count("--constraint=nvlink") == 1
    assert submission.command == (
        "site-submit",
        "--machine",
        "/scratch/job",
        "/scratch/job/submit.sbatch",
    )
    content = submission.script.read_text(encoding="utf-8")
    assert content.startswith("#!/bin/zsh\n")
    assert "module load cuda" in content
    assert "echo complete" in content


def test_slurm_submission_parses_custom_job_id_and_omit_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    profile = SlurmProfile.from_mapping(
        {
            "resource_mapping": {"gpu": "omit"},
            "scheduler_commands": {
                "submit": {
                    "command": "submit-wrapper",
                    "args": ["{script}"],
                    "job_id_pattern": "^job=(\\d+)$",
                }
            },
        }
    )
    transport = FakeTransport([CommandResult(0, "", ""), CommandResult(0, "job=42\n", "")])
    scheduler = SlurmScheduler(transport, profile=profile)
    submission = scheduler.submit(
        ("true",), ResourceRequest(gpu_count=1), work_dir="/work", dry_run=False
    )
    assert submission.scheduler_id == "42"
    assert "not emitted" in submission.warnings[0]
    assert transport.commands[-1] == ("submit-wrapper", "/work/submit.sbatch")


def test_legacy_scheduler_options_remain_static_directives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    scheduler = SlurmScheduler(FakeTransport(), options={"partition": "gpu"})
    submission = scheduler.submit(("true",), ResourceRequest(), work_dir="/work", dry_run=True)
    assert "--partition=gpu" in submission.directives
