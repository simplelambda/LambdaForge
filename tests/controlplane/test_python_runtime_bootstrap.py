"""Behavioral coverage for managed Python discovery, provisioning and reuse."""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lambdaforge.controlplane import (
    ClusterProfile,
    CommandResult,
    EnvironmentIdentity,
    MicromambaArtifactStore,
    NoCompatiblePythonRuntimeError,
    PythonRuntimeResolver,
    Transport,
)


class RuntimeResolutionTransport(Transport):
    """Model a Linux host with bounded Python and Conda-family executable discovery."""

    def __init__(
        self,
        pythons: dict[str, str],
        *,
        manager: str | None = None,
        fail_create: bool = False,
        create_delay: float = 0.0,
    ) -> None:
        self.pythons = dict(pythons)
        self.manager = manager
        self.fail_create = fail_create
        self.create_delay = create_delay
        self.commands: list[tuple[str, ...]] = []
        self.files: dict[str, str] = {}
        self.directories: set[str] = set()
        self.executables: set[str] = set()
        self.binary_files: dict[str, bytes] = {}

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del cwd, timeout
        values = tuple(command)
        self.commands.append(values)
        if values[:2] == ("uname", "-s"):
            return CommandResult(0, "Linux\n")
        if values[:2] == ("uname", "-m"):
            return CommandResult(0, "x86_64\n")
        if values and values[0] == "cat":
            value = self.files.get(values[1])
            return CommandResult(0, value) if value is not None else CommandResult(1)
        if values[:2] == ("test", "-f"):
            return CommandResult(0 if values[2] in self.files else 1)
        if values[:2] in {("test", "-d"), ("test", "-x")}:
            target = values[2]
            exists = (
                target in self.directories
                if values[1] == "-d"
                else target in self.executables or target in self.pythons
            )
            return CommandResult(0 if exists else 1)
        if values and values[0] == "which":
            name = values[-1]
            if name == self.manager:
                return CommandResult(0, f"/opt/{name}\n")
            return CommandResult(1)
        if values and values[0] == "mkdir":
            if "-p" not in values and values[-1] in self.directories:
                return CommandResult(1, stderr="already exists")
            self.directories.add(values[-1])
            return CommandResult(0)
        if values and values[0] == "rmdir":
            self.directories.discard(values[-1])
            return CommandResult(0)
        if values and values[0] == "sha256sum":
            content = self.binary_files.get(values[1])
            if content is None:
                return CommandResult(1)
            return CommandResult(0, f"{hashlib.sha256(content).hexdigest()}  {values[1]}\n")
        if values and values[0] == "chmod":
            self.executables.add(values[-1])
            return CommandResult(0)
        if values and values[0] == "rm":
            self.files.pop(values[-1], None)
            self.directories.discard(values[-1])
            return CommandResult(0)
        if values and values[0] == "mv":
            source, destination = values[-2:]
            if source in self.files:
                self.files[destination] = self.files.pop(source)
            if source in self.binary_files:
                self.binary_files[destination] = self.binary_files.pop(source)
            if source in self.executables:
                self.executables.remove(source)
                self.executables.add(destination)
            for path, version in tuple(self.pythons.items()):
                if path == source or path.startswith(f"{source}/"):
                    self.pythons[f"{destination}{path[len(source) :]}"] = version
                    del self.pythons[path]
            for path, content in tuple(self.files.items()):
                if path.startswith(f"{source}/"):
                    self.files[f"{destination}{path[len(source) :]}"] = content
                    del self.files[path]
            self.directories.add(destination)
            return CommandResult(0)
        uses_manager = any(
            value in {"/opt/conda", "/opt/mamba"} or "micromamba-" in value for value in values
        )
        if "create" in values and uses_manager:
            if self.fail_create:
                return CommandResult(1, stderr="solver failed")
            time.sleep(self.create_delay)
            prefix = values[values.index("--prefix") + 1]
            requested = next(
                value.split("=", 1)[1] for value in values if value.startswith("python=")
            )
            self.pythons[f"{prefix}/bin/python"] = f"{requested}.9"
            self.directories.add(prefix)
            return CommandResult(0)
        if "list" in values and uses_manager:
            return CommandResult(
                0,
                json.dumps(
                    [
                        {"name": "python", "version": "3.13.9", "build_string": "h1"},
                        {"name": "pip", "version": "25.1", "build_string": "pyh"},
                    ]
                ),
            )
        executable = next((value for value in values if value in self.pythons), None)
        if executable is not None and "Path(sys.argv[1]).write_text" in " ".join(values):
            self.files[values[-2]] = values[-1]
            return CommandResult(0)
        if executable is not None and "platform.python_version" in " ".join(values):
            version = self.pythons[executable]
            return CommandResult(
                0,
                json.dumps(
                    {
                        "version": version,
                        "implementation": "CPython",
                        "system": "Linux",
                        "architecture": "x86_64",
                        "executable": executable,
                        "pip": True,
                        "venv": True,
                    }
                )
                + "\n",
            )
        if uses_manager and values[-1] == "--version":
            reported = "2.8.1" if any("micromamba-" in value for value in values) else "25.1"
            return CommandResult(0, f"{reported}\n")
        return CommandResult(1, stderr=f"unsupported test command: {values}")

    def put(self, source: str | Path, destination: str | Path) -> None:
        path = Path(source)
        if path.is_dir():
            self.directories.add(str(destination))
        else:
            self.binary_files[str(destination)] = path.read_bytes()


class RuntimeArtifacts:
    """Provide exact local manager bytes and an offline package cache without networking."""

    VERSION = "2.8.1-0"

    def __init__(self, root: Path) -> None:
        self.binary = root / "micromamba"
        self.binary.write_bytes(b"verified-micromamba")
        self.packages = root / "packages"
        self.packages.mkdir()

    def artifact(self, platform_tag: str) -> tuple[Path, str]:
        assert platform_tag == "linux-64"
        return self.binary, hashlib.sha256(self.binary.read_bytes()).hexdigest()

    def offline_packages(self, platform_tag: str, python_version: str) -> Path:
        assert platform_tag == "linux-64" and python_version == "3.13"
        return self.packages


def managed_profile(
    *, strategy: str = "auto", allow_managed_install: bool = True
) -> ClusterProfile:
    """Create one new-style managed cluster policy through its public YAML parser."""
    return ClusterProfile.from_mapping(
        "gpu",
        {
            "transport": "ssh",
            "host": "gpu.example",
            "workspace": "/work/user",
            "environment": "managed",
            "python": {
                "strategy": strategy,
                "executable": "python3",
                "allow_managed_install": allow_managed_install,
            },
        },
    )


def test_auto_selects_a_compatible_existing_alternative() -> None:
    transport = RuntimeResolutionTransport({"python3": "3.9.21", "python3.12": "3.12.11"})

    runtime = PythonRuntimeResolver().resolve(managed_profile(), transport)

    assert runtime.executable == "python3.12"
    assert runtime.version == "3.12.11"
    assert not runtime.managed


def test_auto_uses_a_compatible_configured_python_without_managed_install() -> None:
    transport = RuntimeResolutionTransport({"python3": "3.13.7"})

    runtime = PythonRuntimeResolver().resolve(managed_profile(), transport)

    assert runtime.executable == "python3"
    assert runtime.version == "3.13.7"
    assert runtime.provider == "existing"
    assert not any("create" in command for command in transport.commands)


def test_consumer_constraint_participates_in_existing_python_selection() -> None:
    transport = RuntimeResolutionTransport(
        {"python3": "3.9.21", "python3.13": "3.13.4", "python3.12": "3.12.11"}
    )

    runtime = PythonRuntimeResolver().resolve(
        managed_profile(), transport, requirements=(">=3.11,<3.13",)
    )

    assert runtime.version == "3.12.11"


def test_incompatible_constraint_intersection_fails_before_installation() -> None:
    transport = RuntimeResolutionTransport({"python3": "3.9.21"}, manager="conda")

    with pytest.raises(NoCompatiblePythonRuntimeError, match="No compatible Python runtime"):
        PythonRuntimeResolver().resolve(
            managed_profile(), transport, requirements=(">=3.15", "<3.15")
        )

    assert not any("create" in command for command in transport.commands)


def test_existing_conda_creates_and_reuses_a_dedicated_runtime_prefix() -> None:
    transport = RuntimeResolutionTransport({"python3": "3.9.21"}, manager="conda")
    resolver = PythonRuntimeResolver()
    profile = managed_profile()

    first = resolver.resolve(profile, transport)
    second = resolver.resolve(profile, transport)

    assert first.managed and first.action == "install"
    assert second.runtime_id == first.runtime_id and second.action == "reuse"
    creates = [command for command in transport.commands if "create" in command]
    assert len(creates) == 1
    assert "--prefix" in creates[0]
    assert not any("activate" in command for command in transport.commands)
    assert first.executable.startswith("/work/user/.lambdaforge/cache/runtimes/")


def test_concurrent_runtime_requests_create_once_then_reuse() -> None:
    transport = RuntimeResolutionTransport(
        {"python3": "3.9.21"}, manager="conda", create_delay=0.3
    )
    resolver = PythonRuntimeResolver(lock_timeout=3.0)
    profile = managed_profile()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(resolver.resolve, profile, transport) for _ in range(2)]
        results = [future.result() for future in futures]

    assert results[0].runtime_id == results[1].runtime_id
    assert {result.action for result in results} == {"install", "reuse"}
    assert sum("create" in command for command in transport.commands) == 1


def test_rejected_runtime_advances_to_the_next_python_minor() -> None:
    """PyTorch wheel rejection must not loop on the same cached managed runtime."""
    transport = RuntimeResolutionTransport({"python3": "3.9.21"}, manager="conda")
    resolver = PythonRuntimeResolver()
    profile = managed_profile(strategy="managed")

    first = resolver.resolve(profile, transport)
    second = resolver.resolve(
        profile, transport, excluded_runtime_ids=(first.runtime_id,)
    )

    assert first.version.startswith("3.13")
    assert second.version.startswith("3.12")
    assert second.runtime_id != first.runtime_id


def test_runtime_identity_changes_the_managed_environment_identity() -> None:
    wheel = ({"name": "lambdaforge.whl", "sha256": "sha256:x", "size_bytes": 1},)
    first = EnvironmentIdentity.create(
        wheel,
        python_requirement="==3.12.*",
        offline=False,
        dependency_policy={"python_runtime": {"runtime_id": "python-runtime-a"}},
    )
    second = EnvironmentIdentity.create(
        wheel,
        python_requirement="==3.13.*",
        offline=False,
        dependency_policy={"python_runtime": {"runtime_id": "python-runtime-b"}},
    )

    assert first.environment_id != second.environment_id


def test_disabled_managed_install_fails_without_mutating_the_host() -> None:
    transport = RuntimeResolutionTransport({"python3": "3.9.21"})

    with pytest.raises(NoCompatiblePythonRuntimeError, match="installation is disabled"):
        PythonRuntimeResolver().resolve(managed_profile(allow_managed_install=False), transport)

    assert not any(re.search(r"\bcreate\b", " ".join(command)) for command in transport.commands)


def test_dry_run_plans_but_does_not_download_or_create_a_runtime() -> None:
    transport = RuntimeResolutionTransport({"python3": "3.9.21"})

    runtime = PythonRuntimeResolver().resolve(managed_profile(), transport, dry_run=True)

    assert not runtime.ready and runtime.action == "install"
    assert runtime.version == "3.13"
    assert not any("create" in command for command in transport.commands)


def test_missing_manager_is_staged_verified_and_kept_inside_cluster_cache(
    tmp_path: Path,
) -> None:
    transport = RuntimeResolutionTransport({"python3": "3.9.21"})
    resolver = PythonRuntimeResolver(RuntimeArtifacts(tmp_path))  # type: ignore[arg-type]

    runtime = resolver.resolve(managed_profile(), transport)

    assert runtime.provider == "micromamba" and runtime.managed
    manager = next(path for path in transport.executables if "runtime-managers" in path)
    assert manager.startswith("/work/user/.lambdaforge/cache/")
    assert any(command[0] == "sha256sum" for command in transport.commands)


def test_offline_runtime_packages_are_prefetched_locally_and_staged(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    descriptor = managed_profile().to_dict(include_defaults=False)
    descriptor.pop("name", None)
    descriptor["wheelhouse"] = str(wheelhouse)
    profile = ClusterProfile.from_mapping("gpu", descriptor)
    transport = RuntimeResolutionTransport({"python3": "3.9.21"}, manager="conda")
    resolver = PythonRuntimeResolver(RuntimeArtifacts(tmp_path))  # type: ignore[arg-type]

    runtime = resolver.resolve(profile, transport)

    assert runtime.managed
    create = next(command for command in transport.commands if "create" in command)
    assert "--offline" in create
    assert any("runtime-packages" in value for value in create)


def test_corrupt_manager_download_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    store = MicromambaArtifactStore(tmp_path / "installers")
    monkeypatch.setitem(
        store.ARTIFACTS,
        "linux-64",
        ("micromamba-linux-64", hashlib.sha256(b"expected").hexdigest()),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: Response(b"corrupt"),
    )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        store.artifact("linux-64")

    assert not tuple((tmp_path / "installers").rglob("micromamba-linux-64"))


def test_failed_runtime_creation_cleans_staging_and_never_activates() -> None:
    transport = RuntimeResolutionTransport({"python3": "3.9.21"}, manager="conda", fail_create=True)

    with pytest.raises(RuntimeError, match="solver failed"):
        PythonRuntimeResolver().resolve(managed_profile(), transport)

    assert "/work/user/.lambdaforge/state/active-python-runtime.json" not in transport.files
    assert any(
        command[:2] == ("rm", "-rf") and ".tmp-" in command[-1] for command in transport.commands
    )
