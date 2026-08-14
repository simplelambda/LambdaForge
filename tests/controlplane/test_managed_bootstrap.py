"""Managed bootstrap wheel resolution independent of the active virtual environment layout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lambdaforge.controlplane import (
    ClusterCatalog,
    ClusterProfile,
    ClusterService,
    CommandResult,
    PreparedEnvironment,
    ProjectWheelBuilder,
    PythonRuntime,
    TorchInstallationPlan,
    Transport,
)


class FakeInstalledDistribution:
    """Expose only the metadata required to repack an installed wheel."""

    version = "0.7.2"
    metadata = {"Name": "lambdaforge"}
    files: tuple[object, ...] = ()

    def __init__(self, direct_url: str | None = None) -> None:
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        values = {
            "direct_url.json": self.direct_url,
            "METADATA": (
                "Metadata-Version: 2.4\nName: lambdaforge\nVersion: 0.7.2\n"
                "Requires-Python: >=3.10\n"
            ),
            "WHEEL": (
                "Wheel-Version: 1.0\nGenerator: LambdaForge test\n"
                "Root-Is-Purelib: true\nTag: py3-none-any\n"
            ),
            "entry_points.txt": (
                "[console_scripts]\nlf = lambdaforge:version\nlambdaforge = lambdaforge:version\n"
            ),
            "top_level.txt": "lambdaforge\n",
        }
        return values.get(filename)

    def locate_file(self, path: object) -> Path:
        raise AssertionError(f"No installed data files were declared: {path}")


class BootstrapTransport(Transport):
    """Accept workspace operations while reporting an empty remote bootstrap cache."""

    def run(self, command: Sequence[str], *, cwd: str | Path | None = None) -> CommandResult:
        del cwd
        return CommandResult(1) if command[:2] == ("test", "-f") else CommandResult(0)

    def put(self, source: str | Path, destination: str | Path) -> None:
        assert Path(source).is_dir()
        assert destination

    def get(self, source: str | Path, destination: str | Path) -> None:
        raise AssertionError(f"Bootstrap should not download {source} to {destination}.")


class BootstrapEnvironmentProvider:
    """Acknowledge the staged environment without running remote pip in this unit test."""

    def prepare(
        self,
        profile: ClusterProfile,
        transport: Transport,
        bundle: Any,
        *,
        remote_bundle_dir: str | Path,
    ) -> PreparedEnvironment:
        del profile, transport, remote_bundle_dir
        return PreparedEnvironment(bundle.environment_id, "/remote/env/bin/python", False)


class BootstrapFactory:
    """Return deterministic transport and environment test doubles."""

    def __init__(self) -> None:
        self.remote = BootstrapTransport()
        self.environment = BootstrapEnvironmentProvider()

    def transport(self, profile: ClusterProfile) -> BootstrapTransport:
        del profile
        return self.remote

    def environment_provider(self, profile: ClusterProfile) -> BootstrapEnvironmentProvider:
        del profile
        return self.environment


class BootstrapCudaResolver:
    """Avoid hardware probing while preserving an exact managed dependency plan."""

    def resolve(
        self,
        profile: ClusterProfile,
        transport: Transport,
        *,
        python_executable: str | None = None,
    ) -> TorchInstallationPlan:
        del profile, transport, python_executable
        return TorchInstallationPlan(
            channel="cpu",
            version="2.1.0",
            index_url="https://download.pytorch.org/whl/cpu",
            accelerator="cpu",
            python_version="3.10",
        )


class RecordingWheelBuilder:
    """Record that bootstrap resolves an installed distribution instead of a guessed root."""

    def __init__(self, wheel: Path) -> None:
        self.wheel = wheel
        self.calls: list[tuple[str, Path | None]] = []

    def build_installed(
        self, distribution_name: str, *, source_hint: str | Path | None = None
    ) -> Path:
        self.calls.append(
            (distribution_name, Path(source_hint).resolve() if source_hint is not None else None)
        )
        return self.wheel


class BootstrapRuntimeResolver:
    """Return a verified runtime while this test isolates framework-wheel resolution."""

    def resolve(
        self, profile: ClusterProfile, transport: Transport, **kwargs: Any
    ) -> PythonRuntime:
        del profile, transport, kwargs
        return PythonRuntime(
            "python-runtime-test",
            "/usr/bin/python3.10",
            "3.10.14",
            "CPython",
            "Linux",
            "x86_64",
            "existing",
            None,
            False,
            True,
            "reuse",
        )

    def activate(
        self, profile: ClusterProfile, transport: Transport, runtime: PythonRuntime
    ) -> None:
        del profile, transport, runtime


def test_installed_distribution_repacking_is_deterministic_and_installable(tmp_path: Path) -> None:
    """A normal wheel install can bootstrap again without its original checkout or index."""
    package = tmp_path / "installed" / "lambdaforge"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'version = "0.7.2"\n__version__ = version\n', encoding="utf-8"
    )
    schemas = package / "schemas"
    schemas.mkdir()
    (schemas / "task.schema.json").write_text("{}\n", encoding="utf-8")
    builder = ProjectWheelBuilder(tmp_path / "cache")
    installed = FakeInstalledDistribution()

    first = builder._pack_installed(installed, package)  # type: ignore[arg-type]
    second = builder._pack_installed(installed, package)  # type: ignore[arg-type]

    assert first == second
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert "lambdaforge/__init__.py" in names
        assert "lambdaforge/schemas/task.schema.json" in names
        assert "lambdaforge-0.7.2.dist-info/RECORD" in names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    target = tmp_path / "target"
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(first),
        ),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr
    environment = dict(os.environ, PYTHONPATH=str(target))
    imported = subprocess.run(
        (sys.executable, "-c", "import lambdaforge; print(lambdaforge.__version__)"),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        cwd=tmp_path,
        env=environment,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "0.7.2"


def test_editable_distribution_resolves_direct_url_not_virtualenv_parent(tmp_path: Path) -> None:
    """The PEP 610 source URL wins when ``__file__`` lives under an unrelated environment."""
    project = tmp_path / "LambdaForge"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='lambdaforge'\n", encoding="utf-8")
    installed = FakeInstalledDistribution(
        json.dumps({"url": project.as_uri(), "dir_info": {"editable": True}})
    )

    resolved = ProjectWheelBuilder.installed_project_root(
        installed,  # type: ignore[arg-type]
        source_hint=tmp_path / "consumer/.venv/lib/python3.14",
    )

    assert resolved == project.resolve()


def test_cluster_bootstrap_requests_the_installed_framework_distribution(tmp_path: Path) -> None:
    """Regress the former ``.venv/lib/pythonX/pyproject.toml`` source-root assumption."""
    wheel = tmp_path / "lambdaforge-0.7.2-py3-none-any.whl"
    wheel.write_bytes(b"framework-wheel")
    builder = RecordingWheelBuilder(wheel)
    profile = ClusterProfile(
        "gpu",
        transport="ssh",
        host="gpu.example",
        workspace="/work/user",
        environment="managed",
    )
    service = ClusterService(
        ClusterCatalog({"gpu": profile}),
        BootstrapFactory(),  # type: ignore[arg-type]
        tmp_path / "control",
        BootstrapCudaResolver(),  # type: ignore[arg-type]
        builder,  # type: ignore[arg-type]
        BootstrapRuntimeResolver(),  # type: ignore[arg-type]
    )

    result = service.bootstrap("gpu")

    assert result.python == "/remote/env/bin/python"
    assert len(builder.calls) == 1
    assert builder.calls[0][0] == "lambdaforge"


def test_cluster_bootstrap_dry_run_does_not_build_or_stage_wheels(tmp_path: Path) -> None:
    wheel = tmp_path / "lambdaforge-0.7.2-py3-none-any.whl"
    wheel.write_bytes(b"unused")
    builder = RecordingWheelBuilder(wheel)
    profile = ClusterProfile(
        "gpu",
        transport="ssh",
        host="gpu.example",
        workspace="/work/user",
        environment="managed",
    )
    factory = BootstrapFactory()
    service = ClusterService(
        ClusterCatalog({"gpu": profile}),
        factory,  # type: ignore[arg-type]
        tmp_path / "control",
        BootstrapCudaResolver(),  # type: ignore[arg-type]
        builder,  # type: ignore[arg-type]
        BootstrapRuntimeResolver(),  # type: ignore[arg-type]
    )

    result = service.bootstrap("gpu", dry_run=True)

    assert result.planned
    assert result.runtime is not None
    assert builder.calls == []
