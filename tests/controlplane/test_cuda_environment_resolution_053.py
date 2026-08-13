"""Focused managed CUDA environment selection for LambdaForge 0.5.3."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.CudaCompatibilityResolver import CudaCompatibilityResolver
from lambdaforge.controlplane.Doctor import Doctor
from lambdaforge.controlplane.EnvironmentIdentity import EnvironmentIdentity
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.TorchInstallationPlan import TorchInstallationPlan
from lambdaforge.controlplane.TorchInstallationPolicy import TorchInstallationPolicy
from lambdaforge.controlplane.Transport import Transport


class ResolutionTransport(Transport):
    """Return controlled remote Python, NVIDIA and official-index facts."""

    def __init__(
        self,
        *,
        driver: str | None,
        capabilities: tuple[str, ...] = (),
        available: dict[str, str] | None = None,
        python: str = "3.13",
    ) -> None:
        self.driver = driver
        self.capabilities = capabilities
        self.available = available or {}
        self.python = python
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: Sequence[str], *, cwd: str | Path | None = None) -> CommandResult:
        del cwd
        values = tuple(command)
        self.commands.append(values)
        if values and values[0] == "nvidia-smi":
            if self.driver is None:
                return CommandResult(1, "", "nvidia-smi unavailable")
            rows = "\n".join(f"{self.driver}, {item}" for item in self.capabilities)
            return CommandResult(0, rows + "\n")
        if "pip" in values and "index" in values:
            index = values[values.index("--index-url") + 1]
            channel = index.rsplit("/", 1)[-1]
            version = self.available.get(channel)
            if version is None:
                return CommandResult(1, "", "No matching distribution")
            return CommandResult(0, f"torch ({version})\nAvailable versions: {version}\n")
        if "platform.machine()" in " ".join(values):
            return CommandResult(0, f"{self.python}\nx86_64\n")
        return CommandResult(0, "", "")

    def put(self, source: str | Path, destination: str | Path) -> None:
        del source, destination


class InstallationTransport(Transport):
    """Capture managed-environment mutation without running pip."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.files: set[str] = set()

    def run(self, command: Sequence[str], *, cwd: str | Path | None = None) -> CommandResult:
        del cwd
        values = tuple(command)
        self.commands.append(values)
        if values[:2] == ("test", "-f"):
            return CommandResult(0 if values[2] in self.files else 1)
        if "Path(sys.argv[1]).write_text" in " ".join(values):
            self.files.add(values[-2])
        return CommandResult(0, "0.6.0 2.12.1+cu126 12.6 True\n", "")

    def put(self, source: str | Path, destination: str | Path) -> None:
        del source, destination


class DoctorTransport(Transport):
    """Expose a GPU whose installed Torch build cannot initialize CUDA."""

    def run(self, command: Sequence[str], *, cwd: str | Path | None = None) -> CommandResult:
        del cwd
        values = tuple(command)
        source = " ".join(values)
        if values and values[0] == "cat":
            return CommandResult(0, "/work/.lambdaforge/environments/current/bin/python\n")
        if values and values[0] == "nvidia-smi":
            return CommandResult(0, "NVIDIA H100, 535.183.01, 9.0\n")
        if "torch.cuda.is_available" in source:
            return CommandResult(
                2,
                "available= False runtime= 13.0 devices= 2 error= RuntimeError: driver too old\n",
            )
        if values and values[-1] == "--version":
            return CommandResult(0, "Python 3.13.13\n")
        if "import torch; print(torch.__version__)" in source:
            return CommandResult(0, "2.13.0+cu130\n")
        if "import lambdaforge" in source:
            return CommandResult(0, "0.6.0\n")
        return CommandResult(0, "")

    def put(self, source: str | Path, destination: str | Path) -> None:
        del source, destination


class DoctorFactory:
    """Inject one diagnostic transport."""

    def __init__(self, transport: DoctorTransport) -> None:
        self.instance = transport

    def transport(self, profile: ClusterProfile) -> DoctorTransport:
        del profile
        return self.instance


def test_h100_with_535_driver_selects_native_cu121_not_newer_runtime() -> None:
    transport = ResolutionTransport(
        driver="535.183.01",
        capabilities=("9.0", "9.0"),
        available={"cu121": "2.5.1+cu121", "cu118": "2.7.1+cu118"},
    )
    profile = ClusterProfile(
        "h100",
        transport="ssh",
        host="h100",
        workspace="/work",
        environment="managed",
    )

    plan = CudaCompatibilityResolver().resolve(profile, transport)

    assert plan.channel == "cu121"
    assert plan.version == "2.5.1+cu121"
    assert plan.require_cuda
    assert all(
        channel not in " ".join(command)
        for channel in ("cu130", "cu128", "cu126", "cu124")
        for command in transport.commands
    )


def test_pascal_gpu_conservatively_selects_cu118() -> None:
    transport = ResolutionTransport(
        driver="470.199.02",
        capabilities=("6.1",),
        available={"cu118": "2.7.1+cu118"},
        python="3.11",
    )
    profile = ClusterProfile(
        "p40",
        transport="ssh",
        host="p40",
        workspace="/work",
        environment="managed",
    )

    plan = CudaCompatibilityResolver().resolve(profile, transport)

    assert plan.channel == "cu118"
    assert plan.compute_capabilities == ("6.1",)


def test_missing_compatible_wheel_fails_instead_of_falling_back_to_wrong_cuda() -> None:
    transport = ResolutionTransport(driver="535.183.01", capabilities=("6.1",), python="3.13")
    profile = ClusterProfile(
        "legacy",
        transport="ssh",
        host="legacy",
        workspace="/work",
        environment="managed",
    )

    with pytest.raises(RuntimeError, match="No compatible official PyTorch wheel"):
        CudaCompatibilityResolver().resolve(profile, transport)


def test_cpu_policy_is_explicit_and_does_not_require_nvidia() -> None:
    transport = ResolutionTransport(driver=None, available={"cpu": "2.12.1+cpu"}, python="3.13")
    profile = ClusterProfile(
        "cpu",
        transport="ssh",
        host="cpu",
        workspace="/work",
        environment="managed",
        pytorch=TorchInstallationPolicy("cpu", False),
    )

    plan = CudaCompatibilityResolver().resolve(profile, transport)

    assert plan.accelerator == "cpu"
    assert not plan.require_cuda


def test_reviewed_channel_can_defer_validation_on_gpu_less_login_host() -> None:
    transport = ResolutionTransport(
        driver=None,
        available={"cu121": "2.5.1+cu121"},
        python="3.12",
    )
    profile = ClusterProfile(
        "slurm-login",
        transport="ssh",
        host="login",
        workspace="/work",
        environment="managed",
        pytorch=TorchInstallationPolicy("cu121", False),
    )

    plan = CudaCompatibilityResolver().resolve(profile, transport)

    assert plan.channel == "cu121"
    assert not plan.require_cuda


def test_environment_identity_changes_with_exact_torch_plan() -> None:
    wheels = ({"name": "lambdaforge.whl", "sha256": "sha256:x", "size_bytes": 1},)
    first = EnvironmentIdentity.create(
        wheels,
        python_requirement=">=3.10",
        offline=False,
        dependency_policy={"pytorch": {"channel": "cu126", "version": "2.12.1+cu126"}},
    )
    second = EnvironmentIdentity.create(
        wheels,
        python_requirement=">=3.10",
        offline=False,
        dependency_policy={"pytorch": {"channel": "cu130", "version": "2.13.0+cu130"}},
    )

    assert first.environment_id != second.environment_id


def test_managed_install_pins_resolved_torch_before_framework(tmp_path: Path) -> None:
    package = tmp_path / "lambdaforge-0.6.0-py3-none-any.whl"
    package.write_bytes(b"wheel")
    plan = TorchInstallationPlan(
        "cu126",
        "2.12.1+cu126",
        "https://download.pytorch.org/whl/cu126",
        "cuda",
        "535.183.01",
        ("9.0",),
        "3.13",
        "x86_64",
        require_cuda=True,
    )
    identity = EnvironmentIdentity.create(
        ({"name": package.name, "sha256": "sha256:x", "size_bytes": 5},),
        python_requirement=">=3.13",
        offline=False,
        dependency_policy={"pytorch": plan.to_dict()},
    )
    bundle = ExecutionBundle(
        "bundle",
        tmp_path,
        tmp_path / "config.yaml",
        tmp_path / "manifest.json",
        5,
        environment_id=identity.environment_id,
        package_names=(package.name,),
        environment_policy=identity.dependency_policy,
    )
    profile = ClusterProfile(
        "h100",
        transport="ssh",
        host="h100",
        workspace="/work",
        environment="managed",
    )
    transport = InstallationTransport()

    ManagedEnvironmentProvider().prepare(
        profile, transport, bundle, remote_bundle_dir="/remote/bundle"
    )

    pip_commands = tuple(command for command in transport.commands if "pip" in command)
    assert pip_commands[0][-1] == "torch==2.12.1+cu126"
    assert "--index-url" in pip_commands[0]
    assert "--constraint" in pip_commands[1]
    assert package.name in pip_commands[1][-1]


def test_doctor_fails_when_visible_gpu_cannot_initialize_cuda() -> None:
    profile = ClusterProfile(
        "h100",
        transport="ssh",
        host="h100",
        workspace="/work",
        environment="managed",
    )
    transport = DoctorTransport()

    report = Doctor(
        ClusterCatalog({"h100": profile}),
        factory=DoctorFactory(transport),  # type: ignore[arg-type]
    ).check("h100")

    cuda = next(check for check in report.checks if check.name == "cuda")
    assert not report.ok
    assert not cuda.ok
    assert "driver too old" in cuda.message
