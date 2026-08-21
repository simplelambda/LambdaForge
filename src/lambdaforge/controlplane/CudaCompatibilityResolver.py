"""Remote NVIDIA driver to official PyTorch wheel resolver."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import PurePosixPath

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.python_runtime import PythonRuntimeRequirements
from lambdaforge.controlplane.TorchInstallationPlan import TorchInstallationPlan
from lambdaforge.controlplane.Transport import Transport


class NoCompatibleTorchWheelError(RuntimeError):
    """Signal that another otherwise valid Python candidate may solve wheel availability."""


class CudaCompatibilityResolver:
    """Select a conservative available PyTorch channel from remote facts."""

    INDEX_ROOT = "https://download.pytorch.org/whl"
    MINOR_COMPATIBILITY_MINIMUM_DRIVER = {
        "cu118": (450, 80),
        "cu121": (525, 0),
        "cu124": (525, 0),
        "cu126": (525, 0),
        "cu128": (525, 0),
        "cu130": (580, 0),
    }
    NATIVE_MINIMUM_DRIVER = {
        "cu118": (520, 61, 5),
        "cu121": (530, 30, 2),
        "cu124": (550, 54, 14),
        "cu126": (560, 28, 3),
        "cu128": (570, 26),
        "cu130": (580, 65, 6),
    }
    MINIMUM_COMPUTE_CAPABILITY = {"cu128": (7, 5), "cu130": (7, 5)}

    def resolve(
        self,
        profile: ClusterProfile,
        transport: Transport,
        *,
        python_executable: str | None = None,
    ) -> TorchInstallationPlan:
        """Probe remote Python/GPU facts and resolve one installable exact wheel version."""
        executable = python_executable or profile.python
        python = self._python(profile, transport, executable)
        requirement = PythonRuntimeRequirements.framework()
        if not PythonRuntimeRequirements.compatible(python[0], (requirement,)):
            displayed_requirement = requirement.replace(">=", ">= ")
            raise RuntimeError(
                f"LambdaForge requires Python {displayed_requirement}, but "
                f"clusters.{profile.name}.python "
                f"resolves to Python {python[0]}. Configure a "
                "supported cluster Python executable or load the site's Python module before "
                "bootstrap; python.strategy=existing never installs or changes system Python. "
                "Use python.strategy=auto or managed to permit a user-space runtime."
            )
        driver, capabilities = self._nvidia(transport)
        has_cuda = driver is not None and bool(capabilities)
        required = profile.pytorch.require_cuda
        require_cuda = has_cuda if required is None else required
        candidates: tuple[str, ...]
        if profile.wheelhouse is not None:
            if require_cuda and not has_cuda:
                raise RuntimeError(
                    "CUDA is required but nvidia-smi did not expose a driver and "
                    "compute capability."
                )
            accelerator = (
                "cpu"
                if profile.pytorch.channel == "cpu"
                else "cuda"
                if profile.pytorch.channel != "auto" or has_cuda
                else "cpu"
            )
            return TorchInstallationPlan(
                "wheelhouse",
                None,
                None,
                accelerator,
                driver,
                capabilities,
                python[0],
                python[1],
                "Offline wheelhouse; installed Torch bytes are part of environment identity.",
                require_cuda and accelerator == "cuda",
            )
        requested = profile.pytorch.channel
        if requested == "cpu":
            if require_cuda:
                raise RuntimeError("pytorch.channel=cpu conflicts with require_cuda=true.")
            candidates = ("cpu",)
        elif requested != "auto":
            if driver is None:
                if required is not False:
                    raise RuntimeError(
                        f"pytorch.channel={requested} cannot be verified without a visible NVIDIA "
                        "driver. On a GPU-less scheduler login host, use require_cuda=false only "
                        "for an administrator-reviewed channel and validate inside an allocation."
                    )
            else:
                self._validate_channel(requested, driver, capabilities)
            candidates = (requested,)
            require_cuda = driver is not None if required is None else required
        elif not has_cuda:
            if require_cuda:
                raise RuntimeError(
                    "CUDA is required but nvidia-smi did not expose a driver and "
                    "compute capability."
                )
            candidates = ("cpu",)
        else:
            assert driver is not None
            candidates = self._automatic_channels(driver, capabilities)
        active = self._active_verified_plan(
            profile,
            transport,
            python=python,
            driver=driver,
            capabilities=capabilities,
            candidates=candidates,
            require_cuda=require_cuda,
        )
        if active is not None:
            return active
        failures: list[str] = []
        for channel in candidates:
            index = f"{self.INDEX_ROOT}/{channel}"
            version = self._available_version(profile, transport, index, executable)
            if version is not None:
                accelerator = "cpu" if channel == "cpu" else "cuda"
                return TorchInstallationPlan(
                    channel,
                    version,
                    index,
                    accelerator,
                    driver,
                    capabilities,
                    python[0],
                    python[1],
                    self._reason(channel, driver, capabilities),
                    require_cuda and accelerator == "cuda",
                )
            failures.append(channel)
        raise NoCompatibleTorchWheelError(
            "No compatible official PyTorch wheel was found for remote Python "
            f"{python[0]} ({python[1]}) in channels {tuple(failures)}. Choose another configured "
            "Python (older clusters commonly need Python 3.10-3.12), set an explicit supported "
            "pytorch.channel, or provide a reviewed target-compatible wheelhouse."
        )

    @staticmethod
    def _active_verified_plan(
        profile: ClusterProfile,
        transport: Transport,
        *,
        python: tuple[str, str],
        driver: str | None,
        capabilities: tuple[str, ...],
        candidates: tuple[str, ...],
        require_cuda: bool,
    ) -> TorchInstallationPlan | None:
        """Reuse a verified immutable environment receipt when all host facts still match."""
        if profile.environment != "managed" or profile.storage is None:
            return None
        pointer = PurePosixPath(profile.storage.state_root) / "active-environment"
        active = transport.run(("cat", str(pointer)))
        if active.returncode or not active.stdout.strip():
            return None
        executable = PurePosixPath(active.stdout.strip())
        marker = executable.parent.parent / ".lambdaforge-environment.json"
        receipt = transport.run(("cat", str(marker)))
        if receipt.returncode:
            return None
        try:
            value = json.loads(receipt.stdout)
            policy = value.get("environment_policy", {})
            raw_plan = policy.get("pytorch", {}) if isinstance(policy, dict) else {}
            plan = TorchInstallationPlan.from_mapping(raw_plan)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        if (
            plan.channel not in candidates
            or plan.version is None
            or plan.index_url is None
            or plan.python_version != python[0]
            or plan.architecture != python[1]
            or plan.driver_version != driver
            or tuple(plan.compute_capabilities) != capabilities
            or plan.require_cuda != require_cuda
        ):
            return None
        return plan

    @staticmethod
    def _python(profile: ClusterProfile, transport: Transport, executable: str) -> tuple[str, str]:
        result = transport.run(
            (
                *profile.command_prefix,
                executable,
                "-c",
                "import platform,sys; "
                "print(f'{sys.version_info.major}.{sys.version_info.minor}'); "
                "print(platform.machine())",
            )
        )
        lines = result.stdout.strip().splitlines()
        if result.returncode or len(lines) < 2:
            raise RuntimeError(
                "Could not inspect remote Python version/architecture before dependency selection: "
                f"{result.stderr.strip()}"
            )
        return lines[0], lines[1]

    @classmethod
    def _nvidia(cls, transport: Transport) -> tuple[str | None, tuple[str, ...]]:
        result = transport.run(
            (
                "nvidia-smi",
                "--query-gpu=driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            )
        )
        if result.returncode:
            visible = transport.run(("nvidia-smi", "-L"))
            if visible.returncode == 0 and visible.stdout.strip():
                raise RuntimeError(
                    "NVIDIA GPUs are visible but driver/compute capability could not be queried "
                    "safely. Set an explicit reviewed pytorch.channel or provide a wheelhouse."
                )
            return None, ()
        drivers: set[str] = set()
        capabilities: set[str] = set()
        for line in result.stdout.splitlines():
            values = tuple(item.strip() for item in line.split(","))
            if len(values) != 2 or re.fullmatch(r"\d+(?:\.\d+)+", values[0]) is None:
                continue
            if re.fullmatch(r"\d+\.\d+", values[1]) is None:
                continue
            drivers.add(values[0])
            capabilities.add(values[1])
        if not drivers or not capabilities:
            raise RuntimeError(
                "nvidia-smi returned GPU rows but no parseable driver/compute capability. Set an "
                "explicit reviewed pytorch.channel or provide a wheelhouse."
            )
        driver = min(drivers, key=cls._version)
        return driver, tuple(sorted(capabilities, key=cls._version))

    def _available_version(
        self,
        profile: ClusterProfile,
        transport: Transport,
        index_url: str,
        executable: str,
    ) -> str | None:
        result = transport.run(
            (
                *profile.command_prefix,
                executable,
                "-m",
                "pip",
                "index",
                "versions",
                "torch",
                "--index-url",
                index_url,
                "--disable-pip-version-check",
            )
        )
        if result.returncode:
            return None
        match = re.search(r"Available versions:\s*([^\r\n]+)", result.stdout + result.stderr)
        if match is None:
            return None
        versions = tuple(item.strip() for item in match.group(1).split(",") if item.strip())
        return versions[0] if versions else None

    def _automatic_channels(self, driver: str, capabilities: Sequence[str]) -> tuple[str, ...]:
        version = self._version(driver)
        minimum_capability = min((self._version(item) for item in capabilities), default=(0,))
        if minimum_capability < (7, 0):
            if version < self.MINOR_COMPATIBILITY_MINIMUM_DRIVER["cu118"]:
                raise RuntimeError(
                    f"NVIDIA driver {driver} is below the documented cu118 minor-compatibility "
                    "floor. Use a reviewed existing environment or upgrade the driver."
                )
            return ("cu118",)
        compatible = tuple(
            channel
            for channel in ("cu130", "cu128", "cu126", "cu124", "cu121", "cu118")
            if version >= self.NATIVE_MINIMUM_DRIVER[channel]
            and minimum_capability >= self.MINIMUM_COMPUTE_CAPABILITY.get(channel, (0,))
        )
        if compatible:
            return compatible
        if version >= self.MINOR_COMPATIBILITY_MINIMUM_DRIVER["cu118"]:
            return ("cu118",)
        raise RuntimeError(
            f"NVIDIA driver {driver} is below the documented cu118 minor-compatibility floor. "
            "Use a reviewed existing environment or upgrade the driver."
        )

    def _validate_channel(self, channel: str, driver: str, capabilities: Sequence[str]) -> None:
        minimum = self.MINOR_COMPATIBILITY_MINIMUM_DRIVER[channel]
        if self._version(driver) < minimum:
            raise RuntimeError(
                f"pytorch.channel={channel} requires NVIDIA driver >= {minimum[0]}.{minimum[1]}, "
                f"but the cluster reports {driver}."
            )
        compute_minimum = self.MINIMUM_COMPUTE_CAPABILITY.get(channel, (0,))
        if min(self._version(item) for item in capabilities) < compute_minimum:
            raise RuntimeError(
                f"pytorch.channel={channel} is not selected safely for compute capability "
                f"{tuple(capabilities)}; use cu118 or a reviewed wheelhouse."
            )

    @staticmethod
    def _reason(channel: str, driver: str | None, capabilities: Sequence[str]) -> str:
        if channel == "cpu":
            return "No compatible NVIDIA devices were detected or CPU was explicitly selected."
        return (
            f"Selected {channel} for NVIDIA driver {driver} and compute capabilities "
            f"{tuple(capabilities)}; the exact wheel exists for the configured remote Python."
        )

    @staticmethod
    def _version(value: str) -> tuple[int, ...]:
        """Parse a dotted numeric NVIDIA version for deterministic comparisons."""
        return tuple(int(item) for item in value.split("."))
