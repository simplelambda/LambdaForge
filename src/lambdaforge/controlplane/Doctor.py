"""Local and remote preflight diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.python_runtime import PythonRuntimeRequirements
from lambdaforge.controlplane.PythonRuntimeResolver import PythonRuntimeResolver
from lambdaforge.controlplane.SlurmProfile import SlurmProfile
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """Represent one actionable and portable environment diagnostic."""

    name: str
    ok: bool
    message: str
    fix: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        """Return a machine-readable check."""
        return {"name": self.name, "ok": self.ok, "message": self.message, "fix": self.fix}


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Collect diagnostic checks for one local or remote profile."""

    cluster: str
    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        """Return whether all required checks passed."""
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        """Return the stable report envelope."""
        return {
            "cluster": self.cluster,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
        }

    def summary(self) -> str:
        """Render actionable terminal output."""
        lines = [f"LambdaForge doctor ({self.cluster}): {'OK' if self.ok else 'ISSUES'}"]
        for check in self.checks:
            lines.append(f"[{'ok' if check.ok else '!!'}] {check.name}: {check.message}")
            if not check.ok and check.fix:
                lines.append(f"     fix: {check.fix}")
        return "\n".join(lines)


class Doctor:
    """Check access, Python, framework, scheduler and optional CUDA visibility."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()

    def check(
        self, cluster: str = "local", *, config_path: str | Path | None = None
    ) -> DoctorReport:
        """Run read-only diagnostics through the selected transport."""
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        transport = self.factory.transport(profile)
        checks: list[DoctorCheck] = []
        try:
            connection = transport.run(("true",))
        except Exception as error:
            return DoctorReport(
                cluster,
                (
                    DoctorCheck(
                        "connection",
                        False,
                        str(error),
                        (
                            "Verify the credential reference, SSH host key, network and timeout; "
                            "OpenSSH profiles should verify ssh config/agent/ProxyJump."
                        ),
                    ),
                ),
            )
        checks.append(
            DoctorCheck(
                "connection",
                connection.returncode == 0,
                connection.stderr.strip() or "Transport connection is available.",
                "Verify SSH config, agent, host key and network connectivity.",
            )
        )
        checks.append(
            DoctorCheck(
                "authentication",
                True,
                (
                    "Authentication is delegated to OpenSSH configuration/agent."
                    if profile.auth.mode == "openssh"
                    else (
                        f"Password authentication succeeded using {profile.auth.credential!r}."
                        if profile.auth.credential is not None
                        else "Interactive password authentication succeeded."
                    )
                ),
                None,
            )
        )
        selected_python = profile.python
        active_environment = False
        if profile.environment == "managed":
            pointer = PurePosixPath(profile.storage.state_root) / "active-environment"
            active = transport.run(("cat", str(pointer)))
            if active.returncode or not active.stdout.strip():
                legacy = PurePosixPath(profile.workspace) / ".lambdaforge" / "active-environment"
                active = transport.run(("cat", str(legacy)))
            if active.returncode == 0 and active.stdout.strip():
                selected_python = active.stdout.strip()
                active_environment = True
            checks.append(
                DoctorCheck(
                    "managed-environment",
                    active.returncode == 0 and bool(active.stdout.strip()),
                    active.stdout.strip() or "No active managed environment is recorded.",
                    f"Run 'lambdaforge clusters bootstrap {cluster}'.",
                )
            )
        workspace = transport.run(("test", "-d", profile.workspace))
        checks.append(
            DoctorCheck(
                "workspace",
                workspace.returncode == 0,
                profile.workspace if workspace.returncode == 0 else "Workspace is missing.",
                f"Run 'lambdaforge clusters bootstrap {cluster}'.",
            )
        )
        requirement = PythonRuntimeRequirements.framework()
        system_python = transport.run((*profile.command_prefix, profile.python, "--version"))
        system_message = (
            system_python.stdout or system_python.stderr
        ).strip() or "Python was not found."
        system_match = re.search(r"Python\s+(\d+(?:\.\d+){1,2})", system_message)
        system_compatible = bool(
            system_match
            and PythonRuntimeRequirements.compatible(system_match.group(1), (requirement,))
        )
        checks.append(
            DoctorCheck(
                "system-python" if profile.environment == "managed" else "python",
                system_python.returncode == 0 and system_match is not None,
                f"{system_message} ({'compatible' if system_compatible else 'incompatible'} "
                f"with {requirement})",
                (
                    f"Set clusters.{cluster}.python.executable to a working Python."
                    if system_python.returncode
                    else None
                ),
            )
        )
        if profile.environment == "managed":
            runtime = PythonRuntimeResolver().active(profile, transport)
            if runtime is None:
                checks.append(
                    DoctorCheck(
                        "python-runtime",
                        True,
                        (
                            "No managed runtime is active yet; "
                            f"python.strategy={profile.runtime_policy.strategy} can resolve one "
                            "during bootstrap."
                        ),
                    )
                )
            else:
                runtime_probe = transport.run(
                    (*profile.command_prefix, runtime.executable, "--version")
                )
                runtime_message = (
                    runtime_probe.stdout or runtime_probe.stderr
                ).strip() or runtime.executable
                runtime_match = re.search(r"Python\s+(\d+(?:\.\d+){1,2})", runtime_message)
                runtime_ok = bool(
                    runtime_probe.returncode == 0
                    and runtime_match
                    and PythonRuntimeRequirements.compatible(runtime_match.group(1), (requirement,))
                )
                checks.append(
                    DoctorCheck(
                        "python-runtime",
                        runtime_ok,
                        f"{runtime_message}; {runtime.executable} ({runtime.provider})",
                        f"Rerun 'lf clusters bootstrap {cluster}' to repair the managed runtime.",
                    )
                )
            if active_environment:
                managed = transport.run((*profile.command_prefix, selected_python, "--version"))
                managed_message = (
                    managed.stdout or managed.stderr
                ).strip() or "Python was not found."
                managed_match = re.search(r"Python\s+(\d+(?:\.\d+){1,2})", managed_message)
                checks.append(
                    DoctorCheck(
                        "managed-python",
                        bool(
                            managed.returncode == 0
                            and managed_match
                            and PythonRuntimeRequirements.compatible(
                                managed_match.group(1), (requirement,)
                            )
                        ),
                        f"{managed_message}; {selected_python}",
                        f"Rerun 'lf clusters bootstrap {cluster}'.",
                    )
                )
        if profile.project_module:
            project = transport.run(
                (
                    *profile.command_prefix,
                    selected_python,
                    "-c",
                    f"import {profile.project_module}",
                )
            )
            checks.append(
                DoctorCheck(
                    "consumer-project",
                    project.returncode == 0,
                    project.stdout.strip() or project.stderr.strip() or profile.project_module,
                    "Bootstrap the managed environment or install the exact consumer project.",
                )
            )
        framework = transport.run(
            (
                *profile.command_prefix,
                selected_python,
                "-c",
                (
                    "import lambdaforge; "
                    f"assert lambdaforge.__version__ == {LambdaForgeVersion.CURRENT!r}; "
                    "print(lambdaforge.__version__)"
                ),
            )
        )
        checks.append(
            DoctorCheck(
                "lambdaforge",
                framework.returncode == 0,
                framework.stdout.strip() or framework.stderr.strip(),
                "Install the same LambdaForge release in the selected environment.",
            )
        )
        if profile.scheduler == "slurm":
            slurm_profile = cast(SlurmProfile, profile.slurm_profile)
            for executable in slurm_profile.executables():
                scheduler = transport.run(("which", executable))
                checks.append(
                    DoctorCheck(
                        f"scheduler-command:{executable}",
                        scheduler.returncode == 0,
                        scheduler.stdout.strip() or scheduler.stderr.strip() or "Not found.",
                        "Load the cluster's scheduler client environment or correct "
                        "scheduler_commands in the profile.",
                    )
                )
            try:
                _, warnings = slurm_profile.resource_mapping.render(
                    ResourceRequest(
                        cpu_cores=2,
                        ram_bytes=1024**3,
                        gpu_count=1,
                        runtime_seconds=60,
                    )
                )
                checks.append(
                    DoctorCheck(
                        "scheduler-resource-mapping",
                        True,
                        (
                            "Resource templates are valid. WARNING: " + " ".join(warnings)
                            if warnings
                            else "Resource templates are valid and emit CPU/memory/GPU/time."
                        ),
                        None,
                    )
                )
            except (TypeError, ValueError) as error:
                checks.append(
                    DoctorCheck(
                        "scheduler-resource-mapping",
                        False,
                        str(error),
                        "Correct resource_mapping placeholders and options in the cluster profile.",
                    )
                )
            partitions = slurm_profile.directives.get("partition", ())
            for partition in partitions:
                partition_command = slurm_profile.info.render(
                    {"partition": str(partition)}, allowed={"partition"}
                )
                partition_check = transport.run(partition_command)
                checks.append(
                    DoctorCheck(
                        f"scheduler-partition:{partition}",
                        partition_check.returncode == 0 and bool(partition_check.stdout.strip()),
                        partition_check.stdout.strip()
                        or partition_check.stderr.strip()
                        or "Partition not visible.",
                        "Correct scheduler_directives.partition or request cluster access.",
                    )
                )
        pytorch = transport.run(
            (
                *profile.command_prefix,
                selected_python,
                "-c",
                "import torch; print(torch.__version__)",
            )
        )
        checks.append(
            DoctorCheck(
                "pytorch",
                pytorch.returncode == 0,
                pytorch.stdout.strip() or pytorch.stderr.strip(),
                "Install a PyTorch build compatible with the cluster's Python and driver.",
            )
        )
        nvidia = transport.run(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            )
        )
        gpu_visible = nvidia.returncode == 0 and bool(nvidia.stdout.strip())
        cuda_required = self._requires_cuda(Path(config_path)) if config_path is not None else False
        cuda_expected = cuda_required or (
            gpu_visible
            and profile.pytorch.channel != "cpu"
            and profile.pytorch.require_cuda is not False
        )
        checks.append(
            DoctorCheck(
                "nvidia-driver",
                nvidia.returncode == 0 or not cuda_expected,
                nvidia.stdout.strip()
                or nvidia.stderr.strip()
                or "No NVIDIA GPU/driver was detected; CPU execution remains available.",
                "Expose a supported NVIDIA driver/GPU or use an explicit CPU PyTorch policy.",
            )
        )
        cuda = transport.run(
            (
                *profile.command_prefix,
                selected_python,
                "-c",
                (
                    "import sys, torch; available=torch.cuda.is_available(); error=''; "
                    "\nif not available:\n"
                    " try: torch.cuda.init()\n"
                    " except Exception as exc: error=f'{type(exc).__name__}: {exc}'\n"
                    "print('available=', available, 'runtime=', torch.version.cuda, "
                    "'devices=', torch.cuda.device_count(), 'error=', error); "
                    f"sys.exit(0 if available or {not cuda_expected!r} else 2)"
                ),
            )
        )
        checks.append(
            DoctorCheck(
                "cuda",
                cuda.returncode == 0,
                cuda.stdout.strip() or cuda.stderr.strip(),
                "Select a cluster-provided PyTorch/CUDA environment; LambdaForge never installs "
                "drivers or the system CUDA toolkit. For managed environments rerun bootstrap "
                "with pytorch.channel=auto or an explicit compatible channel.",
            )
        )
        bundle_cache = transport.run(("test", "-d", profile.storage.bundle_root))
        checks.append(
            DoctorCheck(
                "bundle-cache",
                bundle_cache.returncode == 0,
                "Bundle cache is ready." if bundle_cache.returncode == 0 else "Cache is missing.",
                f"Run 'lambdaforge clusters bootstrap {cluster}'.",
            )
        )
        if config_path is not None:
            checks.extend(self._data_checks(cluster, Path(config_path)))
        return DoctorReport(cluster, tuple(checks))

    def _data_checks(self, cluster: str, config_path: Path) -> tuple[DoctorCheck, ...]:
        """Resolve logical data through the same Registry-first policy used at runtime."""
        try:
            from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
            from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
            from lambdaforge.data.DataCatalog import DataCatalog
            from lambdaforge.data.DatasetRegistry import DatasetRegistry
            from lambdaforge.data.DatasetResolver import DatasetResolver

            materialized = AuthoringConfig.from_yaml(config_path).materialize().to_dict()
            references = ExecutionBundleBuilder._experiment_dataset_references(materialized)
            if not references:
                return (
                    DoctorCheck(
                        "datasets",
                        True,
                        "The configuration declares no logical dataset references.",
                    ),
                )
            extensions = materialized.get("extensions", {})
            authoring = extensions.get("authoring", {}) if isinstance(extensions, dict) else {}
            catalog_value = authoring.get("data_catalog") if isinstance(authoring, dict) else None
            catalog = None
            if catalog_value is not None:
                source = Path(str(catalog_value))
                source = source if source.is_absolute() else (config_path.resolve().parent / source)
                catalog = DataCatalog.from_yaml(source)
            profile = self.catalog.get(cluster)
            resolver = DatasetResolver(
                DatasetRegistry(DatasetRegistry.project_path(config_path.resolve().parent)),
                catalog,
                environment=profile.data_environment or cluster,
                managed_environment=cluster,
                source_dir=config_path.resolve().parent,
            )
            resolved = tuple(resolver.resolve(reference) for reference in references)
            return (
                DoctorCheck(
                    "datasets",
                    True,
                    "Logical datasets resolve exactly: "
                    + ", ".join(value.exact_reference for value in resolved),
                ),
            )
        except Exception as error:
            return (
                DoctorCheck(
                    "datasets",
                    False,
                    f"{error.__class__.__name__}: {error}",
                    "Register/materialize managed data or fix the explicit external catalog.",
                ),
            )

    @staticmethod
    def _requires_cuda(config_path: Path) -> bool:
        """Return whether an authored config explicitly requests one or more GPUs."""
        try:
            from lambdaforge.configuration.AuthoringConfig import AuthoringConfig

            values = AuthoringConfig.from_yaml(config_path).materialize().to_dict()
            extensions = values.get("extensions", {})
            authoring = extensions.get("authoring", {}) if isinstance(extensions, dict) else {}
            resources = authoring.get("resources", {}) if isinstance(authoring, dict) else {}
            if not isinstance(resources, dict):
                return False
            gpus = resources.get("gpus", 0)
            return gpus == "auto" or int(gpus) > 0
        except (OSError, TypeError, ValueError):
            return False
