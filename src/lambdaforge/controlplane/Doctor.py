"""Local and remote preflight diagnostics."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.DoctorCheck import DoctorCheck
from lambdaforge.controlplane.DoctorReport import DoctorReport
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


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
        transport = self.factory.transport(profile)
        checks: list[DoctorCheck] = []
        connection = transport.run(("true",))
        checks.append(
            DoctorCheck(
                "connection",
                connection.returncode == 0,
                connection.stderr.strip() or "Transport connection is available.",
                "Verify SSH config, agent, host key and network connectivity.",
            )
        )
        selected_python = profile.python
        if profile.environment == "managed":
            pointer = PurePosixPath(profile.workspace) / ".lambdaforge" / "active-environment"
            active = transport.run(("cat", str(pointer)))
            if active.returncode == 0 and active.stdout.strip():
                selected_python = active.stdout.strip()
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
        python = transport.run((*profile.command_prefix, selected_python, "--version"))
        checks.append(
            DoctorCheck(
                "python",
                python.returncode == 0,
                (python.stdout or python.stderr).strip() or "Python was not found.",
                f"Set clusters.{cluster}.python to a working Python executable.",
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
            scheduler = transport.run(("sinfo", "--version"))
            checks.append(
                DoctorCheck(
                    "slurm",
                    scheduler.returncode == 0,
                    scheduler.stdout.strip() or scheduler.stderr.strip(),
                    "Load the cluster's SLURM client environment or correct the profile.",
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
        cuda_required = self._requires_cuda(Path(config_path)) if config_path is not None else False
        cuda = transport.run(
            (
                *profile.command_prefix,
                selected_python,
                "-c",
                (
                    "import sys, torch; available=torch.cuda.is_available(); "
                    "print('available=', available, 'runtime=', torch.version.cuda, "
                    "'devices=', torch.cuda.device_count()); "
                    f"sys.exit(0 if available or {not cuda_required!r} else 2)"
                ),
            )
        )
        checks.append(
            DoctorCheck(
                "cuda",
                cuda.returncode == 0,
                cuda.stdout.strip() or cuda.stderr.strip(),
                "Select a cluster-provided PyTorch/CUDA environment; LambdaForge never installs "
                "drivers or the system CUDA toolkit.",
            )
        )
        bundle_cache = transport.run(
            ("test", "-d", f"{profile.workspace.rstrip('/')}/.lambdaforge/bundles")
        )
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
        """Validate catalog declarations and target locations for one requested config."""
        try:
            from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
            from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
            from lambdaforge.data.DataCatalog import DataCatalog

            materialized = AuthoringConfig.from_yaml(config_path).materialize().to_dict()
            extensions = materialized.get("extensions", {})
            authoring = extensions.get("authoring", {}) if isinstance(extensions, dict) else {}
            catalog_value = authoring.get("data_catalog") if isinstance(authoring, dict) else None
            if catalog_value is None:
                return (
                    DoctorCheck(
                        "data-catalog",
                        True,
                        "The configuration declares no data catalog.",
                        "Add data_catalog only when logical datasets are used.",
                    ),
                )
            source = Path(str(catalog_value))
            source = source if source.is_absolute() else (config_path.resolve().parent / source)
            catalog = DataCatalog.from_yaml(source)
            names = ExecutionBundleBuilder._experiment_dataset_names(materialized)
            environment = self.catalog.get(cluster).data_environment or cluster
            missing = [name for name in names if not self._has_location(catalog, name, environment)]
            return (
                DoctorCheck(
                    "data-catalog",
                    not missing,
                    (
                        f"Datasets {names} resolve for {environment!r}."
                        if not missing
                        else f"Missing {environment!r} locations for {tuple(missing)}."
                    ),
                    "Add each target location or replicate the dataset explicitly.",
                ),
            )
        except Exception as error:
            return (
                DoctorCheck(
                    "data-catalog",
                    False,
                    f"{error.__class__.__name__}: {error}",
                    "Fix the config/catalog and run validate before submission.",
                ),
            )

    @staticmethod
    def _has_location(catalog: object, name: str, environment: str) -> bool:
        try:
            catalog.resolve(f"dataset:{name}", environment=environment)  # type: ignore[attr-defined]
        except (KeyError, TypeError, ValueError):
            return False
        return True

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
