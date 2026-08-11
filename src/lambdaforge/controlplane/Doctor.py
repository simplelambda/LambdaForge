"""Local and remote preflight diagnostics."""

from __future__ import annotations

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.DoctorCheck import DoctorCheck
from lambdaforge.controlplane.DoctorReport import DoctorReport


class Doctor:
    """Check access, Python, framework, scheduler and optional CUDA visibility."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()

    def check(self, cluster: str = "local") -> DoctorReport:
        """Run read-only diagnostics through the selected transport."""
        profile = self.catalog.get(cluster)
        transport = self.factory.transport(profile)
        checks: list[DoctorCheck] = []
        python = transport.run((*profile.command_prefix, profile.python, "--version"))
        checks.append(
            DoctorCheck(
                "python",
                python.returncode == 0,
                (python.stdout or python.stderr).strip() or "Python was not found.",
                f"Set clusters.{cluster}.python to a working Python executable.",
            )
        )
        framework = transport.run(
            (
                *profile.command_prefix,
                profile.python,
                "-c",
                "import lambdaforge; print(lambdaforge.__version__)",
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
        cuda = transport.run(
            (
                *profile.command_prefix,
                profile.python,
                "-c",
                "import torch; print(torch.cuda.is_available(), torch.version.cuda)",
            )
        )
        checks.append(
            DoctorCheck(
                "torch-cuda",
                cuda.returncode == 0,
                cuda.stdout.strip() or cuda.stderr.strip(),
                "Install a PyTorch build matching the visible NVIDIA driver when GPU is required.",
            )
        )
        return DoctorReport(cluster, tuple(checks))
