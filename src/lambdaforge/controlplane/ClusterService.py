"""Application service for cluster workspace and environment preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path, PurePosixPath

from lambdaforge.controlplane.ClusterBootstrapResult import ClusterBootstrapResult
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.EnvironmentIdentity import EnvironmentIdentity
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.ProjectWheelBuilder import ProjectWheelBuilder


class ClusterService:
    """Prepare a cluster entirely in user space through configured providers."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
        cache_root: str | Path = ".lambdaforge/control",
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.cache_root = Path(cache_root).resolve()

    def bootstrap(
        self, cluster: str, *, wheelhouse: str | Path | None = None
    ) -> ClusterBootstrapResult:
        """Create workspace and idempotently verify/install the configured environment."""
        profile = self.catalog.get(cluster)
        if wheelhouse is not None:
            profile = replace(profile, wheelhouse=str(Path(wheelhouse).expanduser().resolve()))
        transport = self.factory.transport(profile)
        created = transport.run(
            (
                "mkdir",
                "-p",
                str(PurePosixPath(profile.workspace) / ".lambdaforge" / "bundles"),
                str(PurePosixPath(profile.workspace) / ".lambdaforge" / "environments"),
            )
        )
        if created.returncode:
            raise RuntimeError(f"Could not create cluster workspace: {created.stderr.strip()}")
        if profile.environment == "existing":
            probe = transport.run(
                (*profile.command_prefix, profile.python, "-c", "import lambdaforge, torch")
            )
            if probe.returncode:
                raise RuntimeError(
                    "The existing environment is not ready. Install the pinned LambdaForge "
                    f"release and project dependencies: {probe.stderr.strip()}"
                )
            return ClusterBootstrapResult(cluster, "existing", profile.python, True)

        framework_root = Path(__file__).resolve().parents[3]
        wheel = ProjectWheelBuilder(self.cache_root / "wheels").build(framework_root)
        descriptors = [
            {
                "name": wheel.name,
                "sha256": f"sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}",
                "size_bytes": wheel.stat().st_size,
            }
        ]
        if profile.wheelhouse is not None:
            for dependency in sorted(Path(profile.wheelhouse).expanduser().glob("*.whl")):
                descriptors.append(
                    {
                        "name": f"wheelhouse/{dependency.name}",
                        "sha256": f"sha256:{hashlib.sha256(dependency.read_bytes()).hexdigest()}",
                        "size_bytes": dependency.stat().st_size,
                    }
                )
        identity = EnvironmentIdentity.create(
            descriptors, python_requirement=">=3.10", offline=profile.wheelhouse is not None
        )
        directory = self.cache_root / "bootstrap" / identity.environment_id
        packages = directory / "packages"
        packages.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel, packages / wheel.name)
        wheelhouse = directory / "wheelhouse"
        wheelhouse.mkdir(parents=True, exist_ok=True)
        if profile.wheelhouse is not None:
            for dependency in sorted(Path(profile.wheelhouse).expanduser().glob("*.whl")):
                shutil.copy2(dependency, wheelhouse / dependency.name)
        manifest = directory / "manifest.json"
        identity_payload = identity.to_dict()
        manifest.write_text(
            json.dumps(identity_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        bundle = ExecutionBundle(
            f"bootstrap-{identity.environment_id}",
            directory,
            manifest,
            manifest,
            sum(item.stat().st_size for item in directory.rglob("*") if item.is_file()),
            environment_id=identity.environment_id,
            package_names=(wheel.name,),
            offline=identity.offline,
        )
        remote = (
            PurePosixPath(profile.workspace)
            / ".lambdaforge"
            / "bootstrap"
            / identity.environment_id
        )
        cached = transport.run(("test", "-f", str(remote / "manifest.json")))
        if cached.returncode != 0:
            parent = transport.run(("mkdir", "-p", str(remote.parent)))
            if parent.returncode:
                raise RuntimeError(f"Could not create bootstrap cache: {parent.stderr.strip()}")
            transport.put(directory, str(remote))
        prepared = self.factory.environment_provider(profile).prepare(
            profile, transport, bundle, remote_bundle_dir=str(remote)
        )
        return ClusterBootstrapResult(
            cluster, prepared.environment_id, prepared.python, prepared.reused
        )
