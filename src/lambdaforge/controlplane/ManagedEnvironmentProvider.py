"""Portable venv/pip managed environment provider."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.EnvironmentProvider import EnvironmentProvider
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.PreparedEnvironment import PreparedEnvironment
from lambdaforge.controlplane.Transport import Transport


class ManagedEnvironmentProvider(EnvironmentProvider):
    """Create idempotent user-space venvs from staged exact wheels."""

    def prepare(
        self,
        profile: ClusterProfile,
        transport: Transport,
        bundle: ExecutionBundle,
        *,
        remote_bundle_dir: str | Path,
    ) -> PreparedEnvironment:
        """Create once, install exact wheels, verify, then atomically mark ready."""
        if bundle.environment_id is None:
            raise ValueError("Managed environment bundles require an environment identity.")
        remote = PurePosixPath(str(remote_bundle_dir))
        environment = (
            PurePosixPath(profile.workspace)
            / ".lambdaforge"
            / "environments"
            / bundle.environment_id
        )
        python = environment / "bin" / "python"
        marker = environment / ".lambdaforge-environment.json"
        cached = transport.run(("test", "-f", str(marker)))
        if cached.returncode == 0:
            verified = transport.run(
                (str(python), "-c", "import lambdaforge; print(lambdaforge.__version__)")
            )
            if verified.returncode == 0:
                self._activate(profile, transport, str(python))
                return PreparedEnvironment(bundle.environment_id, str(python), True)

        created = transport.run(
            (*profile.command_prefix, profile.python, "-m", "venv", str(environment))
        )
        if created.returncode:
            raise RuntimeError(
                "Could not create the managed venv. Ensure the configured Python provides venv: "
                f"{created.stderr.strip()}"
            )
        wheel_dir = remote / "packages"
        wheel_names = tuple(bundle.package_names)
        if not wheel_names:
            raise ValueError("Managed environment bundle contains no exact wheels.")
        install = [str(python), "-m", "pip", "install"]
        if bundle.offline:
            install.extend(("--no-index", "--find-links", str(remote / "wheelhouse")))
        else:
            install.extend(("--find-links", str(remote / "wheelhouse")))
        install.extend(str(wheel_dir / name) for name in wheel_names)
        installed = transport.run(tuple(install))
        if installed.returncode:
            hint = (
                "The offline wheelhouse is incomplete for the remote platform."
                if bundle.offline
                else "Dependency installation failed; inspect pip output and cluster connectivity."
            )
            raise RuntimeError(f"{hint} {installed.stderr.strip()}")
        verified = transport.run(
            (str(python), "-c", "import lambdaforge, torch; print(lambdaforge.__version__)")
        )
        if verified.returncode:
            raise RuntimeError(
                f"Managed environment verification failed: {verified.stderr.strip()}"
            )
        payload = json.dumps(
            {"environment_id": bundle.environment_id, "packages": list(wheel_names)},
            sort_keys=True,
        )
        marked = transport.run(
            (
                str(python),
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
                str(marker),
                payload,
            )
        )
        if marked.returncode:
            raise RuntimeError(f"Could not mark managed environment ready: {marked.stderr.strip()}")
        self._activate(profile, transport, str(python))
        return PreparedEnvironment(bundle.environment_id, str(python), False)

    @staticmethod
    def _activate(profile: ClusterProfile, transport: Transport, python: str) -> None:
        """Publish the interpreter selected by the latest successful preparation."""
        pointer = PurePosixPath(profile.workspace) / ".lambdaforge" / "active-environment"
        result = transport.run(
            (
                python,
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
                str(pointer),
                f"{python}\n",
            )
        )
        if result.returncode:
            raise RuntimeError(f"Could not publish the active managed environment: {result.stderr}")
