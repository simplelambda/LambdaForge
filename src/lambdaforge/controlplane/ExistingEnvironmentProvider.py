"""Verification provider for administrator/user managed environments."""

from __future__ import annotations

from pathlib import Path

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.EnvironmentProvider import EnvironmentProvider
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.PreparedEnvironment import PreparedEnvironment
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


class ExistingEnvironmentProvider(EnvironmentProvider):
    """Require the configured interpreter to contain the expected framework release."""

    def prepare(
        self,
        profile: ClusterProfile,
        transport: Transport,
        bundle: ExecutionBundle,
        *,
        remote_bundle_dir: str | Path,
    ) -> PreparedEnvironment:
        """Verify the existing interpreter without installing or mutating it."""
        del bundle, remote_bundle_dir
        result = transport.run(
            (
                *profile.command_prefix,
                profile.python,
                "-c",
                (
                    "import lambdaforge; "
                    f"assert lambdaforge.__version__ == {LambdaForgeVersion.CURRENT!r}, "
                    "(lambdaforge.__version__)"
                ),
            )
        )
        if result.returncode:
            raise RuntimeError(
                "The configured existing environment does not contain the expected "
                f"LambdaForge {LambdaForgeVersion.CURRENT}: {result.stderr.strip()}"
            )
        return PreparedEnvironment("existing", profile.python, True)
