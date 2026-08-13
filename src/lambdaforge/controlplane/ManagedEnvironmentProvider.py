"""Portable venv/pip managed environment provider."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.EnvironmentProvider import EnvironmentProvider
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.PreparedEnvironment import PreparedEnvironment
from lambdaforge.controlplane.TorchInstallationPlan import TorchInstallationPlan
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


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
        policy = dict(bundle.environment_policy or {})
        raw_plan = policy.get("pytorch")
        plan = (
            TorchInstallationPlan.from_mapping(raw_plan) if isinstance(raw_plan, Mapping) else None
        )
        cached = transport.run(("test", "-f", str(marker)))
        if cached.returncode == 0:
            verified = self._verify(transport, str(python), plan)
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
        constraint: PurePosixPath | None = None
        if not bundle.offline and plan is not None:
            if plan.version is None or plan.index_url is None:
                raise ValueError("Online managed environments require an exact PyTorch plan.")
            torch_install = transport.run(
                (
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--index-url",
                    plan.index_url,
                    f"torch=={plan.version}",
                )
            )
            if torch_install.returncode:
                raise RuntimeError(
                    f"Could not install resolved PyTorch {plan.version} from {plan.channel}: "
                    f"{torch_install.stderr.strip()}"
                )
            constraint = remote / "torch-constraint.txt"
            written = transport.run(
                (
                    str(python),
                    "-c",
                    "from pathlib import Path; import sys; "
                    "Path(sys.argv[1]).write_text(sys.argv[2])",
                    str(constraint),
                    f"torch=={plan.version}\n",
                )
            )
            if written.returncode:
                raise RuntimeError(
                    f"Could not write the managed PyTorch constraint: {written.stderr.strip()}"
                )
        install = [str(python), "-m", "pip", "install"]
        if bundle.offline:
            install.extend(("--no-index", "--find-links", str(remote / "wheelhouse")))
        else:
            install.extend(("--find-links", str(remote / "wheelhouse")))
        if constraint is not None:
            install.extend(("--constraint", str(constraint)))
        install.extend(str(wheel_dir / name) for name in wheel_names)
        installed = transport.run(tuple(install))
        if installed.returncode:
            hint = (
                "The offline wheelhouse is incomplete for the remote platform."
                if bundle.offline
                else "Dependency installation failed; inspect pip output and cluster connectivity."
            )
            raise RuntimeError(f"{hint} {installed.stderr.strip()}")
        verified = self._verify(transport, str(python), plan)
        if verified.returncode:
            raise RuntimeError(
                f"Managed environment verification failed: {verified.stderr.strip()}"
            )
        payload = json.dumps(
            {
                "environment_id": bundle.environment_id,
                "packages": list(wheel_names),
                "environment_policy": policy,
            },
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
    def _verify(
        transport: Transport, python: str, plan: TorchInstallationPlan | None
    ) -> CommandResult:
        """Verify exact framework import and required CUDA initialization."""
        require_cuda = bool(plan and plan.require_cuda)
        expected_torch = plan.version if plan is not None else None
        code = (
            "import lambdaforge,sys,torch\n"
            f"assert lambdaforge.__version__ == {LambdaForgeVersion.CURRENT!r}\n"
            f"assert {expected_torch!r} is None or torch.__version__ == {expected_torch!r}\n"
            f"required={require_cuda!r}\n"
            "available=torch.cuda.is_available()\n"
            "if required and not available: sys.exit(3)\n"
            "if available:\n"
            " for device in range(torch.cuda.device_count()):\n"
            "  with torch.cuda.device(device):\n"
            "   probe=torch.ones(1, device='cuda').mul_(2)\n"
            "   assert probe.item() == 2\n"
            "print(lambdaforge.__version__, torch.__version__, torch.version.cuda, available)\n"
        )
        return transport.run((python, "-c", code))

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
