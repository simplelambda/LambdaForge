"""Portable venv/pip managed environment provider."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from uuid import uuid4

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.EnvironmentProvider import EnvironmentProvider
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.NativeEnvironment import (
    NativeEnvironmentError,
    NativeEnvironmentPlan,
)
from lambdaforge.controlplane.PreparedEnvironment import PreparedEnvironment
from lambdaforge.controlplane.TlsTrust import TlsTrust
from lambdaforge.controlplane.TorchInstallationPlan import TorchInstallationPlan
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


class ManagedEnvironmentProvider(EnvironmentProvider):
    """Create immutable pip-only venvs or unified native Conda prefixes."""

    def prepare(
        self,
        profile: ClusterProfile,
        transport: Transport,
        bundle: ExecutionBundle,
        *,
        remote_bundle_dir: str | Path,
    ) -> PreparedEnvironment:
        """Build under an advisory marker so cache GC fails closed during mutation."""
        assert profile.storage is not None
        cache_root = PurePosixPath(profile.storage.cache_root)
        if bundle.environment_id is None:
            raise ValueError("Managed environment bundles require an environment identity.")
        marker = cache_root / f".environment-build-{bundle.environment_id}.lock"
        completion = (
            PurePosixPath(profile.storage.environment_root)
            / bundle.environment_id
            / ".lambdaforge-environment.json"
        )
        created = transport.run(("mkdir", "-p", str(cache_root)))
        if created.returncode:
            raise RuntimeError(f"Could not create managed cache root: {created.stderr.strip()}")
        acquired = self._acquire(transport, marker, completion)
        if not acquired:
            return self._prepare(
                profile,
                transport,
                bundle,
                remote_bundle_dir=remote_bundle_dir,
            )
        try:
            return self._prepare(
                profile,
                transport,
                bundle,
                remote_bundle_dir=remote_bundle_dir,
            )
        finally:
            transport.run(("rmdir", str(marker)))

    def _prepare(
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
        assert profile.storage is not None
        environment_root = PurePosixPath(profile.storage.environment_root)
        environment = environment_root / bundle.environment_id
        python = environment / "bin" / "python"
        marker = environment / ".lambdaforge-environment.json"
        policy = dict(bundle.environment_policy or {})
        raw_plan = policy.get("pytorch")
        plan = (
            TorchInstallationPlan.from_mapping(raw_plan) if isinstance(raw_plan, Mapping) else None
        )
        raw_runtime = policy.get("python_runtime")
        raw_trust = raw_runtime.get("tls_trust") if isinstance(raw_runtime, Mapping) else None
        trust = TlsTrust.from_mapping(raw_trust if isinstance(raw_trust, Mapping) else None)
        raw_native = policy.get("native_environment")
        native = (
            NativeEnvironmentPlan.from_mapping(raw_native)
            if isinstance(raw_native, Mapping)
            else None
        )
        cached = transport.run(("test", "-f", str(marker)))
        if cached.returncode == 0:
            verified = self._verify_reusable(
                profile, transport, environment, str(python), plan, trust, native
            )
            if verified.returncode == 0:
                self._activate(profile, transport, str(python))
                return PreparedEnvironment(bundle.environment_id, str(python), True)
        legacy_environment = (
            PurePosixPath(profile.workspace)
            / ".lambdaforge"
            / "environments"
            / bundle.environment_id
        )
        legacy_python = legacy_environment / "bin" / "python"
        legacy_marker = legacy_environment / ".lambdaforge-environment.json"
        legacy = transport.run(("test", "-f", str(legacy_marker)))
        if legacy.returncode == 0:
            verified = self._verify_reusable(
                profile,
                transport,
                legacy_environment,
                str(legacy_python),
                plan,
                trust,
                native,
            )
            if verified.returncode == 0:
                self._activate(profile, transport, str(legacy_python))
                return PreparedEnvironment(bundle.environment_id, str(legacy_python), True)
        temporary = environment_root / f".{bundle.environment_id}.tmp-{uuid4().hex}"
        temporary_python = temporary / "bin" / "python"
        temporary_marker = temporary / ".lambdaforge-environment.json"
        created_root = transport.run(("mkdir", "-p", str(environment_root)))
        if created_root.returncode:
            raise RuntimeError(f"Could not create environment cache: {created_root.stderr.strip()}")
        created = (
            self._create_native(
                profile,
                transport,
                native,
                temporary,
                remote,
                trust,
            )
            if native is not None
            else transport.run(
                (
                    *profile.command_prefix,
                    *(() if trust is None else trust.prefix()),
                    profile.python,
                    "-m",
                    "venv",
                    str(temporary),
                )
            )
        )
        if created.returncode:
            self._cleanup(transport, temporary)
            kind = "native Conda prefix" if native is not None else "managed venv"
            raise RuntimeError(f"Could not create the {kind}: {created.stderr.strip()}")
        if native is not None:
            inventory = self._verify_native_inventory(profile, transport, temporary, native, trust)
            if inventory.returncode:
                self._cleanup(transport, temporary)
                raise NativeEnvironmentError(
                    "Native package inventory differs from the resolved environment identity: "
                    f"{inventory.stderr.strip()}"
                )
        wheel_dir = remote / "packages"
        wheel_names = tuple(bundle.package_names)
        if not wheel_names:
            self._cleanup(transport, temporary)
            raise ValueError("Managed environment bundle contains no exact wheels.")
        constraint: PurePosixPath | None = None
        if not bundle.offline and plan is not None:
            if plan.version is None or plan.index_url is None:
                raise ValueError("Online managed environments require an exact PyTorch plan.")
            torch_install = transport.run(
                (
                    *(() if trust is None else trust.prefix()),
                    str(temporary_python),
                    "-m",
                    "pip",
                    "install",
                    "--cache-dir",
                    str(PurePosixPath(profile.storage.cache_root) / "pip"),
                    "--index-url",
                    plan.index_url,
                    f"torch=={plan.version}",
                )
            )
            if torch_install.returncode:
                self._cleanup(transport, temporary)
                raise RuntimeError(
                    f"Could not install resolved PyTorch {plan.version} from {plan.channel}: "
                    f"{torch_install.stderr.strip()}"
                )
            constraint = remote / "torch-constraint.txt"
            written = transport.run(
                (
                    *(() if trust is None else trust.prefix()),
                    str(temporary_python),
                    "-c",
                    "from pathlib import Path; import sys; "
                    "Path(sys.argv[1]).write_text(sys.argv[2])",
                    str(constraint),
                    f"torch=={plan.version}\n",
                )
            )
            if written.returncode:
                self._cleanup(transport, temporary)
                raise RuntimeError(
                    f"Could not write the managed PyTorch constraint: {written.stderr.strip()}"
                )
        install = [
            *(() if trust is None else trust.prefix()),
            str(temporary_python),
            "-m",
            "pip",
            "install",
            "--cache-dir",
            str(PurePosixPath(profile.storage.cache_root) / "pip"),
        ]
        if bundle.offline:
            install.extend(("--no-index", "--find-links", str(remote / "wheelhouse")))
        if constraint is not None:
            install.extend(("--constraint", str(constraint)))
        install.extend(str(wheel_dir / name) for name in wheel_names)
        installed = transport.run(tuple(install))
        if installed.returncode:
            self._cleanup(transport, temporary)
            hint = (
                "The offline wheelhouse is incomplete for the remote platform."
                if bundle.offline
                else "Dependency installation failed; inspect pip output and cluster connectivity."
            )
            raise RuntimeError(f"{hint} {installed.stderr.strip()}")
        checked = transport.run(
            (
                *(() if trust is None else trust.prefix()),
                str(temporary_python),
                "-m",
                "pip",
                "check",
            )
        )
        if checked.returncode:
            self._cleanup(transport, temporary)
            raise RuntimeError(
                "Managed Python dependencies are inconsistent after installation: "
                f"{checked.stdout.strip() or checked.stderr.strip()}"
            )
        if native is not None:
            inventory = self._verify_native_inventory(profile, transport, temporary, native, trust)
            if inventory.returncode:
                self._cleanup(transport, temporary)
                raise NativeEnvironmentError(
                    "Pip installation changed the identity-bearing native inventory: "
                    f"{inventory.stderr.strip()}"
                )
        verified = self._verify(transport, str(temporary_python), plan, trust, native)
        if verified.returncode:
            self._cleanup(transport, temporary)
            raise RuntimeError(
                f"Managed environment verification failed: {verified.stderr.strip()}"
            )
        native_tools: list[object] = []
        if native is not None:
            native_evidence = self._native_tools(transport, str(temporary_python), native)
            if native_evidence.returncode:
                self._cleanup(transport, temporary)
                raise NativeEnvironmentError(
                    f"Required native executable verification failed: {native_evidence.stderr}"
                )
            try:
                parsed_tools = json.loads(native_evidence.stdout or "[]")
            except json.JSONDecodeError as error:
                self._cleanup(transport, temporary)
                raise RuntimeError("Native executable evidence is not valid JSON.") from error
            if not isinstance(parsed_tools, list):
                self._cleanup(transport, temporary)
                raise RuntimeError("Native executable evidence has an invalid shape.")
            native_tools = parsed_tools
        payload = json.dumps(
            {
                "environment_id": bundle.environment_id,
                "packages": list(wheel_names),
                "environment_policy": policy,
                "native_tools": native_tools,
            },
            sort_keys=True,
        )
        marked = transport.run(
            (
                *(() if trust is None else trust.prefix()),
                str(temporary_python),
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
                str(temporary_marker),
                payload,
            )
        )
        if marked.returncode:
            self._cleanup(transport, temporary)
            raise RuntimeError(f"Could not mark managed environment ready: {marked.stderr.strip()}")
        # Only complete verified environments receive their content-addressed final name.
        if cached.returncode == 0:
            stale = environment_root / f".{bundle.environment_id}.stale-{uuid4().hex}"
            moved = transport.run(("mv", str(environment), str(stale)))
            if moved.returncode == 0:
                self._cleanup(transport, stale)
        published = transport.run(
            (
                profile.python,
                "-c",
                "import os,sys; os.rename(sys.argv[1], sys.argv[2])",
                str(temporary),
                str(environment),
            )
        )
        if published.returncode:
            self._cleanup(transport, temporary)
            # Another bootstrap may have won the race; accept it only after verification.
            verified = self._verify_reusable(
                profile, transport, environment, str(python), plan, trust, native
            )
            if verified.returncode:
                raise RuntimeError(
                    f"Could not atomically publish managed environment: {published.stderr.strip()}"
                )
        self._activate(profile, transport, str(python))
        return PreparedEnvironment(bundle.environment_id, str(python), False)

    @staticmethod
    def _verify(
        transport: Transport,
        python: str,
        plan: TorchInstallationPlan | None,
        trust: TlsTrust | None = None,
        native: NativeEnvironmentPlan | None = None,
    ) -> CommandResult:
        """Verify exact framework import and required CUDA initialization."""
        require_cuda = bool(plan and plan.require_cuda)
        expected_torch = plan.version if plan is not None else None
        code = (
            "import lambdaforge,ssl,sys,torch\n"
            f"assert lambdaforge.__version__ == {LambdaForgeVersion.CURRENT!r}\n"
            f"assert {expected_torch!r} is None or torch.__version__ == {expected_torch!r}\n"
            "assert ssl.create_default_context().get_ca_certs(), 'no trusted TLS CAs'\n"
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
        verified = transport.run((*(() if trust is None else trust.prefix()), python, "-c", code))
        if verified.returncode or native is None:
            return verified
        return ManagedEnvironmentProvider._native_tools(transport, python, native)

    @staticmethod
    def _verify_reusable(
        profile: ClusterProfile,
        transport: Transport,
        prefix: PurePosixPath,
        python: str,
        plan: TorchInstallationPlan | None,
        trust: TlsTrust | None,
        native: NativeEnvironmentPlan | None,
    ) -> CommandResult:
        """Verify runtime behavior and exact native inventory before cache reuse."""
        verified = ManagedEnvironmentProvider._verify(transport, python, plan, trust, native)
        if verified.returncode or native is None:
            return verified
        return ManagedEnvironmentProvider._verify_native_inventory(
            profile, transport, prefix, native, trust
        )

    @staticmethod
    def _create_native(
        profile: ClusterProfile,
        transport: Transport,
        native: NativeEnvironmentPlan,
        temporary: PurePosixPath,
        remote_bundle: PurePosixPath,
        trust: TlsTrust | None,
    ) -> CommandResult:
        """Create the one final prefix from an exact native plan, never through activation."""
        assert profile.storage is not None
        cache_root = PurePosixPath(profile.storage.cache_root)
        manager = (
            cache_root
            / "runtime-managers"
            / f"micromamba-{native.manager_version}-{native.platform}"
        )
        package_root = (
            cache_root / "native-packages" / native.package_cache_id / "packages"
            if native.package_cache_id is not None
            else cache_root / "conda-pkgs"
        )
        command = [
            *profile.command_prefix,
            "env",
            f"MAMBA_ROOT_PREFIX={cache_root / 'micromamba-root'}",
            f"CONDA_PKGS_DIRS={package_root}",
            "CONDARC=/dev/null",
            "MAMBARC=/dev/null",
            *(() if trust is None else trust.assignments()),
            str(manager),
            "create",
            "--yes",
            "--prefix",
            str(temporary),
            "--platform",
            native.platform,
        ]
        if native.offline:
            command.extend(
                (
                    "--offline",
                    "--file",
                    str(remote_bundle / "native" / native.specification_name),
                )
            )
        else:
            command.append("--override-channels")
            for channel in native.channels:
                command.extend(("--channel", channel))
            command.extend(native.exact_packages)
        return transport.run(tuple(command), timeout=900.0)

    @classmethod
    def _verify_native_inventory(
        cls,
        profile: ClusterProfile,
        transport: Transport,
        prefix: PurePosixPath,
        native: NativeEnvironmentPlan,
        trust: TlsTrust | None,
    ) -> CommandResult:
        """Compare regular conda-meta records with the identity-bearing solve."""
        assert profile.storage is not None
        manager = (
            PurePosixPath(profile.storage.cache_root)
            / "runtime-managers"
            / f"micromamba-{native.manager_version}-{native.platform}"
        )
        listed = transport.run(
            (
                *profile.command_prefix,
                *(() if trust is None else trust.prefix()),
                str(manager),
                "list",
                "--json",
                "--prefix",
                str(prefix),
            )
        )
        if listed.returncode:
            return listed
        metadata_code = (
            "# lambdaforge-conda-meta-inventory\n"
            "import json,sys\n"
            "from pathlib import Path\n"
            "records=[]\n"
            "try:\n"
            " for path in sorted(Path(sys.argv[1]).glob('*.json')):\n"
            "  item=json.loads(path.read_text(encoding='utf-8'))\n"
            "  if not isinstance(item,dict): raise TypeError(f'{path.name} is not an object')\n"
            "  records.append(item)\n"
            "except Exception as error:\n"
            " print(f'{type(error).__name__}: {error}',file=sys.stderr)\n"
            " raise SystemExit(2)\n"
            "print(json.dumps(records,sort_keys=True))\n"
        )
        metadata = transport.run(
            (
                *profile.command_prefix,
                str(prefix / "bin" / "python"),
                "-c",
                metadata_code,
                str(prefix / "conda-meta"),
            )
        )
        if metadata.returncode:
            detail = (metadata.stderr or metadata.stdout).strip().splitlines()
            return CommandResult(
                1,
                stderr=(
                    "Installed Conda metadata could not be read"
                    + (f": {detail[-1][:300]}" if detail else ".")
                ),
            )
        try:
            raw_list = json.loads(listed.stdout)
            raw_metadata = json.loads(metadata.stdout)
            if not isinstance(raw_list, list) or not isinstance(raw_metadata, list):
                raise TypeError("inventory responses must be arrays")
            listed_by_package: dict[tuple[str, str, str], Mapping[str, object]] = {}
            for item in raw_list:
                if not isinstance(item, Mapping):
                    raise TypeError("micromamba list contains a non-object record")
                key = cls._native_package_key(item)
                if key in listed_by_package:
                    raise ValueError(f"micromamba list contains duplicate package {key[0]!r}")
                listed_by_package[key] = item
            actual: set[tuple[str, str, str, str, str]] = set()
            conda_keys: set[tuple[str, str, str]] = set()
            for item in raw_metadata:
                if not isinstance(item, Mapping):
                    raise TypeError("conda-meta contains a non-object record")
                key = cls._native_package_key(item)
                if key in conda_keys:
                    raise ValueError(f"conda-meta contains duplicate package {key[0]!r}")
                conda_keys.add(key)
                subdir = str(item.get("subdir", "")).strip()
                if not subdir:
                    raise ValueError(f"conda-meta lacks subdir for package {key[0]!r}")
                listed_item = listed_by_package.get(key, {})
                channel = str(
                    listed_item.get(
                        "channel",
                        listed_item.get("base_url", item.get("channel", item.get("base_url", ""))),
                    )
                ).rstrip("/")
                if not channel:
                    raise ValueError(f"installed metadata lacks channel for package {key[0]!r}")
                actual.add((*key, channel, subdir))
            unmanaged = [
                key
                for key, item in listed_by_package.items()
                if key not in conda_keys
                and str(item.get("channel", "")).strip().lower() not in {"pypi", "pypi_0"}
            ]
            if unmanaged:
                names = ", ".join(repr(item[0]) for item in sorted(unmanaged)[:5])
                suffix = f" (+{len(unmanaged) - 5} more)" if len(unmanaged) > 5 else ""
                raise ValueError(f"conda-meta is missing for installed package(s) {names}{suffix}")
            expected = {
                (
                    item["name"],
                    item["version"],
                    item["build"],
                    item["channel"].rstrip("/"),
                    item["subdir"],
                )
                for item in native.package_inventory
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            return CommandResult(1, stderr=f"Invalid native package inventory: {error}")
        if actual != expected:
            return CommandResult(
                1,
                stderr=cls._summarize_native_inventory_difference(expected, actual),
            )
        return CommandResult(0, listed.stdout)

    @staticmethod
    def _native_package_key(item: Mapping[str, object]) -> tuple[str, str, str]:
        """Read the stable package key shared by solver, list and conda-meta JSON."""
        name = str(item.get("name", "")).strip()
        version = str(item.get("version", "")).strip()
        build = str(item.get("build_string", item.get("build", ""))).strip()
        if not name or not version or not build:
            raise ValueError("installed package metadata lacks name, version or build")
        return name, version, build

    @staticmethod
    def _summarize_native_inventory_difference(
        expected: set[tuple[str, str, str, str, str]],
        actual: set[tuple[str, str, str, str, str]],
        *,
        limit: int = 6,
    ) -> str:
        """Describe a bounded package-level diff instead of dumping whole environments."""
        fields = ("version", "build", "channel", "subdir")
        expected_by_name = {item[0]: item for item in expected}
        actual_by_name = {item[0]: item for item in actual}
        differences: list[str] = []
        for name in sorted(expected_by_name.keys() | actual_by_name.keys()):
            wanted = expected_by_name.get(name)
            observed = actual_by_name.get(name)
            if wanted is None:
                differences.append(f"unexpected {name!r}")
                continue
            if observed is None:
                differences.append(f"missing {name!r}")
                continue
            changed = [
                f"{field} {wanted[index]!r} -> {observed[index]!r}"
                for index, field in enumerate(fields, start=1)
                if wanted[index] != observed[index]
            ]
            if changed:
                differences.append(f"changed {name!r}: {', '.join(changed)}")
        shown = differences[:limit]
        if len(differences) > limit:
            shown.append(f"+{len(differences) - limit} more")
        return f"{len(differences)} package difference(s): {'; '.join(shown)}"

    @staticmethod
    def _native_tools(
        transport: Transport,
        python: str,
        native: NativeEnvironmentPlan,
    ) -> CommandResult:
        """Verify required executables and report their exact Conda package provenance."""
        if not native.required_executables:
            return CommandResult(0, "[]")
        code = (
            "import json,os,subprocess,sys\n"
            "from pathlib import Path\n"
            "prefix=Path(sys.executable).parent.parent\n"
            "records=[]\n"
            "metadata=[]\n"
            "for item in sorted((prefix/'conda-meta').glob('*.json')):\n"
            " try: metadata.append(json.loads(item.read_text()))\n"
            " except Exception: pass\n"
            "for name in json.loads(sys.argv[1]):\n"
            " path=(prefix/'bin'/name).resolve()\n"
            " if not path.is_file() or not os.access(path,os.X_OK):\n"
            "  raise SystemExit(f'missing executable: {name}')\n"
            " relative=path.relative_to(prefix).as_posix()\n"
            " owner=next((item for item in metadata if relative in item.get('files',[])),None)\n"
            " if owner is None: raise SystemExit(f'no Conda package owns executable: {name}')\n"
            " version=None\n"
            " for args in (['--version'],['version']):\n"
            "  try:\n"
            "   result=subprocess.run([str(path),*args],capture_output=True,text=True,timeout=10)\n"
            "  except Exception: continue\n"
            "  if result.returncode==0:\n"
            "   version=(result.stdout or result.stderr).strip()[:2000] or None; break\n"
            " if version is None: raise SystemExit(f'could not query executable version: {name}')\n"
            " records.append({'name':name,'path':str(path),'version':version,"
            "'package':owner.get('name'),'package_version':owner.get('version'),"
            "'build':owner.get('build'),'channel':owner.get('channel'),"
            "'subdir':owner.get('subdir')})\n"
            "print(json.dumps(records,sort_keys=True))\n"
        )
        return transport.run((python, "-c", code, json.dumps(native.required_executables)))

    @staticmethod
    def _activate(profile: ClusterProfile, transport: Transport, python: str) -> None:
        """Publish the interpreter selected by the latest successful preparation."""
        assert profile.storage is not None
        pointer = PurePosixPath(profile.storage.state_root) / "active-environment"
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

    @staticmethod
    def _cleanup(transport: Transport, path: PurePosixPath) -> None:
        """Remove one exact provider-created temporary path, never a broad root."""
        if (
            not path.name.startswith(".env-")
            and ".tmp-" not in path.name
            and ".stale-" not in path.name
        ):
            raise ValueError(f"Refusing to clean an unexpected environment path: {path}")
        transport.run(("rm", "-rf", str(path)))

    @staticmethod
    def _acquire(
        transport: Transport,
        lock: PurePosixPath,
        completion: PurePosixPath,
        *,
        timeout: float = 30.0,
    ) -> bool:
        """Serialize identical environment creation with a bounded remote mkdir lock."""
        deadline = time.monotonic() + timeout
        while True:
            if transport.run(("mkdir", str(lock))).returncode == 0:
                return True
            if transport.run(("test", "-f", str(completion))).returncode == 0:
                return False
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for managed environment lock {lock}.")
            time.sleep(0.2)
