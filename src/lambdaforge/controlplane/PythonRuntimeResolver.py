"""Bounded discovery and provisioning of cluster Python runtimes."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from lambdaforge.controlplane.MicromambaArtifactStore import MicromambaArtifactStore
from lambdaforge.controlplane.python_runtime import (
    NoCompatiblePythonRuntimeError,
    PythonRuntime,
    PythonRuntimePolicy,
    PythonRuntimeRequirements,
)
from lambdaforge.controlplane.Transport import Transport

if TYPE_CHECKING:
    from lambdaforge.controlplane.ClusterProfile import ClusterProfile


class PythonRuntimeResolver:
    """Resolve one compatible interpreter and provision only a bounded managed fallback."""

    DEFAULT_MINOR_CANDIDATES = ("3.13", "3.12", "3.11", "3.10", "3.14")
    ACTIVE_POINTER = "active-python-runtime.json"
    MARKER = ".lambdaforge-python-runtime.json"

    def __init__(
        self,
        artifacts: MicromambaArtifactStore | None = None,
        *,
        lock_timeout: float = 30.0,
    ) -> None:
        self.artifacts = artifacts or MicromambaArtifactStore()
        self.lock_timeout = lock_timeout

    def resolve(
        self,
        profile: ClusterProfile,
        transport: Transport,
        *,
        requirements: Sequence[str | None] = (),
        excluded_runtime_ids: Sequence[str] = (),
        dry_run: bool = False,
    ) -> PythonRuntime:
        """Resolve/provision one candidate without mutating system Python or shell startup files."""
        policy = profile.runtime_policy
        required = PythonRuntimeRequirements.normalized(
            (PythonRuntimeRequirements.framework(), *requirements)
        )
        detected: list[str] = []
        excluded = set(excluded_runtime_ids)
        active = self.active(profile, transport)
        if (
            active is not None
            and active.runtime_id not in excluded
            and (policy.strategy != "managed" or active.managed)
        ):
            probed = self._probe(profile, transport, active.executable)
            self._record_probe(detected, active.executable, probed)
            if probed is not None and self._acceptable(probed, policy, required):
                return self._with_action(probed, active, "reuse")

        if policy.strategy != "managed":
            names = self._existing_candidates(policy)
            for executable in names:
                runtime = self._probe(profile, transport, executable)
                self._record_probe(detected, executable, runtime)
                if (
                    runtime is not None
                    and runtime.runtime_id not in excluded
                    and self._acceptable(runtime, policy, required)
                ):
                    return runtime
            if policy.strategy == "existing":
                raise self._failure(profile, policy, required, names, detected)

        if not policy.allow_managed_install:
            raise self._failure(
                profile,
                policy,
                required,
                self._existing_candidates(policy),
                detected,
            )

        system, architecture = self._platform(transport)
        platform_tag = self.platform_tag(system, architecture)
        manager = self._manager(profile, transport)
        provider = manager[0] if manager is not None else "micromamba"
        provider_version = manager[2] if manager is not None else self.artifacts.VERSION
        versions = (policy.version,) if policy.version else self.DEFAULT_MINOR_CANDIDATES
        for requested in versions:
            if requested is None or not self._minor_compatible(requested, required):
                continue
            request_id = self._request_id(
                requested, system, architecture, provider, provider_version
            )
            cached = self._request_cache(profile, transport, request_id)
            if cached is not None and cached.runtime_id in excluded:
                continue
            if (
                cached is not None
                and cached.runtime_id not in excluded
                and (policy.strategy != "managed" or cached.managed)
            ):
                probed = self._probe(profile, transport, cached.executable)
                self._record_probe(detected, cached.executable, probed)
                if probed is not None and self._acceptable(probed, policy, required):
                    return self._with_action(probed, cached, "reuse")
            planned_path = str(
                PurePosixPath(self._runtime_root(profile)) / request_id / "bin" / "python"
            )
            planned = PythonRuntime(
                request_id,
                planned_path,
                requested,
                "CPython",
                system,
                architecture,
                provider,
                provider_version,
                True,
                False,
                "install",
            )
            if request_id in excluded:
                continue
            if dry_run:
                return planned
            if manager is None:
                manager = self._install_manager(profile, transport, platform_tag)
                provider, _, provider_version = manager
            return self._create(
                profile,
                transport,
                manager,
                requested,
                system,
                architecture,
                platform_tag,
                request_id,
            )
        raise self._failure(
            profile,
            policy,
            required,
            self._existing_candidates(policy),
            detected,
        )

    def active(self, profile: ClusterProfile, transport: Transport) -> PythonRuntime | None:
        """Read the last successfully activated runtime without provisioning anything."""
        assert profile.storage is not None
        pointer = PurePosixPath(profile.storage.state_root) / self.ACTIVE_POINTER
        return self._read_runtime(transport, pointer)

    def activate(
        self, profile: ClusterProfile, transport: Transport, runtime: PythonRuntime
    ) -> None:
        """Publish a runtime only after its package environment has also verified successfully."""
        assert profile.storage is not None
        pointer = PurePosixPath(profile.storage.state_root) / self.ACTIVE_POINTER
        self._write_json(transport, runtime.executable, pointer, runtime.to_dict())

    @staticmethod
    def platform_tag(system: str, architecture: str) -> str:
        """Map a bounded Linux platform probe to conda/micromamba subdir names."""
        if system.lower() != "linux":
            raise RuntimeError("Automatic managed Python currently supports Linux clusters only.")
        normalized = architecture.lower()
        suffix = {
            "x86_64": "64",
            "amd64": "64",
            "aarch64": "aarch64",
            "arm64": "aarch64",
            "ppc64le": "ppc64le",
        }.get(normalized)
        if suffix is None:
            raise RuntimeError(f"Unsupported managed Python architecture: {architecture!r}.")
        return f"linux-{suffix}"

    def _create(
        self,
        profile: ClusterProfile,
        transport: Transport,
        manager: tuple[str, str, str],
        requested: str,
        system: str,
        architecture: str,
        platform_tag: str,
        request_id: str,
    ) -> PythonRuntime:
        assert profile.storage is not None
        provider, executable, provider_version = manager
        cache_root = PurePosixPath(profile.storage.cache_root)
        runtime_root = PurePosixPath(self._runtime_root(profile))
        lock = cache_root / f".python-runtime-{request_id}.lock"
        acquired = self._acquire(transport, lock, runtime_root / request_id / self.MARKER)
        if not acquired:
            cached = self._request_cache(profile, transport, request_id)
            if cached is not None:
                return cached
            raise RuntimeError(f"Timed out waiting for managed Python runtime lock {lock}.")
        temporary = runtime_root / f".{request_id}.tmp-{uuid4().hex}"
        try:
            cached = self._request_cache(profile, transport, request_id)
            if (
                cached is not None
                and self._probe(profile, transport, cached.executable) is not None
            ):
                return self._with_action(cached, cached, "reuse")
            created_root = transport.run(("mkdir", "-p", str(runtime_root)))
            if created_root.returncode:
                raise RuntimeError(
                    f"Could not create managed runtime root: {created_root.stderr.strip()}"
                )
            packages = str(cache_root / "conda-pkgs")
            prefix = (
                *profile.command_prefix,
                "env",
                f"MAMBA_ROOT_PREFIX={cache_root / 'micromamba-root'}",
                f"CONDA_PKGS_DIRS={packages}",
                "CONDARC=/dev/null",
                "MAMBARC=/dev/null",
                executable,
                "create",
                "--yes",
                "--prefix",
                str(temporary),
                "--override-channels",
                "--channel",
                "conda-forge",
            )
            offline_packages: Path | None = None
            if profile.wheelhouse is not None:
                offline_packages = self.artifacts.offline_packages(platform_tag, requested)
                remote_packages = cache_root / "runtime-packages" / request_id
                exists = transport.run(("test", "-d", str(remote_packages)))
                if exists.returncode != 0:
                    parent = transport.run(("mkdir", "-p", str(remote_packages.parent)))
                    if parent.returncode:
                        raise RuntimeError(
                            f"Could not create offline runtime cache: {parent.stderr.strip()}"
                        )
                    transport.put(offline_packages, str(remote_packages))
                prefix = (
                    *profile.command_prefix,
                    "env",
                    f"MAMBA_ROOT_PREFIX={cache_root / 'micromamba-root'}",
                    f"CONDA_PKGS_DIRS={remote_packages}",
                    "CONDARC=/dev/null",
                    "MAMBARC=/dev/null",
                    executable,
                    "create",
                    "--yes",
                    "--offline",
                    "--prefix",
                    str(temporary),
                )
            command = (*prefix, f"python={requested}", "pip")
            created = transport.run(command)
            if created.returncode:
                self._cleanup(transport, temporary)
                raise RuntimeError(
                    f"Could not create managed Python {requested} with {provider}: "
                    f"{created.stderr.strip()}"
                )
            temporary_python = str(temporary / "bin" / "python")
            probed = self._probe(profile, transport, temporary_python)
            if probed is None:
                self._cleanup(transport, temporary)
                raise RuntimeError("Managed Python verification failed before publication.")
            packages_result = transport.run(
                (*profile.command_prefix, executable, "list", "--json", "--prefix", str(temporary))
            )
            if packages_result.returncode:
                self._cleanup(transport, temporary)
                raise RuntimeError(
                    f"Could not inventory the managed Python runtime: "
                    f"{packages_result.stderr.strip()}"
                )
            try:
                package_inventory = json.loads(packages_result.stdout)
            except json.JSONDecodeError as error:
                self._cleanup(transport, temporary)
                raise RuntimeError(
                    "Managed runtime package inventory is not valid JSON."
                ) from error
            if not isinstance(package_inventory, list) or not all(
                isinstance(item, Mapping) for item in package_inventory
            ):
                self._cleanup(transport, temporary)
                raise RuntimeError("Managed runtime package inventory has an invalid shape.")
            package_inventory.sort(
                key=lambda item: (
                    str(item.get("name", "")),
                    str(item.get("version", "")),
                    str(item.get("build_string", item.get("build", ""))),
                    json.dumps(item, sort_keys=True, separators=(",", ":")),
                )
            )
            normalized_inventory = json.dumps(
                package_inventory, sort_keys=True, separators=(",", ":")
            )
            package_fingerprint = hashlib.sha256(normalized_inventory.encode()).hexdigest()
            actual_id = self._runtime_id(
                probed.version,
                probed.implementation,
                system,
                architecture,
                provider,
                provider_version,
                package_fingerprint,
            )
            destination = runtime_root / actual_id
            runtime = PythonRuntime(
                actual_id,
                str(destination / "bin" / "python"),
                probed.version,
                probed.implementation,
                system,
                architecture,
                provider,
                provider_version,
                True,
                True,
                "install",
                package_fingerprint,
            )
            marker = temporary / self.MARKER
            self._write_json(transport, temporary_python, marker, runtime.to_dict())
            published = transport.run(("mv", "-T", str(temporary), str(destination)))
            if published.returncode:
                self._cleanup(transport, temporary)
                existing = self._read_runtime(transport, destination / self.MARKER)
                if existing is None or self._probe(profile, transport, existing.executable) is None:
                    raise RuntimeError(
                        f"Could not atomically publish managed runtime: {published.stderr.strip()}"
                    )
                runtime = self._with_action(existing, existing, "reuse")
            self._publish_request(profile, transport, request_id, runtime)
            return runtime
        finally:
            transport.run(("rmdir", str(lock)))

    def _install_manager(
        self, profile: ClusterProfile, transport: Transport, platform_tag: str
    ) -> tuple[str, str, str]:
        assert profile.storage is not None
        local, expected = self.artifacts.artifact(platform_tag)
        root = PurePosixPath(profile.storage.cache_root) / "runtime-managers"
        destination = root / f"micromamba-{self.artifacts.VERSION}-{platform_tag}"
        exists = transport.run(("test", "-x", str(destination)))
        if exists.returncode != 0:
            created = transport.run(("mkdir", "-p", str(root)))
            if created.returncode:
                raise RuntimeError(f"Could not create runtime-manager cache: {created.stderr}")
            temporary = root / f".{destination.name}.{uuid4().hex}.tmp"
            try:
                transport.put(local, str(temporary))
                checksum = transport.run(("sha256sum", str(temporary)))
                observed = checksum.stdout.strip().split(maxsplit=1)[0]
                if checksum.returncode or observed != expected:
                    raise RuntimeError("Staged micromamba checksum verification failed.")
                changed = transport.run(("chmod", "700", str(temporary)))
                if changed.returncode:
                    raise RuntimeError(f"Could not make micromamba executable: {changed.stderr}")
                published = transport.run(("mv", "-T", str(temporary), str(destination)))
                if (
                    published.returncode
                    and transport.run(("test", "-x", str(destination))).returncode
                ):
                    raise RuntimeError(
                        f"Could not publish managed micromamba: {published.stderr.strip()}"
                    )
            finally:
                transport.run(("rm", "-f", str(temporary)))
        version = transport.run((*profile.command_prefix, str(destination), "--version"))
        expected_version = self.artifacts.VERSION.split("-")[0]
        if version.returncode or not version.stdout.strip().startswith(expected_version):
            raise RuntimeError("Managed micromamba verification failed after publication.")
        return "micromamba", str(destination), self.artifacts.VERSION

    def _manager(
        self, profile: ClusterProfile, transport: Transport
    ) -> tuple[str, str, str] | None:
        for name in ("micromamba", "mamba", "conda"):
            located = transport.run((*profile.command_prefix, "which", name))
            executable = located.stdout.strip().splitlines()[0] if located.returncode == 0 else ""
            if not executable:
                continue
            version = transport.run((*profile.command_prefix, executable, "--version"))
            if version.returncode == 0 and version.stdout.strip():
                return name, executable, version.stdout.strip().splitlines()[0]
        return None

    def _probe(
        self, profile: ClusterProfile, transport: Transport, executable: str
    ) -> PythonRuntime | None:
        code = (
            "import importlib.util,json,platform,ssl,sys;"
            "print(json.dumps({'version':platform.python_version(),"
            "'implementation':platform.python_implementation(),'system':platform.system(),"
            "'architecture':platform.machine(),'executable':sys.executable,"
            "'pip':importlib.util.find_spec('pip') is not None,"
            "'venv':importlib.util.find_spec('venv') is not None}))"
        )
        result = transport.run((*profile.command_prefix, executable, "-c", code))
        if result.returncode:
            return None
        try:
            value = json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return None
        if not value.get("pip") or not value.get("venv"):
            return None
        runtime_id = self._runtime_id(
            str(value["version"]),
            str(value["implementation"]),
            str(value["system"]),
            str(value["architecture"]),
            "existing",
            None,
            None,
        )
        return PythonRuntime(
            runtime_id,
            str(value["executable"]),
            str(value["version"]),
            str(value["implementation"]),
            str(value["system"]),
            str(value["architecture"]),
            "existing",
            None,
            False,
            True,
            "reuse",
        )

    @staticmethod
    def _existing_candidates(policy: PythonRuntimePolicy) -> tuple[str, ...]:
        if policy.strategy == "existing":
            return (policy.executable,)
        values = [policy.executable]
        for minor in ("3.14", "3.13", "3.12", "3.11", "3.10"):
            candidate = f"python{minor}"
            if candidate not in values:
                values.append(candidate)
        if "python3" not in values:
            values.append("python3")
        return tuple(values)

    @staticmethod
    def _acceptable(
        runtime: PythonRuntime,
        policy: PythonRuntimePolicy,
        requirements: Sequence[str],
    ) -> bool:
        if policy.version is not None and not SpecifierSet(
            f"=={policy.version}.*" if policy.version.count(".") == 1 else f"=={policy.version}"
        ).contains(Version(runtime.version)):
            return False
        return PythonRuntimeRequirements.compatible(runtime.version, requirements)

    @staticmethod
    def _minor_compatible(version: str, requirements: Sequence[str]) -> bool:
        candidate = version if version.count(".") == 2 else f"{version}.0"
        return PythonRuntimeRequirements.compatible(candidate, requirements)

    @staticmethod
    def _platform(transport: Transport) -> tuple[str, str]:
        system = transport.run(("uname", "-s"))
        architecture = transport.run(("uname", "-m"))
        if system.returncode or architecture.returncode:
            raise RuntimeError("Could not inspect remote platform for managed Python.")
        return system.stdout.strip(), architecture.stdout.strip()

    @staticmethod
    def _runtime_id(
        version: str,
        implementation: str,
        system: str,
        architecture: str,
        provider: str,
        provider_version: str | None,
        package_fingerprint: str | None,
    ) -> str:
        payload = {
            "identity_version": 1,
            "version": version,
            "implementation": implementation,
            "system": system,
            "architecture": architecture,
            "provider": provider,
            "provider_version": provider_version,
            "package_fingerprint": package_fingerprint,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"python-runtime-{digest[:24]}"

    @classmethod
    def _request_id(
        cls,
        version: str,
        system: str,
        architecture: str,
        provider: str,
        provider_version: str,
    ) -> str:
        return cls._runtime_id(
            version, "CPython", system, architecture, provider, provider_version, None
        )

    @staticmethod
    def _runtime_root(profile: ClusterProfile) -> str:
        assert profile.storage is not None
        return str(PurePosixPath(profile.storage.cache_root) / "runtimes")

    def _request_cache(
        self, profile: ClusterProfile, transport: Transport, request_id: str
    ) -> PythonRuntime | None:
        assert profile.storage is not None
        pointer = (
            PurePosixPath(profile.storage.state_root) / "python-runtimes" / f"{request_id}.json"
        )
        return self._read_runtime(transport, pointer)

    def _publish_request(
        self,
        profile: ClusterProfile,
        transport: Transport,
        request_id: str,
        runtime: PythonRuntime,
    ) -> None:
        assert profile.storage is not None
        pointer = (
            PurePosixPath(profile.storage.state_root) / "python-runtimes" / f"{request_id}.json"
        )
        self._write_json(transport, runtime.executable, pointer, runtime.to_dict())

    @staticmethod
    def _read_runtime(transport: Transport, path: PurePosixPath) -> PythonRuntime | None:
        result = transport.run(("cat", str(path)))
        if result.returncode or not result.stdout.strip():
            return None
        try:
            value = json.loads(result.stdout)
            return PythonRuntime.from_mapping(value) if isinstance(value, Mapping) else None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _write_json(
        transport: Transport,
        python: str,
        path: PurePosixPath,
        payload: Mapping[str, Any],
    ) -> None:
        created = transport.run(("mkdir", "-p", str(path.parent)))
        if created.returncode:
            raise RuntimeError(f"Could not create runtime state directory: {created.stderr}")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        written = transport.run(
            (
                python,
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
                str(temporary),
                json.dumps(payload, sort_keys=True) + "\n",
            )
        )
        if written.returncode:
            raise RuntimeError(f"Could not write runtime state: {written.stderr.strip()}")
        published = transport.run(("mv", "-T", str(temporary), str(path)))
        if published.returncode:
            transport.run(("rm", "-f", str(temporary)))
            raise RuntimeError(f"Could not publish runtime state: {published.stderr.strip()}")

    def _acquire(
        self, transport: Transport, lock: PurePosixPath, completion: PurePosixPath
    ) -> bool:
        deadline = time.monotonic() + self.lock_timeout
        while True:
            result = transport.run(("mkdir", str(lock)))
            if result.returncode == 0:
                return True
            if transport.run(("test", "-f", str(completion))).returncode == 0:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.2)

    @staticmethod
    def _cleanup(transport: Transport, path: PurePosixPath) -> None:
        if ".tmp-" not in path.name:
            raise ValueError(f"Refusing to clean unexpected runtime path {path}.")
        transport.run(("rm", "-rf", str(path)))

    @classmethod
    def _with_action(
        cls, probed: PythonRuntime, source: PythonRuntime, action: str
    ) -> PythonRuntime:
        return PythonRuntime(
            source.runtime_id,
            source.executable,
            probed.version,
            probed.implementation,
            probed.system,
            probed.architecture,
            source.provider,
            source.provider_version,
            source.managed,
            True,
            action,
            source.package_fingerprint,
        )

    @staticmethod
    def _failure(
        profile: ClusterProfile,
        policy: PythonRuntimePolicy,
        requirements: Sequence[str],
        candidates: Sequence[str],
        detected: Sequence[str] = (),
    ) -> NoCompatiblePythonRuntimeError:
        managed = "enabled" if policy.allow_managed_install else "disabled"
        return NoCompatiblePythonRuntimeError(
            "No compatible Python runtime could be prepared. "
            f"Probed runtimes: {tuple(detected) or tuple(candidates)}. "
            f"Required: {tuple(requirements)}. "
            f"Managed runtime installation is {managed}. Set clusters.{profile.name}.python."
            "strategy=auto, allow managed installation, or configure python.executable explicitly.",
            cluster=profile.name,
            strategy=policy.strategy,
            requirements=requirements,
            candidates=candidates,
            detected=detected,
        )

    @staticmethod
    def _record_probe(detected: list[str], executable: str, runtime: PythonRuntime | None) -> None:
        """Record bounded non-secret evidence for an eventual resolution diagnostic."""
        evidence = (
            f"{executable} -> Python {runtime.version}"
            if runtime is not None
            else f"{executable} -> unavailable"
        )
        if evidence not in detected:
            detected.append(evidence)
