"""Declarative project-owned native dependency planning for managed environments."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar
from uuid import uuid4

import tomli
import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.python_runtime import PythonRuntime
from lambdaforge.controlplane.PythonRuntimeResolver import PythonRuntimeResolver
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.ImmutableJson import FrozenJsonMapping
from lambdaforge.work.models import atomic_json

_EXECUTABLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
_PACKAGE_NAME = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*)")
_EXACT_PACKAGE_COMPONENT = re.compile(r"[A-Za-z0-9_.!+*<>=~-]+")
_SHA256_FRAGMENT = re.compile(r"(?:sha256=)?([0-9a-fA-F]{64})$")
_TORCH_PACKAGES = frozenset(
    {"torch", "pytorch", "pytorch-cuda", "torchvision", "torchaudio", "cuda", "cudatoolkit"}
)


class NativeEnvironmentError(RuntimeError):
    """Report a project-native environment preparation or verification failure."""


def _safe_exact_package(value: str) -> bool:
    """Accept one channel-qualified exact MatchSpec without shell-like syntax."""
    channel, separator, package = value.rpartition("::")
    if not separator or not channel:
        return False
    parsed = urllib.parse.urlsplit(channel)
    if parsed.scheme:
        if parsed.scheme != "https" or not parsed.netloc:
            return False
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return False
    elif re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", channel) is None:
        return False
    components = package.split("=")
    return (
        len(components) == 3
        and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", components[0]) is not None
        and _EXACT_PACKAGE_COMPONENT.fullmatch(components[1]) is not None
        and _EXACT_PACKAGE_COMPONENT.fullmatch(components[2]) is not None
    )


@dataclass(frozen=True, slots=True)
class NativeEnvironmentSpecification:
    """Validated native package declaration owned by one consumer project."""

    project_root: Path
    manager: str
    source: Path
    kind: str
    sha256: str
    channels: tuple[str, ...]
    dependencies: tuple[str, ...]
    required_executables: tuple[str, ...]
    python_requirement: str | None
    lock_platform: str | None = None
    lock_entries: tuple[str, ...] = ()
    package_cache: Path | None = None
    package_cache_id: str | None = None
    MAXIMUM_SPECIFICATION_BYTES: ClassVar[int] = 1024 * 1024

    @classmethod
    def discover(cls, project_root: str | Path | None) -> NativeEnvironmentSpecification | None:
        """Read ``tool.lambdaforge.environment`` without executing project code."""
        if project_root is None:
            return None
        root = Path(project_root).expanduser().resolve()
        pyproject = root / "pyproject.toml"
        if not pyproject.is_file() or pyproject.is_symlink():
            return None
        with pyproject.open("rb") as stream:
            document = tomli.load(stream)
        tool = document.get("tool", {})
        lambdaforge = tool.get("lambdaforge", {}) if isinstance(tool, Mapping) else {}
        raw = lambdaforge.get("environment") if isinstance(lambdaforge, Mapping) else None
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise TypeError("tool.lambdaforge.environment must be a TOML table.")
        allowed = {
            "manager",
            "file",
            "lockfile",
            "required_executables",
            "required-executables",
            "package_cache",
            "package-cache",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown native environment options: {unknown}.")
        manager = str(raw.get("manager", "conda")).strip().lower()
        if manager != "conda":
            raise ValueError("Native environment manager must currently be 'conda'.")
        file_value = raw.get("file")
        lock_value = raw.get("lockfile")
        if (file_value is None) == (lock_value is None):
            raise ValueError("Native environments require exactly one of 'file' or 'lockfile'.")
        source = cls._project_path(root, file_value if file_value is not None else lock_value)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(
                f"Native environment specification is missing or unsafe: {source}"
            )
        if source.stat().st_size > cls.MAXIMUM_SPECIFICATION_BYTES:
            raise ValueError("Native environment specifications cannot exceed 1 MiB.")
        executable_values = raw.get("required_executables", raw.get("required-executables", ()))
        if not isinstance(executable_values, list | tuple):
            raise TypeError("required_executables must be an array of executable names.")
        executables = tuple(str(value).strip() for value in executable_values)
        if any(_EXECUTABLE.fullmatch(value) is None for value in executables):
            raise ValueError(
                "required_executables accepts bare executable names, not paths or commands."
            )
        if len(executables) != len(set(executables)):
            raise ValueError("required_executables cannot contain duplicates.")
        cache_value = raw.get("package_cache", raw.get("package-cache"))
        package_cache = cls._project_path(root, cache_value) if cache_value is not None else None
        if package_cache is not None and (not package_cache.is_dir() or package_cache.is_symlink()):
            raise FileNotFoundError(f"Native package_cache is missing or unsafe: {package_cache}")
        digest = f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}"
        if lock_value is not None:
            if package_cache is None:
                raise ValueError(
                    "An explicit native lockfile requires package_cache for offline provisioning."
                )
            platform, entries, dependencies, requirement = cls._parse_lock(source)
            cls._validate_lock_cache(entries, package_cache)
            cache_id = cls._fingerprint_directory(package_cache)
            return cls(
                root,
                manager,
                source,
                "explicit-lock",
                digest,
                (),
                dependencies,
                executables,
                requirement,
                platform,
                entries,
                package_cache,
                cache_id,
            )
        channels, dependencies, requirement = cls._parse_environment(source)
        if package_cache is not None:
            raise ValueError(
                "Offline native provisioning requires an explicit lockfile; package_cache cannot "
                "be combined with a solver-driven environment file."
            )
        return cls(
            root,
            manager,
            source,
            "environment-file",
            digest,
            channels,
            dependencies,
            executables,
            requirement,
        )

    def planning_summary(self, *, platform: str | None = None) -> dict[str, Any]:
        """Return a read-only explanation suitable for bootstrap dry-run output."""
        return {
            "manager": self.manager,
            "kind": self.kind,
            "file": str(self.source.relative_to(self.project_root)),
            "sha256": self.sha256,
            "channels": list(self.channels),
            "requested_packages": list(self.dependencies),
            "required_executables": list(self.required_executables),
            "python_requirement": self.python_requirement,
            "platform": platform or self.lock_platform,
            "offline": self.kind == "explicit-lock",
            "package_cache": str(self.package_cache) if self.package_cache else None,
            "connectivity": (
                "not required; exact package bytes were verified in package_cache"
                if self.kind == "explicit-lock"
                else "required on first solve for the declared channels"
            ),
            "environment_action": "resolve exactly, then reuse or create by content identity",
            "status": "exact solve deferred until provisioning",
        }

    @staticmethod
    def _project_path(root: Path, value: Any) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise TypeError("Native environment paths must be non-empty strings.")
        raw = Path(value).expanduser()
        selected = (raw if raw.is_absolute() else root / raw).resolve()
        if not selected.is_relative_to(root):
            raise ValueError("Native environment files and caches must stay inside the project.")
        return selected

    @classmethod
    def _parse_environment(
        cls, source: Path
    ) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("Native environment.yml must contain a mapping.")
        unknown = sorted(set(value) - {"name", "channels", "dependencies"})
        if unknown:
            raise ValueError(
                f"Unsupported environment.yml fields {unknown}; hooks, variables and prefixes "
                "are intentionally unavailable."
            )
        raw_channels = value.get("channels", ("conda-forge",))
        raw_dependencies = value.get("dependencies", ())
        if not isinstance(raw_channels, list | tuple) or not raw_channels:
            raise TypeError("environment.yml channels must be a non-empty array.")
        if not isinstance(raw_dependencies, list | tuple) or not raw_dependencies:
            raise TypeError("environment.yml dependencies must be a non-empty array.")
        channels = tuple(cls._safe_channel(item) for item in raw_channels)
        dependencies: list[str] = []
        python_requirement: str | None = None
        for raw_dependency in raw_dependencies:
            if not isinstance(raw_dependency, str) or not raw_dependency.strip():
                raise TypeError(
                    "Native dependencies must be Conda package strings; nested pip sections and "
                    "arbitrary install commands are not supported."
                )
            dependency = raw_dependency.strip()
            if any(character in dependency for character in ("\n", "\r", "\0")):
                raise ValueError("Native package specifications must be single-line strings.")
            name_match = _PACKAGE_NAME.match(dependency)
            if name_match is None:
                raise ValueError(f"Invalid native package specification: {dependency!r}.")
            name = name_match.group(1).lower()
            if name in _TORCH_PACKAGES:
                raise ValueError(
                    f"Native package {name!r} conflicts with LambdaForge's PyTorch/CUDA plan. "
                    "Declare Torch through the cluster profile instead."
                )
            if name == "python":
                python_requirement = cls._python_requirement(dependency)
            dependencies.append(dependency)
        return channels, tuple(dependencies), python_requirement

    @staticmethod
    def _python_requirement(dependency: str) -> str:
        constraint = dependency[len("python") :].strip()
        if constraint.startswith("=") and not constraint.startswith(("==", ">=", "<=")):
            selected = constraint[1:].strip()
            if re.fullmatch(r"3\.\d+", selected):
                constraint = f"=={selected}.*"
            else:
                constraint = f"=={selected}"
        if not constraint:
            return ""
        try:
            SpecifierSet(constraint)
        except InvalidSpecifier as error:
            raise ValueError(
                f"Python dependency {dependency!r} cannot be reconciled with Requires-Python."
            ) from error
        return constraint

    @staticmethod
    def _safe_channel(value: Any) -> str:
        if (
            not isinstance(value, str)
            or not value.strip()
            or any(character in value for character in ("\n", "\r", "\0"))
        ):
            raise ValueError("Conda channels must be non-empty single-line strings.")
        selected = value.strip().rstrip("/")
        parsed = urllib.parse.urlsplit(selected)
        if parsed.scheme:
            valid_location = parsed.scheme == "https" and bool(parsed.netloc)
        else:
            valid_location = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", selected) is not None
        if (
            not valid_location
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Conda channels must be simple names or credential-free HTTPS locations."
            )
        return selected

    @classmethod
    def _parse_lock(cls, source: Path) -> tuple[str, tuple[str, ...], tuple[str, ...], str | None]:
        platform: str | None = None
        explicit = False
        entries: list[str] = []
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("# platform:"):
                platform = line.split(":", 1)[1].strip()
                continue
            if line.startswith("#"):
                continue
            if line == "@EXPLICIT":
                explicit = True
                continue
            if not explicit:
                raise ValueError("Native lockfiles must use Conda's @EXPLICIT format.")
            parsed = urllib.parse.urlsplit(line)
            if (
                parsed.scheme not in {"https", "file"}
                or parsed.username
                or parsed.password
                or parsed.query
            ):
                raise ValueError("Locked native packages require credential-free HTTPS/file URLs.")
            fragment = parsed.fragment
            if _SHA256_FRAGMENT.fullmatch(fragment) is None:
                raise ValueError("Every locked native package URL must carry an exact SHA-256.")
            entries.append(line)
        if not explicit or not entries or platform is None:
            raise ValueError(
                "Native lockfiles require '# platform: SUBDIR', @EXPLICIT and package URLs."
            )
        if platform not in {"linux-64", "linux-aarch64", "linux-ppc64le"}:
            raise ValueError(f"Unsupported native lockfile platform: {platform!r}.")
        dependencies: list[str] = []
        python_requirement: str | None = None
        for entry in entries:
            filename = Path(urllib.parse.urlsplit(entry).path).name
            stem = filename.removesuffix(".conda").removesuffix(".tar.bz2")
            match = re.fullmatch(r"(.+)-([0-9][^-]*)-([^-]+)", stem)
            if match is None:
                raise ValueError(f"Could not identify locked Conda package {filename!r}.")
            name, version, build = match.groups()
            normalized = name.lower()
            if normalized in _TORCH_PACKAGES:
                raise ValueError(
                    f"Locked native package {name!r} conflicts with LambdaForge's "
                    "PyTorch/CUDA plan."
                )
            dependencies.append(f"{name}={version}={build}")
            if normalized == "python":
                minor = ".".join(version.split(".")[:2])
                python_requirement = f"=={minor}.*"
        return platform, tuple(entries), tuple(dependencies), python_requirement

    @staticmethod
    def _fingerprint_directory(root: Path) -> str:
        digest = hashlib.sha256()
        found = False
        for item in sorted(root.rglob("*")):
            if item.is_symlink():
                raise ValueError(f"Native package_cache cannot contain symbolic links: {item}")
            if not item.is_file():
                continue
            found = True
            digest.update(item.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        if not found:
            raise ValueError("Native package_cache cannot be empty.")
        return hashlib.sha256(digest.digest()).hexdigest()[:24]

    @staticmethod
    def _validate_lock_cache(entries: Sequence[str], root: Path) -> None:
        """Require every exact locked package byte in the explicit offline cache."""
        available: dict[str, Path] = {}
        for item in root.rglob("*"):
            if item.is_symlink():
                raise ValueError(f"Native package_cache cannot contain symbolic links: {item}")
            if item.is_file():
                if item.name in available:
                    raise ValueError(
                        f"Native package_cache contains duplicate package name {item.name!r}."
                    )
                available[item.name] = item
        for entry in entries:
            parsed = urllib.parse.urlsplit(entry)
            filename = Path(parsed.path).name
            package = available.get(filename)
            if package is None:
                raise FileNotFoundError(
                    f"Locked native package {filename!r} is absent from package_cache."
                )
            expected_match = _SHA256_FRAGMENT.fullmatch(parsed.fragment)
            assert expected_match is not None
            digest = hashlib.sha256()
            with package.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            observed = digest.hexdigest()
            if observed.lower() != expected_match.group(1).lower():
                raise ValueError(f"Native package_cache checksum mismatch for {filename!r}.")


@dataclass(frozen=True, slots=True)
class NativeEnvironmentPlan:
    """Exact native package solve used as part of EnvironmentIdentity."""

    specification_sha256: str
    specification_name: str
    specification_kind: str
    platform: str
    manager_version: str
    channels: tuple[str, ...]
    requested_packages: tuple[str, ...]
    exact_packages: tuple[str, ...]
    package_inventory: tuple[Mapping[str, str], ...]
    required_executables: tuple[str, ...]
    offline: bool
    package_cache_id: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.specification_sha256) is None:
            raise ValueError("Native specification_sha256 must be an exact SHA-256 digest.")
        if Path(self.specification_name).name != self.specification_name:
            raise ValueError("Native specification_name must be a safe basename.")
        if self.platform not in {"linux-64", "linux-aarch64", "linux-ppc64le"}:
            raise ValueError(f"Unsupported native environment platform: {self.platform!r}.")
        if re.fullmatch(r"[A-Za-z0-9_.+-]+", self.manager_version) is None:
            raise ValueError("Native manager_version contains unsafe characters.")
        if any(_EXECUTABLE.fullmatch(value) is None for value in self.required_executables):
            raise ValueError("Native required_executables contains an invalid name.")
        if any(
            not value or any(character in value for character in ("\n", "\r", "\0"))
            for value in self.exact_packages
        ):
            raise ValueError("Native exact_packages contains an invalid package specification.")
        if any(not _safe_exact_package(value) for value in self.exact_packages):
            raise ValueError("Native exact_packages contains an unsafe package specification.")
        if self.offline and not self.package_cache_id:
            raise ValueError("Offline native plans require a package_cache_id.")
        if (
            self.package_cache_id is not None
            and re.fullmatch(r"[0-9a-f]{24}", self.package_cache_id) is None
        ):
            raise ValueError("Native package_cache_id must be a content digest.")
        object.__setattr__(
            self,
            "package_inventory",
            tuple(FrozenJsonMapping(value) for value in self.package_inventory),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return identity-safe native environment policy without machine paths."""
        return {
            "native_environment_version": 1,
            "manager": "micromamba",
            "manager_version": self.manager_version,
            "specification_sha256": self.specification_sha256,
            "specification_name": self.specification_name,
            "specification_kind": self.specification_kind,
            "platform": self.platform,
            "channels": list(self.channels),
            "requested_packages": list(self.requested_packages),
            "exact_packages": list(self.exact_packages),
            "package_inventory": [dict(value) for value in self.package_inventory],
            "required_executables": list(self.required_executables),
            "offline": self.offline,
            "package_cache_id": self.package_cache_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NativeEnvironmentPlan:
        """Restore and minimally validate a persisted native solve."""
        if int(value.get("native_environment_version", 0)) != 1:
            raise ValueError("Unsupported native environment plan version.")
        inventory = value.get("package_inventory", ())
        if not isinstance(inventory, list | tuple) or not all(
            isinstance(item, Mapping) for item in inventory
        ):
            raise TypeError("Native package inventory must be an array of mappings.")
        return cls(
            str(value["specification_sha256"]),
            str(value["specification_name"]),
            str(value["specification_kind"]),
            str(value["platform"]),
            str(value["manager_version"]),
            tuple(str(item) for item in value.get("channels", ())),
            tuple(str(item) for item in value.get("requested_packages", ())),
            tuple(str(item) for item in value.get("exact_packages", ())),
            tuple({str(key): str(item) for key, item in record.items()} for record in inventory),
            tuple(str(item) for item in value.get("required_executables", ())),
            bool(value.get("offline", False)),
            str(value["package_cache_id"]) if value.get("package_cache_id") else None,
        )


class NativeEnvironmentPlanner:
    """Resolve native packages once and cache the exact cross-platform solve locally."""

    def __init__(
        self,
        root: str | Path = ".lambdaforge/control/native-environments",
        runtime_resolver: PythonRuntimeResolver | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.runtime_resolver = runtime_resolver or PythonRuntimeResolver()

    def plan(
        self,
        specification: NativeEnvironmentSpecification,
        profile: ClusterProfile,
        transport: Transport,
        runtime: PythonRuntime,
    ) -> NativeEnvironmentPlan:
        """Return an exact package plan, solving only when no valid local receipt exists."""
        platform = PythonRuntimeResolver.platform_tag(runtime.system, runtime.architecture)
        if specification.lock_platform and specification.lock_platform != platform:
            raise ValueError(
                f"Native lockfile targets {specification.lock_platform}, not remote {platform}."
            )
        if specification.python_requirement and not SpecifierSet(
            specification.python_requirement
        ).contains(Version(runtime.version), prereleases=False):
            raise ValueError(
                f"Native environment requires Python {specification.python_requirement}, but "
                f"the resolved runtime is {runtime.version}."
            )
        manager = self.runtime_resolver.ensure_micromamba(profile, transport, platform)
        manager_version = manager[2]
        key = self._plan_key(specification, platform, manager_version, runtime.version)
        receipt = self.root / f"native-plan-{key}.json"
        if receipt.is_file() and not receipt.is_symlink():
            try:
                cached = NativeEnvironmentPlan.from_mapping(
                    json.loads(receipt.read_text(encoding="utf-8"))
                )
                if self._matches(cached, specification, platform, manager_version):
                    return cached
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                pass
        remote_source, remote_packages = self._stage_inputs(
            specification, profile, transport, runtime.executable
        )
        command = self._solve_command(
            specification,
            profile,
            manager[1],
            platform,
            remote_source,
            remote_packages,
            key,
            runtime,
        )
        resolved = transport.run(command, timeout=300.0)
        if resolved.returncode:
            mode = "offline lock/cache" if specification.kind == "explicit-lock" else "channels"
            raise RuntimeError(
                f"Could not resolve native packages from {mode} for {platform}: "
                f"{resolved.stderr.strip()}"
            )
        inventory = self._inventory(resolved.stdout)
        self._validate_inventory(inventory, runtime, platform)
        requested_packages = self._requested_packages(specification, runtime)
        exact = tuple(
            f"{item['channel']}::{item['name']}={item['version']}={item['build']}"
            for item in inventory
        )
        plan = NativeEnvironmentPlan(
            specification.sha256,
            specification.source.name,
            specification.kind,
            platform,
            manager_version,
            specification.channels,
            requested_packages,
            exact,
            inventory,
            specification.required_executables,
            specification.kind == "explicit-lock",
            specification.package_cache_id,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(receipt, plan.to_dict())
        return plan

    def cached_plan(
        self,
        specification: NativeEnvironmentSpecification,
        runtime: PythonRuntime,
    ) -> NativeEnvironmentPlan | None:
        """Read a previous exact solve without network, transport or filesystem mutation."""
        platform = PythonRuntimeResolver.platform_tag(runtime.system, runtime.architecture)
        key = self._plan_key(
            specification,
            platform,
            self.runtime_resolver.artifacts.VERSION,
            runtime.version,
        )
        receipt = self.root / f"native-plan-{key}.json"
        if not receipt.is_file() or receipt.is_symlink():
            return None
        try:
            cached = NativeEnvironmentPlan.from_mapping(
                json.loads(receipt.read_text(encoding="utf-8"))
            )
            return (
                cached
                if self._matches(
                    cached,
                    specification,
                    platform,
                    self.runtime_resolver.artifacts.VERSION,
                )
                else None
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    @staticmethod
    def _matches(
        plan: NativeEnvironmentPlan,
        specification: NativeEnvironmentSpecification,
        platform: str,
        manager_version: str,
    ) -> bool:
        return bool(
            plan.specification_sha256 == specification.sha256
            and plan.specification_name == specification.source.name
            and plan.platform == platform
            and plan.manager_version == manager_version
            and plan.required_executables == specification.required_executables
            and plan.package_cache_id == specification.package_cache_id
            and plan.offline == (specification.kind == "explicit-lock")
        )

    @staticmethod
    def _plan_key(
        specification: NativeEnvironmentSpecification,
        platform: str,
        manager_version: str,
        python_version: str,
    ) -> str:
        key_payload = {
            "specification": specification.sha256,
            "platform": platform,
            "manager_version": manager_version,
            "python_minor": ".".join(python_version.split(".")[:2]),
            "package_cache_id": specification.package_cache_id,
        }
        return hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

    @staticmethod
    def _solve_command(
        specification: NativeEnvironmentSpecification,
        profile: ClusterProfile,
        manager: str,
        platform: str,
        remote_source: PurePosixPath,
        remote_packages: PurePosixPath | None,
        key: str,
        runtime: PythonRuntime,
    ) -> tuple[str, ...]:
        assert profile.storage is not None
        cache_root = PurePosixPath(profile.storage.cache_root)
        command = [
            *profile.command_prefix,
            "env",
            f"MAMBA_ROOT_PREFIX={cache_root / 'micromamba-root'}",
            f"CONDA_PKGS_DIRS={remote_packages or cache_root / 'conda-pkgs'}",
            "CONDARC=/dev/null",
            "MAMBARC=/dev/null",
            *(runtime.tls_trust.assignments() if runtime.tls_trust is not None else ()),
            manager,
            "create",
            "--dry-run",
            "--json",
            "--prefix",
            str(cache_root / "native-solve" / key),
            "--platform",
            platform,
        ]
        if specification.kind == "explicit-lock":
            command.extend(("--offline", "--file", str(remote_source)))
        else:
            command.append("--override-channels")
            for channel in specification.channels:
                command.extend(("--channel", channel))
            command.extend(NativeEnvironmentPlanner._requested_packages(specification, runtime))
        return tuple(command)

    @staticmethod
    def _requested_packages(
        specification: NativeEnvironmentSpecification,
        runtime: PythonRuntime,
    ) -> tuple[str, ...]:
        """Add only the baseline packages needed to install and verify exact wheels."""
        if specification.kind == "explicit-lock":
            return specification.dependencies
        selected = list(specification.dependencies)
        names = {
            match.group(1).lower()
            for item in selected
            if (match := _PACKAGE_NAME.match(item)) is not None
        }
        required = (
            ("python", f"python={'.'.join(runtime.version.split('.')[:2])}"),
            ("pip", "pip"),
            ("ca-certificates", "ca-certificates"),
            ("openssl", "openssl"),
        )
        selected.extend(package for name, package in required if name not in names)
        return tuple(selected)

    def _stage_inputs(
        self,
        specification: NativeEnvironmentSpecification,
        profile: ClusterProfile,
        transport: Transport,
        remote_python: str,
    ) -> tuple[PurePosixPath, PurePosixPath | None]:
        assert profile.storage is not None
        cache_root = PurePosixPath(profile.storage.cache_root)
        digest = specification.sha256.removeprefix("sha256:")
        remote_spec = cache_root / "native-specifications" / digest / specification.source.name
        if transport.run(("test", "-f", str(remote_spec))).returncode:
            created = transport.run(("mkdir", "-p", str(remote_spec.parent)))
            if created.returncode:
                raise RuntimeError(f"Could not create native specification cache: {created.stderr}")
            temporary_spec = remote_spec.with_name(f".{remote_spec.name}.tmp-{uuid4().hex}")
            try:
                transport.put(specification.source, str(temporary_spec))
                checksum = transport.run(("sha256sum", str(temporary_spec)))
                observed = checksum.stdout.strip().split(maxsplit=1)[0]
                if checksum.returncode or observed != digest:
                    raise RuntimeError("Staged native specification checksum verification failed.")
                published = transport.run(("mv", "-T", str(temporary_spec), str(remote_spec)))
                if (
                    published.returncode
                    and transport.run(("test", "-f", str(remote_spec))).returncode
                ):
                    raise RuntimeError(
                        f"Could not atomically publish native specification: {published.stderr}"
                    )
            finally:
                transport.run(("rm", "-f", str(temporary_spec)))
        if specification.package_cache is None or specification.package_cache_id is None:
            return remote_spec, None
        remote_root = cache_root / "native-packages" / specification.package_cache_id
        marker = remote_root / ".lambdaforge-native-cache.json"
        if transport.run(("test", "-f", str(marker))).returncode == 0:
            return remote_spec, remote_root / "packages"
        temporary = remote_root.parent / f".{remote_root.name}.tmp-{uuid4().hex}"
        created = transport.run(("mkdir", "-p", str(temporary)))
        if created.returncode:
            raise RuntimeError(f"Could not create native package staging: {created.stderr}")
        try:
            transport.put(specification.package_cache, str(temporary / "packages"))
            payload = json.dumps(
                {
                    "native_package_cache_version": 1,
                    "package_cache_id": specification.package_cache_id,
                },
                sort_keys=True,
            )
            marked = transport.run(
                (
                    remote_python,
                    "-c",
                    "from pathlib import Path; import sys; "
                    "Path(sys.argv[1]).write_text(sys.argv[2])",
                    str(temporary / marker.name),
                    payload,
                )
            )
            if marked.returncode:
                raise RuntimeError(f"Could not mark native package cache: {marked.stderr}")
            published = transport.run(("mv", "-T", str(temporary), str(remote_root)))
            if published.returncode and transport.run(("test", "-f", str(marker))).returncode:
                raise RuntimeError(
                    f"Could not atomically publish native package cache: {published.stderr}"
                )
        finally:
            transport.run(("rm", "-rf", str(temporary)))
        return remote_spec, remote_root / "packages"

    @staticmethod
    def _inventory(output: str) -> tuple[Mapping[str, str], ...]:
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as error:
            raise RuntimeError("Native package solver did not return valid JSON.") from error
        actions = payload.get("actions", {}) if isinstance(payload, Mapping) else {}
        candidates = (
            actions.get("LINK") or actions.get("FETCH") if isinstance(actions, Mapping) else None
        )
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Native package solver returned no exact package inventory.")
        normalized: list[Mapping[str, str]] = []
        for raw in candidates:
            if not isinstance(raw, Mapping):
                raise RuntimeError("Native package solver returned an invalid inventory entry.")
            name = str(raw.get("name", "")).strip()
            version = str(raw.get("version", "")).strip()
            build = str(raw.get("build_string", raw.get("build", ""))).strip()
            channel = str(raw.get("channel", raw.get("base_url", ""))).strip()
            subdir = str(raw.get("subdir", "")).strip()
            if not name or not version or not build or not channel or not subdir:
                raise RuntimeError(
                    "Native package inventory lacks name, version, build, channel or subdir."
                )
            parsed_channel = urllib.parse.urlsplit(channel)
            if parsed_channel.username or parsed_channel.password or parsed_channel.query:
                raise ValueError("Resolved native package channels contain credentials or queries.")
            normalized.append(
                {
                    "name": name,
                    "version": version,
                    "build": build,
                    "channel": channel,
                    "subdir": subdir,
                }
            )
        return tuple(
            sorted(
                normalized,
                key=lambda item: (
                    item["name"],
                    item["version"],
                    item["build"],
                    item["channel"],
                ),
            )
        )

    @staticmethod
    def _validate_inventory(
        inventory: Sequence[Mapping[str, str]], runtime: PythonRuntime, platform: str
    ) -> None:
        wrong_platform = sorted(
            {
                item.get("subdir", "")
                for item in inventory
                if item.get("subdir", "") not in {"", "noarch", platform}
            }
        )
        if wrong_platform:
            raise ValueError(f"Native solve for {platform} returned packages for {wrong_platform}.")
        by_name = {item["name"].lower(): item for item in inventory}
        conflict = sorted(_TORCH_PACKAGES.intersection(by_name))
        if conflict:
            raise ValueError(
                f"Resolved native packages conflict with LambdaForge's PyTorch plan: {conflict}."
            )
        python = by_name.get("python")
        if python is None:
            raise ValueError("Resolved native environment contains no Python package.")
        expected = ".".join(runtime.version.split(".")[:2])
        observed = ".".join(python["version"].split(".")[:2])
        if observed != expected:
            raise ValueError(
                f"Native solve selected Python {python['version']}, but runtime/PyTorch planning "
                f"selected Python {runtime.version}."
            )
        if "pip" not in by_name:
            raise ValueError(
                "Resolved native environment contains no pip package for exact wheels."
            )


__all__ = [
    "NativeEnvironmentError",
    "NativeEnvironmentPlan",
    "NativeEnvironmentPlanner",
    "NativeEnvironmentSpecification",
]
