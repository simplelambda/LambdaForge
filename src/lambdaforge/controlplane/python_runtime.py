"""Python runtime policy, identity and standard metadata constraints."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from lambdaforge.controlplane.TlsTrust import TlsTrust


class NoCompatiblePythonRuntimeError(RuntimeError):
    """Report that no permitted Python runtime satisfies all declared constraints."""

    def __init__(
        self,
        message: str,
        *,
        cluster: str | None = None,
        strategy: str | None = None,
        requirements: Sequence[str] = (),
        candidates: Sequence[str] = (),
        detected: Sequence[str] = (),
    ) -> None:
        self.cluster = cluster
        self.strategy = strategy
        self.requirements = tuple(requirements)
        self.candidates = tuple(candidates)
        self.detected = tuple(detected)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PythonRuntimePolicy:
    """Select an existing, automatically resolved or guaranteed managed Python runtime."""

    strategy: str = "existing"
    executable: str = "python"
    version: str | None = None
    allow_managed_install: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in {"auto", "existing", "managed"}:
            raise ValueError("python.strategy must be auto, existing or managed.")
        if not self.executable.strip() or "\n" in self.executable:
            raise ValueError("python.executable must be one non-empty executable path or name.")
        if self.version is not None and re.fullmatch(r"3\.\d+(?:\.\d+)?", self.version) is None:
            raise ValueError("python.version must be a Python 3 minor or patch version.")
        if not isinstance(self.allow_managed_install, bool):
            raise TypeError("python.allow_managed_install must be boolean.")

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | str | None) -> PythonRuntimePolicy:
        """Parse the new mapping while preserving the old explicit-string contract."""
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls("existing", value)
        if not isinstance(value, Mapping):
            raise TypeError("python must be an executable string or a runtime-policy mapping.")
        allowed = value.get("allow_managed_install", True)
        if not isinstance(allowed, bool):
            raise TypeError("python.allow_managed_install must be boolean.")
        return cls(
            strategy=str(value.get("strategy", "existing")),
            executable=str(value.get("executable", "python3")),
            version=str(value["version"]) if value.get("version") is not None else None,
            allow_managed_install=allowed,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable cluster-profile representation."""
        return {
            "strategy": self.strategy,
            "executable": self.executable,
            "version": self.version,
            "allow_managed_install": self.allow_managed_install,
        }


@dataclass(frozen=True, slots=True)
class PythonRuntime:
    """Describe one probed or planned interpreter independently of its package environment."""

    runtime_id: str
    executable: str
    version: str
    implementation: str
    system: str
    architecture: str
    provider: str
    provider_version: str | None
    managed: bool
    ready: bool
    action: str
    package_fingerprint: str | None = None
    tls_trust: TlsTrust | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return identity-safe runtime evidence."""
        return {
            "runtime_id": self.runtime_id,
            "executable": self.executable,
            "version": self.version,
            "implementation": self.implementation,
            "system": self.system,
            "architecture": self.architecture,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "managed": self.managed,
            "ready": self.ready,
            "action": self.action,
            "package_fingerprint": self.package_fingerprint,
            "tls_trust": self.tls_trust.to_dict() if self.tls_trust is not None else None,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PythonRuntime:
        """Restore a runtime pointer or environment-policy record."""
        return cls(
            runtime_id=str(value["runtime_id"]),
            executable=str(value["executable"]),
            version=str(value["version"]),
            implementation=str(value.get("implementation", "CPython")),
            system=str(value["system"]),
            architecture=str(value["architecture"]),
            provider=str(value["provider"]),
            provider_version=(
                str(value["provider_version"]) if value.get("provider_version") else None
            ),
            managed=bool(value.get("managed", False)),
            ready=bool(value.get("ready", True)),
            action=str(value.get("action", "reuse")),
            package_fingerprint=(
                str(value["package_fingerprint"]) if value.get("package_fingerprint") else None
            ),
            tls_trust=TlsTrust.from_mapping(
                value.get("tls_trust") if isinstance(value.get("tls_trust"), Mapping) else None
            ),
        )


class PythonRuntimeRequirements:
    """Read and intersect standard ``Requires-Python`` declarations."""

    @staticmethod
    def framework() -> str:
        """Read the requirement carried by the installed LambdaForge release metadata."""
        metadata = distribution("lambdaforge").metadata
        requirement = metadata["Requires-Python"] if "Requires-Python" in metadata else None
        if not requirement:
            raise RuntimeError("Installed LambdaForge metadata has no Requires-Python field.")
        return str(requirement)

    @staticmethod
    def project(project_root: str | Path | None) -> str | None:
        """Read a consumer project's declaration when a local source project is available."""
        if project_root is None:
            return None
        pyproject = Path(project_root).resolve() / "pyproject.toml"
        if not pyproject.is_file():
            return None
        import tomli

        with pyproject.open("rb") as stream:
            project = tomli.load(stream).get("project", {})
        value = project.get("requires-python") if isinstance(project, Mapping) else None
        return str(value) if value else None

    @classmethod
    def normalized(cls, values: Sequence[str | None]) -> tuple[str, ...]:
        """Validate and deduplicate constraints without inventing another solver."""
        result: list[str] = []
        for value in values:
            if not value or value in result:
                continue
            try:
                SpecifierSet(value)
            except InvalidSpecifier as error:
                raise ValueError(f"Invalid Requires-Python constraint {value!r}.") from error
            result.append(value)
        return tuple(result)

    @classmethod
    def compatible(cls, version: str, requirements: Sequence[str | None]) -> bool:
        """Return whether one concrete interpreter satisfies every declared constraint."""
        candidate = Version(version)
        return all(
            SpecifierSet(requirement).contains(candidate, prereleases=False)
            for requirement in cls.normalized(requirements)
        )
