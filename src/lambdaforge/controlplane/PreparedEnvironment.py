"""Prepared execution-environment result."""

from dataclasses import dataclass
from typing import Any

from lambdaforge.controlplane.NativeLibraryPolicy import NativeLibraryPolicy


@dataclass(frozen=True, slots=True)
class PreparedEnvironment:
    """Exact interpreter, optional pre-start loader policy and current verification evidence.

    ``python`` is a path, not an activated shell. Native commands must retain
    ``library_policy`` before starting it; inherited descendants then use the same policy.
    """

    environment_id: str
    python: str
    reused: bool
    dependency_resolution_id: str | None = None
    library_policy: NativeLibraryPolicy | None = None
    library_evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable preparation result."""
        return {
            "environment_id": self.environment_id,
            "python": self.python,
            "reused": self.reused,
            "dependency_resolution_id": self.dependency_resolution_id,
            "native_library_policy": self.library_policy.to_dict() if self.library_policy else None,
            "native_libraries": self.library_evidence,
        }
