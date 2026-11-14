"""Cluster bootstrap result."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.ImmutableJson import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class ClusterBootstrapResult:
    """Describe an idempotent workspace/environment preparation."""

    cluster: str
    environment_id: str
    python: str
    reused: bool
    pytorch: Mapping[str, Any] | None = None
    runtime: Mapping[str, Any] | None = None
    native_environment: Mapping[str, Any] | None = None
    planned: bool = False
    pruned_environments: tuple[str, ...] = ()
    cleanup_blocked_reason: str | None = None
    tool_runtimes: tuple[Mapping[str, Any], ...] = ()
    native_library_policy: Mapping[str, Any] | None = None
    native_libraries: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("native_library_policy", "native_libraries"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, FrozenJsonMapping(value))
        object.__setattr__(
            self, "tool_runtimes", tuple(FrozenJsonMapping(value) for value in self.tool_runtimes)
        )
        if self.pytorch is not None:
            object.__setattr__(self, "pytorch", FrozenJsonMapping(self.pytorch))
        if self.runtime is not None:
            object.__setattr__(self, "runtime", FrozenJsonMapping(self.runtime))
        if self.native_environment is not None:
            object.__setattr__(
                self,
                "native_environment",
                FrozenJsonMapping(self.native_environment),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable result."""
        return {
            "cluster": self.cluster,
            "environment_id": self.environment_id,
            "python": self.python,
            "reused": self.reused,
            "pytorch": copy.deepcopy(self.pytorch),
            "runtime": copy.deepcopy(self.runtime),
            "native_environment": copy.deepcopy(self.native_environment),
            "native_library_policy": copy.deepcopy(self.native_library_policy),
            "native_libraries": copy.deepcopy(self.native_libraries),
            "planned": self.planned,
            "pruned_environments": list(self.pruned_environments),
            "cleanup_blocked_reason": self.cleanup_blocked_reason,
            "tool_runtimes": [dict(value) for value in self.tool_runtimes],
        }
