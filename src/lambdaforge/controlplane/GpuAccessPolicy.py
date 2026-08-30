"""Portable policy for obtaining GPU access on heterogeneous clusters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuAccessPolicy:
    """Describe who allocates GPUs and whether existing use is admissible."""

    mode: str = "auto"
    command_prefix: tuple[str, ...] = ()

    MODES = frozenset({"auto", "exclusive", "shared", "command", "scheduler"})

    def __post_init__(self) -> None:
        if self.mode not in self.MODES:
            raise ValueError(f"GPU access mode must be one of {sorted(self.MODES)}.")
        if any(not value or "\n" in value or "\x00" in value for value in self.command_prefix):
            raise ValueError(
                "GPU command-prefix arguments must be non-empty and contain no NUL/newline."
            )
        if self.mode == "command" and not self.command_prefix:
            raise ValueError("GPU access mode 'command' requires command_prefix.")
        if self.mode != "command" and self.command_prefix:
            raise ValueError("GPU command_prefix is valid only when mode='command'.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | str | None) -> GpuAccessPolicy:
        """Normalize the compact string or explicit mapping form."""
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise TypeError("gpu_access must be a mode string or mapping.")
        unknown = set(value) - {"mode", "command_prefix"}
        if unknown:
            raise ValueError(f"Unknown gpu_access field(s): {sorted(unknown)}.")
        raw = value.get("command_prefix", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise TypeError("gpu_access.command_prefix must be an argv list.")
        return cls(str(value.get("mode", "auto")), tuple(str(item) for item in raw))

    def effective_mode(self, scheduler: str) -> str:
        """Resolve ``auto`` without probing or mutating the target."""
        if self.mode == "auto":
            return "scheduler" if scheduler == "slurm" else "exclusive"
        return self.mode

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "command_prefix": list(self.command_prefix)}


__all__ = ["GpuAccessPolicy"]
