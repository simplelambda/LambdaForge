"""Portable policy for obtaining GPU access on heterogeneous clusters."""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuAccessPolicy:
    """Describe who allocates GPUs and whether existing use is admissible."""

    mode: str = "auto"
    command_prefix: tuple[str, ...] = ()
    claim_command: tuple[str, ...] = ()
    release_command: tuple[str, ...] = ()
    visibility_command: tuple[str, ...] = ()

    MODES = frozenset({"auto", "exclusive", "shared", "command", "scheduler"})

    def __post_init__(self) -> None:
        if self.mode not in self.MODES:
            raise ValueError(f"GPU access mode must be one of {sorted(self.MODES)}.")
        arguments = (
            *self.command_prefix,
            *self.claim_command,
            *self.release_command,
            *self.visibility_command,
        )
        if any(not value or "\n" in value or "\x00" in value for value in arguments):
            raise ValueError("GPU command arguments must be non-empty and contain no NUL/newline.")
        if self.mode == "command" and not self.command_prefix:
            raise ValueError("GPU access mode 'command' requires command_prefix.")
        if self.mode != "command" and (
            self.command_prefix
            or self.claim_command
            or self.release_command
            or self.visibility_command
        ):
            raise ValueError(
                "GPU command_prefix/claim_command/release_command are valid only when "
                "mode='command'."
            )
        if bool(self.claim_command) != bool(self.release_command):
            raise ValueError(
                "gpu_access.claim_command and release_command must be configured together so "
                "reservations cannot leak."
            )
        formatter = string.Formatter()
        for argument in self.claim_command:
            for _, field, format_spec, conversion in formatter.parse(argument):
                if field is None:
                    continue
                if field != "gpu_count" or format_spec or conversion:
                    raise ValueError(
                        "gpu_access.claim_command accepts only the {gpu_count} placeholder."
                    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | str | None) -> GpuAccessPolicy:
        """Normalize the compact string or explicit mapping form."""
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise TypeError("gpu_access must be a mode string or mapping.")
        unknown = set(value) - {
            "mode",
            "command_prefix",
            "claim_command",
            "release_command",
            "visibility_command",
        }
        if unknown:
            raise ValueError(f"Unknown gpu_access field(s): {sorted(unknown)}.")
        raw = value.get("command_prefix", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise TypeError("gpu_access.command_prefix must be an argv list.")
        raw_claim = value.get("claim_command", ())
        if not isinstance(raw_claim, Sequence) or isinstance(raw_claim, (str, bytes, bytearray)):
            raise TypeError("gpu_access.claim_command must be an argv list.")
        raw_release = value.get("release_command", ())
        if not isinstance(raw_release, Sequence) or isinstance(
            raw_release, (str, bytes, bytearray)
        ):
            raise TypeError("gpu_access.release_command must be an argv list.")
        raw_visibility = value.get("visibility_command", ())
        if not isinstance(raw_visibility, Sequence) or isinstance(
            raw_visibility, (str, bytes, bytearray)
        ):
            raise TypeError("gpu_access.visibility_command must be an argv list.")
        return cls(
            str(value.get("mode", "auto")),
            tuple(str(item) for item in raw),
            tuple(str(item) for item in raw_claim),
            tuple(str(item) for item in raw_release),
            tuple(str(item) for item in raw_visibility),
        )

    def effective_mode(self, scheduler: str) -> str:
        """Resolve ``auto`` without probing or mutating the target."""
        if self.mode == "auto":
            return "scheduler" if scheduler == "slurm" else "exclusive"
        return self.mode

    def wrap(self, command: Sequence[str]) -> tuple[str, ...]:
        """Wrap GPU-sensitive argv without a shell when the site requires it."""
        prefix = self.command_prefix if self.mode == "command" else ()
        return (*prefix, *(str(item) for item in command))

    def claim(self, gpu_count: int) -> tuple[str, ...]:
        """Render the explicitly configured per-submission claim command."""
        if gpu_count < 1 or not self.claim_command:
            return ()
        values = {"gpu_count": str(gpu_count)}
        return tuple(argument.format_map(values) for argument in self.claim_command)

    def release(self) -> tuple[str, ...]:
        """Return cleanup argv paired with an explicit claim."""
        return self.release_command

    def visibility_probe(self) -> tuple[str, ...]:
        """Return an argv command that reports the allocation's current opaque tokens.

        The common ``gpu exec`` launcher contract has a sibling ``gpu env`` command.  Profiles
        can state a different command explicitly; LambdaForge never falls back to nvidia-smi,
        because physical visibility is not proof of ownership.
        """
        if self.visibility_command:
            return self.visibility_command
        if self.mode == "command" and len(self.command_prefix) >= 2:
            if self.command_prefix[-1] == "exec":
                return (*self.command_prefix[:-1], "env")
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "command_prefix": list(self.command_prefix),
            "claim_command": list(self.claim_command),
            "release_command": list(self.release_command),
            "visibility_command": list(self.visibility_command),
        }


__all__ = ["GpuAccessPolicy"]
