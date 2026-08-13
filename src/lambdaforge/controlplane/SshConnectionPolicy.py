"""Independent SSH connection and command-channel policy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SshConnectionPolicy:
    """Separate connection/authentication deadlines and idle connection reuse."""

    connect_timeout: float = 15.0
    auth_timeout: float = 15.0
    banner_timeout: float = 15.0
    keepalive: float = 30.0
    multiplex: bool = True
    persist: float = 60.0
    command_timeout: float | None = None

    def __post_init__(self) -> None:
        for name in ("connect_timeout", "auth_timeout", "banner_timeout"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"connection.{name} must be positive.")
        if self.keepalive < 0 or self.persist < 0:
            raise ValueError("connection keepalive/persist cannot be negative.")
        if self.command_timeout is not None and self.command_timeout <= 0:
            raise ValueError("connection.command_timeout must be positive or null.")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        legacy_timeout: float = 15.0,
    ) -> SshConnectionPolicy:
        """Parse duration values while treating legacy ssh_timeout as connect-only."""
        source = dict(value or {})
        return cls(
            connect_timeout=cls._seconds(source.get("connect_timeout", legacy_timeout)),
            auth_timeout=cls._seconds(source.get("auth_timeout", legacy_timeout)),
            banner_timeout=cls._seconds(source.get("banner_timeout", legacy_timeout)),
            keepalive=cls._seconds(source.get("keepalive", 30.0), allow_zero=True),
            multiplex=bool(source.get("multiplex", True)),
            persist=cls._seconds(source.get("persist", 60.0), allow_zero=True),
            command_timeout=(
                cls._seconds(source["command_timeout"])
                if source.get("command_timeout") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, float | bool | None]:
        """Return a portable, non-secret policy."""
        return {
            "connect_timeout": self.connect_timeout,
            "auth_timeout": self.auth_timeout,
            "banner_timeout": self.banner_timeout,
            "keepalive": self.keepalive,
            "multiplex": self.multiplex,
            "persist": self.persist,
            "command_timeout": self.command_timeout,
        }

    @staticmethod
    def _seconds(value: object, *, allow_zero: bool = False) -> float:
        if isinstance(value, bool):
            raise TypeError("SSH durations cannot be boolean.")
        if isinstance(value, (int, float)):
            seconds = float(value)
        else:
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*", str(value))
            if match is None:
                raise ValueError(f"Invalid SSH duration: {value!r}.")
            scale = {None: 1.0, "ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
            seconds = float(match.group(1)) * scale[match.group(2)]
        if seconds < 0 or (seconds == 0 and not allow_zero):
            raise ValueError("SSH durations must be positive.")
        return seconds
