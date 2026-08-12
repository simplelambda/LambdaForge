"""Redaction-safe cluster authentication descriptor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ClusterAuthentication:
    """Select authentication without ever containing a credential value."""

    mode: str = "openssh"
    credential: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"openssh", "password"}:
            raise ValueError("Cluster authentication must be 'openssh' or 'password'.")
        if self.mode == "openssh" and self.credential is not None:
            raise ValueError("OpenSSH authentication does not use a credential reference.")
        if self.credential is not None and not self.credential.startswith(("keyring:", "env:")):
            raise ValueError("Credential references must start with 'keyring:' or 'env:'.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | str | None) -> ClusterAuthentication:
        """Parse the compact string or explicit YAML authentication form."""
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise TypeError("auth must be a string or mapping.")
        return cls(str(value.get("mode", "openssh")), value.get("credential"))

    def to_dict(self) -> dict[str, str | None]:
        """Return references only; a secret value cannot enter this object."""
        return {"mode": self.mode, "credential": self.credential}
