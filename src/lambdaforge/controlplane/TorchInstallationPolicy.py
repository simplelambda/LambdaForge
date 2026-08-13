"""User-facing managed PyTorch installation policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TorchInstallationPolicy:
    """Select automatic or explicit official PyTorch wheel channels."""

    channel: str = "auto"
    require_cuda: bool | None = None

    CHANNELS = {"auto", "cpu", "cu118", "cu121", "cu124", "cu126", "cu128", "cu130"}

    def __post_init__(self) -> None:
        if self.channel not in self.CHANNELS:
            raise ValueError(f"pytorch.channel must be one of {tuple(sorted(self.CHANNELS))}.")
        if self.require_cuda is not None and not isinstance(self.require_cuda, bool):
            raise TypeError("pytorch.require_cuda must be true, false or null/auto.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | str | None) -> TorchInstallationPolicy:
        """Parse compact channel or explicit policy YAML."""
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise TypeError("pytorch must be a channel string or mapping.")
        required = value.get("require_cuda")
        if required == "auto":
            required = None
        return cls(str(value.get("channel", "auto")), required)

    def to_dict(self) -> dict[str, str | bool]:
        """Return the durable non-secret policy descriptor."""
        return {
            "channel": self.channel,
            "require_cuda": "auto" if self.require_cuda is None else self.require_cuda,
        }
