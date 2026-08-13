"""Resolved PyTorch installation plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TorchInstallationPlan:
    """Record the exact remote-compatible PyTorch wheel decision."""

    channel: str
    version: str | None
    index_url: str | None
    accelerator: str
    driver_version: str | None = None
    compute_capabilities: tuple[str, ...] = ()
    python_version: str | None = None
    architecture: str | None = None
    reason: str = ""
    require_cuda: bool = False

    def __post_init__(self) -> None:
        if self.accelerator not in {"cpu", "cuda"}:
            raise ValueError("Torch installation accelerator must be cpu or cuda.")
        if self.channel != "wheelhouse" and (self.version is None or self.index_url is None):
            raise ValueError("Online Torch installation plans require exact version and index URL.")
        if self.require_cuda and self.accelerator != "cuda":
            raise ValueError("A CUDA-required plan cannot select the CPU accelerator.")

    def to_dict(self) -> dict[str, Any]:
        """Return an identity-safe dependency plan."""
        return {
            "channel": self.channel,
            "version": self.version,
            "index_url": self.index_url,
            "accelerator": self.accelerator,
            "driver_version": self.driver_version,
            "compute_capabilities": list(self.compute_capabilities),
            "python_version": self.python_version,
            "architecture": self.architecture,
            "reason": self.reason,
            "require_cuda": self.require_cuda,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TorchInstallationPlan:
        """Restore a plan carried by an execution bundle."""
        return cls(
            channel=str(value["channel"]),
            version=str(value["version"]) if value.get("version") else None,
            index_url=str(value["index_url"]) if value.get("index_url") else None,
            accelerator=str(value["accelerator"]),
            driver_version=(str(value["driver_version"]) if value.get("driver_version") else None),
            compute_capabilities=tuple(str(item) for item in value.get("compute_capabilities", ())),
            python_version=(str(value["python_version"]) if value.get("python_version") else None),
            architecture=str(value["architecture"]) if value.get("architecture") else None,
            reason=str(value.get("reason", "")),
            require_cuda=bool(value.get("require_cuda", False)),
        )
