"""Validated named execution-cluster profile."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class ClusterProfile:
    """Describe access, scheduler, workspace and environment policy."""

    name: str
    transport: str = "local"
    scheduler: str = "local"
    host: str | None = None
    workspace: str = ".lambdaforge/remote"
    python: str = "python"
    data_environment: str | None = None
    ssh_options: tuple[str, ...] = ()
    command_prefix: tuple[str, ...] = ()
    scheduler_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Cluster profile names cannot be empty.")
        if self.transport not in {"local", "ssh"}:
            raise ValueError("Cluster transport must be 'local' or 'ssh'.")
        if self.scheduler not in {"local", "slurm"}:
            raise ValueError("Cluster scheduler must be 'local' or 'slurm'.")
        if self.transport == "ssh" and not self.host:
            raise ValueError("SSH cluster profiles require host.")
        if self.transport == "ssh" and not self.workspace.startswith("/"):
            raise ValueError("SSH cluster workspaces must be absolute paths.")
        object.__setattr__(self, "scheduler_options", FrozenJsonMapping(self.scheduler_options))

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> ClusterProfile:
        """Construct one profile from portable YAML values."""
        options = value.get("ssh_options", ())
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes, bytearray)):
            raise TypeError("ssh_options must be a list of OpenSSH arguments.")
        prefix = value.get("command_prefix", ())
        if not isinstance(prefix, Sequence) or isinstance(prefix, (str, bytes, bytearray)):
            raise TypeError("command_prefix must be an argument list.")
        return cls(
            name=name,
            transport=str(value.get("transport", "local")),
            scheduler=str(value.get("scheduler", "local")),
            host=str(value["host"]) if value.get("host") is not None else None,
            workspace=str(value.get("workspace", ".lambdaforge/remote")),
            python=str(value.get("python", "python")),
            data_environment=(
                str(value["data_environment"])
                if value.get("data_environment") is not None
                else None
            ),
            ssh_options=tuple(str(item) for item in options),
            command_prefix=tuple(str(item) for item in prefix),
            scheduler_options=copy.deepcopy(value.get("scheduler_options", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a redaction-safe profile descriptor."""
        return {
            "name": self.name,
            "transport": self.transport,
            "scheduler": self.scheduler,
            "host": self.host,
            "workspace": self.workspace,
            "python": self.python,
            "data_environment": self.data_environment or self.name,
            "ssh_options": list(self.ssh_options),
            "command_prefix": list(self.command_prefix),
            "scheduler_options": copy.deepcopy(self.scheduler_options),
        }
