"""Validated named execution-cluster profile."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from lambdaforge.controlplane.ClusterAuthentication import ClusterAuthentication
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.python_runtime import PythonRuntimePolicy
from lambdaforge.controlplane.SlurmProfile import SlurmProfile
from lambdaforge.controlplane.SshConnectionPolicy import SshConnectionPolicy
from lambdaforge.controlplane.TorchInstallationPolicy import TorchInstallationPolicy
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class ClusterProfile:
    """Describe access, scheduler, workspace and environment policy."""

    name: str
    transport: str = "local"
    scheduler: str = "local"
    host: str | None = None
    user: str | None = None
    port: int = 22
    auth: ClusterAuthentication = field(default_factory=ClusterAuthentication)
    known_hosts: str | None = None
    ssh_timeout: float = 15.0
    connection: SshConnectionPolicy = field(default_factory=SshConnectionPolicy)
    workspace: str = ".lambdaforge/remote"
    storage: ClusterStoragePolicy | None = None
    python: str = "python"
    environment: str = "existing"
    wheelhouse: str | None = None
    pytorch: TorchInstallationPolicy = field(default_factory=TorchInstallationPolicy)
    project_module: str | None = None
    data_environment: str | None = None
    ssh_options: tuple[str, ...] = ()
    command_prefix: tuple[str, ...] = ()
    scheduler_options: Mapping[str, Any] = field(default_factory=dict)
    slurm_profile: SlurmProfile | Mapping[str, Any] | None = None
    python_runtime: PythonRuntimePolicy | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Cluster profile names cannot be empty.")
        if self.transport not in {"local", "ssh"}:
            raise ValueError("Cluster transport must be 'local' or 'ssh'.")
        if self.scheduler not in {"local", "slurm"}:
            raise ValueError("Cluster scheduler must be 'local' or 'slurm'.")
        if self.environment not in {"existing", "managed"}:
            raise ValueError("Cluster environment must be 'existing' or 'managed'.")
        runtime = self.python_runtime
        if runtime is not None and not isinstance(runtime, PythonRuntimePolicy):
            runtime = PythonRuntimePolicy.from_value(runtime)
            object.__setattr__(self, "python_runtime", runtime)
        if runtime is not None and self.python != runtime.executable:
            object.__setattr__(self, "python", runtime.executable)
        pytorch = self.pytorch
        if not isinstance(pytorch, TorchInstallationPolicy):
            pytorch = TorchInstallationPolicy.from_mapping(pytorch)
            object.__setattr__(self, "pytorch", pytorch)
        if self.transport == "ssh" and not self.host:
            raise ValueError("SSH cluster profiles require host.")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("SSH port must be an integer between 1 and 65535.")
        if self.user is not None and (
            not self.user.strip() or "\n" in self.user or "@" in self.user
        ):
            raise ValueError("SSH user must be a non-empty user name without '@' or newlines.")
        if self.ssh_timeout <= 0:
            raise ValueError("SSH timeout must be positive.")
        connection = self.connection
        if not isinstance(connection, SshConnectionPolicy):
            connection = SshConnectionPolicy.from_mapping(
                connection, legacy_timeout=self.ssh_timeout
            )
            object.__setattr__(self, "connection", connection)
        if self.transport == "ssh" and not self.workspace.startswith("/"):
            raise ValueError("SSH cluster workspaces must be absolute paths.")
        storage = self.storage
        if not isinstance(storage, ClusterStoragePolicy):
            storage = ClusterStoragePolicy.from_mapping(storage, workspace=self.workspace)
            object.__setattr__(self, "storage", storage)
        if (
            self.project_module is not None
            and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", self.project_module) is None
        ):
            raise ValueError("project_module must be a fully qualified Python module name.")
        object.__setattr__(self, "scheduler_options", FrozenJsonMapping(self.scheduler_options))
        auth = self.auth
        if not isinstance(auth, ClusterAuthentication):
            auth = ClusterAuthentication.from_mapping(auth)
            object.__setattr__(self, "auth", auth)
        if auth.mode == "password" and self.transport != "ssh":
            raise ValueError("Password authentication is available only for SSH profiles.")
        slurm_profile = self.slurm_profile
        if not isinstance(slurm_profile, SlurmProfile):
            slurm_profile = SlurmProfile.from_mapping(
                slurm_profile, legacy_options=self.scheduler_options
            )
            object.__setattr__(self, "slurm_profile", slurm_profile)

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> ClusterProfile:
        """Construct one profile from portable YAML values."""
        options = value.get("ssh_options", ())
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes, bytearray)):
            raise TypeError("ssh_options must be a list of OpenSSH arguments.")
        prefix = value.get("command_prefix", ())
        if not isinstance(prefix, Sequence) or isinstance(prefix, (str, bytes, bytearray)):
            raise TypeError("command_prefix must be an argument list.")
        raw_python = value.get("python", "python")
        runtime = PythonRuntimePolicy.from_value(raw_python)
        return cls(
            name=name,
            transport=str(value.get("transport", "local")),
            scheduler=str(value.get("scheduler", "local")),
            host=str(value["host"]) if value.get("host") is not None else None,
            user=str(value["user"]) if value.get("user") is not None else None,
            port=int(value.get("port", 22)),
            auth=ClusterAuthentication.from_mapping(value.get("auth")),
            known_hosts=(str(value["known_hosts"]) if value.get("known_hosts") else None),
            ssh_timeout=float(value.get("ssh_timeout", 15.0)),
            connection=SshConnectionPolicy.from_mapping(
                value.get("connection"),
                legacy_timeout=float(value.get("ssh_timeout", 15.0)),
            ),
            workspace=str(value.get("workspace", ".lambdaforge/remote")),
            storage=ClusterStoragePolicy.from_mapping(
                value.get("storage"),
                workspace=str(value.get("workspace", ".lambdaforge/remote")),
            ),
            python=runtime.executable,
            python_runtime=(runtime if isinstance(raw_python, Mapping) else None),
            environment=str(value.get("environment", "existing")),
            wheelhouse=(str(value["wheelhouse"]) if value.get("wheelhouse") else None),
            pytorch=TorchInstallationPolicy.from_mapping(value.get("pytorch")),
            project_module=(str(value["project_module"]) if value.get("project_module") else None),
            data_environment=(
                str(value["data_environment"])
                if value.get("data_environment") is not None
                else None
            ),
            ssh_options=tuple(str(item) for item in options),
            command_prefix=tuple(str(item) for item in prefix),
            scheduler_options=copy.deepcopy(value.get("scheduler_options", {})),
            slurm_profile=SlurmProfile.from_mapping(
                {
                    key: copy.deepcopy(value[key])
                    for key in (
                        "resource_mapping",
                        "scheduler_directives",
                        "scheduler_commands",
                        "job_script",
                    )
                    if key in value
                },
                legacy_options=value.get("scheduler_options", {}),
            ),
        )

    def to_dict(self, *, include_defaults: bool = True) -> dict[str, Any]:
        """Return a redaction-safe effective or compact profile descriptor."""
        slurm = cast(SlurmProfile, self.slurm_profile)
        slurm_descriptor = slurm.to_dict(include_defaults=include_defaults)
        if not include_defaults and "scheduler_directives" in slurm_descriptor:
            explicit = dict(slurm_descriptor["scheduler_directives"])
            for key, value in self.scheduler_options.items():
                if explicit.get(key) == value:
                    explicit.pop(key)
            if explicit:
                slurm_descriptor["scheduler_directives"] = explicit
            else:
                slurm_descriptor.pop("scheduler_directives")
        return {
            "name": self.name,
            "transport": self.transport,
            "scheduler": self.scheduler,
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "auth": self.auth.to_dict(),
            "known_hosts": self.known_hosts,
            "ssh_timeout": self.ssh_timeout,
            "connection": self.connection.to_dict(),
            "workspace": self.workspace,
            "storage": cast(ClusterStoragePolicy, self.storage).to_dict(),
            "python": (
                self.python_runtime.to_dict() if self.python_runtime is not None else self.python
            ),
            "environment": self.environment,
            "wheelhouse": self.wheelhouse,
            "pytorch": self.pytorch.to_dict(),
            "project_module": self.project_module,
            "data_environment": self.data_environment or self.name,
            "ssh_options": list(self.ssh_options),
            "command_prefix": list(self.command_prefix),
            "scheduler_options": copy.deepcopy(self.scheduler_options),
            **(slurm_descriptor if self.scheduler == "slurm" else {}),
        }

    @property
    def runtime_policy(self) -> PythonRuntimePolicy:
        """Return the explicit policy or the backward-compatible string interpretation."""
        return self.python_runtime or PythonRuntimePolicy("existing", self.python)
