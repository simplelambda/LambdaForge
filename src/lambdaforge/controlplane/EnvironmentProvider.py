"""Remote environment-provider boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.PreparedEnvironment import PreparedEnvironment
from lambdaforge.controlplane.Transport import Transport


class EnvironmentProvider(ABC):
    """Prepare or verify one user-space interpreter for an execution bundle."""

    @abstractmethod
    def prepare(
        self,
        profile: ClusterProfile,
        transport: Transport,
        bundle: ExecutionBundle,
        *,
        remote_bundle_dir: str | Path,
    ) -> PreparedEnvironment:
        """Return the interpreter that must execute the bundle."""
