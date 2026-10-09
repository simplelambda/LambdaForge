"""Control-plane command and file transport boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from lambdaforge.controlplane.BinaryCommand import BinaryCommand
from lambdaforge.controlplane.CommandResult import CommandResult


class Transport(ABC):
    """Execute argument vectors and stage small control bundles."""

    @abstractmethod
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run argv with an optional command deadline independent of connection setup."""

    @abstractmethod
    def put(self, source: str | Path, destination: str | Path) -> None:
        """Copy one small file or directory to an explicit destination."""

    def get(self, source: str | Path, destination: str | Path) -> None:
        """Retrieve one explicit small file/directory when the provider supports it."""
        raise NotImplementedError(f"{type(self).__name__} does not support retrieval.")

    def stream(self, command: Sequence[str]) -> AbstractContextManager[BinaryCommand]:
        """Stream an explicit durable-data command, separate from small control messages."""
        raise NotImplementedError(f"{type(self).__name__} does not support binary streaming.")
