"""Control-plane command and file transport boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.controlplane.CommandResult import CommandResult


class Transport(ABC):
    """Execute argument vectors and stage small control bundles."""

    @abstractmethod
    def run(self, command: Sequence[str], *, cwd: str | Path | None = None) -> CommandResult:
        """Run a command without interpreting local shell syntax."""

    @abstractmethod
    def put(self, source: str | Path, destination: str | Path) -> None:
        """Copy one small file or directory to an explicit destination."""

    def get(self, source: str | Path, destination: str | Path) -> None:
        """Retrieve one explicit small file/directory when the provider supports it."""
        raise NotImplementedError(f"{type(self).__name__} does not support retrieval.")
