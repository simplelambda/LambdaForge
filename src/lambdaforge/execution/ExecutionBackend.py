"""Execution backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.execution.BackendSubmission import BackendSubmission
from lambdaforge.execution.ResourceRequest import ResourceRequest


class ExecutionBackend(ABC):
    """Translate a portable command/resource plan into a backend submission."""

    @abstractmethod
    def submit(
        self,
        command: Sequence[str],
        resources: ResourceRequest,
        *,
        work_dir: str | Path,
        dry_run: bool = True,
    ) -> BackendSubmission:
        """Preview or submit one command without shell interpretation."""
