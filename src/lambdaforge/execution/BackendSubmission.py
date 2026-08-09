"""Portable backend submission result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BackendSubmission:
    """Record backend identity, generated artifact and optional scheduler job id."""

    backend: str
    artifact: Path | None = None
    job_id: str | None = None
    submitted: bool = False
