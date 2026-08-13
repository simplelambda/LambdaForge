"""Reuse-safe operating-system process identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Identify one LambdaForge child without trusting a reusable PID alone."""

    pid: int
    process_group: int
    create_time: float
    command_sha256: str
    job_id: str

    @classmethod
    def create(
        cls, pid: int, process_group: int, command: Sequence[str], job_id: str
    ) -> ProcessIdentity:
        import psutil

        digest = hashlib.sha256(
            json.dumps(list(command), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(pid, process_group, psutil.Process(pid).create_time(), digest, job_id)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProcessIdentity:
        return cls(
            int(value["pid"]),
            int(value["process_group"]),
            float(value["create_time"]),
            str(value["command_sha256"]),
            str(value["job_id"]),
        )

    def matches(self) -> bool:
        """Verify creation time and command before any signal is sent."""
        try:
            import psutil

            process = psutil.Process(self.pid)
            if abs(process.create_time() - self.create_time) > 0.01:
                return False
            digest = hashlib.sha256(
                json.dumps(process.cmdline(), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            return digest == self.command_sha256
        except Exception:
            return False

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "pid": self.pid,
            "process_group": self.process_group,
            "create_time": self.create_time,
            "command_sha256": self.command_sha256,
            "job_id": self.job_id,
        }
