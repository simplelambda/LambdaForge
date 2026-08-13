"""Typed timeout raised by a transport command."""

from __future__ import annotations


class RemoteCommandTimeout(TimeoutError):
    """Distinguish a caller-selected command deadline from connection failure."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        self.operation = operation
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(
            f"Remote command exceeded its explicit {self.timeout_seconds:g}s command timeout: "
            f"{operation}. This is independent from SSH connection timeout and job runtime."
        )
