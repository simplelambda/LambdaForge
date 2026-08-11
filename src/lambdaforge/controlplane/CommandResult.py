"""Result of one transport command."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Keep stdout, stderr and exit status provider-neutral."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
