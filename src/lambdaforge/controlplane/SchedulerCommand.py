"""Safe configurable scheduler command."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SchedulerCommand:
    """Represent a scheduler executable and argv template without a shell."""

    executable: str
    arguments: tuple[str, ...] = ()
    job_id_pattern: str | None = None

    def __post_init__(self) -> None:
        if not self.executable.strip() or "\x00" in self.executable or "\n" in self.executable:
            raise ValueError("Scheduler executable must be a non-empty argv item.")
        if any("\x00" in item or "\n" in item for item in self.arguments):
            raise ValueError("Scheduler command arguments cannot contain NUL or newlines.")
        if self.job_id_pattern is not None:
            try:
                compiled = re.compile(self.job_id_pattern)
            except re.error as error:
                raise ValueError(f"Invalid scheduler job_id_pattern: {error}.") from error
            if compiled.groups < 1:
                raise ValueError("job_id_pattern must contain a capturing group for the job id.")

    @classmethod
    def from_value(
        cls, value: Mapping[str, Any] | Sequence[str] | str | None, *, default: SchedulerCommand
    ) -> SchedulerCommand:
        """Parse a string, argv list or explicit command mapping."""
        if value is None:
            return default
        if isinstance(value, str):
            return cls(value, default.arguments, default.job_id_pattern)
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            items = tuple(str(item) for item in value)
            if not items:
                raise ValueError("Scheduler command argv cannot be empty.")
            return cls(items[0], items[1:], default.job_id_pattern)
        if not isinstance(value, Mapping):
            raise TypeError("Scheduler commands must be a string, argv list or mapping.")
        raw_arguments = value.get("args", value.get("arguments", default.arguments))
        if not isinstance(raw_arguments, Sequence) or isinstance(
            raw_arguments, (str, bytes, bytearray)
        ):
            raise TypeError("Scheduler command args must be a list.")
        return cls(
            str(value.get("command", value.get("executable", default.executable))),
            tuple(str(item) for item in raw_arguments),
            (
                str(value["job_id_pattern"])
                if value.get("job_id_pattern") is not None
                else default.job_id_pattern
            ),
        )

    def render(self, values: Mapping[str, str], *, allowed: set[str]) -> tuple[str, ...]:
        """Render only approved full or embedded placeholders into argv items."""
        rendered: list[str] = [self.executable]
        formatter = string.Formatter()
        for argument in self.arguments:
            for _, field, format_spec, conversion in formatter.parse(argument):
                if field is None:
                    continue
                if field not in allowed or format_spec or conversion:
                    raise ValueError(f"Unsafe scheduler command placeholder {{{field}}}.")
            try:
                rendered.append(argument.format_map(values))
            except KeyError as error:
                raise ValueError(f"Unavailable scheduler command placeholder: {error}.") from error
        return tuple(rendered)

    def to_dict(self) -> dict[str, Any]:
        """Return a portable command descriptor."""
        value: dict[str, Any] = {"command": self.executable, "args": list(self.arguments)}
        if self.job_id_pattern is not None:
            value["job_id_pattern"] = self.job_id_pattern
        return value
