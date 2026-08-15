"""Plain-terminal and JSON renderers for the shared diagnostic model."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.diagnostics.models import ErrorDiagnostic


class DiagnosticRenderer:
    """Render one semantic diagnostic without relying on ANSI colour."""

    def human(
        self,
        value: ErrorDiagnostic,
        *,
        debug: bool = False,
        exception_type: str | None = None,
        traceback_text: str | None = None,
    ) -> str:
        """Render compact sections and reveal technical traceback only in debug mode."""
        lines = [value.category.heading, "-" * len(value.category.heading), "", value.title]
        if value.summary != value.title:
            lines.extend(("", value.summary))
        self._section(lines, "Why", (value.reason,))
        self._section(lines, "Impact", value.impact)
        context = tuple(
            f"{key}: {self._plain(item)}"
            for key, item in value.context.items()
            if item not in (None, "", (), [], {})
        )
        self._section(lines, "Context", context)
        self._section(lines, "Fix", value.fixes)
        commands = tuple(f"{label}:\n  {command}" for label, command in value.commands)
        self._section(lines, "Next action", commands)
        diagnostics = (
            (f"Record: {value.diagnostic_path}",) if value.diagnostic_path is not None else ()
        )
        self._section(lines, "Diagnostics", diagnostics)
        if value.details:
            self._section(lines, "Details", value.details)
        if debug:
            technical = []
            if exception_type:
                technical.append(f"Exception: {exception_type}")
            if traceback_text:
                technical.append(traceback_text.rstrip())
            self._section(lines, "Debug", tuple(technical))
        return SecretRedactor.redact("\n".join(lines).rstrip() + "\n")

    def json(
        self,
        value: ErrorDiagnostic,
        *,
        debug: bool = False,
        exception_type: str | None = None,
        traceback_text: str | None = None,
    ) -> str:
        """Render equivalent stable fields, with internals gated behind debug."""
        payload = self._redact_value(value.to_dict())
        if debug:
            payload["debug"] = self._redact_value(
                {"exception_type": exception_type, "traceback": traceback_text}
            )
        return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    @staticmethod
    def _section(lines: list[str], title: str, values: Sequence[str]) -> None:
        if not values:
            return
        lines.extend(("", title, "-" * len(title)))
        lines.extend(str(value) for value in values)

    @staticmethod
    def _plain(value: Any) -> str:
        if isinstance(value, Mapping):
            return json.dumps(value, sort_keys=True, default=str)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return ", ".join(str(item) for item in value)
        return str(value)

    @classmethod
    def _redact_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): (
                    SecretRedactor.MARKER
                    if SecretRedactor.is_secret_key(str(key))
                    else cls._redact_value(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._redact_value(item) for item in value]
        return SecretRedactor.redact(value) if isinstance(value, str) else value
