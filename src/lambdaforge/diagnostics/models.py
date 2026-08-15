"""Stable structured diagnostics shared by CLI, JSON and persistent records."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    """Small user-facing failure taxonomy; values are stable machine contracts."""

    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    ENVIRONMENT = "environment"
    EXECUTION = "execution"
    RESOURCE = "resource"
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    DATA = "data"
    STORAGE = "storage"
    OPERATION_REFUSED = "operation_refused"
    INTERNAL = "internal"
    WARNING = "warning"
    CANCELLED = "cancelled"

    @property
    def heading(self) -> str:
        """Return the terminal heading without leaking a Python exception class."""
        return {
            self.CONFIGURATION: "CONFIGURATION ERROR",
            self.VALIDATION: "VALIDATION ERROR",
            self.ENVIRONMENT: "ENVIRONMENT ERROR",
            self.EXECUTION: "EXECUTION FAILED",
            self.RESOURCE: "RESOURCE ERROR",
            self.CONNECTION: "CONNECTION ERROR",
            self.AUTHENTICATION: "AUTHENTICATION ERROR",
            self.DATA: "DATA ERROR",
            self.STORAGE: "STORAGE ERROR",
            self.OPERATION_REFUSED: "OPERATION REFUSED",
            self.INTERNAL: "INTERNAL ERROR",
            self.WARNING: "WARNING",
            self.CANCELLED: "CANCELLED",
        }[self]

    @property
    def code(self) -> str:
        """Return one stable code per contract category, not per message sentence."""
        suffix = {
            self.CONFIGURATION: "CONFIG",
            self.VALIDATION: "VALIDATION",
            self.ENVIRONMENT: "ENV",
            self.EXECUTION: "EXECUTION",
            self.RESOURCE: "RESOURCE",
            self.CONNECTION: "CONNECTION",
            self.AUTHENTICATION: "AUTH",
            self.DATA: "DATA",
            self.STORAGE: "STORAGE",
            self.OPERATION_REFUSED: "REFUSED",
            self.INTERNAL: "INTERNAL",
            self.WARNING: "WARNING",
            self.CANCELLED: "CANCELLED",
        }[self]
        return f"LF-{suffix}"

    @property
    def exit_code(self) -> int:
        """Map categories to a compact documented process-exit policy."""
        if self in {self.WARNING}:
            return 0
        if self is self.CANCELLED:
            return 130
        if self in {self.CONFIGURATION, self.VALIDATION, self.OPERATION_REFUSED}:
            return 2
        if self in {self.ENVIRONMENT, self.CONNECTION, self.AUTHENTICATION}:
            return 3
        if self in {self.EXECUTION, self.DATA}:
            return 4
        if self in {self.RESOURCE, self.STORAGE}:
            return 5
        return 10


class RetryDisposition(str, Enum):
    """Describe whether retrying now, after a fix, or not at all is meaningful."""

    IMMEDIATE = "immediate"
    AFTER_FIX = "after_fix"
    NO = "no"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ErrorDiagnostic:
    """Represent WHAT/WHY/IMPACT/FIX/NEXT ACTION without preformatted walls of text."""

    category: ErrorCategory
    title: str
    summary: str
    reason: str
    impact: tuple[str, ...]
    fixes: tuple[str, ...]
    commands: tuple[tuple[str, str], ...]
    context: Mapping[str, Any]
    retryable: RetryDisposition = RetryDisposition.AFTER_FIX
    operation: str | None = None
    job_id: str | None = None
    diagnostic_path: str | None = None
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "impact", tuple(str(value) for value in self.impact))
        object.__setattr__(self, "fixes", tuple(str(value) for value in self.fixes))
        object.__setattr__(
            self,
            "commands",
            tuple((str(label), str(command)) for label, command in self.commands),
        )
        object.__setattr__(self, "context", copy.deepcopy(dict(self.context)))
        object.__setattr__(self, "details", tuple(str(value) for value in self.details))

    @property
    def exit_code(self) -> int:
        return self.category.exit_code

    @property
    def code(self) -> str:
        return self.category.code

    def with_diagnostic_path(self, path: str) -> ErrorDiagnostic:
        """Attach a record after atomic persistence without mutating domain evidence."""
        return replace(self, diagnostic_path=path)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable machine-readable error envelope."""
        return {
            "status": (
                "warning"
                if self.category is ErrorCategory.WARNING
                else "cancelled"
                if self.category is ErrorCategory.CANCELLED
                else "error"
            ),
            "category": self.category.value,
            "code": self.code,
            "exit_code": self.exit_code,
            "title": self.title,
            "message": self.summary,
            "reason": self.reason,
            "impact": list(self.impact),
            "fixes": list(self.fixes),
            "commands": [{"label": label, "command": command} for label, command in self.commands],
            "context": copy.deepcopy(dict(self.context)),
            "retryable": self.retryable.value,
            "operation": self.operation,
            "job_id": self.job_id,
            "diagnostic_record": self.diagnostic_path,
            "details": list(self.details),
        }


class LambdaForgeError(RuntimeError):
    """Carry one structured expected diagnostic while preserving exception chaining."""

    def __init__(self, diagnostic: ErrorDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.summary)


def diagnostic(
    category: ErrorCategory,
    title: str,
    summary: str,
    *,
    reason: str,
    impact: Sequence[str] = (),
    fixes: Sequence[str] = (),
    commands: Sequence[tuple[str, str]] = (),
    context: Mapping[str, Any] | None = None,
    retryable: RetryDisposition = RetryDisposition.AFTER_FIX,
    operation: str | None = None,
    job_id: str | None = None,
    details: Sequence[str] = (),
) -> ErrorDiagnostic:
    """Construct a normalized diagnostic without adding a class for each sentence."""
    return ErrorDiagnostic(
        category,
        title,
        summary,
        reason,
        tuple(impact),
        tuple(fixes),
        tuple(commands),
        dict(context or {}),
        retryable,
        operation,
        job_id,
        None,
        tuple(details),
    )
