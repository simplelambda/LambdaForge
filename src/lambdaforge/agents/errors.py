"""Agent-facing error semantics over existing domain diagnostics, never another runner."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

import yaml

from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.diagnostics.models import ErrorCategory, LambdaForgeError, RetryDisposition

ErrorCode = Literal[
    "NOT_FOUND",
    "AMBIGUOUS_SELECTOR",
    "INVALID_ARGUMENT",
    "SCIENTIFIC_INCOMPATIBILITY",
    "PERMISSION_DENIED",
    "APPROVAL_REQUIRED",
    "APPROVAL_EXPIRED",
    "PLAN_STALE",
    "OPERATION_CONFLICT",
    "PROVIDER_UNREACHABLE",
    "PROVIDER_UNCERTAIN",
    "EVIDENCE_UNAVAILABLE",
    "EVIDENCE_CORRUPT",
    "UNSUPPORTED",
    "DEPENDENCY_MISSING",
    "TIMEOUT",
    "RESPONSE_TOO_LARGE",
    "INTERNAL_ERROR",
]


@dataclass(frozen=True)
class AgentError:
    """Bounded public diagnostic; retryability is advice, never approval for another apply."""

    code: ErrorCode
    message: str
    retryable: bool = False
    retry_after: float | None = None
    diagnostic_reference: str | None = None
    recommended_action: str | None = None
    original_object_id: str | None = None


class AgentOperationError(ValueError):
    """Explicit adapter failure where a native exception has no finer machine contract."""

    def __init__(self, code: ErrorCode, message: str, *, recommended_action: str | None = None):
        super().__init__(message)
        self.code = code
        self.recommended_action = recommended_action


class AgentApprovalError(PermissionError):
    """Keep existing PermissionError compatibility while identifying exact approval refusal."""

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code


def classify_error(error: Exception, *, knowledge: bool = False) -> AgentError:
    """Use exception types/native categories, not fragile matching of human error sentences."""
    code: ErrorCode
    retryable = False
    action = None
    reference = None
    message = str(error)
    if isinstance(error, AgentOperationError | AgentApprovalError):
        code = error.code
        action = getattr(error, "recommended_action", None)
    elif isinstance(error, LambdaForgeError):
        native = error.diagnostic
        native_codes: dict[ErrorCategory, ErrorCode] = {
            ErrorCategory.CONFIGURATION: "INVALID_ARGUMENT",
            ErrorCategory.VALIDATION: "SCIENTIFIC_INCOMPATIBILITY",
            ErrorCategory.CONNECTION: "PROVIDER_UNREACHABLE",
            ErrorCategory.AUTHENTICATION: "PERMISSION_DENIED",
            ErrorCategory.DATA: "EVIDENCE_CORRUPT",
            ErrorCategory.ENVIRONMENT: "DEPENDENCY_MISSING",
            ErrorCategory.OPERATION_REFUSED: "OPERATION_CONFLICT",
        }
        code = native_codes.get(native.category, "INTERNAL_ERROR")
        message = native.summary
        retryable = native.retryable is RetryDisposition.IMMEDIATE
        reference = native.diagnostic_path
        action = native.fixes[0] if native.fixes else None
    elif isinstance(error, PermissionError):
        code = "PERMISSION_DENIED"
    elif isinstance(error, FileNotFoundError | KeyError):
        code = "NOT_FOUND"
    elif isinstance(error, TimeoutError):
        code, retryable = "TIMEOUT", True
    elif isinstance(error, ConnectionError):
        code, retryable = "PROVIDER_UNREACHABLE", True
    elif isinstance(error, OverflowError):
        code = "RESPONSE_TOO_LARGE"
        action = "Request a smaller page or selected detail; do not increase the safety bound."
    elif isinstance(error, sqlite3.DatabaseError):
        code = "EVIDENCE_UNAVAILABLE"
        if knowledge:
            message = "Knowledge index is unreadable; canonical Markdown remains unchanged."
            action = "Use lf ai knowledge reindex --rebuild."
            message += " " + action
    elif isinstance(error, ValueError | yaml.YAMLError):
        code = "INVALID_ARGUMENT"
    elif isinstance(error, RuntimeError | OSError):
        code = "EVIDENCE_UNAVAILABLE"
    else:
        code = "INTERNAL_ERROR"
        message = "An internal agent adapter failed; no successful operation is confirmed."
        action = "Inspect the native diagnostic locally; do not automatically repeat a mutation."
    return AgentError(
        code,
        SecretRedactor.redact(message)[-1800:],
        retryable,
        diagnostic_reference=SecretRedactor.redact(reference) if reference else None,
        recommended_action=SecretRedactor.redact(action)[:1000] if action else None,
    )
