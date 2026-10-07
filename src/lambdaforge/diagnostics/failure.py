"""Pure failure policy shared by execution, recovery and human diagnostics.

This module never imports Torch, launches a retry or examines a filesystem. A worker-boundary
exception is different from an exception returned by consumer code, even with the same type.
"""

from __future__ import annotations

import errno
from collections.abc import Mapping
from typing import Any

from lambdaforge.diagnostics.models import (
    ErrorCategory,
    FailureDisposition,
    LambdaForgeError,
    RetryDisposition,
)


def classify_failure(
    failure: Mapping[str, Any] | BaseException | None, *, phase: str | None = None
) -> FailureDisposition:
    """Classify typed or persisted failure evidence without trusting prose as a retry grant."""
    if isinstance(failure, LambdaForgeError):
        structured = failure.diagnostic
        if structured.failure_disposition is not None:
            return structured.failure_disposition
        return FailureDisposition(
            structured.category,
            "typed_control_plane_error",
            "wait_for_verified_ownership"
            if structured.category is ErrorCategory.CONNECTION
            else "user_action"
            if structured.category is ErrorCategory.CANCELLED
            else "after_fix",
            structured.retryable,
            termination_type=(
                "aborted"
                if structured.category is ErrorCategory.CANCELLED
                else "infrastructure_failed"
            ),
        )
    if isinstance(failure, BaseException):
        kinds = {base.__name__ for base in type(failure).__mro__}
        message = str(failure).lower()
        error_number = failure.errno if isinstance(failure, OSError) else None
    else:
        record = failure or {}
        kinds = {str(record.get("type", "")).rsplit(".", 1)[-1]}
        ancestry = record.get("exception_types")
        if isinstance(ancestry, list | tuple):
            kinds.update(str(value) for value in ancestry[:16])
        message = str(record.get("message", "")).lower()
        error_number = record.get("errno")
        phase = phase or str(record.get("phase", "run work"))
        # Persisted typed diagnostics preserve control-plane categories across worker boundaries.
        structured_record = record.get("diagnostic")
        if isinstance(structured_record, Mapping):
            if "LambdaForgeError" in kinds:
                supplied_disposition = structured_record.get("failure_disposition")
                if isinstance(supplied_disposition, Mapping):
                    try:
                        return FailureDisposition.from_dict(supplied_disposition)
                    except (ValueError, TypeError, KeyError):
                        pass  # Invalid metadata never grants automatic retry.
            try:
                category = ErrorCategory(str(structured_record.get("category")))
            except ValueError:
                category = ErrorCategory.EXECUTION
            if "LambdaForgeError" in kinds or category not in {
                ErrorCategory.EXECUTION,
                ErrorCategory.RESOURCE,
            }:
                try:
                    retryable = RetryDisposition(str(structured_record.get("retryable")))
                except ValueError:
                    retryable = RetryDisposition.UNKNOWN
                return FailureDisposition(
                    category,
                    "typed_control_plane_error",
                    "wait_for_verified_ownership"
                    if category is ErrorCategory.CONNECTION
                    else "user_action"
                    if category is ErrorCategory.CANCELLED
                    else "after_fix",
                    retryable,
                    termination_type=(
                        "aborted"
                        if category is ErrorCategory.CANCELLED
                        else "infrastructure_failed"
                    ),
                )
    # Exact allocation failures only. CUDA illegal-access/dtype/kernel errors are not OOMs.
    if kinds & {"OutOfMemoryError", "CUDAOutOfMemoryError"} or (
        kinds & {"RuntimeError", "AcceleratorError"}
        and any(
            marker in message
            for marker in ("cuda out of memory", "hip out of memory", "cublas_status_alloc_failed")
        )
    ):
        return FailureDisposition(
            ErrorCategory.RESOURCE,
            "gpu_memory_allocation",
            "retry_safer_placement",
            RetryDisposition.IMMEDIATE,
            True,
            "resource_failed",
        )
    if phase == "worker-process" and kinds & {
        "BrokenProcessPool",
        "ChildProcessError",
        "ConnectionResetError",
        "EOFError",
    }:
        return FailureDisposition(
            ErrorCategory.EXECUTION,
            "lost_worker",
            "retry_same_run",
            RetryDisposition.IMMEDIATE,
            True,
            "infrastructure_failed",
        )
    if error_number in {errno.ENOSPC, errno.EDQUOT} or (
        kinds & {"OSError", "IOError"}
        and any(marker in message for marker in ("no space left on device", "disk quota exceeded"))
    ):
        return FailureDisposition(
            ErrorCategory.STORAGE,
            "storage_capacity",
            "after_fix",
            RetryDisposition.AFTER_FIX,
            termination_type="infrastructure_failed",
        )
    if kinds & {"KeyboardInterrupt", "CancelledError"}:
        return FailureDisposition(
            ErrorCategory.CANCELLED,
            "cancelled",
            "user_action",
            RetryDisposition.NO,
            termination_type="aborted",
        )
    return FailureDisposition(
        ErrorCategory.EXECUTION,
        "consumer_exception" if phase != "worker-process" else "worker_setup_error",
        "after_fix",
        RetryDisposition.AFTER_FIX,
        termination_type="scientific_failed"
        if phase != "worker-process"
        else "infrastructure_failed",
    )
