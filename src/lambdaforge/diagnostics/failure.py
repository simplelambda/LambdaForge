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
    diagnostic,
)


def provider_observation_category(error: Mapping[str, Any]) -> ErrorCategory:
    """Classify failed reads without turning every exception into a network outage.

    Use confirmed exception types, not arbitrary message substrings or scientific state.
    Unclassified query/data errors leave connectivity unknown, even after repeated polls.
    """
    kind = str(error.get("type"))
    if kind in {"AuthenticationError", "AuthenticationException", "BadAuthenticationType"}:
        return ErrorCategory.AUTHENTICATION
    if kind in {
        "TimeoutError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "SSHError",
        "SSHException",
        "NoValidConnectionsError",
    }:
        return ErrorCategory.CONNECTION
    return ErrorCategory.WARNING


def operational_diagnostic(
    health: Mapping[str, Any],
    *,
    state: str,
    failure: Mapping[str, Any] | None = None,
    failure_phase: str | None = None,
    failure_category: str | None = None,
) -> dict[str, Any] | None:
    """Explain provider/runtime facts using the native diagnostic taxonomy.

    This is presentation, not dispatch permission. In particular, an exit signal, a CUDA
    message or missing metrics cannot prove OOM or process loss. Scientific failure records
    must be supplied explicitly; the observer never opens their artifact/epoch files.
    """
    facts = health.get("provider_evidence", {})
    facts = facts if isinstance(facts, Mapping) else {}
    source = str(facts.get("source") or health.get("source") or "unobserved")
    reason = str(health.get("reason") or "NO_PROVIDER_OBSERVATION")
    status = str(health.get("health") or "unknown")
    error = health.get("last_error")
    error = error if isinstance(error, Mapping) else {}
    raw = str(facts.get("raw_state", "")).split(" ")[0].rstrip("+")
    confirmed = health.get("freshness") == "fresh" and status in {"healthy", "lost_confirmed"}
    disposition: FailureDisposition
    title: str
    recommendation: str
    if error:
        # A malformed/unreadable observation is not necessarily an unreachable host.
        category = provider_observation_category(error)
        disposition = FailureDisposition(
            category,
            "provider_observation_failed",
            "wait_for_verified_ownership",
            RetryDisposition.AFTER_FIX,
            termination_type="unknown",
        )
        title = "Provider observation failed; computation status is unconfirmed."
        recommendation = "Restore provider access and reconcile ownership before recovery."
    elif status == "lost_confirmed":
        disposition = FailureDisposition(
            ErrorCategory.EXECUTION,
            "owned_processes_lost",
            "deliberate_study_recovery",
            RetryDisposition.AFTER_FIX,
            termination_type="infrastructure_failed",
        )
        title = "The execution host confirms that the owned processes are absent."
        recommendation = "Preview native retry; preserve completed Runs and validate checkpoints."
    elif status in {"degraded", "stale", "unknown"}:
        disposition = FailureDisposition(
            ErrorCategory.WARNING,
            "observation_unconfirmed",
            "wait_for_verified_ownership",
            RetryDisposition.UNKNOWN,
            termination_type="unknown",
        )
        title = "Execution is not currently verified; silence is not proof of failure."
        recommendation = "Inspect provider, heartbeat and child ownership; do not duplicate work."
    elif source in {"squeue", "sacct"} and raw in {
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
        "CANCELLED",
        "BOOT_FAIL",
    }:
        category, specific, title, recommendation = {
            "NODE_FAIL": (
                ErrorCategory.EXECUTION,
                "scheduler_node_failure",
                "SLURM reports a node failure.",
                "Check node/storage availability, then preview deliberate recovery.",
            ),
            "BOOT_FAIL": (
                ErrorCategory.ENVIRONMENT,
                "scheduler_boot_failure",
                "SLURM reports a boot failure.",
                "Inspect the node/runtime before selecting a compatible recovery placement.",
            ),
            "OUT_OF_MEMORY": (
                ErrorCategory.RESOURCE,
                "scheduler_memory_limit",
                "SLURM reports a memory-limit failure.",
                "Inspect host/cgroup memory limits; this alone does not identify CUDA OOM.",
            ),
            "PREEMPTED": (
                ErrorCategory.EXECUTION,
                "scheduler_preemption",
                "SLURM reports preemption.",
                "Check requeue/ownership before retrying the interrupted scientific Runs.",
            ),
            "TIMEOUT": (
                ErrorCategory.RESOURCE,
                "scheduler_time_limit",
                "SLURM reports a time-limit expiry.",
                "Review the allocation duration and compatible checkpoint recovery.",
            ),
            "CANCELLED": (
                ErrorCategory.CANCELLED,
                "scheduler_cancellation",
                "SLURM reports cancellation.",
                "Cancellation does not identify its requester; recovery remains deliberate.",
            ),
        }[raw]
        disposition = FailureDisposition(
            category,
            specific,
            "deliberate_study_recovery",
            RetryDisposition.AFTER_FIX,
            termination_type="aborted" if raw == "CANCELLED" else "infrastructure_failed",
        )
    elif failure is not None:
        disposition = classify_failure(failure, phase=failure_phase)
        title = "Persisted failure evidence identifies the execution error."
        recommendation = (
            "Review safer placement and live headroom before deliberate recovery."
            if disposition.reason == "gpu_memory_allocation"
            else "Correct the reported failure; generic CUDA/consumer errors are not OOM retries."
        )
    elif state in {"failed", "timeout", "cancelled"}:
        try:
            category = (
                ErrorCategory(failure_category) if failure_category else ErrorCategory.EXECUTION
            )
        except ValueError:
            category = ErrorCategory.EXECUTION
        disposition = FailureDisposition(
            category,
            "terminal_cause_unavailable",
            "inspect_persisted_failure",
            RetryDisposition.AFTER_FIX,
            termination_type="unknown",
        )
        title = "Execution is terminal; its scientific cause has not been inspected."
        recommendation = "Read the persisted failure/logs before changing science or retrying."
    else:
        return None
    job_id = str(health.get("job_id") or "")
    event_time = (
        facts.get("finished_at_utc")
        or health.get("last_attempt_at_utc")
        or health.get("last_confirmed_at_utc")
    )
    return {
        **diagnostic(
            disposition.category,
            title,
            reason,
            reason=disposition.reason,
            impact=("Scientific evidence and physical Attempt costs remain unchanged.",),
            fixes=(recommendation,),
            commands=(("Inspect exact Job", f"lf jobs show {job_id} --json"),),
            context={
                "source": source,
                "confirmation": "verified" if confirmed else "unconfirmed",
                "event_time_utc": event_time,
                "cluster": health.get("cluster"),
                "scheduler_id": health.get("scheduler_id"),
                "execution_id": health.get("execution_id"),
                "failure_phase": failure_phase,
            },
            retryable=disposition.retryable,
            operation="operational observation",
            job_id=job_id,
            failure_disposition=disposition,
        ).to_dict(),
        "severity": "warning" if disposition.category is ErrorCategory.WARNING else "error",
    }


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
    if "StorageOwnershipError" in kinds:
        return FailureDisposition(
            ErrorCategory.STORAGE,
            "storage_ownership",
            "wait_for_verified_ownership",
            RetryDisposition.AFTER_FIX,
            termination_type="infrastructure_failed",
        )
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
