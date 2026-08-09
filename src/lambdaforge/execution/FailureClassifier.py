"""Conservative failure classification."""

from __future__ import annotations

from lambdaforge.execution.FailureCategory import FailureCategory


class FailureClassifier:
    """Classify known signals/errors without treating unknown failures as retryable."""

    def classify(
        self, error: BaseException | str, *, exit_code: int | None = None
    ) -> FailureCategory:
        """Return a stable terminal category from explicit evidence."""
        text = str(error).lower()
        if exit_code in {-15, 143} or "preempt" in text:
            return FailureCategory.PREEMPTED
        if isinstance(error, KeyboardInterrupt) or exit_code in {-2, 130}:
            return FailureCategory.CANCELLED
        if "cuda out of memory" in text or "cuda error: out of memory" in text:
            return FailureCategory.GPU_OOM
        if "out of memory" in text or "cannot allocate memory" in text:
            return FailureCategory.CPU_OOM
        if isinstance(error, (TimeoutError, ConnectionError)):
            return FailureCategory.TRANSIENT
        if isinstance(error, (TypeError, ValueError, KeyError, ImportError)):
            return FailureCategory.USER_ERROR
        return FailureCategory.UNKNOWN
