"""Bounded retry and attempt-lineage policy."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from lambdaforge.execution.FailureCategory import FailureCategory
from lambdaforge.execution.FailureClassifier import FailureClassifier

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry only declared categories with bounded exponential backoff."""

    max_attempts: int = 1
    backoff_seconds: float = 0.0
    retry_categories: tuple[FailureCategory, ...] = (
        FailureCategory.TRANSIENT,
        FailureCategory.PREEMPTED,
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.backoff_seconds < 0:
            raise ValueError("Retry max_attempts must be positive and backoff non-negative.")

    def execute(self, operation: Callable[[int, str | None], T]) -> T:
        """Execute with attempt number and parent lineage, re-raising terminal errors."""
        parent: str | None = None
        classifier = FailureClassifier()
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation(attempt, parent)
            except Exception as error:
                category = classifier.classify(error)
                if attempt == self.max_attempts or category not in self.retry_categories:
                    raise
                parent = f"attempt-{attempt}"
                if self.backoff_seconds:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError("Retry policy exhausted without a terminal result.")
