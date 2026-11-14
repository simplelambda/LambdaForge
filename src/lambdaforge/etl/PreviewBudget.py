"""Shared cooperative budgets for explicit domain reads; never a plugin sandbox."""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from typing import Any

from lambdaforge.catalog.errors import ContractError


class PreviewBudget:
    """Global rows/bytes/files/deadline and conservative decoded-size allocation guard.

    Share one instance across ordinary Python reader calls. Byte counters include encoded
    input actually read; memory bounds guard decoded rows/chunks, not process RSS or arbitrary
    plugin allocations. Deadlines/cancellation are cooperative; use provider I/O timeouts too.
    """

    def __init__(
        self,
        max_rows: int = 100,
        max_bytes: int = 1024**2,
        max_seconds: float = 5,
        max_memory: int = 8 * 1024**2,
        max_files: int = 8,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        if any(type(v) is not int or v < 1 for v in (max_rows, max_bytes, max_memory, max_files)):
            raise ContractError("INVALID_BUDGET", "Positive integer preview limits required.")
        if isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ContractError("INVALID_BUDGET", "A finite positive deadline is required.")
        self.max_rows, self.max_bytes, self.max_memory, self.max_files = (
            max_rows,
            max_bytes,
            max_memory,
            max_files,
        )
        self.deadline = time.monotonic() + max_seconds
        self.cancelled = cancelled
        self.rows = self.bytes = self.files = 0
        self.retained = 0

    def check(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise ContractError("CANCELLED", "Read cancelled.")
        if time.monotonic() > self.deadline:
            raise ContractError("READ_LIMIT", "Read deadline exceeded.")

    def open_file(self) -> None:
        self.check()
        if self.files >= self.max_files:
            raise ContractError("READ_LIMIT", "Global file budget exceeded.")
        self.files += 1

    def charge(self, count: int) -> None:
        self.check()
        if self.bytes + count > self.max_bytes:
            raise ContractError("READ_LIMIT", "Global input byte budget exceeded.")
        self.bytes += count

    def size(self, value: Any) -> int:
        """JSON rows only: no repr/pickle or flattening of scientific arrays."""
        try:
            encoded_size = len(
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                "UNSUPPORTED_VALUE", "Expected portable finite table values."
            ) from error
        seen: set[int] = set()

        def deep_size(item: Any) -> int:
            if id(item) in seen:
                return 0
            seen.add(id(item))
            total = sys.getsizeof(item)
            if isinstance(item, dict):
                total += sum(deep_size(k) + deep_size(v) for k, v in item.items())
            elif isinstance(item, (list, tuple)):
                total += sum(deep_size(v) for v in item)
            return total

        size = max(encoded_size * 4, deep_size(value))
        if size > self.max_memory:
            raise ContractError("READ_LIMIT", "Decoded row exceeds memory guard.")
        return size

    def row(self, value: Any) -> None:
        self.check()
        if self.full:
            raise ContractError("READ_LIMIT", "Global row budget exceeded.")
        self.size(value)
        self.rows += 1

    @property
    def full(self) -> bool:
        return self.rows >= self.max_rows

    def retain(self, size: int) -> None:
        """Charge retained preview objects globally, not streaming rows already consumed."""
        self.check()
        if self.retained + size > self.max_memory:
            raise ContractError("READ_LIMIT", "Global retained preview exceeds memory guard.")
        self.retained += size

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "preview-budget-v1",
            "rows": self.rows,
            "bytes": self.bytes,
            "files": self.files,
            "retained_bytes_guard": self.retained,
            "limits": {
                "rows": self.max_rows,
                "bytes": self.max_bytes,
                "memory": self.max_memory,
                "files": self.max_files,
            },
            "enforcement": "cooperative",
        }
