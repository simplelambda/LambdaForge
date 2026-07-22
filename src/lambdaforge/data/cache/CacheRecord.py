"""Closable payload returned by a dataset cache backend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


class CacheRecord:
    """Own a bytes-like payload and any resource needed to read it."""

    def __init__(
        self,
        payload: Any,
        close_callback: Callable[[], None] | None = None,
        close_callbacks: Sequence[Callable[[], None]] = (),
        token: str | None = None,
    ) -> None:
        self.payload = payload
        callbacks = ([close_callback] if close_callback is not None else []) + list(close_callbacks)
        self._close_callbacks = tuple(callbacks)
        self.token = token
        self._closed = False

    def __enter__(self) -> CacheRecord:
        """Return this live record."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Release a mapped file or other backend resource."""
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        """Release the payload resource exactly once."""
        if self._closed:
            return
        self._closed = True
        failure: BaseException | None = None
        if isinstance(self.payload, memoryview):
            try:
                self.payload.release()
            except BaseException as error:
                failure = error
        for callback in self._close_callbacks:
            try:
                callback()
            except BaseException as error:
                failure = failure or error
        if failure is not None:
            raise failure
