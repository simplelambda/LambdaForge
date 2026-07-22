"""Run-scoped collection of successfully resolved plugin descriptors."""

from __future__ import annotations

import os
from contextvars import Token
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING

from lambdaforge.plugins.PluginDescriptor import PluginDescriptor

if TYPE_CHECKING:
    from lambdaforge.plugins.PluginRegistry import PluginRegistry


class PluginUsageSession:
    """Collect successful plugin resolutions made in one execution context.

    Sessions are created by :meth:`PluginRegistry.usage_session` and act as
    context managers. They are isolated from earlier validations and runs,
    deduplicate repeated/cache-hit resolutions, and intentionally stop at the
    process boundary.
    """

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry
        self._process_id = os.getpid()
        self._descriptors: dict[PluginDescriptor, None] = {}
        self._token: Token[tuple[PluginUsageSession, ...]] | None = None
        self._lock = RLock()

    def __enter__(self) -> PluginUsageSession:
        """Activate this session in the current context."""
        if self._token is not None:
            raise RuntimeError("A plugin usage session cannot be entered twice.")
        if os.getpid() != self._process_id:
            raise RuntimeError("A plugin usage session cannot cross a process boundary.")
        self._token = self._registry._activate_usage_session(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the previous context even when the run raises."""
        del exc_type, exc_value, traceback
        token = self._token
        self._token = None
        if token is not None and os.getpid() == self._process_id:
            self._registry._deactivate_usage_session(token)

    def descriptors(self) -> tuple[PluginDescriptor, ...]:
        """Return an immutable, deterministic snapshot of this run's plugins."""
        if os.getpid() != self._process_id:
            return ()
        with self._lock:
            return tuple(sorted(self._descriptors, key=PluginDescriptor.sort_key))

    def _record(self, descriptor: PluginDescriptor) -> None:
        """Record one successful resolution for the owning registry."""
        if os.getpid() != self._process_id:
            return
        with self._lock:
            self._descriptors[descriptor] = None
