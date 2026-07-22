"""Lazy availability and import guard for optional tracking dependencies."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

from lambdaforge.tracking.TrackingBackend import TrackingBackend
from lambdaforge.tracking.TrackingDependencyError import TrackingDependencyError


class TrackingDependencyGuard:
    """Check and load one tracking dependency only at its point of use."""

    def __init__(self, backend: TrackingBackend | str) -> None:
        """Bind the guard to one normalized backend identifier."""
        self._backend = TrackingBackend.from_value(backend)

    @property
    def backend(self) -> TrackingBackend:
        """Return the backend protected by this guard."""
        return self._backend

    @property
    def is_available(self) -> bool:
        """Return whether the dependency can be resolved without importing it."""
        return self._available_dependency() is not None

    def _available_dependency(self) -> str | None:
        """Resolve the first supported dependency without importing it."""
        for dependency in self._backend.dependencies:
            if dependency in sys.modules:
                return dependency
            try:
                if importlib.util.find_spec(dependency) is not None:
                    return dependency
            except (ModuleNotFoundError, ValueError):
                continue
        return None

    def require(self) -> None:
        """Raise an actionable error unless a dependency is discoverable."""
        if not self.is_available:
            raise TrackingDependencyError(self._backend)

    def import_dependency(self) -> ModuleType:
        """Import and return the dependency after an explicit availability check."""
        dependency = self._available_dependency()
        if dependency is None:
            raise TrackingDependencyError(self._backend)
        try:
            return importlib.import_module(dependency)
        except ModuleNotFoundError as error:
            if error.name == dependency:
                raise TrackingDependencyError(self._backend) from error
            raise
