"""Public error for a tracking backend whose optional dependency is absent."""

from __future__ import annotations

from lambdaforge.tracking.TrackingBackend import TrackingBackend


class TrackingDependencyError(ImportError):
    """Explain how to install the dependency required by a tracking backend."""

    def __init__(self, backend: TrackingBackend | str) -> None:
        """Build a stable, actionable optional-dependency error."""
        normalized = TrackingBackend.from_value(backend)
        self.backend = normalized
        self.dependency = normalized.dependency
        self.dependencies = normalized.dependencies
        self.extra = normalized.extra
        self.install_hint = normalized.install_hint
        alternatives = " or ".join(repr(name) for name in self.dependencies)
        super().__init__(
            f"Tracking backend {normalized.value!r} requires the optional "
            f"dependency {alternatives}. Install the recommended backend with: "
            f"{self.install_hint}"
        )
