"""Canonical identifiers and installation metadata for tracking backends."""

from __future__ import annotations

from enum import Enum


class TrackingBackend(str, Enum):
    """Identify a supported optional experiment-tracking runtime."""

    MLFLOW = "mlflow"
    TENSORBOARD = "tensorboard"
    WEIGHTS_AND_BIASES = "wandb"

    @property
    def dependency(self) -> str:
        """Return the canonical dependency used in errors and provenance."""
        return self.dependencies[0]

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Return accepted import targets in deterministic preference order."""
        if self is TrackingBackend.TENSORBOARD:
            return ("tensorboard", "tensorboardX")
        return (self.value,)

    @property
    def extra(self) -> str:
        """Return the LambdaForge optional-extra name for this backend."""
        return self.value

    @property
    def install_hint(self) -> str:
        """Return the smallest pip command that enables this backend."""
        return f"pip install 'lambdaforge[{self.extra}]'"

    @classmethod
    def from_value(cls, value: TrackingBackend | str) -> TrackingBackend:
        """Normalize a public enum or configuration string."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as error:
            choices = ", ".join(backend.value for backend in cls)
            raise ValueError(f"Unknown tracking backend {value!r}. Options: {choices}.") from error
