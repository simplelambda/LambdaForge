"""Task-level loss and metric logging policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskLoggingConfig:
    """Control which task values reach Lightning and its progress bar.

    Metric objects still determine *what* is computed. This policy determines
    how total/individual losses and computed metric scalars are published.
    """

    log_total_loss: bool = True
    log_individual_losses: bool = True
    loss_on_step: bool = False
    loss_on_epoch: bool = True
    loss_prog_bar: bool = True
    individual_loss_prog_bar: bool = False
    metric_prog_bar: bool = True
    logger: bool = False
    sync_dist: bool = True

    @classmethod
    def from_value(
        cls,
        value: TaskLoggingConfig | Mapping[str, Any] | None,
    ) -> TaskLoggingConfig:
        """Normalize an object, YAML mapping or missing value."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("logging must be a TaskLoggingConfig, mapping, or null.")

    def __post_init__(self) -> None:
        """Reject a total-loss policy that publishes to no destination."""
        if self.log_total_loss and not (self.loss_on_step or self.loss_on_epoch):
            raise ValueError(
                "At least one of loss_on_step or loss_on_epoch must be true "
                "when log_total_loss is enabled."
            )
