"""Callback for cooperative process shutdown."""

from __future__ import annotations

from typing import Any

from lambdaforge.integrations.Lightning import CallbackBase


class StopEventCallback(CallbackBase):
    """Stop a Lightning loop when an external multiprocessing event is set."""

    def __init__(self, stop_event: Any) -> None:
        super().__init__()
        self.stop_event = stop_event

    def on_train_batch_start(self, trainer: Any, *_: Any) -> None:
        self._check(trainer)

    def on_validation_batch_start(self, trainer: Any, *_: Any) -> None:
        self._check(trainer)

    def on_test_batch_start(self, trainer: Any, *_: Any) -> None:
        self._check(trainer)

    def _check(self, trainer: Any) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            trainer.should_stop = True
