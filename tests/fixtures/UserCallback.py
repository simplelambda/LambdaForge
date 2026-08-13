"""Example project-defined callback used by YAML extension tests."""

from pathlib import Path
from typing import Any

from lambdaforge.integrations.Lightning import CallbackBase


class UserCallback(CallbackBase):
    """Write a marker proving that a YAML-provided callback was invoked."""

    def __init__(self, marker_path: str) -> None:
        super().__init__()
        self.marker_path = Path(marker_path)

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Persist the integration marker."""
        del pl_module
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        self.marker_path.write_text("callback invoked", encoding="utf-8")
