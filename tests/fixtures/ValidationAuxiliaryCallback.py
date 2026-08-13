"""Neutral validation callback demonstrating one-forward diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from lambdaforge.integrations.Lightning import CallbackBase


class ValidationAuxiliaryCallback(CallbackBase):
    """Stream one bounded auxiliary statistic from ordinary validation outputs."""

    def __init__(self, artifact_path: str) -> None:
        super().__init__()
        self.artifact_path = Path(artifact_path)
        self._values: list[float] = []

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Reset only the bounded per-epoch summaries."""
        del trainer, pl_module
        self._values.clear()

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Consume existing predictions without invoking the model again."""
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        model_outputs = outputs.get("model_outputs", {}) if isinstance(outputs, Mapping) else {}
        logits = model_outputs.get("user_logits") if isinstance(model_outputs, Mapping) else None
        if isinstance(logits, torch.Tensor):
            self._values.append(float(logits.detach().mean().cpu()))

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Publish an ordinary val_* metric usable by checkpointing and HPO."""
        if bool(getattr(trainer, "sanity_checking", False)) or not self._values:
            return
        score = sum(self._values) / len(self._values)
        pl_module.log("val_auxiliary_score", score, on_epoch=True, sync_dist=True)

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Write project artifacts on rank zero only."""
        del pl_module
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        self.artifact_path.write_text("rank-zero", encoding="utf-8")
