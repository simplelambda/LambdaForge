"""Explicit train/fold-scoped learned normalization, stored as portable JSON state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lambdaforge.catalog.errors import ContractError


class TrainStandardizer:
    """Fit one state per training split/fold. Never fit while transforming validation/test."""

    def __init__(self) -> None:
        self._state: dict[str, Any] | None = None

    def fit(
        self, values: Any, *, ids: Sequence[str], splits: Sequence[str], fold: str = "default"
    ) -> TrainStandardizer:
        import numpy as np

        if self._state is not None:
            raise ContractError(
                "STATE_ALREADY_FITTED", "Create independent state for another fold/trial."
            )
        data = np.asarray(values, dtype=np.float64)
        if (
            data.ndim != 2
            or len(ids) != len(data)
            or len(splits) != len(data)
            or not len(data)
            or set(splits) != {"train"}
        ):
            raise ContractError(
                "SPLIT_LEAKAGE", "Fit requires matching identified train-only rows."
            )
        if len(set(ids)) != len(ids) or not np.isfinite(data).all():
            raise ContractError("INVALID_DATA", "Training ids must be unique and data finite.")
        scale = data.std(axis=0)
        self._state = {
            "state_version": "train-standardizer-v1",
            "fold": fold,
            "train_ids": list(ids),
            "mean": data.mean(axis=0).tolist(),
            "scale": np.where(scale > 0, scale, 1).tolist(),
        }
        return self

    def transform(self, values: Any, *, fold: str = "default") -> Any:
        import numpy as np

        if self._state is None or fold != self._state["fold"]:
            raise ContractError(
                "FOLD_STATE_MISMATCH", "Fit/restore matching training-fold state first."
            )
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(self._state["mean"]):
            raise ContractError("SHAPE_MISMATCH", "Standardizer width differs from training.")
        return (data - self._state["mean"]) / self._state["scale"]

    def state_dict(self) -> dict[str, Any]:
        import copy

        if self._state is None:
            raise ContractError("UNFITTED_STATE", "No fitted state.")
        return copy.deepcopy(self._state)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> TrainStandardizer:
        """Restore portable fold-specific state, never refit on validation data."""
        import copy

        import numpy as np

        if (
            set(state) != {"state_version", "fold", "train_ids", "mean", "scale"}
            or state["state_version"] != "train-standardizer-v1"
        ):
            raise ContractError("CONTRACT_VERSION", "Unsupported standardizer state.")
        mean, scale = np.asarray(state["mean"]), np.asarray(state["scale"])
        if (
            mean.ndim != 1
            or not len(mean)
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ContractError("INVALID_STATE", "Invalid standardizer statistics.")
        if not state["train_ids"] or len(set(state["train_ids"])) != len(state["train_ids"]):
            raise ContractError("INVALID_STATE", "Training ids must be explicit and unique.")
        result = cls()
        result._state = copy.deepcopy(state)
        return result
