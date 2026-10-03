"""Small unambiguous registry for objective shorthand and known bounded ranges."""

from __future__ import annotations

from typing import Any


class MetricRegistry:
    """Resolve only standard metrics whose direction and range are unambiguous."""

    _STANDARD: dict[str, tuple[str, tuple[float, float]]] = {
        "accuracy": ("max", (0.0, 1.0)),
        "balanced_accuracy": ("max", (0.0, 1.0)),
        "auprc": ("max", (0.0, 1.0)),
        "auroc": ("max", (0.0, 1.0)),
        "f1": ("max", (0.0, 1.0)),
        "precision": ("max", (0.0, 1.0)),
        "recall": ("max", (0.0, 1.0)),
        "mcc": ("max", (-1.0, 1.0)),
        "kappa": ("max", (-1.0, 1.0)),
        "loss": ("min", (0.0, float("inf"))),
    }

    @classmethod
    def resolve(cls, name: str) -> dict[str, Any] | None:
        base = name
        for prefix in ("train_", "val_", "validation_", "test_"):
            if base.startswith(prefix):
                base = base[len(prefix) :]
                break
        value = cls._STANDARD.get(base)
        if value is None:
            return None
        mode, bounds = value
        result: dict[str, Any] = {"metric": name, "mode": mode}
        if all(bound != float("inf") for bound in bounds):
            result["range"] = list(bounds)
        return result

    @classmethod
    def catalog_defaults(cls) -> dict[str, dict[str, Any]]:
        """Snapshot known directions/ranges for immutable analysis provenance."""
        return {
            name: definition
            for prefix in ("", "train_", "val_", "validation_", "test_")
            for base in cls._STANDARD
            if (definition := cls.resolve(name := prefix + base)) is not None
        }


__all__ = ["MetricRegistry"]
