"""Optional Optuna study adapter."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any


class OptunaSearch:
    """Create reproducible TPE studies with optional ASHA/Hyperband-style pruning."""

    def __init__(
        self,
        *,
        study_name: str,
        storage: str | None = None,
        seed: int = 0,
        direction: str = "minimize",
        pruner: str | None = None,
    ) -> None:
        self.study_name = study_name
        self.storage = storage
        self.seed = seed
        self.direction = direction
        self.pruner = pruner

    def optimize(self, objective: Callable[[Any], float], *, trials: int) -> Any:
        """Run an Optuna study while leaving scheduling/storage choices explicit."""
        if trials < 1:
            raise ValueError("Optuna trial count must be positive.")
        try:
            optuna = importlib.import_module("optuna")
        except ImportError as error:
            raise ImportError("OptunaSearch requires the optional 'optuna' package.") from error
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruners: Mapping[str, Any] = {
            "asha": optuna.pruners.SuccessiveHalvingPruner,
            "hyperband": optuna.pruners.HyperbandPruner,
        }
        if self.pruner is not None and self.pruner not in pruners:
            raise ValueError("Optuna pruner must be 'asha', 'hyperband' or null.")
        pruner = pruners[self.pruner]() if self.pruner else optuna.pruners.NopPruner()
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(objective, n_trials=trials)
        return study
