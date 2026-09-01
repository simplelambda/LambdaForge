"""Optional BoTorch acquisition over LambdaForge's deterministic candidate pool."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from lambdaforge.hpo.AdaptiveSampler import CandidateObservation
from lambdaforge.hpo.SurvivalModel import SurvivalEstimate


@dataclass(frozen=True, slots=True)
class _EncodedColumn:
    name: str
    numeric_bounds: tuple[float, float] | None
    categories: Mapping[str, int] | None
    conditional: bool


class BayesianSampler:
    """Fit a noise-aware mixed GP and rank finite candidates by joint qLogNEI.

    LambdaForge owns candidate identity and mixed-value encoding; BoTorch remains an optional
    mathematical provider. Conditional inactivity is represented by an explicit activity feature,
    never by pretending that an inactive value is an ordinary category.
    """

    def __init__(self, candidates: Mapping[int, Mapping[str, Any]], *, mode: str) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("Bayesian sampler mode must be min or max.")
        self.candidates = {
            int(trial): {str(name): value for name, value in parameters.items()}
            for trial, parameters in candidates.items()
        }
        self.mode = mode
        self._vectors, self._categorical = self._encode(self.candidates)

    @staticmethod
    def available() -> bool:
        """Return whether the optional provider imports cleanly."""
        try:
            importlib.import_module("botorch")
            importlib.import_module("gpytorch")
        except ImportError:
            return False
        return True

    def propose(
        self,
        observations: Sequence[CandidateObservation],
        *,
        selected: Sequence[int],
        pending: Sequence[int] = (),
        pending_fidelity: Mapping[int, float] | None = None,
        censored: Sequence[int] = (),
        survival: Mapping[int, SurvivalEstimate] | None = None,
        count: int,
    ) -> tuple[int, ...]:
        """Rank unseen pool members using a fresh exact GP posterior."""
        available = [trial for trial in sorted(self.candidates) if trial not in set(selected)]
        observed = [item for item in observations if item.trial in self._vectors]
        censored_trials = [trial for trial in censored if trial in self._vectors]
        dimension = len(next(iter(self._vectors.values()), ())) + 1
        if count < 1 or not available:
            return ()
        if len(observed) < max(4, dimension + 1):
            raise RuntimeError("Insufficient observations for Bayesian proposal.")
        provider = self._provider()
        train_x = torch.tensor(
            [(*self._vectors[item.trial], min(1.0, max(0.0, item.fidelity))) for item in observed],
            dtype=torch.double,
        )
        sign = 1.0 if self.mode == "max" else -1.0
        train_y = torch.tensor(
            [[sign * float(item.value)] for item in observed], dtype=torch.double
        )
        standard_errors = [item.standard_error for item in observed]
        train_yvar = (
            torch.tensor(
                [[max(float(error) ** 2, 1e-10)] for error in standard_errors if error is not None],
                dtype=torch.double,
            )
            if all(
                error is not None and math.isfinite(float(error)) and float(error) >= 0
                for error in standard_errors
            )
            else None
        )
        outcome = provider["Standardize"](m=1)
        model_kwargs = {
            "train_X": train_x,
            "train_Y": train_y,
            "train_Yvar": train_yvar,
            "outcome_transform": outcome,
        }
        if self._categorical:
            model = provider["MixedSingleTaskGP"](
                cat_dims=list(self._categorical),
                **model_kwargs,
            )
        else:
            model = provider["SingleTaskMultiFidelityGP"](
                data_fidelities=[dimension - 1],
                **model_kwargs,
            )
        mll = provider["ExactMarginalLogLikelihood"](model.likelihood, model)
        provider["fit_gpytorch_mll"](mll)
        proposed: list[int] = []
        with torch.no_grad():
            while available and len(proposed) < count:
                pending_points = [*pending, *proposed]
                pending_tensor = (
                    torch.tensor(
                        [
                            (
                                *self._vectors[trial],
                                min(
                                    1.0,
                                    max(
                                        0.0,
                                        float((pending_fidelity or {}).get(trial, 1.0)),
                                    ),
                                ),
                            )
                            for trial in pending_points
                        ],
                        dtype=torch.double,
                    )
                    if pending_points
                    else None
                )
                acquisition = provider["qLogNoisyExpectedImprovement"](
                    model=model,
                    X_baseline=train_x,
                    X_pending=pending_tensor,
                    prune_baseline=True,
                )
                values = torch.tensor(
                    [(*self._vectors[trial], 1.0) for trial in available], dtype=torch.double
                ).unsqueeze(-2)
                raw_scores = acquisition(values).reshape(-1).tolist()
                finite = [float(score) for score in raw_scores if math.isfinite(float(score))]
                low, high = (min(finite), max(finite)) if finite else (0.0, 0.0)
                span = high - low
                scored = []
                for trial, score in zip(available, raw_scores, strict=True):
                    score = float(score)
                    if not math.isfinite(score):
                        continue
                    if censored_trials or survival is not None:
                        normalized = (score - low) / span if span > 1e-12 else 0.5
                        if survival is not None and trial in survival:
                            estimate = survival[trial]
                            uncertainty = estimate.upper - estimate.lower
                            score = (
                                normalized + 0.25 * (estimate.probability - 0.5) + 0.1 * uncertainty
                            )
                        elif censored_trials:
                            distance = min(
                                self._distance(trial, censored_trial)
                                for censored_trial in censored_trials
                            )
                            score = normalized - 0.2 * (1 - distance)
                    scored.append((score, -trial, trial))
                chosen = (
                    max(scored)[2]
                    if scored
                    else max(
                        available,
                        key=lambda trial: (
                            min(self._distance(trial, selected) for selected in proposed)
                            if proposed
                            else 0.0,
                            -trial,
                        ),
                    )
                )
                proposed.append(chosen)
                available.remove(chosen)
        return tuple(proposed)

    def _distance(self, left: int, right: int) -> float:
        """Return normalized numeric distance plus categorical Hamming distance."""
        first, second = self._vectors[left], self._vectors[right]
        categorical = set(self._categorical)
        distances = [
            (0.0 if first[index] == second[index] else 1.0)
            if index in categorical
            else min(1.0, abs(first[index] - second[index]))
            for index in range(len(first))
        ]
        return sum(distances) / len(distances) if distances else 0.0

    @staticmethod
    def _provider() -> dict[str, Any]:
        try:
            models = importlib.import_module("botorch.models")
            fidelity_models = importlib.import_module("botorch.models.gp_regression_fidelity")
            transforms = importlib.import_module("botorch.models.transforms.outcome")
            acquisition = importlib.import_module("botorch.acquisition.logei")
            fit = importlib.import_module("botorch.fit")
            mlls = importlib.import_module("gpytorch.mlls")
        except ImportError as error:
            raise ImportError(
                "Bayesian HPO requires 'pip install lambdaforge[adaptive-hpo]'."
            ) from error
        return {
            "MixedSingleTaskGP": models.MixedSingleTaskGP,
            "SingleTaskMultiFidelityGP": fidelity_models.SingleTaskMultiFidelityGP,
            "Standardize": transforms.Standardize,
            "qLogNoisyExpectedImprovement": acquisition.qLogNoisyExpectedImprovement,
            "fit_gpytorch_mll": fit.fit_gpytorch_mll,
            "ExactMarginalLogLikelihood": mlls.ExactMarginalLogLikelihood,
        }

    @staticmethod
    def _encode(
        candidates: Mapping[int, Mapping[str, Any]],
    ) -> tuple[dict[int, tuple[float, ...]], tuple[int, ...]]:
        names = sorted({name for candidate in candidates.values() for name in candidate})
        columns: list[_EncodedColumn] = []
        for name in names:
            present = [candidate[name] for candidate in candidates.values() if name in candidate]
            numeric = bool(present) and all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in present
            )
            conditional = len(present) != len(candidates)
            if numeric:
                values = [float(value) for value in present]
                column = _EncodedColumn(name, (min(values), max(values)), None, conditional)
            else:
                keys = sorted({json.dumps(value, sort_keys=True, default=str) for value in present})
                column = _EncodedColumn(
                    name, None, {key: index for index, key in enumerate(keys)}, conditional
                )
            columns.append(column)
        categorical: list[int] = []
        vectors: dict[int, tuple[float, ...]] = {}
        for trial, candidate in candidates.items():
            vector: list[float] = []
            for column in columns:
                active = column.name in candidate
                if column.numeric_bounds is not None:
                    low, high = column.numeric_bounds
                    width = high - low
                    value = float(candidate[column.name]) if active else low
                    vector.append(0.0 if width <= 0 else (value - low) / width)
                else:
                    assert column.categories is not None
                    key = (
                        json.dumps(candidate[column.name], sort_keys=True, default=str)
                        if active
                        else ""
                    )
                    vector.append(float(column.categories.get(key, 0)))
                    categorical.append(len(vector) - 1)
                if column.conditional:
                    vector.append(1.0 if active else 0.0)
                    categorical.append(len(vector) - 1)
            vectors[int(trial)] = tuple(vector)
        return vectors, tuple(sorted(set(categorical)))


__all__ = ["BayesianSampler"]
