"""Optional BoTorch Gaussian-process adaptive searcher."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.AdaptiveSearcher import AdaptiveSearcher
from lambdaforge.hpo.LearningCurveModel import LearningCurveModel
from lambdaforge.hpo.SearchSpace import SearchSpace


class BoTorchSearcher(AdaptiveSearcher):
    """Encapsulate probabilistic GP/KG APIs without exposing provider types."""

    def __init__(
        self,
        *,
        seed: int = 0,
        direction: str = "maximize",
        acquisition: str = "knowledge_gradient",
        raw_samples: int = 256,
        num_restarts: int = 10,
        max_budget: int | None = None,
        refresh_interval: int = 1,
    ) -> None:
        if direction not in {"maximize", "minimize"}:
            raise ValueError("BoTorch direction must be maximize or minimize.")
        if raw_samples < 1 or num_restarts < 1 or refresh_interval < 1:
            raise ValueError("BoTorch sampling, restarts and refresh interval must be positive.")
        self.seed = int(seed)
        self.direction = direction
        self.acquisition = acquisition
        self.raw_samples = int(raw_samples)
        self.num_restarts = int(num_restarts)
        self.max_budget = max_budget
        self.refresh_interval = int(refresh_interval)
        self._cached_history: tuple[tuple[object, ...], ...] | None = None
        self._cached_model: Any | None = None
        self._cached_train_y: torch.Tensor | None = None

    def propose(
        self,
        space: SearchSpace,
        state: AdaptiveOptimizerState,
        *,
        count: int = 1,
    ) -> tuple[dict[str, object], ...]:
        """Fit a GP and optimize KG/EI, raising so the controller can record fallback."""
        if count != 1:
            raise ValueError("BoTorchSearcher currently proposes one asynchronous point at a time.")
        observations = tuple(
            observation
            for observation in state.observations
            if observation.score is not None and not observation.oom
        )
        refreshed_count = len(observations) - (len(observations) % self.refresh_interval)
        observations = observations[:refreshed_count]
        config_ids = sorted({observation.config_id for observation in observations})
        if len(config_ids) < max(3, space.dimension + 1):
            raise RuntimeError("Insufficient observations for Bayesian search.")
        try:
            models = importlib.import_module("botorch.models")
            fit = importlib.import_module("botorch.fit")
            mlls = importlib.import_module("gpytorch.mlls")
            analytic = importlib.import_module("botorch.acquisition.analytic")
            kg_module = importlib.import_module("botorch.acquisition.knowledge_gradient")
            logei_module = importlib.import_module("botorch.acquisition.logei")
            optim = importlib.import_module("botorch.optim")
        except ImportError as error:
            raise ImportError(
                "Bayesian adaptive HPO requires the optional 'adaptive-hpo' extra."
            ) from error
        dtype = torch.double
        budget = self.max_budget or max(
            (observation.budget for observation in state.observations), default=1
        )
        training_state = AdaptiveOptimizerState(
            study_fingerprint=state.study_fingerprint,
            controller_seed=state.controller_seed,
            configurations=state.configurations,
            observations=observations,
        )
        curve_model = LearningCurveModel()
        train_x = torch.tensor(
            [space.encode(state.configurations[config_id]) for config_id in config_ids],
            dtype=dtype,
        )
        sign = 1.0 if self.direction == "maximize" else -1.0
        scores: list[list[float]] = []
        for config_id in config_ids:
            prediction = curve_model.predict_configuration(
                training_state, config_id, max_budget=budget
            )
            scores.append([sign * prediction.mean])
        train_y = torch.tensor(scores, dtype=dtype)
        history = (
            ("context", budget, space.dimension, tuple(config_ids), self.direction),
            *(
                (
                    observation.action_id,
                    observation.config_id,
                    observation.seed,
                    observation.budget,
                    observation.score,
                    observation.curve,
                )
                for observation in observations
            ),
        )
        if history != self._cached_history:
            model = models.SingleTaskGP(train_x, train_y)
            mll = mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            fit.fit_gpytorch_mll(mll)
            self._cached_history = history
            self._cached_model = model
            self._cached_train_y = train_y
        else:
            assert self._cached_model is not None and self._cached_train_y is not None
            model = self._cached_model
            train_y = self._cached_train_y
        torch.manual_seed(self.seed + state.decision_index)
        bounds = torch.stack(
            (torch.zeros(space.dimension, dtype=dtype), torch.ones(space.dimension, dtype=dtype))
        )
        if self.acquisition == "knowledge_gradient":
            posterior_mean = analytic.PosteriorMean(model)
            _, current_value = optim.optimize_acqf(
                posterior_mean,
                bounds=bounds,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
            acquisition: Any = kg_module.qKnowledgeGradient(
                model, num_fantasies=32, current_value=current_value.max()
            )
        else:
            acquisition = logei_module.qLogExpectedImprovement(model, best_f=train_y.max())
        pending = [space.encode(action.parameters) for action in state.pending_actions.values()]
        if pending:
            acquisition.set_X_pending(torch.tensor(pending, dtype=dtype))
        candidate, _ = optim.optimize_acqf(
            acquisition,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        values = space.decode(candidate.detach().cpu().reshape(-1).tolist())
        if space.identifier(values) in state.configurations:
            raise RuntimeError("Bayesian acquisition proposed an existing configuration.")
        return (values,)
