"""Optional mixed, multi-fidelity BoTorch adaptive searcher."""

from __future__ import annotations

import importlib
import math
from contextlib import nullcontext
from typing import Any

import torch

from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.AdaptiveSearcher import AdaptiveSearcher
from lambdaforge.hpo.SearchSpace import SearchSpace


class BoTorchSearcher(AdaptiveSearcher):
    """Fit ``f(x, b)`` and optimize asynchronous multi-fidelity KG or log-EI.

    Numeric-only spaces use BoTorch's fidelity kernel. Mixed spaces use
    ``MixedSingleTaskGP`` so unordered dimensions have Hamming rather than interval geometry;
    normalized budget remains an explicit continuous fidelity feature. All observed curve points,
    not heuristic final projections, train the model.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        direction: str = "maximize",
        acquisition: str = "knowledge_gradient",
        raw_samples: int = 256,
        num_restarts: int = 10,
        max_budget: int | None = None,
        min_budget: int = 1,
        budget_step: int = 1,
        refresh_interval: int = 1,
    ) -> None:
        if direction not in {"maximize", "minimize"}:
            raise ValueError("BoTorch direction must be maximize or minimize.")
        if acquisition not in {"knowledge_gradient", "expected_improvement"}:
            raise ValueError(
                "BoTorch acquisition must be knowledge_gradient or expected_improvement."
            )
        if (
            raw_samples < 1
            or num_restarts < 1
            or min_budget < 1
            or budget_step < 1
            or refresh_interval < 1
        ):
            raise ValueError("BoTorch sampling, fidelity and refresh settings must be positive.")
        self.seed = int(seed)
        self.direction = direction
        self.acquisition = acquisition
        self.raw_samples = int(raw_samples)
        self.num_restarts = int(num_restarts)
        self.max_budget = max_budget
        self.min_budget = int(min_budget)
        self.budget_step = int(budget_step)
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
        """Propose one novel point, letting the controller record a safe fallback on failure."""
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
        provider = self._provider()
        dtype = torch.double
        budget = self.max_budget or max(
            (observation.budget for observation in observations), default=1
        )
        points: dict[tuple[str, int, int], float] = {}
        for observation in observations:
            for fidelity, score in observation.curve:
                if 0 <= fidelity <= budget and math.isfinite(score):
                    points[(observation.config_id, observation.seed, fidelity)] = score
            if observation.score is not None:
                points[(observation.config_id, observation.seed, observation.budget)] = (
                    observation.score
                )
        sign = 1.0 if self.direction == "maximize" else -1.0
        train_x = torch.tensor(
            [
                [*space.encode(state.configurations[config_id]), fidelity / budget]
                for config_id, _, fidelity in points
            ],
            dtype=dtype,
        )
        train_y = torch.tensor(
            [[sign * score] for score in points.values()],
            dtype=dtype,
        )
        history = (
            (
                "context",
                budget,
                space.model_dimension,
                space.categorical_indices,
                tuple(config_ids),
                self.direction,
            ),
            *(
                (config_id, seed, fidelity, score)
                for (config_id, seed, fidelity), score in points.items()
            ),
        )
        if history != self._cached_history:
            model = self._fit_with_retry(provider, space, train_x, train_y)
            self._cached_history = history
            self._cached_model = model
            self._cached_train_y = train_y
        else:
            assert self._cached_model is not None and self._cached_train_y is not None
            model = self._cached_model
            train_y = self._cached_train_y

        torch.manual_seed(self.seed + state.decision_index)
        lower, upper = space.model_bounds()
        bounds = torch.tensor(
            [[*lower, 0.0], [*upper, 1.0]],
            dtype=dtype,
        )
        fidelity_index = space.model_dimension
        pending = [
            [*space.encode(action.parameters), action.target_budget / budget]
            for action in state.pending_actions.values()
        ]
        acquisition = self._acquisition(
            provider,
            model,
            train_x,
            train_y,
            bounds,
            fidelity_index,
            pending,
        )
        fixed_features = self._fixed_features(space, fidelity_index, budget)
        candidate, _ = provider["optim"].optimize_acqf_mixed(
            acq_function=acquisition,
            bounds=bounds,
            fixed_features_list=fixed_features,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
        values = space.decode_model(
            candidate.detach().cpu().reshape(-1).tolist()[: space.model_dimension]
        )
        if space.identifier(values) in state.configurations:
            raise RuntimeError("Bayesian acquisition proposed an existing configuration.")
        return (values,)

    @staticmethod
    def _provider() -> dict[str, Any]:
        try:
            return {
                "models": importlib.import_module("botorch.models"),
                "fidelity_models": importlib.import_module("botorch.models.gp_regression_fidelity"),
                "transforms": importlib.import_module("botorch.models.transforms.outcome"),
                "fit": importlib.import_module("botorch.fit"),
                "mlls": importlib.import_module("gpytorch.mlls"),
                "settings": importlib.import_module("gpytorch.settings"),
                "acquisition": importlib.import_module("botorch.acquisition"),
                "fixed": importlib.import_module("botorch.acquisition.fixed_feature"),
                "kg": importlib.import_module("botorch.acquisition.knowledge_gradient"),
                "logei": importlib.import_module("botorch.acquisition.logei"),
                "cost": importlib.import_module("botorch.models.cost"),
                "cost_aware": importlib.import_module("botorch.acquisition.cost_aware"),
                "utils": importlib.import_module("botorch.acquisition.utils"),
                "optim": importlib.import_module("botorch.optim"),
            }
        except ImportError as error:
            raise ImportError(
                "Bayesian adaptive HPO requires the optional 'adaptive-hpo' extra."
            ) from error

    def _fit_with_retry(
        self,
        provider: dict[str, Any],
        space: SearchSpace,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> Any:
        last_error: Exception | None = None
        for attempt, jitter in enumerate((None, 1e-4)):
            try:
                outcome = provider["transforms"].Standardize(m=1)
                if space.categorical_indices:
                    model = provider["models"].MixedSingleTaskGP(
                        train_x,
                        train_y,
                        cat_dims=list(space.categorical_indices),
                        outcome_transform=outcome,
                    )
                else:
                    model = provider["fidelity_models"].SingleTaskMultiFidelityGP(
                        train_x,
                        train_y,
                        data_fidelities=[space.model_dimension],
                        outcome_transform=outcome,
                    )
                mll = provider["mlls"].ExactMarginalLogLikelihood(model.likelihood, model)
                context = (
                    provider["settings"].cholesky_jitter(jitter)
                    if jitter is not None
                    else nullcontext()
                )
                with context:
                    provider["fit"].fit_gpytorch_mll(
                        mll,
                        optimizer_kwargs={"options": {"maxiter": 100 if attempt else 200}},
                    )
                return model
            except Exception as error:
                last_error = error
        assert last_error is not None
        raise RuntimeError(
            f"BoTorch surrogate failed after safe-numerics retry: {type(last_error).__name__}: "
            f"{last_error}"
        ) from last_error

    def _acquisition(
        self,
        provider: dict[str, Any],
        model: Any,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        bounds: torch.Tensor,
        fidelity_index: int,
        pending: list[list[float]],
    ) -> Any:
        x_pending = torch.tensor(pending, dtype=torch.double) if pending else None
        if self.acquisition == "knowledge_gradient":
            target = {fidelity_index: 1.0}
            projected = provider["utils"].project_to_target_fidelity(
                X=train_x,
                target_fidelities=target,
            )
            current_value = model.posterior(projected).mean.max()
            cost_model = provider["cost"].AffineFidelityCostModel(
                fidelity_weights={fidelity_index: 1.0},
                fixed_cost=max(1e-6, self.min_budget / max(1, self.max_budget or self.min_budget)),
            )
            cost_utility = provider["cost_aware"].InverseCostWeightedUtility(cost_model=cost_model)
            return provider["kg"].qMultiFidelityKnowledgeGradient(
                model=model,
                num_fantasies=32,
                current_value=current_value,
                cost_aware_utility=cost_utility,
                project=lambda value: provider["utils"].project_to_target_fidelity(
                    X=value,
                    target_fidelities=target,
                ),
                X_pending=x_pending,
            )
        return provider["logei"].qLogExpectedImprovement(
            model,
            best_f=train_y.max(),
            X_pending=x_pending,
        )

    def _fixed_features(
        self,
        space: SearchSpace,
        fidelity_index: int,
        max_budget: int,
    ) -> list[dict[int, float]]:
        levels = list(range(self.min_budget, max_budget + 1, self.budget_step))
        if not levels or levels[-1] != max_budget:
            levels.append(max_budget)
        categorical = space.categorical_assignments() or ({},)
        return [
            {**assignment, fidelity_index: level / max_budget}
            for assignment in categorical
            for level in levels
        ]
