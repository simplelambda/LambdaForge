"""Lazy mature classical estimators; not differentiable torch modules or another trainer."""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from lambdaforge.catalog.errors import ContractError


class SklearnEstimator:
    """Explicit backend/capabilities, seed and thread policy; call inside Work.run.

    No automatic scaling/splitting or implicit GPU use. Save/load uses optional skops rather
    than pickle; the caller must approve exactly the untrusted types returned by inspect_file.
    Native Work checkpoints/outputs own the resulting file and HPO owns selected parameters.
    """

    ALGORITHMS = {
        "decision_tree_classifier": ("sklearn.tree", "DecisionTreeClassifier", True, True),
        "decision_tree_regressor": ("sklearn.tree", "DecisionTreeRegressor", True, False),
        "random_forest_classifier": ("sklearn.ensemble", "RandomForestClassifier", True, True),
        "random_forest_regressor": ("sklearn.ensemble", "RandomForestRegressor", True, False),
        "gradient_boosting_classifier": (
            "sklearn.ensemble",
            "GradientBoostingClassifier",
            True,
            True,
        ),
        "gradient_boosting_regressor": (
            "sklearn.ensemble",
            "GradientBoostingRegressor",
            True,
            False,
        ),
        "svc": ("sklearn.svm", "SVC", True, False),
        "svr": ("sklearn.svm", "SVR", False, False),
    }

    def __init__(
        self,
        algorithm: str,
        parameters: dict[str, Any] | None = None,
        seed: int = 0,
        threads: int = 1,
    ) -> None:
        if (
            algorithm not in self.ALGORITHMS
            or type(threads) is not int
            or threads < 1
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise ContractError("INVALID_ESTIMATOR", "Unknown algorithm or invalid thread limit.")
        self.algorithm, self.parameters, self.seed, self.threads = (
            algorithm,
            copy.deepcopy(dict(parameters or {})),
            seed,
            threads,
        )
        self._estimator: Any = None
        try:
            json.dumps(self.parameters, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ContractError(
                "INVALID_PARAMETERS", "Estimator parameters must be JSON."
            ) from error

    def provenance(self) -> dict[str, Any]:
        """Project backend identity without importing sklearn or fitting an estimator."""
        try:
            version = importlib.metadata.version("scikit-learn")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return {
            "backend": "scikit-learn",
            "backend_version": version,
            "algorithm": self.algorithm,
            "parameters": copy.deepcopy(self.parameters),
            "seed": self.seed,
            "threads": self.threads,
            "capabilities": self.capabilities,
        }

    @property
    def capabilities(self) -> dict[str, Any]:
        spec = self.ALGORITHMS[self.algorithm]
        return {
            "fit": True,
            "predict": True,
            "predict_proba": spec[3]
            or (self.algorithm == "svc" and self.parameters.get("probability") is True),
            "transform": False,
            "differentiable": False,
            "device": "cpu",
        }

    def fit(self, x: Any, y: Any) -> SklearnEstimator:
        try:
            module, name, seeded, _ = self.ALGORITHMS[self.algorithm]
            cls = getattr(importlib.import_module(module), name)
        except ImportError as error:
            raise ContractError("MISSING_DEPENDENCY", "Install lambdaforge[estimators].") from error
        parameters = dict(self.parameters)
        if "random_state" in parameters or "n_jobs" in parameters:
            raise ContractError(
                "RESOURCE_OVERRIDE", "Use explicit seed/threads, not hidden backend overrides."
            )
        if seeded:
            parameters["random_state"] = self.seed
        if self.algorithm.startswith("random_forest"):
            parameters["n_jobs"] = self.threads
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=self.threads):
            self._estimator = cls(**parameters).fit(x, y)
        return self

    def predict(self, x: Any) -> Any:
        if self._estimator is None:
            raise ContractError("UNFITTED_STATE", "Fit/restore estimator first.")
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=self.threads):
            return self._estimator.predict(x)

    def predict_proba(self, x: Any) -> Any:
        if not self.capabilities["predict_proba"]:
            raise ContractError(
                "UNSUPPORTED_CAPABILITY", "This estimator does not expose predict_proba."
            )
        if self._estimator is None:
            raise ContractError("UNFITTED_STATE", "Fit/restore estimator first.")
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=self.threads):
            return self._estimator.predict_proba(x)

    def save(self, file: str | Path) -> None:
        if self._estimator is None:
            raise ContractError("UNFITTED_STATE", "No fitted estimator to save.")
        try:
            sio = importlib.import_module("skops.io")
        except ImportError as error:
            raise ContractError(
                "MISSING_DEPENDENCY", "Install lambdaforge[estimator-persistence]."
            ) from error
        if Path(file).exists():
            raise FileExistsError("Use a new managed checkpoint/output file.")
        sio.dump(
            {
                "estimator": self._estimator,
                "algorithm": self.algorithm,
                "parameters": self.parameters,
                "seed": self.seed,
                "threads": self.threads,
                "state_version": "sklearn-estimator-v1",
                "provenance": self.provenance(),
            },
            file,
        )

    @staticmethod
    def inspect_file(file: str | Path) -> list[str]:
        """Report skops types requiring explicit trust; never load pickle."""
        try:
            sio = importlib.import_module("skops.io")
        except ImportError as error:
            raise ContractError(
                "MISSING_DEPENDENCY", "Install lambdaforge[estimator-persistence]."
            ) from error
        return list(sio.get_untrusted_types(file=file))

    @classmethod
    def load(cls, file: str | Path, *, trusted_types: list[str]) -> SklearnEstimator:
        """Restore only after explicit review of the exact types reported by inspect_file."""
        import sklearn

        del sklearn  # Require the optional backend before deserializing its state.
        if set(trusted_types) != set(cls.inspect_file(file)):
            raise ContractError("UNTRUSTED_TYPES", "Review the exact unknown types before loading.")
        sio = importlib.import_module("skops.io")
        state = sio.load(file, trusted=trusted_types)
        if not isinstance(state, dict) or state.get("state_version") != "sklearn-estimator-v1":
            raise ContractError("CONTRACT_VERSION", "Unsupported estimator state.")
        result = cls(state["algorithm"], state["parameters"], state["seed"], state["threads"])
        module, name, _, _ = result.ALGORITHMS[result.algorithm]
        if not isinstance(state["estimator"], getattr(importlib.import_module(module), name)):
            raise ContractError(
                "INVALID_STATE", "Estimator type differs from its declared algorithm."
            )
        result._estimator = state["estimator"]
        return result
