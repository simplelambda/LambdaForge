"""Reproducible provider-neutral random search."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.hpo.Trial import Trial


class RandomSearch:
    """Sample explicit search distributions using a private seeded RNG."""

    def __init__(self, space: Mapping[str, Mapping[str, Any]], *, seed: int = 0) -> None:
        self.space = {str(path): dict(spec) for path, spec in space.items()}
        self.seed = seed

    def trials(self, count: int) -> tuple[Trial, ...]:
        """Return deterministic unique trials with conditional parameters."""
        if count < 1:
            raise ValueError("Random search trial count must be positive.")
        rng = random.Random(self.seed)
        output: list[Trial] = []
        seen: set[str] = set()
        attempts = 0
        while len(output) < count and attempts < count * 100:
            attempts += 1
            parameters: dict[str, Any] = {}
            for path, spec in self.space.items():
                condition = spec.get("when")
                if condition is not None:
                    if not isinstance(condition, Mapping) or any(
                        parameters.get(key) != value for key, value in condition.items()
                    ):
                        continue
                parameters[path] = self._sample(spec, rng)
            encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
            fingerprint = "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                output.append(Trial(len(output), parameters, self.seed, fingerprint))
        if len(output) != count:
            raise ValueError("Search space cannot produce the requested number of unique trials.")
        return tuple(output)

    def materialize(self, base: Mapping[str, Any], count: int) -> tuple[dict[str, Any], ...]:
        """Apply sampled dotted paths to independent configuration copies."""
        output: list[dict[str, Any]] = []
        for trial in self.trials(count):
            config = json.loads(json.dumps(base))
            for path, value in trial.parameters.items():
                ExperimentConfig.set_value(config, path, value)
            config.setdefault("extensions", {})["hpo_trial"] = {
                "number": trial.number,
                "seed": trial.seed,
                "fingerprint": trial.fingerprint,
                "parameters": dict(trial.parameters),
            }
            output.append(config)
        return tuple(output)

    @staticmethod
    def _sample(spec: Mapping[str, Any], rng: random.Random) -> Any:
        kind = spec.get("type", "choice")
        if kind == "choice":
            values = spec.get("values")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
                raise ValueError("Choice search parameters require a non-empty values sequence.")
            return values[rng.randrange(len(values))]
        low, high = float(spec["low"]), float(spec["high"])
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("Search bounds must be finite with high > low.")
        if kind == "uniform":
            return rng.uniform(low, high)
        if kind == "loguniform":
            if low <= 0:
                raise ValueError("Log-uniform search requires low > 0.")
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        if kind == "int":
            return rng.randint(math.ceil(low), math.floor(high))
        raise ValueError(f"Unsupported random search distribution: {kind!r}.")
