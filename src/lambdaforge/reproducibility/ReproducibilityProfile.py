"""Explicit reproducibility policy."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ReproducibilityProfile:
    """Apply documented fast/repeatable/strict deterministic behavior."""

    name: str = "repeatable"
    seed: int = 0
    deterministic_algorithms: bool = False
    cudnn_benchmark: bool = False

    @classmethod
    def named(cls, name: str, *, seed: int = 0) -> ReproducibilityProfile:
        """Create one built-in policy by name."""
        profiles = {
            "fast": cls("fast", seed, False, True),
            "repeatable": cls("repeatable", seed, False, False),
            "strict": cls("strict", seed, True, False),
        }
        if name not in profiles:
            raise ValueError("Reproducibility profile must be fast, repeatable or strict.")
        return profiles[name]

    def apply(self) -> None:
        """Seed supported RNGs and configure deterministic torch behavior."""
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(self.deterministic_algorithms)
        torch.backends.cudnn.benchmark = self.cudnn_benchmark

    @staticmethod
    def fingerprints(
        config: Mapping[str, Any],
        *,
        infrastructure_paths: Sequence[str] = (
            "execution",
            "trainer.devices",
            "trainer.accelerator",
        ),
    ) -> dict[str, str]:
        """Return full infrastructure and science-only fingerprints."""
        complete = json.loads(json.dumps(config))
        scientific = json.loads(json.dumps(config))
        for path in infrastructure_paths:
            parts = path.split(".")
            node: Any = scientific
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    node = None
                    break
                node = node[part]
            if isinstance(node, dict):
                node.pop(parts[-1], None)

        def digest(value: Any) -> str:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            return "sha256:" + hashlib.sha256(encoded).hexdigest()

        return {"scientific": digest(scientific), "infrastructure": digest(complete)}
