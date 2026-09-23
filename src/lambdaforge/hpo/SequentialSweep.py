"""Anytime-valid paired inference for automatically replicated sweeps."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SweepSequentialDecision:
    """One auditable decision made after a complete shared-seed block."""

    stop: bool
    conclusion: str
    blocks: int
    leader: int | None
    radius: float | None
    reason: str
    means: Mapping[int, float]
    policy_version: str = "paired-hoeffding-cs-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "stop": self.stop,
            "conclusion": self.conclusion,
            "complete_shared_seed_blocks": self.blocks,
            "leader": self.leader,
            "simultaneous_radius": self.radius,
            "reason": self.reason,
            "means": {str(key): value for key, value in self.means.items()},
        }


class PairedSweepSequentialAnalyzer:
    """Compare sweep cells with a simultaneous time-uniform Hoeffding bound.

    Each update consumes only seeds observed for every cell.  At look ``n`` the error allocation
    ``alpha / ((K-1) n (n+1))`` is summable over time and competitors, so repeatedly inspecting the
    interval does not turn an ordinary fixed-sample interval into an invalid stopping rule.
    Secondary descriptions remain exploratory; only the returned winner/equivalence conclusions
    drive automatic stopping.
    """

    def __init__(
        self,
        *,
        mode: str,
        bounds: tuple[float, float],
        practical_margin: float | None = None,
        alpha: float = 0.05,
        minimum_blocks: int = 2,
    ) -> None:
        low, high = bounds
        if mode not in {"min", "max"}:
            raise ValueError("Sweep objective mode must be min or max.")
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("Automatic sweep replication requires a finite objective range.")
        if not 0 < alpha < 1:
            raise ValueError("Sequential alpha must be strictly between zero and one.")
        self.mode = mode
        self.low = float(low)
        self.high = float(high)
        self.practical_margin = practical_margin
        self.alpha = alpha
        self.minimum_blocks = minimum_blocks

    def evaluate(
        self, values: Mapping[int, Mapping[int, float]]
    ) -> SweepSequentialDecision:
        candidates = tuple(sorted(values))
        if not candidates:
            return SweepSequentialDecision(False, "COLLECTING", 0, None, None, "No evidence.", {})
        shared = set(values[candidates[0]])
        for candidate in candidates[1:]:
            shared.intersection_update(values[candidate])
        seeds = tuple(sorted(shared))
        means = {
            candidate: statistics.fmean(values[candidate][seed] for seed in seeds)
            for candidate in candidates
            if seeds
        }
        if len(candidates) == 1 and seeds:
            return SweepSequentialDecision(
                True,
                "SINGLE_CELL_COMPLETE",
                len(seeds),
                candidates[0],
                0.0,
                "The fixed design has one cell and its required shared block completed.",
                means,
            )
        if len(seeds) < self.minimum_blocks:
            return SweepSequentialDecision(
                False,
                "COLLECTING",
                len(seeds),
                None,
                None,
                "At least two complete shared-seed blocks are required for sequential inference.",
                means,
            )
        leader = (max if self.mode == "max" else min)(means, key=means.__getitem__)
        n = len(seeds)
        comparisons = max(1, len(candidates) - 1)
        look_alpha = self.alpha / (comparisons * n * (n + 1))
        # A paired difference lies in [-(high-low), +(high-low)], whose range is 2R.
        objective_span = self.high - self.low
        radius = 2.0 * objective_span * math.sqrt(math.log(2.0 / look_alpha) / (2.0 * n))
        sign = 1.0 if self.mode == "max" else -1.0
        gaps = [
            sign * (means[leader] - means[candidate])
            for candidate in candidates
            if candidate != leader
        ]
        if gaps and all(gap - radius > 0.0 for gap in gaps):
            return SweepSequentialDecision(
                True,
                "PREFERRED",
                n,
                leader,
                radius,
                "One cell is better than every competitor under simultaneous anytime-valid bounds.",
                means,
            )
        if self.practical_margin is not None:
            pair_differences = [
                abs(means[left] - means[right])
                for index, left in enumerate(candidates)
                for right in candidates[index + 1 :]
            ]
            if pair_differences and all(
                difference + radius <= self.practical_margin
                for difference in pair_differences
            ):
                return SweepSequentialDecision(
                    True,
                    "PRACTICALLY_EQUIVALENT",
                    n,
                    leader,
                    radius,
                    "All simultaneous paired differences are inside the authored practical margin.",
                    means,
                )
        return SweepSequentialDecision(
            False,
            "UNRESOLVED",
            n,
            leader,
            radius,
            "Another complete shared-seed block can still change the supported conclusion.",
            means,
        )


__all__ = ["PairedSweepSequentialAnalyzer", "SweepSequentialDecision"]
