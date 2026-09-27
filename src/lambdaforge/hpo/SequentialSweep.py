"""Anytime-valid paired inference for automatically replicated sweeps."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lambdaforge.hpo.ScientificConclusions import ScientificRelation

_POLICY_VERSION = "paired-pm-eb-cs-v1"


@dataclass(frozen=True, slots=True)
class SweepSequentialDecision:
    """One auditable decision made from complete shared-seed blocks only."""

    stop: bool
    conclusion: str
    blocks: int
    leader: int | None
    radius: float | None
    reason: str
    means: Mapping[int, float]
    relations: tuple[ScientificRelation, ...] = ()
    practical_top_set: tuple[int, ...] = ()
    inferior_set: tuple[int, ...] = ()
    reference: int | None = None
    alpha: float = 0.05
    primary_comparisons: int = 0
    policy_version: str = _POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "stop": self.stop,
            "conclusion": self.conclusion,
            "complete_shared_seed_blocks": self.blocks,
            "leader": self.leader,
            "point_estimate_leader": self.leader,
            "simultaneous_radius": self.radius,
            "reason": self.reason,
            "means": {str(key): value for key, value in self.means.items()},
            "relations": [value.to_dict() for value in self.relations],
            "practical_top_set": list(self.practical_top_set),
            "inferior_set": list(self.inferior_set),
            "reference": self.reference,
            "formal_sequential_evidence": {
                "state": "RESOLVED" if self.stop else "UNRESOLVED",
                "family_alpha": self.alpha,
                "simultaneous_coverage_level": 1.0 - self.alpha,
                "primary_comparisons": self.primary_comparisons,
                "multiplicity_method": "bonferroni-primary-family",
                "method": self.policy_version,
            },
        }


class PairedSweepSequentialAnalyzer:
    """Resolve practical sweep questions with paired PM-EB confidence sequences.

    Seed-matched objective differences are mapped from ``[-R, R]`` to ``[0, 1]`` and
    Theorem 2 of Waudby-Smith & Ramdas, *Estimating means of bounded random variables by
    betting* (JRSS-B 2024, doi:10.1093/jrsssb/qkad009) is applied. Its predictable plug-in
    empirical-Bernstein sequence is

    ``center_t +/- (log(2/alpha) + sum(v_i psi_E(lambda_i))) / sum(lambda_i)``

    with ``v_i=4(X_i-mu_hat_{i-1})^2``, ``psi_E(l)=(-log(1-l)-l)/4`` and the paper's
    variance-adaptive predictable ``lambda_i`` capped at 1/2. The bound is time-uniform, so
    optional stopping is valid. Bonferroni is applied only to the authored primary family:
    treatment-vs-reference when a reference exists, otherwise the pairwise family required by
    practical top-set inference. Observations must share a conditional mean and remain within the
    authored finite bounds; no normality assumption is made.

    The historical Hoeffding interval remains available as a conservative test oracle, not as
    the default policy.
    """

    def __init__(
        self,
        *,
        mode: str,
        bounds: tuple[float, float],
        practical_margin: float | None = None,
        alpha: float = 0.05,
        minimum_blocks: int = 2,
        reference: int | None = None,
    ) -> None:
        low, high = bounds
        if mode not in {"min", "max"}:
            raise ValueError("Sweep objective mode must be min or max.")
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("Automatic sweep replication requires a finite objective range.")
        if practical_margin is not None and (
            not math.isfinite(practical_margin) or practical_margin < 0
        ):
            raise ValueError("practical_margin must be finite and non-negative.")
        if not 0 < alpha < 1:
            raise ValueError("Sequential alpha must be strictly between zero and one.")
        self.mode = mode
        self.low = float(low)
        self.high = float(high)
        self.practical_margin = practical_margin
        self.alpha = alpha
        self.minimum_blocks = minimum_blocks
        self.reference = reference

    def evaluate(
        self, values: Mapping[int, Mapping[int, float]]
    ) -> SweepSequentialDecision:
        candidates = tuple(sorted(values))
        if not candidates:
            return self._decision(False, "COLLECTING", 0, None, None, "No evidence.", {})
        if self.reference is not None and self.reference not in candidates:
            raise ValueError("The authored sweep reference does not identify a candidate.")
        shared = set(values[candidates[0]])
        for candidate in candidates[1:]:
            shared.intersection_update(values[candidate])
        seeds = tuple(sorted(shared))
        self._validate_observations(values, candidates, seeds)
        means = {
            candidate: statistics.fmean(values[candidate][seed] for seed in seeds)
            for candidate in candidates
            if seeds
        }
        if len(candidates) == 1 and seeds:
            return self._decision(
                True,
                "SINGLE_CELL_COMPLETE",
                len(seeds),
                candidates[0],
                0.0,
                "The fixed design has one cell and its required shared block completed.",
                means,
                top=(candidates[0],),
            )
        if len(seeds) < self.minimum_blocks:
            return self._decision(
                False,
                "COLLECTING",
                len(seeds),
                None,
                None,
                f"At least {self.minimum_blocks} complete shared-seed blocks are required for "
                "sequential inference.",
                means,
            )
        leader = (max if self.mode == "max" else min)(means, key=means.__getitem__)
        primary_pairs = self._primary_pairs(candidates)
        pair_alpha = self.alpha / max(1, len(primary_pairs))
        sign = 1.0 if self.mode == "max" else -1.0
        relations: list[ScientificRelation] = []
        radii: list[float] = []
        for left, right in primary_pairs:
            differences = [
                sign * (values[left][seed] - values[right][seed]) for seed in seeds
            ]
            lower, upper = self._pm_eb_interval(differences, pair_alpha)
            estimate = statistics.fmean(differences)
            relation = self._relation(lower, upper)
            relations.append(
                ScientificRelation(
                    left,
                    right,
                    relation,
                    estimate,
                    lower,
                    upper,
                    self.practical_margin,
                )
            )
            radii.append(max(estimate - lower, upper - estimate))
        radius = max(radii, default=0.0)
        if self.reference is not None:
            return self._reference_decision(means, seeds, leader, radius, tuple(relations))
        return self._top_set_decision(means, seeds, leader, radius, tuple(relations))

    def _reference_decision(
        self,
        means: Mapping[int, float],
        seeds: Sequence[int],
        leader: int,
        radius: float,
        relations: tuple[ScientificRelation, ...],
    ) -> SweepSequentialDecision:
        assert self.reference is not None
        if all(value.resolved for value in relations):
            equivalent = tuple(
                value.left for value in relations if value.relation == "PRACTICALLY_EQUIVALENT"
            )
            top = tuple(
                sorted(
                    {
                        self.reference,
                        *(
                            value.left
                            for value in relations
                            if value.relation
                            in {"MATERIALLY_BETTER", "PRACTICALLY_EQUIVALENT"}
                        ),
                    }
                )
            )
            if self.practical_margin is not None and len(equivalent) == len(relations):
                conclusion = "ALL_PRACTICALLY_EQUIVALENT"
            elif len(relations) == 1:
                conclusion = {
                    "MATERIALLY_BETTER": "BETTER_THAN_REFERENCE",
                    "MATERIALLY_WORSE": "WORSE_THAN_REFERENCE",
                    "PRACTICALLY_EQUIVALENT": "EQUIVALENT_TO_REFERENCE",
                }[relations[0].relation]
            else:
                conclusion = "PARTIALLY_ORDERED"
            return self._decision(
                True,
                conclusion,
                len(seeds),
                leader,
                radius,
                "Every treatment-versus-reference question is resolved under one simultaneous "
                "anytime-valid family.",
                means,
                relations=relations,
                top=top,
                inferior=tuple(
                    value.left for value in relations if value.relation == "MATERIALLY_WORSE"
                ),
            )
        return self._unresolved(means, seeds, leader, radius, relations)

    def _top_set_decision(
        self,
        means: Mapping[int, float],
        seeds: Sequence[int],
        leader: int,
        radius: float,
        relations: tuple[ScientificRelation, ...],
    ) -> SweepSequentialDecision:
        candidates = tuple(sorted(means))
        directed: dict[tuple[int, int], str] = {}
        inverse = {
            "MATERIALLY_BETTER": "MATERIALLY_WORSE",
            "MATERIALLY_WORSE": "MATERIALLY_BETTER",
        }
        for value in relations:
            directed[(int(value.left), int(value.right))] = value.relation
            directed[(int(value.right), int(value.left))] = inverse.get(
                value.relation, value.relation
            )
        inferior = {
            candidate
            for candidate in candidates
            if any(
                directed.get((candidate, other)) == "MATERIALLY_WORSE"
                for other in candidates
                if other != candidate
            )
        }
        top = tuple(candidate for candidate in candidates if candidate not in inferior)
        top_resolved = bool(top) and all(
            directed.get((left, right)) in {"MATERIALLY_BETTER", "PRACTICALLY_EQUIVALENT"}
            for left in top
            for right in top
            if left != right
        )
        inferior_resolved = all(
            any(
                directed.get((candidate, superior)) == "MATERIALLY_WORSE"
                for superior in top
            )
            for candidate in inferior
        )
        if top_resolved and inferior_resolved and len(top) + len(inferior) == len(candidates):
            conclusion = (
                "PREFERRED"
                if len(top) == 1
                else "ALL_PRACTICALLY_EQUIVALENT"
                if len(top) == len(candidates)
                else "PRACTICAL_TOP_SET"
            )
            return self._decision(
                True,
                conclusion,
                len(seeds),
                leader,
                radius,
                "Practical top-set membership is stable under simultaneous anytime-valid paired "
                "relations; ordering within an equivalent top set is unnecessary.",
                means,
                relations=relations,
                top=top,
                inferior=tuple(sorted(inferior)),
            )
        return self._unresolved(means, seeds, leader, radius, relations, top=top)

    def _unresolved(
        self,
        means: Mapping[int, float],
        seeds: Sequence[int],
        leader: int,
        radius: float,
        relations: tuple[ScientificRelation, ...],
        *,
        top: tuple[int, ...] = (),
    ) -> SweepSequentialDecision:
        reason = (
            "No practical_margin is defined; LambdaForge is resolving exact ordering, which may "
            "require substantially more replication."
            if self.practical_margin is None
            else "At least one primary practical relation can still change the supported "
            "conclusion; another complete shared-seed block is informative."
        )
        return self._decision(
            False,
            "UNRESOLVED",
            len(seeds),
            leader,
            radius,
            reason,
            means,
            relations=relations,
            top=top,
        )

    def _decision(
        self,
        stop: bool,
        conclusion: str,
        blocks: int,
        leader: int | None,
        radius: float | None,
        reason: str,
        means: Mapping[int, float],
        *,
        relations: tuple[ScientificRelation, ...] = (),
        top: tuple[int, ...] = (),
        inferior: tuple[int, ...] = (),
    ) -> SweepSequentialDecision:
        return SweepSequentialDecision(
            stop=stop,
            conclusion=conclusion,
            blocks=blocks,
            leader=leader,
            radius=radius,
            reason=reason,
            means=means,
            relations=relations,
            practical_top_set=top,
            inferior_set=inferior,
            reference=self.reference,
            alpha=self.alpha,
            primary_comparisons=(
                len(self._primary_pairs(tuple(sorted(means)))) if len(means) > 1 else 0
            ),
        )

    def _primary_pairs(self, candidates: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        if self.reference is not None:
            return tuple((value, self.reference) for value in candidates if value != self.reference)
        return tuple(
            (left, right)
            for index, left in enumerate(candidates)
            for right in candidates[index + 1 :]
        )

    def _relation(self, lower: float, upper: float) -> str:
        margin = self.practical_margin or 0.0
        if lower > margin:
            return "MATERIALLY_BETTER"
        if upper < -margin:
            return "MATERIALLY_WORSE"
        if self.practical_margin is not None and lower >= -margin and upper <= margin:
            return "PRACTICALLY_EQUIVALENT"
        return "UNRESOLVED"

    def _pm_eb_interval(
        self, differences: Sequence[float], alpha: float
    ) -> tuple[float, float]:
        """Return the PM-EB confidence sequence using the authored objective span."""
        if not differences:
            return (-math.inf, math.inf)
        objective_span = self.high - self.low
        transformed = [
            min(1.0, max(0.0, (float(value) + objective_span) / (2.0 * objective_span)))
            for value in differences
        ]
        weighted_sum = 0.0
        weight = 0.0
        penalty = math.log(2.0 / alpha)
        running_mean = 0.5
        running_variance = 0.25
        residual_sum = 0.0
        observed_sum = 0.0
        for index, value in enumerate(transformed, start=1):
            lam = min(
                0.5,
                math.sqrt(
                    2.0 * math.log(2.0 / alpha)
                    / max(1e-15, running_variance * index * math.log(index + 1.0))
                ),
            )
            residual = value - running_mean
            psi = (-math.log1p(-lam) - lam) / 4.0
            penalty += 4.0 * residual * residual * psi
            weighted_sum += lam * value
            weight += lam
            observed_sum += value
            running_mean = (0.5 + observed_sum) / (index + 1.0)
            residual_sum += (value - running_mean) ** 2
            running_variance = (0.25 + residual_sum) / (index + 1.0)
        center = weighted_sum / weight
        half_width = penalty / weight
        low = max(0.0, center - half_width)
        high = min(1.0, center + half_width)
        return (
            (2.0 * low - 1.0) * objective_span,
            (2.0 * high - 1.0) * objective_span,
        )

    @staticmethod
    def hoeffding_interval(
        differences: Sequence[float], *, objective_span: float, alpha: float
    ) -> tuple[float, float]:
        """Historical summable-look Hoeffding interval retained as a test oracle."""
        n = len(differences)
        if n == 0:
            return (-objective_span, objective_span)
        look_alpha = alpha / (n * (n + 1))
        radius = 2.0 * objective_span * math.sqrt(math.log(2.0 / look_alpha) / (2.0 * n))
        mean = statistics.fmean(differences)
        return (max(-objective_span, mean - radius), min(objective_span, mean + radius))

    def _validate_observations(
        self,
        values: Mapping[int, Mapping[int, float]],
        candidates: Sequence[int],
        seeds: Sequence[int],
    ) -> None:
        for candidate in candidates:
            for seed in seeds:
                value = float(values[candidate][seed])
                if not math.isfinite(value) or not self.low <= value <= self.high:
                    raise ValueError(
                        f"Sweep objective {value!r} for candidate {candidate}, seed {seed} is "
                        f"outside the authored bounds [{self.low}, {self.high}]."
                    )


__all__ = ["PairedSweepSequentialAnalyzer", "SweepSequentialDecision"]
