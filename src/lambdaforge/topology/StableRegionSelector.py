"""Explicit longest-stable-interval policy consuming an existing merge tree."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import MappingProxyType

import numpy as np

from .MergeTree import MergeTree
from .RegionSelection import RegionSelection
from .StableRegion import StableRegion


class StableRegionSelector:
    """Select the widest interval satisfying all explicit constraints per branch.

    Candidate endpoints are that branch's growth events plus the observed field
    floor (or death boundary). Equal widths prefer higher endpoint stability,
    then higher upper level. At least ``min_samples`` endpoints are required;
    they are filtration samples, NOT independent statistical observations.
    ``disjoint`` greedily keeps widest intervals first and rejects overlapping
    positive-membership supports, without trimming or renormalizing them.
    ``allow`` explicitly allows overlapping parent/child regions.
    Essential branches use the observed floor, not infinite persistence.
    """

    def __init__(
        self,
        *,
        min_persistence: float = 0,
        min_mass: float = 0,
        min_width: float = 0,
        min_samples: int = 2,
        min_iou: float = 0.8,
        overlap: str = "disjoint",
        max_membership_bytes: int | None = 256 * 1024 * 1024,
    ) -> None:
        if any(not np.isfinite(v) or v < 0 for v in (min_persistence, min_mass, min_width)):
            raise ValueError("Persistence, mass and width minima must be finite and nonnegative.")
        if not np.isfinite(min_iou) or not 0 < min_iou <= 1:
            raise ValueError("min_iou must be in (0, 1].")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2.")
        if overlap not in {"allow", "disjoint"}:
            raise ValueError("overlap must be 'allow' or 'disjoint'.")
        if max_membership_bytes is not None and (
            isinstance(max_membership_bytes, bool)
            or not isinstance(max_membership_bytes, int)
            or max_membership_bytes < 1
        ):
            raise ValueError("max_membership_bytes must be positive or null.")
        self.min_persistence = min_persistence
        self.min_mass = min_mass
        self.min_width = min_width
        self.min_samples = min_samples
        self.min_iou = min_iou
        self.overlap = overlap
        self.max_membership_bytes = max_membership_bytes

    def select(self, tree: MergeTree) -> RegionSelection:
        """Return selected regions and a reason for every unselected branch."""
        rejected: dict[int, str] = {}
        proposals: list[tuple[float, float, float, int, float, int]] = []
        for branch in tree.branches:
            if branch.persistence is not None and branch.persistence < self.min_persistence:
                rejected[branch.id] = "persistence below minimum"
                continue
            curve = tree.mass_curve(branch.id)
            floor = float(tree.values.min()) if branch.death is None else branch.death
            # Essential persistence is undefined; compare only observed span.
            if branch.essential and branch.birth - floor < self.min_persistence:
                rejected[branch.id] = "observed essential span below minimum"
                continue
            levels, masses = curve[:, 0].tolist(), curve[:, 1].tolist()
            if levels[-1] > floor:
                levels.append(floor)
                masses.append(masses[-1])  # left-open death boundary, pre-merge mass
            lower_masses = [masses[0], *masses[:-1]]
            best: tuple[float, float, float, int, float, int] | None = None
            low = 0
            for high in range(len(levels)):
                low = max(low, high)
                while (
                    low + 1 < len(levels) and lower_masses[low + 1] <= masses[high] / self.min_iou
                ):
                    low += 1
                width = levels[high] - levels[low]
                if (
                    width <= 0
                    or width < self.min_width
                    or masses[high] < self.min_mass
                    or low - high + 1 < self.min_samples
                    or lower_masses[low] == 0
                ):
                    continue
                stability = masses[high] / lower_masses[low]
                item = (width, stability, levels[high], branch.id, levels[low], low - high + 1)
                if best is None or item[:3] > best[:3]:
                    best = item
            if best is None:
                rejected[branch.id] = "no interval satisfies width, mass, samples and weighted IoU"
            else:
                proposals.append(best)
        regions: list[StableRegion] = []
        occupied = np.zeros(len(tree.values), dtype=bool)
        for _, stability, upper_level, branch_id, lower_level, samples in sorted(
            proposals, key=lambda p: (-p[0], -p[1], -p[2], p[3])
        ):
            # No mask extends to the other side of a death event.
            support = tree.region_at(branch_id, float(np.nextafter(lower_level, upper_level)))
            if self.overlap == "disjoint" and occupied[support].any():
                rejected[branch_id] = "support overlaps an earlier widest accepted interval"
                continue
            required = (len(regions) + 1) * len(tree.values) * 8
            if self.max_membership_bytes is not None and required > self.max_membership_bytes:
                raise MemoryError(
                    "Region memberships exceed max_membership_bytes; select fewer regions."
                )
            membership = uniform_membership(tree, branch_id, lower_level, upper_level)
            support = np.flatnonzero(membership > 0)
            core = tree.region_at(branch_id, upper_level)
            region = StableRegion(
                branch_id,
                (lower_level, upper_level),
                core,
                support,
                membership,
                float(membership @ tree.vertex_weights),
                float(tree.vertex_weights[support].sum()),
                tree.branches[branch_id].persistence,
                stability,
                samples,
            )
            regions.append(region)
            occupied[support] = True
        return RegionSelection(tuple(regions), MappingProxyType(rejected), self.overlap)


def uniform_membership(tree: MergeTree, branch_id: int, lower: float, upper: float) -> np.ndarray:
    """Exact deterministic integral of branch membership under uniform u in (lower, upper].

    Each newly attached subtree is visited once. Lower may equal death because
    that endpoint has zero measure; no threshold at/below death is followed.
    """
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower >= upper
        or not np.isfinite(float(upper) - float(lower))
    ):
        raise ValueError("Membership requires finite lower < upper.")
    tree._event_at(branch_id, upper)
    branch = tree.branches[branch_id]
    if branch.death is not None and lower < branch.death:
        raise ValueError("Membership interval cannot cross branch death.")
    return _membership_from_cdf(
        tree, branch_id, lambda level: max(0.0, min(upper, level) - lower) / (upper - lower)
    )


def _membership_from_cdf(
    tree: MergeTree, branch_id: int, cdf: Callable[[float], float]
) -> np.ndarray:
    """Assign threshold-CDF mass when each disjoint subtree first joins the branch."""
    membership = np.zeros(len(tree.values))
    for index in tree._histories[branch_id]:
        event = tree.events[index]
        fraction = cdf(event.level)
        if fraction == 0:
            break
        membership[list(event.added_vertices)] = fraction
        for child in event.children:
            if tree.events[child].branch != branch_id:
                membership[tree._region(child)] = fraction
    membership.setflags(write=False)
    return membership


def discrete_membership(
    tree: MergeTree, branch_id: int, thresholds: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Normalize an explicit finite nonnegative threshold measure, never spatial masses."""
    levels = np.asarray(thresholds, dtype=float)
    masses = np.asarray(weights, dtype=float)
    if (
        levels.ndim != 1
        or not len(levels)
        or masses.shape != levels.shape
        or not np.isfinite(levels).all()
        or not np.isfinite(masses).all()
        or np.any(masses < 0)
        or not 0 < masses.sum() < np.inf
    ):
        raise ValueError(
            "Thresholds and finite nonnegative weights must match, with positive total."
        )
    # Validate all levels, including zero-weight ones, before accumulating.
    for level in levels:
        tree._event_at(branch_id, float(level))
    total = math.fsum(masses)
    # Correctly rounded subset sums preserve [0,1], unlike repeated normalized
    # increments (100 times 0.01 can exceed one). No clipping/repair is needed.
    return _membership_from_cdf(
        tree, branch_id, lambda level: math.fsum(masses[levels <= level]) / total
    )
