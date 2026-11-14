"""Exact threshold integrals, left-open merges and explicit overlap selection."""

import numpy as np
import pytest

from lambdaforge.topology import (
    StableRegionSelector,
    SuperlevelMergeTree,
    discrete_membership,
    uniform_membership,
    weighted_dice,
    weighted_iou,
)


def chain(values, weights=None):
    return SuperlevelMergeTree().fit(
        np.asarray(values, dtype=float),
        np.array([(i, i + 1) for i in range(len(values) - 1)], dtype=int).reshape(-1, 2),
        weights,
    )


def test_exact_integral_weighted_mass_and_no_death_crossing():
    tree = chain([5, 2, 4, 1, 3], np.array([1, 2, 3, 4, 5]))
    phi = uniform_membership(tree, 0, 1, 5)
    assert phi == pytest.approx([1, 0.25, 0.25, 0, 0])
    assert phi @ tree.vertex_weights == pytest.approx(2.25)
    assert uniform_membership(tree, 1, 2, 4) == pytest.approx([0, 0, 1, 0, 0])
    with pytest.raises(ValueError, match="death"):
        uniform_membership(tree, 1, 1, 4)
    with pytest.raises(ValueError, match="lifetime"):
        discrete_membership(tree, 1, np.array([2.0, 3.0]), np.array([0.0, 1.0]))
    discrete = discrete_membership(tree, 0, np.array([5.0, 2.0]), np.array([1.0, 3.0]))
    assert discrete == pytest.approx([1, 0.75, 0.75, 0, 0])
    assert not discrete.flags.writeable


def test_selection_widest_interval_not_tiny_iou_maximum():
    tree = chain([5, 4, 3, 2, 1])
    result = StableRegionSelector(min_iou=0.5, min_width=2, min_mass=1).select(tree)
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.interval == (1.0, 4.0)  # width 3 beats perfect-IoU intervals of width 1
    assert region.stability == pytest.approx(0.5)
    assert region.mass == pytest.approx(3)
    assert region.support_mass == 4
    assert region.persistence is None
    assert region.membership == pytest.approx([1, 1, 2 / 3, 1 / 3, 0])
    assert set(region.support) == {0, 1, 2, 3}


def test_no_manufactured_intervals_and_threshold_sample_constraint():
    tree = chain([2, 2, 2])
    result = StableRegionSelector().select(tree)
    assert not result.regions and "no interval" in result.rejected[0]
    tree = chain([3, 2, 1])
    for selector in [
        StableRegionSelector(min_width=10),
        StableRegionSelector(min_mass=10),
        StableRegionSelector(min_samples=5),
        StableRegionSelector(min_persistence=5),
    ]:
        assert not selector.select(tree).regions


def test_nested_regions_overlap_policy_and_plateaus():
    tree = chain([5, 2, 4, 1, 3])
    allowed = StableRegionSelector(min_iou=0.1, overlap="allow").select(tree)
    assert len(allowed.regions) == 3
    disjoint = StableRegionSelector(min_iou=0.1).select(tree)
    used = set()
    for region in disjoint.regions:
        assert used.isdisjoint(region.support)
        used.update(region.support)
        assert np.all((region.membership >= 0) & (region.membership <= 1))
    assert len(disjoint.regions) < len(allowed.regions)
    assert any("overlaps" in reason for reason in disjoint.rejected.values())
    plateau = chain([4, 4, 1])
    result = StableRegionSelector(min_iou=1).select(plateau)
    assert result.regions[0].membership == pytest.approx([1, 1, 0])


@pytest.mark.parametrize("seed", range(15))
def test_integral_matches_eventwise_midpoint_reference(seed):
    rng = np.random.default_rng(seed)
    values = rng.uniform(-2, 5, size=12)
    tree = chain(values)
    for branch in tree.branches:
        lower = branch.death if branch.death is not None else values.min()
        upper = branch.birth
        if upper == lower:
            continue
        breakpoints = np.unique(np.r_[lower, values[(values > lower) & (values < upper)], upper])
        expected = np.zeros(len(values))
        for low, high in zip(breakpoints[:-1], breakpoints[1:], strict=True):
            expected[tree.region_at(branch.id, (low + high) / 2)] += (high - low) / (upper - lower)
        assert uniform_membership(tree, branch.id, lower, upper) == pytest.approx(expected)


def test_limits_and_invalid_policy():
    with pytest.raises(ValueError):
        StableRegionSelector(min_iou=0)
    with pytest.raises(ValueError):
        StableRegionSelector(overlap="trim")
    with pytest.raises(MemoryError):
        StableRegionSelector(max_membership_bytes=1).select(chain([3, 2, 1]))


def test_weighted_binary_and_fuzzy_overlap():
    first = np.array([True, True, False])
    second = np.array([False, True, True])
    weights = np.array([1.0, 2.0, 3.0])
    assert weighted_iou(first, second, weights=weights) == pytest.approx(2 / 6)
    assert weighted_dice(first, second, weights=weights) == pytest.approx(4 / 8)
    assert weighted_dice(first, first) == 1
    assert weighted_iou(np.array([0.2, 0.8]), np.array([0.6, 0.4])) == pytest.approx(0.6 / 1.4)
    assert weighted_iou(np.zeros(3), np.zeros(3)) == 1
    assert weighted_dice(np.zeros(3), np.zeros(3), empty_value=0) == 0
    with pytest.raises(ValueError):
        weighted_iou(np.array([2.0]), np.array([1.0]))


def test_discrete_membership_accumulation_never_exceeds_one():
    tree = chain([3, 1])
    membership = discrete_membership(tree, 0, np.full(100, 3.0), np.ones(100))
    assert membership.tolist() == [1.0, 0.0]
