"""Analytic persistence and exhaustive graph-component oracles, not snapshot tests."""

from collections import Counter

import numpy as np
import pytest

from lambdaforge.topology import SuperlevelMergeTree


def partitions(components):
    return {frozenset(map(int, component)) for component in components}


def exhaustive(values, edges, threshold):
    remaining = set(np.flatnonzero(values >= threshold))
    result = []
    while remaining:
        todo = [remaining.pop()]
        component = set(todo)
        while todo:
            vertex = todo.pop()
            for a, b in edges:
                other = b if a == vertex else a if b == vertex else None
                if other in remaining:
                    remaining.remove(other)
                    component.add(other)
                    todo.append(other)
        result.append(component)
    return partitions(result)


def test_chain_elder_rule_and_death_boundary():
    tree = SuperlevelMergeTree().fit(
        np.array([5.0, 2.0, 4.0, 1.0, 3.0]),
        np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    assert tree.persistence_diagram() == ((5.0, None), (4.0, 2.0), (3.0, 1.0))
    assert tree.branches[0].children == (1, 2)
    assert tree.branches[1].persistence == 2
    assert tree.branches[0].essential and tree.branches[0].persistence is None
    assert set(tree.region_at(0, 2)) == {0, 1, 2}
    assert set(tree.region_at(1, 2.01)) == {2}
    with pytest.raises(ValueError, match="lifetime"):
        tree.region_at(1, 2)
    assert np.array_equal(tree.mass_curve(0), [[5, 1], [2, 6], [1, 15]])
    assert not tree.values.flags.writeable
    assert not tree.region_at(0, 2).flags.writeable


def test_plateau_disconnected_cycles_ties_and_negative_values():
    values = np.array([0.0, 0.0, -2.0, 0.0, 0.0, -3.0, -7.0])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 2], [2, 2], [0, 1]])
    tree = SuperlevelMergeTree().fit(values, edges)
    assert len(tree.branches) == 4
    assert tree.branches[0].birth_vertices == (0, 1)
    assert tree.branches[1].birth_vertices == (3, 4)
    assert tree.branches[1].parent == 0  # equal birth, minimum birth vertex elder
    assert sum(b.essential for b in tree.branches) == 3
    assert partitions(tree.components_at(-2)) == {frozenset([0, 1, 2, 3, 4])}
    for threshold in [-10, -7, -3, -2, 0, 1]:
        assert partitions(tree.components_at(threshold)) == exhaustive(values, edges, threshold)


@pytest.mark.parametrize("seed", range(30))
def test_against_exhaustive_components_and_permutations(seed):
    rng = np.random.default_rng(seed)
    n = 12
    values = rng.integers(-3, 4, n).astype(float)
    edges = np.array(
        [(i, j) for i in range(n) for j in range(i) if rng.random() < 0.2], dtype=np.int64
    ).reshape(-1, 2)
    builder = SuperlevelMergeTree()
    tree = builder.fit(values, edges)
    transformed = builder.fit(np.exp(values), edges[::-1, ::-1])
    permutation = rng.permutation(n)
    inverse = np.argsort(permutation)
    permuted = builder.fit(values[permutation], inverse[edges])
    assert Counter(tree.persistence_diagram()) == Counter(permuted.persistence_diagram())
    assert [b.parent for b in tree.branches] == [b.parent for b in transformed.branches]
    for threshold in np.arange(-4.0, 4.5, 0.5):
        expected = exhaustive(values, edges, threshold)
        assert partitions(tree.components_at(threshold)) == expected
        assert partitions(transformed.components_at(np.exp(threshold))) == expected
        assert partitions(permutation[c] for c in permuted.components_at(threshold)) == expected
    for branch in tree.branches:
        for level, mass in tree.mass_curve(branch.id):
            region = tree.region_at(branch.id, level)
            assert frozenset(region) in exhaustive(values, edges, level)
            assert len(region) == mass
    assert len(tree.events) <= n
    assert len(tree._order) == n


def test_no_edges_and_deep_chain_without_recursive_traversal():
    n = 3000
    values = np.arange(n, dtype=float)
    tree = SuperlevelMergeTree().fit(values, np.empty((0, 2), dtype=int))
    assert len(tree.branches) == n and all(b.essential for b in tree.branches)
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    tree = SuperlevelMergeTree().fit(values, edges)
    assert len(tree.branches) == 1
    assert len(tree.region_at(0, 0)) == n


@pytest.mark.parametrize(
    "values,edges,weights",
    [
        ([], np.empty((0, 2), dtype=int), None),
        ([np.nan], np.empty((0, 2), dtype=int), None),
        ([1.0], np.array([[0, 1]]), None),
        ([1.0], np.array([[0.0, 0.0]]), None),
        ([1.0], np.array([[0, 0]]), [0.0]),
        ([1.0], np.array([[0, 0]]), [-1.0]),
    ],
)
def test_validation(values, edges, weights):
    with pytest.raises(ValueError):
        SuperlevelMergeTree().fit(values, edges, weights)
