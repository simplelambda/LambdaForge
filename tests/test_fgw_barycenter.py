"""Numerical barycenters use POT, including nonzero diagonal and final objective checks."""

import numpy as np
import pytest

from lambdaforge.distances import FGWBarycenter, FusedGromovWassersteinDistance, StructuredMeasure

ot = pytest.importorskip("ot")


def object_at(offset=0.0):
    return StructuredMeasure(
        np.array([[0.0, 2.0], [2.0, 0.0]]),
        features=np.array([[offset], [offset + 2]]),
        weights=[1, 1],
    )


def test_one_object_exact_initialization_and_permutation():
    obj = object_at()
    solver = FGWBarycenter(2, seed=3)
    result = solver.fit([obj], init_structure=obj.structure, init_features=obj.features)
    assert result.objective == pytest.approx(0, abs=1e-10)
    assert result.converged and result.prototype.relational
    assert result.prototype.structure == pytest.approx(obj.structure)
    perm = [1, 0]
    permuted = StructuredMeasure(
        obj.structure[np.ix_(perm, perm)], obj.features[perm], obj.weights[perm]
    )
    second = solver.fit([permuted], init_structure=obj.structure, init_features=obj.features)
    assert second.objective == pytest.approx(result.objective, abs=1e-10)
    assert not result.object_weights.flags.writeable


def test_nonzero_relational_diagonal_not_zeroed_and_real_pot_comparison():
    a, b = object_at(), object_at(2)
    result = FGWBarycenter(1, seed=10).fit(
        [a, b], object_weights=[1, 3], init_structure=[[0.0]], init_features=[[0.0]]
    )
    # With one support node, every source pair maps to it; mean structural relation is 1.
    assert result.prototype.structure[0, 0] == pytest.approx(1.0)
    assert result.prototype.features[0, 0] == pytest.approx(2.5)
    assert sum(result.object_weights) == 1
    expected = sum(
        w * FusedGromovWassersteinDistance()(result.prototype, obj)
        for w, obj in zip(result.object_weights, [a, b], strict=True)
    )
    assert result.objective == pytest.approx(expected)
    x, c, log = ot.gromov.fgw_barycenters(
        1,
        [a.features.copy(), b.features.copy()],
        [a.structure.copy(), b.structure.copy()],
        ps=[a.weights.copy(), b.weights.copy()],
        lambdas=np.array([0.25, 0.75]),
        p=np.array([1.0]),
        alpha=0.5,
        max_iter=100,
        tol=1e-7,
        init_C=np.array([[0.0]]),
        init_X=np.array([[0.0]]),
        warmstartT=True,
        log=True,
        random_state=result.start_seed,
    )
    assert result.prototype.features == pytest.approx(x)
    assert result.prototype.structure == pytest.approx(c)
    assert result.feature_errors == pytest.approx(log["err_feature"])
    with pytest.raises(ValueError, match="zero diagonal"):
        StructuredMeasure(c, x)
    assert result.diagnostics["backend_objective"] is None
    assert "post-optimization" in result.diagnostics["objective_source"]
    for comparison in result.comparisons:
        assert comparison.transport.sum(axis=1) == pytest.approx(result.prototype.weights)


def test_weights_multistart_seed_and_no_false_convergence():
    a, b = object_at(), object_at(1)
    solver = FGWBarycenter(2, seed=42, n_init=3, max_iter=10)
    first = solver.fit([a, b], support_weights=[1, 3])
    second = solver.fit([a, b], support_weights=[1, 3])
    assert first.prototype.weights == pytest.approx([0.25, 0.75])
    assert first.objective == pytest.approx(second.objective)
    assert first.start_seed == second.start_seed
    assert first.objective == min(start["objective"] for start in first.restarts if start["valid"])
    short = FGWBarycenter(1, max_iter=1, seed=1).fit(
        [a], init_structure=[[0.0]], init_features=[[10.0]]
    )
    assert not short.converged and short.iterations == 1 and short.warnings


@pytest.mark.parametrize(
    "kwargs", [{"alpha": 0}, {"alpha": 1}, {"n_init": 0}, {"max_iter": 0}, {"threads": 0}]
)
def test_invalid_solver_parameters(kwargs):
    with pytest.raises(ValueError):
        FGWBarycenter(2, **kwargs)


def test_invalid_inputs_initialization_and_memory():
    obj = object_at()
    with pytest.raises(MemoryError):
        FGWBarycenter(2, max_memory_bytes=1).fit([obj])
    with pytest.raises(ValueError, match="same dimension"):
        FGWBarycenter(2).fit([obj, StructuredMeasure(obj.structure, np.zeros((2, 2)))])
    with pytest.raises(ValueError, match="numeric"):
        FGWBarycenter(2).fit([StructuredMeasure(obj.structure, [{"a": 1}, {"a": 2}])])
    with pytest.raises(ValueError, match="strictly positive"):
        FGWBarycenter(2).fit([obj], support_weights=[0, 1])
    with pytest.raises(ValueError, match="init_structure"):
        FGWBarycenter(2).fit([obj], init_structure=[[0.0]])
    with pytest.raises(ValueError, match="underflow"):
        FGWBarycenter(2).fit([obj], support_weights=[np.nextafter(0.0, 1.0), 1e300])


def test_invalid_backend_does_not_become_a_plausible_prototype(monkeypatch):
    def invalid(*args, **kwargs):
        return (
            np.zeros((1, 1)),
            np.array([[np.nan]]),
            {"err_feature": [0.0], "err_structure": [0.0]},
        )

    monkeypatch.setattr(ot.gromov, "fgw_barycenters", invalid)
    with pytest.raises(RuntimeError, match="No valid"):
        FGWBarycenter(1).fit([object_at()])


def test_pot_or_stopping_is_not_reported_as_both_changes_converged(monkeypatch):
    obj = object_at()

    def stopped(*args, **kwargs):
        return (
            obj.features.copy(),
            obj.structure.copy(),
            {"err_feature": [0.0], "err_structure": [1.0], "Ts_iter": []},
        )

    monkeypatch.setattr(ot.gromov, "fgw_barycenters", stopped)
    result = FGWBarycenter(2).fit([obj])
    assert not result.converged
    assert not result.diagnostics["backend_both_changes_small"]
    assert all(comparison.converged for comparison in result.comparisons)
    assert any("only one" in warning for warning in result.warnings)


def test_weighted_unequal_objects_remain_compatible_with_distance_and_clustering():
    from lambdaforge.clustering import DBSCAN
    from lambdaforge.distances import FGWMatrixBuilder

    a = StructuredMeasure([[0.0, 1.0], [1.0, 0.0]], [[0.0], [1.0]], weights=[1.0, 3.0])
    b = StructuredMeasure(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        [[0.0], [0.5], [1.0]],
        weights=[1.0, 2.0, 1.0],
    )
    result = FGWBarycenter(2, seed=3, n_init=2).fit([a, b], object_weights=[1.0, 2.0])
    assert result.object_weights == pytest.approx([1 / 3, 2 / 3])
    assert np.isfinite(result.objective)
    for obj, comparison in zip([a, b], result.comparisons, strict=True):
        assert comparison.transport.sum(axis=0) == pytest.approx(obj.weights)
    # Native FGW matrices can feed the public precomputed clustering API directly.
    matrix = FGWMatrixBuilder(FusedGromovWassersteinDistance()).build([a, b], symmetric=True)
    labels = DBSCAN(distance="precomputed", eps=100, min_samples=1).cluster(matrix.values)
    assert labels.n_clusters == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"structure": [[0, 1j], [1j, 0]]}, {"structure": [[0, 1], [1, 0]], "weights": [1j, 1j]}],
)
def test_complex_relations_are_not_silently_cast_to_real(kwargs):
    with pytest.raises(ValueError, match="real"):
        StructuredMeasure(**kwargs)


def test_invalid_restart_is_recorded_and_a_valid_restart_can_win(monkeypatch):
    original = ot.gromov.fgw_barycenters
    calls = []

    def one_invalid(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return np.zeros((1, 1)), np.array([[np.nan]]), {}
        return original(*args, **kwargs)

    monkeypatch.setattr(ot.gromov, "fgw_barycenters", one_invalid)
    result = FGWBarycenter(1, n_init=2, seed=10).fit([object_at()])
    assert not result.restarts[0]["valid"] and result.restarts[1]["valid"]
    assert result.start_seed == result.restarts[1]["seed"]
    assert result.n_support == 1 and result.alpha == 0.5
