"""Whole-object transport invariants, with a real optional POT solver."""

from __future__ import annotations

import builtins
import subprocess
import sys

import numpy as np
import pytest

from lambdaforge.distances import FusedGromovWassersteinDistance as FGW
from lambdaforge.distances import StructuredMeasure


def measure(features=(0.0, 1.0), weights=None):
    return StructuredMeasure([[0, 1], [1, 0]], features=features, weights=weights)


def solve(source, target, **kwargs):
    pytest.importorskip("ot")
    return FGW(**kwargs).solve(source, target)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 1.0])
def test_identity_and_marginals(alpha):
    obj = measure(weights=[1, 3])
    result = solve(obj, obj, alpha=alpha)
    assert result.value == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(result.transport.sum(axis=0), [0.25, 0.75])
    np.testing.assert_allclose(result.transport.sum(axis=1), [0.25, 0.75])
    assert result.converged and result.marginal_error < 1e-10
    assert not result.transport.flags.writeable
    assert obj.total_mass == 4


def test_feature_only_transport_has_analytic_squared_cost():
    source = StructuredMeasure([[0]], features=[[0]], weights=[2])
    target = StructuredMeasure([[0, 2], [2, 0]], features=[[1], [3]], weights=[1, 3])
    result = solve(source, target, alpha=0)
    assert result.value == pytest.approx(7)
    np.testing.assert_allclose(result.transport, [[0.25, 0.75]])


def test_structural_cost_matches_explicit_four_index_objective():
    source = measure(weights=[1, 3])
    target = StructuredMeasure.from_points([[0], [2], [3]], features=[1, 2, 3])
    result = solve(source, target, alpha=0.4)
    structure = sum(
        (source.structure[i, k] - target.structure[j, other]) ** 2
        * result.transport[i, j]
        * result.transport[k, other]
        for i in range(2)
        for k in range(2)
        for j in range(3)
        for other in range(3)
    )
    features = sum(
        (source.features[i] - target.features[j]) ** 2 * result.transport[i, j]
        for i in range(2)
        for j in range(3)
    )
    assert result.structure_cost == pytest.approx(structure)
    assert result.feature_cost == pytest.approx(features)
    assert result.value == pytest.approx(0.6 * features + 0.4 * structure)
    assert result.objective_history[-1] == pytest.approx(result.value)


def test_permutation_invariance_with_feasible_warm_start():
    pytest.importorskip("ot")
    source = measure(features=[0, 10], weights=[1, 3])
    target = measure(features=[10, 0], weights=[3, 1])
    result = FGW().solve(source, target, initial_coupling=[[0, 0.25], [0.75, 0]])
    assert result.value == pytest.approx(0, abs=1e-12)


def test_arbitrary_features_and_precomputed_distances_agree():
    pytest.importorskip("ot")
    source = measure(features=[{"kind": "a"}, {"kind": "b"}])
    target = measure(features=[{"kind": "a"}, {"kind": "c"}])
    solver = FGW(feature_metric=lambda a, b: float(a["kind"] != b["kind"]))
    result = solver.solve(source, target)
    direct = FGW().solve(source, target, feature_distances=[[0, 1], [1, 1]])
    assert result.value == pytest.approx(direct.value)


def test_structure_only_and_zero_mass_nodes():
    source = measure(features=None, weights=[0, 2])
    result = solve(source, source, alpha=1)
    assert result.value == pytest.approx(0)
    np.testing.assert_allclose(result.transport, [[0, 0], [0, 1]])


def test_pairwise_is_an_object_distance_matrix_not_a_clusterer():
    pytest.importorskip("ot")
    solver = FGW()
    objects = [measure(), measure(features=[2, 3])]
    result = solver.pairwise(objects, symmetric=True)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(result.diagonal(), 0, atol=1e-12)
    assert not hasattr(solver, "cluster")
    assert solver.pairwise([]).shape == (0, 0)


@pytest.mark.parametrize(
    "structure", [[], [[0, 1]], [[0, -1], [-1, 0]], [[0, 1], [2, 0]], [[1]], [[float("inf")]]]
)
def test_invalid_geometry(structure):
    with pytest.raises(ValueError):
        StructuredMeasure(structure)


@pytest.mark.parametrize("weights", [[0, 0], [-1, 1], [1], [1, float("nan")]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        measure(weights=weights)


def test_memory_guard_precedes_backend_and_cross_allocation():
    with pytest.raises(MemoryError, match="workspace"):
        FGW(max_memory_bytes=1).solve(measure(), measure())
    with pytest.raises(MemoryError, match="intrinsic"):
        StructuredMeasure.from_points([[0], [1]], max_structure_bytes=1)
    with pytest.raises(MemoryError, match="Pairwise"):
        FGW(max_memory_bytes=1).pairwise([measure()])
    with pytest.raises(MemoryError, match="plus dense FGW workspace"):
        FGW(max_memory_bytes=640).pairwise([measure()])


@pytest.mark.parametrize(
    "option",
    [
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": True},
        {"alpha": float("nan")},
        {"max_iter": 0},
        {"max_iter": True},
        {"tol_abs": 0},
        {"tol_rel": float("inf")},
        {"max_memory_bytes": -1},
        {"feature_metric": 12},
    ],
)
def test_invalid_solver_policy(option):
    with pytest.raises((ValueError, TypeError)):
        FGW(**option)


def test_failed_transport_solver_is_not_reported_as_valid_evidence(monkeypatch):
    gromov = pytest.importorskip("ot.gromov")
    monkeypatch.setattr(
        gromov,
        "fused_gromov_wasserstein",
        lambda *a, **k: (np.diag([0.5, 0.5]), {"warning": "LP failed", "loss": [0, 0]}),
    )
    with pytest.raises(RuntimeError, match="transport solver failed: LP failed"):
        FGW().solve(measure(), measure())


def test_missing_optional_solver_has_actionable_error(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("ot"):
            raise ImportError("test missing POT")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ImportError, match=r"lambdaforge\[optimal-transport\]"):
        FGW().solve(measure(), measure())


@pytest.mark.parametrize(
    "distances", [[[0]], [[-1, 0], [0, 1]], [[float("inf"), 0], [0, 1]], [[1e300, 0], [0, 1]]]
)
def test_invalid_feature_distances(distances):
    pytest.importorskip("ot")
    with pytest.raises(ValueError):
        FGW().solve(measure(), measure(), feature_distances=distances)


def test_invalid_coupling_and_missing_features():
    pytest.importorskip("ot")
    with pytest.raises(ValueError, match="marginals"):
        FGW().solve(measure(), measure(), initial_coupling=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="needs features"):
        FGW().solve(measure(features=None), measure())
    huge = StructuredMeasure([[0, 1e300], [1e300, 0]], features=[0, 1])
    with pytest.raises(ValueError, match="Structural distances overflow"):
        FGW().solve(huge, huge)


def test_iteration_limit_does_not_claim_convergence():
    result = solve(measure(), measure(features=[3, 1]), max_iter=1, tol_rel=1e-15, tol_abs=1e-15)
    assert result.iterations == 1 and not result.converged


def test_import_does_not_load_optional_backend_or_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lambdaforge.distances; "
            "from lambdaforge.distances import FusedGromovWassersteinDistance; "
            "assert not any(x in sys.modules for x in ('ot', 'torch', 'sklearn'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
