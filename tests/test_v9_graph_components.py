"""Generic V9 primitives: equations, geometry, gradients, support and bounded memory."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest
import torch
from torch import nn

from lambdaforge.nn.models.graph import (
    GINE,
    GVP,
    GVPGNN,
    CosineCutoff,
    DimeNet,
    DimeNetBasis,
    DimeNetInteraction,
    DimeNetPlusPlusInteraction,
    GaussianRadialBasis,
    GINELayer,
    GraphStructuralEncoding,
    GraphTriplets,
    NNConv,
    NNConvLayer,
    PaiNN,
    SchNet,
    SineRadialBasis,
    StructuralGraphTransformer,
    StructuralGraphTransformerLayer,
    VectorDropout,
    edge_geometry,
)


@pytest.fixture
def data():
    torch.manual_seed(17)
    x = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    positions = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    edges = torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3, 0], [1, 0, 2, 0, 3, 1, 3, 2, 1]])
    attributes = torch.randn(edges.shape[1], 2, dtype=torch.double, requires_grad=True)
    return x, positions, edges, attributes


def test_gine_exact_relu_message_and_root():
    layer = GINELayer(2, 2, 2, epsilon=0.5).double()
    layer.mlp = nn.Identity()
    with torch.no_grad():
        layer.edge_projection.weight.copy_(torch.eye(2))
    x = torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=torch.double)
    edges = torch.tensor([[0, 0], [1, 1]])
    attrs = torch.tensor([[-2.0, 1.0], [1.0, -4.0]], dtype=torch.double)
    actual = layer(x, edges, attrs)
    expected = 1.5 * x
    expected[1] += torch.tensor([2.0, 3.0])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("bases", [None, 1, 3])
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_nnconv_matches_explicit_edge_matrices_and_chunks(data, bases, aggregation):
    x, _, edges, attrs = data
    layer = NNConvLayer(
        4,
        3,
        2,
        num_bases=bases,
        aggregation=aggregation,
        center_conditioner=bases is not None,
        message_chunk_size=2,
    ).double()
    actual = layer(x, edges, attrs)
    weights = layer.conditioner(attrs)
    if bases is not None:
        weights = weights - layer.conditioner(torch.zeros_like(attrs))
        matrices = layer.base_weight + torch.einsum("eb,bio->eio", weights, layer.bases)
    else:
        matrices = weights.reshape(-1, 4, 3)
    expected = x.new_zeros((5, 3))
    count = x.new_zeros(5)
    for index, (source, target) in enumerate(edges.t()):
        expected[target] = expected[target] + x[source] @ matrices[index]
        count[target] += 1
    if aggregation == "mean":
        expected = expected / count.clamp_min(1)[:, None]
    expected = expected + layer.root(x)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize(
    "name",
    [
        "gine",
        "nn-full",
        "nn-bases",
        "schnet",
        "gvp",
        "painn",
        "directional-bilinear",
        "directional-factorized",
    ],
)
def test_graph_operators_permutation_gradients_and_empty_graph(data, name):
    x, positions, edges, attrs = data
    if name.startswith("directional"):
        pytest.importorskip("scipy")
        model = DimeNet(
            4,
            3,
            hidden_channels=8,
            num_layers=2,
            num_radial=2,
            num_spherical=3,
            edge_channels=2,
            interaction_type=name.split("-")[1],
        ).double()
    elif name == "gine":
        model = GINELayer(4, 3, 2).double()
    elif name.startswith("nn"):
        model = NNConvLayer(4, 3, 2, num_bases=2 if name == "nn-bases" else None).double()
    elif name == "schnet":
        model = SchNet(4, 3, hidden_channels=8, num_layers=2, edge_channels=2).double()
    elif name == "gvp":
        model = GVPGNN(
            4, 3, scalar_channels=8, vector_channels=3, num_layers=2, edge_channels=2
        ).double()
    else:
        model = PaiNN(4, 3, hidden_channels=8, num_layers=2, edge_channels=2).double()

    def forward(features, coords, routed, attributes):
        if name in {"gine", "nn-full", "nn-bases"}:
            return model(features, routed, attributes)
        distances, directions = edge_geometry(coords, routed)
        if name == "schnet":
            return model(features, routed, distances, attributes)
        if name == "gvp":
            return model(features, routed, directions[:, None, :], attributes)
        if name == "painn":
            return model(features, routed, distances, directions, attributes)
        triplets = GraphTriplets.from_edges(routed, len(features))
        return model(
            features, routed, distances, triplets, triplets.angles(coords, routed), attributes
        )

    actual = forward(x, positions, edges, attrs)
    assert actual.shape == (5, 3) and torch.isfinite(actual).all()
    # Reorder nodes and edges independently, preserving explicit feature alignment.
    permutation = torch.tensor([4, 1, 3, 0, 2])
    inverse = torch.argsort(permutation)
    order = torch.randperm(edges.shape[1])
    changed = forward(
        x[permutation], positions[permutation], inverse[edges[:, order]], attrs[order]
    )
    torch.testing.assert_close(changed, actual[permutation], atol=1e-9, rtol=1e-9)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    empty = forward(x.detach(), positions.detach(), edges[:, :0], attrs[:0])
    assert empty.shape == (5, 3) and torch.isfinite(empty).all()
    no_nodes = forward(x.detach()[:0], positions.detach()[:0], edges[:, :0], attrs[:0])
    assert no_nodes.shape == (0, 3)


@pytest.mark.parametrize("family", ["gvp", "painn"])
@pytest.mark.parametrize("reflection", [False, True])
def test_persistent_vectors_are_o3_equivariant_and_readout_invariant(data, family, reflection):
    x, positions, edges, attrs = data
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))
    if (torch.linalg.det(orthogonal) < 0).item() != reflection:
        orthogonal[:, 0] *= -1
    shifted = positions @ orthogonal.T + torch.tensor([9.0, -3.0, 4.0])
    dist, directions = edge_geometry(positions, edges)
    dist2, directions2 = edge_geometry(shifted, edges)
    if family == "gvp":
        model = (
            GVPGNN(4, 3, scalar_channels=8, vector_channels=3, num_layers=3, edge_channels=2)
            .double()
            .eval()
        )
        args = (x, edges, directions[:, None, :], attrs)
        args2 = (x, edges, directions2[:, None, :], attrs)
    else:
        model = PaiNN(4, 3, hidden_channels=8, num_layers=3, edge_channels=2).double().eval()
        args = (x, edges, dist, directions, attrs)
        args2 = (x, edges, dist2, directions2, attrs)
    scalars, vectors = model.forward_with_vectors(*args)
    scalars2, vectors2 = model.forward_with_vectors(*args2)
    assert vectors.abs().sum() > 0
    torch.testing.assert_close(scalars2, scalars, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(vectors2, vectors @ orthogonal.T, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(model(*args2), model(*args), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("vector_in,vector_out", [(0, 0), (0, 2), (2, 0), (2, 3)])
def test_gvp_zero_vectors_and_gradients(vector_in, vector_out, gate):
    block = GVP(4, vector_in, 3, vector_out, vector_gate=gate).double()
    scalars = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    vectors = torch.zeros(5, vector_in, 3, dtype=torch.double, requires_grad=True)
    s, v = block(scalars, vectors)
    assert s.shape == (5, 3) and v.shape == (5, vector_out, 3)
    (s.square().sum() + v.square().sum()).backward()
    assert torch.isfinite(scalars.grad).all()
    if vectors.grad is not None:
        assert torch.isfinite(vectors.grad).all()


def test_vector_dropout_uses_one_mask_per_channel():
    layer = VectorDropout(0.5).train()
    vectors = torch.ones(128, 8, 3)
    result = layer(vectors)
    assert torch.equal(result[..., 0], result[..., 1]) and torch.equal(
        result[..., 0], result[..., 2]
    )
    assert (result == 0).any() and (result != 0).any()
    torch.testing.assert_close(layer.eval()(vectors), vectors)


def test_radial_support_analytic_zero_and_invalid_geometry():
    distances = torch.tensor([0.0, 1.0, 5.0, 6.0], dtype=torch.double, requires_grad=True)
    sine = SineRadialBasis(3, 5).double()
    torch.testing.assert_close(sine(distances)[0], sine.frequencies)
    cutoff = CosineCutoff(5)
    torch.testing.assert_close(cutoff(distances)[2:], torch.zeros(2, dtype=torch.double))
    assert torch.isfinite(GaussianRadialBasis(1, 5).double()(distances)).all()
    sine(distances).sum().backward()
    assert torch.isfinite(distances.grad).all()
    with pytest.raises(ValueError, match="Zero-length"):
        edge_geometry(torch.zeros(2, 3), torch.tensor([[0], [1]]))
    with pytest.raises(ValueError):
        cutoff(torch.tensor([-1.0]))


def test_triplet_types_no_immediate_return_limits_and_angles():
    edges = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])
    triplets = GraphTriplets.from_edges(edges, 3)
    pairs = set(zip(triplets.incoming.tolist(), triplets.outgoing.tolist(), strict=True))
    assert pairs == {(0, 3), (1, 3), (4, 2)}
    positions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    torch.testing.assert_close(triplets.angles(positions, edges), torch.full((3,), math.pi / 2))
    with pytest.raises(MemoryError):
        GraphTriplets.from_edges(edges, 3, max_triplets=2)
    with pytest.raises(ValueError, match="self-loops"):
        GraphTriplets.from_edges(torch.tensor([[0], [0]]), 1)


def test_spherical_basis_matches_scipy_and_has_finite_autograd():
    special = pytest.importorskip("scipy.special")
    basis = DimeNetBasis(2, 4, cutoff=4).double()
    distances = torch.tensor([0.001, 0.5, 2.0, 4.0], dtype=torch.double, requires_grad=True)
    angles = torch.tensor([0.0, 0.8, math.pi], dtype=torch.double, requires_grad=True)
    triplets = GraphTriplets(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
    radial, spherical = basis(distances, angles, triplets)
    for degree in range(4):
        for row, edge in enumerate(triplets.incoming):
            normalized = float(distances[edge].detach()) / 4
            for index in range(2):
                root = float(basis.roots[degree, index])
                expected = special.spherical_jn(degree, root * normalized) * float(
                    basis.normalizers[degree, index]
                )
                expected *= float(basis._envelope(torch.tensor(normalized, dtype=torch.double)))
                expected *= math.sqrt((2 * degree + 1) / (4 * math.pi)) * special.eval_legendre(
                    degree, math.cos(float(angles[row].detach()))
                )
                assert float(spherical[row, degree * 2 + index].detach()) == pytest.approx(
                    expected, rel=1e-8, abs=1e-10
                )
    (radial.square().sum() + spherical.square().sum()).backward()
    assert torch.isfinite(distances.grad).all() and torch.isfinite(angles.grad).all()
    torch.testing.assert_close(radial[-1], torch.zeros(2, dtype=torch.double))


@pytest.mark.parametrize("factorized", [False, True])
def test_directional_interactions_match_explicit_triplet_sum(factorized):
    cls = DimeNetPlusPlusInteraction if factorized else DimeNetInteraction
    layer = cls(4, 2, 3, num_before_skip=0, num_after_skip=0).double()
    x = torch.randn(5, 4, dtype=torch.double)
    radial = torch.randn(5, 2, dtype=torch.double)
    spherical = torch.randn(3, 6, dtype=torch.double)
    triplets = GraphTriplets(torch.tensor([0, 1, 1]), torch.tensor([2, 3, 4]))
    incoming = torch.nn.functional.silu(layer.incoming(x)) * layer.radial(radial)
    if factorized:
        incoming = torch.nn.functional.silu(layer.down(incoming))
        aggregate = x.new_zeros(5, layer.down.out_features)
    else:
        aggregate = torch.zeros_like(x)
    for row, (first, second) in enumerate(zip(triplets.incoming, triplets.outgoing, strict=True)):
        basis = layer.spherical(spherical[row])
        if factorized:
            value = incoming[first] * basis
        else:
            value = torch.einsum("b,c,obc->o", basis, incoming[first], layer.bilinear)
        aggregate[second] += value
    if factorized:
        aggregate = torch.nn.functional.silu(layer.up(aggregate))
    hidden = torch.nn.functional.silu(layer.outgoing(x)) + aggregate
    expected = x + torch.nn.functional.silu(layer.skip(hidden))
    torch.testing.assert_close(layer(x, radial, spherical, triplets), expected)


def test_global_attention_padding_batch_isolation_permutation_and_gradients():
    torch.manual_seed(3)
    edges = torch.tensor([[0, 1, 2, 1], [1, 2, 1, 0]])
    encoding = GraphStructuralEncoding.from_edges(edges, 4, num_spatial_codes=8)
    x = torch.randn(2, 4, 4, dtype=torch.double, requires_grad=True)
    spatial = encoding.spatial_indices.repeat(2, 1, 1)
    degree_in = encoding.in_degree.repeat(2, 1)
    degree_out = encoding.out_degree.repeat(2, 1)
    valid = torch.tensor([[True, True, True, True], [True, True, True, False]])
    model = (
        StructuralGraphTransformer(
            4, 3, hidden_channels=8, num_layers=2, num_heads=2, num_spatial_codes=8
        )
        .double()
        .eval()
    )
    y = model(x, spatial, degree_in, degree_out, valid)
    assert (y[1, -1] == 0).all()
    perm = torch.tensor([2, 0, 3, 1])
    changed = model(
        x[:, perm],
        spatial[:, perm][:, :, perm],
        degree_in[:, perm],
        degree_out[:, perm],
        valid[:, perm],
    )
    torch.testing.assert_close(changed, y[:, perm])
    changed_x = x.detach().clone()
    changed_x[1] += 50
    torch.testing.assert_close(model(changed_x, spatial, degree_in, degree_out, valid)[0], y[0])
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.isfinite(x.grad).all()


def test_structural_encoding_ties_disconnected_and_memory_guards():
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    encoding = GraphStructuralEncoding.from_edges(edges, 5)
    assert encoding.spatial_indices[3, 0] == 3
    assert encoding.spatial_indices[0, 3] == 0
    assert encoding.spatial_indices[4, 4] == 1
    perm = torch.tensor([3, 0, 4, 2, 1])
    inverse = torch.argsort(perm)
    other = GraphStructuralEncoding.from_edges(inverse[edges], 5)
    torch.testing.assert_close(other.spatial_indices, encoding.spatial_indices[perm][:, perm])
    with pytest.raises(MemoryError):
        GraphStructuralEncoding.from_edges(edges, 5, max_pairwise_bytes=1)
    layer = StructuralGraphTransformerLayer(4, num_heads=2, max_attention_bytes=1)
    with pytest.raises(MemoryError):
        layer(
            torch.randn(1, 5, 4),
            encoding.spatial_indices[None],
            encoding.in_degree[None],
            encoding.out_degree[None],
        )


def test_importing_graph_catalog_does_not_import_optional_scipy_or_pyg():
    # Lightning/Torch may themselves import SciPy in some distributions; block only
    # accesses originating in graph modules, not dependencies' unrelated import policy.
    script = """
import builtins
original = builtins.__import__
def checked(name, globals=None, locals=None, fromlist=(), level=0):
    origin = (globals or {}).get('__name__','')
    graph_origin = origin.startswith('lambdaforge.nn.models.graph')
    if graph_origin and name.startswith(('scipy','torch_geometric')):
        raise AssertionError('eager optional graph dependency')
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = checked
import lambdaforge.nn.models.graph
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["gine", "nn-full", "nn-bases"])
def test_message_stacks_state_dict_roundtrip(data, kind):
    x, _, edges, attributes = data
    if kind == "gine":
        model = GINE(4, 3, 2, hidden_channels=(8,))
        restored = GINE(4, 3, 2, hidden_channels=(8,))
    else:
        bases = 3 if kind == "nn-bases" else None
        model = NNConv(4, 3, 2, hidden_channels=(8,), num_bases=bases)
        restored = NNConv(4, 3, 2, hidden_channels=(8,), num_bases=bases)
    model, restored = model.double(), restored.double()
    restored.load_state_dict(model.state_dict())
    result = model(x, edges, attributes)
    torch.testing.assert_close(restored(x, edges, attributes), result)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("kind", ["schnet", "bilinear", "factorized"])
def test_distance_and_angle_encoders_are_rigid_frame_invariant(data, kind):
    x, positions, edges, attributes = data
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))
    shifted = positions @ rotation.T + torch.tensor([4.0, 3.0, -2.0])
    distances, _ = edge_geometry(positions, edges)
    other_distances, _ = edge_geometry(shifted, edges)
    if kind == "schnet":
        model = SchNet(4, 3, hidden_channels=8, num_layers=2, edge_channels=2).double()
        actual = model(x, edges, distances, attributes)
        changed = model(x, edges, other_distances, attributes)
    else:
        pytest.importorskip("scipy")
        model = DimeNet(
            4,
            3,
            hidden_channels=8,
            num_layers=2,
            num_radial=2,
            num_spherical=3,
            edge_channels=2,
            interaction_type=kind,
        ).double()
        triplets = GraphTriplets.from_edges(edges, len(x))
        actual = model(x, edges, distances, triplets, triplets.angles(positions, edges), attributes)
        changed = model(
            x, edges, other_distances, triplets, triplets.angles(shifted, edges), attributes
        )
    torch.testing.assert_close(changed, actual, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("kind", ["gine", "nn", "schnet", "gvp", "painn", "directional"])
def test_cpu_bfloat16_autocast_has_finite_backward(data, kind):
    x, positions, edges, attributes = (
        value.detach().float() if value.is_floating_point() else value for value in data
    )
    x.requires_grad_(True)
    distances, directions = edge_geometry(positions, edges)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        if kind == "gine":
            model = GINE(4, 3, 2, hidden_channels=(8,))
            result = model(x, edges, attributes)
        elif kind == "nn":
            model = NNConv(4, 3, 2, hidden_channels=(8,), num_bases=2)
            result = model(x, edges, attributes)
        elif kind == "schnet":
            model = SchNet(4, 3, hidden_channels=8, num_layers=2, edge_channels=2)
            result = model(x, edges, distances, attributes)
        elif kind == "gvp":
            model = GVPGNN(
                4, 3, scalar_channels=8, vector_channels=3, num_layers=2, edge_channels=2
            )
            result = model(x, edges, directions[:, None, :], attributes)
        elif kind == "painn":
            model = PaiNN(4, 3, hidden_channels=8, num_layers=2, edge_channels=2)
            result = model(x, edges, distances, directions, attributes)
        else:
            pytest.importorskip("scipy")
            model = DimeNet(
                4,
                3,
                hidden_channels=8,
                num_layers=2,
                num_radial=2,
                num_spherical=3,
                edge_channels=2,
            )
            triplets = GraphTriplets.from_edges(edges, len(x))
            result = model(
                x, edges, distances, triplets, triplets.angles(positions, edges), attributes
            )
        assert torch.isfinite(result).all()
        result.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_declared_memory_and_input_failures(data):
    x, positions, edges, attributes = data
    for invalid in (edges.float(), edges.bool(), torch.tensor([[0], [99]])):
        with pytest.raises((TypeError, IndexError)):
            GINELayer(4, 3, 2).double()(x, invalid, attributes)
    with pytest.raises(ValueError):
        NNConvLayer(4, 3, 2, aggregation="median")
    with pytest.raises(ValueError):
        NNConvLayer(4, 3, 2, center_conditioner=True)
    with pytest.raises(ValueError):
        GraphStructuralEncoding.from_edges(edges, 5, num_spatial_codes=1)
    with pytest.raises(ValueError):
        StructuralGraphTransformerLayer(7, num_heads=2)
    with pytest.raises(ValueError):
        GVP(4, -1, 4, 3)
    distances, directions = edge_geometry(positions, edges)
    model = PaiNN(4, 3, hidden_channels=8, num_layers=1).double()
    with pytest.raises(ValueError, match="unit"):
        model(x, edges, distances, torch.zeros_like(directions))
    scipy = pytest.importorskip("scipy")
    assert scipy is not None
    triplets = GraphTriplets.from_edges(edges, len(x))
    directional = DimeNet(
        4, 3, hidden_channels=8, num_layers=1, num_radial=2, num_spherical=3, max_triplet_bytes=1
    ).double()
    with pytest.raises(MemoryError):
        directional(x, edges, distances, triplets, triplets.angles(positions, edges))


def test_optional_directional_dependency_failure_is_explicit(monkeypatch):
    import builtins

    from lambdaforge.nn.models.graph.geometric.DimeNetBasis import _constants

    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    _constants.cache_clear()
    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ImportError, match="graph-directional"):
        DimeNetBasis(2, 2)


def test_directional_cutoff_far_outside_support_has_zero_finite_gradient():
    pytest.importorskip("scipy")
    basis = DimeNetBasis(2, 2).float()
    distances = torch.tensor([1e12, 1e20], requires_grad=True)
    angles = torch.tensor([0.8], requires_grad=True)
    triplets = GraphTriplets(torch.tensor([0]), torch.tensor([1]))
    radial, spherical = basis(distances, angles, triplets)
    assert torch.equal(radial, torch.zeros_like(radial))
    assert torch.equal(spherical, torch.zeros_like(spherical))
    (radial.sum() + spherical.sum()).backward()
    assert torch.equal(distances.grad, torch.zeros_like(distances))


def test_spherical_recurrence_matches_all_validated_orders():
    special = pytest.importorskip("scipy.special")
    z = torch.logspace(-6, 2, 80, dtype=torch.double)
    for order in range(8):
        expected = torch.from_numpy(special.spherical_jn(order, z.numpy()))
        torch.testing.assert_close(DimeNetBasis._bessel(order, z), expected, atol=1e-12, rtol=1e-9)
