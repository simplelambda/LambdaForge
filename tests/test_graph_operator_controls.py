"""Public composition/source controls without consumer graph-equation reimplementations."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from lambdaforge.nn.models import ECMP
from lambdaforge.nn.models.graph import (
    GCN,
    DimeNet,
    DimeNetBasis,
    DimeNetInteraction,
    DimeNetPlusPlusInteraction,
    GCNLayer,
    GINELayer,
    GINLayer,
    GraphSAGE,
    GraphSAGELayer,
    GraphTriplets,
    PaiNN,
    PaiNNLayer,
    PNALayer,
    SchNet,
    SchNetLayer,
    StructuralGraphTransformerLayer,
    edge_geometry,
)
from lambdaforge.nn.models.Scatter import Scatter


@pytest.fixture
def sample():
    torch.manual_seed(41)
    x = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    pos = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    # Pair 0/1 occurs twice: two distinct relation incidences, not a coalesced edge.
    edges = torch.tensor([[0, 1, 0, 1, 1, 2, 2, 3], [1, 0, 1, 0, 2, 1, 3, 2]])
    attrs = torch.randn(8, 2, dtype=torch.double, requires_grad=True)
    return x, pos, edges, attrs


def gcn_reference(layer, x, edges, modulation=None):
    source, target = edges
    topological = x.new_ones(len(source))
    mod = x.new_ones(len(source)) if modulation is None else modulation
    if layer.add_self_loops:
        nonself = source != target
        nodes = torch.arange(len(x))
        source = torch.cat((source[nonself], nodes))
        target = torch.cat((target[nonself], nodes))
        topological = torch.cat(
            (topological[nonself], x.new_full((len(x),), layer.self_loop_weight))
        )
        mod = torch.cat((mod[nonself], x.new_ones(len(x))))
    indeg = x.new_zeros(len(x))
    outdeg = x.new_zeros(len(x))
    for s, t, w in zip(source, target, topological, strict=True):
        outdeg[s] += w
        indeg[t] += w
    transformed = layer.linear(x)
    result = x.new_zeros((len(x), layer.out_channels))
    for s, t, w, m in zip(source, target, topological, mod, strict=True):
        result[t] += transformed[s] * (
            w
            / outdeg[s].clamp_min(layer.degree_epsilon).sqrt()
            / indeg[t].clamp_min(layer.degree_epsilon).sqrt()
            * m
        )
    return result if layer.bias is None else result + layer.bias


@pytest.mark.parametrize("loops", [False, True])
def test_gcn_signed_modulation_keeps_topological_degrees_and_one_root(loops):
    x = torch.tensor([[1.0], [2.0], [3.0], [9.0]], dtype=torch.double, requires_grad=True)
    edges = torch.tensor([[0, 0, 1, 0], [1, 1, 2, 0]])
    weights = torch.tensor([2.0, -1.0, 0.0, -9.0], dtype=torch.double, requires_grad=True)
    layer = GCNLayer(1, 1, bias=False, add_self_loops=loops, self_loop_weight=2).double()
    with torch.no_grad():
        layer.linear.weight.fill_(1)
    torch.testing.assert_close(layer(x, edges, weights), gcn_reference(layer, x, edges, weights))
    assert torch.equal(layer(x, edges), layer(x, edges, torch.ones_like(weights)))
    torch.testing.assert_close(layer(x, edges), gcn_reference(layer, x, edges))
    layer(x, edges, weights).sum().backward()
    assert torch.isfinite(weights.grad).all()
    if loops:
        assert weights.grad[-1] == 0  # Canonical loops replace supplied loop modulation.
    empty = edges[:, :0]
    assert torch.isfinite(layer(x, empty, weights[:0])).all()


def test_sage_mean_uses_incidence_count_not_weight_sum():
    layer = GraphSAGELayer(1, 1, root_weight=False, bias=False).double()
    with torch.no_grad():
        layer.neighbor_linear.weight.fill_(1)
    x = torch.tensor([[2.0], [5.0], [7.0], [9.0]], dtype=torch.double)
    edges = torch.tensor([[0, 0, 1], [2, 2, 2]])
    weights = torch.tensor([2.0, -1.0, 0.0], dtype=torch.double, requires_grad=True)
    expected = torch.tensor([[0.0], [0.0], [2.0 / 3], [0.0]], dtype=torch.double)
    torch.testing.assert_close(layer(x, edges, weights), expected)
    assert torch.equal(layer(x, edges), layer(x, edges, torch.ones(3, dtype=torch.double)))
    layer(x, edges, weights).sum().backward()
    torch.testing.assert_close(
        weights.grad, torch.tensor([2.0 / 3, 2.0 / 3, 5.0 / 3], dtype=torch.double)
    )


@pytest.mark.parametrize("kind", [GCN, GraphSAGE])
def test_weighted_stacks_serialize_without_new_state_and_preserve_default(sample, kind):
    x, _, edges, _ = sample
    model = kind(4, 3, hidden_channels=[8]).double()
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)
    default = model(x, edges)
    assert torch.equal(default, model(x, edges, torch.ones(8, dtype=torch.double)))
    weights = torch.randn(8, dtype=torch.double, requires_grad=True)
    result = model(x, edges, weights)
    torch.testing.assert_close(restored(x, edges, weights), result)
    result.square().sum().backward()
    assert torch.isfinite(weights.grad).all()
    assert not any("message_weight" in name for name in model.state_dict())


@pytest.mark.parametrize(
    "bad",
    [
        torch.ones(3),
        torch.ones(8, 1),
        torch.ones(8, dtype=torch.long),
        torch.full((8,), float("nan")),
    ],
)
def test_invalid_modulators_are_rejected(sample, bad):
    x, _, edges, _ = sample
    for layer in [GCNLayer(4, 3).double(), GraphSAGELayer(4, 3).double()]:
        with pytest.raises((TypeError, ValueError)):
            layer(x, edges, bad)


@pytest.mark.parametrize("edge_aware", [False, True])
def test_gin_composition_has_exactly_one_root_and_update(edge_aware):
    layer = (
        GINELayer(1, 1, 1, epsilon=0.5).double()
        if edge_aware
        else GINLayer(1, 1, epsilon=0.5).double()
    )
    layer.mlp = nn.Identity()
    if edge_aware:
        with torch.no_grad():
            layer.edge_projection.weight.zero_()
    x = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.double)
    a = torch.tensor([[0, 1], [2, 2]])
    b = torch.tensor([[0], [2]])
    z = torch.tensor([0.25, 0.0], dtype=torch.double, requires_grad=True)
    ma = (
        layer.messages(x, a, torch.zeros(2, 1, dtype=torch.double))
        if edge_aware
        else layer.messages(x, a)
    )
    mb = (
        layer.messages(x, b, torch.zeros(1, 1, dtype=torch.double))
        if edge_aware
        else layer.messages(x, b)
    )
    aa = layer.aggregate(ma, a, 3)
    ab = layer.aggregate(mb, b, 3)
    actual = layer.update(x, z[0] * aa + z[1] * ab)
    expected = 1.5 * x + torch.tensor([[0.0], [0.0], [1.25]], dtype=torch.double)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(z.grad, torch.tensor([5.0, 2.0], dtype=torch.double))
    torch.testing.assert_close(layer.update(x, torch.zeros_like(x)), 1.5 * x)


def legacy_ecmp(model, x, edges, attrs):
    h = model.input_proj(x)
    for message, update, norm in zip(
        model.message_mlps, model.update_mlps, model.norms, strict=True
    ):
        if edges.numel():
            raw = message(torch.cat((h[edges[0]], h[edges[1]], attrs), -1))
            aggregate = Scatter.reduce(raw, edges[1], len(h), model.aggregation)
        else:
            aggregate = h.new_zeros((len(h), model.hidden_dim))
        delta = model.dropout(norm(update(torch.cat((h, aggregate), -1))))
        h = h + delta if model.use_residual else delta
    return h


def legacy_pna(layer, x, edges, attrs):
    messages = layer.pre_mlp(torch.cat((x[edges[1]], x[edges[0]], attrs), -1))
    degree = torch.bincount(edges[1], minlength=len(x)).to(x)
    aggregates = [aggregator.reduce(messages, edges[1], len(x)) for aggregator in layer.aggregators]
    combined = [
        scaler.scale(
            a,
            degree,
            average_degree=layer.average_degree,
            average_log_degree=layer.average_log_degree,
            epsilon=layer.epsilon,
        )
        for scaler in layer.scalers
        for a in aggregates
    ]
    return layer.dropout(layer.post_mlp(torch.cat((x, *combined), -1)))


@pytest.mark.parametrize("family", ["ecmp", "pna"])
@pytest.mark.parametrize("amp", [False, True])
def test_composition_default_equals_original_equations_and_strict_serialization(
    sample, family, amp
):
    x, _, edges, attrs = sample
    x, attrs = x.float(), attrs.float()
    if family == "ecmp":
        model = ECMP(4, hidden_dim=4, num_layers=2, edge_dim=2)
        reference = legacy_ecmp
    else:
        model = PNALayer(
            4, 3, edge_channels=2, scalers=["identity", "amplification", "attenuation"]
        )
        reference = legacy_pna
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        expected = reference(model, x, edges, attrs)
        actual = model(x, edges, attrs)
    assert torch.equal(actual, expected)
    restored = copy.deepcopy(model)
    restored.load_state_dict(before, strict=True)
    assert torch.equal(restored(x, edges, attrs), model(x, edges, attrs))
    assert set(before) == set(model.state_dict())


@pytest.mark.parametrize("family", ["ecmp", "pna"])
def test_relation_composition_uses_public_stages_and_zero_relation_does_not_leak(sample, family):
    x, _, edges, attrs = sample
    z = torch.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True)
    if family == "ecmp":
        first = ECMP(4, hidden_dim=4, num_layers=1, edge_dim=2, norm="identity").double()
        second = copy.deepcopy(first)
        h = first.project_inputs(x)
    else:
        first = PNALayer(4, 3, edge_channels=2, scalers=["identity", "amplification"]).double()
        second = copy.deepcopy(first)
        h = x
    ea, eb = edges[:, :4], edges[:, 4:]
    aa = first.aggregate(first.messages(h, ea, attrs[:4]), ea, len(x))
    ab = second.aggregate(second.messages(h, eb, attrs[4:]), eb, len(x))
    result = first.update(h, z[0] * aa + z[1] * ab)
    torch.testing.assert_close(result, first.update(h, aa))
    # Replacing attributes of a disabled relation must not affect the common update.
    changed = second.aggregate(second.messages(h, eb, attrs[4:] + 100), eb, len(x))
    torch.testing.assert_close(first.update(h, z[0] * aa + z[1] * changed), result)
    result.square().sum().backward()
    assert torch.isfinite(z.grad).all()
    torch.testing.assert_close(attrs.grad[4:], torch.zeros_like(attrs.grad[4:]))
    empty = edges[:, :0]
    empty_messages = first.messages(h, empty, attrs[:0])
    assert torch.isfinite(first.aggregate(empty_messages, empty, len(x))).all()


@pytest.mark.parametrize("family", ["schnet", "painn"])
def test_distance_control_closes_every_radial_cutoff_route_but_not_other_features(sample, family):
    x, pos, edges, attrs = sample
    distances, directions = edge_geometry(pos, edges)
    distances = distances.detach().requires_grad_()
    closed = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    if family == "schnet":
        model = SchNet(4, 3, hidden_channels=8, num_layers=2, edge_channels=2, cutoff=1.0).double()

        def call(d, g):
            return model(x, edges, d, attrs, distance_weights=g)
    else:
        model = PaiNN(4, 3, hidden_channels=8, num_layers=2, edge_channels=2, cutoff=1.0).double()

        def call(d, g):
            return model(x, edges, d, directions, attrs, distance_weights=g)

    baseline = call(distances, None)
    assert torch.equal(baseline, call(distances, torch.ones_like(distances)))
    a = call(distances, closed)
    b = call(distances * 10, closed)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    derivative = torch.autograd.grad(a.sum(), distances, retain_graph=True)[0]
    assert torch.equal(derivative, torch.zeros_like(derivative))
    a.square().sum().backward()
    assert closed.grad is not None and torch.isfinite(closed.grad)
    layer = model.layers[0]
    radial, support = layer.radial_inputs(distances, distance_weights=closed)
    assert torch.equal(radial, torch.zeros_like(radial))
    assert torch.equal(support, torch.ones_like(support))
    with pytest.raises(ValueError):
        call(torch.full_like(distances, float("nan")), closed)


@pytest.mark.parametrize("variant", ["bilinear", "factorized"])
def test_directional_distance_source_control_and_public_basis_boundary(sample, variant):
    pytest.importorskip("scipy")
    x, pos, edges, attrs = sample
    distances, _ = edge_geometry(pos, edges)
    distances = distances.detach().requires_grad_()
    triplets = GraphTriplets.from_edges(edges, len(x))
    angles = triplets.angles(pos, edges)
    model = DimeNet(
        4,
        3,
        hidden_channels=8,
        num_layers=2,
        num_radial=2,
        num_spherical=3,
        edge_channels=2,
        interaction_type=variant,
    ).double()
    g = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    actual = model(x, edges, distances, triplets, angles, attrs, distance_weights=g)
    changed = model(x, edges, distances * 50, triplets, angles, attrs, distance_weights=g)
    assert torch.equal(actual, changed)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), distances, retain_graph=True)[0],
        torch.zeros_like(distances),
    )
    actual.sum().backward(retain_graph=True)
    assert torch.isfinite(g.grad)
    radial, spherical = model.basis(distances, angles, triplets, distance_weights=g)
    assert torch.equal(radial, torch.ones_like(radial))
    torch.testing.assert_close(
        model.forward_from_basis(x, edges, radial, spherical, triplets, attrs), actual
    )
    # Separate relation-specific supports, reused natively rather than changing physical lengths.
    near = DimeNetBasis(2, 3, cutoff=1.0).double()
    far = DimeNetBasis(2, 3, cutoff=100.0).double()
    r_near, s_near = near(distances, angles, triplets)
    r_far, s_far = far(distances, angles, triplets)
    covalent = torch.tensor([True, True, True, True, False, False, False, False])
    radial = torch.where(covalent[:, None], r_far, r_near)
    spherical = torch.where(covalent[triplets.incoming, None], s_far, s_near)
    assert (radial[covalent].abs().sum(-1) > 0).all()
    assert torch.isfinite(
        model.forward_from_basis(x, edges, radial, spherical, triplets, attrs)
    ).all()
    with pytest.raises(ValueError):
        model(x, edges, torch.zeros_like(distances), triplets, angles, attrs, distance_weights=g)


@pytest.mark.parametrize("variant", ["bilinear", "factorized"])
def test_directional_relation_paths_are_suppressed_without_thresholding(sample, variant):
    pytest.importorskip("scipy")
    x, pos, edges, attrs = sample
    distances, _ = edge_geometry(pos, edges)
    triplets = GraphTriplets.from_edges(edges, len(x))
    angles = triplets.angles(pos, edges)
    model = DimeNet(
        4,
        3,
        hidden_channels=8,
        num_layers=2,
        num_radial=2,
        num_spherical=3,
        edge_channels=2,
        interaction_type=variant,
    ).double()
    types = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    z = torch.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True)
    weights = z[types]
    assert torch.equal(
        model(x, edges, distances, triplets, angles, attrs),
        model(
            x, edges, distances, triplets, angles, attrs, message_weights=torch.ones_like(weights)
        ),
    )
    radial, spherical = model.basis(distances, angles, triplets)
    result = model.forward_from_basis(
        x, edges, radial, spherical, triplets, attrs, message_weights=weights
    )
    changed_attrs = attrs + types[:, None] * 100
    changed_r = radial + types[:, None] * 100
    changed_s = spherical + types[triplets.incoming, None] * 100
    changed = model.forward_from_basis(
        x, edges, changed_r, changed_s, triplets, changed_attrs, message_weights=weights
    )
    torch.testing.assert_close(changed, result, rtol=0, atol=0)
    result.square().mean().backward(retain_graph=True)
    assert torch.isfinite(z.grad).all()
    assert torch.equal(attrs.grad[types == 1], torch.zeros_like(attrs.grad[types == 1]))
    empty_relations = torch.zeros_like(weights, requires_grad=True)
    all_closed = model.forward_from_basis(
        x, edges, radial, spherical, triplets, attrs, message_weights=empty_relations
    )
    expected = model.root(x)
    for readout in model.node_readouts:
        expected = expected + readout(x.new_zeros((len(x), model.hidden_channels)))
    torch.testing.assert_close(all_closed, expected, rtol=0, atol=0)
    # Mixed typed triplets were retained, not rebuilt or selected by gate values.
    assert (types[triplets.incoming] != types[triplets.outgoing]).any()


@pytest.mark.parametrize("kind", [DimeNetInteraction, DimeNetPlusPlusInteraction])
def test_directional_block_fractional_weights_have_explicit_path_semantics(kind):
    layer = kind(3, 2, 2, num_before_skip=0, num_after_skip=0).double()
    x = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    r = torch.randn(4, 2, dtype=torch.double)
    s = torch.randn(2, 4, dtype=torch.double)
    t = GraphTriplets(torch.tensor([0, 1]), torch.tensor([2, 3]))
    g = torch.tensor([0.2, 0.7, 0.4, 0.8], dtype=torch.double, requires_grad=True)
    # Scaling spherical rows applies the exact product to either linear interaction.
    scaled = s * (g[t.incoming] * g[t.outgoing])[:, None]
    expected = layer(x, r, scaled, t) * g[:, None]
    actual = layer(x, r, s, t, message_weights=g)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert torch.isfinite(g.grad).all()


def test_disabled_attributes_are_not_disabled_relations(sample):
    x, _, edges, attrs = sample
    layer = GINELayer(4, 4, 2).double()
    messages = layer.messages(x, edges, torch.zeros_like(attrs))
    assert torch.equal(messages, x[edges[0]].relu())
    aggregate = layer.aggregate(messages, edges, len(x))
    assert aggregate.abs().sum() > 0
    expected = layer.update(x, aggregate * 0)
    torch.testing.assert_close(layer.update(x, torch.zeros_like(aggregate)), expected)


@pytest.mark.parametrize("kind", [GCNLayer, GraphSAGELayer])
def test_weighted_permutation_reversal_and_disjoint_graph_isolation(sample, kind):
    x, _, edges, _ = sample
    layer = kind(4, 3).double()
    g = torch.randn(8, dtype=torch.double, requires_grad=True)
    output = layer(x, edges, g)
    order = torch.randperm(edges.shape[1])
    perm = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.argsort(perm)
    torch.testing.assert_close(layer(x[perm], inverse[edges[:, order]], g[order]), output[perm])
    # Reversing the storage orientation of undirected pairs swaps the two
    # directed incidences and their aligned attributes/modulators, not their meaning.
    reversed_order = torch.arange(edges.shape[1]).reshape(-1, 2).flip(1).flatten()
    torch.testing.assert_close(layer(x, edges[:, reversed_order], g[reversed_order]), output)
    both_x = torch.cat((x, x + 20), 0)
    both_edges = torch.cat((edges, edges + len(x)), 1)
    both_g = torch.cat((g, g))
    result = layer(both_x, both_edges, both_g)
    torch.testing.assert_close(result[: len(x)], output)
    torch.testing.assert_close(result[len(x) :], layer(x + 20, edges, g))


@pytest.mark.parametrize("family", ["schnet", "painn", "directional"])
@pytest.mark.parametrize("amp", [False, True])
def test_controlled_geometry_has_finite_gradients_serializes_and_respects_o3(sample, family, amp):
    x, pos, edges, attrs = sample
    x = x.detach().float().requires_grad_()
    pos = pos.detach().float()
    attrs = attrs.detach().float()
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(rotation) > 0:
        rotation[:, 0] *= -1
    transformed = pos @ rotation.T + torch.tensor([3.0, 4.0, 5.0])
    d, u = edge_geometry(pos, edges)
    d2, u2 = edge_geometry(transformed, edges)
    source = torch.tensor(0.3, requires_grad=True)
    if family == "schnet":
        model = SchNet(4, 3, hidden_channels=8, num_layers=2, edge_channels=2, cutoff=5)
        args = (x, edges, d, attrs)
        args2 = (x, edges, d2, attrs)
        extra = {}
    elif family == "painn":
        model = PaiNN(4, 3, hidden_channels=8, num_layers=2, edge_channels=2, cutoff=5)
        args = (x, edges, d, u, attrs)
        args2 = (x, edges, d2, u2, attrs)
        extra = {}
    else:
        pytest.importorskip("scipy")
        model = DimeNet(
            4, 3, hidden_channels=8, num_layers=2, num_radial=2, num_spherical=3, edge_channels=2
        )
        t = GraphTriplets.from_edges(edges, len(x))
        args = (x, edges, d, t, t.angles(pos, edges), attrs)
        args2 = (x, edges, d2, t, t.angles(transformed, edges), attrs)
        extra = {
            "message_weights": torch.tensor(
                [1.0, 1.0, 0.0, 0.0, 0.7, 0.7, 0.5, 0.5], requires_grad=True
            )
        }
    saved = copy.deepcopy(model)
    saved.load_state_dict(model.state_dict(), strict=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        actual = model(*args, distance_weights=source, **extra)
        restored = saved(*args, distance_weights=source, **extra)
        changed = model(*args2, distance_weights=source, **extra)
    assert torch.equal(actual, restored)
    torch.testing.assert_close(
        changed, actual, atol=0.03 if amp else 2e-5, rtol=0.03 if amp else 2e-5
    )
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(source.grad)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_controlled_directional_basis_path_keeps_memory_and_shape_guards(sample):
    pytest.importorskip("scipy")
    x, pos, edges, attrs = sample
    d, _ = edge_geometry(pos, edges)
    t = GraphTriplets.from_edges(edges, len(x))
    model = DimeNet(
        4,
        3,
        hidden_channels=8,
        num_layers=1,
        num_radial=2,
        num_spherical=3,
        edge_channels=2,
        max_triplet_bytes=1,
    ).double()
    r, s = model.basis(d, t.angles(pos, edges), t)
    with pytest.raises(MemoryError):
        model.forward_from_basis(
            x, edges, r, s, t, attrs, message_weights=torch.zeros(8, dtype=x.dtype)
        )


@pytest.mark.parametrize("family", ["schnet", "painn"])
def test_public_radial_input_boundary_preserves_relation_specific_support(sample, family):
    x, pos, edges, attrs = sample
    d, u = edge_geometry(pos, edges)
    if family == "schnet":
        branch = SchNetLayer(4, num_radial=3, radius=2, cutoff=None, edge_channels=2).double()
        common = SchNetLayer(4, num_radial=3, radius=2, cutoff=0.001, edge_channels=2).double()
    else:
        branch = PaiNNLayer(4, num_radial=3, radius=2, cutoff=None, edge_channels=2).double()
        common = PaiNNLayer(4, num_radial=3, radius=2, cutoff=0.001, edge_channels=2).double()
    r, support = branch.radial_inputs(d)
    assert torch.equal(support, torch.ones_like(support))
    gate = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.double, requires_grad=True
    )
    v = x.new_zeros((len(x), 4, 3))
    if family == "schnet":
        actual = common.forward_from_radial(x, edges, r, support * gate, attrs)
        changed = common.forward_from_radial(x, edges, r, torch.zeros_like(support), attrs)
        reference = common.forward_from_radial(
            x, edges, r * 100, torch.zeros_like(support), attrs + 100
        )
        assert actual.abs().sum() > 0
        torch.testing.assert_close(changed, reference, atol=0, rtol=0)
        actual.sum().backward()
    else:
        s, w = common.forward_from_radial(x, v, edges, r, u, support * gate, attrs)
        zero = common.forward_from_radial(x, v, edges, r, u, torch.zeros_like(support), attrs)
        zero2 = common.forward_from_radial(
            x, v, edges, r * 100, u, torch.zeros_like(support), attrs + 100
        )
        for a, b in zip(zero, zero2, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        (s.square().sum() + w.square().sum()).backward()
    assert torch.isfinite(gate.grad).all()


def test_pna_public_split_matches_amp_forward_with_declared_degree_dtype(sample):
    x, _, edges, attrs = sample
    x, attrs = x.float(), attrs.float()
    layer = PNALayer(4, 3, edge_channels=2, scalers=["identity", "amplification", "attenuation"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = layer(x, edges, attrs)
        messages = layer.messages(x, edges, attrs)
        aggregate = layer.aggregate(messages, edges, len(x), degree_dtype=x.dtype)
        actual = layer.update(x, aggregate)
    assert torch.equal(actual, expected)


def test_dense_score_guard_accounts_for_fp32_structural_bias_under_bf16_autocast():
    layer = StructuralGraphTransformerLayer(4, num_heads=2, max_attention_bytes=100)
    x = torch.randn(1, 4, 4, dtype=torch.bfloat16)
    # 32 bf16 scores would be 64 bytes, but fp32 spatial bias promotes them to 128.
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(MemoryError):
        layer(
            x,
            torch.ones(1, 4, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
        )
