"""Public graph composition/source controls on disposable synthetic tensors only."""

from __future__ import annotations

import argparse

import torch

from lambdaforge.nn.models import ECMP
from lambdaforge.nn.models.graph import (
    DimeNet,
    DimeNetBasis,
    GCNLayer,
    GINELayer,
    GraphSAGELayer,
    GraphTriplets,
    PaiNN,
    PaiNNLayer,
    PNALayer,
    SchNet,
    SchNetLayer,
    edge_geometry,
)


def main() -> None:
    """Demonstrate independent messages, one update and explicit geometric routes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directional", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(11)
    x = torch.randn(5, 4, requires_grad=True)
    positions = torch.randn(5, 3)
    a = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    b = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])  # pair 0/1 deliberately overlaps
    ea, eb = torch.randn(4, 2), torch.randn(4, 2)
    za = torch.tensor(1.0, requires_grad=True)
    zb = torch.tensor(0.0, requires_grad=True)
    # A consumer may compute 1+C(e)-C(0). LF does NOT learn or invent that policy.
    signed_modulation = torch.tensor([1.5, -0.5, 1.0, 0.0], requires_grad=True)
    outputs = {
        "GCN/modulated": GCNLayer(4, 4, add_self_loops=False)(x, a, signed_modulation),
        "GraphSAGE/modulated": GraphSAGELayer(4, 4)(x, a, signed_modulation),
    }
    ga, gb = GINELayer(4, 4, 2), GINELayer(4, 4, 2)
    sum_a = ga.aggregate(ga.messages(x, a, ea), a, len(x))
    sum_b = gb.aggregate(gb.messages(x, b, eb), b, len(x))
    outputs["GINE/common-update"] = ga.update(x, za * sum_a + zb * sum_b)

    ma = ECMP(4, hidden_dim=8, num_layers=1, edge_dim=2)
    mb = ECMP(4, hidden_dim=8, num_layers=1, edge_dim=2)
    h = ma.project_inputs(x)
    sum_a = ma.aggregate(ma.messages(h, a, ea), a, len(x))
    sum_b = mb.aggregate(mb.messages(h, b, eb), b, len(x))
    outputs["ECMP/common-update"] = ma.update(h, za * sum_a + zb * sum_b)

    pa = PNALayer(
        4,
        4,
        edge_channels=2,
        message_channels=8,
        scalers=["identity", "amplification"],
        average_log_degree=1.0,
    )
    pb = PNALayer(
        4,
        4,
        edge_channels=2,
        message_channels=8,
        scalers=["identity", "amplification"],
        average_log_degree=1.0,
    )
    sum_a = pa.aggregate(pa.messages(x, a, ea), a, len(x), degree_dtype=x.dtype)
    sum_b = pb.aggregate(pb.messages(x, b, eb), b, len(x), degree_dtype=x.dtype)
    outputs["PNA/common-update"] = pa.update(x, za * sum_a + zb * sum_b)

    edges = torch.cat((a, b), dim=1)
    attrs = torch.cat((ea, eb))
    distances, directions = edge_geometry(positions, edges)
    source = torch.tensor(0.0, requires_grad=True)  # distance disabled, valid geometry retained
    outputs["SchNet/distance-closed"] = SchNet(
        4, 4, hidden_channels=8, num_layers=2, edge_channels=2, cutoff=2.0
    )(x, edges, distances, attrs, distance_weights=source)
    outputs["PaiNN/distance-closed"] = PaiNN(
        4, 4, hidden_channels=8, num_layers=2, edge_channels=2, cutoff=2.0
    )(x, edges, distances, directions, attrs, distance_weights=source)
    # One native node update with an authored support policy per relation.
    is_a = torch.arange(edges.shape[1]) < a.shape[1]
    edge_multiplier = torch.cat((za.expand(a.shape[1]), zb.expand(b.shape[1])))
    scalar_layer = SchNetLayer(4, num_radial=3, cutoff=2.0, edge_channels=2)
    radial, support = scalar_layer.radial_inputs(distances, distance_weights=source)
    support = torch.where(is_a, torch.ones_like(support), support) * edge_multiplier
    outputs["SchNet/explicit-support"] = scalar_layer.forward_from_radial(
        x, edges, radial, support, attrs
    )
    vector_layer = PaiNNLayer(4, num_radial=3, cutoff=2.0, edge_channels=2)
    radial, support = vector_layer.radial_inputs(distances, distance_weights=source)
    support = torch.where(is_a, torch.ones_like(support), support) * edge_multiplier
    scalar_state, vector_state = vector_layer.forward_from_radial(
        x, x.new_zeros((len(x), 4, 3)), edges, radial, directions, support, attrs
    )
    outputs["PaiNN/explicit-support"] = scalar_state
    outputs["PaiNN/vector-state"] = vector_state
    if args.directional:
        triplets = GraphTriplets.from_edges(edges, len(x))
        angles = triplets.angles(positions, edges)
        # Separate native bases preserve the non-spatial relation's authored support.
        # Distances/angles stay physical; only derived source routes are controlled.
        basis_a = DimeNetBasis(2, 3, cutoff=100.0)
        basis_b = DimeNetBasis(2, 3, cutoff=2.0)
        ra, sa = basis_a(distances, angles, triplets, distance_weights=source)
        rb, sb = basis_b(distances, angles, triplets, distance_weights=source)
        is_a = torch.arange(edges.shape[1]) < a.shape[1]
        radial = torch.where(is_a[:, None], ra, rb)
        spherical = torch.where(is_a[triplets.incoming, None], sa, sb)
        messages = torch.cat((za.expand(a.shape[1]), zb.expand(b.shape[1])))
        for variant in ("bilinear", "factorized"):
            encoder = DimeNet(
                4,
                4,
                hidden_channels=8,
                num_layers=2,
                num_radial=2,
                num_spherical=3,
                edge_channels=2,
                interaction_type=variant,
            )
            outputs[f"DimeNet/{variant}"] = encoder.forward_from_basis(
                x, edges, radial, spherical, triplets, attrs, message_weights=messages
            )
    for label, result in outputs.items():
        assert torch.isfinite(result).all()
        print(f"{label}: {tuple(result.shape)}")
    sum(value.square().mean() for value in outputs.values()).backward()
    assert x.grad is not None and source.grad is not None
    assert za.grad is not None and zb.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(source.grad)
    assert torch.isfinite(za.grad) and torch.isfinite(zb.grad)
    print("Finite gradients; no Jobs, datasets or application state modified.")


if __name__ == "__main__":
    main()
