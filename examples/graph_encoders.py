"""Tiny generic graph encoders; no Work, dataset, cluster or domain-specific policy."""

from __future__ import annotations

import argparse

import torch

from lambdaforge.nn.models.graph import (
    GINE,
    GVPGNN,
    DimeNet,
    GraphStructuralEncoding,
    GraphTriplets,
    NNConv,
    PaiNN,
    SchNet,
    StructuralGraphTransformer,
    edge_geometry,
)


def main() -> None:
    """Exercise small synthetic node embeddings, optionally including Bessel directionality."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directional", action="store_true", help="Requires graph-directional extra"
    )
    args = parser.parse_args()
    torch.manual_seed(7)
    features = torch.randn(4, 5)
    coordinates = torch.randn(4, 3)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    attributes = torch.randn(edges.shape[1], 2)
    distances, directions = edge_geometry(coordinates, edges)
    outputs = {
        "GINE": GINE(5, 8, 2, hidden_channels=(16,))(features, edges, attributes),
        "NNConv/bases": NNConv(
            5, 8, 2, hidden_channels=(16,), num_bases=3, center_conditioner=True
        )(features, edges, attributes),
        "SchNet": SchNet(5, 8, hidden_channels=16, num_layers=2, edge_channels=2)(
            features, edges, distances, attributes
        ),
        "GVP-GNN": GVPGNN(
            5, 8, scalar_channels=16, vector_channels=4, num_layers=2, edge_channels=2
        )(features, edges, directions[:, None, :], attributes),
        "PaiNN": PaiNN(5, 8, hidden_channels=16, num_layers=2, edge_channels=2)(
            features, edges, distances, directions, attributes
        ),
    }
    encoding = GraphStructuralEncoding.from_edges(edges, len(features))
    outputs["Structural/global"] = StructuralGraphTransformer(
        5, 8, hidden_channels=16, num_layers=2
    )(
        features[None],
        encoding.spatial_indices[None],
        encoding.in_degree[None],
        encoding.out_degree[None],
    )[0]
    if args.directional:
        triplets = GraphTriplets.from_edges(edges, len(features))
        angles = triplets.angles(coordinates, edges)
        for variant in ("bilinear", "factorized"):
            model = DimeNet(
                5,
                8,
                hidden_channels=16,
                num_layers=2,
                num_radial=3,
                num_spherical=3,
                edge_channels=2,
                interaction_type=variant,
            )
            outputs[f"DimeNet/{variant}"] = model(
                features, edges, distances, triplets, angles, attributes
            )
    for name, value in outputs.items():
        assert value.shape == (4, 8) and torch.isfinite(value).all()
        value.square().mean().backward()
        print(f"{name}: {tuple(value.shape)}, finite forward/backward")


if __name__ == "__main__":
    main()
