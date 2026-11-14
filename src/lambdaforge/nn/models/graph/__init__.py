"Composable graph neural-network layers and model stacks."

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.nn.models.graph.attention import (
        GATv2,
        GATv2Layer,
        GraphStructuralEncoding,
        GraphTransformer,
        GraphTransformerLayer,
        StructuralGraphTransformer,
        StructuralGraphTransformerLayer,
    )
    from lambdaforge.nn.models.graph.equivariant import (
        EGNN,
        GVP,
        GVPGNN,
        EGNNLayer,
        EquivariantOutputMode,
        EquivariantTensorAdapter,
        GVPGNNLayer,
        PaiNN,
        PaiNNLayer,
        TensorFieldNetwork,
        VectorDropout,
    )
    from lambdaforge.nn.models.graph.GAT import GAT
    from lambdaforge.nn.models.graph.GATLayer import GATLayer
    from lambdaforge.nn.models.graph.GCN import GCN
    from lambdaforge.nn.models.graph.GCNLayer import GCNLayer
    from lambdaforge.nn.models.graph.geometric import (
        CosineCutoff,
        DimeNet,
        DimeNetBasis,
        DimeNetInteraction,
        DimeNetPlusPlusInteraction,
        GaussianRadialBasis,
        GraphTriplets,
        SchNet,
        SchNetLayer,
        SineRadialBasis,
    )
    from lambdaforge.nn.models.graph.GIN import GIN
    from lambdaforge.nn.models.graph.GINLayer import GINLayer
    from lambdaforge.nn.models.graph.GraphReadout import GraphReadout
    from lambdaforge.nn.models.graph.GraphSAGE import GraphSAGE
    from lambdaforge.nn.models.graph.GraphSAGELayer import GraphSAGELayer
    from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill
    from lambdaforge.nn.models.graph.message_passing import (
        GINE,
        PNA,
        DegreeScaler,
        GINELayer,
        NNConv,
        NNConvLayer,
        PNAAggregator,
        PNALayer,
        RelationalGCN,
        RelationalGCNLayer,
    )
    from lambdaforge.nn.models.graph.validation import edge_geometry

LazyExports.install(
    __name__,
    {
        "GINE": ("lambdaforge.nn.models.graph.message_passing", "GINE"),
        "NNConv": ("lambdaforge.nn.models.graph.message_passing", "NNConv"),
        "GraphStructuralEncoding": (
            "lambdaforge.nn.models.graph.attention",
            "GraphStructuralEncoding",
        ),
        "StructuralGraphTransformer": (
            "lambdaforge.nn.models.graph.attention",
            "StructuralGraphTransformer",
        ),
        "StructuralGraphTransformerLayer": (
            "lambdaforge.nn.models.graph.attention",
            "StructuralGraphTransformerLayer",
        ),
        "GVP": ("lambdaforge.nn.models.graph.equivariant", "GVP"),
        "GVPGNN": ("lambdaforge.nn.models.graph.equivariant", "GVPGNN"),
        "GVPGNNLayer": ("lambdaforge.nn.models.graph.equivariant", "GVPGNNLayer"),
        "PaiNN": ("lambdaforge.nn.models.graph.equivariant", "PaiNN"),
        "PaiNNLayer": ("lambdaforge.nn.models.graph.equivariant", "PaiNNLayer"),
        "VectorDropout": ("lambdaforge.nn.models.graph.equivariant", "VectorDropout"),
        "CosineCutoff": ("lambdaforge.nn.models.graph.geometric", "CosineCutoff"),
        "DimeNet": ("lambdaforge.nn.models.graph.geometric", "DimeNet"),
        "DimeNetBasis": ("lambdaforge.nn.models.graph.geometric", "DimeNetBasis"),
        "DimeNetInteraction": ("lambdaforge.nn.models.graph.geometric", "DimeNetInteraction"),
        "DimeNetPlusPlusInteraction": (
            "lambdaforge.nn.models.graph.geometric",
            "DimeNetPlusPlusInteraction",
        ),
        "GaussianRadialBasis": ("lambdaforge.nn.models.graph.geometric", "GaussianRadialBasis"),
        "GraphTriplets": ("lambdaforge.nn.models.graph.geometric", "GraphTriplets"),
        "SchNet": ("lambdaforge.nn.models.graph.geometric", "SchNet"),
        "SchNetLayer": ("lambdaforge.nn.models.graph.geometric", "SchNetLayer"),
        "SineRadialBasis": ("lambdaforge.nn.models.graph.geometric", "SineRadialBasis"),
        "GINELayer": ("lambdaforge.nn.models.graph.message_passing", "GINELayer"),
        "NNConvLayer": ("lambdaforge.nn.models.graph.message_passing", "NNConvLayer"),
        "edge_geometry": ("lambdaforge.nn.models.graph.validation", "edge_geometry"),
        "DegreeScaler": ("lambdaforge.nn.models.graph.message_passing", "DegreeScaler"),
        "EGNN": ("lambdaforge.nn.models.graph.equivariant", "EGNN"),
        "EGNNLayer": ("lambdaforge.nn.models.graph.equivariant", "EGNNLayer"),
        "EquivariantOutputMode": (
            "lambdaforge.nn.models.graph.equivariant",
            "EquivariantOutputMode",
        ),
        "EquivariantTensorAdapter": (
            "lambdaforge.nn.models.graph.equivariant",
            "EquivariantTensorAdapter",
        ),
        "GAT": ("lambdaforge.nn.models.graph.GAT", "GAT"),
        "GATLayer": ("lambdaforge.nn.models.graph.GATLayer", "GATLayer"),
        "GATv2": ("lambdaforge.nn.models.graph.attention", "GATv2"),
        "GATv2Layer": ("lambdaforge.nn.models.graph.attention", "GATv2Layer"),
        "GCN": ("lambdaforge.nn.models.graph.GCN", "GCN"),
        "GCNLayer": ("lambdaforge.nn.models.graph.GCNLayer", "GCNLayer"),
        "GIN": ("lambdaforge.nn.models.graph.GIN", "GIN"),
        "GINLayer": ("lambdaforge.nn.models.graph.GINLayer", "GINLayer"),
        "GraphReadout": ("lambdaforge.nn.models.graph.GraphReadout", "GraphReadout"),
        "GraphSAGE": ("lambdaforge.nn.models.graph.GraphSAGE", "GraphSAGE"),
        "GraphSAGELayer": ("lambdaforge.nn.models.graph.GraphSAGELayer", "GraphSAGELayer"),
        "GraphSelfLoopFill": ("lambdaforge.nn.models.graph.GraphSelfLoopFill", "GraphSelfLoopFill"),
        "GraphTransformer": ("lambdaforge.nn.models.graph.attention", "GraphTransformer"),
        "GraphTransformerLayer": ("lambdaforge.nn.models.graph.attention", "GraphTransformerLayer"),
        "PNA": ("lambdaforge.nn.models.graph.message_passing", "PNA"),
        "PNAAggregator": ("lambdaforge.nn.models.graph.message_passing", "PNAAggregator"),
        "PNALayer": ("lambdaforge.nn.models.graph.message_passing", "PNALayer"),
        "RelationalGCN": ("lambdaforge.nn.models.graph.message_passing", "RelationalGCN"),
        "RelationalGCNLayer": ("lambdaforge.nn.models.graph.message_passing", "RelationalGCNLayer"),
        "TensorFieldNetwork": ("lambdaforge.nn.models.graph.equivariant", "TensorFieldNetwork"),
    },
)
__all__ = [
    "GINE",
    "NNConv",
    "GraphStructuralEncoding",
    "StructuralGraphTransformer",
    "StructuralGraphTransformerLayer",
    "GVP",
    "GVPGNN",
    "GVPGNNLayer",
    "PaiNN",
    "PaiNNLayer",
    "VectorDropout",
    "CosineCutoff",
    "DimeNet",
    "DimeNetBasis",
    "DimeNetInteraction",
    "DimeNetPlusPlusInteraction",
    "GaussianRadialBasis",
    "GraphTriplets",
    "SchNet",
    "SchNetLayer",
    "SineRadialBasis",
    "GINELayer",
    "NNConvLayer",
    "edge_geometry",
    "DegreeScaler",
    "EGNN",
    "EGNNLayer",
    "EquivariantOutputMode",
    "EquivariantTensorAdapter",
    "GAT",
    "GATLayer",
    "GATv2",
    "GATv2Layer",
    "GCN",
    "GCNLayer",
    "GIN",
    "GINLayer",
    "GraphReadout",
    "GraphSAGE",
    "GraphSAGELayer",
    "GraphSelfLoopFill",
    "GraphTransformer",
    "GraphTransformerLayer",
    "PNA",
    "PNAAggregator",
    "PNALayer",
    "RelationalGCN",
    "RelationalGCNLayer",
    "TensorFieldNetwork",
]
