"""Composable graph neural-network layers and model stacks."""

from lambdaforge.nn.models.graph.attention import (
    GATv2,
    GATv2Layer,
    GraphTransformer,
    GraphTransformerLayer,
)
from lambdaforge.nn.models.graph.equivariant import (
    EGNN,
    EGNNLayer,
    EquivariantOutputMode,
    EquivariantTensorAdapter,
    TensorFieldNetwork,
)
from lambdaforge.nn.models.graph.GAT import GAT
from lambdaforge.nn.models.graph.GATLayer import GATLayer
from lambdaforge.nn.models.graph.GCN import GCN
from lambdaforge.nn.models.graph.GCNLayer import GCNLayer
from lambdaforge.nn.models.graph.GIN import GIN
from lambdaforge.nn.models.graph.GINLayer import GINLayer
from lambdaforge.nn.models.graph.GraphReadout import GraphReadout
from lambdaforge.nn.models.graph.GraphSAGE import GraphSAGE
from lambdaforge.nn.models.graph.GraphSAGELayer import GraphSAGELayer
from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill
from lambdaforge.nn.models.graph.message_passing import (
    PNA,
    DegreeScaler,
    PNAAggregator,
    PNALayer,
    RelationalGCN,
    RelationalGCNLayer,
)

__all__ = [
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
