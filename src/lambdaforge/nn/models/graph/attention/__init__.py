"Sparse and memory-bounded global structural graph attention."

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.nn.models.graph.attention.GATv2 import GATv2
    from lambdaforge.nn.models.graph.attention.GATv2Layer import GATv2Layer
    from lambdaforge.nn.models.graph.attention.GraphStructuralEncoding import (
        GraphStructuralEncoding,
    )
    from lambdaforge.nn.models.graph.attention.GraphTransformer import GraphTransformer
    from lambdaforge.nn.models.graph.attention.GraphTransformerLayer import (
        GraphTransformerLayer,
    )
    from lambdaforge.nn.models.graph.attention.StructuralGraphTransformer import (
        StructuralGraphTransformer,
    )
    from lambdaforge.nn.models.graph.attention.StructuralGraphTransformerLayer import (
        StructuralGraphTransformerLayer,
    )

LazyExports.install(
    __name__,
    {
        "GraphStructuralEncoding": (
            "lambdaforge.nn.models.graph.attention.GraphStructuralEncoding",
            "GraphStructuralEncoding",
        ),
        "StructuralGraphTransformer": (
            "lambdaforge.nn.models.graph.attention.StructuralGraphTransformer",
            "StructuralGraphTransformer",
        ),
        "StructuralGraphTransformerLayer": (
            "lambdaforge.nn.models.graph.attention.StructuralGraphTransformerLayer",
            "StructuralGraphTransformerLayer",
        ),
        "GATv2": ("lambdaforge.nn.models.graph.attention.GATv2", "GATv2"),
        "GATv2Layer": ("lambdaforge.nn.models.graph.attention.GATv2Layer", "GATv2Layer"),
        "GraphTransformer": (
            "lambdaforge.nn.models.graph.attention.GraphTransformer",
            "GraphTransformer",
        ),
        "GraphTransformerLayer": (
            "lambdaforge.nn.models.graph.attention.GraphTransformerLayer",
            "GraphTransformerLayer",
        ),
    },
)
__all__ = [
    "GraphStructuralEncoding",
    "StructuralGraphTransformer",
    "StructuralGraphTransformerLayer",
    "GATv2",
    "GATv2Layer",
    "GraphTransformer",
    "GraphTransformerLayer",
]
