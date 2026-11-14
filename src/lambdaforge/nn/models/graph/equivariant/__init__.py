"E(n)-equivariant graph models with no compiled dependency."

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.nn.models.graph.equivariant.EGNN import EGNN
    from lambdaforge.nn.models.graph.equivariant.EGNNLayer import EGNNLayer
    from lambdaforge.nn.models.graph.equivariant.EquivariantOutputMode import (
        EquivariantOutputMode,
    )
    from lambdaforge.nn.models.graph.equivariant.EquivariantTensorAdapter import (
        EquivariantTensorAdapter,
    )
    from lambdaforge.nn.models.graph.equivariant.GVP import GVP
    from lambdaforge.nn.models.graph.equivariant.GVPGNN import GVPGNN
    from lambdaforge.nn.models.graph.equivariant.GVPGNNLayer import GVPGNNLayer
    from lambdaforge.nn.models.graph.equivariant.PaiNN import PaiNN
    from lambdaforge.nn.models.graph.equivariant.PaiNNLayer import PaiNNLayer
    from lambdaforge.nn.models.graph.equivariant.TensorFieldNetwork import TensorFieldNetwork
    from lambdaforge.nn.models.graph.equivariant.VectorDropout import VectorDropout

LazyExports.install(
    __name__,
    {
        "EGNN": ("lambdaforge.nn.models.graph.equivariant.EGNN", "EGNN"),
        "EGNNLayer": ("lambdaforge.nn.models.graph.equivariant.EGNNLayer", "EGNNLayer"),
        "EquivariantOutputMode": (
            "lambdaforge.nn.models.graph.equivariant.EquivariantOutputMode",
            "EquivariantOutputMode",
        ),
        "EquivariantTensorAdapter": (
            "lambdaforge.nn.models.graph.equivariant.EquivariantTensorAdapter",
            "EquivariantTensorAdapter",
        ),
        "TensorFieldNetwork": (
            "lambdaforge.nn.models.graph.equivariant.TensorFieldNetwork",
            "TensorFieldNetwork",
        ),
        "GVP": ("lambdaforge.nn.models.graph.equivariant.GVP", "GVP"),
        "GVPGNN": ("lambdaforge.nn.models.graph.equivariant.GVPGNN", "GVPGNN"),
        "GVPGNNLayer": ("lambdaforge.nn.models.graph.equivariant.GVPGNNLayer", "GVPGNNLayer"),
        "PaiNN": ("lambdaforge.nn.models.graph.equivariant.PaiNN", "PaiNN"),
        "PaiNNLayer": ("lambdaforge.nn.models.graph.equivariant.PaiNNLayer", "PaiNNLayer"),
        "VectorDropout": ("lambdaforge.nn.models.graph.equivariant.VectorDropout", "VectorDropout"),
    },
)
__all__ = [
    "EGNN",
    "EGNNLayer",
    "EquivariantOutputMode",
    "EquivariantTensorAdapter",
    "TensorFieldNetwork",
    "GVP",
    "GVPGNN",
    "GVPGNNLayer",
    "PaiNN",
    "PaiNNLayer",
    "VectorDropout",
]
