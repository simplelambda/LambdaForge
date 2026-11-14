"""Relation-aware and multi-aggregation graph message-passing models."""

from lambdaforge.nn.models.graph.message_passing.DegreeScaler import DegreeScaler
from lambdaforge.nn.models.graph.message_passing.GINE import GINE
from lambdaforge.nn.models.graph.message_passing.GINELayer import GINELayer
from lambdaforge.nn.models.graph.message_passing.NNConv import NNConv
from lambdaforge.nn.models.graph.message_passing.NNConvLayer import NNConvLayer
from lambdaforge.nn.models.graph.message_passing.PNA import PNA
from lambdaforge.nn.models.graph.message_passing.PNAAggregator import PNAAggregator
from lambdaforge.nn.models.graph.message_passing.PNALayer import PNALayer
from lambdaforge.nn.models.graph.message_passing.RelationalGCN import RelationalGCN
from lambdaforge.nn.models.graph.message_passing.RelationalGCNLayer import (
    RelationalGCNLayer,
)

__all__ = [
    "GINE",
    "NNConv",
    "DegreeScaler",
    "GINELayer",
    "NNConvLayer",
    "PNA",
    "PNAAggregator",
    "PNALayer",
    "RelationalGCN",
    "RelationalGCNLayer",
]
