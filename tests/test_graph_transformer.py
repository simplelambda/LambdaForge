"""Contract tests for the native sparse edge-aware graph Transformer."""

from __future__ import annotations

import pytest
import torch

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.nn.models.graph.attention.GraphTransformer import GraphTransformer
from lambdaforge.nn.models.graph.attention.GraphTransformerLayer import (
    GraphTransformerLayer,
)
from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill


class TestGraphTransformer:
    """Verify sparse attention, edge conditioning and configurable blocks."""

    @staticmethod
    def graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one directed graph with two-dimensional edge features."""
        generator = torch.Generator().manual_seed(31)
        x = torch.randn(5, 4, generator=generator)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 1, 4], [1, 2, 0, 2, 4, 3]],
            dtype=torch.long,
        )
        edge_features = torch.randn(6, 2, generator=generator)
        return x, edge_index, edge_features

    def test_layer_attention_is_normalized_and_all_parameters_receive_gradients(self) -> None:
        x, edge_index, edge_features = self.graph()
        x.requires_grad_()
        edge_features.requires_grad_()
        layer = GraphTransformerLayer(
            4,
            6,
            num_heads=2,
            edge_channels=2,
            feedforward_channels=9,
            beta=True,
            self_loop_edge_fill=GraphSelfLoopFill.MEAN,
        )
        parameter_count = sum(parameter.numel() for parameter in layer.parameters())

        output, routed, attention = layer.forward_with_attention(
            x,
            edge_index,
            edge_features,
        )
        output.square().mean().backward()

        assert output.shape == (5, 6)
        assert attention.shape == (routed.shape[1], 2)
        for node in range(x.shape[0]):
            incoming = routed[1] == node
            assert torch.allclose(attention[incoming].sum(dim=0), torch.ones(2))
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert edge_features.grad is not None and torch.isfinite(edge_features.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in layer.parameters()
            if parameter.requires_grad
        )
        assert sum(parameter.numel() for parameter in layer.parameters()) == parameter_count

    def test_stack_supports_per_layer_heads_widths_and_post_norm(self) -> None:
        x, edge_index, edge_features = self.graph()
        model = GraphTransformer(
            4,
            3,
            hidden_channels=[8],
            heads=[2, 1],
            concatenate_heads=[True, False],
            edge_channels=2,
            feedforward_channels=[12, 7],
            pre_norm=[True, False],
            feature_dropout=[0.0, 0.0],
            attention_dropout=[0.0, 0.0],
            feedforward_dropout=[0.0, 0.0],
        )

        output, routed, attention = model.forward_with_attention(
            x,
            edge_index,
            edge_features,
        )

        assert output.shape == (5, 3)
        assert len(routed) == len(attention) == 2
        assert attention[0].shape[1] == 2
        assert attention[1].shape[1] == 1

    def test_edge_features_only_affect_reachable_destinations_without_loops(self) -> None:
        layer = GraphTransformerLayer(
            3,
            3,
            edge_channels=1,
            add_self_loops=False,
            residual=False,
            normalization="identity",
        )
        layer.eval()
        x = torch.randn(3, 3)
        edge_index = torch.tensor([[0], [1]])

        baseline = layer(x, edge_index, torch.zeros(1, 1))
        changed = layer(x, edge_index, torch.ones(1, 1))

        assert torch.equal(baseline[[0, 2]], changed[[0, 2]])
        assert not torch.allclose(baseline[1], changed[1])

    def test_self_loops_are_replaced_and_empty_graphs_remain_finite(self) -> None:
        x = torch.randn(3, 4)
        duplicated = torch.tensor([[0, 0, 1, 1], [1, 0, 1, 1]])
        layer = GraphTransformerLayer(4, 4, num_heads=2)

        output, routed, attention = layer.forward_with_attention(x, duplicated)

        self_edges = routed[:, routed[0] == routed[1]]
        assert torch.equal(self_edges, torch.tensor([[0, 1, 2], [0, 1, 2]]))
        assert torch.isfinite(output).all()
        assert torch.isfinite(attention).all()

        empty_layer = GraphTransformerLayer(
            4,
            4,
            num_heads=2,
            add_self_loops=False,
            pre_norm=False,
        )
        empty = empty_layer(x, torch.empty((2, 0), dtype=torch.long))
        assert empty.shape == (3, 4)
        assert torch.isfinite(empty).all()

    def test_disjoint_union_equals_independent_execution(self) -> None:
        x, edge_index, edge_features = self.graph()
        first_nodes = 3
        first_mask = (edge_index[0] < first_nodes) & (edge_index[1] < first_nodes)
        second_mask = (edge_index[0] >= first_nodes) & (edge_index[1] >= first_nodes)
        first_edges = edge_index[:, first_mask]
        second_edges = edge_index[:, second_mask] - first_nodes
        disjoint_edges = torch.cat(
            (first_edges, second_edges + torch.tensor([[first_nodes], [first_nodes]])),
            dim=1,
        )
        disjoint_features = torch.cat(
            (edge_features[first_mask], edge_features[second_mask]),
            dim=0,
        )
        model = GraphTransformer(
            4,
            4,
            hidden_channels=[4],
            heads=2,
            edge_channels=2,
        )
        model.eval()

        combined = model(x, disjoint_edges, disjoint_features)
        expected = torch.cat(
            (
                model(x[:first_nodes], first_edges, edge_features[first_mask]),
                model(x[first_nodes:], second_edges, edge_features[second_mask]),
            ),
            dim=0,
        )

        assert torch.allclose(combined, expected, atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
    def test_non_integer_edge_indices_are_rejected(self, dtype: torch.dtype) -> None:
        layer = GraphTransformerLayer(4, 4)
        with pytest.raises(TypeError, match="integer dtype"):
            layer(torch.randn(3, 4), torch.tensor([[0, 1], [1, 2]], dtype=dtype))

    def test_configuration_errors_are_explicit(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            GraphTransformerLayer(4, 5, num_heads=2)
        with pytest.raises(ValueError, match="beta"):
            GraphTransformerLayer(4, 4, residual=False, beta=True)
        with pytest.raises(ValueError, match="exactly 2"):
            GraphTransformer(4, 2, hidden_channels=[4], heads=[1])

    @pytest.mark.parametrize(
        ("name", "value", "error", "message"),
        [
            ("concatenate_heads", 1, TypeError, "boolean"),
            ("add_self_loops", 0, TypeError, "boolean"),
            ("pre_norm", "false", TypeError, "boolean"),
            ("residual", 1, TypeError, "boolean"),
            ("beta", "false", TypeError, "boolean"),
            ("bias", 1, TypeError, "boolean"),
            ("feature_dropout", False, TypeError, "real number"),
            ("attention_dropout", "0.1", TypeError, "real number"),
            ("feedforward_dropout", float("nan"), ValueError, "finite"),
        ],
    )
    def test_layer_rejects_coerced_flags_and_invalid_dropout(
        self,
        name: str,
        value: object,
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            GraphTransformerLayer(4, 4, **{name: value})

    @pytest.mark.parametrize(
        ("fill", "error", "message"),
        [
            (True, TypeError, "self_loop_edge_fill"),
            ("median", ValueError, "GraphSelfLoopFill"),
            (float("inf"), ValueError, "finite"),
        ],
    )
    def test_self_loop_fill_is_validated_eagerly(
        self,
        fill: object,
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            GraphTransformerLayer(4, 4, self_loop_edge_fill=fill)

        enum_layer = GraphTransformerLayer(4, 4, self_loop_edge_fill="mean")
        numeric_layer = GraphTransformerLayer(4, 4, self_loop_edge_fill=2)
        assert enum_layer.self_loop_edge_fill is GraphSelfLoopFill.MEAN
        assert numeric_layer.self_loop_edge_fill == 2.0

    def test_zero_node_stack_preserves_empty_shapes(self) -> None:
        model = GraphTransformer(
            4,
            3,
            hidden_channels=[6],
            heads=[2, 1],
            edge_channels=2,
        )

        output, routed, attention = model.forward_with_attention(
            torch.empty((0, 4)),
            torch.empty((2, 0), dtype=torch.int32),
            torch.empty((0, 2)),
        )

        assert output.shape == (0, 3)
        assert [tuple(edges.shape) for edges in routed] == [(2, 0), (2, 0)]
        assert [tuple(weights.shape) for weights in attention] == [(0, 2), (0, 1)]

    def test_yaml_object_factory_builds_graph_transformer_stack(self) -> None:
        model = ObjectFactory.build(
            {
                "target": (
                    "lambdaforge.nn.models.graph.attention.GraphTransformer.GraphTransformer"
                ),
                "params": {
                    "in_channels": 4,
                    "out_channels": 3,
                    "hidden_channels": [6],
                    "heads": [2, 1],
                    "concatenate_heads": [True, False],
                    "edge_channels": 2,
                    "feedforward_channels": [9, 7],
                    "self_loop_edge_fill": ["mean", 0.5],
                    "pre_norm": [True, False],
                },
            }
        )
        x, edge_index, edge_features = self.graph()

        assert isinstance(model, GraphTransformer)
        assert model(x, edge_index, edge_features).shape == (5, 3)
