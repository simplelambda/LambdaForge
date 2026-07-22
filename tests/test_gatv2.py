"""Numerical, configuration and edge-alignment tests for native GATv2."""

from __future__ import annotations

import pytest
import torch

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.nn.models.graph.attention.GATv2 import GATv2
from lambdaforge.nn.models.graph.attention.GATv2Layer import GATv2Layer
from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill


class TestGATv2:
    """Verify dynamic sparse attention without optional graph dependencies."""

    @staticmethod
    def graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return node and edge features for a small directed graph."""
        x = torch.randn(5, 4, requires_grad=True)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 1, 4], [1, 2, 0, 2, 4, 3]],
            dtype=torch.long,
        )
        edge_features = torch.randn(edge_index.shape[1], 2, requires_grad=True)
        return x, edge_index, edge_features

    def test_layer_implements_dynamic_edge_aware_score_equation(self) -> None:
        layer = GATv2Layer(
            2,
            1,
            concatenate_heads=False,
            edge_channels=1,
            negative_slope=0.0,
            add_self_loops=False,
            bias=False,
        )
        with torch.no_grad():
            layer.source_projection.weight.copy_(torch.tensor([[1.0, 0.0]]))
            assert layer.destination_projection is not None
            layer.destination_projection.weight.copy_(torch.tensor([[0.0, 1.0]]))
            assert layer.edge_projection is not None
            layer.edge_projection.weight.copy_(torch.tensor([[2.0]]))
            layer.attention.fill_(1.0)
        x = torch.tensor([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        edge_index = torch.tensor([[0, 2], [1, 1]])
        edge_features = torch.tensor([[1.0], [0.0]])

        output, routed_edges, alpha = layer.forward_with_attention(
            x,
            edge_index,
            edge_features,
        )

        expected_alpha = torch.softmax(torch.tensor([23.0, 25.0]), dim=0)
        expected_node = (expected_alpha * torch.tensor([1.0, 5.0])).sum()
        assert torch.equal(routed_edges, edge_index)
        assert torch.allclose(alpha[:, 0], expected_alpha)
        assert torch.allclose(output[:, 0], torch.tensor([0.0, expected_node, 0.0]))

    def test_stack_shapes_gradients_and_per_layer_configuration(self) -> None:
        x, edge_index, edge_features = self.graph()
        model = GATv2(
            4,
            3,
            hidden_channels=[8, 8],
            heads=[2, 2, 3],
            concatenate_heads=[True, True, False],
            share_weights=[False, True, False],
            edge_channels=2,
            activation=["elu", "gelu"],
            normalization=["layernorm", "layernorm"],
            feature_dropout=[0.0, 0.1, 0.0],
            attention_dropout=[0.0, 0.0, 0.0],
            negative_slope=[0.1, 0.2, 0.3],
            add_self_loops=[True, False, True],
            self_loop_fill=["mean", 0.0, 1.5],
            residual=[False, True, False],
            bias=[True, False, True],
        )

        output = model(x, edge_index, edge_features)
        output.square().mean().backward()

        assert output.shape == (5, 3)
        assert [layer.share_weights for layer in model.layers] == [False, True, False]
        assert [layer.out_channels for layer in model.layers] == [8, 8, 3]
        assert x.grad is not None
        assert edge_features.grad is not None
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    def test_attention_is_normalized_before_dropout(self) -> None:
        x, edge_index, edge_features = self.graph()
        layer = GATv2Layer(
            4,
            3,
            num_heads=2,
            edge_channels=2,
            attention_dropout=0.85,
            add_self_loops=True,
        )
        layer.train()

        _, routed_edges, alpha = layer.forward_with_attention(x, edge_index, edge_features)

        assert alpha.shape == (routed_edges.shape[1], 2)
        for node in range(x.shape[0]):
            incoming = routed_edges[1] == node
            assert torch.allclose(
                alpha[incoming].sum(dim=0),
                torch.ones(2),
                atol=1e-6,
            )

    @pytest.mark.parametrize(
        "fill",
        [GraphSelfLoopFill.ZERO, "mean", 2.5],
    )
    def test_self_loops_are_replaced_and_edge_features_remain_aligned(
        self,
        fill: GraphSelfLoopFill | str | float,
    ) -> None:
        torch.manual_seed(9)
        layer = GATv2Layer(
            2,
            2,
            edge_channels=1,
            add_self_loops=True,
            self_loop_fill=fill,
        )
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        base_edges = torch.tensor([[0], [1]])
        duplicated_edges = torch.tensor([[0, 0, 0, 1], [1, 0, 0, 1]])
        base_features = torch.tensor([[3.0]])
        duplicated_features = torch.tensor([[3.0], [90.0], [91.0], [92.0]])

        expected, expected_edges, expected_alpha = layer.forward_with_attention(
            x,
            base_edges,
            base_features,
        )
        actual, actual_edges, actual_alpha = layer.forward_with_attention(
            x,
            duplicated_edges,
            duplicated_features,
        )

        routed = torch.tensor([[0, 0, 1], [1, 0, 1]])
        assert torch.equal(expected_edges, routed)
        assert torch.equal(actual_edges, routed)
        assert torch.allclose(actual, expected)
        assert torch.allclose(actual_alpha, expected_alpha)

    def test_shared_projection_and_head_averaging_preserve_requested_width(self) -> None:
        layer = GATv2Layer(
            4,
            5,
            num_heads=3,
            concatenate_heads=False,
            share_weights=True,
            add_self_loops=False,
        )
        output = layer(
            torch.randn(4, 4),
            torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32),
        )

        assert layer.destination_projection is None
        assert layer.out_channels == 5
        assert output.shape == (4, 5)

    def test_empty_edges_and_empty_edge_features_are_finite(self) -> None:
        model = GATv2(
            4,
            2,
            hidden_channels=[6],
            heads=[2, 1],
            edge_channels=3,
            add_self_loops=False,
        )
        output = model(
            torch.randn(3, 4),
            torch.empty((2, 0), dtype=torch.int32),
            torch.empty((0, 3)),
        )

        assert output.shape == (3, 2)
        assert torch.isfinite(output).all()

    def test_yaml_object_factory_constructs_complete_stack(self) -> None:
        model = ObjectFactory.build(
            {
                "target": "lambdaforge.nn.models.graph.attention.GATv2.GATv2",
                "params": {
                    "in_channels": 4,
                    "out_channels": 2,
                    "hidden_channels": [6],
                    "heads": [2, 1],
                    "concatenate_heads": [True, False],
                    "share_weights": [True, False],
                    "edge_channels": 1,
                    "activation": "gelu",
                    "normalization": "layernorm",
                    "self_loop_fill": "mean",
                },
            }
        )

        output = model(
            torch.randn(3, 4),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
            torch.randn(2, 1),
        )
        assert isinstance(model, GATv2)
        assert output.shape == (3, 2)

    @pytest.mark.parametrize(
        ("arguments", "error", "message"),
        [
            ({"in_channels": True, "out_channels_per_head": 2}, TypeError, "integer"),
            ({"in_channels": 2, "out_channels_per_head": 0}, ValueError, "positive"),
            (
                {"in_channels": 2, "out_channels_per_head": 2, "edge_channels": -1},
                ValueError,
                "non-negative",
            ),
            (
                {"in_channels": 2, "out_channels_per_head": 2, "attention_dropout": 1.0},
                ValueError,
                r"\[0, 1\)",
            ),
            (
                {"in_channels": 2, "out_channels_per_head": 2, "self_loop_fill": "median"},
                ValueError,
                "GraphSelfLoopFill",
            ),
            (
                {"in_channels": 2, "out_channels_per_head": 2, "self_loop_fill": float("nan")},
                ValueError,
                "finite",
            ),
        ],
    )
    def test_layer_rejects_invalid_configuration(
        self,
        arguments: dict[str, object],
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            GATv2Layer(**arguments)

    def test_layer_validates_graph_and_edge_feature_contracts(self) -> None:
        x = torch.randn(3, 2)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        layer = GATv2Layer(2, 2, edge_channels=1)

        with pytest.raises(ValueError, match="required"):
            layer(x, edge_index)
        with pytest.raises(ValueError, match="shape"):
            layer(x, edge_index, torch.randn(3, 1))
        with pytest.raises(TypeError, match="integer dtype"):
            layer(x, edge_index.float(), torch.randn(2, 1))
        with pytest.raises(IndexError, match="outside"):
            layer(x, torch.tensor([[0, 3], [1, 2]]), torch.randn(2, 1))

    @pytest.mark.parametrize(
        ("arguments", "error", "message"),
        [
            ({"hidden_channels": [4], "heads": [1]}, ValueError, "exactly 2"),
            ({"hidden_channels": [7], "heads": [2, 1]}, ValueError, "divisible"),
            ({"hidden_channels": [4], "heads": [True, 1]}, TypeError, "integer"),
            ({"hidden_channels": [4], "feature_dropout": [0.0, 1.0]}, ValueError, "Dropout"),
            (
                {"hidden_channels": [4], "attention_dropout": [0.0, float("nan")]},
                ValueError,
                "finite",
            ),
            ({"hidden_channels": [4], "residual": [True]}, ValueError, "exactly 2"),
        ],
    )
    def test_stack_rejects_inconsistent_per_layer_configuration(
        self,
        arguments: dict[str, object],
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            GATv2(4, 2, **arguments)
