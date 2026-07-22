"""Numerical and contract tests for pure-PyTorch graph models."""

import pytest
import torch

from lambdaforge.experiments import ObjectFactory
from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.graph import (
    GAT,
    GCN,
    GIN,
    GATLayer,
    GCNLayer,
    GraphReadout,
    GraphSAGE,
    GraphSAGELayer,
)
from lambdaforge.nn.pooling.sparse import (
    SparseAttentionPooling,
    SparseMaxPooling,
    SparseMeanPooling,
)


class TestGraphModels:
    """Verify shapes, gradients, attention and sparse edge cases."""

    @staticmethod
    def graph() -> tuple[torch.Tensor, torch.Tensor]:
        """Return one small directed graph with an isolated input pattern."""
        x = torch.randn(5, 4, requires_grad=True)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 1, 4], [1, 2, 0, 2, 4, 3]],
            dtype=torch.long,
        )
        return x, edge_index

    @pytest.mark.parametrize(
        "model",
        [
            GCN(4, 3, hidden_channels=[8, 8], residual=True),
            GraphSAGE(4, 3, hidden_channels=[8]),
            GAT(
                4,
                3,
                hidden_channels=[8],
                heads=[2, 1],
                concatenate_heads=[True, False],
            ),
            GIN(4, 3, hidden_channels=[8], trainable_epsilon=True),
        ],
    )
    def test_stacks_have_expected_shape_and_gradients(self, model: torch.nn.Module) -> None:
        x, edge_index = self.graph()
        output = model(x, edge_index)
        assert output.shape == (5, 3)
        output.square().mean().backward()
        assert x.grad is not None
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    def test_gat_attention_is_normalized_per_destination_and_head(self) -> None:
        x, edge_index = self.graph()
        layer = GATLayer(4, 3, num_heads=2, add_self_loops=True)
        output, routed_edges, attention = layer.forward_with_attention(x, edge_index)
        assert output.shape == (5, 6)
        assert attention.shape == (routed_edges.shape[1], 2)
        for node in range(x.shape[0]):
            incoming = routed_edges[1] == node
            assert torch.allclose(attention[incoming].sum(dim=0), torch.ones(2))

    def test_gcn_uses_directed_source_and_destination_degrees(self) -> None:
        layer = GCNLayer(1, 1, bias=False, add_self_loops=False)
        with torch.no_grad():
            layer.linear.weight.fill_(1.0)
        x = torch.tensor([[1.0], [2.0], [3.0]])
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])

        output = layer(x, edge_index)

        expected = torch.tensor(
            [
                [0.0],
                [1.0 / torch.sqrt(torch.tensor(2.0))],
                [0.5 + torch.sqrt(torch.tensor(2.0))],
            ]
        )
        assert torch.allclose(output, expected)

    def test_gcn_adds_bias_once_after_aggregation(self) -> None:
        layer = GCNLayer(1, 1, bias=True, add_self_loops=False)
        with torch.no_grad():
            layer.linear.weight.zero_()
            assert layer.bias is not None
            layer.bias.fill_(2.0)

        output = layer(torch.ones(3, 1), torch.tensor([[0, 1], [2, 2]]))

        assert layer.linear.bias is None
        assert torch.equal(output, torch.full((3, 1), 2.0))

    def test_gcn_replaces_existing_self_loops_instead_of_duplicating_them(self) -> None:
        layer = GCNLayer(1, 1, bias=False, add_self_loops=True, self_loop_weight=2.0)
        with torch.no_grad():
            layer.linear.weight.fill_(1.0)
        x = torch.tensor([[1.0], [3.0]])
        without_loops = torch.tensor([[0], [1]])
        with_duplicate_loops = torch.tensor([[0, 0, 0, 1], [1, 0, 0, 1]])

        expected = layer(x, without_loops)
        actual = layer(x, with_duplicate_loops)

        assert torch.allclose(actual, expected)

    @pytest.mark.parametrize(
        "model",
        [
            GCN(4, 2, add_self_loops=False),
            GraphSAGE(4, 2),
            GAT(4, 2, add_self_loops=False),
            GIN(4, 2),
        ],
    )
    def test_empty_edge_lists_are_finite(self, model: torch.nn.Module) -> None:
        output = model(torch.randn(3, 4), torch.empty((2, 0), dtype=torch.long))
        assert output.shape == (3, 2)
        assert torch.isfinite(output).all()

    def test_graphsage_max_matches_directed_reference_with_isolated_nodes(self) -> None:
        layer = GraphSAGELayer(
            1,
            1,
            aggregation=Aggregation.MAX,
            root_weight=False,
            bias=False,
        )
        with torch.no_grad():
            layer.neighbor_linear.weight.fill_(1.0)
        x = torch.tensor([[1.0], [4.0], [2.0], [10.0]], requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])

        output = layer(x, edge_index)
        output.sum().backward()

        assert torch.equal(output, torch.tensor([[0.0], [0.0], [0.0], [4.0]]))
        assert x.grad is not None and torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("num_nodes", [0, 3])
    @pytest.mark.parametrize("aggregation", list(Aggregation))
    def test_graphsage_empty_edges_keep_projected_parameters_connected(
        self,
        num_nodes: int,
        aggregation: Aggregation,
    ) -> None:
        layer = GraphSAGELayer(
            2,
            3,
            aggregation=aggregation,
            root_weight=False,
            project_neighbors=True,
            bias=False,
        )
        x = torch.randn(num_nodes, 2, requires_grad=True)

        output = layer(x, torch.empty((2, 0), dtype=torch.int32))
        output.sum().backward()

        assert output.shape == (num_nodes, 3)
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in layer.parameters()
            if parameter.requires_grad
        )

    @pytest.mark.parametrize(
        "model",
        [
            GCN(4, 2),
            GraphSAGE(4, 2),
            GAT(4, 2),
            GIN(4, 2),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
    def test_graph_models_reject_non_integer_edge_indices(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype,
    ) -> None:
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=dtype)
        with pytest.raises(TypeError, match="integer dtype"):
            model(torch.randn(3, 4), edge_index)

    @pytest.mark.parametrize(
        "model",
        [
            GCN(4, 2),
            GraphSAGE(4, 2),
            GAT(4, 2),
            GIN(4, 2),
        ],
    )
    def test_graph_models_accept_non_long_integer_edge_indices(
        self,
        model: torch.nn.Module,
    ) -> None:
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
        output = model(torch.randn(3, 4), edge_index)
        assert output.shape == (3, 2)

    @pytest.mark.parametrize(
        "model_type",
        [GCN, GraphSAGE, GAT, GIN],
    )
    @pytest.mark.parametrize("normalization", ["layernorm", "batchnorm"])
    def test_graph_stacks_support_node_feature_normalizations(
        self,
        model_type: type[torch.nn.Module],
        normalization: str,
    ) -> None:
        model = model_type(4, 2, hidden_channels=[4], normalization=normalization)
        x, edge_index = self.graph()
        output = model(x, edge_index)
        assert output.shape == (5, 2)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("model_type", [GCN, GraphSAGE, GAT, GIN])
    def test_graph_stacks_reject_instance_norm_layout(
        self,
        model_type: type[torch.nn.Module],
    ) -> None:
        with pytest.raises(ValueError, match="InstanceNorm is incompatible"):
            model_type(4, 2, hidden_channels=[4], normalization="instancenorm")

    @pytest.mark.parametrize("model_type", [GCN, GraphSAGE, GAT])
    def test_graph_stacks_reject_spatial_batch_norm_variants(
        self,
        model_type: type[torch.nn.Module],
    ) -> None:
        with pytest.raises(ValueError, match=r"BatchNorm\(dim=1\)"):
            model_type(
                4,
                2,
                hidden_channels=[4],
                normalization="batchnorm",
                normalization_kwargs={"dim": 2},
            )

    @pytest.mark.parametrize("model_type", [GCN, GraphSAGE, GAT])
    def test_graph_stacks_support_group_norm_over_node_features(
        self,
        model_type: type[torch.nn.Module],
    ) -> None:
        model = model_type(
            4,
            2,
            hidden_channels=[4],
            normalization="groupnorm",
            normalization_kwargs={"num_groups": 2},
        )
        x, edge_index = self.graph()
        assert model(x, edge_index).shape == (5, 2)

    def test_gat_is_constructible_from_yaml_specification(self) -> None:
        model = ObjectFactory.build(
            {
                "target": "lambdaforge.nn.models.GAT",
                "params": {
                    "in_channels": 4,
                    "out_channels": 2,
                    "hidden_channels": [6],
                    "heads": [2, 1],
                    "concatenate_heads": [True, False],
                    "activation": "gelu",
                },
            }
        )
        assert isinstance(model, GAT)

    def test_gat_rejects_non_divisible_concatenated_width(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            GAT(4, 2, hidden_channels=[7], heads=[2, 1])

    def test_sparse_readout_encapsulates_graph_level_prediction(self) -> None:
        x, edge_index = self.graph()
        group_index = torch.tensor([0, 0, 0, 1, 1])
        model = GraphReadout(
            encoder=GCN(4, 6, hidden_channels=[8]),
            pooling=SparseAttentionPooling(6, hidden_features=4),
            head=torch.nn.Linear(6, 2),
        )
        output = model(x, edge_index, group_index)
        assert output.shape == (2, 2)
        output.sum().backward()
        assert x.grad is not None

    def test_sparse_pooling_defines_empty_group_values(self) -> None:
        x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        groups = torch.tensor([0, 2])
        mean = SparseMeanPooling()(x, groups, num_groups=4)
        maximum = SparseMaxPooling()(x, groups, num_groups=4)
        assert torch.equal(mean[1], torch.zeros(2))
        assert torch.equal(maximum[3], torch.zeros(2))
