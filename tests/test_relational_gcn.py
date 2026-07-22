"""Numerical and configuration contracts for relational graph convolutions."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.graph.message_passing.RelationalGCN import RelationalGCN
from lambdaforge.nn.models.graph.message_passing.RelationalGCNLayer import (
    RelationalGCNLayer,
)


class TestRelationalGCN:
    """Verify directed routing, relation reduction, bases and stack behavior."""

    @pytest.mark.parametrize(
        ("aggregation", "expected"),
        [
            (Aggregation.SUM, 14.0),
            (Aggregation.MEAN, 10.0),
        ],
    )
    def test_manual_relation_weights_and_pair_normalization(
        self,
        aggregation: Aggregation,
        expected: float,
    ) -> None:
        layer = RelationalGCNLayer(
            1,
            1,
            num_relations=2,
            aggregation=aggregation,
            root_weight=False,
            bias=False,
        )
        assert layer.relation_weight is not None
        with torch.no_grad():
            layer.relation_weight.copy_(torch.tensor([[[2.0]], [[3.0]]]))

        x = torch.tensor([[1.0], [3.0], [0.0], [2.0]])
        edge_index = torch.tensor([[0, 1, 3], [2, 2, 2]])
        edge_types = torch.tensor([0, 0, 1])

        output = layer(x, edge_index, edge_types)

        assert torch.equal(output[[0, 1, 3]], torch.zeros(3, 1))
        assert output[2].item() == pytest.approx(expected)

    def test_basis_decomposition_matches_explicit_relation_matrices(self) -> None:
        explicit = RelationalGCNLayer(
            2,
            2,
            num_relations=3,
            aggregation="sum",
            root_weight=False,
            bias=False,
        )
        decomposed = RelationalGCNLayer(
            2,
            2,
            num_relations=3,
            num_bases=2,
            aggregation="sum",
            root_weight=False,
            bias=False,
        )
        bases = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 2.0]],
                [[0.0, 1.0], [-1.0, 0.0]],
            ]
        )
        coefficients = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 2.0],
            ]
        )
        expected_weights = torch.einsum("rb,bio->rio", coefficients, bases)
        assert explicit.relation_weight is not None
        assert decomposed.basis_weight is not None
        assert decomposed.basis_coefficients is not None
        with torch.no_grad():
            explicit.relation_weight.copy_(expected_weights)
            decomposed.basis_weight.copy_(bases)
            decomposed.basis_coefficients.copy_(coefficients)

        x = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
        edge_types = torch.tensor([0, 1, 2, 2])

        assert not explicit.uses_basis_decomposition
        assert decomposed.uses_basis_decomposition
        assert torch.equal(decomposed.effective_relation_weights(), expected_weights)
        assert torch.equal(
            decomposed(x, edge_index, edge_types),
            explicit(x, edge_index, edge_types),
        )

    @pytest.mark.parametrize("aggregation", ["sum", "mean"])
    @pytest.mark.parametrize("num_bases", [None, 2])
    def test_message_chunks_match_unchunked_relational_aggregation(
        self,
        aggregation: str,
        num_bases: int | None,
    ) -> None:
        reference = RelationalGCNLayer(
            3,
            2,
            num_relations=4,
            num_bases=num_bases,
            message_chunk_size=None,
            aggregation=aggregation,
        )
        chunked = RelationalGCNLayer(
            3,
            2,
            num_relations=4,
            num_bases=num_bases,
            message_chunk_size=1,
            aggregation=aggregation,
        )
        chunked.load_state_dict(reference.state_dict())
        x = torch.randn(6, 3)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 1, 0], [3, 3, 3, 4, 4, 4, 5, 5]],
            dtype=torch.int32,
        )
        edge_types = torch.tensor([2, 0, 2, 1, 1, 3, 0, 3], dtype=torch.int16)

        expected = reference(x, edge_index, edge_types)
        actual = chunked(x, edge_index, edge_types)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("num_bases", [None, 2])
    def test_sparse_relation_path_never_calls_batched_matrix_multiply(
        self,
        monkeypatch: pytest.MonkeyPatch,
        num_bases: int | None,
    ) -> None:
        forbidden = Mock(side_effect=AssertionError("torch.bmm must not be used by R-GCN"))
        monkeypatch.setattr(torch, "bmm", forbidden)
        layer = RelationalGCNLayer(
            2,
            2,
            num_relations=3,
            num_bases=num_bases,
            message_chunk_size=2,
        )

        output = layer(
            torch.randn(4, 2),
            torch.tensor([[0, 1, 2], [2, 2, 3]]),
            torch.tensor([0, 1, 2]),
        )

        assert output.shape == (4, 2)
        forbidden.assert_not_called()

    def test_root_projection_handles_isolated_nodes_and_bias_is_added_once(self) -> None:
        layer = RelationalGCNLayer(
            1,
            1,
            num_relations=2,
            aggregation="sum",
            root_weight=True,
            bias=True,
        )
        assert layer.relation_weight is not None
        assert layer.root_linear is not None
        assert layer.bias is not None
        with torch.no_grad():
            layer.relation_weight.zero_()
            layer.root_linear.weight.fill_(2.0)
            layer.bias.fill_(1.0)

        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        edge_index = torch.tensor([[0, 1, 3], [2, 2, 2]])
        edge_types = torch.tensor([0, 1, 0])

        output = layer(x, edge_index, edge_types)

        assert torch.equal(output, 2.0 * x + 1.0)
        assert output[0].item() == pytest.approx(3.0)

    @pytest.mark.parametrize("message_chunk_size", [1, None])
    def test_empty_int32_edges_and_zero_node_graphs_are_supported(
        self,
        message_chunk_size: int | None,
    ) -> None:
        layer = RelationalGCNLayer(
            3,
            2,
            num_relations=4,
            message_chunk_size=message_chunk_size,
        )
        edge_index = torch.empty((2, 0), dtype=torch.int32)
        edge_types = torch.empty((0,), dtype=torch.int32)

        output = layer(torch.randn(5, 3), edge_index, edge_types)
        empty_output = layer(torch.empty(0, 3), edge_index, edge_types)

        assert output.shape == (5, 2)
        assert torch.isfinite(output).all()
        assert empty_output.shape == (0, 2)
        assert torch.isfinite(empty_output).all()

    @pytest.mark.parametrize("num_nodes", [0, 3])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("root_weight", [False, True])
    @pytest.mark.parametrize("num_bases", [None, 2])
    def test_empty_edges_keep_every_trainable_parameter_in_autograd(
        self,
        num_nodes: int,
        bias: bool,
        root_weight: bool,
        num_bases: int | None,
    ) -> None:
        layer = RelationalGCNLayer(
            3,
            2,
            num_relations=3,
            num_bases=num_bases,
            root_weight=root_weight,
            bias=bias,
            message_chunk_size=1,
        )
        x = torch.randn(num_nodes, 3, requires_grad=True)
        edge_index = torch.empty((2, 0), dtype=torch.int32)
        edge_types = torch.empty(0, dtype=torch.int16)

        output = layer(x, edge_index, edge_types)
        with torch.no_grad():
            expected = x.new_zeros((num_nodes, 2))
            if layer.root_linear is not None:
                expected = expected + layer.root_linear(x)
            if layer.bias is not None:
                expected = expected + layer.bias

        assert torch.equal(output.detach(), expected)
        output.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        if layer.root_linear is None:
            assert torch.count_nonzero(x.grad) == 0
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in layer.parameters()
            if parameter.requires_grad
        )
        relation_parameters = (
            (layer.relation_weight,)
            if layer.relation_weight is not None
            else (layer.basis_weight, layer.basis_coefficients)
        )
        assert all(
            parameter is not None
            and parameter.grad is not None
            and torch.count_nonzero(parameter.grad) == 0
            for parameter in relation_parameters
        )

    @pytest.mark.parametrize("num_nodes", [0, 4])
    @pytest.mark.parametrize("num_bases", [[None, 2], [2, None]])
    def test_empty_edges_propagate_zero_gradients_through_the_complete_stack(
        self,
        num_nodes: int,
        num_bases: list[int | None],
    ) -> None:
        model = RelationalGCN(
            3,
            2,
            num_relations=3,
            hidden_channels=[4],
            num_bases=num_bases,
            root_weight=[False, False],
            bias=[False, False],
            message_chunk_size=[1, None],
        )
        x = torch.randn(num_nodes, 3, requires_grad=True)

        output = model(
            x,
            torch.empty((2, 0), dtype=torch.int32),
            torch.empty(0, dtype=torch.int16),
        )

        assert torch.equal(output, torch.zeros(num_nodes, 2))
        output.sum().backward()
        assert x.grad is not None
        assert torch.count_nonzero(x.grad) == 0
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad) == 0
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    def test_stack_propagates_gradients_through_direct_and_basis_layers(self) -> None:
        model = RelationalGCN(
            3,
            2,
            num_relations=4,
            hidden_channels=[5, 4],
            num_bases=[2, None, 3],
            message_chunk_size=[1, 2, None],
            aggregation=["sum", "mean", "sum"],
            activation=["gelu", "relu"],
            normalization=["layernorm", "identity"],
            dropout=[0.0, 0.0, 0.0],
            residual=[False, True, False],
            root_weight=[True, False, True],
            bias=[True, False, True],
        )
        x = torch.randn(6, 3, requires_grad=True)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 5, 0, 4]],
            dtype=torch.int32,
        )
        edge_types = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int16)

        output = model(x, edge_index, edge_types)
        output.square().mean().backward()

        assert output.shape == (6, 2)
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    def test_per_layer_configuration_and_final_residual_are_encapsulated(self) -> None:
        model = RelationalGCN(
            2,
            2,
            num_relations=2,
            num_bases=[1],
            message_chunk_size=[1],
            aggregation=["sum"],
            dropout=[0.0],
            residual=[True],
            root_weight=[False],
            bias=[False],
        )
        layer = model.layers[0]
        assert isinstance(model.dropouts[0], nn.Identity)
        assert layer.num_bases == 1
        assert layer.message_chunk_size == 1
        assert layer.aggregation is Aggregation.SUM
        assert layer.root_linear is None
        assert layer.bias is None
        assert layer.basis_weight is not None
        assert layer.basis_coefficients is not None
        with torch.no_grad():
            layer.basis_weight.zero_()
            layer.basis_coefficients.zero_()

        x = torch.randn(3, 2)
        output = model(
            x,
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

        assert torch.equal(output, x)

    @pytest.mark.parametrize(
        ("edge_index", "edge_types", "error", "message"),
        [
            (
                torch.tensor([[0.0], [1.0]]),
                torch.tensor([0]),
                TypeError,
                "integer dtype",
            ),
            (
                torch.tensor([[0], [3]]),
                torch.tensor([0]),
                IndexError,
                "outside",
            ),
            (
                torch.tensor([[0], [1]]),
                torch.tensor([0.0]),
                TypeError,
                "integer dtype",
            ),
            (
                torch.tensor([[0], [1]]),
                torch.tensor([2]),
                IndexError,
                "relation outside",
            ),
            (
                torch.tensor([[0], [1]]),
                torch.tensor([0, 1]),
                ValueError,
                "shape",
            ),
        ],
    )
    def test_edge_contract_errors_are_explicit(
        self,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        error: type[Exception],
        message: str,
    ) -> None:
        layer = RelationalGCNLayer(2, 2, num_relations=2)
        with pytest.raises(error, match=message):
            layer(torch.randn(3, 2), edge_index, edge_types)

    def test_constructor_and_stack_validation_are_strict(self) -> None:
        with pytest.raises(TypeError, match="in_channels"):
            RelationalGCNLayer(True, 2, num_relations=2)
        with pytest.raises(ValueError, match="cannot exceed"):
            RelationalGCNLayer(2, 2, num_relations=2, num_bases=3)
        with pytest.raises(TypeError, match="message_chunk_size"):
            RelationalGCNLayer(2, 2, num_relations=2, message_chunk_size=True)
        with pytest.raises(ValueError, match="message_chunk_size"):
            RelationalGCNLayer(2, 2, num_relations=2, message_chunk_size=0)
        with pytest.raises(ValueError, match="sum.*mean"):
            RelationalGCNLayer(2, 2, num_relations=2, aggregation=Aggregation.MAX)
        with pytest.raises(TypeError, match="floating-point"):
            RelationalGCNLayer(2, 2, num_relations=2)(
                torch.ones(3, 2, dtype=torch.int64),
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        with pytest.raises(ValueError, match="exactly 2"):
            RelationalGCN(
                2,
                2,
                num_relations=2,
                hidden_channels=[3],
                num_bases=[1],
            )
        with pytest.raises(TypeError, match="dropout"):
            RelationalGCN(2, 2, num_relations=2, dropout=True)
        with pytest.raises(TypeError, match="residual"):
            RelationalGCN(2, 2, num_relations=2, residual=[1])
        with pytest.raises(ValueError, match="message_chunk_size must contain exactly 2"):
            RelationalGCN(
                2,
                2,
                num_relations=2,
                hidden_channels=[3],
                message_chunk_size=[1],
            )

    def test_model_is_constructible_from_yaml_object_specification(self) -> None:
        model = ObjectFactory.build(
            {
                "target": (
                    "lambdaforge.nn.models.graph.message_passing.RelationalGCN.RelationalGCN"
                ),
                "params": {
                    "in_channels": 3,
                    "out_channels": 2,
                    "num_relations": 3,
                    "hidden_channels": [4],
                    "num_bases": [2, None],
                    "message_chunk_size": [1, None],
                    "aggregation": ["mean", "sum"],
                    "activation": "gelu",
                    "normalization": "layernorm",
                    "dropout": [0.0, 0.0],
                    "residual": [False, False],
                    "root_weight": [True, True],
                    "bias": [True, False],
                },
            }
        )

        assert isinstance(model, RelationalGCN)
        output = model(
            torch.randn(4, 3),
            torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )
        assert output.shape == (4, 2)
