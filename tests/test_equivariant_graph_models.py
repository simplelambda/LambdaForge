"""Numerical and integration tests for dependency-light E(n)-equivariant graphs."""

from __future__ import annotations

import pytest
import torch

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.graph.equivariant import (
    EGNN,
    EGNNLayer,
    EquivariantOutputMode,
)
from lambdaforge.nn.models.graph.GraphReadout import GraphReadout
from lambdaforge.nn.pooling.sparse.SparseMeanPooling import SparseMeanPooling


class TestEquivariantGraphModels:
    """Verify shape, gradients and the promised E(n) transformation contract."""

    @staticmethod
    def graph(
        *,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scalar node/edge features and coordinates for one directed graph."""
        generator = torch.Generator().manual_seed(17)
        x = torch.randn(4, 3, generator=generator, dtype=dtype)
        coordinates = torch.randn(4, 3, generator=generator, dtype=dtype)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]],
            dtype=torch.long,
        )
        edge_features = torch.randn(6, 2, generator=generator, dtype=dtype)
        return x, edge_index, coordinates, edge_features

    def test_layer_and_stack_have_finite_gradients_without_lazy_parameters(self) -> None:
        x, edge_index, coordinates, edge_features = self.graph()
        x.requires_grad_()
        coordinates.requires_grad_()
        edge_features.requires_grad_()
        model = EGNN(
            3,
            2,
            hidden_channels=[5],
            edge_channels=2,
            message_channels=[7, 6],
            layer_kwargs=[
                {"attention": True, "coordinate_tanh": True},
                {"normalize_displacements": True},
            ],
            normalization="layernorm",
        )
        parameter_count = model.num_parameters()
        state_keys = tuple(model.state_dict())

        features, updated_coordinates = model.forward_with_coordinates(
            x,
            edge_index,
            coordinates,
            edge_features,
        )
        loss = features.square().mean() + updated_coordinates.square().mean()
        loss.backward()

        assert features.shape == (4, 2)
        assert updated_coordinates.shape == (4, 3)
        assert torch.isfinite(features).all()
        assert torch.isfinite(updated_coordinates).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()
        assert edge_features.grad is not None and torch.isfinite(edge_features.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        assert model.num_parameters() == parameter_count
        assert tuple(model.state_dict()) == state_keys

    @pytest.mark.parametrize(
        "orthogonal",
        [
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
        ],
    )
    def test_features_are_invariant_and_coordinates_are_en_equivariant(
        self,
        orthogonal: torch.Tensor,
    ) -> None:
        x, edge_index, coordinates, edge_features = self.graph(dtype=torch.float64)
        translation = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float64)
        transformed = coordinates @ orthogonal.T + translation
        model = EGNN(
            3,
            4,
            hidden_channels=[6],
            edge_channels=2,
            message_channels=8,
            layer_kwargs={
                "coordinate_tanh": True,
                "coordinate_scale": 0.2,
            },
        ).to(dtype=torch.float64)
        model.eval()

        features, output_coordinates = model.forward_with_coordinates(
            x,
            edge_index,
            coordinates,
            edge_features,
        )
        transformed_features, transformed_coordinates = model.forward_with_coordinates(
            x,
            edge_index,
            transformed,
            edge_features,
        )

        assert torch.allclose(transformed_features, features, atol=1e-10, rtol=1e-9)
        assert torch.allclose(
            transformed_coordinates,
            output_coordinates @ orthogonal.T + translation,
            atol=1e-10,
            rtol=1e-9,
        )

    def test_node_permutation_and_edge_order_preserve_the_computation(self) -> None:
        x, edge_index, coordinates, edge_features = self.graph(dtype=torch.float64)
        model = EGNN(3, 4, edge_channels=2, message_channels=6).to(dtype=torch.float64)
        model.eval()
        expected_features, expected_coordinates = model.forward_with_coordinates(
            x,
            edge_index,
            coordinates,
            edge_features,
        )
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        edge_order = torch.tensor([4, 1, 5, 0, 3, 2])

        actual_features, actual_coordinates = model.forward_with_coordinates(
            x[permutation],
            inverse[edge_index[:, edge_order]],
            coordinates[permutation],
            edge_features[edge_order],
        )

        assert torch.allclose(actual_features, expected_features[permutation], atol=1e-10)
        assert torch.allclose(actual_coordinates, expected_coordinates[permutation], atol=1e-10)

    def test_empty_edges_are_finite_and_do_not_move_coordinates(self) -> None:
        x = torch.randn(3, 2)
        coordinates = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        edge_index = torch.empty((2, 0), dtype=torch.long)
        layer = EGNNLayer(
            2,
            3,
            edge_channels=1,
            normalize_displacements=True,
            coordinate_tanh=True,
        )

        features, updated = layer(
            x,
            edge_index,
            coordinates,
            torch.empty((0, 1)),
        )

        assert features.shape == (3, 3)
        assert torch.isfinite(features).all()
        assert torch.equal(updated, coordinates)

    @pytest.mark.parametrize("num_nodes", [0, 3])
    def test_empty_edges_keep_every_enabled_parameter_connected(
        self,
        num_nodes: int,
    ) -> None:
        x = torch.randn(num_nodes, 3, requires_grad=True)
        coordinates = torch.randn(num_nodes, 4, requires_grad=True)
        edge_features = torch.empty((0, 2), requires_grad=True)
        layer = EGNNLayer(
            3,
            4,
            edge_channels=2,
            message_channels=5,
            attention=True,
        )

        features, updated = layer(
            x,
            torch.empty((2, 0), dtype=torch.int32),
            coordinates,
            edge_features,
        )
        (features.square().sum() + updated.square().sum()).backward()

        assert features.shape == (num_nodes, 4)
        assert torch.equal(updated, coordinates)
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()
        assert edge_features.grad is not None and torch.isfinite(edge_features.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in layer.parameters()
            if parameter.requires_grad
        )

    def test_max_message_aggregation_matches_manual_reference_with_isolates(self) -> None:
        layer = EGNNLayer(
            1,
            1,
            message_channels=1,
            message_hidden_channels=[],
            node_hidden_channels=[],
            message_aggregation=Aggregation.MAX,
            residual=False,
            update_coordinates=False,
            bias=False,
        )
        with torch.no_grad():
            layer.message_mlp.output.weight.copy_(torch.tensor([[0.0, 1.0, 0.0]]))
            layer.node_mlp.output.weight.copy_(torch.tensor([[0.0, 1.0]]))
        x = torch.tensor([[-2.0], [3.0], [1.0], [10.0]], requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [2, 2, 1]])

        output, _ = layer(x, edge_index, torch.zeros(4, 2))
        output.sum().backward()

        assert torch.equal(output, torch.tensor([[0.0], [1.0], [3.0], [0.0]]))
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in layer.parameters()
            if parameter.requires_grad
        )

    def test_coordinate_updates_can_be_disabled_exactly(self) -> None:
        x, edge_index, coordinates, edge_features = self.graph()
        x.requires_grad_()
        coordinates.requires_grad_()
        edge_features.requires_grad_()
        layer = EGNNLayer(3, 3, edge_channels=2, update_coordinates=False)
        features, updated = layer(x, edge_index, coordinates, edge_features)
        features.square().mean().backward()

        assert layer.coordinate_mlp is None
        assert updated is coordinates
        assert all(
            parameter.grad is not None
            for parameter in layer.parameters()
            if parameter.requires_grad
        )

    def test_disabled_residual_has_no_disconnected_projection(self) -> None:
        x, edge_index, coordinates, edge_features = self.graph()
        x.requires_grad_()
        coordinates.requires_grad_()
        edge_features.requires_grad_()
        layer = EGNNLayer(3, 4, edge_channels=2, residual=False)

        features, updated = layer(x, edge_index, coordinates, edge_features)
        (features.square().mean() + updated.square().mean()).backward()

        assert layer.residual_projection is None
        assert all(
            parameter.grad is not None
            for parameter in layer.parameters()
            if parameter.requires_grad
        )

    def test_feature_dropout_keeps_the_residual_identity_clean(self) -> None:
        layer = EGNNLayer(
            2,
            2,
            feature_dropout=0.9,
            update_coordinates=False,
            residual=True,
        )
        with torch.no_grad():
            for parameter in layer.node_mlp.parameters():
                parameter.zero_()
        x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        coordinates = torch.zeros(2, 3)

        output, _ = layer(
            x,
            torch.empty((2, 0), dtype=torch.long),
            coordinates,
        )

        assert torch.equal(output, x)

    def test_mapping_mode_and_graph_readout_remain_composable(self) -> None:
        x, edge_index, coordinates, edge_features = self.graph()
        mapping_model = EGNN(
            3,
            4,
            edge_channels=2,
            output_mode=EquivariantOutputMode.MAPPING,
            feature_output_key="embedding",
            coordinate_output_key="geometry",
        )
        output = mapping_model(x, edge_index, coordinates, edge_features)
        assert isinstance(output, dict)
        assert set(output) == {"embedding", "geometry"}
        assert mapping_model.output_schema == {
            "embedding": "Tensor[N, F]",
            "geometry": "Tensor[N, D]",
        }
        assert EGNN(3, 4).output_schema == {"output": "Tensor[N, F]"}

        readout = GraphReadout(
            encoder=EGNN(3, 4, edge_channels=2),
            pooling=SparseMeanPooling(),
            head=torch.nn.Linear(4, 2),
        )
        groups = torch.tensor([0, 0, 1, 1])
        pooled = readout(
            x,
            edge_index,
            groups,
            coordinates=coordinates,
            edge_features=edge_features,
        )
        assert pooled.shape == (2, 2)

    @pytest.mark.parametrize(
        ("coordinates", "message"),
        [
            (torch.randn(3, 2), "shape"),
            (torch.ones(4, 2, dtype=torch.int64), "floating-point"),
            (torch.ones(4, 2, dtype=torch.float64), "device and dtype"),
        ],
    )
    def test_invalid_coordinate_contracts_are_rejected(
        self,
        coordinates: torch.Tensor,
        message: str,
    ) -> None:
        layer = EGNNLayer(3, 3)
        with pytest.raises((TypeError, ValueError), match=message):
            layer(
                torch.randn(4, 3),
                torch.empty((2, 0), dtype=torch.long),
                coordinates,
            )

    def test_layer_rejects_coordinate_reductions_that_break_equivariance(self) -> None:
        with pytest.raises(ValueError, match="sum.*mean"):
            EGNNLayer(3, 3, coordinate_aggregation=Aggregation.MAX)

    @pytest.mark.parametrize(
        ("name", "value", "error", "message"),
        [
            ("residual", "false", TypeError, "boolean"),
            ("update_coordinates", 0, TypeError, "boolean"),
            ("normalize_displacements", 1, TypeError, "boolean"),
            ("coordinate_tanh", "false", TypeError, "boolean"),
            ("attention", 1, TypeError, "boolean"),
            ("bias", 1, TypeError, "boolean"),
            ("feature_dropout", False, TypeError, "real number"),
            ("message_dropout", "0.1", TypeError, "real number"),
            ("update_dropout", float("nan"), ValueError, "finite"),
            ("distance_epsilon", True, TypeError, "real number"),
            ("distance_scale", "1.0", TypeError, "real number"),
            ("coordinate_scale", float("inf"), ValueError, "finite"),
        ],
    )
    def test_layer_rejects_coerced_flags_and_invalid_real_values(
        self,
        name: str,
        value: object,
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            EGNNLayer(3, 3, **{name: value})

    def test_stack_passes_feature_dropout_into_layers_and_reserves_ownership(self) -> None:
        model = EGNN(
            3,
            2,
            hidden_channels=[4],
            feature_dropout=[0.2, 0.3],
        )

        assert [layer.feature_dropout.p for layer in model.layers] == [0.2, 0.3]
        assert not hasattr(model, "feature_dropouts")
        with pytest.raises(ValueError, match="feature_dropout"):
            EGNN(
                3,
                2,
                feature_dropout=0.2,
                layer_kwargs={"feature_dropout": 0.1},
            )

    def test_zero_node_stack_preserves_feature_and_coordinate_shapes(self) -> None:
        model = EGNN(
            3,
            2,
            hidden_channels=[4],
            edge_channels=1,
            feature_dropout=[0.1, 0.0],
        )

        features, coordinates = model.forward_with_coordinates(
            torch.empty((0, 3)),
            torch.empty((2, 0), dtype=torch.int32),
            torch.empty((0, 5)),
            torch.empty((0, 1)),
        )

        assert features.shape == (0, 2)
        assert coordinates.shape == (0, 5)

    def test_yaml_object_factory_builds_equivariant_stack(self) -> None:
        model = ObjectFactory.build(
            {
                "target": "lambdaforge.nn.models.graph.equivariant.EGNN.EGNN",
                "params": {
                    "in_channels": 3,
                    "out_channels": 2,
                    "hidden_channels": [5],
                    "edge_channels": 2,
                    "message_channels": [7, 6],
                    "feature_dropout": [0.1, 0.0],
                    "normalization": "layernorm",
                    "layer_kwargs": [
                        {"attention": True, "coordinate_tanh": True},
                        {"normalize_displacements": True},
                    ],
                },
            }
        )
        x, edge_index, coordinates, edge_features = self.graph()

        output = model(x, edge_index, coordinates, edge_features)

        assert isinstance(model, EGNN)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 2)
