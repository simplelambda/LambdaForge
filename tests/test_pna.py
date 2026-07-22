"""Numerical and configuration tests for native PNA graph layers."""

from __future__ import annotations

import math

import pytest
import torch

from lambdaforge.experiments import ObjectFactory
from lambdaforge.nn.models.graph.message_passing.DegreeScaler import DegreeScaler
from lambdaforge.nn.models.graph.message_passing.PNA import PNA
from lambdaforge.nn.models.graph.message_passing.PNAAggregator import PNAAggregator
from lambdaforge.nn.models.graph.message_passing.PNALayer import PNALayer


class TestPNA:
    """Verify reducers, degree scaling, edge data and stack construction."""

    def test_aggregators_match_manual_segment_references(self) -> None:
        messages = torch.tensor([[1.0, 5.0], [3.0, 1.0], [4.0, 8.0], [8.0, 2.0]])
        destination = torch.tensor([0, 0, 2, 2])
        expected = {
            PNAAggregator.MEAN: torch.tensor([[2.0, 3.0], [0.0, 0.0], [6.0, 5.0], [0.0, 0.0]]),
            PNAAggregator.MIN: torch.tensor([[1.0, 1.0], [0.0, 0.0], [4.0, 2.0], [0.0, 0.0]]),
            PNAAggregator.MAX: torch.tensor([[3.0, 5.0], [0.0, 0.0], [8.0, 8.0], [0.0, 0.0]]),
            PNAAggregator.STD: torch.tensor([[1.0, 2.0], [0.0, 0.0], [2.0, 3.0], [0.0, 0.0]]),
        }

        for aggregator, reference in expected.items():
            assert torch.allclose(aggregator.reduce(messages, destination, 4), reference)

    def test_degree_scalers_match_manual_factors(self) -> None:
        degree = torch.tensor([0.0, 1.0, 3.0])
        average_degree = 2.0
        average_log_degree = math.log(3.0)
        epsilon = 0.25
        expected = {
            DegreeScaler.IDENTITY: torch.ones(3),
            DegreeScaler.AMPLIFICATION: degree.log1p() / average_log_degree,
            DegreeScaler.ATTENUATION: average_log_degree / degree.log1p().clamp_min(epsilon),
            DegreeScaler.LINEAR: degree / average_degree,
            DegreeScaler.INVERSE_LINEAR: average_degree / degree.clamp_min(epsilon),
        }

        for scaler, reference in expected.items():
            actual = scaler.factor(
                degree,
                average_degree=average_degree,
                average_log_degree=average_log_degree,
                epsilon=epsilon,
            )
            assert torch.allclose(actual, reference)

    def test_layer_message_and_post_mlp_follow_directed_reference(self) -> None:
        layer = PNALayer(
            1,
            1,
            aggregators="mean",
            scalers="identity",
            message_channels=1,
            pre_mlp_hidden_channels=[],
            post_mlp_hidden_channels=[],
            bias=False,
        )
        with torch.no_grad():
            layer.pre_mlp.output.weight.copy_(torch.tensor([[0.0, 1.0]]))
            layer.post_mlp.output.weight.copy_(torch.tensor([[0.0, 1.0]]))
        x = torch.tensor([[1.0], [3.0], [10.0]])
        edge_index = torch.tensor([[0, 1], [2, 2]])

        output = layer(x, edge_index)

        assert torch.allclose(output, torch.tensor([[0.0], [0.0], [2.0]]))

    def test_edge_features_and_parameters_receive_gradients(self) -> None:
        layer = PNALayer(
            3,
            4,
            aggregators=["mean", "max", "std"],
            scalers=["identity", "linear"],
            edge_channels=2,
            message_channels=5,
            pre_mlp_hidden_channels=[7],
            post_mlp_hidden_channels=[6],
            average_degree=1.5,
            average_log_degree=0.8,
            activation="gelu",
            dropout=0.1,
        )
        x = torch.randn(5, 3, requires_grad=True)
        edge_features = torch.randn(4, 2, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]])

        output = layer(x, edge_index, edge_features)
        output.square().sum().backward()

        assert output.shape == (5, 4)
        assert torch.isfinite(output).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert edge_features.grad is not None and torch.isfinite(edge_features.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in layer.parameters()
            if parameter.requires_grad
        )

    def test_layer_creates_no_parameters_during_forward(self) -> None:
        layer = PNALayer(3, 2, message_channels=4)
        before = tuple((name, id(parameter)) for name, parameter in layer.named_parameters())

        layer(torch.randn(3, 3), torch.tensor([[0, 1], [1, 2]]))

        after = tuple((name, id(parameter)) for name, parameter in layer.named_parameters())
        assert after == before

    @pytest.mark.parametrize("edge_channels", [0, 2])
    def test_empty_edges_and_isolated_nodes_are_finite(self, edge_channels: int) -> None:
        layer = PNALayer(
            3,
            2,
            aggregators=["mean", "min", "max", "std"],
            scalers=[
                "identity",
                "amplification",
                "attenuation",
                "linear",
                "inverse_linear",
            ],
            edge_channels=edge_channels,
            epsilon=1e-6,
        )
        edge_features = torch.empty((0, edge_channels)) if edge_channels else None

        output = layer(
            torch.randn(4, 3),
            torch.empty((2, 0), dtype=torch.long),
            edge_features,
        )

        assert output.shape == (4, 2)
        assert torch.isfinite(output).all()

    def test_inverse_scalers_keep_half_precision_isolated_nodes_finite(self) -> None:
        layer = PNALayer(
            2,
            2,
            aggregators=["mean"],
            scalers=["attenuation", "inverse_linear"],
        ).half()
        output = layer(
            torch.randn(3, 2, dtype=torch.float16),
            torch.empty((2, 0), dtype=torch.long),
        )
        assert torch.isfinite(output).all()

    def test_default_internal_mlps_make_activation_architecturally_effective(self) -> None:
        torch.manual_seed(123)
        relu_model = PNA(3, 2, activation="relu")
        torch.manual_seed(123)
        silu_model = PNA(3, 2, activation="silu")
        x = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, 2.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]])

        assert tuple(relu_model.state_dict()) == tuple(silu_model.state_dict())
        assert not torch.allclose(
            relu_model(x, edge_index),
            silu_model(x, edge_index),
        )

    def test_pna_stack_supports_per_layer_configuration_and_residuals(self) -> None:
        model = PNA(
            3,
            2,
            hidden_channels=[3, 5],
            aggregators=["mean", "max"],
            scalers=["identity", "amplification"],
            edge_channels=1,
            message_channels=[4, 5, 6],
            average_degree=[1.0, 2.0, 3.0],
            average_log_degree=[0.5, 0.75, 1.0],
            epsilon=[1e-6, 1e-5, 1e-4],
            dropout=[0.0, 0.1, 0.0],
            activation=["relu", "gelu", "silu"],
            activation_kwargs=[{}, {}, {}],
            normalization=["layernorm", "identity"],
            residual=[True, False],
            bias=[False, True, True],
        )
        x = torch.randn(4, 3, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 1]])
        edge_features = torch.randn(3, 1)

        output = model(x, edge_index, edge_features)
        output.sum().backward()

        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 2)
        assert torch.isfinite(output).all()
        assert x.grad is not None

    def test_pna_is_constructible_from_yaml_compatible_specification(self) -> None:
        model = ObjectFactory.build(
            {
                "target": "lambdaforge.nn.models.graph.message_passing.PNA.PNA",
                "params": {
                    "in_channels": 4,
                    "out_channels": 2,
                    "hidden_channels": [6],
                    "aggregators": ["mean", "max", "std"],
                    "scalers": ["identity", "attenuation"],
                    "average_degree": [2.0, 2.0],
                    "average_log_degree": [1.0, 1.0],
                    "normalization": "layernorm",
                },
            }
        )

        assert isinstance(model, PNA)
        assert model(torch.randn(3, 4), torch.tensor([[0, 1], [1, 2]])).shape == (3, 2)

    @pytest.mark.parametrize(
        ("keyword", "value", "match"),
        [
            ("aggregators", [], "at least one"),
            ("aggregators", ["mean", "mean"], "duplicate"),
            ("scalers", [], "at least one"),
            ("scalers", ["identity", "identity"], "duplicate"),
            ("average_degree", 0.0, "positive and finite"),
            ("average_log_degree", float("inf"), "positive and finite"),
            ("epsilon", 0.0, "positive and finite"),
            ("dropout", 1.0, r"\[0, 1\)"),
        ],
    )
    def test_invalid_layer_configuration_fails_eagerly(
        self,
        keyword: str,
        value: object,
        match: str,
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=match):
            PNALayer(3, 2, **{keyword: value})

    def test_edge_feature_contract_is_strict(self) -> None:
        layer = PNALayer(3, 2, edge_channels=2)
        x = torch.randn(3, 3)
        edge_index = torch.tensor([[0, 1], [1, 2]])

        with pytest.raises(ValueError, match="required"):
            layer(x, edge_index)
        with pytest.raises(ValueError, match="shape"):
            layer(x, edge_index, torch.randn(3, 2))

    def test_layer_kwargs_allow_exact_per_layer_architectures(self) -> None:
        model = PNA(
            3,
            2,
            hidden_channels=[4],
            layer_kwargs=[
                {
                    "aggregators": ["mean"],
                    "scalers": ["identity"],
                    "message_channels": 5,
                    "pre_mlp_hidden_channels": [6],
                },
                {
                    "aggregators": ["max", "std"],
                    "scalers": ["amplification", "attenuation"],
                    "message_channels": 7,
                    "post_mlp_hidden_channels": [8, 6],
                },
            ],
        )

        first, second = model.layers
        assert isinstance(first, PNALayer)
        assert isinstance(second, PNALayer)
        assert first.message_channels == 5
        assert first.aggregators == (PNAAggregator.MEAN,)
        assert second.message_channels == 7
        assert second.aggregators == (PNAAggregator.MAX, PNAAggregator.STD)
        assert second.scalers == (DegreeScaler.AMPLIFICATION, DegreeScaler.ATTENUATION)
        with pytest.raises(ValueError, match="stack-owned"):
            PNA(3, 2, layer_kwargs={"edge_channels": 4})

    def test_stack_rejects_wrong_per_layer_lengths(self) -> None:
        with pytest.raises(ValueError, match="average_degree must contain exactly 2"):
            PNA(3, 2, hidden_channels=[4], average_degree=[1.0])
