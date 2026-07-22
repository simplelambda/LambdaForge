"""Contract tests for shared sparse graph data and reduction objects."""

from __future__ import annotations

import pytest
import torch

from lambdaforge.nn.models import Aggregation, Scatter
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill


class TestGraphPrimitives:
    """Keep sparse reductions and aligned edge data deterministic and finite."""

    def test_extended_reductions_define_empty_segments_as_zero(self) -> None:
        source = torch.tensor([[1.0, 5.0], [3.0, 1.0], [7.0, 9.0]])
        index = torch.tensor([0, 0, 2])

        assert torch.equal(
            Scatter.maximum(source, index, 4),
            torch.tensor([[3.0, 5.0], [0.0, 0.0], [7.0, 9.0], [0.0, 0.0]]),
        )
        assert torch.equal(
            Scatter.minimum(source, index, 4),
            torch.tensor([[1.0, 1.0], [0.0, 0.0], [7.0, 9.0], [0.0, 0.0]]),
        )
        assert torch.allclose(
            Scatter.standard_deviation(source, index, 4),
            torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        )
        assert torch.equal(
            Scatter.reduce(source, index, 4, Aggregation.MAX),
            Scatter.maximum(source, index, 4),
        )

    def test_mean_preserves_scalar_shape_and_std_has_finite_constant_gradients(self) -> None:
        scalar = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
        index = torch.tensor([0, 0, 1])
        mean = Scatter.mean(scalar, index, 3)
        assert mean.shape == (3,)
        assert torch.equal(mean, torch.tensor([2.0, 5.0, 0.0]))

        constant = torch.tensor([[1.0], [2.0]], requires_grad=True)
        standard_deviation = Scatter.standard_deviation(
            constant,
            torch.tensor([0, 1]),
            2,
        )
        assert torch.equal(standard_deviation, torch.zeros_like(standard_deviation))
        standard_deviation.sum().backward()
        assert constant.grad is not None
        assert torch.isfinite(constant.grad).all()

    def test_edge_features_and_relations_are_normalized_without_truncation(self) -> None:
        reference = torch.zeros(3, 2, dtype=torch.float64)
        features = GraphEdgeData.normalize_features(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            edge_channels=2,
            edge_count=2,
            reference=reference,
        )
        relations = GraphEdgeData.normalize_relation_types(
            torch.tensor([0, 2], dtype=torch.int16),
            edge_count=2,
            num_relations=3,
            device=reference.device,
        )

        assert features is not None
        assert features.dtype == torch.float64
        assert relations.dtype == torch.long
        with pytest.raises(TypeError, match="integer dtype"):
            GraphEdgeData.normalize_relation_types(
                torch.tensor([False, True]),
                edge_count=2,
                num_relations=3,
                device=reference.device,
            )

    @pytest.mark.parametrize(
        ("fill", "expected"),
        [
            (
                GraphSelfLoopFill.ZERO,
                torch.tensor([[10.0], [20.0], [0.0], [0.0], [0.0]]),
            ),
            (
                GraphSelfLoopFill.MEAN,
                torch.tensor([[10.0], [20.0], [0.0], [10.0], [20.0]]),
            ),
            (
                2.5,
                torch.tensor([[10.0], [20.0], [2.5], [2.5], [2.5]]),
            ),
        ],
    )
    def test_self_loops_are_replaced_and_features_stay_aligned(
        self,
        fill: GraphSelfLoopFill | float,
        expected: torch.Tensor,
    ) -> None:
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 0, 2, 2]])
        edge_features = torch.tensor([[10.0], [99.0], [20.0], [88.0]])

        routed, aligned = GraphEdgeData.replace_self_loops(
            edge_index,
            device=edge_index.device,
            num_nodes=3,
            edge_features=edge_features,
            fill=fill,
        )

        assert torch.equal(routed, torch.tensor([[0, 1, 0, 1, 2], [1, 2, 0, 1, 2]]))
        assert aligned is not None
        assert torch.equal(aligned, expected)
