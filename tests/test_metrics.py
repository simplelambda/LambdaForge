"""Metric correctness and state contracts."""

import math

import torch

from lambdaforge.metrics import (
    MAE,
    BinaryAccuracy,
    MulticlassAccuracy,
    SpearmanCorrelation,
)


class TestMetrics:
    """Exercise binary, multiclass and regression metric families."""

    def test_binary_accuracy_accumulates_batches(self) -> None:
        metric = BinaryAccuracy()
        metric.update({"probs": torch.tensor([0.9, 0.2])}, {"y": torch.tensor([1, 0])})
        metric.update({"probs": torch.tensor([0.8])}, {"y": torch.tensor([0])})
        assert metric.compute() == 2 / 3

    def test_regression_mae(self) -> None:
        metric = MAE()
        metric.update({"pred": torch.tensor([1.0, 4.0])}, {"y": torch.tensor([2.0, 2.0])})
        assert metric.compute() == 1.5

    def test_spearman_uses_average_tie_ranks(self) -> None:
        metric = SpearmanCorrelation()
        metric.update({"pred": torch.tensor([1.0, 1.0, 2.0])}, {"y": torch.tensor([1.0, 1.0, 3.0])})
        assert math.isclose(metric.compute(), 1.0, rel_tol=1e-6)

    def test_multiclass_accuracy_has_usable_default_constructor(self) -> None:
        metric = MulticlassAccuracy()
        metric.update(
            {"logits": torch.tensor([[4.0, 0.0], [0.0, 3.0]])},
            {"y": torch.tensor([0, 1])},
        )
        assert metric.compute() == 1.0

    def test_multiclass_merge_ignores_empty_distributed_rank(self) -> None:
        metric = MulticlassAccuracy()
        metric.merge_distributed_state(
            {"predictions": torch.empty((0, 0)), "targets": torch.empty(0, dtype=torch.long)}
        )
        metric.merge_distributed_state(
            {
                "predictions": torch.tensor([[3.0, 0.0]]),
                "targets": torch.tensor([0]),
            }
        )
        assert metric.compute() == 1.0
