"""Correctness, memory and distributed contracts for streaming curve metrics."""

import math
from unittest.mock import Mock

import pytest
import torch

from lambdaforge.metrics import (
    BinaryAUPRC,
    BinaryAUROC,
    MetricAlias,
    StreamingBinaryAUPRC,
    StreamingBinaryAUROC,
)


class TestStreamingBinaryMetrics:
    """Exercise fixed-memory binary AUROC and average precision."""

    @pytest.mark.parametrize("metric_type", [StreamingBinaryAUROC, StreamingBinaryAUPRC])
    def test_perfect_ranking_is_one(self, metric_type: type) -> None:
        metric = metric_type(num_bins=16)
        metric.update(
            {"probs": torch.tensor([0.1, 0.2, 0.8, 0.9])},
            {"y": torch.tensor([0, 0, 1, 1])},
        )
        assert metric.compute() == 1.0

    def test_reversed_ranking_has_known_values(self) -> None:
        outputs = {"probs": torch.tensor([0.9, 0.8, 0.2, 0.1])}
        batch = {"y": torch.tensor([0, 0, 1, 1])}
        auroc = StreamingBinaryAUROC(num_bins=16)
        auprc = StreamingBinaryAUPRC(num_bins=16)
        auroc.update(outputs, batch)
        auprc.update(outputs, batch)
        assert auroc.compute() == 0.0
        assert math.isclose(auprc.compute(), 5 / 12)

    @pytest.mark.parametrize("metric_type", [StreamingBinaryAUROC, StreamingBinaryAUPRC])
    def test_scores_in_one_bin_are_treated_as_ties(self, metric_type: type) -> None:
        metric = metric_type(num_bins=2)
        metric.update(
            {"probs": torch.tensor([0.1, 0.4])},
            {"y": torch.tensor([0, 1])},
        )
        assert metric.compute() == 0.5

    @pytest.mark.parametrize("metric_type", [StreamingBinaryAUROC, StreamingBinaryAUPRC])
    def test_multiple_batches_equal_one_batch(self, metric_type: type) -> None:
        predictions = torch.tensor([0.05, 0.4, 0.6, 0.95])
        targets = torch.tensor([0, 1, 0, 1])
        combined = metric_type(num_bins=32)
        split = metric_type(num_bins=32)
        combined.update({"probs": predictions}, {"y": targets})
        split.update({"probs": predictions[:2]}, {"y": targets[:2]})
        split.update({"probs": predictions[2:]}, {"y": targets[2:]})
        assert split.compute() == combined.compute()
        for key in ("positive_counts", "negative_counts"):
            assert torch.equal(
                split.distributed_state()[key],
                combined.distributed_state()[key],
            )

    @pytest.mark.parametrize("metric_type", [StreamingBinaryAUROC, StreamingBinaryAUPRC])
    def test_explicit_logits_match_probabilities(self, metric_type: type) -> None:
        logits = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        targets = torch.tensor([0, 1, 0, 1])
        from_logits = metric_type(num_bins=64, pred_key="logits", from_logits=True)
        from_probabilities = metric_type(num_bins=64)
        from_logits.update({"logits": logits}, {"y": targets})
        from_probabilities.update({"probs": torch.sigmoid(logits)}, {"y": targets})
        assert from_logits.compute() == from_probabilities.compute()

    @pytest.mark.parametrize(
        ("streaming_type", "exact_type"),
        [(StreamingBinaryAUROC, BinaryAUROC), (StreamingBinaryAUPRC, BinaryAUPRC)],
    )
    def test_matches_exact_metric_without_cross_class_bin_collisions(
        self, streaming_type: type, exact_type: type
    ) -> None:
        predictions = torch.tensor([0.05, 0.15, 0.35, 0.55, 0.75, 0.95])
        targets = torch.tensor([0, 1, 0, 1, 1, 0])
        streaming = streaming_type(num_bins=100)
        exact = exact_type()
        streaming.update({"probs": predictions}, {"y": targets})
        exact.update({"probs": predictions}, {"y": targets})
        assert math.isclose(streaming.compute(), exact.compute(), abs_tol=1e-7)

    @pytest.mark.parametrize("metric_type", [StreamingBinaryAUROC, StreamingBinaryAUPRC])
    def test_empty_and_single_class_states_are_undefined(self, metric_type: type) -> None:
        metric = metric_type(num_bins=8)
        assert math.isnan(metric.compute())
        metric.update({"probs": torch.tensor([0.2, 0.8])}, {"y": torch.tensor([1, 1])})
        assert math.isnan(metric.compute())

    def test_invalid_inputs_fail_before_changing_state(self) -> None:
        invalid_batches = [
            ({"probs": torch.tensor([float("nan")])}, {"y": torch.tensor([1])}),
            ({"probs": torch.tensor([1.1])}, {"y": torch.tensor([1])}),
            ({"probs": torch.tensor([0.5])}, {"y": torch.tensor([0.5])}),
            ({"probs": torch.tensor([0.2, 0.3])}, {"y": torch.tensor([1])}),
        ]
        metric = StreamingBinaryAUROC(num_bins=8)
        for outputs, batch in invalid_batches:
            with pytest.raises(ValueError):
                metric.update(outputs, batch)
        state = metric.distributed_state()
        assert state["positive_counts"].sum().item() == 0
        assert state["negative_counts"].sum().item() == 0

    def test_persistent_state_size_is_independent_of_sample_count(self) -> None:
        metric = StreamingBinaryAUROC(num_bins=32)
        for _ in range(20):
            metric.update(
                {"probs": torch.linspace(0, 1, 101)},
                {"y": torch.arange(101) % 2},
            )
        state = metric.distributed_state()
        counts = (state["positive_counts"], state["negative_counts"])
        assert all(value.shape == (32,) for value in counts)
        assert sum(value.numel() * value.element_size() for value in counts) == 16 * 32
        assert sum(value.sum().item() for value in counts) == 20 * 101

    def test_state_round_trip_is_independent_and_validated(self) -> None:
        metric = StreamingBinaryAUPRC(num_bins=8)
        metric.update(
            {"probs": torch.tensor([0.1, 0.4, 0.8])},
            {"y": torch.tensor([0, 1, 1])},
        )
        state = metric.state_dict()
        restored = StreamingBinaryAUPRC(num_bins=8)
        restored.load_state_dict(state)
        state["positive_counts"].zero_()
        assert restored.compute() == metric.compute()
        with pytest.raises(ValueError):
            StreamingBinaryAUPRC(num_bins=4).load_state_dict(restored.state_dict())

    def test_invalid_merge_does_not_partially_change_state(self) -> None:
        metric = StreamingBinaryAUROC(num_bins=4)
        invalid_state = {
            "positive_counts": torch.tensor([1, 0, 0, 0], dtype=torch.int64),
            "negative_counts": torch.zeros(3, dtype=torch.int64),
        }
        with pytest.raises(ValueError):
            metric.merge_distributed_state(invalid_state)
        assert metric.distributed_state()["positive_counts"].sum().item() == 0

    @pytest.mark.parametrize("metric_type", [StreamingBinaryAUROC, StreamingBinaryAUPRC])
    def test_merged_rank_state_matches_combined_data(self, metric_type: type) -> None:
        first = metric_type(num_bins=16)
        second = metric_type(num_bins=16)
        merged = metric_type(num_bins=16)
        combined = metric_type(num_bins=16)
        first.update({"probs": torch.tensor([0.1, 0.7])}, {"y": torch.tensor([0, 1])})
        second.update({"probs": torch.tensor([0.3, 0.9])}, {"y": torch.tensor([1, 0])})
        combined.update(
            {"probs": torch.tensor([0.1, 0.7, 0.3, 0.9])},
            {"y": torch.tensor([0, 1, 1, 0])},
        )
        merged.merge_distributed_state(first.distributed_state())
        merged.merge_distributed_state(second.distributed_state())
        assert merged.compute() == combined.compute()

    def test_tensor_all_reduce_is_bounded_and_idempotent(self, monkeypatch) -> None:
        metric = StreamingBinaryAUROC(num_bins=8)
        metric.update({"probs": torch.tensor([0.2, 0.8])}, {"y": torch.tensor([0, 1])})
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")

        def all_reduce(counts: torch.Tensor, op: object) -> None:
            calls.append(tuple(counts.shape))
            assert op == torch.distributed.ReduceOp.SUM
            counts.mul_(2)

        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        metric.synchronize()
        metric.synchronize()
        assert calls == [(2, 8)]
        state = metric.distributed_state()
        assert state["positive_counts"].sum().item() == 2
        assert state["negative_counts"].sum().item() == 2
        with pytest.raises(RuntimeError, match="Reset"):
            metric.update({"probs": torch.tensor([0.5])}, {"y": torch.tensor([1])})
        metric.reset()
        metric.update({"probs": torch.tensor([0.5])}, {"y": torch.tensor([1])})

    def test_metric_alias_delegates_specialized_synchronization(self) -> None:
        metric = StreamingBinaryAUROC(num_bins=8)
        synchronization = Mock()
        metric.synchronize = synchronization
        MetricAlias(metric=metric, name="bounded_auc").synchronize()
        synchronization.assert_called_once_with()

    @pytest.mark.parametrize("num_bins", [True, 1, 2.5])
    def test_invalid_bin_count_is_rejected(self, num_bins: object) -> None:
        with pytest.raises(ValueError):
            StreamingBinaryAUROC(num_bins=num_bins)

    @pytest.mark.parametrize(
        "kwargs",
        [{"pred_key": ""}, {"target_key": 1}, {"from_logits": "yes"}],
    )
    def test_invalid_configuration_types_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises((TypeError, ValueError)):
            StreamingBinaryAUROC(**kwargs)
