"""Bounded-state multiclass curve metric contracts."""

import json
import math

import pytest
import torch

from lambdaforge.experiments import ObjectFactory
from lambdaforge.metrics import (
    MulticlassCurveAverage,
    StreamingMulticlassAUPRC,
    StreamingMulticlassAUROC,
    UndefinedClassPolicy,
)


class TestStreamingMulticlassMetrics:
    """Verify reductions, memory bounds, validation and YAML construction."""

    METRIC_TYPES = (StreamingMulticlassAUROC, StreamingMulticlassAUPRC)

    @pytest.mark.parametrize("metric_type", METRIC_TYPES)
    @pytest.mark.parametrize("average", list(MulticlassCurveAverage))
    def test_perfect_probabilities_score_one(self, metric_type: type, average: object) -> None:
        metric = metric_type(
            num_classes=3,
            num_bins=32,
            average=average,
            pred_key="probs",
            from_logits=False,
        )
        metric.update(
            {
                "probs": torch.tensor(
                    [
                        [0.90, 0.05, 0.05],
                        [0.05, 0.90, 0.05],
                        [0.05, 0.05, 0.90],
                        [0.80, 0.10, 0.10],
                    ]
                )
            },
            {"y": torch.tensor([0, 1, 2, 0])},
        )
        assert metric.compute() == 1.0
        assert metric.compute_per_class() == (1.0, 1.0, 1.0)

    @pytest.mark.parametrize("metric_type", METRIC_TYPES)
    def test_split_batches_and_merged_states_match_combined(self, metric_type: type) -> None:
        probabilities = torch.tensor(
            [
                [0.70, 0.20, 0.10],
                [0.20, 0.60, 0.20],
                [0.10, 0.20, 0.70],
                [0.40, 0.35, 0.25],
            ]
        )
        targets = torch.tensor([0, 1, 2, 1])
        first = metric_type(num_classes=3, num_bins=64, pred_key="p", from_logits=False)
        second = metric_type(num_classes=3, num_bins=64, pred_key="p", from_logits=False)
        combined = metric_type(num_classes=3, num_bins=64, pred_key="p", from_logits=False)
        first.update({"p": probabilities[:2]}, {"y": targets[:2]})
        second.update({"p": probabilities[2:]}, {"y": targets[2:]})
        combined.update({"p": probabilities}, {"y": targets})
        first.merge_distributed_state(second.distributed_state())
        assert first.compute_per_class() == combined.compute_per_class()
        assert first.compute() == combined.compute()

    @pytest.mark.parametrize("metric_type", METRIC_TYPES)
    def test_state_is_bounded_by_classes_times_bins(self, metric_type: type) -> None:
        metric = metric_type(num_classes=4, num_bins=17)
        for _ in range(10):
            metric.update(
                {"logits": torch.randn(101, 4)},
                {"y": torch.arange(101) % 4},
            )
        state = metric.distributed_state()
        assert state["positive_counts"].shape == (4, 17)
        assert state["negative_counts"].shape == (4, 17)
        assert metric.persistent_state_bytes == 16 * 4 * 17
        assert sum(t.numel() * t.element_size() for t in state.values()) == 16 * 4 * 17

    @pytest.mark.parametrize("metric_type", METRIC_TYPES)
    def test_undefined_class_policy_is_explicit(self, metric_type: type) -> None:
        payload = {"p": torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])}
        batch = {"y": torch.tensor([0, 1])}
        metrics = [
            metric_type(
                num_classes=3,
                num_bins=16,
                pred_key="p",
                from_logits=False,
                undefined_class_policy=policy,
            )
            for policy in UndefinedClassPolicy
        ]
        for metric in metrics:
            metric.update(payload, batch)
        ignore, propagate, zero = metrics
        assert math.isnan(ignore.compute_per_class()[2])
        assert ignore.compute() == 1.0
        assert math.isnan(propagate.compute())
        assert math.isclose(zero.compute(), 2.0 / 3.0)

    @pytest.mark.parametrize("metric_type", METRIC_TYPES)
    def test_micro_uses_all_one_vs_rest_pairs(self, metric_type: type) -> None:
        metric = metric_type(
            num_classes=3,
            num_bins=100,
            average=MulticlassCurveAverage.MICRO,
            pred_key="p",
            from_logits=False,
        )
        metric.update(
            {"p": torch.tensor([[0.60, 0.30, 0.10], [0.20, 0.50, 0.30], [0.20, 0.20, 0.60]])},
            {"y": torch.tensor([0, 1, 2])},
        )
        assert metric.compute() == 1.0

    def test_high_resolution_histograms_match_exact_curve_metrics(self) -> None:
        from torchmetrics.classification import (
            BinaryAUROC,
            BinaryAveragePrecision,
            MulticlassAUROC,
            MulticlassAveragePrecision,
        )

        generator = torch.Generator().manual_seed(4)
        probabilities = torch.softmax(torch.randn(20, 3, generator=generator), dim=1)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        one_hot = torch.nn.functional.one_hot(targets, 3).reshape(-1)
        cases = (
            (
                StreamingMulticlassAUROC,
                MulticlassAUROC,
                BinaryAUROC,
            ),
            (
                StreamingMulticlassAUPRC,
                MulticlassAveragePrecision,
                BinaryAveragePrecision,
            ),
        )
        for streaming_type, exact_type, binary_type in cases:
            macro = streaming_type(
                num_classes=3,
                num_bins=100_000,
                pred_key="p",
                from_logits=False,
            )
            micro = streaming_type(
                num_classes=3,
                num_bins=100_000,
                average="micro",
                pred_key="p",
                from_logits=False,
            )
            macro.update({"p": probabilities}, {"y": targets})
            micro.update({"p": probabilities}, {"y": targets})
            exact_macro = float(exact_type(num_classes=3, average="macro")(probabilities, targets))
            exact_per_class = exact_type(num_classes=3, average=None)(probabilities, targets)
            exact_micro = float(binary_type()(probabilities.reshape(-1), one_hot))
            assert math.isclose(macro.compute(), exact_macro, abs_tol=1e-6)
            assert all(
                math.isclose(actual, float(expected), abs_tol=1e-6)
                for actual, expected in zip(
                    macro.compute_per_class(),
                    exact_per_class,
                    strict=True,
                )
            )
            assert math.isclose(micro.compute(), exact_micro, abs_tol=1e-6)

    def test_tensor_synchronization_uses_bounded_collective(self, monkeypatch) -> None:
        metric = StreamingMulticlassAUROC(num_classes=3, num_bins=8)
        metric.update(
            {"logits": torch.tensor([[2.0, 0.0, -1.0], [-1.0, 2.0, 0.0]])},
            {"y": torch.tensor([0, 1])},
        )
        calls: list[tuple[int, ...]] = []
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
        assert calls == [(2, 3, 8)]
        assert metric.distributed_state()["positive_counts"].sum().item() == 4

    def test_yaml_factory_constructs_public_metric(self) -> None:
        metric = ObjectFactory.build(
            {
                "target": "lambdaforge.metrics.StreamingMulticlassAUROC",
                "params": {
                    "num_classes": 3,
                    "num_bins": 32,
                    "average": "weighted",
                },
            }
        )
        assert isinstance(metric, StreamingMulticlassAUROC)
        assert metric.average is MulticlassCurveAverage.WEIGHTED

    def test_state_round_trip_has_same_scores(self) -> None:
        metric = StreamingMulticlassAUROC(num_classes=3, num_bins=8)
        metric.update({"logits": torch.randn(5, 3)}, {"y": torch.tensor([0, 1, 2, 0, 1])})
        restored = StreamingMulticlassAUROC(num_classes=3, num_bins=8)
        restored.load_state_dict(metric.state_dict())
        assert restored.compute_per_class() == metric.compute_per_class()
        assert json.dumps({"num_classes": restored.num_classes, "num_bins": restored.num_bins})

    @pytest.mark.parametrize(
        ("outputs", "batch", "error"),
        [
            ({"logits": torch.randn(2, 4)}, {"y": torch.tensor([0, 1])}, ValueError),
            ({"logits": torch.randn(2, 3)}, {"y": torch.tensor([0, 3])}, ValueError),
            ({"logits": torch.randn(2, 3)}, {"y": torch.tensor([0.0, 1.0])}, TypeError),
            (
                {"probs": torch.tensor([[0.2, 0.2, 0.2]])},
                {"y": torch.tensor([0])},
                ValueError,
            ),
        ],
    )
    def test_invalid_updates_do_not_change_state(
        self, outputs: dict, batch: dict, error: type[Exception]
    ) -> None:
        metric = StreamingMulticlassAUROC(
            num_classes=3,
            num_bins=8,
            pred_key="probs" if "probs" in outputs else "logits",
            from_logits="logits" in outputs,
        )
        with pytest.raises(error):
            metric.update(outputs, batch)
        assert metric.distributed_state()["positive_counts"].sum().item() == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_classes": 1},
            {"num_classes": True},
            {"num_classes": 3, "num_bins": 1},
            {"num_classes": 3, "average": "none"},
            {"num_classes": 3, "undefined_class_policy": "guess"},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises((TypeError, ValueError)):
            StreamingMulticlassAUROC(**kwargs)
