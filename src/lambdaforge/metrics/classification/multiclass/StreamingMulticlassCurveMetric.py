"""Fixed-memory one-vs-rest state for multiclass curve metrics."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from lambdaforge.metrics.classification.multiclass.MulticlassCurveAverage import (
    MulticlassCurveAverage,
)
from lambdaforge.metrics.classification.multiclass.UndefinedClassPolicy import (
    UndefinedClassPolicy,
)
from lambdaforge.metrics.Metric import Metric


class StreamingMulticlassCurveMetric(Metric):
    """Accumulate class probabilities in bounded one-vs-rest histograms.

    Persistent state consists of two CPU int64 tensors with shape
    (num_classes, num_bins), occupying exactly 16 * num_classes * num_bins
    bytes regardless of sample count.
    """

    def __init__(
        self,
        name: str,
        *,
        num_classes: int,
        num_bins: int = 4096,
        average: MulticlassCurveAverage | str = MulticlassCurveAverage.MACRO,
        undefined_class_policy: UndefinedClassPolicy | str = UndefinedClassPolicy.IGNORE,
        pred_key: str = "logits",
        target_key: str = "y",
        from_logits: bool = True,
        validate_probability_sum: bool = True,
        probability_tolerance: float = 1e-5,
    ) -> None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer greater than or equal to 2.")
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer greater than or equal to 2.")
        if not isinstance(pred_key, str) or not pred_key:
            raise ValueError("pred_key must be a non-empty string.")
        if not isinstance(target_key, str) or not target_key:
            raise ValueError("target_key must be a non-empty string.")
        if not isinstance(from_logits, bool):
            raise TypeError("from_logits must be a bool.")
        if not isinstance(validate_probability_sum, bool):
            raise TypeError("validate_probability_sum must be a bool.")
        if (
            isinstance(probability_tolerance, bool)
            or not isinstance(probability_tolerance, (int, float))
            or not math.isfinite(float(probability_tolerance))
            or float(probability_tolerance) < 0
        ):
            raise ValueError("probability_tolerance must be a finite non-negative number.")
        super().__init__(name=name, higher_is_better=True)
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.average = MulticlassCurveAverage(average)
        self.undefined_class_policy = UndefinedClassPolicy(undefined_class_policy)
        self.pred_key = pred_key
        self.target_key = target_key
        self.from_logits = from_logits
        self.validate_probability_sum = validate_probability_sum
        self.probability_tolerance = float(probability_tolerance)
        self.reset()

    @property
    def persistent_state_bytes(self) -> int:
        """Return the exact byte count occupied by both histograms."""
        return 16 * self.num_classes * self.num_bins

    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        """Add one batch without retaining sample-level tensors."""
        del context
        if self._synchronized:
            raise RuntimeError("Reset a synchronized streaming metric before updating it again.")
        predictions = outputs[self.pred_key]
        targets = batch[self.target_key]
        if not torch.is_tensor(predictions):
            raise TypeError(f"outputs[{self.pred_key!r}] must be a torch.Tensor.")
        if not torch.is_tensor(targets):
            raise TypeError(f"batch[{self.target_key!r}] must be a torch.Tensor.")
        if predictions.ndim != 2 or predictions.shape[1] != self.num_classes:
            raise ValueError(
                "Multiclass predictions must have shape "
                f"(N, {self.num_classes}); received {tuple(predictions.shape)}."
            )
        if not predictions.is_floating_point() or torch.is_complex(predictions):
            raise TypeError("Multiclass curve predictions must use a real floating dtype.")
        if targets.is_floating_point() or torch.is_complex(targets) or targets.dtype == torch.bool:
            raise TypeError("Multiclass curve targets must use an integer class dtype.")

        scores = predictions.detach().to(device="cpu", dtype=torch.float64)
        target_values = targets.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        if scores.shape[0] != target_values.numel():
            raise ValueError("Predictions and targets must have the same batch size.")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Multiclass curve predictions must contain only finite values.")
        if target_values.numel() and not bool(
            ((target_values >= 0) & (target_values < self.num_classes)).all()
        ):
            raise ValueError(f"Targets must be class indices in [0, {self.num_classes - 1}].")

        probabilities = torch.softmax(scores, dim=1) if self.from_logits else scores
        if not self.from_logits:
            if not bool(((probabilities >= 0) & (probabilities <= 1)).all()):
                raise ValueError(
                    "Probability scores must be within [0, 1]; set from_logits=True for logits."
                )
            if self.validate_probability_sum and probabilities.shape[0]:
                row_sums = probabilities.sum(dim=1)
                if not bool(
                    torch.isclose(
                        row_sums,
                        torch.ones_like(row_sums),
                        atol=self.probability_tolerance,
                        rtol=0.0,
                    ).all()
                ):
                    raise ValueError(
                        "Multiclass probability rows must sum to one within probability_tolerance."
                    )
        if target_values.numel() == 0:
            return

        indices = torch.floor(probabilities * self.num_bins).to(dtype=torch.long)
        indices.clamp_(min=0, max=self.num_bins - 1)
        offsets = torch.arange(self.num_classes, dtype=torch.long).unsqueeze(0) * self.num_bins
        total_counts = torch.bincount(
            (indices + offsets).reshape(-1),
            minlength=self.num_classes * self.num_bins,
        ).reshape(self.num_classes, self.num_bins)
        rows = torch.arange(target_values.numel(), dtype=torch.long)
        positive_flat = indices[rows, target_values] + target_values * self.num_bins
        positive_counts = torch.bincount(
            positive_flat,
            minlength=self.num_classes * self.num_bins,
        ).reshape(self.num_classes, self.num_bins)
        self._positive_counts.add_(positive_counts)
        self._negative_counts.add_(total_counts - positive_counts)

    def compute(self) -> float:
        """Return the configured scalar reduction over histogram curves."""
        if self.average is MulticlassCurveAverage.MICRO:
            return self._score_counts(
                self._positive_counts.sum(dim=0),
                self._negative_counts.sum(dim=0),
            )
        scores = self.compute_per_class()
        supports = self._positive_counts.sum(dim=1).tolist()
        return self._reduce(scores, supports)

    def compute_per_class(self) -> tuple[float, ...]:
        """Return one one-vs-rest score for every class, including NaNs."""
        return tuple(
            self._score_counts(self._positive_counts[index], self._negative_counts[index])
            for index in range(self.num_classes)
        )

    def reset(self) -> None:
        """Discard all counts and allow a new accumulation lifecycle."""
        shape = (self.num_classes, self.num_bins)
        self._positive_counts = torch.zeros(shape, dtype=torch.int64)
        self._negative_counts = torch.zeros(shape, dtype=torch.int64)
        self._synchronized = False

    def state_dict(self) -> dict[str, Any]:
        """Return independent fixed-size tensors suitable for checkpointing."""
        return {
            "positive_counts": self._positive_counts.clone(),
            "negative_counts": self._negative_counts.clone(),
            "synchronized": self._synchronized,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore validated histogram state."""
        positive_counts = self._validated_counts(state, "positive_counts")
        negative_counts = self._validated_counts(state, "negative_counts")
        synchronized = state.get("synchronized", False)
        if not isinstance(synchronized, bool):
            raise TypeError("Streaming metric state 'synchronized' must be a bool.")
        self._positive_counts = positive_counts
        self._negative_counts = negative_counts
        self._synchronized = synchronized

    def distributed_state(self) -> dict[str, torch.Tensor]:
        """Return cloned additive histograms for manual or DDP merging."""
        return {
            "positive_counts": self._positive_counts.clone(),
            "negative_counts": self._negative_counts.clone(),
        }

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Add one worker's fixed-size state without partial mutation."""
        if self._synchronized:
            raise RuntimeError("Cannot merge into an already synchronized streaming metric.")
        positive_counts = self._validated_counts(state, "positive_counts")
        negative_counts = self._validated_counts(state, "negative_counts")
        self._positive_counts.add_(positive_counts)
        self._negative_counts.add_(negative_counts)

    def synchronize(self) -> None:
        """Sum bounded histogram tensors across initialized DDP ranks."""
        import torch.distributed as distributed

        if self._synchronized:
            return
        if not distributed.is_available() or not distributed.is_initialized():
            return
        if distributed.get_world_size() <= 1:
            return
        counts = torch.stack((self._positive_counts, self._negative_counts))
        backend = str(distributed.get_backend()).lower()
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL metric synchronization requires an available CUDA device.")
            counts = counts.to(torch.device("cuda", torch.cuda.current_device()))
        distributed.all_reduce(counts, op=distributed.ReduceOp.SUM)
        counts = counts.to(device="cpu", dtype=torch.int64)
        self._positive_counts = counts[0].clone()
        self._negative_counts = counts[1].clone()
        self._synchronized = True

    def _reduce(self, scores: tuple[float, ...], supports: list[int]) -> float:
        values = list(scores)
        defined = [math.isfinite(value) for value in values]
        if self.undefined_class_policy is UndefinedClassPolicy.NAN and not all(defined):
            return float("nan")
        if self.undefined_class_policy is UndefinedClassPolicy.ZERO:
            values = [value if valid else 0.0 for value, valid in zip(values, defined, strict=True)]
            defined = [True] * len(values)
        selected = [index for index, valid in enumerate(defined) if valid]
        if not selected:
            return float("nan")
        if self.average is MulticlassCurveAverage.MACRO:
            return float(sum(values[index] for index in selected) / len(selected))
        total_support = sum(supports[index] for index in selected)
        if total_support == 0:
            return float("nan")
        return float(sum(values[index] * supports[index] for index in selected) / total_support)

    def _validated_counts(self, state: Mapping[str, Any], key: str) -> torch.Tensor:
        if key not in state:
            raise KeyError(f"Streaming metric state is missing {key!r}.")
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Streaming metric state {key!r} must be a torch.Tensor.")
        counts = value.detach().to(device="cpu")
        expected_shape = (self.num_classes, self.num_bins)
        if counts.dtype != torch.int64:
            raise TypeError(f"Streaming metric state {key!r} must use torch.int64.")
        if counts.shape != expected_shape:
            raise ValueError(f"Streaming metric state {key!r} must have shape {expected_shape}.")
        if bool((counts < 0).any()):
            raise ValueError(f"Streaming metric state {key!r} cannot contain negative counts.")
        return counts.clone()

    @abstractmethod
    def _score_counts(
        self,
        positive_counts: torch.Tensor,
        negative_counts: torch.Tensor,
    ) -> float:
        """Reduce one positive/negative histogram pair to a scalar curve score."""
