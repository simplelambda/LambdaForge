"""Fixed-memory histogram state for streaming binary curve metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from lambdaforge.metrics.Metric import Metric


class StreamingBinaryCurveMetric(Metric):
    """Accumulate binary scores in fixed-size positive and negative histograms.

    Its two CPU int64 histograms occupy exactly 16 * num_bins bytes and never
    grow with the number of samples. Subclasses reduce them to a curve metric.
    The from_logits flag is explicit because per-batch detection would create
    inconsistent bins.
    """

    def __init__(
        self,
        name: str,
        pred_key: str = "probs",
        target_key: str = "y",
        num_bins: int = 4096,
        from_logits: bool = False,
    ) -> None:
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer greater than or equal to 2.")
        if not isinstance(pred_key, str) or not pred_key:
            raise ValueError("pred_key must be a non-empty string.")
        if not isinstance(target_key, str) or not target_key:
            raise ValueError("target_key must be a non-empty string.")
        if not isinstance(from_logits, bool):
            raise TypeError("from_logits must be a bool.")
        super().__init__(name=name, higher_is_better=True)
        self.pred_key = pred_key
        self.target_key = target_key
        self.num_bins = num_bins
        self.from_logits = from_logits
        self.reset()

    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        """Add one batch without retaining any individual sample."""
        del context
        if self._synchronized:
            raise RuntimeError("Reset a synchronized streaming metric before updating it again.")
        predictions = outputs[self.pred_key]
        targets = batch[self.target_key]
        if not torch.is_tensor(predictions):
            raise TypeError(f"outputs[{self.pred_key!r}] must be a torch.Tensor.")
        if not torch.is_tensor(targets):
            raise TypeError(f"batch[{self.target_key!r}] must be a torch.Tensor.")
        if torch.is_complex(predictions) or torch.is_complex(targets):
            raise ValueError("Binary curve predictions and targets must be real-valued.")

        scores = predictions.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        target_values = targets.detach().reshape(-1).to(device="cpu")
        if scores.numel() != target_values.numel():
            raise ValueError("Predictions and targets must contain the same number of values.")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Binary curve predictions must contain only finite values.")
        if target_values.is_floating_point() and not bool(torch.isfinite(target_values).all()):
            raise ValueError("Binary curve targets must contain only finite values.")
        if not bool(((target_values == 0) | (target_values == 1)).all()):
            raise ValueError("Binary curve targets must contain only zero or one.")

        probabilities = torch.sigmoid(scores) if self.from_logits else scores
        if not self.from_logits and not bool(((probabilities >= 0) & (probabilities <= 1)).all()):
            raise ValueError(
                "Probability scores must be within [0, 1]; set from_logits=True for logits."
            )
        indices = torch.floor(probabilities * self.num_bins).to(dtype=torch.long)
        indices.clamp_(min=0, max=self.num_bins - 1)
        positive_mask = target_values == 1
        self._positive_counts.add_(torch.bincount(indices[positive_mask], minlength=self.num_bins))
        self._negative_counts.add_(torch.bincount(indices[~positive_mask], minlength=self.num_bins))

    def reset(self) -> None:
        """Discard all counts and allow a new accumulation lifecycle."""
        self._positive_counts = torch.zeros(self.num_bins, dtype=torch.int64)
        self._negative_counts = torch.zeros(self.num_bins, dtype=torch.int64)
        self._synchronized = False

    def state_dict(self) -> dict[str, Any]:
        """Return independent tensors suitable for checkpointing."""
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
        """Return cloned additive state for manual or generic DDP merging."""
        return {
            "positive_counts": self._positive_counts.clone(),
            "negative_counts": self._negative_counts.clone(),
        }

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Add one rank's fixed-size histogram state."""
        if self._synchronized:
            raise RuntimeError("Cannot merge into an already synchronized streaming metric.")
        positive_counts = self._validated_counts(state, "positive_counts")
        negative_counts = self._validated_counts(state, "negative_counts")
        self._positive_counts.add_(positive_counts)
        self._negative_counts.add_(negative_counts)

    def synchronize(self) -> None:
        """Sum histogram counts with a bounded tensor collective across DDP ranks."""
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

    def _validated_counts(self, state: Mapping[str, Any], key: str) -> torch.Tensor:
        """Return one independent, non-negative and correctly shaped count tensor."""
        if key not in state:
            raise KeyError(f"Streaming metric state is missing {key!r}.")
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Streaming metric state {key!r} must be a torch.Tensor.")
        counts = value.detach().to(device="cpu")
        if counts.dtype != torch.int64:
            raise TypeError(f"Streaming metric state {key!r} must use torch.int64.")
        if counts.shape != (self.num_bins,):
            raise ValueError(f"Streaming metric state {key!r} must have shape ({self.num_bins},).")
        if bool((counts < 0).any()):
            raise ValueError(f"Streaming metric state {key!r} cannot contain negative counts.")
        return counts.clone()
