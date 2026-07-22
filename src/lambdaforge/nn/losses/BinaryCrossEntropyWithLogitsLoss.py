"""Implementation of the BinaryCrossEntropyWithLogitsLoss object."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class BinaryCrossEntropyWithLogitsLoss(Loss):
    r"""Binary cross-entropy loss with logits.

    This loss expects raw logits, not sigmoid probabilities.

    Parameters
    ----------
    output_key : str
        Key used to read logits from model outputs.
    target_key : str
        Key used to read targets from the batch.
    weight : float
        Multiplicative factor applied to the loss.
    reduction : Reduction | str
        Scalar reduction mode: ``"mean"`` or ``"sum"``.
    pos_weight : torch.Tensor | Sequence[float] | None
        Optional positive class weights. Sequences are convenient in YAML.
    name : str
        Unique log name, useful when configuring multiple BCE terms.
    """

    def __init__(
        self,
        output_key: str = "logits",
        target_key: str = "target",
        weight: float = 1.0,
        reduction: Reduction | str = Reduction.MEAN,
        pos_weight: torch.Tensor | Sequence[float] | None = None,
        name: str = "binary_cross_entropy_with_logits",
    ) -> None:
        super().__init__(name=name, weight=weight)

        self.output_key = output_key
        self.target_key = target_key
        self.reduction = Reduction.from_value(reduction)

        resolved_pos_weight = (
            None
            if pos_weight is None
            else torch.as_tensor(pos_weight, dtype=torch.float32).detach().clone()
        )
        if resolved_pos_weight is not None and resolved_pos_weight.numel() < 1:
            raise ValueError("pos_weight cannot be empty.")
        self.pos_weight: torch.Tensor | None
        self.register_buffer("pos_weight", resolved_pos_weight)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del context

        logits = outputs[self.output_key]
        target = batch[self.target_key].to(dtype=logits.dtype)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=(
                None if self.pos_weight is None else self.pos_weight.to(dtype=logits.dtype)
            ),
            reduction=self.reduction.value,
        )
        return self.weight * loss
