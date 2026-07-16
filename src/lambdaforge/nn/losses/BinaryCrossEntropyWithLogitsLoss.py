"""Implementation of the BinaryCrossEntropyWithLogitsLoss object."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss


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
    reduction : str
        Reduction mode: ``"mean"``, ``"sum"`` or ``"none"``.
    pos_weight : torch.Tensor | None
        Optional positive class weight.
    name : str
        Unique log name, useful when configuring multiple BCE terms.
    """

    def __init__(
        self,
        output_key: str = "logits",
        target_key: str = "target",
        weight: float = 1.0,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
        name: str = "binary_cross_entropy_with_logits",
    ) -> None:
        super().__init__(name=name, weight=weight)

        self.output_key = output_key
        self.target_key = target_key
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

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
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
        return self.weight * loss
