"""Loss interface and built-in objective functions."""

from lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss import (
    BinaryCrossEntropyWithLogitsLoss,
)
from lambdaforge.nn.losses.Loss import Loss

__all__ = ["BinaryCrossEntropyWithLogitsLoss", "Loss"]
