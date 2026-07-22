"""Loss interface and built-in objective functions."""

from lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss import (
    BinaryCrossEntropyWithLogitsLoss,
)
from lambdaforge.nn.losses.BinaryFocalLoss import BinaryFocalLoss
from lambdaforge.nn.losses.ContrastiveLoss import ContrastiveLoss
from lambdaforge.nn.losses.CrossEntropyLoss import CrossEntropyLoss
from lambdaforge.nn.losses.DiceLoss import DiceLoss
from lambdaforge.nn.losses.HuberLoss import HuberLoss
from lambdaforge.nn.losses.InfoNCELoss import InfoNCELoss
from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.MeanAbsoluteErrorLoss import MeanAbsoluteErrorLoss
from lambdaforge.nn.losses.MeanSquaredErrorLoss import MeanSquaredErrorLoss
from lambdaforge.nn.losses.MulticlassFocalLoss import MulticlassFocalLoss
from lambdaforge.nn.losses.Reduction import Reduction
from lambdaforge.nn.losses.SmoothL1Loss import SmoothL1Loss
from lambdaforge.nn.losses.TripletMarginLoss import TripletMarginLoss
from lambdaforge.nn.losses.TverskyLoss import TverskyLoss
from lambdaforge.nn.losses.VariationalAutoEncoderLoss import VariationalAutoEncoderLoss

__all__ = [
    "BinaryCrossEntropyWithLogitsLoss",
    "BinaryFocalLoss",
    "ContrastiveLoss",
    "CrossEntropyLoss",
    "DiceLoss",
    "HuberLoss",
    "InfoNCELoss",
    "Loss",
    "MeanAbsoluteErrorLoss",
    "MeanSquaredErrorLoss",
    "MulticlassFocalLoss",
    "Reduction",
    "SmoothL1Loss",
    "TripletMarginLoss",
    "TverskyLoss",
    "VariationalAutoEncoderLoss",
]
