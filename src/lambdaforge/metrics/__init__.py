"""Task-agnostic stateful metrics."""

from lambdaforge.metrics.classification import (
    BinaryAccuracy,
    BinaryAUPRC,
    BinaryAUROC,
    BinaryBalancedAccuracy,
    BinaryCohenKappa,
    BinaryConfusionCounts,
    BinaryF1,
    BinaryMCC,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassAUPRC,
    MulticlassAUROC,
    MulticlassBalancedAccuracy,
    MulticlassF1,
)
from lambdaforge.metrics.Metric import Metric
from lambdaforge.metrics.MetricAlias import MetricAlias
from lambdaforge.metrics.regression import (
    MAE,
    MSE,
    RMSE,
    MeanMetric,
    PearsonCorrelation,
    R2Score,
    SpearmanCorrelation,
)

__all__ = [
    "BinaryAccuracy",
    "BinaryAUPRC",
    "BinaryAUROC",
    "BinaryBalancedAccuracy",
    "BinaryCohenKappa",
    "BinaryConfusionCounts",
    "BinaryF1",
    "BinaryMCC",
    "BinaryPrecision",
    "BinaryRecall",
    "BinarySpecificity",
    "MAE",
    "MSE",
    "MeanMetric",
    "Metric",
    "MetricAlias",
    "MulticlassAccuracy",
    "MulticlassAUPRC",
    "MulticlassAUROC",
    "MulticlassBalancedAccuracy",
    "MulticlassF1",
    "PearsonCorrelation",
    "R2Score",
    "RMSE",
    "SpearmanCorrelation",
]
