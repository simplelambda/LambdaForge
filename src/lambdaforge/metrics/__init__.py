"""Task-agnostic stateful metrics."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
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
        MulticlassCurveAverage,
        MulticlassF1,
        StreamingBinaryAUPRC,
        StreamingBinaryAUROC,
        StreamingBinaryCurveMetric,
        StreamingMulticlassAUPRC,
        StreamingMulticlassAUROC,
        StreamingMulticlassCurveMetric,
        UndefinedClassPolicy,
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
    "MulticlassCurveAverage",
    "MulticlassF1",
    "PearsonCorrelation",
    "R2Score",
    "RMSE",
    "SpearmanCorrelation",
    "StreamingBinaryAUPRC",
    "StreamingBinaryAUROC",
    "StreamingBinaryCurveMetric",
    "StreamingMulticlassAUPRC",
    "StreamingMulticlassAUROC",
    "StreamingMulticlassCurveMetric",
    "UndefinedClassPolicy",
]

_REGRESSION_NAMES = {
    "MAE",
    "MSE",
    "RMSE",
    "MeanMetric",
    "PearsonCorrelation",
    "R2Score",
    "SpearmanCorrelation",
}
LazyExports.install(
    __name__,
    {
        name: (
            f"lambdaforge.metrics.{name}"
            if name in {"Metric", "MetricAlias"}
            else "lambdaforge.metrics.regression"
            if name in _REGRESSION_NAMES
            else "lambdaforge.metrics.classification",
            name,
        )
        for name in __all__
    },
)
