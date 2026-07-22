"""Multiclass classification metric implementations."""

from lambdaforge.metrics.classification.multiclass.MulticlassAccuracy import (
    MulticlassAccuracy,
)
from lambdaforge.metrics.classification.multiclass.MulticlassAUPRC import MulticlassAUPRC
from lambdaforge.metrics.classification.multiclass.MulticlassAUROC import MulticlassAUROC
from lambdaforge.metrics.classification.multiclass.MulticlassBalancedAccuracy import (
    MulticlassBalancedAccuracy,
)
from lambdaforge.metrics.classification.multiclass.MulticlassCurveAverage import (
    MulticlassCurveAverage,
)
from lambdaforge.metrics.classification.multiclass.MulticlassF1 import MulticlassF1
from lambdaforge.metrics.classification.multiclass.MulticlassMetric import MulticlassMetric
from lambdaforge.metrics.classification.multiclass.StreamingMulticlassAUPRC import (
    StreamingMulticlassAUPRC,
)
from lambdaforge.metrics.classification.multiclass.StreamingMulticlassAUROC import (
    StreamingMulticlassAUROC,
)
from lambdaforge.metrics.classification.multiclass.StreamingMulticlassCurveMetric import (
    StreamingMulticlassCurveMetric,
)
from lambdaforge.metrics.classification.multiclass.UndefinedClassPolicy import (
    UndefinedClassPolicy,
)

__all__ = [
    "MulticlassAccuracy",
    "MulticlassAUPRC",
    "MulticlassAUROC",
    "MulticlassBalancedAccuracy",
    "MulticlassCurveAverage",
    "MulticlassF1",
    "MulticlassMetric",
    "StreamingMulticlassAUPRC",
    "StreamingMulticlassAUROC",
    "StreamingMulticlassCurveMetric",
    "UndefinedClassPolicy",
]
