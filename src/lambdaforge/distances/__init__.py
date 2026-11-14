"""Distances between complete structured objects, independent of neural point distances."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from .FGWBarycenter import FGWBarycenter
    from .FGWBarycenterResult import FGWBarycenterResult
    from .FGWMatrixBlock import FGWMatrixBlock
    from .FGWMatrixBuilder import FGWMatrixBuilder
    from .FGWMatrixResult import FGWMatrixResult
    from .FGWResult import FGWResult
    from .FusedGromovWassersteinDistance import FusedGromovWassersteinDistance
    from .StructuredMeasure import StructuredMeasure
    from .validation import validate_dissimilarity

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.distances.{name}", name)
        for name in (
            "FGWResult",
            "FusedGromovWassersteinDistance",
            "StructuredMeasure",
            "FGWBarycenter",
            "FGWBarycenterResult",
            "FGWMatrixBuilder",
            "FGWMatrixBlock",
            "FGWMatrixResult",
        )
    }
    | {"validate_dissimilarity": ("lambdaforge.distances.validation", "validate_dissimilarity")},
)

__all__ = [
    "FGWBarycenter",
    "FGWBarycenterResult",
    "FGWMatrixBuilder",
    "FGWMatrixBlock",
    "FGWMatrixResult",
    "FGWResult",
    "FusedGromovWassersteinDistance",
    "StructuredMeasure",
    "validate_dissimilarity",
]
