"""Distances between complete structured objects, independent of neural point distances."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from .FGWResult import FGWResult
    from .FusedGromovWassersteinDistance import FusedGromovWassersteinDistance
    from .StructuredMeasure import StructuredMeasure

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.distances.{name}", name)
        for name in ("FGWResult", "FusedGromovWassersteinDistance", "StructuredMeasure")
    },
)

__all__ = ["FGWResult", "FusedGromovWassersteinDistance", "StructuredMeasure"]
