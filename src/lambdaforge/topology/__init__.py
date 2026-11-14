"""Generic graph topology; no clustering, execution engine or domain-specific geometry."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from .Branch import Branch
    from .MergeEvent import MergeEvent
    from .MergeTree import MergeTree
    from .overlap import weighted_dice, weighted_iou
    from .RegionSelection import RegionSelection
    from .StableRegion import StableRegion
    from .StableRegionSelector import (
        StableRegionSelector,
        discrete_membership,
        uniform_membership,
    )
    from .SuperlevelMergeTree import SuperlevelMergeTree

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.topology.{name}", name)
        for name in (
            "Branch",
            "MergeEvent",
            "MergeTree",
            "SuperlevelMergeTree",
            "StableRegion",
            "RegionSelection",
        )
    }
    | {name: ("lambdaforge.topology.overlap", name) for name in ("weighted_iou", "weighted_dice")}
    | {
        name: ("lambdaforge.topology.StableRegionSelector", name)
        for name in (
            "StableRegionSelector",
            "uniform_membership",
            "discrete_membership",
        )
    },
)

__all__ = [
    "Branch",
    "MergeEvent",
    "MergeTree",
    "SuperlevelMergeTree",
    "StableRegion",
    "StableRegionSelector",
    "RegionSelection",
    "uniform_membership",
    "discrete_membership",
    "weighted_iou",
    "weighted_dice",
]
