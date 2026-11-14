"""Accepted spatial regions and explicit rejection evidence."""

from collections.abc import Mapping
from dataclasses import dataclass

from .StableRegion import StableRegion


@dataclass(frozen=True, slots=True)
class RegionSelection:
    """Accepted spatial regions and explicit per-branch rejection reasons."""

    regions: tuple[StableRegion, ...]
    rejected: Mapping[int, str]
    overlap_policy: str
