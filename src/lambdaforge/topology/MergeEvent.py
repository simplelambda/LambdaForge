"""A changed superlevel component represented without copying its full vertex set."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MergeEvent:
    """Filtration event owning newly activated vertices and references to older children."""

    level: float
    added_vertices: tuple[int, ...]
    children: tuple[int, ...]
    branch: int
    mass: float
