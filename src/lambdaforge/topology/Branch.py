"""Persistent H0 branch evidence, independent of any selection policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Branch:
    """Elder-rule branch; an essential branch has death/persistence=None.

    Branch IDs are deterministic for a fixed vertex indexing. Equal-birth elders
    are chosen by smallest birth-plateau vertex ID. Permuting those IDs can rename
    tied branches, not change the superlevel component partitions.
    """

    id: int
    birth_vertices: tuple[int, ...]
    birth: float
    death: float | None
    parent: int | None
    children: tuple[int, ...]
    merge_event: int | None

    @property
    def essential(self) -> bool:
        return self.death is None

    @property
    def persistence(self) -> float | None:
        return None if self.death is None else self.birth - self.death
