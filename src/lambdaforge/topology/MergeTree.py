"""Immutable event hierarchy for a vertex-superlevel H0 filtration."""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

from .Branch import Branch
from .MergeEvent import MergeEvent


class MergeTree:
    """Read-only fitted hierarchy. Regions are output-sized views of event subtrees.

    Every event owns only newly activated vertices; a DFS interval indexes each
    complete component. No list of full regions is retained at every threshold.
    A branch exists at birth >= u > death (including its birth, excluding death).
    """

    def __init__(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        branches: tuple[Branch, ...],
        events: tuple[MergeEvent, ...],
        roots: tuple[int, ...],
        histories: tuple[tuple[int, ...], ...],
    ) -> None:
        self.values = values.copy()
        self.vertex_weights = weights.copy()
        self.values.setflags(write=False)
        self.vertex_weights.setflags(write=False)
        self.branches = branches
        self.events = events
        self.roots = roots
        self._histories = histories
        self._levels = tuple(tuple(-events[e].level for e in history) for history in histories)
        starts = np.empty(len(events), dtype=np.int64)
        ends = np.empty(len(events), dtype=np.int64)
        order: list[int] = []
        stack = [(root, False) for root in reversed(roots)]
        while stack:
            index, visited = stack.pop()
            if visited:
                ends[index] = len(order)
            else:
                starts[index] = len(order)
                order.extend(events[index].added_vertices)
                stack.append((index, True))
                stack.extend((child, False) for child in reversed(events[index].children))
        self._order = np.asarray(order, dtype=np.int64)
        self._order.setflags(write=False)
        self._starts, self._ends = starts, ends

    @staticmethod
    def _threshold(threshold: float) -> float:
        level = float(threshold)
        if not np.isfinite(level):
            raise ValueError("A filtration threshold must be finite.")
        return level

    def _region(self, event: int) -> np.ndarray:
        return self._order[self._starts[event] : self._ends[event]]

    def components_at(self, threshold: float) -> tuple[np.ndarray, ...]:
        """Return connected vertex sets of {values >= threshold}, without graph traversal."""
        level = self._threshold(threshold)
        selected: list[np.ndarray] = []
        stack = list(reversed(self.roots))
        while stack:
            index = stack.pop()
            event = self.events[index]
            if event.level >= level:
                selected.append(self._region(index))
            else:
                stack.extend(reversed(event.children))
        return tuple(selected)

    def _event_at(self, branch_id: int, threshold: float) -> int:
        level = self._threshold(threshold)
        branch = self._branch(branch_id)
        if level > branch.birth or (branch.death is not None and level <= branch.death):
            raise ValueError("Threshold is outside the branch lifetime (death, birth].")
        position = bisect_right(self._levels[branch_id], -level) - 1
        return self._histories[branch_id][position]

    def _branch(self, branch_id: int) -> Branch:
        if (
            isinstance(branch_id, bool)
            or not isinstance(branch_id, int)
            or not 0 <= branch_id < len(self.branches)
        ):
            raise ValueError("Unknown branch ID.")
        return self.branches[branch_id]

    def region_at(self, branch_id: int, threshold: float) -> np.ndarray:
        """Follow an elder-rule branch, never across its death/merge into its parent."""
        return self._region(self._event_at(branch_id, threshold))

    def mass_curve(self, branch_id: int) -> np.ndarray:
        """Descending event levels and post-event weighted masses, shape [events, 2]."""
        self._branch(branch_id)
        result = np.asarray(
            [(self.events[e].level, self.events[e].mass) for e in self._histories[branch_id]]
        )
        result.setflags(write=False)
        return result

    def persistence_diagram(self) -> tuple[tuple[float, float | None], ...]:
        """Birth/death pairs; None explicitly denotes an essential H0 class."""
        return tuple((branch.birth, branch.death) for branch in self.branches)
