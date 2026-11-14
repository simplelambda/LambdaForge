"""Union-find superlevel filtration, processing equal-valued vertices atomically."""

from __future__ import annotations

import numpy as np

from .Branch import Branch
from .MergeEvent import MergeEvent
from .MergeTree import MergeTree


class SuperlevelMergeTree:
    """Build H0 persistence for a finite scalar field on an undirected graph.

    Edges use unambiguous shape [E, 2]. Loops/duplicates do not change topology.
    Plateau batches avoid spurious zero-lifetime branches. Ties use the elder
    rule, then the smallest birth vertex ID. Time O(N log N + E alpha(N));
    storage O(N+E), excluding returned query regions. No statistical confidence
    or significance is attached to topological prominence.
    """

    def fit(
        self,
        values: np.ndarray,
        edges: np.ndarray,
        vertex_weights: np.ndarray | None = None,
    ) -> MergeTree:
        raw = np.asarray(values)
        if (
            raw.ndim != 1
            or not len(raw)
            or np.iscomplexobj(raw)
            or not np.issubdtype(raw.dtype, np.number)
        ):
            raise ValueError("values must be a non-empty real numeric vector [N].")
        field = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(field).all():
            raise ValueError("values must be finite.")
        if not np.isfinite(float(field.max()) - float(field.min())):
            raise ValueError("The field span exceeds float64; explicitly rescale the field.")
        n = len(field)
        links = np.asarray(edges)
        if links.ndim != 2 or links.shape[1] != 2 or not np.issubdtype(links.dtype, np.integer):
            raise ValueError("edges must be integer vertex indices with shape [E, 2].")
        if np.any(links < 0) or np.any(links >= n):
            raise ValueError("Edge vertex indices must be in [0, N).")
        raw_weights = np.ones(n) if vertex_weights is None else np.asarray(vertex_weights)
        if np.iscomplexobj(raw_weights) or not np.issubdtype(raw_weights.dtype, np.number):
            raise ValueError("vertex_weights must be real numeric values.")
        weights = np.asarray(raw_weights, dtype=float)
        if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("vertex_weights must be finite nonnegative values [N].")
        total = float(np.sum(weights, dtype=np.longdouble))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("vertex_weights must have a finite positive total mass.")
        neighbors: list[list[int]] = [[] for _ in range(n)]
        for left, right in links:
            neighbors[int(left)].append(int(right))
            neighbors[int(right)].append(int(left))
        parents = np.arange(n)
        sizes = np.ones(n, dtype=np.int64)
        active = np.zeros(n, dtype=bool)
        root_event = np.full(n, -1, dtype=np.int64)

        def find(vertex: int) -> int:
            while parents[vertex] != vertex:
                parents[vertex] = parents[parents[vertex]]
                vertex = int(parents[vertex])
            return vertex

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                if sizes[a] < sizes[b]:
                    a, b = b, a
                parents[b] = a
                sizes[a] += sizes[b]

        births: list[tuple[int, ...]] = []
        levels: list[float] = []
        deaths: dict[int, tuple[float, int, int]] = {}
        histories: list[list[int]] = []
        events: list[MergeEvent] = []
        order = np.argsort(-field, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(field[order][1:] != field[order][:-1]) + 1, n]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            vertices = order[start:end]
            level = float(field[vertices[0]])
            previous = {find(v) for new in vertices for v in neighbors[int(new)] if active[v]}
            old_events = {root: int(root_event[root]) for root in previous}
            active[vertices] = True
            for new in vertices:
                for neighbor in neighbors[int(new)]:
                    if active[neighbor]:
                        union(int(new), neighbor)
            groups: dict[int, list[int]] = {}
            for new in vertices:
                groups.setdefault(find(int(new)), []).append(int(new))
            children: dict[int, list[int]] = {root: [] for root in groups}
            for old, event in old_events.items():
                children[find(old)].append(event)
            for root, added in sorted(groups.items(), key=lambda item: min(item[1])):
                child_events = tuple(sorted(children[root]))
                if child_events:
                    live = [events[e].branch for e in child_events]
                    branch = min(live, key=lambda b: (-levels[b], births[b][0]))
                    for loser in live:
                        if loser != branch:
                            deaths[loser] = (level, branch, len(events))
                else:
                    branch = len(births)
                    births.append(tuple(sorted(added)))
                    levels.append(level)
                    histories.append([])
                mass = float(weights[added].sum()) + sum(events[e].mass for e in child_events)
                index = len(events)
                events.append(MergeEvent(level, tuple(sorted(added)), child_events, branch, mass))
                histories[branch].append(index)
                root_event[root] = index
        descendants: dict[int, list[int]] = {b: [] for b in range(len(births))}
        for loser, (_, parent, _) in deaths.items():
            descendants[parent].append(loser)
        branches = tuple(
            Branch(
                b,
                birth,
                levels[b],
                deaths[b][0] if b in deaths else None,
                deaths[b][1] if b in deaths else None,
                tuple(descendants[b]),
                deaths[b][2] if b in deaths else None,
            )
            for b, birth in enumerate(births)
        )
        roots = tuple(sorted(int(root_event[i]) for i in range(n) if parents[i] == i))
        return MergeTree(
            field,
            weights,
            branches,
            tuple(events),
            roots,
            tuple(tuple(history) for history in histories),
        )
