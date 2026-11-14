"""An attributed finite probability measure with explicit intrinsic geometry."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class StructuredMeasure:
    """One weighted object, not a collection of objects to cluster.

    ``structure`` is a finite, nonnegative symmetric [N,N] dissimilarity matrix with
    zero diagonal. Triangle inequalities are not tested. ``features`` has N rows/items;
    these can be numeric vectors or application-defined objects when a custom feature
    metric/precomputed cross-distance is supplied to FGW. Relative nonnegative weights
    are normalized explicitly here to probabilities; ``total_mass`` retains their sum.
    This is balanced probability FGW, not unbalanced transport of the original masses.
    Explicit ``relational=True`` permits nonzero diagonal symmetric relationships
    (for example a POT barycenter); it does not turn those relations into a metric.
    """

    structure: Any
    features: Any = None
    weights: Any = None
    relational: bool = field(default=False, kw_only=True)
    total_mass: float = field(init=False)

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.structure) or np.iscomplexobj(self.weights):
            raise ValueError("structure and weights must be real, not complex.")
        structure = np.array(self.structure, dtype=np.float64, copy=True)
        if structure.ndim != 2 or not len(structure) or structure.shape[0] != structure.shape[1]:
            raise ValueError("structure must be a nonempty square [N,N] matrix.")
        if not np.isfinite(structure).all() or np.any(structure < 0):
            raise ValueError(
                "structure must be finite and nonnegative; disconnected infinities fail."
            )
        if not np.allclose(structure, structure.T, atol=1e-12, rtol=1e-10):
            raise ValueError("structure must be symmetric (undirected relationships).")
        if not isinstance(self.relational, bool):
            raise TypeError("relational must be an explicit boolean.")
        if not self.relational and not np.allclose(np.diag(structure), 0, atol=1e-12, rtol=0):
            raise ValueError("structure must have a zero diagonal.")
        count = len(structure)
        weights = (
            np.ones(count)
            if self.weights is None
            else np.array(self.weights, dtype=np.float64, copy=True)
        )
        if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("weights must be finite nonnegative [N] values.")
        mass = float(weights.sum())
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("weights must have finite positive total mass.")
        weights /= mass
        features = copy.deepcopy(self.features)
        if features is not None:
            try:
                size = len(features)
            except TypeError as error:
                raise ValueError("features must have one row/item per point.") from error
            if size != count:
                raise ValueError("features must have one row/item per point.")
            if isinstance(features, np.ndarray):
                features.setflags(write=False)
            elif isinstance(features, list):
                features = tuple(features)
        structure.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "total_mass", mass)

    @classmethod
    def from_points(
        cls,
        points: ArrayLike,
        *,
        features: Any = None,
        weights: ArrayLike | None = None,
        metric: str = "euclidean",
        max_structure_bytes: int = 256 * 1024**2,
    ) -> StructuredMeasure:
        """Build intrinsic distances from [N,D] coordinates, without normalizing units.

        Coordinates are only geometry: pass features explicitly if they should also be
        node attributes. A caller can instead supply shortest paths or any finite
        undirected dissimilarities to the constructor. SciPy is provided by the OT extra.
        """
        if np.iscomplexobj(points):
            raise ValueError("points must be real, not complex.")
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
            raise ValueError("points must be a finite nonempty [N,D] array.")
        if (
            isinstance(max_structure_bytes, bool)
            or not isinstance(max_structure_bytes, int)
            or max_structure_bytes <= 0
        ):
            raise ValueError("max_structure_bytes must be a positive integer.")
        if len(values) ** 2 * 8 > max_structure_bytes:
            raise MemoryError("Dense intrinsic distance matrix exceeds max_structure_bytes.")
        try:
            from scipy.spatial.distance import cdist
        except ImportError as error:
            raise ImportError(
                "Install lambdaforge[optimal-transport] for point geometry."
            ) from error
        return cls(cdist(values, values, metric=metric), features=features, weights=weights)
