"""Declared backend capabilities, not a separate distance registry."""

EUCLIDEAN = frozenset({"euclidean", "squared_euclidean"})
DENSITY_METRICS = frozenset({"euclidean", "manhattan", "minkowski", "chebyshev", "cosine"})
