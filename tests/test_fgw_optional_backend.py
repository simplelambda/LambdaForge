"""New transport APIs remain importable and diagnostic without the optional solver."""

import builtins
from importlib.metadata import PackageNotFoundError

import numpy as np
import pytest

from lambdaforge.distances import (
    FGWBarycenter,
    FGWMatrixBuilder,
    FusedGromovWassersteinDistance,
    StructuredMeasure,
)


def test_barycenter_missing_backend_is_actionable(monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "ot.gromov":
            raise ImportError("test missing backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    obj = StructuredMeasure([[0.0, 1.0], [1.0, 0.0]], np.array([[0.0], [1.0]]))
    with pytest.raises(ImportError, match="optimal-transport"):
        FGWBarycenter(2).fit([obj])


def test_matrix_missing_backend_metadata_is_actionable(monkeypatch):
    import lambdaforge.distances.matrix_identity as identity

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(identity, "version", missing)
    obj = StructuredMeasure([[0.0, 1.0], [1.0, 0.0]], np.array([[0.0], [1.0]]))
    with pytest.raises(ImportError, match="optimal-transport"):
        FGWMatrixBuilder(FusedGromovWassersteinDistance()).build([obj])
