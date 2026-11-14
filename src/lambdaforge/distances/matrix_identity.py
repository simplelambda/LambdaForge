"""Versioned numerical-operation identities for native resumable matrix blocks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import numpy as np

from lambdaforge._version import VERSION

from .StructuredMeasure import StructuredMeasure

if TYPE_CHECKING:
    from .FusedGromovWassersteinDistance import FusedGromovWassersteinDistance


def json_digest(value: Any) -> str:
    """Hash strict deterministic JSON, never repr/object addresses or machine paths."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def measure_digest(measure: StructuredMeasure, encoder: Callable[[Any], bytes] | None) -> str:
    """Exact normalized numerical content; opaque features need an explicit encoder."""
    digest = hashlib.sha256()
    for value in (measure.structure, measure.weights):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(json_digest(array.shape).encode())
        digest.update(array.data.cast("B"))
    digest.update(json_digest([measure.total_mass, measure.relational]).encode())
    if measure.features is None:
        digest.update(b"no-features")
    elif encoder is not None:
        payload = encoder(measure.features)
        if not isinstance(payload, bytes):
            raise TypeError("feature_encoder must return deterministic exact bytes.")
        digest.update(b"declared-feature-encoding:" + payload)
    else:
        array = np.asarray(measure.features)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise ValueError("Opaque features require an explicit deterministic feature_encoder.")
        if not np.isfinite(array).all():
            raise ValueError("Numeric features must be finite.")
        array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        digest.update(json_digest([array.dtype.str, array.shape]).encode())
        digest.update(array.data.cast("B"))
    return digest.hexdigest()


def solver_descriptor(
    distance: FusedGromovWassersteinDistance, metric_identity: str | None
) -> dict[str, Any]:
    """Include backend versions and authored custom-metric contract, not a guessed function hash."""
    if callable(distance.feature_metric) and not metric_identity:
        raise ValueError(
            "Callable metrics require metric_identity identifying their implementation/revision."
        )
    try:
        backend_version = version("POT")
    except PackageNotFoundError as error:
        raise ImportError("Install lambdaforge[optimal-transport] for FGW blocks.") from error
    return {
        "schema": "fgw-matrix-v1",
        "lambdaforge": VERSION,
        "POT": backend_version,
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "alpha": distance.alpha,
        "metric": metric_identity if callable(distance.feature_metric) else distance.feature_metric,
        "max_iter": distance.max_iter,
        "tol_rel": distance.tol_rel,
        "tol_abs": distance.tol_abs,
    }
