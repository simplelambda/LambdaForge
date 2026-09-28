"""Central human formatting for authored scientific parameters.

Machine mappings are never changed; these helpers only produce compact presentation labels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from lambdaforge.hpo.ParameterSpace import ParameterDescriptor, ParameterSpace


def _compact_exponent(value: str) -> str:
    return value.replace("e-0", "e-").replace("e+0", "e+")


def format_parameter_value(
    value: Any,
    descriptor: ParameterDescriptor | None = None,
) -> str:
    """Format one parameter without losing exact values in the underlying JSON."""
    if value is None:
        return "—"
    if descriptor is not None and descriptor.kind in {"categorical", "boolean", "integer"}:
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if descriptor is not None and descriptor.scale == "log":
            if value == 0:
                return f"{value:.3g}"
            return _compact_exponent(f"{value:.3e}")
        magnitude = abs(value)
        if magnitude and (magnitude < 1e-3 or magnitude >= 1e5):
            return _compact_exponent(f"{value:.4g}")
        return f"{value:.6g}"
    return str(value)


def format_parameter_vector(
    parameters: Mapping[str, Any],
    parameter_space: ParameterSpace | None = None,
) -> str:
    """Return a readable compact vector while leaving machine parameter values untouched."""
    parts: list[str] = []
    for name, value in sorted(parameters.items()):
        descriptor = None
        if parameter_space is not None and name in parameter_space.names:
            descriptor = parameter_space.descriptor(name)
        parts.append(f"{name}={format_parameter_value(value, descriptor)}")
    return ", ".join(parts)


__all__ = ["format_parameter_value", "format_parameter_vector"]
