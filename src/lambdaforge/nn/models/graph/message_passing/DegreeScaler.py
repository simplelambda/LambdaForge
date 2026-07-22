"""Type-safe degree scalers for Principal Neighbourhood Aggregation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import Enum

import torch


class DegreeScaler(str, Enum):
    r"""Select one PNA degree-dependent rescaling policy.

    `amplification` and `attenuation` compare :math:`log(d + 1)`
    against its training-set average. `linear` and `inverse_linear`
    compare the incoming degree directly against its training-set average.
    Denominators are clamped by `epsilon` so isolated nodes stay finite.
    """

    IDENTITY = "identity"
    AMPLIFICATION = "amplification"
    ATTENUATION = "attenuation"
    LINEAR = "linear"
    INVERSE_LINEAR = "inverse_linear"

    def factor(
        self,
        degree: torch.Tensor,
        *,
        average_degree: float,
        average_log_degree: float,
        epsilon: float,
    ) -> torch.Tensor:
        """Return one multiplicative factor for every node degree."""
        self.validate_statistics(average_degree, average_log_degree, epsilon)
        if not isinstance(degree, torch.Tensor):
            raise TypeError("degree must be a torch.Tensor.")
        if degree.ndim != 1:
            raise ValueError("degree must have shape (N,).")
        if degree.dtype == torch.bool or degree.is_complex():
            raise TypeError("degree must contain real numeric values.")
        if degree.numel() and (not bool(torch.isfinite(degree).all()) or bool((degree < 0).any())):
            raise ValueError("degree values must be finite and non-negative.")
        if not degree.is_floating_point():
            degree = degree.to(dtype=torch.get_default_dtype())
        elif degree.dtype in {torch.float16, torch.bfloat16}:
            degree = degree.to(dtype=torch.float32)
        if self is DegreeScaler.IDENTITY:
            return torch.ones_like(degree)
        if self is DegreeScaler.AMPLIFICATION:
            return degree.log1p() / average_log_degree
        if self is DegreeScaler.ATTENUATION:
            return average_log_degree / degree.log1p().clamp_min(epsilon)
        if self is DegreeScaler.LINEAR:
            return degree / average_degree
        return average_degree / degree.clamp_min(epsilon)

    def scale(
        self,
        values: torch.Tensor,
        degree: torch.Tensor,
        *,
        average_degree: float,
        average_log_degree: float,
        epsilon: float,
    ) -> torch.Tensor:
        """Scale node values while preserving all feature dimensions."""
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")
        if not isinstance(degree, torch.Tensor):
            raise TypeError("degree must be a torch.Tensor.")
        if values.ndim < 1 or values.shape[0] != degree.shape[0]:
            raise ValueError("values and degree must have the same leading dimension.")
        working_dtype = (
            torch.float32 if values.dtype in {torch.float16, torch.bfloat16} else values.dtype
        )
        working_values = values.to(dtype=working_dtype)
        factor = self.factor(
            degree.to(device=values.device, dtype=working_dtype),
            average_degree=average_degree,
            average_log_degree=average_log_degree,
            epsilon=epsilon,
        )
        scaled = working_values * factor.view(-1, *([1] * (values.ndim - 1)))
        return scaled.to(dtype=values.dtype)

    @classmethod
    def normalize_many(
        cls,
        values: DegreeScaler | str | Iterable[DegreeScaler | str],
    ) -> tuple[DegreeScaler, ...]:
        """Return a non-empty, duplicate-free tuple of scaler policies."""
        raw_values = (values,) if isinstance(values, (cls, str)) else tuple(values)
        normalized = tuple(cls(value) for value in raw_values)
        if not normalized:
            raise ValueError("scalers must contain at least one value.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("scalers cannot contain duplicate values.")
        return normalized

    @staticmethod
    def validate_statistics(
        average_degree: float,
        average_log_degree: float,
        epsilon: float,
    ) -> None:
        """Require finite, strictly positive reference statistics."""
        for name, value in (
            ("average_degree", average_degree),
            ("average_log_degree", average_log_degree),
            ("epsilon", epsilon),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
