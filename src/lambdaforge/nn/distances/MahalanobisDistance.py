"""Implementation of the MahalanobisDistance object."""

from __future__ import annotations

import math

import torch
from torch import nn

from lambdaforge.nn.distances.Distance import Distance


class MahalanobisDistance(Distance):
    r"""Pairwise Mahalanobis distance with a positive-semidefinite metric.

    The configured precision matrix is factorized once during initialization.
    When trainable, the factor is optimized instead of the precision matrix,
    so the effective metric remains positive semidefinite throughout training.

    Parameters
    ----------
    num_features : int
        Input feature dimension.
    precision_matrix : torch.Tensor | None
        Optional symmetric positive-semidefinite matrix with shape
        ``(num_features, num_features)``. The identity is used by default.
    trainable : bool
        Whether the metric factor is learnable. Default: ``False``.
    squared : bool
        Return squared distances instead of distances. Default: ``False``.
    eigenvalue_tolerance : float
        Non-negative tolerance used for symmetry and PSD validation. Small
        negative eigenvalues within this tolerance are clamped to zero.
        Default: ``1e-7``.
    """

    def __init__(
        self,
        num_features: int,
        precision_matrix: torch.Tensor | None = None,
        trainable: bool = False,
        squared: bool = False,
        eigenvalue_tolerance: float = 1e-7,
    ) -> None:
        super().__init__()
        if isinstance(num_features, bool) or not isinstance(num_features, int):
            raise TypeError("num_features must be an integer")
        if num_features <= 0:
            raise ValueError("num_features must be greater than zero")
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean")
        if not isinstance(squared, bool):
            raise TypeError("squared must be a boolean")
        if isinstance(eigenvalue_tolerance, bool) or not isinstance(
            eigenvalue_tolerance, (int, float)
        ):
            raise TypeError("eigenvalue_tolerance must be a real number")
        if not math.isfinite(float(eigenvalue_tolerance)) or float(eigenvalue_tolerance) < 0.0:
            raise ValueError("eigenvalue_tolerance must be finite and non-negative")

        self.num_features = num_features
        self.trainable = trainable
        self.squared = squared
        self.eigenvalue_tolerance = float(eigenvalue_tolerance)

        if precision_matrix is None:
            factor = torch.eye(num_features)
        else:
            if not isinstance(precision_matrix, torch.Tensor):
                raise TypeError("precision_matrix must be a torch.Tensor or None")
            if not precision_matrix.is_floating_point():
                raise TypeError("precision_matrix must be a floating-point tensor")
            if precision_matrix.ndim != 2 or precision_matrix.shape != (
                num_features,
                num_features,
            ):
                raise ValueError(
                    f"precision_matrix must have shape ({num_features}, {num_features})"
                )
            if not torch.isfinite(precision_matrix).all().item():
                raise ValueError("precision_matrix must contain only finite values")
            if not torch.allclose(
                precision_matrix,
                precision_matrix.transpose(0, 1),
                rtol=self.eigenvalue_tolerance,
                atol=self.eigenvalue_tolerance,
            ):
                raise ValueError("precision_matrix must be symmetric")
            eigenvalues, eigenvectors = torch.linalg.eigh(precision_matrix)
            if eigenvalues.min().item() < -self.eigenvalue_tolerance:
                raise ValueError("precision_matrix must be positive semidefinite")
            factor = eigenvectors @ torch.diag(eigenvalues.clamp_min(0.0).sqrt())

        if trainable:
            self.factor = nn.Parameter(factor)
        else:
            self.register_buffer("factor", factor)

    @property
    def precision_matrix(self) -> torch.Tensor:
        """Return the current positive-semidefinite precision matrix."""
        return self.factor @ self.factor.transpose(0, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("x and y must both have shape (B, T, F)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same batch size")
        if x.shape[-1] != self.num_features or y.shape[-1] != self.num_features:
            raise ValueError(f"x and y must have feature dimension {self.num_features}")
        if x.device != y.device or x.dtype != y.dtype:
            raise ValueError("x and y must have the same device and dtype")
        if not x.is_floating_point() or not y.is_floating_point():
            raise TypeError("x and y must be floating-point tensors")
        if x.device != self.factor.device or x.dtype != self.factor.dtype:
            raise ValueError("inputs and Mahalanobis factor must have the same device and dtype")

        distances = torch.cdist(x @ self.factor, y @ self.factor, p=2.0)
        return distances.square() if self.squared else distances

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, trainable={self.trainable}, "
            f"squared={self.squared}, eigenvalue_tolerance={self.eigenvalue_tolerance}"
        )
