"""Contract for hierarchical two-dimensional feature extractors."""

from __future__ import annotations

from abc import abstractmethod

import torch

from lambdaforge.nn.models.Model import Model


class HierarchicalBackbone2D(Model):
    """Base class for image encoders that expose ordered feature maps.

    Implementations publish one channel count per stage through
    ``feature_channels`` and return the corresponding NCHW tensors from fine
    to coarse resolution.  Dense-prediction heads can therefore depend on a
    small stable contract instead of knowing a concrete backbone class.
    """

    feature_channels: tuple[int, ...]

    @abstractmethod
    def forward_feature_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return one NCHW tensor per stage, ordered from fine to coarse."""
        raise NotImplementedError

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final pooled representation before any prediction head."""
        raise NotImplementedError
