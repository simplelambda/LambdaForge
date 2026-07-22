"""Composable siamese neural network."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lambdaforge.nn.models.composition.SiameseMerge import SiameseMerge
from lambdaforge.nn.models.Model import Model


class SiameseModel(Model):
    """Encode a pair with shared weights and compare their embeddings.

    A custom ``comparator`` may replace every built-in merge and receives
    ``(left_embedding, right_embedding)``. The optional ``head`` is then
    applied to the comparison tensor. This keeps the ordinary forward result a
    tensor while :meth:`encode` and :meth:`compare_embeddings` expose the
    reusable stages.
    """

    def __init__(
        self,
        encoder: nn.Module,
        merge: SiameseMerge | str = SiameseMerge.ABSOLUTE_DIFFERENCE,
        comparator: nn.Module | None = None,
        head: nn.Module | None = None,
        merge_dimension: int = -1,
        keep_distance_dimension: bool = True,
        normalize_embeddings: bool = False,
        normalization_p: float = 2.0,
        normalization_dimension: int = -1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module.")
        for name, module in (("comparator", comparator), ("head", head)):
            if module is not None and not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be a torch.nn.Module or None.")
        for name, value in (
            ("merge_dimension", merge_dimension),
            ("normalization_dimension", normalization_dimension),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if (
            isinstance(normalization_p, bool)
            or not math.isfinite(float(normalization_p))
            or float(normalization_p) <= 0
        ):
            raise ValueError("normalization_p must be a finite positive number.")
        if isinstance(eps, bool) or not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError("eps must be a finite positive number.")

        self.encoder = encoder
        self.merge = SiameseMerge.from_value(merge)
        self.comparator = comparator
        self.head = head if head is not None else nn.Identity()
        self.merge_dimension = merge_dimension
        self.keep_distance_dimension = keep_distance_dimension
        self.normalize_embeddings = normalize_embeddings
        self.normalization_p = float(normalization_p)
        self.normalization_dimension = normalization_dimension
        self.eps = float(eps)

    def encode(self, x: Tensor) -> Tensor:
        """Return one normalized or raw shared-encoder embedding."""
        embedding = self.encoder(x)
        if not isinstance(embedding, Tensor):
            raise TypeError("encoder must return a Tensor.")
        if self.normalize_embeddings:
            embedding = F.normalize(
                embedding,
                p=self.normalization_p,
                dim=self.normalization_dimension,
                eps=self.eps,
            )
        return embedding

    def compare_embeddings(self, left: Tensor, right: Tensor) -> Tensor:
        """Combine two already encoded tensors using the configured operation."""
        if self.comparator is not None:
            compared = self.comparator(left, right)
            if not isinstance(compared, Tensor):
                raise TypeError("comparator must return a Tensor.")
            return compared
        if left.shape != right.shape:
            raise ValueError("Built-in siamese merges require embeddings with identical shapes.")
        if self.merge is SiameseMerge.ABSOLUTE_DIFFERENCE:
            return (left - right).abs()
        if self.merge is SiameseMerge.DIFFERENCE:
            return left - right
        if self.merge is SiameseMerge.PRODUCT:
            return left * right
        if self.merge is SiameseMerge.CONCATENATE:
            return torch.cat([left, right], dim=self.merge_dimension)
        if self.merge is SiameseMerge.L1_DISTANCE:
            return torch.linalg.vector_norm(
                left - right,
                ord=1,
                dim=self.merge_dimension,
                keepdim=self.keep_distance_dimension,
            )
        if self.merge is SiameseMerge.L2_DISTANCE:
            return torch.linalg.vector_norm(
                left - right,
                ord=2,
                dim=self.merge_dimension,
                keepdim=self.keep_distance_dimension,
            )
        similarity = F.cosine_similarity(
            left,
            right,
            dim=self.merge_dimension,
            eps=self.eps,
        )
        if self.keep_distance_dimension:
            similarity = similarity.unsqueeze(self.merge_dimension)
        return similarity

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Encode, compare and transform a pair of inputs."""
        compared = self.compare_embeddings(self.encode(left), self.encode(right))
        output = self.head(compared)
        if not isinstance(output, Tensor):
            raise TypeError("head must return a Tensor.")
        return output
