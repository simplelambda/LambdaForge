"""Geometric vector perceptron for arbitrary scalar and polar-vector channels."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.validation import aligned, channels, positive


class GVP(nn.Module):
    """Map (s[N,S], V[N,C,3]) to scalars and E(3)-equivariant polar vectors.

    Channel mixing has no Cartesian bias. Original norm-dependent vector activation
    and the explicitly selected scalar-gated variant are distinguished by vector_gate.
    This is a primitive, not a protein featurizer or chirality detector.
    """

    def __init__(
        self,
        scalar_in: int,
        vector_in: int,
        scalar_out: int,
        vector_out: int,
        hidden_vectors: int | None = None,
        vector_gate: bool = False,
        activate: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        channels(scalar_in=scalar_in, scalar_out=scalar_out)
        for name, value in (("vector_in", vector_in), ("vector_out", vector_out)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        hidden = max(vector_in, vector_out, 1) if hidden_vectors is None else hidden_vectors
        channels(hidden_vectors=hidden)
        if not isinstance(vector_gate, bool) or not isinstance(activate, bool):
            raise TypeError("vector_gate and activate must be Boolean.")
        self.scalar_in, self.vector_in, self.vector_out = scalar_in, vector_in, vector_out
        self.epsilon = positive(epsilon, "epsilon")
        self.vector_gate, self.activate = vector_gate, activate
        self.vector_hidden = nn.Linear(vector_in, hidden, bias=False) if vector_in else None
        self.scalar = nn.Linear(scalar_in + (hidden if vector_in else 0), scalar_out)
        self.vector = (
            nn.Linear(hidden, vector_out, bias=False) if vector_in and vector_out else None
        )
        self.gate = (
            nn.Linear(scalar_out, vector_out) if vector_gate and vector_out and vector_in else None
        )

    def forward(
        self, scalars: torch.Tensor, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a stable tuple even when vector_out=0; vector dimensions are channels,3."""
        if (
            scalars.ndim != 2
            or scalars.shape[1] != self.scalar_in
            or not scalars.is_floating_point()
        ):
            raise ValueError("Malformed scalar states.")
        vectors = aligned(vectors, (len(scalars), self.vector_in, 3), scalars, "vectors")
        hidden = (
            self.vector_hidden(vectors.transpose(-1, -2))
            if self.vector_hidden is not None
            else None
        )
        scalar_input = (
            scalars
            if hidden is None
            else torch.cat((scalars, hidden.square().sum(-2).clamp_min(self.epsilon).sqrt()), -1)
        )
        preactivation = self.scalar(scalar_input)
        output = scalars.new_zeros((len(scalars), self.vector_out, 3))
        if self.vector is not None:
            output = self.vector(hidden).transpose(-1, -2)
            if self.gate is not None:
                output = output * self.gate(preactivation).sigmoid().unsqueeze(-1)
            elif self.activate:
                output = (
                    output
                    * output.square().sum(-1, keepdim=True).clamp_min(self.epsilon).sqrt().sigmoid()
                )
        return (preactivation.relu() if self.activate else preactivation), output
