"""Implementation of the GatedAttentionPooling object."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lambdaforge.nn.pooling.Pooling import Pooling


class GatedAttentionPooling(Pooling):
    r"""Gated attention pooling from Ilse et al. (2018).

    Extends :class:`AttentionPooling` with a sigmoid gate that modulates the
    tanh branch element-wise before computing the scalar attention score:

    .. math::

        a_i = \frac{
            \exp\!\bigl(\mathbf{w}^\top
                \bigl[\tanh(\mathbf{V}\, h_i^\top)
                \odot \sigma(\mathbf{U}\, h_i^\top)\bigr]\bigr)
        }{
            \sum_j \exp\!\bigl(\mathbf{w}^\top
                \bigl[\tanh(\mathbf{V}\, h_j^\top)
                \odot \sigma(\mathbf{U}\, h_j^\top)\bigr]\bigr)
        }

    .. math::

        z = \sum_i a_i \, h_i

    The gate :math:`\sigma(\mathbf{U} h_i^\top)` lets the network suppress
    entire projection dimensions, giving finer control over which aspects of
    each feature vector drive the attention score.

    The scores of masked-out positions are set to :math:`-\infty` before
    the softmax so that they receive zero weight.

    Reference
    ---------
    Ilse, M., Tomczak, J., & Welling, M. (2018).
    Attention-based Deep Multiple Instance Learning.
    *ICML 2018*. https://arxiv.org/abs/1802.04712

    Parameters
    ----------
    in_features : int
        Dimensionality of each input feature vector ``D``.
    hidden_features : int
        Internal projection dimensionality ``L`` shared by the tanh branch
        (``V``) and the sigmoid gate (``U``).
    name : str | None
        Optional name used to identify the pooling layer.

    Shape convention
    ----------------
    B:
        Batch size.

    N:
        Number of elements per sample.  For example, vertices, atoms, nodes,
        tokens, or neighbors in a local patch.

    D:
        Feature dimension per element.

    L:
        Attention hidden dimension (``hidden_features``).

    x:
        Shape ``(B, N, D)``.
    mask:
        Shape ``(B, N)``.  Float or bool — ``1`` / ``True`` = valid.
    output:
        Shape ``(B, D)``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 128,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self.V = nn.Linear(in_features, hidden_features, bias=False)
        self.U = nn.Linear(in_features, hidden_features, bias=False)
        self.w = nn.Linear(hidden_features, 1, bias=False)

    def attention_weights(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Return the normalised attention weights without applying them.

        Parameters
        ----------
        x    : ``(B, N, D)``
        mask : ``(B, N)`` — optional validity mask.

        Returns
        -------
        torch.Tensor
            Attention weights of shape ``(B, N)``.
        """
        gate = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))  # (B, N, L)
        A = self.w(gate).squeeze(-1)  # (B, N)

        if mask is not None:
            A = A.masked_fill(~mask.bool(), float("-inf"))

        return F.softmax(A, dim=1)  # (B, N)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        A = self.attention_weights(x, mask)  # (B, N)
        return torch.einsum("bn,bnd->bd", A, x)  # (B, D)
