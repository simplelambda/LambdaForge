"""Implementation of the MomentPooling object."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class MomentPooling(Pooling):
    r"""Second-order moment pooling over the set dimension.

    Computes the mean and the unbiased standard deviation across the set and
    concatenates them into a single representation:

    .. math::

        \mu_d = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} h_{i,d}

    .. math::

        \sigma_d = \sqrt{
            \frac{1}{|\mathcal{V}| - 1}
            \sum_{i \in \mathcal{V}} (h_{i,d} - \mu_d)^2 + \varepsilon
        }

    .. math::

        z = [\,\mu \,\Vert\, \sigma\,]

    Capturing variance alongside the mean encodes both the typical value and
    the spread of features across the set.  This is particularly informative
    when set elements sample a continuous underlying signal (e.g. features
    defined at every point of a local neighborhood) and the diversity across
    the set carries discriminative information.

    When a sample has only one valid element the standard deviation is set
    to zero (unbiased estimate is undefined).

    Parameters
    ----------
    eps : float
        Small constant added inside the square root for numerical stability.
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

    x:
        Shape ``(B, N, D)``.
    mask:
        Shape ``(B, N)``.  Float or bool — ``1`` / ``True`` = valid.
    output:
        Shape ``(B, 2 * D)``.  First ``D`` channels are the mean; last ``D``
        channels are the standard deviation.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            m = mask.to(dtype=x.dtype)  # (B, N)
            count = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
            x_m = x * m.unsqueeze(-1)  # (B, N, D)
            mean = x_m.sum(dim=1) / count  # (B, D)

            # Unbiased std: divide by max(count - 1, 1)
            denom = (count - 1).clamp_min(1.0)
            diff = (x - mean.unsqueeze(1)) * m.unsqueeze(-1)  # (B, N, D)
            var = (diff**2).sum(dim=1) / denom  # (B, D)
        else:
            mean = x.mean(dim=1)  # (B, D)
            var = x.var(dim=1, unbiased=True)  # (B, D)

        std = torch.sqrt(var + self.eps)  # (B, D)

        return torch.cat([mean, std], dim=-1)  # (B, 2D)
