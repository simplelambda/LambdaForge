"""Bessel/real m=0 spherical features for directional interactions."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import nn

from lambdaforge.nn.models.graph.geometric.GraphTriplets import GraphTriplets
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.validation import channels, positive


@lru_cache(maxsize=32)
def _constants(orders: int, radial: int) -> tuple[list[list[float]], list[list[float]]]:
    try:
        from scipy.optimize import brentq
        from scipy.special import spherical_jn
    except ImportError as error:
        raise ImportError(
            "DimeNetBasis construction requires lambdaforge[graph-directional]."
        ) from error
    roots, normalizers = [], []
    for order in range(orders):
        found = []
        left = 1e-6
        for index in range(1, 8 * (orders + radial + 4)):
            right = index * math.pi / 4
            if spherical_jn(order, left) * spherical_jn(order, right) < 0:
                root = float(brentq(lambda x, degree=order: spherical_jn(degree, x), left, right))
                found.append(root)
                if len(found) == radial:
                    break
            left = right
        if len(found) != radial:
            raise RuntimeError("Could not bracket the requested spherical Bessel roots.")
        roots.append(found)
        normalizers.append([math.sqrt(2) / abs(float(spherical_jn(order + 1, r))) for r in found])
    return roots, normalizers


class DimeNetBasis(nn.Module):
    """Trainable sine radial basis and normalized spherical Bessel x real Y_l0.

    Distances are explicit (E,), angles (T,), and incoming edge IDs select spherical
    radial rows. The polynomial envelope is the DimeNet convention (exponent+1).
    SciPy is required only at construction for root constants; forward is pure Torch
    and differentiable. Spherical recurrence is evaluated in float64 for stability
    then cast back. No coordinates, neighbourhoods or chemical labels are inferred.
    """

    def __init__(
        self,
        num_radial: int = 6,
        num_spherical: int = 7,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ) -> None:
        super().__init__()
        channels(
            num_radial=num_radial, num_spherical=num_spherical, envelope_exponent=envelope_exponent
        )
        if num_radial > 64 or num_spherical > 8:
            raise ValueError("Validated basis limits are num_radial<=64 and num_spherical<=8.")
        self.num_radial, self.num_spherical = num_radial, num_spherical
        self.cutoff = positive(cutoff, "cutoff")
        self.exponent = envelope_exponent + 1
        roots, norms = _constants(num_spherical, num_radial)
        self.register_buffer("roots", torch.tensor(roots, dtype=torch.float64))
        self.register_buffer("normalizers", torch.tensor(norms, dtype=torch.float64))
        self.frequencies = nn.Parameter(
            torch.arange(1, num_radial + 1, dtype=torch.get_default_dtype()) * math.pi
        )

    def _envelope(self, x: torch.Tensor) -> torch.Tensor:
        p = self.exponent
        inside = x < 1
        # Never evaluate large outside-support powers, even in a discarded where
        # branch: their backward can otherwise produce 0*inf -> NaN.
        x = torch.where(inside, x, torch.ones_like(x))
        return torch.where(
            inside,
            1 / x
            - (p + 1) * (p + 2) / 2 * x.pow(p - 1)
            + p * (p + 2) * x.pow(p)
            - p * (p + 1) / 2 * x.pow(p + 1),
            torch.zeros_like(x),
        )

    @staticmethod
    def _bessel(order: int, z: torch.Tensor) -> torch.Tensor:
        # Small-argument power series avoids cancellation in upward recurrence.
        small = z <= order + 1
        stable = torch.where(small, torch.ones_like(z) * (order + 2), z)
        j0 = torch.sin(stable) / stable
        if order == 0:
            recurrence = j0
        else:
            j1 = torch.sin(stable) / stable.square() - torch.cos(stable) / stable
            for degree in range(1, order):
                j0, j1 = j1, (2 * degree + 1) / stable * j1 - j0
            recurrence = j1
        series_z = torch.where(small, z, torch.zeros_like(z))
        denominator = math.prod(range(1, 2 * order + 2, 2))
        term = series_z.pow(order) / denominator
        series = term
        for k in range(1, 40):
            term = term * (-series_z.square()) / (2 * k * (2 * order + 2 * k + 1))
            series = series + term
        return torch.where(small, series, recurrence)

    def forward(
        self,
        distances: torch.Tensor,
        angles: torch.Tensor,
        triplets: GraphTriplets,
        *,
        distance_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return radial (E,R) and angular-radial (T,L*R) features."""
        if (
            distances.ndim != 1
            or not distances.is_floating_point()
            or not torch.isfinite(distances).all()
            or (distances <= 0).any()
        ):
            raise ValueError("DimeNet distances must be finite strictly positive (E,).")
        incoming, _ = triplets.validate(len(distances), distances.device)
        if (
            angles.shape != incoming.shape
            or not angles.is_floating_point()
            or not torch.isfinite(angles).all()
            or (angles < 0).any()
            or (angles > math.pi).any()
        ):
            raise ValueError("angles must be finite (T,) in [0,pi].")
        x = distances / self.cutoff
        supported_x = torch.where(x < 1, x, torch.ones_like(x))
        radial = self._envelope(x)[:, None] * torch.sin(supported_x[:, None] * self.frequencies)
        weights = GraphEdgeData.normalize_message_weights(
            distance_weights,
            edge_count=len(distances),
            reference=distances,
            name="distance_weights",
            allow_scalar=True,
        )
        if weights is not None:
            # A constant-one radial route leaves angle/message/readout processing
            # meaningful; zero would disable them all through bias-free filters.
            radial = (1 - weights[:, None]) + weights[:, None] * radial
        cosine = angles.to(device=distances.device, dtype=torch.float64).cos()
        legendre = [torch.ones_like(cosine)]
        if self.num_spherical > 1:
            legendre.append(cosine)
        for degree in range(1, self.num_spherical - 1):
            legendre.append(
                ((2 * degree + 1) * cosine * legendre[-1] - degree * legendre[-2]) / (degree + 1)
            )
        features = []
        for degree in range(self.num_spherical):
            z = supported_x.double()[:, None] * self.roots[degree].double()
            bessel = self._bessel(degree, z) * self.normalizers[degree].double()
            bessel = bessel * self._envelope(x.double())[:, None]
            if weights is not None:
                bessel = (1 - weights.double()[:, None]) + weights.double()[:, None] * bessel
            harmonic = math.sqrt((2 * degree + 1) / (4 * math.pi)) * legendre[degree]
            features.append(bessel[incoming] * harmonic[:, None])
        return radial, torch.cat(features, -1).to(distances)
