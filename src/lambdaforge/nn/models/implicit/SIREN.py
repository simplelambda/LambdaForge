"""Sinusoidal Representation Network for implicit neural representations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from lambdaforge.nn.models.Model import Model


class SIREN(Model):
    r"""Coordinate network with periodic activations and SIREN initialization.

    An integer ``hidden`` creates that many layers of ``hidden_features``
    neurons. A sequence supplies every hidden width directly. The first layer
    is initialized uniformly in ``[-1/fan_in, 1/fan_in]``; later sine layers
    use ``[-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega]``. These defaults are
    the principled initialization from the SIREN paper, while explicit bounds
    make every layer overrideable for controlled experiments.

    Parameters
    ----------
    first_omega:
        Frequency multiplier of the first hidden sine layer.
    hidden_omega:
        Scalar or one value per hidden layer after the first.
    output_omega:
        Frequency used only when ``outermost_linear=False``.
    first_weight_bound, hidden_weight_bounds, output_weight_bound:
        Optional absolute uniform bounds replacing the paper defaults.
    bias_bounds:
        Optional scalar or one bound per linear layer. ``None`` preserves
        PyTorch's standard bias initialization, matching the reference code.
    output_transform:
        Optional injected module applied after the linear or sine output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int | Sequence[int] = 3,
        hidden_features: int = 256,
        first_omega: float = 30.0,
        hidden_omega: float | Sequence[float] = 30.0,
        output_omega: float = 30.0,
        outermost_linear: bool = True,
        bias: bool | Sequence[bool] = True,
        first_weight_bound: float | None = None,
        hidden_weight_bounds: float | Sequence[float] | None = None,
        output_weight_bound: float | None = None,
        bias_bounds: float | Sequence[float] | None = None,
        output_transform: nn.Module | None = None,
    ) -> None:
        super().__init__()
        for dimension_name, dimension_value in (
            ("in_features", in_features),
            ("out_features", out_features),
        ):
            if (
                isinstance(dimension_value, bool)
                or not isinstance(dimension_value, int)
                or dimension_value < 1
            ):
                raise ValueError(f"{dimension_name} must be a positive integer.")
        if (
            isinstance(hidden_features, bool)
            or not isinstance(hidden_features, int)
            or hidden_features < 1
        ):
            raise ValueError("hidden_features must be a positive integer.")
        if isinstance(hidden, bool):
            raise TypeError("hidden must be a positive integer or a sequence of widths.")
        if isinstance(hidden, int):
            if hidden < 1:
                raise ValueError("hidden must create at least one sine layer.")
            hidden_sizes = [hidden_features] * hidden
        elif isinstance(hidden, Sequence) and not isinstance(hidden, str | bytes):
            hidden_sizes = list(hidden)
            if not hidden_sizes or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 1
                for size in hidden_sizes
            ):
                raise ValueError("Every hidden layer width must be a positive integer.")
        else:
            raise TypeError("hidden must be a positive integer or a sequence of widths.")
        for omega_name, omega_value in (
            ("first_omega", first_omega),
            ("output_omega", output_omega),
        ):
            self._validate_positive(omega_name, omega_value)

        later_count = len(hidden_sizes) - 1
        hidden_omegas = self._expand_values(
            hidden_omega,
            later_count,
            "hidden_omega",
        )
        hidden_bounds = self._expand_optional_values(
            hidden_weight_bounds,
            later_count,
            "hidden_weight_bounds",
        )
        bias_flags = self._expand_bools(bias, len(hidden_sizes) + 1, "bias")
        expanded_bias_bounds = self._expand_optional_values(
            bias_bounds,
            len(hidden_sizes) + 1,
            "bias_bounds",
        )
        for bound_name, bound_value in (
            ("first_weight_bound", first_weight_bound),
            ("output_weight_bound", output_weight_bound),
        ):
            if bound_value is not None:
                self._validate_non_negative(bound_name, bound_value)
        if output_transform is not None and not isinstance(output_transform, nn.Module):
            raise TypeError("output_transform must be a torch.nn.Module or None.")

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.first_omega = float(first_omega)
        self.hidden_omegas = [float(value) for value in hidden_omegas]
        self.output_omega = float(output_omega)
        self.outermost_linear = outermost_linear
        self.output_transform = output_transform if output_transform is not None else nn.Identity()

        sizes = [in_features, *hidden_sizes, out_features]
        self.linears = nn.ModuleList(
            nn.Linear(sizes[index], sizes[index + 1], bias=bias_flags[index])
            for index in range(len(sizes) - 1)
        )

        self.first_weight_bound = (
            1.0 / in_features if first_weight_bound is None else float(first_weight_bound)
        )
        resolved_hidden_bounds: list[float] = []
        for index, configured in enumerate(hidden_bounds, start=1):
            default = math.sqrt(6.0 / sizes[index]) / self.hidden_omegas[index - 1]
            resolved_hidden_bounds.append(default if configured is None else float(configured))
        self.hidden_weight_bounds = resolved_hidden_bounds

        if outermost_linear:
            initialization_omega = (
                self.hidden_omegas[-1] if self.hidden_omegas else self.first_omega
            )
        else:
            initialization_omega = self.output_omega
        default_output_bound = math.sqrt(6.0 / hidden_sizes[-1]) / initialization_omega
        self.output_weight_bound = (
            default_output_bound if output_weight_bound is None else float(output_weight_bound)
        )
        self.bias_bounds = expanded_bias_bounds
        self.reset_parameters()

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        """Reapply the configured SIREN initialization in-place."""
        with torch.no_grad():
            first_linear = cast(nn.Linear, self.linears[0])
            first_linear.weight.uniform_(
                -self.first_weight_bound,
                self.first_weight_bound,
                generator=generator,
            )
            for linear, bound in zip(
                self.linears[1:-1],
                self.hidden_weight_bounds,
                strict=True,
            ):
                cast(nn.Linear, linear).weight.uniform_(-bound, bound, generator=generator)
            output_linear = cast(nn.Linear, self.linears[-1])
            output_linear.weight.uniform_(
                -self.output_weight_bound,
                self.output_weight_bound,
                generator=generator,
            )
            for module, bias_bound in zip(self.linears, self.bias_bounds, strict=True):
                linear = cast(nn.Linear, module)
                if linear.bias is not None and bias_bound is not None:
                    linear.bias.uniform_(-bias_bound, bias_bound, generator=generator)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the continuous coordinate representation."""
        self._validate_input(x)
        value = x
        for index, linear in enumerate(self.linears[:-1]):
            omega = self.first_omega if index == 0 else self.hidden_omegas[index - 1]
            value = torch.sin(omega * linear(value))
        value = self.linears[-1](value)
        if not self.outermost_linear:
            value = torch.sin(self.output_omega * value)
        transformed = self.output_transform(value)
        if not isinstance(transformed, Tensor):
            raise TypeError("output_transform must return a Tensor.")
        return transformed

    def activations(self, x: Tensor) -> tuple[Tensor, ...]:
        """Return hidden sine activations followed by the final output."""
        self._validate_input(x)
        values: list[Tensor] = []
        value = x
        for index, linear in enumerate(self.linears[:-1]):
            omega = self.first_omega if index == 0 else self.hidden_omegas[index - 1]
            value = torch.sin(omega * linear(value))
            values.append(value)
        value = self.linears[-1](value)
        if not self.outermost_linear:
            value = torch.sin(self.output_omega * value)
        transformed = self.output_transform(value)
        if not isinstance(transformed, Tensor):
            raise TypeError("output_transform must return a Tensor.")
        values.append(transformed)
        return tuple(values)

    def _validate_input(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError("SIREN input must be a Tensor.")
        if not torch.is_floating_point(x):
            raise TypeError("SIREN requires floating-point inputs.")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input shape (..., {self.in_features}); got {tuple(x.shape)}."
            )

    @staticmethod
    def _expand_values(
        value: float | Sequence[float],
        count: int,
        name: str,
    ) -> list[float]:
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            values = list(value)
        else:
            values = [float(value)] * count
        if len(values) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        for item in values:
            SIREN._validate_positive(name, item)
        return [float(item) for item in values]

    @staticmethod
    def _expand_optional_values(
        value: float | Sequence[float] | None,
        count: int,
        name: str,
    ) -> list[float | None]:
        if value is None:
            return [None] * count
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            values: list[float | None] = list(value)
        else:
            values = [float(value)] * count
        if len(values) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        for item in values:
            if item is not None:
                SIREN._validate_non_negative(name, item)
        return values

    @staticmethod
    def _expand_bools(value: bool | Sequence[bool], count: int, name: str) -> list[bool]:
        if isinstance(value, Sequence):
            values = list(value)
        else:
            values = [value] * count
        if len(values) != count or any(not isinstance(item, bool) for item in values):
            raise ValueError(f"{name} must be a bool or contain exactly {count} booleans.")
        return values

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be a finite positive number.")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a finite non-negative number.")
