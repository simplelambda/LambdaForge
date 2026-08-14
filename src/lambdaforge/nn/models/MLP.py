"""Implementation of the MLP object."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch
import torch.nn as nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.rectifiers import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization

ActivationSpec: TypeAlias = type[Activation] | str | list[type[Activation] | str] | None
NormalizationSpec: TypeAlias = type[Normalization] | str | list[type[Normalization] | str] | None
DropoutSpec: TypeAlias = float | list[float] | None
KwargsSpec: TypeAlias = dict[str, Any] | list[dict[str, Any]] | None


class MLP(Model):
    r"""Multi-Layer Perceptron.

    This class builds a fully-connected neural network using hidden linear
    layers followed by optional normalization, activation and dropout.

    The final output layer is only a linear layer. No normalization,
    activation or dropout is applied to the final output.

    Layer structure
    ---------------
    Hidden layers use:

        Linear -> Normalization -> Activation -> Dropout -> Optional residual

    The output layer uses:

        Linear

    Hidden sizes
    ------------
    ``hidden`` controls the hidden architecture.

    If ``hidden`` is ``None`` or ``0``, no hidden layers are created and the
    model becomes a single linear layer:

        in_features -> out_features

    If ``hidden`` is an integer, it is interpreted as the number of hidden
    layers. Their sizes are interpolated between ``in_features`` and
    ``out_features``:

        hidden=3

    creates three intermediate sizes between input and output.

    If ``hidden`` is a list of integers, it is interpreted directly as the
    hidden layer sizes:

        hidden=[128, 256, 128]

    Activation
    ----------
    ``activation`` must be an activation class, a list of activation classes,
    or ``None``.

    Recommended usage:

        activation=ReLU

    If a single class is passed, the same activation type is used for all
    hidden layers, but each layer gets its own instance.

    If a list is passed, its length must match the number of hidden layers:

        activation=[ReLU, GELU, ReLU]

    If ``activation`` is ``None``, ``ReLU`` is used by default.

    Activation parameters can be passed through ``activation_kwargs``:

        activation=LeakyReLU
        activation_kwargs={"negative_slope": 0.2}

    For per-layer parameters, pass a list:

        activation_kwargs=[
            {"negative_slope": 0.1},
            {"negative_slope": 0.2},
        ]

    Normalization
    -------------
    ``normalization`` must be a normalization class, a list of normalization
    classes, or ``None``.

    Recommended usage:

        normalization=LayerNorm

    If a single class is passed, the same normalization type is used for all
    hidden layers, but each layer gets its own instance.

    If a list is passed, its length must match the number of hidden layers:

        normalization=[LayerNorm, RMSNorm, IdentityNorm]

    If ``normalization`` is ``None``, ``IdentityNorm`` is used by default.

    Normalization classes are expected to follow this constructor convention:

        Normalization(features, **kwargs)

    where ``features`` is the output size of the current hidden layer.

    This allows the MLP to instantiate any custom normalization without
    hardcoding its type.

    Example:

        normalization=LayerNorm
        normalization_kwargs={"eps": 1e-6}

    For per-layer parameters, pass a list:

        normalization_kwargs=[
            {"eps": 1e-5},
            {"eps": 1e-6},
        ]

    Dropout
    -------
    ``dropout`` can be a single float, a list of floats, or ``None``.

    If a single float is passed, the same dropout probability is used for all
    hidden layers.

    If a list is passed, its length must match the number of hidden layers.

    Residual connections
    --------------------
    If ``residual=True``, a residual connection is applied only when the input
    and output tensor shapes of a hidden layer match.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    hidden : int | list[int] | None
        Hidden architecture definition.
    activation : type[Activation] | list[type[Activation]] | None
        Activation class or classes.
    normalization : type[Normalization] | list[type[Normalization]] | None
        Normalization class or classes.
    dropout : float | list[float] | None
        Dropout probability or probabilities.
    residual : bool
        Whether to use residual connections when shapes match.
    activation_kwargs : dict[str, Any] | list[dict[str, Any]] | None
        Keyword arguments used to instantiate activation modules.
    normalization_kwargs : dict[str, Any] | list[dict[str, Any]] | None
        Keyword arguments used to instantiate normalization modules.
    bias : bool
        Whether linear layers use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int | list[int] | None = None,
        activation: ActivationSpec = ReLU,
        normalization: NormalizationSpec = IdentityNorm,
        dropout: DropoutSpec = 0.0,
        residual: bool = False,
        activation_kwargs: KwargsSpec = None,
        normalization_kwargs: KwargsSpec = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive.")
        if isinstance(hidden, int) and hidden < 0:
            raise ValueError("hidden must be non-negative when given as a count.")

        if hidden is None:
            hidden = []
        if isinstance(hidden, int):
            if hidden == 0:
                hidden = []
            else:
                hidden = [
                    round(in_features + (out_features - in_features) * i / (hidden + 1))
                    for i in range(1, hidden + 1)
                ]

        n_hidden = len(hidden)
        if any(size < 1 for size in hidden):
            raise ValueError("Every hidden layer size must be positive.")

        if activation is None:
            activation = [ReLU] * n_hidden
        elif isinstance(activation, type | str):
            activation = [activation] * n_hidden
        else:
            activation = list(activation)

        if normalization is None:
            normalization = [IdentityNorm] * n_hidden
        elif isinstance(normalization, type | str):
            normalization = [normalization] * n_hidden
        else:
            normalization = list(normalization)

        if activation_kwargs is None:
            activation_kwargs = [{} for _ in range(n_hidden)]
        elif isinstance(activation_kwargs, dict):
            activation_kwargs = [activation_kwargs] * n_hidden
        else:
            activation_kwargs = list(activation_kwargs)

        if normalization_kwargs is None:
            normalization_kwargs = [{} for _ in range(n_hidden)]
        elif isinstance(normalization_kwargs, dict):
            normalization_kwargs = [normalization_kwargs] * n_hidden
        else:
            normalization_kwargs = list(normalization_kwargs)

        if dropout is None:
            dropout = [0.0] * n_hidden
        elif isinstance(dropout, int | float):
            dropout = [float(dropout)] * n_hidden
        else:
            dropout = list(dropout)

        per_layer_specs = (
            ("activation", activation),
            ("normalization", normalization),
            ("activation_kwargs", activation_kwargs),
            ("normalization_kwargs", normalization_kwargs),
            ("dropout", dropout),
        )
        for name, values in per_layer_specs:
            if len(values) != n_hidden:
                raise ValueError(f"{name} must contain exactly {n_hidden} values.")
        if any(not 0.0 <= probability < 1.0 for probability in dropout):
            raise ValueError("dropout probabilities must be in [0, 1).")

        sizes = [in_features] + hidden + [out_features]

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(n_hidden):
            layer_in = sizes[i]
            layer_out = sizes[i + 1]

            self.linears.append(nn.Linear(layer_in, layer_out, bias=bias))
            norm_cls = ComponentRegistry.resolve_normalization(normalization[i])
            activation_cls = ComponentRegistry.resolve_activation(activation[i])

            self.norms.append(norm_cls(layer_out, **normalization_kwargs[i]))
            self.activations.append(activation_cls(**activation_kwargs[i]))
            self.dropouts.append(nn.Dropout(p=dropout[i]) if dropout[i] > 0 else nn.Identity())

        self.output = nn.Linear(sizes[-2], sizes[-1], bias=bias)
        self.sizes = sizes
        self.use_residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply the MLP.

        Shape convention
        ----------------
        B:
            Batch size.

        \*:
            Any number of additional leading dimensions.

        F_in:
            Input feature dimension (``in_features``).

        F_out:
            Output feature dimension (``out_features``).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, F_in)`` or more generally
            ``(*, F_in)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, F_out)`` or more generally
            ``(*, F_out)``.
        """
        for i in range(len(self.linears)):
            identity = x

            x = self.linears[i](x)
            x = self.norms[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)

            if self.use_residual and x.shape == identity.shape:
                x = x + identity

        return self.output(x)
