"""Implementation of the CNN2D object."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch
import torch.nn as nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ReLU import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.Normalization import Normalization

ActivationSpec: TypeAlias = type[Activation] | str | list[type[Activation] | str] | None
NormalizationSpec: TypeAlias = type[Normalization] | str | list[type[Normalization] | str] | None
DropoutSpec: TypeAlias = float | list[float] | None
KwargsSpec: TypeAlias = dict[str, Any] | list[dict[str, Any]] | None


class CNN2D(Model):
    r"""2D Convolutional Neural Network.

    This class builds a stack of 2D convolutional layers followed by
    optional normalization, activation and dropout — the exact 2D analogue
    of :class:`MLP`.

    The final output layer is only a convolution. No normalization,
    activation or dropout is applied to the final output.

    Layer structure
    ---------------
    Hidden layers use:

        Conv2d -> Normalization -> Activation -> Dropout -> Optional residual

    The output layer uses:

        Conv2d

    Hidden channels
    ---------------
    ``hidden_channels`` controls the hidden architecture.

    If ``hidden_channels`` is ``None`` or ``0``, no hidden layers are
    created and the model becomes a single convolutional layer:

        in_channels -> out_channels

    If ``hidden_channels`` is an integer, it is interpreted as the number
    of hidden layers.  Their channel counts are interpolated between
    ``in_channels`` and ``out_channels``:

        hidden_channels=3

    creates three intermediate channel counts between input and output.

    If ``hidden_channels`` is a list of integers, it is interpreted
    directly as the hidden layer channel counts:

        hidden_channels=[64, 128, 64]

    Activation
    ----------
    ``activation`` must be an activation class, a list of activation
    classes, or ``None``.

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

    For per‑layer parameters, pass a list:

        activation_kwargs=[
            {"negative_slope": 0.1},
            {"negative_slope": 0.2},
        ]

    Normalization
    -------------
    ``normalization`` must be a normalization class, a list of
    normalization classes, or ``None``.

    Recommended usage:

        normalization=BatchNorm
        normalization_kwargs={"dim": 2}

    If a single class is passed, the same normalization type is used for
    all hidden layers, but each layer gets its own instance.

    If a list is passed, its length must match the number of hidden layers:

        normalization=[BatchNorm, LayerNorm, IdentityNorm]

    If ``normalization`` is ``None``, ``BatchNorm`` is used by default
    (the standard choice for 2D convolutions).

    Normalization classes are expected to follow this constructor
    convention:

        Normalization(features, **kwargs)

    where ``features`` is the output channel count of the current hidden
    layer.

    Dropout
    -------
    ``dropout`` can be a single float, a list of floats, or ``None``.

    If a single float is passed, the same dropout probability is used for
    all hidden layers.  Uses :class:`torch.nn.Dropout2d`.

    If a list is passed, its length must match the number of hidden layers.

    Residual connections
    --------------------
    If ``residual=True``, a residual connection is applied only when the
    input and output tensor shapes of a hidden layer match.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int | list[int] | None
        Hidden channel architecture definition.
    kernel_size : int | tuple[int, int]
        Spatial size of the convolution kernel for all layers.
    stride : int | tuple[int, int]
        Stride for all layers.
    padding : int | tuple[int, int] | None
        Padding.  If ``None``, defaults to ``kernel_size // 2`` ("same"
        padding for odd kernel sizes).
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
        Whether convolutional layers use bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | list[int] | None = None,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        activation: ActivationSpec = ReLU,
        normalization: NormalizationSpec = BatchNorm,
        dropout: DropoutSpec = 0.0,
        residual: bool = False,
        activation_kwargs: KwargsSpec = None,
        normalization_kwargs: KwargsSpec = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive.")
        if isinstance(hidden_channels, int) and hidden_channels < 0:
            raise ValueError("hidden_channels must be non-negative when given as a count.")

        resolved_padding: int | tuple[int, int]
        if padding is None:
            if isinstance(kernel_size, int):
                resolved_padding = kernel_size // 2
            else:
                resolved_padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            resolved_padding = padding

        if hidden_channels is None:
            hidden_channels = []
        if isinstance(hidden_channels, int):
            if hidden_channels == 0:
                hidden_channels = []
            else:
                hidden_channels = [
                    round(in_channels + (out_channels - in_channels) * i / (hidden_channels + 1))
                    for i in range(1, hidden_channels + 1)
                ]

        n_hidden = len(hidden_channels)
        if any(channels < 1 for channels in hidden_channels):
            raise ValueError("Every hidden channel count must be positive.")

        # --- Normalize specs to per‑layer lists -------------------------
        if activation is None:
            activation = [ReLU] * n_hidden
        elif isinstance(activation, type | str):
            activation = [activation] * n_hidden
        else:
            activation = list(activation)

        if normalization is None:
            normalization = [BatchNorm] * n_hidden
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

        channels = [in_channels] + hidden_channels + [out_channels]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(n_hidden):
            layer_in = channels[i]
            layer_out = channels[i + 1]

            self.convs.append(
                nn.Conv2d(
                    layer_in,
                    layer_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=resolved_padding,
                    bias=bias,
                )
            )

            normalization_cls = ComponentRegistry.resolve_normalization(normalization[i])
            activation_cls = ComponentRegistry.resolve_activation(activation[i])
            normalization_params = dict(normalization_kwargs[i])
            if normalization_cls is BatchNorm:
                normalization_params.setdefault("dim", 2)
            self.norms.append(normalization_cls(layer_out, **normalization_params))
            self.activations.append(activation_cls(**activation_kwargs[i]))
            self.dropouts.append(nn.Dropout2d(p=dropout[i]) if dropout[i] > 0 else nn.Identity())

        self.output = nn.Conv2d(
            channels[-2],
            channels[-1],
            kernel_size=kernel_size,
            stride=stride,
            padding=resolved_padding,
            bias=bias,
        )

        self.channels = channels
        self.use_residual = residual

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply the CNN2D.

        Shape convention
        ----------------
        B:
            Batch size.

        C_in:
            Number of input channels (``in_channels``).

        C_out:
            Number of output channels (``out_channels``).

        H, W:
            Spatial height and width of the input feature map.

        H', W':
            Spatial height and width of the output feature map.  Equal to
            ``H, W`` when ``stride=1`` and padding preserves spatial size
            (the default).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, C_out, H', W')``.
        """
        for i in range(len(self.convs)):
            identity = x

            x = self.convs[i](x)
            x = self.norms[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)

            if self.use_residual and x.shape == identity.shape:
                x = x + identity

        return self.output(x)
