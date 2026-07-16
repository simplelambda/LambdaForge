"""Implementation of the Identity object."""

from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class Identity(Activation):
    """Identity activation (passthrough / no-op).

    Formula:
        f(x) = x

    Leaves the input unchanged. Useful as a placeholder when an
    activation slot is required but no transformation is desired
    (e.g. final layer of a regressor, or when the model already
    includes its own activation).

    Parameters
    ----------
    name : str | None
        Optional name to identify this activation instance.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x
