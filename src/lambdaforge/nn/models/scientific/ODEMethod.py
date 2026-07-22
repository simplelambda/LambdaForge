"""Supported dependency-free fixed-step ODE integration methods."""

from enum import Enum


class ODEMethod(str, Enum):
    """Select the accuracy/cost tradeoff of native ODE integration."""

    EULER = "euler"
    MIDPOINT = "midpoint"
    RK4 = "rk4"
