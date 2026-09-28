from __future__ import annotations

from lambdaforge.hpo.ParameterSpace import ParameterDescriptor
from lambdaforge.scientific_format import format_parameter_value


def test_log_parameter_is_compact_without_mutating_machine_value() -> None:
    value = 0.00001422634036982495
    descriptor = ParameterDescriptor(
        "learning_rate", "continuous", low=1e-6, high=1e-2, scale="log"
    )

    rendered = format_parameter_value(value, descriptor)

    assert rendered == "1.423e-5"
    assert value == 0.00001422634036982495


def test_discrete_parameter_remains_exact() -> None:
    descriptor = ParameterDescriptor("width", "integer", low=32, high=256)

    assert format_parameter_value(128, descriptor) == "128"
