"""Immutable result of one architecture numerical-parity check."""

from __future__ import annotations

from typing import Any

from lambdaforge.experiments.JsonResult import JsonResult


class ArchitectureConformanceResult(JsonResult):
    """Expose parity outcome, provenance and checksums as typed JSON data."""

    def __init__(
        self,
        *,
        name: str,
        passed: bool,
        source: str,
        parameter_count: int,
        expected_parameter_count: int,
        output_shape: tuple[int, ...],
        expected_output_shape: tuple[int, ...],
        state_checksum: str,
        output_checksum: str,
        maximum_absolute_error: float,
        maximum_relative_error: float,
        message: str = "",
    ) -> None:
        self.name = name
        self.passed = bool(passed)
        self.source = source
        self.parameter_count = int(parameter_count)
        self.expected_parameter_count = int(expected_parameter_count)
        self.output_shape = tuple(output_shape)
        self.expected_output_shape = tuple(expected_output_shape)
        self.state_checksum = state_checksum
        self.output_checksum = output_checksum
        self.maximum_absolute_error = float(maximum_absolute_error)
        self.maximum_relative_error = float(maximum_relative_error)
        self.message = message
        self._freeze_mapping(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable result mapping."""
        return {
            "name": self.name,
            "passed": self.passed,
            "source": self.source,
            "parameter_count": self.parameter_count,
            "expected_parameter_count": self.expected_parameter_count,
            "output_shape": list(self.output_shape),
            "expected_output_shape": list(self.expected_output_shape),
            "state_checksum": self.state_checksum,
            "output_checksum": self.output_checksum,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "message": self.message,
        }
