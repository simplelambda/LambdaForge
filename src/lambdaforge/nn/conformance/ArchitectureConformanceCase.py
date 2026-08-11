"""One deterministic architecture initialization and numerical reference case."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn

from lambdaforge.nn.conformance.ArchitectureConformanceResult import (
    ArchitectureConformanceResult,
)


class ArchitectureConformanceCase:
    """Rebuild a tiny model and compare it with a provenance-linked reference."""

    REFERENCE_VERSION = 1

    def __init__(
        self,
        *,
        name: str,
        model_factory: Callable[[], nn.Module],
        inputs: Sequence[Any],
        expected_output: torch.Tensor,
        expected_state: Mapping[str, torch.Tensor],
        source: str,
        expected_parameter_count: int,
        keyword_inputs: Mapping[str, Any] | None = None,
        output_key: str | None = None,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-5,
    ) -> None:
        if not name or not source:
            raise ValueError("name and source provenance must be non-empty.")
        if not callable(model_factory):
            raise TypeError("model_factory must be callable.")
        if not torch.is_tensor(expected_output):
            raise TypeError("expected_output must be a tensor.")
        if expected_parameter_count < 0:
            raise ValueError("expected_parameter_count must be non-negative.")
        if absolute_tolerance < 0 or relative_tolerance < 0:
            raise ValueError("Conformance tolerances must be non-negative.")
        self.name = name
        self.model_factory = model_factory
        self.inputs = tuple(inputs)
        self.keyword_inputs = dict(keyword_inputs or {})
        self.expected_output = expected_output.detach().cpu()
        self.expected_state = {key: value.detach().cpu() for key, value in expected_state.items()}
        self.source = source
        self.expected_parameter_count = expected_parameter_count
        self.output_key = output_key
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)

    @classmethod
    def capture(
        cls,
        *,
        name: str,
        model_factory: Callable[[], nn.Module],
        inputs: Sequence[Any],
        source: str,
        seed: int = 0,
        keyword_inputs: Mapping[str, Any] | None = None,
        output_key: str | None = None,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-5,
    ) -> ArchitectureConformanceCase:
        """Capture deterministic initialization and output for a tiny fixture."""
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = model_factory()
            if not isinstance(model, nn.Module):
                raise TypeError("model_factory must return a torch.nn.Module.")
            model.eval()
            with torch.no_grad():
                output = cls._output(
                    model(*tuple(inputs), **dict(keyword_inputs or {})), output_key
                )
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
        return cls(
            name=name,
            model_factory=model_factory,
            inputs=inputs,
            keyword_inputs=keyword_inputs,
            expected_output=output,
            expected_state=state,
            source=source,
            expected_parameter_count=parameter_count,
            output_key=output_key,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

    def run(self, device: torch.device | str = "cpu") -> ArchitectureConformanceResult:
        """Load the reference state and evaluate shape/count/numerical parity."""
        model = self.model_factory()
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return a torch.nn.Module.")
        model.load_state_dict(self.expected_state, strict=True)
        model.to(device).eval()
        args = tuple(self._move(value, device) for value in self.inputs)
        kwargs = {key: self._move(value, device) for key, value in self.keyword_inputs.items()}
        with torch.no_grad():
            output = self._output(model(*args, **kwargs), self.output_key)
        expected = self.expected_output.to(device=device, dtype=output.dtype)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        same_shape = output.shape == expected.shape
        if same_shape and output.numel():
            absolute = (output - expected).abs()
            relative = absolute / expected.abs().clamp_min(1e-12)
            maximum_absolute_error = float(absolute.max().cpu())
            maximum_relative_error = float(relative.max().cpu())
        elif same_shape:
            maximum_absolute_error = 0.0
            maximum_relative_error = 0.0
        else:
            maximum_absolute_error = float("inf")
            maximum_relative_error = float("inf")
        close = same_shape and torch.allclose(
            output,
            expected,
            atol=self.absolute_tolerance,
            rtol=self.relative_tolerance,
        )
        count_matches = parameter_count == self.expected_parameter_count
        passed = bool(close and count_matches)
        problems: list[str] = []
        if not same_shape:
            problems.append("output shape differs")
        elif not close:
            problems.append("output values exceed tolerance")
        if not count_matches:
            problems.append("parameter count differs")
        return ArchitectureConformanceResult(
            name=self.name,
            passed=passed,
            source=self.source,
            parameter_count=parameter_count,
            expected_parameter_count=self.expected_parameter_count,
            output_shape=tuple(output.shape),
            expected_output_shape=tuple(expected.shape),
            state_checksum=self._checksum(self.expected_state.values()),
            output_checksum=self._checksum((output,)),
            maximum_absolute_error=maximum_absolute_error,
            maximum_relative_error=maximum_relative_error,
            message="; ".join(problems),
        )

    def write_reference(self, path: str | Path) -> Path:
        """Write a tiny trusted tensor checkpoint with explicit provenance."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            torch.save(
                {
                    "reference_version": self.REFERENCE_VERSION,
                    "name": self.name,
                    "source": self.source,
                    "expected_parameter_count": self.expected_parameter_count,
                    "expected_output": self.expected_output,
                    "expected_state": self.expected_state,
                    "output_key": self.output_key,
                    "absolute_tolerance": self.absolute_tolerance,
                    "relative_tolerance": self.relative_tolerance,
                },
                temporary,
            )
            # Windows does not permit ``fsync`` on a read-only CRT descriptor.
            # Reopen read/write while preserving the already serialized bytes.
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def from_reference(
        cls,
        path: str | Path,
        *,
        model_factory: Callable[[], nn.Module],
        inputs: Sequence[Any],
        keyword_inputs: Mapping[str, Any] | None = None,
    ) -> ArchitectureConformanceCase:
        """Load a trusted weights-only reference without importing arbitrary code."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("reference_version") != cls.REFERENCE_VERSION
        ):
            raise ValueError("Unsupported architecture reference checkpoint.")
        state = payload.get("expected_state")
        output = payload.get("expected_output")
        if not isinstance(state, dict) or not torch.is_tensor(output):
            raise TypeError("Architecture reference has invalid tensor payloads.")
        return cls(
            name=str(payload["name"]),
            model_factory=model_factory,
            inputs=inputs,
            keyword_inputs=keyword_inputs,
            expected_output=output,
            expected_state=state,
            source=str(payload["source"]),
            expected_parameter_count=int(payload["expected_parameter_count"]),
            output_key=payload.get("output_key"),
            absolute_tolerance=float(payload["absolute_tolerance"]),
            relative_tolerance=float(payload["relative_tolerance"]),
        )

    @staticmethod
    def _output(result: Any, output_key: str | None) -> torch.Tensor:
        if isinstance(result, Mapping):
            key = output_key or "prediction"
            if key not in result:
                raise KeyError(f"Model output has no {key!r} conformance tensor.")
            result = result[key]
        if not torch.is_tensor(result):
            raise TypeError("Conformance models must expose a tensor output.")
        return result

    @classmethod
    def _move(cls, value: Any, device: torch.device | str) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: cls._move(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._move(item, device) for item in value)
        if isinstance(value, list):
            return [cls._move(item, device) for item in value]
        return value

    @staticmethod
    def _checksum(tensors: Sequence[torch.Tensor] | Any) -> str:
        digest = hashlib.sha256()
        for tensor in tensors:
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return f"sha256:{digest.hexdigest()}"
