"""Generic NPZ/NPY contents validator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from lambdaforge.artifacts.ArtifactValidationResult import ArtifactValidationResult
from lambdaforge.artifacts.ArtifactValidator import ArtifactValidator


class NumpyArtifactValidator(ArtifactValidator):
    """Require named arrays, optional shapes and finite numeric values."""

    def __init__(
        self,
        *,
        required_arrays: Sequence[str] = (),
        shapes: Mapping[str, Sequence[int | None]] | None = None,
        finite: bool = False,
        max_elements: int = 5_000_000,
    ) -> None:
        self.required_arrays = tuple(str(value) for value in required_arrays)
        self.shapes = {str(key): tuple(value) for key, value in (shapes or {}).items()}
        self.finite = bool(finite)
        self.max_elements = int(max_elements)

    def validate(self, path: Path) -> ArtifactValidationResult:
        """Validate with pickle disabled and bounded finite scans."""
        errors: list[str] = []
        warnings: list[str] = []
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            return ArtifactValidationResult(False, ("Artifact is missing, symbolic or empty.",))
        try:
            loaded = np.load(
                path, allow_pickle=False, mmap_mode="r" if path.suffix == ".npy" else None
            )
            arrays = {path.stem: loaded} if isinstance(loaded, np.ndarray) else loaded
            names = tuple(arrays.files) if hasattr(arrays, "files") else tuple(arrays)
            for required in self.required_arrays:
                if required not in names:
                    errors.append(f"Required array {required!r} is missing; available: {names}.")
            for name, expected in self.shapes.items():
                if name not in names:
                    continue
                actual = tuple(np.asarray(arrays[name]).shape)
                if len(actual) != len(expected) or any(
                    wanted is not None and actual_value != wanted
                    for actual_value, wanted in zip(actual, expected, strict=False)
                ):
                    errors.append(f"Array {name!r} shape {actual} does not match {expected}.")
            if self.finite:
                for name in names:
                    array = np.asarray(arrays[name])
                    if array.size > self.max_elements:
                        warnings.append(
                            f"Skipped full finite validation for {name!r} ({array.size} elements)."
                        )
                    elif np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
                        errors.append(f"Array {name!r} contains NaN or infinity.")
        except (OSError, TypeError, ValueError) as error:
            errors.append(str(error))
        finally:
            close = getattr(locals().get("loaded"), "close", None)
            if callable(close):
                close()
        return ArtifactValidationResult(not errors, tuple(errors), tuple(warnings))
