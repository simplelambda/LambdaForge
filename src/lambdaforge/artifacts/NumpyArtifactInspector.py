"""Safe bounded NumPy NPY/NPZ inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lambdaforge.artifacts.ArtifactInspection import ArtifactInspection
from lambdaforge.artifacts.ArtifactInspector import ArtifactInspector


class NumpyArtifactInspector(ArtifactInspector):
    """Inspect numeric arrays with pickle disabled and bounded statistics/previews."""

    def __init__(self, *, max_statistics_elements: int = 1_000_000) -> None:
        if max_statistics_elements < 1:
            raise ValueError("max_statistics_elements must be positive.")
        self.max_statistics_elements = int(max_statistics_elements)

    def supports(self, path: Path, *, media_type: str | None = None) -> bool:
        """Recognize only explicit safe NumPy extensions/media types."""
        return path.suffix.lower() in {".npy", ".npz"} or media_type in {
            "application/x-npy",
            "application/x-npz",
        }

    def inspect(
        self,
        path: Path,
        *,
        item: str | None = None,
        rows: int = 20,
        slice_expression: str | None = None,
    ) -> ArtifactInspection:
        """Return shape/dtype/stats plus a preview capped by ``rows``."""
        if rows < 0 or rows > 1000:
            raise ValueError("rows must be between 0 and 1000.")
        source = path.resolve()
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Artifact is missing or symbolic: {source}")
        loaded = np.load(
            source,
            allow_pickle=False,
            mmap_mode="r" if source.suffix.lower() == ".npy" else None,
        )
        try:
            arrays = {source.stem: loaded} if isinstance(loaded, np.ndarray) else loaded
            names = tuple(arrays.files) if hasattr(arrays, "files") else tuple(arrays)
            if item is not None and item not in names:
                raise KeyError(f"Array {item!r} was not found. Available arrays: {names}.")
            selected = (item,) if item is not None else names
            reports = tuple(
                self._array_report(name, np.asarray(arrays[name]), rows, slice_expression)
                for name in selected
            )
        finally:
            close = getattr(loaded, "close", None)
            if callable(close):
                close()
        return ArtifactInspection(
            "NumPy NPZ" if source.suffix.lower() == ".npz" else "NumPy NPY",
            str(source),
            source.stat().st_size,
            reports,
        )

    def _array_report(
        self, name: str, array: np.ndarray, rows: int, slice_expression: str | None
    ) -> dict[str, Any]:
        if array.dtype.hasobject:
            raise ValueError(
                f"Array {name!r} contains Python objects; unsafe pickle loading is disabled."
            )
        viewed = self._slice(array, slice_expression) if slice_expression else array
        flattened = viewed.reshape(-1)
        exact = flattened.size <= self.max_statistics_elements
        if exact:
            sample = flattened
        else:
            indices = np.linspace(
                0, flattened.size - 1, self.max_statistics_elements, dtype=np.int64
            )
            sample = flattened[indices]
        numeric = np.issubdtype(sample.dtype, np.number)
        finite_values: np.ndarray[Any, Any] | None = None
        nan_count: int | None = None
        inf_count: int | None = None
        if numeric:
            if np.issubdtype(sample.dtype, np.inexact):
                nan_count = int(np.isnan(sample).sum())
                inf_count = int(np.isinf(sample).sum())
                finite_values = sample[np.isfinite(sample)]
            else:
                nan_count = 0
                inf_count = 0
                finite_values = sample
        stats: dict[str, float | int | None] = {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
        if finite_values is not None and finite_values.size:
            stats.update(
                {
                    "min": float(np.min(finite_values)),
                    "max": float(np.max(finite_values)),
                    "mean": float(np.mean(finite_values, dtype=np.float64)),
                    "std": float(np.std(finite_values, dtype=np.float64)),
                }
            )
        preview = viewed[:rows].tolist() if viewed.ndim else viewed.tolist()
        return {
            "name": name,
            "shape": list(array.shape),
            "selected_shape": list(viewed.shape),
            "ndim": int(array.ndim),
            "dtype": str(array.dtype),
            "elements": int(array.size),
            "size_bytes": int(array.nbytes),
            **stats,
            "statistics": "exact" if exact else f"sampled:{sample.size}",
            "preview": preview,
        }

    @staticmethod
    def _slice(array: np.ndarray, expression: str) -> np.ndarray:
        """Parse a deliberately small integer/start:stop comma syntax without eval."""
        indices: list[int | slice] = []
        for token in expression.split(","):
            token = token.strip()
            if ":" in token:
                parts = token.split(":")
                if len(parts) > 3:
                    raise ValueError("Array slices use start:stop[:step] syntax.")
                values = [int(part) if part else None for part in parts]
                indices.append(slice(*values))
            else:
                indices.append(int(token))
        result = array[tuple(indices)]
        return np.asarray(result)
