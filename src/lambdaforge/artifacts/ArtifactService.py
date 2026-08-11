"""Application service for safe artifact discovery, inspection, export and validation."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lambdaforge.artifacts.ArtifactInspection import ArtifactInspection
from lambdaforge.artifacts.ArtifactInspector import ArtifactInspector
from lambdaforge.artifacts.ArtifactPluginRegistry import ArtifactPluginRegistry
from lambdaforge.artifacts.ArtifactValidationResult import ArtifactValidationResult
from lambdaforge.artifacts.ArtifactValidator import ArtifactValidator
from lambdaforge.artifacts.GenericArtifactVisualizer import GenericArtifactVisualizer
from lambdaforge.artifacts.NumpyArtifactInspector import NumpyArtifactInspector
from lambdaforge.artifacts.TabularArtifactInspector import TabularArtifactInspector
from lambdaforge.results.ResultService import ResultService
from lambdaforge.visualization.PlotSpec import PlotSpec


class ArtifactService:
    """Coordinate artifact tools while inspectors/renderers retain separate responsibilities."""

    def __init__(
        self,
        *,
        inspectors: Sequence[ArtifactInspector] | None = None,
        plugins: ArtifactPluginRegistry | None = None,
        results: ResultService | None = None,
    ) -> None:
        self.inspectors = tuple(
            inspectors or (NumpyArtifactInspector(), TabularArtifactInspector())
        )
        self.plugins = plugins or ArtifactPluginRegistry()
        self.results = results or ResultService()

    def inspect(
        self,
        path: str | Path,
        *,
        array: str | None = None,
        rows: int = 20,
        slice_expression: str | None = None,
        inspector: str | None = None,
    ) -> ArtifactInspection:
        """Use a built-in format match or one explicit plugin inspector."""
        source = Path(path)
        if inspector is not None:
            provider = self.plugins.load("inspector", inspector)
            selected = provider() if isinstance(provider, type) else provider
        else:
            selected = next(
                (candidate for candidate in self.inspectors if candidate.supports(source)), None
            )
        if selected is None:
            raise ValueError(
                f"No safe inspector supports {source.suffix!r}. Use an explicit plugin or "
                "export to NPY/NPZ/CSV/TSV/JSON/JSONL."
            )
        return selected.inspect(source, item=array, rows=rows, slice_expression=slice_expression)

    def export_array(
        self,
        source: str | Path,
        *,
        array: str,
        destination: str | Path,
    ) -> Path:
        """Export one numeric NPY/NPZ array to bounded-interoperable CSV/JSON/NPY."""
        path = Path(source)
        loaded = np.load(path, allow_pickle=False, mmap_mode="r" if path.suffix == ".npy" else None)
        try:
            arrays = {path.stem: loaded} if isinstance(loaded, np.ndarray) else loaded
            names = tuple(arrays.files) if hasattr(arrays, "files") else tuple(arrays)
            if array not in names:
                raise KeyError(f"Array {array!r} was not found. Available arrays: {names}.")
            value = np.asarray(arrays[array])
            if value.dtype.hasobject:
                raise ValueError("Object arrays cannot be exported safely.")
            output = Path(destination)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix.lower() == ".npy":
                np.save(output, value, allow_pickle=False)
            elif output.suffix.lower() == ".json":
                output.write_text(json.dumps(value.tolist()) + "\n", encoding="utf-8")
            elif output.suffix.lower() == ".csv":
                table = value.reshape(value.shape[0], -1) if value.ndim else value.reshape(1, 1)
                with output.open("w", encoding="utf-8", newline="") as handle:
                    csv.writer(handle).writerows(table.tolist())
            else:
                raise ValueError("Array export must end in .csv, .json or .npy.")
            return output
        finally:
            close = getattr(loaded, "close", None)
            if callable(close):
                close()

    def list(self, selector: str | Path) -> tuple[dict[str, Any], ...]:
        """List logical artifacts from result envelopes plus generated plot sidecars."""
        output: list[dict[str, Any]] = []
        for record in self.results.resolve(selector):
            for logical_name, path in (
                ("best-checkpoint", record.result.best_model_path),
                ("last-checkpoint", record.result.last_model_path),
            ):
                if path:
                    candidate = Path(path)
                    if not candidate.is_absolute():
                        candidate = Path(record.run_dir) / candidate
                    output.append(
                        {
                            "job_or_attempt": record.attempt_id,
                            "logical_name": logical_name,
                            "type": "checkpoint",
                            "size_bytes": candidate.stat().st_size if candidate.is_file() else None,
                            "location": str(candidate),
                            "scope": "local",
                        }
                    )
            artifacts = record.result.get("artifacts", ())
            if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
                for value in artifacts:
                    if isinstance(value, Mapping):
                        output.append(
                            {
                                "job_or_attempt": record.attempt_id,
                                "logical_name": value.get("metadata", {}).get(
                                    "logical_name", Path(str(value.get("path", "artifact"))).name
                                )
                                if isinstance(value.get("metadata"), Mapping)
                                else Path(str(value.get("path", "artifact"))).name,
                                "type": value.get("kind", "artifact"),
                                "size_bytes": value.get("size_bytes"),
                                "location": str(Path(record.run_dir) / str(value.get("path", ""))),
                                "scope": "local",
                            }
                        )
            for sidecar in sorted(Path(record.run_dir).glob("plots/*.plot.json")):
                figure = Path(str(sidecar).removesuffix(".plot.json"))
                if figure.is_file():
                    output.append(
                        {
                            "job_or_attempt": record.attempt_id,
                            "logical_name": figure.name,
                            "type": "plot",
                            "size_bytes": figure.stat().st_size,
                            "location": str(figure),
                            "scope": "local",
                            "specification": str(sidecar),
                        }
                    )
                output.append(
                    {
                        "job_or_attempt": record.attempt_id,
                        "logical_name": sidecar.name.removesuffix(".plot.json") + "-spec",
                        "type": "plot-spec",
                        "size_bytes": sidecar.stat().st_size,
                        "location": str(sidecar),
                        "scope": "local",
                    }
                )
        return tuple(output)

    def visualization_spec(
        self,
        path: str | Path,
        *,
        visualization_type: str,
        roles: Mapping[str, Any],
        visualizer: str | None = None,
    ) -> PlotSpec:
        """Build a graph/point-cloud/mesh spec with explicit semantics."""
        if visualizer is not None:
            provider = self.plugins.load("visualizer", visualizer)
            selected = provider() if isinstance(provider, type) else provider
        else:
            selected = GenericArtifactVisualizer()
        return selected.specification(
            Path(path), visualization_type=visualization_type, roles=roles
        )

    @staticmethod
    def validate(
        path: str | Path, validators: Sequence[ArtifactValidator]
    ) -> ArtifactValidationResult:
        """Combine generic/project validators without hiding any issue."""
        source = Path(path)
        errors: list[str] = []
        warnings: list[str] = []
        if not source.is_file() or source.is_symlink():
            errors.append("Artifact is missing or symbolic.")
        elif source.stat().st_size == 0:
            errors.append("Artifact is empty.")
        for validator in validators:
            result = validator.validate(source)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        return ArtifactValidationResult(not errors, tuple(errors), tuple(warnings))
