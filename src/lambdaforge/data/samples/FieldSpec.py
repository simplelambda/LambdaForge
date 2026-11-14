"""Portable explicit reads from member assets, targets, partitions or metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lambdaforge.catalog.errors import ContractError


@dataclass(frozen=True)
class FieldSpec:
    """One field, not an implicit concatenation of every array in a file.

    Shapes exclude batch; None means ragged/unknown. `path` indexes a JSON/CSV structure,
    not an executable expression. Asset names are logical DatasetMember asset keys.
    """

    name: str
    source: str = "asset"
    asset: str | None = None
    format: str | None = None
    path: tuple[str | int, ...] = ()
    dtype: str | None = None
    shape: tuple[int | None, ...] | None = None
    axes: tuple[str, ...] = ()
    units: str | None = None
    role: str = "input"
    required: bool = True
    missing: str = "error"
    default: Any = None
    collate: str = "stack"

    def __post_init__(self) -> None:
        import json

        if (
            not isinstance(self.name, str)
            or not self.name
            or type(self.required) is not bool
            or self.source not in {"asset", "targets", "metadata", "partitions", "id"}
            or any(type(v) not in {str, int} for v in self.path)
        ):
            raise ContractError("INVALID_SCHEMA", "Invalid field name or source.")
        if self.role not in {
            "input",
            "target",
            "loss",
            "metadata",
            "mask",
            "index",
            "coordinates",
            "vector",
        }:
            raise ContractError("INVALID_SCHEMA", f"Unknown field role {self.role!r}.")
        if self.source == "asset" and (
            not self.asset or self.format not in {"npy", "npz", "json", "csv"}
        ):
            raise ContractError("INVALID_SCHEMA", "Asset fields need a logical asset and format.")
        if self.source != "asset" and (self.asset is not None or self.format is not None):
            raise ContractError("INVALID_SCHEMA", "Only asset sources declare asset/format.")
        if self.missing not in {"error", "none", "default"} or self.collate not in {
            "stack",
            "ragged",
            "pad",
            "list",
        }:
            raise ContractError("INVALID_SCHEMA", "Invalid missing/collate policy.")
        if self.required and self.missing != "error":
            raise ContractError("INVALID_SCHEMA", "Required fields cannot silently default.")
        if self.shape is not None and any(
            v is not None and (type(v) is not int or v < 0) for v in self.shape
        ):
            raise ContractError("INVALID_SCHEMA", "Shape dimensions must be nonnegative or null.")
        if self.axes and (self.shape is None or len(self.axes) != len(self.shape)):
            raise ContractError("INVALID_SCHEMA", "Axes and shape ranks must agree.")
        if self.source == "asset" and self.format == "npz" and len(self.path) != 1:
            raise ContractError("INVALID_SCHEMA", "NPZ fields select exactly one array key.")
        if self.format == "npz" and not isinstance(self.path[0], str):
            raise ContractError("INVALID_SCHEMA", "NPZ array keys must be strings.")
        try:
            json.dumps(self.default, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ContractError(
                "INVALID_SCHEMA", "Defaults must be portable finite JSON."
            ) from error
        if self.dtype is not None:
            import numpy as np

            try:
                dtype = np.dtype(self.dtype)
            except TypeError as error:
                raise ContractError("INVALID_SCHEMA", "Invalid field dtype.") from error
            if dtype.hasobject:
                raise ContractError("UNSAFE_DTYPE", "Object/pickle field types are forbidden.")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> FieldSpec:
        value = dict(value)
        for key in ("path", "axes", "shape"):
            if value.get(key) is not None:
                value[key] = tuple(value[key])
        try:
            return cls(**value)
        except TypeError as error:
            raise ContractError("INVALID_SCHEMA", str(error)) from error

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        result = asdict(self)
        for key in ("path", "shape", "axes"):
            if result[key] is not None:
                result[key] = list(result[key])
        return result

    def validate_value(self, value: Any) -> Any:
        """Validate one decoded field without implicit dtype casts or padding."""
        import numpy as np

        if value is None or (self.dtype is None and self.shape is None):
            return value
        array = np.asarray(value)
        if self.dtype and array.dtype != np.dtype(self.dtype):
            raise ContractError(
                "DTYPE_MISMATCH", f"{self.name}: expected {self.dtype}, got {array.dtype}."
            )
        if self.shape is not None and (
            array.ndim != len(self.shape)
            or any(v is not None and v != n for v, n in zip(self.shape, array.shape, strict=False))
        ):
            raise ContractError("SHAPE_MISMATCH", f"{self.name}: unexpected shape {array.shape}.")
        return array
