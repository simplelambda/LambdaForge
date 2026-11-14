"""Lazy samples over the native immutable member index; not another dataset registry."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.catalog.errors import ContractError
from lambdaforge.data.DatasetArtifact import DatasetArtifact
from lambdaforge.data.index import DatasetIndex
from lambdaforge.data.samples.readers import ReadBudget, read_field
from lambdaforge.data.samples.SampleSchema import SampleSchema


class SampleDataset:
    """Resolve `{dataset: NAME@VERSION}` through Work inputs, then pass that native root here.

    Construction reads only declaration/index metadata, never sample bytes. No automatic remote
    downloads: remote Works use their native placement; materialization uses DatasetService.
    DataLoader-compatible __len__/__getitem__, with no retained decoded arrays or framework runner.
    """

    def __init__(
        self,
        root: str | Path,
        schema: SampleSchema | Mapping[str, Any],
        *,
        partitions: Mapping[str, Any] | None = None,
        max_bytes: int = 32 * 1024**2,
        max_seconds: float = 5,
        max_members: int = 100_000,
        expected_content_id: str | None = None,
    ) -> None:
        from lambdaforge.work.snapshot import validate_path

        self.root = Path(root).absolute()
        validate_path(self.root)
        self.schema = (
            schema if isinstance(schema, SampleSchema) else SampleSchema.from_mapping(schema)
        )
        self.max_bytes, self.max_seconds = max_bytes, max_seconds
        ReadBudget(max_bytes, max_seconds)
        if type(max_members) is not int or max_members < 1:
            raise ContractError("INVALID_BUDGET", "max_members must be positive.")
        declaration = self.root / "dataset-artifact.json"
        validate_path(declaration)
        if declaration.stat().st_size > 1024**2:
            raise ContractError("READ_LIMIT", "Dataset declaration exceeds 1 MiB.")
        artifact = DatasetArtifact.read_json(declaration)
        self.content_id = artifact.dataset_id
        if expected_content_id is not None and self.content_id != expected_content_id:
            raise ContractError(
                "CONTENT_MISMATCH", "Dataset declaration differs from pinned input."
            )
        relative = Path(str(artifact.index.get("path", "index.jsonl")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError("UNSAFE_PATH", "Index path escapes Dataset root.")
        index = self.root / relative
        validate_path(index)
        if index.stat().st_size > 64 * 1024**2:
            raise ContractError("READ_LIMIT", "Sample metadata index exceeds 64 MiB.")
        declared_sha = artifact.index.get("sha256")
        if not declared_sha or DatasetIndex(index).file_sha256().removeprefix("sha256:") != str(
            declared_sha
        ).removeprefix("sha256:"):
            raise ContractError(
                "CONTENT_MISMATCH", "Dataset member index differs from declaration."
            )
        members: list[Any] = []
        seen = set()
        for position, member in enumerate(DatasetIndex(index)):
            if member.member_id in seen:
                raise ContractError("DUPLICATE_ID", "Duplicate Dataset sample identity.")
            seen.add(member.member_id)
            if position >= max_members:
                raise ContractError("READ_LIMIT", "Sample member inventory exceeds limit.")
            if all(member.partitions.get(k) == v for k, v in (partitions or {}).items()):
                members.append(member)
        self.members = tuple(members)

    def __len__(self) -> int:
        return len(self.members)

    def preflight(self) -> dict[str, Any]:
        """Metadata-only structural compatibility. Does not claim decoded values were verified."""
        missing = [
            (m.member_id, f.name)
            for m in self.members
            for f in self.schema.fields
            if f.source == "asset" and f.required and f.asset not in m.assets
        ]
        return {
            "schema_version": "sample-preflight-v1",
            "content_id": self.content_id,
            "members": len(self),
            "compatible": not missing,
            "missing_required": missing[:100],
            "decoded_assets": False,
            "value_validation": "on-access",
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._read(index, ReadBudget(self.max_bytes, self.max_seconds))

    def _read(self, index: int, budget: ReadBudget) -> dict[str, Any]:
        import copy

        member = self.members[index]
        fields, present, assets = {}, {}, {}
        for spec in self.schema.fields:
            value: Any
            try:
                if spec.source == "asset":
                    assert spec.asset is not None
                    asset = member.assets[spec.asset]
                    path = self.root / asset.path
                    if "://" in asset.path or not path.is_relative_to(self.root):
                        raise ContractError(
                            "UNSUPPORTED_LOCATION",
                            "Resolve assets with native Dataset placement first.",
                        )
                    try:
                        value = read_field(
                            path, str(spec.format), spec.path, budget, sha256=asset.sha256
                        )
                    except FileNotFoundError as error:
                        # A missing optional declaration is permitted; bytes declared in an
                        # immutable Dataset disappearing are corruption, never a default.
                        raise ContractError(
                            "CONTENT_MISMATCH", "A declared Dataset asset is missing."
                        ) from error
                    assets[spec.asset] = {"sha256": asset.sha256, "path": asset.path}
                else:
                    value = (
                        member.member_id if spec.source == "id" else getattr(member, spec.source)
                    )
                    for key in spec.path:
                        value = value[key]
                present[spec.name] = True
            except (KeyError, FileNotFoundError, IndexError):
                if spec.required or spec.missing == "error":
                    raise ContractError(
                        "MISSING_FIELD", f"Missing {spec.name!r} for {member.member_id!r}."
                    ) from None
                value = copy.deepcopy(spec.default) if spec.missing == "default" else None
                present[spec.name] = False
            fields[spec.name] = spec.validate_value(value)
        return {
            "id": member.member_id,
            "fields": fields,
            "present": present,
            "partitions": dict(member.partitions),
            "provenance": {
                "content_id": self.content_id,
                "member_id": member.member_id,
                "assets": assets,
            },
        }

    def preview(
        self, *, samples: int = 3, max_bytes: int = 1024**2, max_seconds: float = 2
    ) -> dict[str, Any]:
        """Explicit bounded decoding; returns shapes and small values, not full arrays."""
        import numpy as np

        if type(samples) is not int or not 0 <= samples <= 20:
            raise ContractError("READ_LIMIT", "Preview accepts 0..20 samples.")
        budget = ReadBudget(max_bytes, max_seconds)
        rows = []
        for index in range(min(samples, len(self))):
            sample = self._read(index, budget)
            rows.append(
                {
                    "id": sample["id"],
                    "present": sample["present"],
                    "fields": {
                        k: {
                            "shape": list(np.shape(v)),
                            "dtype": str(np.asarray(v).dtype),
                            "preview": np.asarray(v).reshape(-1)[:6].tolist(),
                        }
                        if v is not None
                        else None
                        for k, v in sample["fields"].items()
                    },
                }
            )
        return {"content_id": self.content_id, "samples": rows, "remaining_bytes": budget.remaining}
