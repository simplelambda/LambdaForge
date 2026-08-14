"""Streaming JSONL index for logical dataset members."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.data.DatasetMember import DatasetMember
from lambdaforge.tasks.TaskArtifact import TaskArtifact


class DatasetIndex:
    """Read, validate, summarize and compare a canonical member index without eager loading."""

    FORMAT = "jsonl"
    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    @classmethod
    def write(cls, path: str | Path, members: Iterable[DatasetMember]) -> DatasetIndex:
        """Atomically persist canonical JSONL records in caller-provided stable order."""
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp")
        seen: set[str] = set()
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for member in members:
                    if member.member_id in seen:
                        raise ValueError(f"Duplicate dataset member ID: {member.member_id!r}.")
                    seen.add(member.member_id)
                    handle.write(
                        json.dumps(
                            member.to_dict(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return cls(destination)

    def __iter__(self) -> Iterator[DatasetMember]:
        """Yield members one at a time and report the exact malformed line."""
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid DatasetIndex JSON on line {line_number}: {error.msg}."
                    ) from error
                if not isinstance(value, Mapping):
                    raise TypeError(f"DatasetIndex line {line_number} must contain an object.")
                yield DatasetMember.from_mapping(value)

    def get(self, member_id: str) -> DatasetMember:
        """Find one member by stable ID without loading the complete index."""
        for member in self:
            if member.member_id == member_id:
                return member
        raise KeyError(f"Unknown dataset member {member_id!r}.")

    def members(
        self,
        *,
        partitions: Mapping[str, Any] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[DatasetMember, ...]:
        """Return a bounded page optionally filtered by arbitrary partition assignments."""
        if offset < 0 or limit < 1 or limit > 10_000:
            raise ValueError("Dataset member pagination requires offset>=0 and 1<=limit<=10000.")
        selected: list[DatasetMember] = []
        skipped = 0
        expected = dict(partitions or {})
        for member in self:
            if any(member.partitions.get(key) != value for key, value in expected.items()):
                continue
            if skipped < offset:
                skipped += 1
                continue
            selected.append(member)
            if len(selected) >= limit:
                break
        return tuple(selected)

    def validate(
        self,
        root: str | Path | None = None,
        *,
        target_schema: Mapping[str, Any] | None = None,
        require_checksums: bool = False,
    ) -> dict[str, Any]:
        """Validate unique IDs and, when a root is supplied, safe declared local assets."""
        dataset_root = Path(root).resolve() if root is not None else None
        seen: set[str] = set()
        errors: list[str] = []
        count = 0
        for member in self:
            count += 1
            if member.member_id in seen:
                errors.append(f"Duplicate member ID: {member.member_id}")
            seen.add(member.member_id)
            if target_schema and any(
                key in target_schema for key in ("$schema", "type", "properties", "required")
            ):
                from jsonschema import Draft202012Validator

                for error in Draft202012Validator(dict(target_schema)).iter_errors(
                    dict(member.targets)
                ):
                    errors.append(f"{member.member_id}.targets: {error.message}")
            if dataset_root is None:
                continue
            for name, asset in member.assets.items():
                if "://" in asset.path:
                    continue
                unresolved = dataset_root / asset.path
                resolved = unresolved.resolve(strict=False)
                if (
                    unresolved.is_symlink()
                    or not resolved.is_relative_to(dataset_root)
                    or not resolved.exists()
                ):
                    errors.append(f"{member.member_id}.{name}: missing or unsafe asset")
                    continue
                if resolved.is_dir() and any(item.is_symlink() for item in resolved.rglob("*")):
                    errors.append(f"{member.member_id}.{name}: directory contains a symlink")
                    continue
                if require_checksums and asset.sha256 is None:
                    errors.append(f"{member.member_id}.{name}: local asset has no checksum")
                    continue
                if asset.sha256 is not None:
                    digest, size = TaskArtifact.fingerprint_path(resolved)
                    if f"sha256:{digest}" != asset.sha256 or (
                        asset.size_bytes is not None and size != asset.size_bytes
                    ):
                        errors.append(f"{member.member_id}.{name}: asset integrity differs")
        return {"valid": not errors, "member_count": count, "errors": errors}

    def summary(self) -> dict[str, Any]:
        """Derive universal counts from logical members rather than manual split fields."""
        partitions: dict[str, Counter[str]] = defaultdict(Counter)
        asset_types: Counter[str] = Counter()
        targets: dict[str, Counter[str]] = defaultdict(Counter)
        members = 0
        missing_assets = 0
        for member in self:
            members += 1
            for name, value in member.partitions.items():
                partitions[name][str(value)] += 1
            for name, value in member.targets.items():
                targets[name][json.dumps(value, sort_keys=True, default=str)] += 1
            for asset in member.assets.values():
                asset_types[asset.kind] += 1
                if asset.sha256 is None:
                    missing_assets += 1
        return {
            "member_count": members,
            "partitions": {
                name: dict(sorted(counts.items())) for name, counts in sorted(partitions.items())
            },
            "targets": {
                name: dict(sorted(counts.items())) for name, counts in sorted(targets.items())
            },
            "asset_types": dict(sorted(asset_types.items())),
            "assets_without_checksum": missing_assets,
        }

    def identity(self, global_assets: Mapping[str, Any] | None = None) -> str:
        """Hash scientific member content in a path-independent canonical stream."""
        digest = hashlib.sha256()
        digest.update(b"lambdaforge-dataset-index-v1\0")
        for member in self:
            digest.update(
                json.dumps(
                    member.identity_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            digest.update(b"\n")
        digest.update(
            json.dumps(
                dict(global_assets or {}),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        return f"sha256:{digest.hexdigest()}"

    def file_sha256(self) -> str:
        """Return the checksum of the concrete persisted JSONL representation."""
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def diff(self, other: DatasetIndex) -> dict[str, Any]:
        """Compare member identities and classify scientific changes."""
        left = {member.member_id: member.identity_dict() for member in self}
        right = {member.member_id: member.identity_dict() for member in other}
        common = sorted(set(left) & set(right))
        changed = [member_id for member_id in common if left[member_id] != right[member_id]]
        partition_changes = [
            member_id
            for member_id in changed
            if left[member_id]["partitions"] != right[member_id]["partitions"]
        ]
        target_changes = [
            member_id
            for member_id in changed
            if left[member_id]["targets"] != right[member_id]["targets"]
        ]
        asset_changes = [
            member_id
            for member_id in changed
            if left[member_id]["assets"] != right[member_id]["assets"]
        ]
        return {
            "added": sorted(set(right) - set(left)),
            "removed": sorted(set(left) - set(right)),
            "changed": changed,
            "partition_changes": partition_changes,
            "target_changes": target_changes,
            "asset_identity_changes": asset_changes,
        }
