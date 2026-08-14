"""Bounded filesystem operations for one registered dataset root."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lambdaforge.data.index import DatasetIndex
from lambdaforge.preprocessing.DatasetArtifact import DatasetArtifact
from lambdaforge.tasks.artifacts import TaskArtifact


class DatasetOperations:
    """Inspect/verify/delete an exact root without scanning unrelated filesystems."""

    @classmethod
    def stats(cls, root: str | Path) -> dict[str, Any]:
        path = Path(root).resolve()
        manifest = path if path.name == "dataset-artifact.json" else path / "dataset-artifact.json"
        artifact = DatasetArtifact.read_json(manifest) if manifest.is_file() else None
        index_summary: dict[str, Any] = {}
        if artifact is not None and artifact.index.get("path"):
            index_path = (path / str(artifact.index["path"])).resolve(strict=False)
            if (
                index_path.is_relative_to(path)
                and index_path.is_file()
                and not index_path.is_symlink()
            ):
                index_summary = DatasetIndex(index_path).summary()
        files = tuple(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())
        return {
            "root": str(path),
            "size_bytes": sum(item.stat().st_size for item in files),
            "file_count": len(files),
            "sample_count": artifact.sample_count if artifact else None,
            "member_count": artifact.member_count if artifact else None,
            "splits": dict(artifact.splits) if artifact else {},
            "partitions": dict(artifact.partitions) if artifact else {},
            "index_summary": index_summary,
            "format": cls._format(files),
            "dataset_id": artifact.dataset_id if artifact else None,
        }

    @classmethod
    def verify(cls, root: str | Path, expected_id: str) -> dict[str, Any]:
        path = Path(root).resolve()
        if path.is_symlink() or not path.exists():
            return {"valid": False, "errors": ["Registered root is absent or a symlink."]}
        manifest = path if path.name == "dataset-artifact.json" else path / "dataset-artifact.json"
        if not manifest.is_file() or manifest.is_symlink():
            return {"valid": False, "errors": ["dataset-artifact.json is missing or unsafe."]}
        artifact = DatasetArtifact.read_json(manifest)
        errors = []
        if artifact.dataset_id != expected_id:
            errors.append("Dataset identity does not match the registry.")
        if artifact.index.get("path"):
            index_path = (path / str(artifact.index["path"])).resolve(strict=False)
            if (
                not index_path.is_relative_to(path)
                or index_path.is_symlink()
                or not index_path.is_file()
            ):
                errors.append("Dataset member index is missing or unsafe.")
            else:
                index = DatasetIndex(index_path)
                expected_sha = artifact.index.get("sha256")
                if expected_sha and index.file_sha256() != expected_sha:
                    errors.append("Dataset member index bytes differ.")
                validation = index.validate(
                    path,
                    target_schema=artifact.target_schema,
                    require_checksums=artifact.dataset_artifact_version >= 2,
                )
                errors.extend(str(item) for item in validation["errors"])
                global_identity = {
                    name: asset.identity_dict()
                    for name, asset in sorted(artifact.global_assets.items())
                }
                if index.identity(global_identity) != artifact.content_id:
                    errors.append("Dataset logical content identity differs from the manifest.")
        for expected in artifact.artifacts:
            candidate = (path / expected.path).resolve(strict=False)
            if (
                not candidate.is_relative_to(path)
                or candidate.is_symlink()
                or not candidate.exists()
            ):
                errors.append(f"Missing or unsafe artifact: {expected.path}")
                continue
            digest, size = TaskArtifact.fingerprint_path(candidate)
            if digest != expected.sha256 or size != expected.size_bytes:
                errors.append(f"Artifact bytes differ: {expected.path}")
        for name, asset in artifact.global_assets.items():
            if "://" in asset.path:
                continue
            candidate = (path / asset.path).resolve(strict=False)
            if (
                not candidate.is_relative_to(path)
                or candidate.is_symlink()
                or not candidate.exists()
            ):
                errors.append(f"Missing or unsafe global asset: {name}")
                continue
            if asset.sha256 is not None:
                digest, size = TaskArtifact.fingerprint_path(candidate)
                if f"sha256:{digest}" != asset.sha256 or (
                    asset.size_bytes is not None and size != asset.size_bytes
                ):
                    errors.append(f"Global asset bytes differ: {name}")
        return {"valid": not errors, "errors": errors, **cls.stats(path)}

    @classmethod
    def members(
        cls,
        root: str | Path,
        *,
        partitions: dict[str, str] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return a bounded logical-member page from a v2 dataset."""
        index = cls._index(root)
        values = index.members(partitions=partitions, offset=offset, limit=limit)
        return {
            "offset": offset,
            "limit": limit,
            "members": [member.to_dict() for member in values],
            "returned": len(values),
        }

    @classmethod
    def member(cls, root: str | Path, member_id: str) -> dict[str, Any]:
        """Return one exact logical member without exposing index storage details."""
        return cls._index(root).get(member_id).to_dict()

    @classmethod
    def diff(cls, left: str | Path, right: str | Path) -> dict[str, Any]:
        """Compare two logical indices at one execution location."""
        return cls._index(left).diff(cls._index(right))

    @staticmethod
    def _index(root: str | Path) -> DatasetIndex:
        path = Path(root).resolve()
        manifest = DatasetArtifact.read_json(path / "dataset-artifact.json")
        relative = manifest.index.get("path")
        if not relative:
            raise ValueError("DatasetArtifact v1 has no logical DatasetIndex.")
        index = (path / str(relative)).resolve(strict=False)
        if not index.is_relative_to(path) or index.is_symlink() or not index.is_file():
            raise ValueError("DatasetIndex is missing or unsafe.")
        return DatasetIndex(index)

    @classmethod
    def delete(cls, root: str | Path, expected_id: str, *, apply: bool) -> dict[str, Any]:
        path = Path(root).resolve()
        dangerous = {Path("/").resolve(), Path.home().resolve()}
        if path in dangerous or len(path.parts) < 3:
            raise ValueError(f"Refusing dangerous dataset root: {path}")
        verification = cls.verify(path, expected_id)
        if not verification.get("valid"):
            raise ValueError("Dataset deletion requires a valid matching DatasetArtifact.")
        if apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return {**verification, "applied": bool(apply), "root": str(path)}

    @staticmethod
    def _format(files: Sequence[Path]) -> str:
        suffixes = {item.suffix.lower().removeprefix(".") for item in files if item.suffix}
        return next(iter(suffixes)) if len(suffixes) == 1 else "mixed" if suffixes else "unknown"

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        values = tuple(argv if argv is not None else sys.argv[1:])
        if len(values) < 2:
            raise SystemExit(
                "Usage: DatasetOperations stats|verify|delete|members|member ROOT [ARGS]"
            )
        operation, root, *rest = values
        if operation == "stats":
            payload = cls.stats(root)
        elif operation == "verify" and rest:
            payload = cls.verify(root, rest[0])
        elif operation == "delete" and rest:
            payload = cls.delete(root, rest[0], apply="--apply" in rest[1:])
        elif operation == "members":
            options = json.loads(rest[0]) if rest else {}
            payload = cls.members(
                root,
                partitions=options.get("partitions"),
                offset=int(options.get("offset", 0)),
                limit=int(options.get("limit", 100)),
            )
        elif operation == "member" and rest:
            payload = cls.member(root, rest[0])
        elif operation == "diff" and rest:
            payload = cls.diff(root, rest[0])
        else:
            raise SystemExit("Invalid DatasetOperations command.")
        print(json.dumps(payload, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(DatasetOperations.main())
