"""Bounded filesystem operations for one registered dataset root."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lambdaforge.preprocessing.DatasetArtifact import DatasetArtifact
from lambdaforge.tasks.TaskArtifact import TaskArtifact


class DatasetOperations:
    """Inspect/verify/delete an exact root without scanning unrelated filesystems."""

    @classmethod
    def stats(cls, root: str | Path) -> dict[str, Any]:
        path = Path(root).resolve()
        manifest = path if path.name == "dataset-artifact.json" else path / "dataset-artifact.json"
        artifact = DatasetArtifact.read_json(manifest) if manifest.is_file() else None
        files = tuple(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())
        return {
            "root": str(path),
            "size_bytes": sum(item.stat().st_size for item in files),
            "file_count": len(files),
            "sample_count": artifact.sample_count if artifact else None,
            "splits": dict(artifact.splits) if artifact else {},
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
        return {"valid": not errors, "errors": errors, **cls.stats(path)}

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
            raise SystemExit("Usage: DatasetOperations stats|verify|delete ROOT [ID] [--apply]")
        operation, root, *rest = values
        if operation == "stats":
            payload = cls.stats(root)
        elif operation == "verify" and rest:
            payload = cls.verify(root, rest[0])
        elif operation == "delete" and rest:
            payload = cls.delete(root, rest[0], apply="--apply" in rest[1:])
        else:
            raise SystemExit("Invalid DatasetOperations command.")
        print(json.dumps(payload, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(DatasetOperations.main())
