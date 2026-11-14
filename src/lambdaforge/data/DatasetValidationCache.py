"""Placement-bound receipts for explicit, reusable scientific dataset validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any

from lambdaforge.data.DatasetArtifact import DatasetArtifact
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.index import DatasetIndex
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json
from lambdaforge.work.models import WorkInput


class DatasetValidationCache:
    """Share validation between Runs, not data, scientific evidence or mutable loader state.

    Receipts are reconstructible, local trusted-cache metadata. A reuse checks every entry's
    device/inode/type/size/mtime/ctime without reading asset bytes. This detects ordinary local
    changes, not hostile filesystem metadata replay. First use, changed metadata or ``force``
    requires native exact verification and the project's explicit scientific validator again.
    POSIX filesystem semantics and immutable managed DatasetArtifact placements are required.
    """

    VERSION = 1
    MAX_ENTRIES = 200_000

    def __init__(self, root: str | Path) -> None:
        """Use an owned reconstructible cache directory, outside all validated datasets."""
        self.root = Path(root).expanduser().absolute()

    def validate(
        self,
        dataset: WorkInput,
        *,
        validator: Callable[[Path], bool | None],
        contract: str,
        version: str,
        implementation: Mapping[str, Any],
        parameters: Mapping[str, Any] | None = None,
        force: bool = False,
        timeout_seconds: float = 3600,
    ) -> dict[str, Any]:
        """Verify once per exact dataset, placement, consumer code and validation contract.

        ``validator(root)`` must raise on rejection, return False, or return True/None on
        success; it must not modify dataset files or train/fit seed-dependent state. ``contract``
        and ``version`` name its rules; ``parameters`` must include all relevant loader settings.
        ``implementation`` is the native consumer code identity, not a seed/Run fingerprint.
        A corrupt receipt fails closed; explicit ``force=True`` may reconstruct it after checks.
        No metadata-only UI or ordinary dataset resolution invokes this expensive operation.
        """
        if os.name != "posix":
            raise ValueError("Reusable dataset validation currently requires POSIX filesystems.")
        if dataset.kind != "dataset" or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", dataset.content_id or ""
        ):
            raise ValueError("Dataset validation requires an exact managed dataset WorkInput.")
        if not contract.strip() or not version.strip() or not callable(validator):
            raise ValueError("An explicit validation contract, version and callable are required.")
        if (
            not implementation.get("provider")
            or implementation.get("provider") == "unversioned"
            or not implementation.get("revision")
        ):
            raise ValueError("Reusable validation requires a verifiable consumer code identity.")
        directory = dataset.path.expanduser().absolute()
        self._safe(directory)
        self._safe(self.root)
        if (
            directory == self.root
            or directory in self.root.parents
            or self.root in directory.parents
        ):
            raise ValueError("Dataset and validation cache directories must not overlap.")
        if not directory.is_dir():
            raise ValueError("Dataset validation requires a complete managed dataset directory.")
        identity = {
            "dataset_id": dataset.content_id,
            "placement": str(directory),
            "contract": contract,
            "version": version,
            "parameters": dict(parameters or {}),
            "implementation": dict(implementation),
            "runtime": [sys.version, sys.prefix, LambdaForgeVersion.CURRENT],
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > 32768:
            raise ValueError("Dataset validation contract exceeds its metadata bound.")
        # Normalize immutable JSON containers to portable JSON; never pickle callbacks/results.
        identity = json.loads(encoded)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        # Same lease convention as native Work-cache GC; collection cannot remove a validator.
        with CrossProcessFileLock(
            self.root.parent / f".{self.root.name}.lease",
            shared=True,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=0.1,
        ):
            self._safe(self.root)
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            ownership = self.root.stat()
            if ownership.st_uid != os.getuid() or ownership.st_mode & 0o022:
                raise ValueError(
                    "Dataset validation cache must be owned and not publicly writable."
                )
            with CrossProcessFileLock(
                self.root / f"{key}.lock",
                shared=False,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=0.1,
            ):
                receipt = self.root / f"{key}.json"
                self._safe(receipt)
                if receipt.exists() and not receipt.is_file():
                    raise ValueError("Dataset validation receipt is not a regular file.")
                snapshot = self._snapshot(directory)
                if receipt.exists() and not force:
                    if not receipt.is_file() or receipt.stat().st_size > 65536:
                        raise ValueError("Unsafe or oversized dataset validation receipt.")
                    try:
                        previous = json.loads(receipt.read_text(encoding="utf-8"))
                    except (ValueError, UnicodeError) as error:
                        raise ValueError(
                            "Corrupt validation receipt; use force=True to revalidate."
                        ) from error
                    if (
                        not isinstance(previous, dict)
                        or previous.get("receipt_version") != self.VERSION
                        or previous.get("identity") != identity
                        or previous.get("valid") is not True
                    ):
                        raise ValueError(
                            "Invalid validation receipt; use force=True to revalidate."
                        )
                    if previous.get("snapshot") == snapshot:
                        return {**previous, "reused": True, "receipt": str(receipt)}
                # A failed explicit revalidation must not leave an old success reusable.
                receipt.unlink(missing_ok=True)
                checked = DatasetOperations.verify(directory, str(dataset.content_id))
                if not checked["valid"]:
                    raise ValueError(f"Dataset integrity validation failed: {checked['errors']}")
                artifact = DatasetArtifact.read_json(directory / "dataset-artifact.json")
                if artifact.dataset_artifact_version != 2 or not artifact.index.get("path"):
                    raise ValueError(
                        "Reusable validation requires a sealed DatasetArtifact v2 index."
                    )
                # Native verification may describe an external URI without fetching its bytes.
                # Such content cannot be covered by a local placement-metadata receipt.
                assets = (
                    asset
                    for member in DatasetIndex(directory / str(artifact.index["path"]))
                    for asset in member.assets.values()
                )
                for asset in chain(artifact.global_assets.values(), assets):
                    if (
                        asset.kind not in {"file", "directory"}
                        or "://" in asset.path
                        or not asset.sha256
                    ):
                        raise ValueError("Reusable validation requires local checksummed assets.")
                result = validator(directory)
                if result is not None and result is not True:
                    raise ValueError("Scientific dataset validator did not accept the dataset.")
                if self._snapshot(directory) != snapshot:
                    raise ValueError(
                        "Dataset changed during validation; no reusable receipt published."
                    )
                document = {
                    "receipt_version": self.VERSION,
                    "identity": identity,
                    "valid": True,
                    "snapshot": snapshot,
                    "validated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                atomic_write_json(receipt, document)
                return {**document, "reused": False, "receipt": str(receipt)}

    @staticmethod
    def _safe(path: Path) -> None:
        if any(value.is_symlink() for value in (path, *path.parents)):
            raise ValueError("Dataset validation paths must not traverse symlinks.")

    @classmethod
    def _snapshot(cls, root: Path) -> str:
        """Bounded metadata-only inventory; no atime (reads must not invalidate receipts)."""
        digest = hashlib.sha256()
        pending = [root]
        count = 0
        while pending:
            path = pending.pop()
            metadata = path.lstat()
            if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
                raise ValueError("Dataset contains a symlink or unsupported filesystem entry.")
            count += 1
            if count > cls.MAX_ENTRIES:
                raise ValueError("Dataset exceeds the reusable validation inventory bound.")
            value = [
                path.relative_to(root).as_posix(),
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            ]
            digest.update(json.dumps(value, separators=(",", ":")).encode() + b"\n")
            if stat.S_ISDIR(metadata.st_mode):
                children = []
                with os.scandir(path) as entries:
                    for entry in entries:
                        children.append(Path(entry.path))
                        if count + len(pending) + len(children) > cls.MAX_ENTRIES:
                            raise ValueError(
                                "Dataset exceeds the reusable validation inventory bound."
                            )
                pending.extend(sorted(children, reverse=True))
        cls._safe(root)
        return "sha256:" + digest.hexdigest()
