"""Versioned immutable contract for one shared persistent-cache namespace."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CacheNamespaceManifest:
    """Prevent processes with incompatible quotas or codecs from sharing files."""

    format_version: int
    namespace: str
    max_bytes: int
    max_entries: int
    record_codec_fingerprint: str

    def __post_init__(self) -> None:
        """Validate exact field types so booleans cannot masquerade as quotas."""
        if type(self.format_version) is not int or self.format_version < 1:
            raise ValueError("format_version must be a positive integer.")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("namespace must be a non-empty string.")
        if type(self.max_bytes) is not int or self.max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")
        if type(self.max_entries) is not int or self.max_entries < 1:
            raise ValueError("max_entries must be a positive integer.")
        if not isinstance(self.record_codec_fingerprint, str) or not self.record_codec_fingerprint:
            raise ValueError("record_codec_fingerprint must be a non-empty string.")

    @classmethod
    def read(cls, path: str | Path) -> CacheNamespaceManifest:
        """Read and validate a namespace manifest without applying defaults."""
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid cache namespace manifest: {source}") from error
        if not isinstance(value, dict):
            raise ValueError("Cache namespace manifest must be a JSON object.")
        expected = {
            "format_version",
            "namespace",
            "max_bytes",
            "max_entries",
            "record_codec_fingerprint",
        }
        if set(value) != expected:
            raise ValueError("Cache namespace manifest fields do not match this format.")
        try:
            return cls(
                format_version=value["format_version"],
                namespace=value["namespace"],
                max_bytes=value["max_bytes"],
                max_entries=value["max_entries"],
                record_codec_fingerprint=value["record_codec_fingerprint"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Cache namespace manifest contains invalid field values.") from error

    def write_atomic(self, path: str | Path) -> None:
        """Flush and atomically publish this manifest in its final directory."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(asdict(self), handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def assert_compatible(self, other: CacheNamespaceManifest) -> None:
        """Fail closed when a process proposes a different shared contract."""
        if self != other:
            raise ValueError(
                "Persistent cache namespace configuration mismatch. Use a new namespace "
                "or clear/recreate it with one shared quota and record codec."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for diagnostics."""
        return asdict(self)
