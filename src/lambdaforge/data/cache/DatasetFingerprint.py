"""Explicit dataset-content, transform and configuration identity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class DatasetFingerprint:
    """Create a canonical fingerprint that can participate in cache keys.

    LambdaForge cannot reliably infer the semantics of arbitrary callables.
    Researchers therefore name the content snapshot and deterministic
    transform explicitly, while configuration mappings are canonicalized.
    """

    FORMAT_VERSION = "1"

    def __init__(
        self,
        content: str,
        transform: str,
        configuration: Mapping[str, Any] | str | None = None,
    ) -> None:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty digest or version identifier.")
        if not isinstance(transform, str) or not transform.strip():
            raise ValueError("transform must be a non-empty deterministic transform identifier.")
        self.content = content.strip()
        self.transform = transform.strip()
        if configuration is None:
            configuration_text = "{}"
        elif isinstance(configuration, str):
            if not configuration:
                raise ValueError("configuration strings must be non-empty.")
            configuration_text = configuration
        elif isinstance(configuration, Mapping):
            self._validate_configuration(configuration, active=set(), depth=0)
            try:
                configuration_text = json.dumps(
                    dict(configuration),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "Dataset fingerprint configuration must contain canonical JSON values."
                ) from error
        else:
            raise TypeError("configuration must be a mapping, string or None.")
        self.configuration = configuration_text
        self.digest = self._combined_digest()

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | Path],
        *,
        transform: str,
        configuration: Mapping[str, Any] | str | None = None,
        chunk_bytes: int = 1_048_576,
    ) -> DatasetFingerprint:
        """Hash an ordered file-content snapshot without retaining file bytes."""
        if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence) or not paths:
            raise ValueError("paths must be a non-empty ordered sequence.")
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
            raise ValueError("chunk_bytes must be a positive integer.")
        digest = hashlib.sha256()
        digest.update(b"LAMBDAFORGE-DATASET-CONTENT\0")
        for index, value in enumerate(paths):
            path = Path(value).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"Dataset fingerprint path is not a file: {path}")
            size = path.stat().st_size
            digest.update(index.to_bytes(8, "big"))
            digest.update(size.to_bytes(16, "big"))
            with path.open("rb") as handle:
                while chunk := handle.read(chunk_bytes):
                    digest.update(chunk)
        return cls(
            content=f"sha256:{digest.hexdigest()}",
            transform=transform,
            configuration=configuration,
        )

    def to_dict(self) -> dict[str, str]:
        """Return the three source components plus their combined digest."""
        return {
            "format_version": self.FORMAT_VERSION,
            "content": self.content,
            "transform": self.transform,
            "configuration": self.configuration,
            "digest": self.digest,
        }

    def _combined_digest(self) -> str:
        digest = hashlib.sha256()
        for component in (
            "lambdaforge-dataset-fingerprint",
            self.FORMAT_VERSION,
            self.content,
            self.transform,
            self.configuration,
        ):
            encoded = component.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    @classmethod
    def _validate_configuration(
        cls,
        value: Any,
        *,
        active: set[int],
        depth: int,
    ) -> None:
        if depth > 128:
            raise ValueError("Dataset fingerprint configuration is too deeply nested.")
        if value is None or type(value) in {bool, int, str}:
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Dataset fingerprint floats must be finite.")
            return
        identity = id(value)
        if isinstance(value, Mapping):
            if identity in active:
                raise ValueError("Dataset fingerprint configuration cannot contain cycles.")
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Dataset fingerprint configuration keys must be strings.")
            active.add(identity)
            try:
                for item in value.values():
                    cls._validate_configuration(item, active=active, depth=depth + 1)
            finally:
                active.remove(identity)
            return
        if isinstance(value, (list, tuple)):
            if identity in active:
                raise ValueError("Dataset fingerprint configuration cannot contain cycles.")
            active.add(identity)
            try:
                for item in value:
                    cls._validate_configuration(item, active=active, depth=depth + 1)
            finally:
                active.remove(identity)
            return
        raise TypeError("Dataset fingerprint configuration contains a non-JSON value.")
