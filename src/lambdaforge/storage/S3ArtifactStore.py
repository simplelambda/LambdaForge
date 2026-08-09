"""Optional S3-compatible artifact-store adapter."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

from lambdaforge.storage.ArtifactReference import ArtifactReference
from lambdaforge.storage.ArtifactStore import ArtifactStore


class S3ArtifactStore(ArtifactStore):
    """Use an injected client or optional boto3 without adding a base dependency."""

    def __init__(
        self, bucket: str, *, prefix: str = "", client: Any = None, name: str = "s3"
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or self._client()
        self.name = name

    def publish(
        self, source: str | Path, *, key: str | None = None, media_type: str | None = None
    ) -> ArtifactReference:
        """Upload content with checksum metadata and verify object metadata."""
        path = Path(source)
        if not path.is_file() or path.is_symlink():
            raise ValueError("S3 artifact sources must be regular files.")
        digest = self._digest(path)
        object_key = self._key(key or f"sha256/{digest}/{path.name}")
        extra: dict[str, Any] = {"Metadata": {"sha256": digest}}
        if media_type:
            extra["ContentType"] = media_type
        self.client.upload_file(str(path), self.bucket, object_key, ExtraArgs=extra)
        return ArtifactReference(
            self.name, object_key, f"sha256:{digest}", path.stat().st_size, media_type
        )

    def stage(self, reference: ArtifactReference, destination: str | Path) -> Path:
        """Download then verify before exposing a staged file."""
        destination_path = Path(destination)
        temporary = destination_path.with_suffix(destination_path.suffix + ".part")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, reference.key, str(temporary))
            if temporary.stat().st_size != reference.size_bytes or self._digest(
                temporary
            ) != reference.sha256.removeprefix("sha256:"):
                raise OSError("Downloaded S3 artifact failed integrity validation.")
            temporary.replace(destination_path)
        finally:
            temporary.unlink(missing_ok=True)
        return destination_path

    def exists(self, reference: ArtifactReference) -> bool:
        """Validate remote size and stored SHA-256 metadata."""
        if reference.store != self.name:
            return False
        try:
            metadata = self.client.head_object(Bucket=self.bucket, Key=reference.key)
        except Exception:
            return False
        return int(metadata.get("ContentLength", -1)) == reference.size_bytes and metadata.get(
            "Metadata", {}
        ).get("sha256") == reference.sha256.removeprefix("sha256:")

    def _key(self, key: str) -> str:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"Unsafe S3 artifact key: {key!r}")
        return f"{self.prefix}/{key}" if self.prefix else key

    @staticmethod
    def _client() -> Any:
        try:
            boto3 = importlib.import_module("boto3")
        except ImportError as error:
            raise ImportError(
                "S3ArtifactStore requires 'boto3' or an injected compatible client."
            ) from error
        return boto3.client("s3")

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
