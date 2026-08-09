"""Lease-coordinated shared artifact cache."""

from __future__ import annotations

from pathlib import Path

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.storage.ArtifactReference import ArtifactReference
from lambdaforge.storage.ArtifactStore import ArtifactStore
from lambdaforge.storage.LocalArtifactStore import LocalArtifactStore


class DistributedArtifactCache:
    """Stage remote content once with shared-filesystem locks and integrity recovery."""

    def __init__(self, root: str | Path, upstream: ArtifactStore) -> None:
        self.cache = LocalArtifactStore(root, name="cache")
        self.upstream = upstream

    def resolve(self, reference: ArtifactReference) -> Path:
        """Return verified cached content, repairing corruption under an exclusive lease."""
        local_reference = ArtifactReference(
            "cache", reference.key, reference.sha256, reference.size_bytes, reference.media_type
        )
        path = self.cache._path(reference.key)
        if self.cache.exists(local_reference):
            return path
        lock_path = path.with_suffix(path.suffix + ".lease")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            lock_path, shared=False, timeout_seconds=120, poll_interval_seconds=0.05
        ):
            if self.cache.exists(local_reference):
                return path
            path.unlink(missing_ok=True)
            staged = self.upstream.stage(reference, path.with_suffix(path.suffix + ".download"))
            try:
                published = self.cache.publish(
                    staged, key=reference.key, media_type=reference.media_type
                )
            finally:
                staged.unlink(missing_ok=True)
            if published.sha256 != reference.sha256:
                path.unlink(missing_ok=True)
                raise OSError("Upstream content does not match its artifact reference.")
        return path

    def invalidate(self, reference: ArtifactReference) -> bool:
        """Remove one cache copy; the authoritative store is never modified."""
        path = self.cache._path(reference.key)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed
