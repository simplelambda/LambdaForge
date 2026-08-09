"""Focused immutable-store and distributed-cache integrity tests."""

from pathlib import Path

from lambdaforge.storage import DistributedArtifactCache, LocalArtifactStore


def test_local_store_staging_and_cache_corruption_recovery(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"scientific artifact")
    upstream = LocalArtifactStore(tmp_path / "upstream", name="upstream")
    reference = upstream.publish(source)
    staged = upstream.stage(reference, tmp_path / "staged.bin")
    assert staged.read_bytes() == source.read_bytes()

    cache = DistributedArtifactCache(tmp_path / "cache", upstream)
    cached = cache.resolve(reference)
    cached.write_bytes(b"corrupt")
    repaired = cache.resolve(reference)
    assert repaired.read_bytes() == source.read_bytes()
    assert cache.invalidate(reference)
