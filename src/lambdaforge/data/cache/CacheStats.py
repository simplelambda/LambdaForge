"""Immutable statistics snapshot for one DatasetCache process replica."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Expose cache behavior and the configured process-local RAM budget."""

    memory_hits: int
    backend_hits: int
    misses: int
    writes: int
    evictions: int
    skipped_oversize: int
    serialization_errors: int
    backend_errors: int
    memory_entries: int
    memory_bytes: int
    max_memory_bytes_per_process: int
    max_memory_entries: int
    backend_entries: int
    backend_bytes: int
    process_id: int
