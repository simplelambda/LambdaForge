"""Bounded-memory, process-isolation and disk-backend cache tests."""

from __future__ import annotations

import importlib
import pickle
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from lambdaforge.data.cache import (
    DatasetCache,
    DiskCacheBackend,
    MemoryMappedCacheBackend,
)


class CountingDataset(Dataset[Any]):
    """Expose access counts so cache hits are visible in assertions and workers."""

    def __init__(self, values: list[Any]) -> None:
        self.values = values
        self.calls: dict[int, int] = {}

    def __len__(self) -> int:
        """Return the number of configured samples."""
        return len(self.values)

    def __getitem__(self, index: int) -> Any:
        """Return a mutable sample carrying its process-local access count."""
        self.calls[index] = self.calls.get(index, 0) + 1
        return {"value": self.values[index], "source_calls": self.calls[index]}


class RepeatedIndexSampler(Sampler[int]):
    """Yield one index twice inside the same DataLoader worker."""

    def __iter__(self) -> Iterator[int]:
        """Yield a deterministic repeated index sequence."""
        return iter((0, 0))

    def __len__(self) -> int:
        """Return the sequence length."""
        return 2


class TestDatasetCache:
    """Verify that memory remains bounded without changing dataset semantics."""

    def test_lru_is_bounded_by_bytes_and_entry_count(self) -> None:
        dataset = CountingDataset(["zero", "one", "two"])
        cache = DatasetCache(
            dataset,
            max_memory_bytes_per_process=1_000_000,
            max_memory_entries=2,
        )

        cache[0]
        cache[1]
        cache[0]
        cache[2]
        cache[1]

        stats = cache.stats()
        assert dataset.calls == {0: 1, 1: 2, 2: 1}
        assert stats.memory_hits == 1
        assert stats.evictions == 2
        assert stats.memory_entries == 2
        assert stats.memory_bytes <= stats.max_memory_bytes_per_process

    def test_cached_none_and_mutable_values_are_isolated(self) -> None:
        dataset = CountingDataset([None, [1, 2]])
        cache = DatasetCache(dataset, max_memory_bytes_per_process=10_000)

        first_none = cache[0]
        second_none = cache[0]
        first_list = cache[1]
        first_list["value"].append(3)
        second_list = cache[1]

        assert first_none["value"] is None
        assert second_none["value"] is None
        assert second_list["value"] == [1, 2]
        assert dataset.calls == {0: 1, 1: 1}

    def test_oversized_sample_is_skipped_without_exceeding_budget(self) -> None:
        dataset = CountingDataset(["large-payload"])
        cache = DatasetCache(dataset, max_memory_bytes_per_process=1)

        cache[0]
        cache[0]

        stats = cache.stats()
        assert dataset.calls == {0: 2}
        assert stats.skipped_oversize == 2
        assert stats.memory_entries == 0
        assert stats.memory_bytes == 0

    def test_spawn_serialization_drops_parent_memory(self) -> None:
        dataset = CountingDataset([42])
        cache = DatasetCache(dataset, max_memory_bytes_per_process=10_000)
        cache[0]

        restored = pickle.loads(pickle.dumps(cache))

        assert restored.stats().memory_entries == 0
        restored[0]
        assert restored.dataset.calls[0] == 2

    @pytest.mark.parametrize(
        ("cache_in_workers", "expected_calls"),
        [(False, [1, 2]), (True, [1, 1])],
    )
    def test_dataloader_worker_cache_is_explicit_opt_in(
        self,
        cache_in_workers: bool,
        expected_calls: list[int],
    ) -> None:
        cache = DatasetCache(
            CountingDataset([7]),
            max_memory_bytes_per_process=10_000,
            cache_in_workers=cache_in_workers,
        )
        loader = DataLoader(
            cache,
            batch_size=None,
            num_workers=1,
            sampler=RepeatedIndexSampler(),
        )

        samples = list(loader)

        assert [sample["source_calls"] for sample in samples] == expected_calls

    def test_inherited_memory_is_discarded_when_process_id_changes(self, monkeypatch) -> None:
        dataset = CountingDataset([9])
        cache = DatasetCache(dataset, max_memory_bytes_per_process=10_000)
        cache[0]
        original_process_id = cache.stats().process_id
        module = importlib.import_module("lambdaforge.data.cache.DatasetCache")
        monkeypatch.setattr(module.os, "getpid", lambda: original_process_id + 1)

        cache[0]

        assert dataset.calls == {0: 2}
        assert cache.stats().process_id == original_process_id + 1
        assert cache.stats().memory_hits == 0

    def test_invalidate_removes_one_sample_without_flushing_other_entries(self) -> None:
        dataset = CountingDataset([1, 2])
        cache = DatasetCache(dataset, max_memory_bytes_per_process=10_000)
        cache[0]
        cache[1]

        cache.invalidate(0)
        cache[0]
        cache[1]

        assert dataset.calls == {0: 2, 1: 1}

    def test_nested_cuda_detection_handles_cycles(self) -> None:
        cyclic: list[Any] = []
        cyclic.append(cyclic)
        assert not DatasetCache._contains_cuda_tensor(cyclic)


class TestDatasetCacheBackends:
    """Exercise atomic bounded disk storage and mapped reads."""

    def test_disk_backend_enforces_byte_and_entry_quotas(self, tmp_path) -> None:
        backend = DiskCacheBackend(
            tmp_path,
            namespace="dataset-v1",
            max_bytes=120,
            max_entries=2,
        )

        for number in range(3):
            assert backend.write(f"{number:064x}", b"12345678")

        assert backend.entry_count == 2
        assert backend.current_bytes <= 120

    def test_memory_mapped_backend_releases_files_on_windows(self, tmp_path) -> None:
        key = f"{1:064x}"
        backend = MemoryMappedCacheBackend(
            tmp_path,
            namespace="mapped-v1",
            max_bytes=100,
        )
        assert backend.write(key, b"mapped-record")

        record = backend.read(key)
        assert record is not None
        with record:
            assert bytes(record.payload) == b"mapped-record"
        backend.remove(key)

        assert backend.entry_count == 0

    def test_corrupt_disk_record_is_removed_and_rebuilt(self, tmp_path) -> None:
        dataset = CountingDataset([torch.tensor([1.0])])
        backend = DiskCacheBackend(tmp_path, "corrupt-v1", max_bytes=10_000)
        cache = DatasetCache(
            dataset,
            max_memory_bytes_per_process=0,
            backend=backend,
        )
        digest = cache._digest(0)
        assert digest is not None
        assert backend.write(digest, b"not-a-pickle")

        first = cache[0]
        second = cache[0]

        assert torch.equal(first["value"], second["value"])
        assert dataset.calls == {0: 1}
        assert cache.stats().serialization_errors == 1
