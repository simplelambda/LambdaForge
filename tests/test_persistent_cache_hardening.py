"""Security, fingerprint, crash-recovery and multiprocess cache tests."""

from __future__ import annotations

import math
import multiprocessing
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from lambdaforge.data import (
    CacheIntegrityError,
    CacheRecordCodec,
    DatasetCache,
    DatasetFingerprint,
    DatasetSerializer,
    DiskCacheBackend,
    MemoryMappedCacheBackend,
    NumpyDatasetSerializer,
)
from lambdaforge.experiments import ObjectFactory
from tests.fixtures.CacheBlockedWriterJob import CacheBlockedWriterJob
from tests.fixtures.CacheCrashJob import CacheCrashJob
from tests.fixtures.CacheWriterJob import CacheWriterJob


class VersionedDataset(Dataset[Any]):
    """Return one configured value while exposing source access counts."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def __len__(self) -> int:
        """Return the single-sample length."""
        return 1

    def __getitem__(self, index: int) -> Any:
        """Return the configured value and count the source read."""
        if index != 0:
            raise IndexError(index)
        self.calls += 1
        return self.value


class TrackingPickleSerializer(DatasetSerializer):
    """Count deserialization calls around trusted test-only pickle bytes."""

    def __init__(self) -> None:
        self.loads_calls = 0

    @property
    def format_fingerprint(self) -> str:
        """Return one stable test format identifier."""
        return "tracking-pickle-v1"

    def dumps(self, value: Any) -> bytes:
        """Serialize test values."""
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: Any) -> Any:
        """Count and deserialize test values."""
        self.loads_calls += 1
        return pickle.loads(payload)


class TestDatasetFingerprint:
    """Verify explicit cache identity changes only with declared inputs."""

    def test_configuration_is_canonical_and_components_are_independent(self) -> None:
        first = DatasetFingerprint(
            content="sha256:abc",
            transform="normalize-v2",
            configuration={"batch": 32, "nested": {"b": 2, "a": 1}},
        )
        reordered = DatasetFingerprint(
            content="sha256:abc",
            transform="normalize-v2",
            configuration={"nested": {"a": 1, "b": 2}, "batch": 32},
        )

        assert first.digest == reordered.digest
        assert first.to_dict()["configuration"] == reordered.to_dict()["configuration"]
        assert (
            first.digest
            != DatasetFingerprint(
                content="sha256:def",
                transform="normalize-v2",
                configuration={"batch": 32, "nested": {"a": 1, "b": 2}},
            ).digest
        )
        assert (
            first.digest
            != DatasetFingerprint(
                content="sha256:abc",
                transform="normalize-v3",
                configuration={"batch": 32, "nested": {"a": 1, "b": 2}},
            ).digest
        )

    def test_from_files_tracks_order_and_content(self, tmp_path: Path) -> None:
        first_path = tmp_path / "first.bin"
        second_path = tmp_path / "second.bin"
        first_path.write_bytes(b"alpha")
        second_path.write_bytes(b"beta")

        initial = DatasetFingerprint.from_files(
            [first_path, second_path],
            transform="decode-v1",
        )
        same = DatasetFingerprint.from_files(
            [first_path, second_path],
            transform="decode-v1",
        )
        reversed_order = DatasetFingerprint.from_files(
            [second_path, first_path],
            transform="decode-v1",
        )
        second_path.write_bytes(b"changed")
        changed = DatasetFingerprint.from_files(
            [first_path, second_path],
            transform="decode-v1",
        )

        assert initial.digest == same.digest
        assert initial.digest != reversed_order.digest
        assert initial.digest != changed.digest

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"content": "", "transform": "v1"},
            {"content": "content", "transform": ""},
            {
                "content": "content",
                "transform": "v1",
                "configuration": {"invalid": math.nan},
            },
        ],
    )
    def test_invalid_fingerprint_sources_fail_closed(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises((TypeError, ValueError)):
            DatasetFingerprint(**kwargs)

    def test_fingerprint_isolates_persistent_records(self, tmp_path: Path) -> None:
        backend = DiskCacheBackend(tmp_path, "fingerprints", max_bytes=100_000)
        first_source = VersionedDataset({"value": "old"})
        second_source = VersionedDataset({"value": "new"})
        first = DatasetCache(
            first_source,
            max_memory_bytes_per_process=0,
            backend=backend,
            fingerprint=DatasetFingerprint("content-v1", "transform-v1"),
        )
        second = DatasetCache(
            second_source,
            max_memory_bytes_per_process=0,
            backend=backend,
            fingerprint=DatasetFingerprint("content-v2", "transform-v1"),
        )

        assert first[0] == {"value": "old"}
        assert second[0] == {"value": "new"}
        assert first_source.calls == 1
        assert second_source.calls == 1
        assert backend.entry_count == 2


class TestCacheRecordCodec:
    """Verify integrity and authentication before dataset deserialization."""

    def test_checksum_and_hmac_have_distinct_semantics(self) -> None:
        checksum = CacheRecordCodec()
        authenticated = CacheRecordCodec(
            "hmac_sha256",
            authentication_key=b"k" * 32,
        )
        payload = b"record"
        associated = b"namespace:key"

        assert not checksum.authenticated
        assert authenticated.authenticated
        assert (
            checksum.decode(
                checksum.encode(payload, associated_data=associated),
                associated_data=associated,
            )
            == payload
        )
        assert (
            authenticated.decode(
                authenticated.encode(payload, associated_data=associated),
                associated_data=associated,
            )
            == payload
        )
        assert checksum.format_fingerprint != authenticated.format_fingerprint

    @pytest.mark.parametrize("mutation", ["header", "tag", "payload", "truncate", "append"])
    def test_hmac_rejects_every_record_region(self, mutation: str) -> None:
        codec = CacheRecordCodec("hmac_sha256", authentication_key=b"s" * 32)
        encoded = bytearray(codec.encode(b"payload", associated_data=b"binding"))
        if mutation == "header":
            encoded[0] ^= 1
        elif mutation == "tag":
            encoded[20] ^= 1
        elif mutation == "payload":
            encoded[-1] ^= 1
        elif mutation == "truncate":
            encoded.pop()
        else:
            encoded.append(0)

        with pytest.raises(CacheIntegrityError):
            codec.decode(encoded, associated_data=b"binding")

    def test_associated_data_prevents_record_substitution(self) -> None:
        codec = CacheRecordCodec("hmac_sha256", authentication_key=b"a" * 32)
        encoded = codec.encode(b"payload", associated_data=b"key-a")

        with pytest.raises(CacheIntegrityError):
            codec.decode(encoded, associated_data=b"key-b")

    def test_hmac_environment_key_is_configurable(self, monkeypatch) -> None:
        monkeypatch.setenv("LAMBDAFORGE_TEST_CACHE_KEY", "z" * 32)
        codec = CacheRecordCodec(
            "hmac_sha256",
            authentication_key_env="LAMBDAFORGE_TEST_CACHE_KEY",
        )

        assert codec.authenticated
        assert codec.key_id is not None


class TestNumpyDatasetSerializer:
    """Verify deterministic safe trees and bounded array materialization."""

    def test_round_trip_preserves_supported_numpy_torch_tree(self) -> None:
        serializer = NumpyDatasetSerializer()
        sample = {
            "array": np.arange(6, dtype=np.float32).reshape(2, 3),
            "tensor": torch.tensor([1, 2, 3], dtype=torch.int64),
            "scalar": np.float64(2.5),
            "values": [None, True, -7, float("nan"), "text", b"bytes"],
            "tuple": ("x", 3.0),
        }

        payload = serializer.dumps(sample)
        restored = serializer.loads(payload)

        assert np.array_equal(restored["array"], sample["array"])
        assert torch.equal(restored["tensor"], sample["tensor"])
        assert restored["scalar"] == sample["scalar"]
        assert math.isnan(restored["values"][3])
        assert restored["values"][5] == b"bytes"
        assert restored["tuple"] == ("x", 3.0)

    def test_encoding_is_deterministic_for_reordered_mappings(self) -> None:
        serializer = NumpyDatasetSerializer(compressed=True)
        first = {"b": torch.tensor([2]), "a": np.array([1], dtype=np.int16)}
        second = {"a": np.array([1], dtype=np.int16), "b": torch.tensor([2])}

        assert serializer.dumps(first) == serializer.dumps(second)

    @pytest.mark.parametrize(
        "value",
        [
            object(),
            {1: "non-string-key"},
            np.array([object()], dtype=object),
            torch.tensor([[1.0]]).to_sparse(),
        ],
    )
    def test_unsafe_or_ambiguous_values_are_rejected(self, value: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            NumpyDatasetSerializer().dumps(value)

    def test_array_limits_apply_before_archive_creation(self) -> None:
        serializer = NumpyDatasetSerializer(max_decoded_bytes=4)

        with pytest.raises(ValueError, match="max_decoded_bytes"):
            serializer.dumps(np.zeros(2, dtype=np.float32))

    def test_invalid_archive_is_rejected_without_pickle(self) -> None:
        serializer = NumpyDatasetSerializer()

        with pytest.raises((ValueError, zipfile.BadZipFile)):
            serializer.loads(b"not-an-archive")


class TestCoordinatedDiskCache:
    """Exercise authenticated storage, exact quotas and anti-ABA deletion."""

    def test_envelope_bytes_count_toward_quota(self, tmp_path: Path) -> None:
        codec = CacheRecordCodec()
        payload = b"12345678"
        backend = DiskCacheBackend(
            tmp_path,
            "envelope-too-small",
            max_bytes=codec.encoded_size(len(payload)) - 1,
            record_codec=codec,
        )

        assert not backend.write(f"{1:064x}", payload)
        assert backend.usage().entries == 0

    def test_tamper_is_removed_before_deserializer_runs(self, tmp_path: Path) -> None:
        codec = CacheRecordCodec("hmac_sha256", authentication_key=b"h" * 32)
        backend = DiskCacheBackend(
            tmp_path,
            "verify-before-load",
            max_bytes=100_000,
            record_codec=codec,
        )
        serializer = TrackingPickleSerializer()
        source = VersionedDataset({"value": "rebuilt"})
        cache = DatasetCache(
            source,
            max_memory_bytes_per_process=0,
            backend=backend,
            serializer=serializer,
            fingerprint=DatasetFingerprint("content-v1", "transform-v1"),
        )
        digest = cache._digest(0)
        assert digest is not None
        assert backend.write(digest, serializer.dumps({"value": "untrusted"}))
        path = backend._path_for(digest)
        encoded = bytearray(path.read_bytes())
        encoded[-1] ^= 1
        path.write_bytes(encoded)

        assert cache[0] == {"value": "rebuilt"}
        assert serializer.loads_calls == 0
        assert source.calls == 1
        assert cache[0] == {"value": "rebuilt"}
        assert serializer.loads_calls == 1

    def test_wrong_hmac_key_fails_manifest_without_mutation(self, tmp_path: Path) -> None:
        first = DiskCacheBackend(
            tmp_path,
            "key-contract",
            max_bytes=10_000,
            record_codec=CacheRecordCodec(
                "hmac_sha256",
                authentication_key=b"a" * 32,
            ),
        )
        key = f"{1:064x}"
        assert first.write(key, b"payload")
        before = first._path_for(key).read_bytes()

        with pytest.raises(ValueError, match="configuration mismatch"):
            DiskCacheBackend(
                tmp_path,
                "key-contract",
                max_bytes=10_000,
                record_codec=CacheRecordCodec(
                    "hmac_sha256",
                    authentication_key=b"b" * 32,
                ),
            )

        assert first._path_for(key).read_bytes() == before

    def test_swapped_record_is_rejected_and_original_remains_valid(self, tmp_path: Path) -> None:
        backend = DiskCacheBackend(
            tmp_path,
            "bound-records",
            max_bytes=10_000,
            record_codec=CacheRecordCodec(
                "hmac_sha256",
                authentication_key=b"x" * 32,
            ),
        )
        first_key = f"{1:064x}"
        second_key = f"{2:064x}"
        assert backend.write(first_key, b"first")
        assert backend.write(second_key, b"second")
        backend._path_for(second_key).write_bytes(backend._path_for(first_key).read_bytes())

        with pytest.raises(CacheIntegrityError):
            backend.read(second_key)
        with backend.read(first_key) as record:
            assert record.payload == b"first"

    def test_conditional_remove_does_not_delete_new_generation(self, tmp_path: Path) -> None:
        backend = DiskCacheBackend(tmp_path, "anti-aba", max_bytes=10_000)
        key = f"{3:064x}"
        assert backend.write(key, b"old")
        old = backend.read(key)
        assert old is not None
        old_token = old.token
        old.close()
        assert backend.write(key, b"new")

        assert not backend.remove_if_unchanged(key, old_token)
        current = backend.read(key)
        assert current is not None
        with current:
            assert current.payload == b"new"

    def test_namespace_rejects_conflicting_shared_quota(self, tmp_path: Path) -> None:
        backend = DiskCacheBackend(
            tmp_path,
            "manifest-contract",
            max_bytes=10_000,
            max_entries=2,
        )
        assert backend.write(f"{4:064x}", b"value")

        with pytest.raises(ValueError, match="configuration mismatch"):
            DiskCacheBackend(
                tmp_path,
                "manifest-contract",
                max_bytes=20_000,
                max_entries=2,
            )

        backend.clear()
        assert backend.manifest_path.exists()
        assert backend.lock_path.exists()
        assert backend.write(f"{5:064x}", b"reused")

    def test_spawn_writers_never_report_usage_over_shared_quota(
        self,
        tmp_path: Path,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        ready_queue = context.Queue()
        result_queue = context.Queue()
        start_event = context.Event()
        payload = b"x" * 64
        max_bytes = CacheRecordCodec().encoded_size(len(payload)) * 3
        processes = [
            context.Process(
                target=CacheWriterJob(
                    tmp_path,
                    "spawn-quota",
                    max_bytes,
                    3,
                    f"{index:064x}",
                    payload,
                    ready_queue,
                    start_event,
                    result_queue,
                )
            )
            for index in range(8)
        ]
        try:
            for process in processes:
                process.start()
            for _ in processes:
                assert ready_queue.get(timeout=20) is True
            start_event.set()
            results = [result_queue.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)

        assert all(written for written, _, _ in results)
        assert all(entries <= 3 and used_bytes <= max_bytes for _, entries, used_bytes in results)
        reopened = DiskCacheBackend(
            tmp_path,
            "spawn-quota",
            max_bytes=max_bytes,
            max_entries=3,
        )
        assert reopened.usage().entries <= 3
        assert reopened.usage().bytes <= max_bytes
        assert not list(reopened.directory.glob("*.tmp"))

    def test_mmap_record_holds_shared_lease_until_close(self, tmp_path: Path) -> None:
        backend = MemoryMappedCacheBackend(
            tmp_path,
            "mmap-lease",
            max_bytes=10_000,
            max_entries=1,
        )
        first_key = f"{10:064x}"
        second_key = f"{11:064x}"
        assert backend.write(first_key, b"first")
        record = backend.read(first_key)
        assert record is not None
        context = multiprocessing.get_context("spawn")
        started_event = context.Event()
        done_event = context.Event()
        process = context.Process(
            target=CacheBlockedWriterJob(
                tmp_path,
                "mmap-lease",
                10_000,
                1,
                second_key,
                b"second",
                started_event,
                done_event,
            )
        )
        try:
            process.start()
            assert started_event.wait(10)
            assert not done_event.wait(0.25)
            record.close()
            assert done_event.wait(10)
            process.join(timeout=10)
            assert process.exitcode == 0
        finally:
            record.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

        assert backend.usage().entries == 1
        assert backend.read(first_key) is None
        current = backend.read(second_key)
        assert current is not None
        current.close()

    @pytest.mark.parametrize(
        ("stage", "exit_code"),
        [("temporary", 91), ("after_replace", 92)],
    )
    def test_crash_recovery_releases_lock_cleans_temp_and_preserves_quota(
        self,
        tmp_path: Path,
        stage: str,
        exit_code: int,
    ) -> None:
        namespace = f"crash-{stage}"
        payload = b"p" * 64
        max_bytes = CacheRecordCodec().encoded_size(len(payload))
        backend = DiskCacheBackend(
            tmp_path,
            namespace,
            max_bytes=max_bytes,
            max_entries=1,
        )
        assert backend.write(f"{20:064x}", payload)
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=CacheCrashJob(
                tmp_path,
                namespace,
                max_bytes,
                1,
                f"{21:064x}",
                payload,
                stage,
            )
        )
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("Injected cache crash process deadlocked.")

        assert process.exitcode == exit_code
        if stage == "temporary":
            assert list(backend.directory.glob("*.tmp"))
        usage = backend.recover()
        assert usage.entries <= 1
        assert usage.bytes <= max_bytes
        assert not list(backend.directory.glob("*.tmp"))
        assert backend.write(f"{22:064x}", payload)

    def test_object_factory_builds_complete_hardened_yaml_tree(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("LAMBDAFORGE_FACTORY_CACHE_KEY", "q" * 32)
        cache = ObjectFactory.build(
            {
                "target": "lambdaforge.data.DatasetCache",
                "params": {
                    "dataset": {
                        "target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset",
                        "params": {"size": 2},
                    },
                    "max_memory_bytes_per_process": 0,
                    "backend": {
                        "target": "lambdaforge.data.MemoryMappedCacheBackend",
                        "params": {
                            "root": str(tmp_path),
                            "namespace": "factory-hardened",
                            "max_bytes": 100_000,
                            "record_codec": {
                                "target": "lambdaforge.data.CacheRecordCodec",
                                "params": {
                                    "integrity": "hmac_sha256",
                                    "authentication_key_env": "LAMBDAFORGE_FACTORY_CACHE_KEY",
                                },
                            },
                        },
                    },
                    "serializer": {
                        "target": "lambdaforge.data.NumpyDatasetSerializer",
                        "params": {"compressed": False},
                    },
                    "fingerprint": {
                        "target": "lambdaforge.data.DatasetFingerprint",
                        "params": {
                            "content": "sha256:fixture",
                            "transform": "identity-v1",
                            "configuration": {"size": 2},
                        },
                    },
                },
            }
        )

        assert isinstance(cache, DatasetCache)
        first = cache[0]
        second = cache[0]
        assert torch.equal(first["x"], second["x"])
        assert cache.stats().backend_hits == 1
