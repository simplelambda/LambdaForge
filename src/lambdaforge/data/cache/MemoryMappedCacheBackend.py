"""Memory-mapped reader for quota-bound dataset cache files."""

from __future__ import annotations

import hashlib
import mmap
import os

from lambdaforge.data.cache.CacheIntegrityError import CacheIntegrityError
from lambdaforge.data.cache.CacheRecord import CacheRecord
from lambdaforge.data.cache.DiskCacheBackend import DiskCacheBackend


class MemoryMappedCacheBackend(DiskCacheBackend):
    """Read disk records through a read-only mmap before deserialization.

    Mapping avoids a separate ``read_bytes`` allocation. The reconstructed
    dataset sample still occupies its normal live Python/Tensor memory.
    """

    def read(self, key: str) -> CacheRecord | None:
        """Verify a mapped record while retaining its shared filesystem lease."""
        self._ensure_process_local_state()
        path = self._path_for(key)
        lease = self._file_lock(shared=True)
        lease.acquire()
        mapped: mmap.mmap | None = None
        token: str | None = None
        try:
            with path.open("rb") as handle:
                mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        except FileNotFoundError:
            lease.release()
            return None
        except (OSError, ValueError) as error:
            lease.release()
            if self.remove_invalid_records:
                self.remove_if_unchanged(key, hashlib.sha256(b"").hexdigest())
            raise CacheIntegrityError("Cache record cannot be memory mapped.") from error
        try:
            token = hashlib.sha256(mapped).hexdigest()
            payload = self.record_codec.decode_view(
                mapped,
                associated_data=self._associated_data(key),
            )
            try:
                os.utime(path, None)
            except OSError:
                pass
            return CacheRecord(
                payload,
                close_callbacks=(mapped.close, lease.release),
                token=token,
            )
        except CacheIntegrityError:
            mapped.close()
            lease.release()
            if self.remove_invalid_records:
                self.remove_if_unchanged(key, token)
            raise
        except Exception:
            mapped.close()
            lease.release()
            raise
