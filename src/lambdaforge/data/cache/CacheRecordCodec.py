"""Versioned checksum or HMAC envelope for persistent cache payloads."""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from typing import Any

from lambdaforge.data.cache.CacheIntegrityError import CacheIntegrityError
from lambdaforge.data.cache.CacheIntegrityMode import CacheIntegrityMode


class CacheRecordCodec:
    """Verify persistent bytes before any dataset deserializer sees them.

    SHA-256 checksums detect accidental corruption but do not authenticate a
    writer. HMAC-SHA256 authenticates records with a secret supplied directly
    from Python or through an environment variable. Neither mode encrypts.
    """

    MAGIC = b"LFCACHE2"
    FORMAT_VERSION = "2"
    SIGNATURE_BYTES = 32
    HEADER_BYTES = len(MAGIC) + 1 + 8 + SIGNATURE_BYTES

    def __init__(
        self,
        integrity: CacheIntegrityMode | str = CacheIntegrityMode.CHECKSUM_SHA256,
        *,
        authentication_key: bytes | None = None,
        authentication_key_env: str | None = None,
        minimum_key_bytes: int = 32,
    ) -> None:
        self.integrity = CacheIntegrityMode(integrity)
        if (
            not isinstance(minimum_key_bytes, int)
            or isinstance(minimum_key_bytes, bool)
            or minimum_key_bytes < 16
        ):
            raise ValueError("minimum_key_bytes must be an integer of at least 16.")
        if authentication_key_env is not None and (
            not isinstance(authentication_key_env, str) or not authentication_key_env
        ):
            raise ValueError("authentication_key_env must be a non-empty string.")
        if authentication_key is not None and not isinstance(authentication_key, bytes):
            raise TypeError("authentication_key must be bytes.")
        if authentication_key is not None and authentication_key_env is not None:
            raise ValueError("Configure authentication_key or authentication_key_env, not both.")

        key = authentication_key
        if authentication_key_env is not None:
            value = os.environ.get(authentication_key_env)
            if value is None:
                raise ValueError(f"Environment variable {authentication_key_env!r} is not defined.")
            key = value.encode("utf-8")
        if self.integrity is CacheIntegrityMode.HMAC_SHA256:
            if key is None:
                raise ValueError("HMAC-SHA256 requires an authentication key.")
            if len(key) < minimum_key_bytes:
                raise ValueError(
                    f"HMAC-SHA256 authentication keys must contain at least "
                    f"{minimum_key_bytes} bytes."
                )
        elif key is not None:
            raise ValueError("Checksum mode does not accept an authentication key.")
        self._authentication_key = key
        self.minimum_key_bytes = minimum_key_bytes
        self.authentication_key_env = authentication_key_env

    @property
    def authenticated(self) -> bool:
        """Return whether this codec authenticates writers with a secret."""
        return self.integrity is CacheIntegrityMode.HMAC_SHA256

    @property
    def key_id(self) -> str | None:
        """Return a non-secret identifier used to detect key mismatch."""
        if self._authentication_key is None:
            return None
        return hashlib.sha256(self._authentication_key).hexdigest()[:16]

    @property
    def format_fingerprint(self) -> str:
        """Identify the envelope mode and HMAC key without exposing the key."""
        key_id = self.key_id or "none"
        return f"lambdaforge-cache-record-{self.FORMAT_VERSION}:{self.integrity.value}:{key_id}"

    def encoded_size(self, payload_bytes: int) -> int:
        """Return complete record bytes for quota planning."""
        if (
            not isinstance(payload_bytes, int)
            or isinstance(payload_bytes, bool)
            or payload_bytes < 0
        ):
            raise ValueError("payload_bytes must be a non-negative integer.")
        return self.HEADER_BYTES + payload_bytes

    def encode(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        associated_data: bytes = b"",
    ) -> bytes:
        """Seal payload bytes and bind them to namespace/key associated data."""
        payload_view = memoryview(payload)
        try:
            mode_byte = self._mode_byte()
            prefix = self.MAGIC + mode_byte + struct.pack(">Q", payload_view.nbytes)
            signature = self._signature(prefix, payload_view, associated_data)
            return prefix + signature + payload_view.tobytes()
        finally:
            payload_view.release()

    def decode(
        self,
        record: bytes | bytearray | memoryview | Any,
        *,
        associated_data: bytes = b"",
    ) -> bytes:
        """Verify one complete record and return independent payload bytes."""
        payload_view = self.decode_view(record, associated_data=associated_data)
        try:
            return payload_view.tobytes()
        finally:
            payload_view.release()

    def decode_view(
        self,
        record: bytes | bytearray | memoryview | Any,
        *,
        associated_data: bytes = b"",
    ) -> memoryview:
        """Verify a record and return a zero-copy view of its payload."""
        view = memoryview(record)
        try:
            if view.nbytes < self.HEADER_BYTES:
                raise CacheIntegrityError("Cache record is truncated.")
            if view[: len(self.MAGIC)].tobytes() != self.MAGIC:
                raise CacheIntegrityError("Cache record has an unknown magic/version.")
            mode_start = len(self.MAGIC)
            mode = view[mode_start : mode_start + 1].tobytes()
            if mode != self._mode_byte():
                raise CacheIntegrityError("Cache record integrity mode does not match the backend.")
            length_start = mode_start + 1
            payload_length = struct.unpack(">Q", view[length_start : length_start + 8])[0]
            if self.HEADER_BYTES + payload_length != view.nbytes:
                raise CacheIntegrityError("Cache record payload length is inconsistent.")
            signature_start = length_start + 8
            supplied = view[signature_start : signature_start + self.SIGNATURE_BYTES].tobytes()
            payload_view = view[self.HEADER_BYTES :]
            prefix = view[:signature_start]
            expected = self._signature(prefix, payload_view, associated_data)
            if not hmac.compare_digest(supplied, expected):
                label = "authentication tag" if self.authenticated else "checksum"
                payload_view.release()
                raise CacheIntegrityError(f"Cache record {label} verification failed.")
            return payload_view
        finally:
            view.release()

    def _mode_byte(self) -> bytes:
        if self.integrity is CacheIntegrityMode.HMAC_SHA256:
            return b"H"
        return b"C"

    def _signature(
        self,
        prefix: bytes | memoryview,
        payload: memoryview,
        associated_data: bytes,
    ) -> bytes:
        domain = b"LAMBDAFORGE-CACHE-RECORD\0"
        associated_prefix = struct.pack(">Q", len(associated_data))
        digest: Any
        if self._authentication_key is None:
            digest = hashlib.sha256()
        else:
            digest = hmac.new(self._authentication_key, digestmod=hashlib.sha256)
        digest.update(domain)
        digest.update(associated_prefix)
        digest.update(associated_data)
        digest.update(prefix)
        digest.update(payload)
        return digest.digest()
