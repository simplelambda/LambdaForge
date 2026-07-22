"""Closed integrity and authentication choices for persistent cache records."""

from enum import Enum


class CacheIntegrityMode(str, Enum):
    """Select whether records use an unkeyed checksum or keyed authentication."""

    CHECKSUM_SHA256 = "checksum_sha256"
    HMAC_SHA256 = "hmac_sha256"
