"""Stable hierarchical seed derivation."""

from __future__ import annotations

import hashlib


class SeedDeriver:
    """Derive independent 32-bit seeds without depending on Python hash randomization."""

    def __init__(self, root_seed: int) -> None:
        if isinstance(root_seed, bool) or not isinstance(root_seed, int):
            raise TypeError("Root seed must be an integer.")
        self.root_seed = root_seed

    def derive(self, *scope: object) -> int:
        """Derive one stable seed for a semantic component path."""
        payload = "\0".join((str(self.root_seed), *(str(value) for value in scope)))
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
