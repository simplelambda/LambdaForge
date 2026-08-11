"""Contract for scalable dataset identity strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.data.DatasetIdentity import DatasetIdentity


class DataIdentityProvider(ABC):
    """Derive logical identity without coupling callers to one hashing policy."""

    @abstractmethod
    def identify(
        self, path: Path, descriptor: Mapping[str, Any], *, source_dir: Path
    ) -> DatasetIdentity:
        """Return the identity for one resolved dataset location."""
