"""Seeded group-preserving order-independent splitting of identified records."""

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from lambdaforge.catalog.errors import ContractError


class PartitionRecords:
    """Hash the explicit group (or id) and seed; never mix a group across partitions.

    Fractions are expected proportions, not exact sizes or stratification. Existing split labels
    are rejected instead of overwritten. Use fold-specific state for every CV fold.
    """

    def __init__(self, fractions: Mapping[str, float], seed: int, group: str = "id") -> None:
        if (
            not fractions
            or any(not math.isfinite(v) or v <= 0 for v in fractions.values())
            or not math.isclose(sum(fractions.values()), 1)
        ):
            raise ContractError("INVALID_PARAMETERS", "Positive split fractions must sum to one.")
        self.fractions, self.seed, self.group = dict(fractions), seed, group

    def __call__(self, records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in records:
            if "split" in row.get("partitions", {}):
                raise ContractError("SPLIT_LEAKAGE", "Refusing to overwrite existing split.")
            encoded = json.dumps(
                [self.seed, row[self.group]], sort_keys=True, allow_nan=False
            ).encode()
            value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") / 2**64
            cumulative = 0.0
            for split, fraction in self.fractions.items():
                cumulative += fraction
                if value < cumulative:
                    yield {**row, "partitions": {**row.get("partitions", {}), "split": split}}
                    break
