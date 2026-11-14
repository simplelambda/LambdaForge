"""Strict id joins with explicit cardinality and split protection."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Literal

from lambdaforge.catalog.errors import ContractError


class JoinRecords:
    """One-to-one joins only. Refuse many-to-many, duplicate IDs and column/split ambiguity."""

    def __init__(
        self,
        how: Literal["left", "inner"] = "left",
        max_right: int = 10000,
        require_match: bool = True,
    ) -> None:
        if how not in {"left", "inner"} or max_right < 1:
            raise ContractError("INVALID_PARAMETERS", "Join needs left/inner and positive limit.")
        self.how, self.max_right, self.require_match = how, max_right, require_match

    def __call__(
        self, left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        indexed: dict[Any, Mapping[str, Any]] = {}
        for row in right:
            key = row["id"]
            if key in indexed or len(indexed) >= self.max_right:
                raise ContractError("JOIN_CARDINALITY", "Right ids must be unique and bounded.")
            indexed[key] = row
        seen = set()
        for row in left:
            key = row["id"]
            if key in seen:
                raise ContractError("JOIN_CARDINALITY", "Left ids must be unique.")
            seen.add(key)
            peer = indexed.get(key)
            if peer is None:
                if self.require_match:
                    raise ContractError("JOIN_MISSING", f"No right record for {key!r}.")
                if self.how == "left":
                    yield dict(row)
                continue
            if row.get("partitions", {}) != peer.get("partitions", {}):
                raise ContractError("SPLIT_LEAKAGE", "Joined records disagree on partitions.")
            overlap = set(row).intersection(peer) - {"id", "partitions"}
            if overlap:
                raise ContractError("JOIN_COLLISION", "Rename overlapping columns before joining.")
            yield {**row, **peer}
