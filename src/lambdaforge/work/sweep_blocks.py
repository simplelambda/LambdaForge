"""Read-only validation of the existing paired-sweep commitment inventory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def sweep_block_inventory(record: Mapping[str, Any]) -> tuple[tuple[int, ...], int | None]:
    """Restore created ordinals and the sole outstanding lookahead, failing closed.

    Version-one records already persist this authority. Do not reconstruct it from
    completion order or silently discard a commitment during recovery.
    """
    if record.get("block_progress_version") != 1:
        raise ValueError("Persisted sweep block inventory is incompatible.")
    blocks = record.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Persisted sweep block inventory is missing.")
    ordinals: list[int] = []
    committed_flags: list[int] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("Persisted sweep block record is corrupt.")
        ordinal = block.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("Persisted sweep block ordinal is invalid.")
        ordinals.append(ordinal)
        if block.get("committed") is True:
            committed_flags.append(ordinal)
    if sorted(ordinals) != list(range(len(ordinals))):
        raise ValueError("Persisted sweep block ordinals are duplicated or discontinuous.")
    lookahead = record.get("lookahead")
    if not isinstance(lookahead, Mapping) or lookahead.get("maximum_blocks") != 1:
        raise ValueError("Persisted sweep lookahead policy is missing or corrupt.")
    committed = lookahead.get("committed_ordinal")
    if committed is not None and (
        isinstance(committed, bool)
        or not isinstance(committed, int)
        or committed != max(ordinals)
    ):
        raise ValueError("Persisted sweep commitment must identify the final created block.")
    if committed_flags != ([] if committed is None else [committed]):
        raise ValueError("Persisted sweep commitment and block records disagree.")
    return tuple(sorted(ordinals)), committed
