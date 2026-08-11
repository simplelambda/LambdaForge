"""Compare normalized scientific identity payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.reproducibility.IdentityExplanation import IdentityExplanation
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity


class IdentityExplainer:
    """Explain identity changes path by path without executing user code."""

    def compare(
        self,
        current_payload: Mapping[str, Any],
        previous_payload: Mapping[str, Any] | None = None,
    ) -> IdentityExplanation:
        """Return digests and semantic differences between two payloads."""
        current = ScientificIdentity.from_payload(current_payload)
        if previous_payload is None:
            return IdentityExplanation(False, current.digest, None, ())
        previous = ScientificIdentity.from_payload(previous_payload)
        changes: list[Mapping[str, Any]] = []
        self._walk(current_payload, previous_payload, "", changes)
        return IdentityExplanation(
            current.digest == previous.digest,
            current.digest,
            previous.digest,
            tuple(changes),
        )

    def _walk(
        self, current: Any, previous: Any, path: str, changes: list[Mapping[str, Any]]
    ) -> None:
        if isinstance(current, Mapping) and isinstance(previous, Mapping):
            for key in sorted(set(current) | set(previous), key=str):
                self._walk(
                    current.get(key, "<missing>"),
                    previous.get(key, "<missing>"),
                    f"{path}.{key}".lstrip("."),
                    changes,
                )
            return
        if (
            isinstance(current, Sequence)
            and isinstance(previous, Sequence)
            and not isinstance(current, (str, bytes, bytearray))
            and not isinstance(previous, (str, bytes, bytearray))
        ):
            for index in range(max(len(current), len(previous))):
                left = current[index] if index < len(current) else "<missing>"
                right = previous[index] if index < len(previous) else "<missing>"
                self._walk(left, right, f"{path}[{index}]", changes)
            return
        if current != previous:
            changes.append({"path": path or "<root>", "previous": previous, "current": current})
