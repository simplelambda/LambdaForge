"""Reusable include/exclude policy for logged scalar names."""

from __future__ import annotations

from collections.abc import Sequence
from fnmatch import fnmatchcase


class LogKeyFilter:
    """Select log keys with shell-style include and exclude patterns."""

    def __init__(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> None:
        self.include = tuple(str(pattern) for pattern in include) if include is not None else None
        self.exclude = tuple(str(pattern) for pattern in (exclude or ()))

    def accepts(self, key: str) -> bool:
        """Return whether ``key`` satisfies both pattern sets."""
        included = self.include is None or any(
            fnmatchcase(key, pattern) for pattern in self.include
        )
        excluded = any(fnmatchcase(key, pattern) for pattern in self.exclude)
        return included and not excluded
