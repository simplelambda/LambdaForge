"""Explicit operator-owned authorization boundary for SQL source mutations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SQLWritePolicy:
    """Supply from trusted operator Python, never from preview/agent-authored booleans.

    This is an explicit API authorization contract, not a security sandbox. A caller already
    executing unrestricted Python can bypass it; UI/session policy must protect its creation.
    """

    allow_insert: bool = False
    allow_update: bool = False
    allow_replace: bool = False
    allow_ddl: bool = False

    def __post_init__(self) -> None:
        if any(
            type(v) is not bool
            for v in (self.allow_insert, self.allow_update, self.allow_replace, self.allow_ddl)
        ):
            raise TypeError(
                "SQL authorization policy fields must be booleans from trusted operator code."
            )
