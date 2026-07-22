"""Value object for exact LambdaForge experiment Schema versions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import total_ordering
from typing import Any, ClassVar


@total_ordering
@dataclass(frozen=True, slots=True)
class ExperimentSchemaVersion:
    """Represent a normalized `MAJOR.MINOR` version or unversioned legacy input."""

    value: str

    CURRENT_VALUE: ClassVar[str] = "1.1"
    UNVERSIONED_VALUE: ClassVar[str] = "unversioned"
    VERSION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)$")

    def __post_init__(self) -> None:
        if self.value == self.UNVERSIONED_VALUE:
            return
        if not isinstance(self.value, str) or self.VERSION_PATTERN.fullmatch(self.value) is None:
            raise ValueError("Schema versions must use the exact string form 'MAJOR.MINOR'.")

    @classmethod
    def current(cls) -> ExperimentSchemaVersion:
        """Return the Schema version implemented by this LambdaForge release."""
        return cls(cls.CURRENT_VALUE)

    @classmethod
    def unversioned(cls) -> ExperimentSchemaVersion:
        """Return the internal marker for a YAML document without a version."""
        return cls(cls.UNVERSIONED_VALUE)

    @classmethod
    def from_value(cls, value: Any) -> ExperimentSchemaVersion:
        """Parse a YAML value without coercing numbers or informal strings."""
        if value is None:
            return cls.unversioned()
        if not isinstance(value, str):
            raise TypeError("schema_version must be a quoted 'MAJOR.MINOR' string.")
        return cls(value)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ExperimentSchemaVersion:
        """Detect a version from one raw configuration mapping."""
        if "schema_version" not in config:
            return cls.unversioned()
        if config["schema_version"] is None:
            raise TypeError("schema_version must be a quoted 'MAJOR.MINOR' string.")
        return cls.from_value(config["schema_version"])

    @property
    def is_unversioned(self) -> bool:
        """Return whether the source omitted `schema_version`."""
        return self.value == self.UNVERSIONED_VALUE

    def to_json_value(self) -> str | None:
        """Return null for legacy input and the exact version otherwise."""
        return None if self.is_unversioned else self.value

    def sort_key(self) -> tuple[int, int]:
        """Return a forward-order key with unversioned input before version zero."""
        if self.is_unversioned:
            return (-1, -1)
        major, minor = self.value.split(".", maxsplit=1)
        return int(major), int(minor)

    def __lt__(self, other: object) -> bool:
        """Compare normalized versions while rejecting unrelated objects."""
        if not isinstance(other, ExperimentSchemaVersion):
            return NotImplemented
        return self.sort_key() < other.sort_key()

    def __str__(self) -> str:
        """Return a user-facing version label."""
        return self.value
