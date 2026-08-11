"""Reference to a named logical dataset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetReference:
    """Represent the portable ``dataset:NAME`` authoring form."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.strip() or ":" in self.name:
            raise ValueError("Dataset reference names must be non-empty and cannot contain ':'.")

    @classmethod
    def parse(cls, value: str) -> DatasetReference:
        """Parse and validate a ``dataset:NAME`` reference."""
        if not value.startswith("dataset:"):
            raise ValueError(f"Dataset references must use dataset:NAME, found {value!r}.")
        return cls(value.removeprefix("dataset:"))

    def __str__(self) -> str:
        return f"dataset:{self.name}"
