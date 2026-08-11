"""Reference to a named logical dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True, slots=True)
class DatasetReference:
    """Represent the portable ``dataset:NAME`` authoring form."""

    name: str
    subpath: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or ":" in self.name:
            raise ValueError("Dataset reference names must be non-empty and cannot contain ':'.")
        if self.subpath is not None:
            path = PurePosixPath(self.subpath)
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise ValueError("Dataset reference subpaths must be relative and contained.")

    @classmethod
    def parse(cls, value: str) -> DatasetReference:
        """Parse and validate a ``dataset:NAME`` reference."""
        if not value.startswith("dataset:"):
            raise ValueError(f"Dataset references must use dataset:NAME, found {value!r}.")
        body = value.removeprefix("dataset:")
        name, separator, subpath = body.partition("/")
        return cls(name, subpath if separator else None)

    def __str__(self) -> str:
        suffix = f"/{self.subpath}" if self.subpath is not None else ""
        return f"dataset:{self.name}{suffix}"

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> DatasetReference:
        """Build an explicit ``{dataset, subpath}`` reference mapping."""
        unexpected = set(value) - {"dataset", "subpath"}
        if unexpected:
            raise ValueError(f"Unexpected dataset reference keys: {sorted(unexpected)}.")
        raw = str(value["dataset"])
        reference = cls.parse(raw) if raw.startswith("dataset:") else cls(raw)
        subpath = value.get("subpath", reference.subpath)
        return cls(reference.name, str(subpath) if subpath is not None else None)
