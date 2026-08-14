"""Reference to a named logical dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True, slots=True)
class DatasetReference:
    """Represent the portable ``dataset:NAME`` authoring form."""

    name: str
    subpath: str | None = None
    version: str | None = None
    content_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or ":" in self.name or "@" in self.name:
            raise ValueError("Dataset reference names must be non-empty and cannot contain ':'/@.")
        if self.version is not None and (not self.version.strip() or "/" in self.version):
            raise ValueError("Dataset reference versions must be non-empty path-free strings.")
        if self.content_id is not None and not self.content_id.startswith("sha256:"):
            raise ValueError("Dataset reference content_id must be a versioned SHA-256.")
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
        selector, separator, subpath = body.partition("/")
        name, version_separator, version = selector.partition("@")
        return cls(
            name,
            subpath if separator else None,
            version if version_separator else None,
        )

    def __str__(self) -> str:
        version = f"@{self.version}" if self.version is not None else ""
        suffix = f"/{self.subpath}" if self.subpath is not None else ""
        return f"dataset:{self.name}{version}{suffix}"

    @property
    def selector(self) -> str:
        """Return the registry selector without scheme or subpath."""
        return f"{self.name}@{self.version}" if self.version is not None else self.name

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> DatasetReference:
        """Build an explicit ``{dataset, subpath}`` reference mapping."""
        unexpected = set(value) - {"dataset", "version", "content_id", "subpath"}
        if unexpected:
            raise ValueError(f"Unexpected dataset reference keys: {sorted(unexpected)}.")
        raw = str(value["dataset"])
        reference = cls.parse(raw) if raw.startswith("dataset:") else cls(raw)
        subpath = value.get("subpath", reference.subpath)
        version = value.get("version", reference.version)
        content_id = value.get("content_id", reference.content_id)
        return cls(
            reference.name,
            str(subpath) if subpath is not None else None,
            str(version) if version is not None else None,
            str(content_id) if content_id is not None else None,
        )
