"""Per-cluster separation of state, cache, mutable runs and datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True, slots=True)
class ClusterStoragePolicy:
    """Resolve operational roots without forcing scientific data into user home."""

    state_root: str
    cache_root: str
    run_root: str
    dataset_root: str | None = None
    cache_max_bytes: int | None = None
    cache_max_age_seconds: float | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None, *, workspace: str
    ) -> ClusterStoragePolicy:
        source = dict(value or {})
        base = PurePosixPath(workspace) / ".lambdaforge"
        return cls(
            state_root=str(source.get("state_root", base / "state")),
            cache_root=str(source.get("cache_root", base / "cache")),
            run_root=str(source.get("run_root", base / "jobs")),
            dataset_root=(
                str(source["dataset_root"]) if source.get("dataset_root") is not None else None
            ),
            cache_max_bytes=(
                cls._bytes(source["cache_max_size"])
                if source.get("cache_max_size") is not None
                else None
            ),
            cache_max_age_seconds=(
                cls._duration(source["cache_max_age"])
                if source.get("cache_max_age") is not None
                else None
            ),
        )

    def __post_init__(self) -> None:
        for name in ("state_root", "cache_root", "run_root"):
            if not getattr(self, name):
                raise ValueError(f"storage.{name} cannot be empty.")
        if self.cache_max_bytes is not None and self.cache_max_bytes <= 0:
            raise ValueError("storage.cache_max_size must be positive.")
        if self.cache_max_age_seconds is not None and self.cache_max_age_seconds <= 0:
            raise ValueError("storage.cache_max_age must be positive.")

    @property
    def bundle_root(self) -> str:
        return str(PurePosixPath(self.cache_root) / "bundles")

    @property
    def environment_root(self) -> str:
        return str(PurePosixPath(self.cache_root) / "environments")

    @property
    def runtime_root(self) -> str:
        return str(PurePosixPath(self.cache_root) / "runtimes")

    @property
    def job_root(self) -> str:
        return self.run_root

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_root": self.state_root,
            "cache_root": self.cache_root,
            "run_root": self.run_root,
            "dataset_root": self.dataset_root,
            "cache_max_size": self.cache_max_bytes,
            "cache_max_age": self.cache_max_age_seconds,
        }

    @staticmethod
    def _bytes(value: object) -> int:
        if isinstance(value, bool):
            raise TypeError("Storage sizes cannot be boolean.")
        if isinstance(value, int):
            return value
        import re

        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?i?B)\s*", str(value), re.I)
        if match is None:
            raise ValueError(f"Invalid storage size: {value!r}.")
        unit = match.group(2).upper()
        binary = "I" in unit
        exponent = {
            "B": 0,
            "KB": 1,
            "KIB": 1,
            "MB": 2,
            "MIB": 2,
            "GB": 3,
            "GIB": 3,
            "TB": 4,
            "TIB": 4,
        }[unit]
        return int(float(match.group(1)) * ((1024 if binary else 1000) ** exponent))

    @staticmethod
    def _duration(value: object) -> float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        import re

        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", str(value), re.I)
        if match is None:
            raise ValueError(f"Invalid storage age: {value!r}.")
        return (
            float(match.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2).lower()]
        )
