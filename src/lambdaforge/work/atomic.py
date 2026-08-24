"""Small suffix-preserving atomic file publication primitives."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4


def atomic_build_file(destination: Path, build: Callable[[Path], Any]) -> Path:
    """Build a regular sibling file, fsync it and atomically promote it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(destination.suffixes)
    stem = destination.name[: -len(suffix)] if suffix else destination.name
    temporary = destination.with_name(f".{stem}.{os.getpid()}.{uuid4().hex}.tmp{suffix}")
    try:
        build(temporary)
        if not temporary.is_file() or temporary.is_symlink():
            raise FileNotFoundError(f"File producer did not create its destination: {temporary}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_write_bytes(destination: Path, value: bytes) -> Path:
    """Atomically replace one file with bytes."""

    def write(target: Path) -> None:
        target.write_bytes(value)

    return atomic_build_file(destination, write)


def atomic_write_text(destination: Path, value: str, *, encoding: str = "utf-8") -> Path:
    """Atomically replace one text file."""

    def write(target: Path) -> None:
        target.write_text(value, encoding=encoding)

    return atomic_build_file(destination, write)


def atomic_write_json(destination: Path, value: Any) -> Path:
    """Atomically replace one file with strict, deterministic UTF-8 JSON."""

    def write(target: Path) -> None:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    return atomic_build_file(destination, write)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["atomic_build_file", "atomic_write_bytes", "atomic_write_json", "atomic_write_text"]
