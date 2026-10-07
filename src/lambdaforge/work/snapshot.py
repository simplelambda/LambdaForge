"""Independent immutable publication copies, opportunistically copy-on-write on Linux."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path


def validate_path(path: Path) -> None:
    """Reject lexical symlink traversal before resolving a snapshot root."""
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"Snapshot paths cannot traverse symbolic links: {path}")


def copy_file(source: str | Path, destination: str | Path) -> str:
    """Try a private reflink; fall back to copying, never mutable source hardlinks."""
    from lambdaforge.controlplane.StorageAdmission import StorageAdmission

    selected = Path(source)
    target = Path(destination)
    if any(path.is_symlink() for path in (selected, *selected.parents, target, *target.parents)):
        raise ValueError("Snapshot copy refuses symbolic files.")
    if not selected.is_file():
        raise ValueError("Snapshot source must be an ordinary file.")
    if target.exists() and os.path.samefile(selected, target):
        raise shutil.SameFileError("Snapshot source and destination must be independent.")
    if os.name == "posix":
        # A reflink does not need a second payload allocation. Check physical safety
        # and inodes first; require full-copy capacity only if the filesystem declines.
        with StorageAdmission.transaction(target, 0, purpose="publication-reflink"):
            cloned = _try_reflink(selected, target)
            if cloned is not None:
                return cloned
    with StorageAdmission.transaction(target, selected.stat().st_size, purpose="publication-copy"):
        return shutil.copy2(selected, str(target))


def _try_reflink(selected: Path, target: Path) -> str | None:
    """Try a private clone, requiring a full-byte lease for any later fallback."""
    import fcntl

    try:
        with (
            selected.open("rb") as reader,
            os.fdopen(
                os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600), "wb"
            ) as writer,
        ):
            fcntl.ioctl(writer.fileno(), 0x40049409, reader.fileno())  # Linux FICLONE
        shutil.copystat(selected, target)
        return str(target)
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL}:
            raise
        target.unlink(missing_ok=True)
        return None


def copy_tree(source: Path, destination: Path) -> None:
    """Copy only safe ordinary trees; no symlink traversal or shared mutable inodes."""
    from lambdaforge.work.managed import _canonical_entries

    validate_path(source)
    validate_path(destination)
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Snapshot destination cannot be inside its source tree.")
    _canonical_entries(source)
    shutil.copytree(source, destination, copy_function=copy_file)
