"""Cross-platform operating-system file lock for local filesystem coordination."""

from __future__ import annotations

import importlib
import math
import os
import stat
import time
from pathlib import Path
from typing import Any, BinaryIO


class CrossProcessFileLock:
    """Acquire a shared or exclusive OS lock with a bounded wait.

    The lock is attached to an open file descriptor, so the operating system
    releases it when the owning process exits, including abrupt termination.
    The lock file itself remains as harmless coordination metadata.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        shared: bool,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> None:
        if not isinstance(shared, bool):
            raise TypeError("shared must be a bool.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number.")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be a positive number.")
        self.path = Path(path)
        self.shared = shared
        self.timeout_seconds = float(timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._handle: BinaryIO | None = None
        self._windows_overlapped: Any | None = None

    def __enter__(self) -> CrossProcessFileLock:
        """Acquire this lease and return it to the context manager."""
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Release this lease on every context-manager exit."""
        del exc_type, exc, traceback
        self.release()

    @property
    def acquired(self) -> bool:
        """Report whether this object currently owns a live lock handle."""
        return self._handle is not None

    def acquire(self) -> None:
        """Wait up to the configured timeout for the filesystem lock."""
        if self._handle is not None:
            raise RuntimeError("Cross-process file lock is already acquired.")
        handle = self._open_safe_handle()
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self._lock_handle(handle)
                self._handle = handle
                return
            except OSError as error:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise TimeoutError(f"Timed out acquiring file lock {self.path}.") from error
                time.sleep(self.poll_interval_seconds)

    def _open_safe_handle(self) -> BinaryIO:
        """Open lock metadata only below a real directory and never through a link."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        parent_metadata = self.path.parent.lstat()
        parent_attributes = getattr(parent_metadata, "st_file_attributes", 0)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_attributes & reparse
        ):
            raise ValueError(f"Lock parent is not a safe directory: {self.path.parent}.")
        try:
            before = self.path.lstat()
        except FileNotFoundError:
            before = None
        if before is not None:
            before_attributes = getattr(before, "st_file_attributes", 0)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before_attributes & reparse
            ):
                raise ValueError(f"Lock path is not a safe regular file: {self.path}.")

        handle = self.path.open("a+b")
        try:
            opened = os.fstat(handle.fileno())
            after = self.path.lstat()
            after_attributes = getattr(after, "st_file_attributes", 0)
            if (
                not stat.S_ISREG(after.st_mode)
                or stat.S_ISLNK(after.st_mode)
                or after_attributes & reparse
                or opened.st_ino != after.st_ino
                or opened.st_dev != after.st_dev
            ):
                raise ValueError(f"Lock path changed or became unsafe: {self.path}.")
            return handle
        except BaseException:
            handle.close()
            raise

    def release(self) -> None:
        """Unlock and close the owning file descriptor exactly once."""
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            self._unlock_handle(handle)
        finally:
            handle.close()

    def __getstate__(self) -> dict[str, object]:
        """Never transfer a live OS lock handle through spawn serialization."""
        if self._handle is not None:
            raise RuntimeError("An acquired cross-process file lock cannot be serialized.")
        return {
            "path": self.path,
            "shared": self.shared,
            "timeout_seconds": self.timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "_handle": None,
            "_windows_overlapped": None,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore an unlocked lease in the receiving process."""
        self.__dict__.update(state)
        self._handle = None
        self._windows_overlapped = None

    def _lock_handle(self, handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            descriptor_runtime = importlib.import_module("msvcrt")
            constants_runtime = importlib.import_module("win32con")
            file_runtime = importlib.import_module("win32file")
            types_runtime = importlib.import_module("pywintypes")
            flags = constants_runtime.LOCKFILE_FAIL_IMMEDIATELY
            if not self.shared:
                flags |= constants_runtime.LOCKFILE_EXCLUSIVE_LOCK
            overlapped = types_runtime.OVERLAPPED()
            error_type = vars(types_runtime)["error"]
            try:
                file_runtime.LockFileEx(
                    descriptor_runtime.get_osfhandle(handle.fileno()),
                    flags,
                    0,
                    1,
                    overlapped,
                )
            except error_type as error:
                raise BlockingIOError(str(error)) from error
            self._windows_overlapped = overlapped
            return
        runtime = importlib.import_module("fcntl")
        mode = runtime.LOCK_SH if self.shared else runtime.LOCK_EX
        runtime.flock(handle.fileno(), mode | runtime.LOCK_NB)

    def _unlock_handle(self, handle: BinaryIO) -> None:
        handle.seek(0)
        overlapped = self._windows_overlapped
        self._windows_overlapped = None
        if os.name == "nt":
            if overlapped is None:
                raise RuntimeError("Windows file lock has no ownership metadata.")
            descriptor_runtime = importlib.import_module("msvcrt")
            file_runtime = importlib.import_module("win32file")
            file_runtime.UnlockFileEx(
                descriptor_runtime.get_osfhandle(handle.fileno()),
                0,
                1,
                overlapped,
            )
            return
        runtime = importlib.import_module("fcntl")
        runtime.flock(handle.fileno(), runtime.LOCK_UN)
