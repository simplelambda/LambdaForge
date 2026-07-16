"""Windows process container with kill-on-owner-close semantics."""

from __future__ import annotations

import os
from typing import Any


class WindowsJobObject:
    """Contain training descendants in a Windows Job Object when available.

    Closing the owning handle terminates every process assigned to the job,
    including descendants. On non-Windows systems, or without ``pywin32``, the
    object becomes a documented no-op and the portable process-tree cleanup
    remains active.
    """

    def __init__(self) -> None:
        self._handle: Any = None
        self._win32api: Any = None
        self._win32con: Any = None
        self._win32job: Any = None
        if os.name != "nt":
            return
        try:
            import win32api
            import win32con
            import win32job
        except ImportError:
            return
        handle = win32job.CreateJobObject(None, "")
        information = win32job.QueryInformationJobObject(
            handle,
            win32job.JobObjectExtendedLimitInformation,
        )
        information["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        win32job.SetInformationJobObject(
            handle,
            win32job.JobObjectExtendedLimitInformation,
            information,
        )
        self._handle = handle
        self._win32api = win32api
        self._win32con = win32con
        self._win32job = win32job

    @property
    def active(self) -> bool:
        """Return whether a native job handle was created."""
        return self._handle is not None

    def assign(self, pid: int | None) -> bool:
        """Assign a newly spawned process to the container."""
        if not self.active or pid is None:
            return False
        process_handle = None
        try:
            access = self._win32con.PROCESS_TERMINATE | self._win32con.PROCESS_SET_QUOTA
            process_handle = self._win32api.OpenProcess(access, False, int(pid))
            self._win32job.AssignProcessToJobObject(self._handle, process_handle)
            return True
        except Exception:
            return False
        finally:
            if process_handle is not None:
                self._win32api.CloseHandle(process_handle)

    def close(self) -> None:
        """Close the job handle, terminating any remaining contained processes."""
        if self._handle is None:
            return
        self._win32api.CloseHandle(self._handle)
        self._handle = None
