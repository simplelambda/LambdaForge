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
        self._initialization_error: str | None = None
        self._last_assignment_error: str | None = None
        if os.name != "nt":
            return
        try:
            import win32api
            import win32con
            import win32job
        except ImportError as error:
            self._initialization_error = self._describe_error(error)
            return
        handle: Any = None
        try:
            # types-pywin32 currently declares this call as returning None even
            # though pywin32 returns the native handle used by every operation
            # below. Keeping the callable dynamic isolates that third-party
            # stub mismatch without weakening typing for this object.
            create_job_object: Any = win32job.CreateJobObject
            handle = create_job_object(None, "")
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
        except Exception as error:
            self._initialization_error = self._describe_error(error)
            if handle is not None:
                try:
                    win32api.CloseHandle(handle)
                except Exception:
                    pass
            return
        self._handle = handle
        self._win32api = win32api
        self._win32con = win32con
        self._win32job = win32job

    @property
    def active(self) -> bool:
        """Return whether a native job handle was created."""
        return self._handle is not None

    @property
    def initialization_error(self) -> str | None:
        """Return the native initialization failure, when one occurred."""
        return self._initialization_error

    @property
    def last_assignment_error(self) -> str | None:
        """Return the most recent native process-assignment failure."""
        return self._last_assignment_error

    @property
    def last_error(self) -> str | None:
        """Return the most relevant native failure for compatibility."""
        return self._last_assignment_error or self._initialization_error

    def assign(self, pid: int | None) -> bool:
        """Assign a newly spawned process to the container."""
        self._last_assignment_error = None
        if pid is None:
            self._last_assignment_error = "Worker PID is unavailable."
            return False
        if not self.active:
            if os.name == "nt":
                self._last_assignment_error = (
                    self._initialization_error or "Windows Job Object is inactive."
                )
            return False
        process_handle = None
        try:
            access = self._win32con.PROCESS_TERMINATE | self._win32con.PROCESS_SET_QUOTA
            process_handle = self._win32api.OpenProcess(access, False, int(pid))
            self._win32job.AssignProcessToJobObject(self._handle, process_handle)
            return True
        except Exception as error:
            self._last_assignment_error = self._describe_error(error)
            return False
        finally:
            if process_handle is not None:
                try:
                    self._win32api.CloseHandle(process_handle)
                except Exception as error:
                    if self._last_assignment_error is None:
                        self._last_assignment_error = self._describe_error(error)

    @staticmethod
    def _describe_error(error: BaseException) -> str:
        return f"{type(error).__name__}: {error}"

    def close(self) -> None:
        """Close the job handle, terminating any remaining contained processes."""
        if self._handle is None:
            return
        self._win32api.CloseHandle(self._handle)
        self._handle = None
