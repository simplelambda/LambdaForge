"""Bounded binary command streams for explicit durable data transfers."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from threading import Lock, Thread
from typing import Any, BinaryIO, cast

from lambdaforge.controlplane.SecretRedactor import SecretRedactor


class BinaryCommand:
    """Drain bounded diagnostics independently of binary stdin/stdout."""

    def __init__(self, stdin: BinaryIO, stdout: BinaryIO, stderr: BinaryIO, owner: Any) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.owner = owner
        self._errors = bytearray()
        self._error_lock = Lock()
        self._drain_error: str | None = None
        self._thread = Thread(target=self._drain, args=(stderr,), daemon=True)
        self._thread.start()

    def _drain(self, stderr: BinaryIO) -> None:
        try:
            while chunk := stderr.read(4096):
                with self._error_lock:
                    self._errors.extend(chunk)
                    if len(self._errors) > 16384:
                        del self._errors[:-16384]
        except (EOFError, OSError) as error:
            self._drain_error = f"{type(error).__name__}: {error}"

    def diagnostics(self) -> dict[str, Any]:
        """Bounded best-effort evidence; a broken SSH connection is not a known exit code."""
        self._thread.join(timeout=0.1)
        code = None
        if hasattr(self.owner, "poll"):
            code = self.owner.poll()
        elif hasattr(self.owner, "exit_status_ready"):
            try:
                if self.owner.exit_status_ready():
                    status = self.owner.recv_exit_status()
                    code = status if status >= 0 else None
            except (EOFError, OSError) as error:
                self._drain_error = f"Exit status unavailable: {type(error).__name__}"
        with self._error_lock:
            errors = bytes(self._errors).decode("utf-8", errors="replace")
        return {
            "exit_code": code,
            "stderr": SecretRedactor.redact(errors)[-2048:],
            "stderr_observation_error": self._drain_error,
        }

    def wait(self) -> None:
        code = self.owner.wait() if hasattr(self.owner, "wait") else self.owner.recv_exit_status()
        self._thread.join(timeout=1.0)
        if code:
            raise RuntimeError(
                (
                    f"Data transfer command exited with code {code}: "
                    if code > 0
                    else "Data transfer peer exit status is unavailable: "
                )
                + self.diagnostics()["stderr"]
            )

    def finish_input(self) -> None:
        """Send EOF on subprocess pipes and Paramiko channels alike."""
        try:
            self.stdin.flush()
        except (EOFError, OSError) as error:
            raise RuntimeError(
                f"Transfer input closed: {type(error).__name__}: "
                f"{SecretRedactor.redact(error)}; peer={self.diagnostics()}"
            ) from error
        if hasattr(self.stdin, "channel"):
            self.stdin.channel.shutdown_write()
        self.stdin.close()


@contextmanager
def subprocess_stream(command: Sequence[str]) -> Iterator[BinaryCommand]:
    """Own a child and its pipes; never buffer an archive in memory or on disk."""
    with subprocess.Popen(
        tuple(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        assert (
            process.stdin is not None and process.stdout is not None and process.stderr is not None
        )
        stream = BinaryCommand(
            cast(BinaryIO, process.stdin),
            cast(BinaryIO, process.stdout),
            cast(BinaryIO, process.stderr),
            process,
        )
        try:
            yield stream
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_error: OSError | EOFError | None = None
            try:
                process.stdin.close()
            except (EOFError, OSError) as error:
                cleanup_error = error
            if process.poll() is None:
                try:
                    # Receiver EOF must get a chance to unwind owned staging before signals.
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error
