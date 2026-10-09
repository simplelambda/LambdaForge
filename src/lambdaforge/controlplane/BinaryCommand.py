"""Bounded binary command streams for explicit durable data transfers."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from threading import Thread
from typing import Any, BinaryIO, cast


class BinaryCommand:
    """Drain bounded diagnostics independently of binary stdin/stdout."""

    def __init__(self, stdin: BinaryIO, stdout: BinaryIO, stderr: BinaryIO, owner: Any) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.owner = owner
        self._errors = bytearray()
        self._thread = Thread(target=self._drain, args=(stderr,), daemon=True)
        self._thread.start()

    def _drain(self, stderr: BinaryIO) -> None:
        while chunk := stderr.read(4096):
            self._errors.extend(chunk)
            if len(self._errors) > 16384:
                del self._errors[:-16384]

    def wait(self) -> None:
        code = self.owner.wait() if hasattr(self.owner, "wait") else self.owner.recv_exit_status()
        self._thread.join()
        if code:
            raise RuntimeError(
                f"Data transfer command exited with code {code}: "
                + self._errors.decode("utf-8", errors="replace")
            )

    def finish_input(self) -> None:
        """Send EOF on subprocess pipes and Paramiko channels alike."""
        try:
            self.stdin.flush()
        except BrokenPipeError:
            # Surface receiver diagnostics (e.g. disk admission), not a generic pipe error.
            self.wait()
            raise
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
            try:
                process.stdin.close()
            except OSError:
                pass
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
