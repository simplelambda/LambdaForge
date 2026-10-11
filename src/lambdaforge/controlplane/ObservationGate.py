"""Cross-process coalescing of native provider probes, not a second observation cache."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.OperationalHealth import timestamp
from lambdaforge.controlplane.OperationalIncidents import read_document, write_document
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class ObservationGate:
    """Serialize one project's cluster observations with a bounded nonblocking wait.

    A concurrent caller retains native last-good Job records, including their original clocks.
    Receipt times never certify the provider. Process death releases the OS lock automatically;
    completed probes commit their actual observations before writing a compact receipt.
    """

    def __init__(self, root: Path, cluster: str, *, scope: str = "inventory") -> None:
        identifier = hashlib.sha256(cluster.encode("utf-8")).hexdigest()
        directory = root / ".operational" / "probes"
        self.path = directory / f"{identifier}.json"
        self.lock = CrossProcessFileLock(
            directory / f"{identifier}.lock",
            shared=False,
            timeout_seconds=0.05,
            poll_interval_seconds=0.01,
        )
        self.requested_at = datetime.now(timezone.utc)
        self.cluster = cluster
        self.scope = scope
        self.probe = False

    def __enter__(self) -> bool:
        """False means reuse native facts without changing freshness or failure counters."""
        if self.path.is_symlink() or self.path.resolve() != self.path.absolute():
            raise ValueError(f"Unsafe provider observation gate: {self.path}")
        try:
            self.lock.acquire()
        except TimeoutError:
            return False
        try:
            last = read_document(self.path)
            finished = timestamp(last.get("completed_at_utc"))
            if (
                finished is not None
                and finished >= self.requested_at
                and last.get("scope") == self.scope
            ):
                self.lock.release()
                return False
            self.probe = True
            return True
        except BaseException:
            self.lock.release()
            raise

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> None:
        """No receipt claims success; only successful commits permit request coalescing."""
        if not self.probe:
            return
        try:
            if error_type is None:
                write_document(
                    self.path,
                    {
                        "probe_receipt_version": 1,
                        "cluster": self.cluster,
                        "scope": self.scope,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
        finally:
            self.lock.release()
