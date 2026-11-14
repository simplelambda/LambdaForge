"""Durable FIFO admission tickets, separate from scientific and provider Job state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4


class ExecutionQueue:
    """One project/placement FIFO; unknown predecessors block, never imply completion."""

    LIMIT = 10000
    TERMINAL = {"succeeded", "failed", "cancelled", "timeout"}

    def __init__(
        self, state_root: str | Path, job_root: str | Path, placement: str = "local"
    ) -> None:
        if not isinstance(placement, str) or not placement or len(placement) > 256:
            raise ValueError("Invalid execution queue placement.")
        self.placement = placement
        self.parent = Path(state_root) / "execution-queue"
        self.root = self.parent / (
            "placement-" + hashlib.sha256(placement.encode()).hexdigest()[:24]
        )
        self.jobs = Path(job_root)

    @staticmethod
    def check_id(job_id: Any) -> str:
        if (
            not isinstance(job_id, str)
            or len(job_id) > 128
            or re.fullmatch(r"job-[a-z0-9-]+", job_id) is None
        ):
            raise ValueError("Unsafe execution queue Job identity.")
        return job_id

    @contextmanager
    def locked(self, *, write: bool = False) -> Iterator[dict[str, Any]]:
        if os.name != "posix":
            raise ValueError("Execution queue metadata requires a POSIX process host.")
        import fcntl

        paths = (self.root, *self.root.parents, self.jobs, *self.jobs.parents)
        if any(p.is_symlink() for p in paths):
            raise ValueError("Execution queue paths must not traverse symbolic links.")
        if (self.parent / "queue.json").exists():
            raise ValueError(
                "Unscoped development queue metadata must be reviewed before admission."
            )
        if write:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.root.exists():
            yield {"version": 1, "placement": self.placement, "entries": []}
            return
        lock = self.root / "queue.lock"
        manifest = self.root / "queue.json"
        if lock.is_symlink() or manifest.is_symlink():
            raise ValueError("Unsafe execution queue metadata.")
        if manifest.exists() and not lock.exists():
            raise ValueError("Execution queue ownership lock is missing.")
        if not write and not lock.exists():
            raise ValueError("Execution queue ownership lock is missing.")
        flags = (os.O_RDWR | os.O_CREAT) if write else os.O_RDONLY
        fd = os.open(lock, flags | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+" if write else "r") as stream:
            mode = fcntl.LOCK_EX if write else fcntl.LOCK_SH
            deadline = time.monotonic() + 10
            while True:
                try:
                    fcntl.flock(stream, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Execution queue metadata is busy; retry observation."
                        ) from None
                    time.sleep(0.05)
            if not manifest.exists():
                stream.seek(0)
                if not write or stream.read(128):
                    raise ValueError(
                        "Missing execution queue manifest; refusing to reset admission."
                    )
                # A missing manifest after an acknowledged initialization is corruption,
                # not an empty queue. A crashed first writer also stays fail-closed.
                stream.write("initialized-v1\n")
                stream.flush()
                os.fsync(stream.fileno())
            if manifest.exists() and manifest.stat().st_size > 8 * 1024**2:
                raise ValueError("Execution queue metadata exceeds its bounded size.")
            document = (
                json.loads(manifest.read_text())
                if manifest.exists()
                else {"version": 1, "placement": self.placement, "entries": []}
            )
            if (
                not isinstance(document, dict)
                or document.get("version") != 1
                or document.get("placement") != self.placement
                or not isinstance(document.get("entries"), list)
                or len(document["entries"]) > self.LIMIT
            ):
                raise ValueError("Corrupt execution queue; restore its owned admission metadata.")
            seen = set()
            for item in document["entries"]:
                if not isinstance(item, dict):
                    raise ValueError("Corrupt execution queue ticket.")
                key = self.check_id(item.get("job_id"))
                if key in seen or not isinstance(item.get("blockers"), list):
                    raise ValueError("Duplicate or corrupt execution queue ticket.")
                seen.add(key)
                for value in item["blockers"]:
                    self.check_id(value)
                if item.get("phase") not in {
                    "pending",
                    "dispatched",
                    "external",
                    "aborted",
                    "released",
                }:
                    raise ValueError("Invalid execution queue admission phase.")
            yield document
            if write:
                encoded = json.dumps(document, sort_keys=True)
                if len(encoded.encode("utf-8")) > 8 * 1024**2:
                    raise ValueError(
                        "Queue update exceeds bounded metadata; no ticket was committed."
                    )
                temporary = self.root / ("." + uuid4().hex + ".tmp")
                try:
                    with temporary.open("x") as out:
                        out.write(encoded)
                        out.flush()
                        os.fsync(out.fileno())
                    os.replace(temporary, manifest)
                    directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    temporary.unlink(missing_ok=True)

    def register(self, job_id: str, blockers: Sequence[str]) -> dict[str, Any]:
        self.check_id(job_id)
        before = sorted({self.check_id(v) for v in blockers} - {job_id})
        with self.locked(write=True) as document:
            existing = next((v for v in document["entries"] if v["job_id"] == job_id), None)
            if existing is not None:
                return dict(existing)
            if len(document["entries"]) >= self.LIMIT:
                raise ValueError("Execution queue is full; inspect its retained admission history.")
            known = {v["job_id"] for v in document["entries"]}
            if len(document["entries"]) + len(set(before) - known) >= self.LIMIT:
                raise ValueError("Execution queue predecessor inventory exceeds its bounded size.")
            document["entries"].extend(
                {"job_id": v, "blockers": [], "phase": "external"} for v in before if v not in known
            )
            item = {"job_id": job_id, "blockers": before, "phase": "pending"}
            document["entries"].append(item)
            return dict(item)

    def mark(self, job_id: str, phase: str) -> None:
        """Abort only an unstarted ticket; a dispatched Job needs native cancellation."""
        self.check_id(job_id)
        if phase not in {"dispatched", "aborted"}:
            raise ValueError("Unknown execution queue transition.")
        with self.locked(write=True) as document:
            item = next((v for v in document["entries"] if v["job_id"] == job_id), None)
            if item is None:
                raise ValueError("Missing execution queue ticket; refusing unordered execution.")
            if item["phase"] == "aborted":
                if phase == "dispatched":
                    raise InterruptedError("Execution queue ticket was cancelled before dispatch.")
                return
            if item["phase"] == "released":
                raise ValueError("Execution queue ticket has already released its admission.")
            if phase == "dispatched" and item["phase"] == "dispatched":
                raise ValueError("Execution queue ticket already has a native supervisor.")
            if phase == "aborted" and item["phase"] == "dispatched":
                raise ValueError("Dispatched queue Job requires native provider cancellation.")
            if (
                phase == "aborted"
                and item["phase"] == "external"
                and (self.jobs / job_id / "state.json").exists()
            ):
                raise ValueError("Observed predecessor requires native provider cancellation.")
            item["phase"] = phase

    def abort_preparation(self, job_id: str) -> None:
        """No-op for an untracked Job; never create a queue merely to cancel preparation."""
        self.check_id(job_id)
        if any(v["job_id"] == job_id for v in self.snapshot()["entries"]):
            self.mark(job_id, "aborted")

    def release(self, job_id: str) -> None:
        """Persist admission release only after native cleanup/observation proves it."""
        self.check_id(job_id)
        if not any(v["job_id"] == job_id for v in self.snapshot()["entries"]):
            return
        with self.locked(write=True) as document:
            item = next((v for v in document["entries"] if v["job_id"] == job_id), None)
            if item is not None and item["phase"] != "aborted":
                item["phase"] = "released"

    def waiting_on(self, job_id: str, observe: Callable[[str], Mapping[str, Any]]) -> list[str]:
        self.check_id(job_id)
        with self.locked() as document:
            entries = document["entries"]
            index = next((i for i, v in enumerate(entries) if v["job_id"] == job_id), None)
            if index is None:
                raise ValueError("Missing execution queue ticket; refusing unordered execution.")
            item = entries[index]
            if item["phase"] == "aborted":
                raise InterruptedError("Execution queue ticket cancelled before scientific start.")
            settled = {v["job_id"] for v in entries if v["phase"] in {"aborted", "released"}}
            before = list(
                dict.fromkeys([*(v["job_id"] for v in entries[:index]), *item["blockers"]])
            )
        waiting = []
        for previous in before:
            if previous in settled:
                continue
            value = observe(previous)
            if value.get("state") not in self.TERMINAL or not value.get("released", False):
                waiting.append(previous)
            else:
                self.release(previous)
        return waiting

    def snapshot(self) -> dict[str, Any]:
        with self.locked() as document:
            return json.loads(json.dumps(document))

    def retained_jobs(self) -> set[str]:
        """Only unsettled dependencies need workspaces; receipts survive their removal."""
        entries = self.snapshot()["entries"]
        settled = {v["job_id"] for v in entries if v["phase"] in {"aborted", "released"}}
        retained: set[str] = set()
        prior: set[str] = set()
        for ticket in entries:
            if ticket["job_id"] in settled:
                continue
            if ticket["phase"] != "external":
                retained.update(prior | (set(ticket["blockers"]) - settled) | {ticket["job_id"]})
            prior.add(ticket["job_id"])
        return retained

    @classmethod
    def for_storage(
        cls, state_root: str | Path, job_root: str | Path
    ) -> tuple[ExecutionQueue, ...]:
        """Inspect scoped metadata only, including queues sharing one storage mount."""
        parent = Path(state_root) / "execution-queue"
        if any(p.is_symlink() for p in (parent, *parent.parents)):
            raise ValueError("Unsafe execution queue storage root.")
        if not parent.exists():
            return ()
        if (parent / "queue.json").exists():
            raise ValueError("Unscoped development queue metadata needs operator review.")
        queues = []
        for child in sorted(parent.iterdir()):
            if not child.name.startswith("placement-"):
                continue
            manifest = child / "queue.json"
            if child.is_symlink() or manifest.is_symlink() or not manifest.is_file():
                raise ValueError("Unsafe or incomplete scoped execution queue metadata.")
            if manifest.stat().st_size > 8 * 1024**2:
                raise ValueError("Execution queue metadata exceeds its bounded size.")
            value = json.loads(manifest.read_text())
            if not isinstance(value, dict) or not isinstance(value.get("placement"), str):
                raise ValueError("Missing execution queue placement ownership.")
            queue = cls(state_root, job_root, value["placement"])
            if queue.root != child:
                raise ValueError("Execution queue placement identity mismatch.")
            queues.append(queue)
        return tuple(queues)
