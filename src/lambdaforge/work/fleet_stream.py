"""Lease-fenced transport of native scalar JSONL, not another metric authority.

Offsets identify exact byte ranges in the owner's append-only files. Complete records only are
sent; a receiver publishes a verified chunk and cursor together. Retrying a read never consumes
events. The native Run logs/metrics remain authoritative on the execution host.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json

CHUNK_BYTES = 32 * 1024
CHANNELS = ("metrics", "training-metrics")


def regular_path(path: Path) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Scalar stream cannot follow a symlink.")
    if path.exists() and not path.is_file():
        raise ValueError("Scalar stream must be a regular file.")


def scalar_records(raw: bytes) -> None:
    """Validate before publishing; never reinterpret a truncated/invalid record as evidence."""
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Scalar stream chunk ends inside a record.")
    for line in raw.splitlines():
        value = json.loads(line)
        # Lightning persists this bounded native display annotation in the same JSONL.
        # Preserve its bytes/order, but never reinterpret it as an objective observation.
        if isinstance(value, dict) and value.get("kind") == "chart-filter":
            if (
                set(value) - {"kind", "include", "exclude", "display_names"}
                or any(
                    not isinstance(value.get(name), list)
                    or any(not isinstance(item, str) for item in value[name])
                    for name in ("include", "exclude")
                )
                or not isinstance(value.get("display_names", {}), dict)
                or any(
                    not isinstance(key, str) or not isinstance(label, str)
                    for key, label in value.get("display_names", {}).items()
                )
            ):
                raise ValueError("Invalid native chart annotation.")
            continue
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("name"), str)
            or not value["name"]
            or isinstance(value.get("value"), bool)
            or not isinstance(value.get("value"), int | float)
            or not math.isfinite(value["value"])
            or (
                value.get("step") is not None
                and (type(value["step"]) is not int or value["step"] < 0)
            )
            or (value.get("split") is not None and not isinstance(value["split"], str))
        ):
            raise ValueError("Invalid native scalar event.")


def read_chunk(path: Path, offset: int, *, terminal: bool) -> dict[str, Any]:
    """Read a bounded prefix without acknowledging or deleting worker evidence."""
    if type(offset) is not int or offset < 0:
        raise ValueError("Scalar stream cursor must be a non-negative byte offset.")
    regular_path(path)
    size = path.stat().st_size if path.exists() else 0
    if offset > size:
        raise ValueError("Native scalar stream shrank behind its acknowledged cursor.")
    raw = b""
    if size:
        with path.open("rb") as stream:
            if offset:
                stream.seek(offset - 1)
                if stream.read(1) != b"\n":
                    raise ValueError("Scalar stream cursor is not a record boundary.")
            stream.seek(offset)
            raw = stream.read(CHUNK_BYTES)
        if raw and b"\n" not in raw:
            if len(raw) == CHUNK_BYTES or terminal:
                raise ValueError("Native scalar record is oversized or incomplete.")
            raw = b""
        elif raw:
            raw = raw[: raw.rfind(b"\n") + 1]
    scalar_records(raw)
    if terminal and offset + len(raw) == size and size and not raw and offset != size:
        raise ValueError("Terminal scalar stream contains an incomplete record.")
    packed = zlib.compress(raw, level=3)
    compressed = len(packed) < len(raw)
    return {
        "offset": offset,
        "next_offset": offset + len(raw),
        "size": size,
        "complete": terminal and offset + len(raw) == size,
        "encoding": "zlib" if compressed else "raw",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(packed if compressed else raw).decode("ascii"),
    }


class ScalarStream:
    """Durable, exact owner-bound receiver over the existing native scalar file format."""

    def __init__(self, root: Path, identity: Mapping[str, Any]) -> None:
        self.root = root.absolute()
        self.identity = dict(identity)
        if (
            not re.fullmatch(r"sha256:[a-f0-9]{64}", str(identity.get("run_key", "")))
            or type(identity.get("attempt")) is not int
            or identity["attempt"] < 1
            or any(
                not isinstance(identity.get(name), str) or not identity[name]
                for name in ("study_identity", "lease_id", "cluster", "shard_id")
            )
        ):
            raise ValueError("Scalar stream requires an exact scientific lease identity.")
        regular_path(self.root / "stream.json")
        self.root.mkdir(parents=True, exist_ok=True)

    def _state(self) -> dict[str, Any]:
        path = self.root / "stream.json"
        regular_path(path)
        if not path.exists():
            if any(self.path(channel).exists() for channel in CHANNELS):
                raise ValueError("Scalar cursor is missing; refusing a guessed reset.")
            return {"stream_version": 1, "identity": self.identity, "channels": {}}
        value = json.loads(path.read_text())
        if (
            not isinstance(value, dict)
            or value.get("stream_version") != 1
            or value.get("identity") != self.identity
            or not isinstance(value.get("channels"), dict)
            or set(value["channels"]) - set(CHANNELS)
            or any(not isinstance(item, dict) for item in value["channels"].values())
        ):
            raise ValueError("Scalar stream identity/version mismatch.")
        return dict(value)

    def path(self, channel: str) -> Path:
        if channel not in CHANNELS:
            raise ValueError("Unknown native scalar channel.")
        path = self.root / (channel + ".jsonl")
        regular_path(path)
        return path

    def cursors(self) -> dict[str, int]:
        with self._lock():
            state = self._state()
            return {channel: self._cursor(state, channel) for channel in CHANNELS}

    def _cursor(self, state: Mapping[str, Any], channel: str) -> int:
        raw = state["channels"].get(channel, {}).get("offset", 0)
        if type(raw) is not int or raw < 0:
            raise ValueError("Corrupt persisted scalar cursor.")
        path = self.path(channel)
        size = path.stat().st_size if path.exists() else 0
        if size < raw:
            raise ValueError("Persisted scalar bytes disagree with their durable cursor.")
        if size > raw:
            # Bytes beyond the commit point were never acknowledged. Recover an interrupted
            # append only in this exact lease's owned mirror, not on the execution host.
            with path.open("r+b") as stream:
                stream.truncate(raw)
                stream.flush()
                os.fsync(stream.fileno())
        return raw

    def _lock(self) -> CrossProcessFileLock:
        return CrossProcessFileLock(
            self.root / ".lock", shared=False, timeout_seconds=5, poll_interval_seconds=0.05
        )

    def ingest(self, payload: Mapping[str, Any]) -> bool:
        if payload.get("identity") != self.identity or payload.get("stream_version") != 1:
            raise ValueError("Received scalar stream belongs to another lease.")
        chunks = payload.get("channels")
        if not isinstance(chunks, Mapping) or set(chunks) != set(CHANNELS):
            raise ValueError("Received scalar channels are incomplete.")
        with self._lock():
            state = self._state()
            decoded: dict[str, tuple[Mapping[str, Any], bytes]] = {}
            for channel, chunk in chunks.items():
                if not isinstance(chunk, Mapping):
                    raise ValueError("Invalid scalar chunk.")
                if not isinstance(chunk.get("data"), str) or len(chunk["data"]) > 4 * (
                    (CHUNK_BYTES + 2) // 3
                ):
                    raise ValueError("Scalar chunk exceeds transport limit.")
                packed = base64.b64decode(chunk["data"], validate=True)
                if len(packed) > CHUNK_BYTES:
                    raise ValueError("Scalar chunk exceeds transport limit.")
                if chunk["encoding"] == "zlib":
                    decoder = zlib.decompressobj()
                    raw = decoder.decompress(packed, CHUNK_BYTES + 1)
                    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                        raise ValueError("Invalid or oversized compressed scalar chunk.")
                elif chunk["encoding"] == "raw":
                    raw = packed
                else:
                    raise ValueError("Unsupported scalar compression.")
                if len(raw) > CHUNK_BYTES or hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
                    raise ValueError("Scalar chunk digest/size mismatch.")
                scalar_records(raw)
                offset, end, size = (chunk[name] for name in ("offset", "next_offset", "size"))
                if (
                    any(type(v) is not int or v < 0 for v in (offset, end, size))
                    or end != offset + len(raw)
                    or end > size
                    or type(chunk.get("complete")) is not bool
                    or (chunk["complete"] and end != size)
                ):
                    raise ValueError("Scalar chunk has invalid accounting.")
                cursor = self._cursor(state, channel)
                if (
                    offset == cursor
                    and state["channels"].get(channel, {}).get("complete", False)
                    and (size != cursor or not chunk["complete"])
                ):
                    raise ValueError("A completed scalar stream cannot change.")
                if offset > cursor or (offset < cursor and end > cursor):
                    raise ValueError("Scalar chunk skips or overlaps acknowledged evidence.")
                if offset < cursor:
                    with self.path(channel).open("rb") as stream:
                        stream.seek(offset)
                        if stream.read(len(raw)) != raw:
                            raise ValueError("Contradictory scalar replay.")
                decoded[channel] = (chunk, raw)
            # The cursor is the transaction commit point. Initialize it before the first
            # append, so recovery can discard an unacknowledged suffix after interruption.
            atomic_write_json(self.root / "stream.json", state)
            for channel, (chunk, raw) in decoded.items():
                cursor = self._cursor(state, channel)
                if chunk["offset"] == cursor:
                    path = self.path(channel)
                    with path.open("ab") as stream:
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                    state["channels"][channel] = {
                        "offset": chunk["next_offset"],
                        "complete": chunk["complete"],
                    }
            atomic_write_json(self.root / "stream.json", state)
            return all(
                state["channels"].get(channel, {}).get("complete", False) for channel in CHANNELS
            )
