"""Spawn-safe crash injector for persistent-cache recovery tests."""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

from lambdaforge.data import DiskCacheBackend


class CacheCrashJob:
    """Exit while holding the namespace lock at one deterministic write stage."""

    def __init__(
        self,
        root: str | Path,
        namespace: str,
        max_bytes: int,
        max_entries: int,
        key: str,
        payload: bytes,
        stage: str,
    ) -> None:
        if stage not in {"temporary", "after_replace"}:
            raise ValueError("stage must be temporary or after_replace.")
        self.root = Path(root)
        self.namespace = namespace
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.key = key
        self.payload = payload
        self.stage = stage

    def __call__(self) -> None:
        backend = DiskCacheBackend(
            self.root,
            self.namespace,
            max_bytes=self.max_bytes,
            max_entries=self.max_entries,
        )
        if self.stage == "temporary":
            with backend._lock, backend._file_lock(shared=False):
                backend._prepare_unlocked()
                encoded = backend.record_codec.encode(
                    self.payload,
                    associated_data=backend._associated_data(self.key),
                )
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=backend.directory,
                    prefix=f".{self.key}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os._exit(91)

        module = importlib.import_module("lambdaforge.data.cache.DiskCacheBackend")
        replace = module.os.replace

        def replace_and_exit(source: object, target: object) -> None:
            replace(source, target)
            os._exit(92)

        module.os.replace = replace_and_exit
        backend.write(self.key, self.payload)
