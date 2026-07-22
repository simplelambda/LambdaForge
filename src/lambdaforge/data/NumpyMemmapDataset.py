"""Safe read-only memory-mapped dataset for numeric NumPy arrays."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyMemmapDataset(Dataset[dict[str, Any]]):
    """Expose aligned .npy arrays as copied mapping-shaped samples.

    Files are opened lazily with mmap_mode="r" and allow_pickle=False. Every
    array must have the same non-scalar leading dimension. Returned samples
    are always copied away from the read-only mapping, so callers may mutate
    tensors or arrays without changing the file or later samples.

    Mappings and locks are process-local. Pickling for spawn drops every open
    handle, while a PID change after fork closes inherited mappings and
    reopens them in the child. Call close for deterministic file release,
    particularly before replacing or deleting files on Windows.

    Parameters
    ----------
    arrays:
        Non-empty mapping from output keys to numeric .npy files.
    as_tensors:
        Convert each writable sample copy to a PyTorch tensor. If false,
        writable NumPy arrays are returned instead.
    """

    def __init__(
        self,
        arrays: Mapping[str, str | Path],
        as_tensors: bool = True,
    ) -> None:
        if not isinstance(arrays, Mapping):
            raise TypeError("arrays must map output names to .npy paths.")
        if not arrays:
            raise ValueError("arrays cannot be empty.")
        if not isinstance(as_tensors, bool):
            raise TypeError("as_tensors must be a bool.")

        paths: dict[str, Path] = {}
        for name, value in arrays.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Every array name must be a non-empty string.")
            path = Path(value).expanduser().resolve()
            if path.suffix.lower() != ".npy":
                raise ValueError(f"NumpyMemmapDataset only accepts .npy files: {path}")
            paths[name] = path

        self.paths = paths
        self.as_tensors = as_tensors
        self._mapped_arrays: dict[str, np.memmap] = {}
        self._length: int | None = None
        self._owner_pid: int | None = None
        self._lock = threading.RLock()

    def __len__(self) -> int:
        """Open headers/mappings on first use and return the shared length."""
        self._ensure_open()
        if self._length is None:
            raise RuntimeError("Memory-mapped arrays opened without a length.")
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return independent writable copies of every aligned array row."""
        self._ensure_open()
        with self._lock:
            sample = {
                name: np.array(array[index], copy=True)
                for name, array in self._mapped_arrays.items()
            }
        if not self.as_tensors:
            return sample
        try:
            return {name: torch.from_numpy(value) for name, value in sample.items()}
        except (TypeError, ValueError) as error:
            raise TypeError(
                "as_tensors=True requires NumPy dtypes supported by torch.from_numpy."
            ) from error

    @property
    def is_open(self) -> bool:
        """Report whether this process currently owns live mappings."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            return False
        with lock:
            return self._owner_pid == os.getpid() and bool(self._mapped_arrays)

    def close(self) -> None:
        """Close every process-local mapping; repeated calls are harmless."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            self._close_mappings(self._mapped_arrays)
            self._mapped_arrays = {}
            self._owner_pid = None

    def __enter__(self) -> NumpyMemmapDataset:
        """Return this dataset for deterministic context-managed cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close all mappings when leaving a context."""
        del exc_type, exc, traceback
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize configuration without mmap handles or thread locks."""
        with self._lock:
            state = dict(self.__dict__)
        state["_mapped_arrays"] = {}
        state["_owner_pid"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore an unopened process-local dataset after spawn."""
        self.__dict__.update(state)
        self._mapped_arrays = {}
        self._owner_pid = None
        self._lock = threading.RLock()

    def __del__(self) -> None:
        """Best-effort fallback for callers that do not close explicitly."""
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        pid = os.getpid()
        with self._lock:
            if self._owner_pid == pid and self._mapped_arrays:
                return

            self._close_mappings(self._mapped_arrays)
            opened: dict[str, np.memmap] = {}
            expected_length: int | None = None
            try:
                for name, path in self.paths.items():
                    try:
                        array = np.load(path, mmap_mode="r", allow_pickle=False)
                    except ValueError as error:
                        raise ValueError(
                            f"Cannot memory-map {path}; object arrays and pickled data are refused."
                        ) from error
                    if not isinstance(array, np.memmap):
                        raise TypeError(f"Expected a memory-mapped .npy array: {path}")
                    opened[name] = array
                    if array.ndim == 0:
                        raise ValueError(
                            f"Memory-mapped dataset array must have a sample axis: {path}"
                        )
                    if array.dtype.hasobject:
                        raise ValueError(
                            f"Object dtype is not allowed in memory-mapped data: {path}"
                        )
                    length = int(array.shape[0])
                    if expected_length is None:
                        expected_length = length
                    elif length != expected_length:
                        raise ValueError(
                            f"Array {name!r} has {length} samples; expected {expected_length}."
                        )
            except Exception:
                self._close_mappings(opened)
                self._mapped_arrays = {}
                self._owner_pid = None
                raise

            self._mapped_arrays = opened
            self._length = expected_length
            self._owner_pid = pid

    @staticmethod
    def _close_mappings(arrays: Mapping[str, np.memmap]) -> None:
        for array in arrays.values():
            mapped_file = getattr(array, "_mmap", None)
            if mapped_file is not None:
                mapped_file.close()
