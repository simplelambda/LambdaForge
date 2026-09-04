"""Small persistent MRU list for Work YAMLs selected in the Research Console."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json


class RecentWorkStore:
    """Persist only bounded local YAML paths; never duplicate Work configuration."""

    VERSION = 1
    MAX_ITEMS = 20

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            path = state_home / "lambdaforge" / "recent-work.json"
        self.path = Path(path).expanduser().resolve()

    def items(self, *, limit: int = 12) -> tuple[dict[str, str], ...]:
        """Return existing YAML files in most-recent-first order."""
        bounded = max(0, min(int(limit), self.MAX_ITEMS))
        if bounded == 0:
            return ()
        result: list[dict[str, str]] = []
        for item in self._read():
            selected = Path(item["path"])
            if selected.suffix.lower() not in {".yaml", ".yml"} or not selected.is_file():
                continue
            result.append(dict(item))
            if len(result) >= bounded:
                break
        return tuple(result)

    def remember(self, path: str | Path, *, name: str | None = None) -> None:
        """Move one validated YAML to the front using an atomic bounded update."""
        selected = Path(path).expanduser().resolve()
        if selected.suffix.lower() not in {".yaml", ".yml"} or not selected.is_file():
            raise ValueError(f"Recent Work configuration must be an existing YAML file: {selected}")
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        with CrossProcessFileLock(
            lock,
            shared=False,
            timeout_seconds=5.0,
            poll_interval_seconds=0.05,
        ):
            previous = [item for item in self._read() if item["path"] != str(selected)]
            entry = {
                "path": str(selected),
                "name": str(name or selected.stem),
                "last_used_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(
                self.path,
                {
                    "version": self.VERSION,
                    "items": [entry, *previous][: self.MAX_ITEMS],
                },
            )

    def _read(self) -> list[dict[str, str]]:
        if not self.path.is_file() or self.path.is_symlink():
            return []
        try:
            value: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(value, dict) or value.get("version") != self.VERSION:
            return []
        result: list[dict[str, str]] = []
        raw_items = value.get("items", ())
        if not isinstance(raw_items, list):
            return []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            path = raw.get("path")
            if not isinstance(path, str) or not Path(path).is_absolute():
                continue
            result.append(
                {
                    "path": path,
                    "name": str(raw.get("name") or Path(path).stem),
                    "last_used_utc": str(raw.get("last_used_utc") or ""),
                }
            )
        return result


__all__ = ["RecentWorkStore"]
