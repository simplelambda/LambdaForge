"""Small persistent MRU list for Work YAMLs selected in the Research Console."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json


class RecentWorkStore:
    """Persist only bounded local YAML paths; never duplicate Work configuration."""

    VERSION = 1
    MAX_ITEMS = 20

    def __init__(self, path: str | Path | None = None) -> None:
        self.project = ProjectContext.discover() if path is None else None
        self.legacy_path: Path | None = None
        if path is None:
            state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            base = state_home / "lambdaforge"
            assert self.project is not None
            self.legacy_path = (base / "recent-work.json").expanduser().resolve()
            path = base / "projects" / self.project.project_id / "recent-work.json"
        self.path = Path(path).expanduser().resolve()

    def items(self, *, limit: int = 12) -> tuple[dict[str, str], ...]:
        """Return existing YAML files in most-recent-first order."""
        bounded = max(0, min(int(limit), self.MAX_ITEMS))
        if bounded == 0:
            return ()
        result: list[dict[str, str]] = []
        for item in self._read():
            if self.project is not None and not self.project.owns_source(item["path"]):
                continue
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
        if self.project is not None and not self.project.owns_source(str(selected)):
            raise ValueError("Select a Work YAML from the current project.")
        if selected.suffix.lower() not in {".yaml", ".yml"} or not selected.is_file():
            raise ValueError(f"Recent Work configuration must be an existing YAML file: {selected}")
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        with CrossProcessFileLock(
            lock,
            shared=False,
            timeout_seconds=5.0,
            poll_interval_seconds=0.05,
        ):
            previous = [
                item
                for item in self._read()
                if item["path"] != str(selected)
                and (self.project is None or self.project.owns_source(item["path"]))
            ]
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
        values = self._read_path(self.path)
        if self.legacy_path is not None:
            values.extend(self._read_path(self.legacy_path))
        result: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in values:
            if item["path"] not in seen:
                result.append(item)
                seen.add(item["path"])
        return sorted(result, key=lambda item: item["last_used_utc"], reverse=True)

    def _read_path(self, path: Path) -> list[dict[str, str]]:
        if not path.is_file() or path.is_symlink():
            return []
        try:
            value: Any = json.loads(path.read_text(encoding="utf-8"))
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
            item_path = raw.get("path")
            if not isinstance(item_path, str) or not Path(item_path).is_absolute():
                continue
            result.append(
                {
                    "path": item_path,
                    "name": str(raw.get("name") or Path(item_path).stem),
                    "last_used_utc": str(raw.get("last_used_utc") or ""),
                }
            )
        return result


__all__ = ["RecentWorkStore"]
