"""Filesystem catalog for canonical results and immutable attempt history."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from lambdaforge.experiments.results.ResultRecord import ResultRecord
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class ResultCatalog:
    """Discover, index and safely select experiment attempts below one root."""

    INDEX_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def index_path(self) -> Path:
        """Return the location of the reproducible catalog snapshot."""
        return self.root / ".lambdaforge" / "result-index.json"

    def records(
        self,
        *,
        status: str | None = None,
        fingerprint: str | None = None,
        include_archived: bool = True,
    ) -> tuple[ResultRecord, ...]:
        """Scan disk and return deterministically ordered matching attempts."""
        discovered: list[ResultRecord] = []
        if not self.root.exists():
            return ()
        for path in sorted(self.root.rglob("result.json")):
            record = self._record(path, archived=False)
            if record is not None:
                discovered.append(record)
        if include_archived:
            for path in sorted(self.root.rglob(".lambdaforge/attempts/result-*.json")):
                record = self._record(path, archived=True)
                if record is not None:
                    discovered.append(record)
        filtered = (
            record
            for record in discovered
            if (status is None or record.status == status)
            and (fingerprint is None or record.config_fingerprint == fingerprint)
        )
        return tuple(sorted(filtered, key=self._sort_key))

    def duplicate_groups(self) -> dict[str, tuple[ResultRecord, ...]]:
        """Return fingerprints represented by more than one persisted attempt."""
        return self._duplicate_groups(self.records())

    @staticmethod
    def _duplicate_groups(
        records: tuple[ResultRecord, ...],
    ) -> dict[str, tuple[ResultRecord, ...]]:
        """Group one consistent catalog snapshot by scientific identity."""
        grouped: dict[str, list[ResultRecord]] = defaultdict(list)
        for record in records:
            if record.config_fingerprint is not None:
                grouped[record.config_fingerprint].append(record)
        return {
            fingerprint: tuple(records)
            for fingerprint, records in sorted(grouped.items())
            if len(records) > 1
        }

    def ambiguous_successes(self) -> dict[str, tuple[ResultRecord, ...]]:
        """Return identities with multiple successful attempts requiring selection."""
        return {
            fingerprint: tuple(record for record in records if record.status == "ok")
            for fingerprint, records in self.duplicate_groups().items()
            if sum(record.status == "ok" for record in records) > 1
        }

    def select(self, *, attempt_id: str) -> ResultRecord:
        """Select exactly one attempt by its stable identifier or fail loudly."""
        matches = [record for record in self.records() if record.attempt_id == attempt_id]
        if len(matches) != 1:
            raise LookupError(
                f"Expected exactly one result for attempt_id={attempt_id!r}, found {len(matches)}."
            )
        return matches[0]

    def write_index(self) -> Path:
        """Atomically publish a locked snapshot for humans and external agents."""
        records = self.records()
        duplicate_groups = self._duplicate_groups(records)
        ambiguous = {
            fingerprint: tuple(record for record in group if record.status == "ok")
            for fingerprint, group in duplicate_groups.items()
            if sum(record.status == "ok" for record in group) > 1
        }
        payload: dict[str, Any] = {
            "index_version": self.INDEX_VERSION,
            "root": str(self.root),
            "records": [record.to_dict() for record in records],
            "duplicate_fingerprints": sorted(duplicate_groups),
            "ambiguous_success_fingerprints": sorted(ambiguous),
        }
        metadata_dir = self.index_path.parent
        metadata_dir.mkdir(parents=True, exist_ok=True)
        lock = CrossProcessFileLock(
            metadata_dir / "result-index.lock",
            shared=False,
            timeout_seconds=30.0,
            poll_interval_seconds=0.05,
        )
        with lock:
            temporary = self.index_path.with_name(
                f".{self.index_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
            )
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.index_path)
            finally:
                temporary.unlink(missing_ok=True)
        return self.index_path

    def summary(self) -> str:
        """Render a concise human-readable audit of the result tree."""
        records = self.records()
        statuses: dict[str, int] = defaultdict(int)
        for record in records:
            statuses[record.status] += 1
        status_text = ", ".join(f"{key}={value}" for key, value in sorted(statuses.items()))
        return (
            f"LambdaForge results: attempts={len(records)}"
            f"{', ' + status_text if status_text else ''}; "
            f"duplicate_identities={len(self.duplicate_groups())}; "
            f"ambiguous_successes={len(self.ambiguous_successes())}."
        )

    def _record(self, path: Path, *, archived: bool) -> ResultRecord | None:
        try:
            result = RunResult.read_json(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        run_dir = path.parents[2] if archived else path.parent
        fingerprint = result.config_fingerprint or self._config_fingerprint(run_dir)
        attempt_id = result.attempt_id or path.stem
        return ResultRecord(
            result=result,
            result_path=path,
            run_dir=run_dir,
            archived=archived,
            config_fingerprint=fingerprint,
            attempt_id=attempt_id,
        )

    @staticmethod
    def _config_fingerprint(run_dir: Path) -> str | None:
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            return None
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None
        return RunFingerprint.digest(payload) if isinstance(payload, dict) else None

    @staticmethod
    def _sort_key(record: ResultRecord) -> tuple[str, str, str]:
        timestamp = record.result.finished_at_utc or record.result.started_at_utc or ""
        return (record.run_dir, timestamp, record.attempt_id)
