"""Bounded descriptive tables of retained scalar evidence, not another result authority."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.agents.contracts import redact

PROVENANCE_COLUMNS = (
    "execution_id",
    "scientific_fingerprint",
    "run_id",
    "attempt_id",
    "attempt_number",
    "status",
    "evidence_kind",
    "metric",
)
DEFAULT_COLUMNS = (*PROVENANCE_COLUMNS, "name", "trial", "seed", "value")
OPTIONAL_COLUMNS = ("parameters", "duration_seconds", "work_class", "termination_type")
NOTICE = (
    "Derived descriptive table; native Execution/Run/Attempt records remain authoritative. "
    "Values are stored scalar snapshots, not inferred best/final/selection scores. "
    "Partial or censored observations are not completed evidence. No statistical refit."
)


def result_table(
    selected: Mapping[str, Any],
    *,
    metrics: Sequence[str] = (),
    runs: Sequence[str] = (),
    trials: Sequence[int] = (),
    columns: Sequence[str] = (),
    attempts: str = "latest",
    offset: int = 0,
    limit: int = 100,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    """Page one long-form table without opening metric series, artifact or checkpoint bytes."""
    if attempts not in {"latest", "all"}:
        raise ValueError("Table Attempts must be latest or all.")
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Table offset must be non-negative and limit must be 1..1000.")
    allowed = set(DEFAULT_COLUMNS + OPTIONAL_COLUMNS)
    if set(columns) - allowed:
        raise ValueError(f"Unknown table columns: {sorted(set(columns) - allowed)}.")
    names = tuple(dict.fromkeys((*PROVENANCE_COLUMNS, *(columns or DEFAULT_COLUMNS))))
    records = selected.get("runs", ())
    if not isinstance(records, list | tuple) or any(
        not isinstance(row, Mapping) for row in records
    ):
        raise ValueError("Corrupt native Run evidence for result table.")
    latest: dict[str, Mapping[str, Any]] = {}
    keys = set()
    for row in records:
        run_id, number = row.get("run_id"), row.get("attempt_number")
        if (
            not isinstance(run_id, str)
            or type(number) is not int
            or number < 1
            or not isinstance(row.get("attempt_id"), str)
            or row.get("execution_id") != selected.get("execution_id")
        ):
            raise ValueError("Result table requires exact native Run/Attempt identities.")
        key = (run_id, number)
        if key in keys:
            raise ValueError("Duplicate native Run/Attempt evidence; cannot export a table.")
        keys.add(key)
        if run_id not in latest or number > latest[run_id]["attempt_number"]:
            latest[run_id] = row
    if set(runs) - set(latest):
        raise KeyError(f"Unknown selected Run IDs: {sorted(set(runs) - set(latest))}.")
    observed_trials = {(row.get("trial") or {}).get("index") for row in records}
    if set(trials) - observed_trials:
        raise KeyError(f"Unknown selected Trials: {sorted(set(trials) - observed_trials)}.")
    records = [
        row
        for row in (latest.values() if attempts == "latest" else records)
        if (not runs or row["run_id"] in runs)
        and (not trials or (row.get("trial") or {}).get("index") in trials)
    ]
    records.sort(key=lambda row: (row["run_id"], row["attempt_number"]))
    known_metrics: set[str] = set()
    for row in records:
        scalar = row.get("metrics", {})
        if not isinstance(scalar, Mapping):
            raise ValueError("Corrupt native scalar metric mapping.")
        if any(not isinstance(name, str) for name in scalar):
            raise ValueError("Native metric names must be exact strings.")
        known_metrics.update(scalar)
    if set(metrics) - known_metrics:
        raise KeyError(
            f"Metrics absent from selected Run evidence: {sorted(set(metrics) - known_metrics)}."
        )
    metric_names: Sequence[str | None] = (
        tuple(dict.fromkeys(metrics)) or tuple(sorted(known_metrics)) or (None,)
    )
    evidence_hash = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {
                    "execution_id": selected.get("execution_id"),
                    "scientific_fingerprint": selected.get("scientific_fingerprint"),
                    "records": records,
                },
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )
    if fingerprint is not None and fingerprint != evidence_hash:
        raise ValueError("Result evidence changed between table pages; restart from offset 0.")
    total = len(records) * len(metric_names)
    if offset > total:
        raise ValueError("Table offset exceeds available rows.")
    rows = []
    used_bytes = 0
    for index in range(offset, min(total, offset + limit)):
        source = records[index // len(metric_names)]
        metric = metric_names[index % len(metric_names)]
        value = source.get("metrics", {}).get(metric) if metric is not None else None
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Non-finite or non-numeric native scalar metric evidence.")
        censored = (
            bool(source.get("pruned")) or source.get("termination_type") == "performance_pruned"
        )
        row = {
            **{name: source.get(name) for name in names},
            "execution_id": selected["execution_id"],
            "scientific_fingerprint": source.get("scientific_fingerprint"),
            "evidence_kind": "censored_snapshot"
            if censored
            else "completed_snapshot"
            if source.get("status") == "succeeded"
            else "partial_snapshot",
            "metric": metric,
            "trial": (source.get("trial") or {}).get("index"),
            "value": value,
        }
        row = {name: redact(row.get(name)) for name in names}
        # Redact the whole mapping too: key names in authored parameter objects matter.
        row = redact(row)
        size = len(json.dumps(row, ensure_ascii=False, allow_nan=False).encode())
        if size > 480 * 1024:
            raise ValueError("One result table row exceeds 480 KiB; omit bulky optional columns.")
        if used_bytes + size > 480 * 1024:
            break
        rows.append(row)
        used_bytes += size
    return {
        "result_table_version": 1,
        "notice": NOTICE,
        "source": {
            "execution_id": selected["execution_id"],
            "scientific_fingerprint": selected.get("scientific_fingerprint"),
            "status": selected.get("status"),
            "imported": bool(selected.get("imported")),
        },
        "evidence_fingerprint": evidence_hash,
        "columns": list(names),
        "rows": rows,
        "offset": offset,
        "total": total,
        "next_offset": offset + len(rows) if offset + len(rows) < total else None,
    }
