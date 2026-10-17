"""Bound operational samples, preserving cumulative peaks; never touch scientific evidence."""

from __future__ import annotations

import gzip
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.work.atomic import atomic_write_json, atomic_write_text

_PEAKS = ("rss_bytes", "gpu_memory_bytes", "process_count", "threads", "cpu_percent")


def record_usage(
    job_dir: Path,
    sample: Mapping[str, Any],
    *,
    limit_bytes: int = 2 * 1024**2,
    retained_samples: int = 256,
) -> dict[str, Any]:
    """Existing supervisor writer; rotate four compressed diagnostic windows and retain peaks.

    Operational samples are not complete Run trajectories or scientific cost/refund state.
    """
    if not 1024 <= limit_bytes <= 8 * 1024**2 or not 1 <= retained_samples <= 1024:
        raise ValueError("Operational history limits are outside bounded policy.")
    paths = [
        job_dir / "usage.jsonl",
        job_dir / "usage-summary.json",
        *(job_dir / f"usage-window-{index}.jsonl.gz" for index in range(4)),
    ]
    if job_dir.resolve() != job_dir.absolute() or any(path.is_symlink() for path in paths):
        raise ValueError("Operational resource history must use owned non-symbolic paths.")
    history, summary_path = paths[:2]
    summary: dict[str, Any] = {}
    if summary_path.exists():
        if summary_path.stat().st_size > 65536:
            raise ValueError("Resource usage summary is oversized.")
        summary = json.loads(summary_path.read_text())
    summary.setdefault("version", 1)
    summary.setdefault("first_sample_at_utc", sample.get("timestamp_utc"))
    summary["last_sample_at_utc"] = sample.get("timestamp_utc")
    summary["sample_count"] = int(summary.get("sample_count", 0)) + 1
    summary.setdefault("compacted_windows", 0)
    peaks = summary.setdefault("lifetime_sampled_peaks", {})
    for key in _PEAKS:
        value = sample.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= value < 2**63
            and math.isfinite(value)
        ):
            peaks[key] = max(value, peaks.get(key, 0))
    encoded = json.dumps(dict(sample), allow_nan=False, sort_keys=True) + "\n"
    if len(encoded.encode()) > 65536:
        raise ValueError("Operational usage sample exceeds 64 KiB.")
    fd = os.open(
        history,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        0o600,
    )
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or os.name == "posix"
            and info.st_uid != os.getuid()
        ):
            raise ValueError("Unsafe resource usage history file.")
        if info.st_size > 8 * 1024**2:
            raise ValueError(
                "Legacy resource history exceeds bounded read; "
                "new samples remain in state metadata."
            )
        stream.write(encoded)
    if history.stat().st_size > limit_bytes:
        if history.stat().st_size > 8 * 1024**2 + 65536:
            raise ValueError(
                "Legacy resource usage history needs scoped cleanup; refusing unbounded read."
            )
        raw = history.read_bytes()
        slot = int(summary["compacted_windows"]) % 4
        archive = paths[2 + slot]
        # A crash leftover must not permanently disable future bounded compaction.
        # Never adopt or remove another temporary file; only clean the one created here.
        fd, name = tempfile.mkstemp(prefix=archive.name + ".", suffix=".tmp", dir=job_dir)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as output:
                with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
                    compressed.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
        retained = raw.splitlines()[-retained_samples:]
        while retained and sum(len(line) + 1 for line in retained) > limit_bytes // 4:
            retained.pop(0)
        atomic_write_text(history, b"\n".join(retained).decode() + ("\n" if retained else ""))
        summary["compacted_windows"] += 1
        summary["complete_timeline"] = False
    else:
        summary.setdefault("complete_timeline", True)
    atomic_write_json(summary_path, summary)
    return summary
