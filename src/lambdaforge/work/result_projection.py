"""Read-only, stdlib-only projections of native Work evidence.

This is a presentation adapter, not a store or an execution mechanism. Only an explicitly
owned Execution and its recorded Run/Attempt identities may be read. Scalar history and logs
are selected-Attempt-only; panel projections never read artifact bytes. The explicit small-output
preview is separately bounded and checksum-verified.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def owned(root: Path, relative: str) -> Path:
    """Reject traversal and symlinks, including missing paths below symbolic parents."""
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("Unsafe Work evidence path.")
    selected = root / path
    if selected.resolve() != selected or not selected.is_relative_to(root):
        raise ValueError("Work evidence cannot contain symlinks or leave its owner.")
    return selected


def read_mapping(path: Path, *, aggregate: bool = False) -> dict[str, Any]:
    if path.resolve() != path or not path.is_file():
        raise ValueError(f"Missing or symbolic Work record: {path}")
    if not aggregate and path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Work metadata exceeds the individual-document safety bound.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Work record must be a JSON object.")
    return value


def attempt_path(root: Path, run: Mapping[str, Any]) -> Path:
    """Relocate by attested native identity, never by stale remote paths or filenames."""
    run_id, attempt_id = run.get("run_id"), run.get("attempt_id")
    if (
        not isinstance(run_id, str)
        or not re.fullmatch(r"run-[A-Za-z0-9_-]+", run_id)
        or not isinstance(attempt_id, str)
        or not re.fullmatch(r"attempt-\d{4}", attempt_id)
        or type(run.get("attempt_number")) is not int
        or run["attempt_number"] != int(attempt_id[8:])
    ):
        raise ValueError("Invalid Work Run/Attempt identity.")
    if run.get("execution_id") != root.name:
        # Portable packages use execution/ as their owned root. Its immutable origin is
        # authoritative; a folder name in a downloaded package is not a scientific identity.
        origin = read_mapping(owned(root, "execution.json"))
        if run.get("execution_id") != origin.get("execution_id"):
            raise ValueError("Work Attempt belongs to another Execution.")
    expected = ("runs", run_id, "attempts", attempt_id)
    if Path(str(run.get("run_dir", ""))).parts[-4:] != expected:
        raise ValueError("Work Attempt path disagrees with its native identity.")
    return owned(root, "/".join(expected))


def is_study(configuration: Mapping[str, Any]) -> bool:
    """Declared design, not metrics or training class names, determines Study presentation."""
    if any(key in configuration for key in ("search", "sweep", "replicates")):
        return True
    seeds = configuration.get("seeds")
    if isinstance(seeds, list) and len(seeds) > 1:
        return True
    return any(
        is_study(child)
        for step in configuration.get("steps", ())
        if isinstance(step, Mapping)
        for child in step.get("parallel", [step])
        if isinstance(child, Mapping)
    )


def curves(root: Path, *, points: int = 200) -> dict[str, Any]:
    """Stream registered scalar files, retaining bounded curves without loading raw history."""
    if type(points) is not int or not 10 <= points <= 500:
        raise ValueError("Curve points must be between 10 and 500.")
    series: dict[str, list[dict[str, Any]]] = {}
    latest: dict[str, Any] = {}
    counts: dict[str, int] = {}
    metadata: dict[str, Any] = {}
    for filename in ("metrics.jsonl", "training-metrics.jsonl"):
        path = owned(root, filename)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            while line := stream.readline(1024 * 1024 + 1):
                if len(line) > 1024 * 1024:
                    raise ValueError("Scalar observation exceeds the metadata safety bound.")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # An in-flight final JSONL line is not fabricated evidence.
                if not isinstance(row, Mapping):
                    continue
                if row.get("kind") == "chart-filter":
                    metadata = dict(row)
                    continue
                value = row.get("value")
                if isinstance(value, bool) or not isinstance(value, int | float):
                    continue
                if not math.isfinite(value) or not isinstance(row.get("name"), str):
                    continue
                name = f"{row['split']}_{row['name']}" if row.get("split") else row["name"]
                counts[name] = counts.get(name, 0) + 1
                step = row.get("step")
                step = step if type(step) is int else counts[name]
                sample = {"step": step, "value": value, "timestamp": row.get("timestamp_utc")}
                latest[name] = sample
                values = series.setdefault(name, [])
                values.append(sample)
                if len(values) > points * 2:
                    values[:] = values[::2]
    for name, values in series.items():
        if values[-1] != latest[name]:
            values.append(latest[name])
        if len(values) > points:
            indices = [round(i * (len(values) - 1) / (points - 1)) for i in range(points)]
            series[name] = [values[i] for i in indices]
    return {"curves": series, "latest": latest, "observations": counts, "chart_filter": metadata}


def log_tail(path: Path, *, lines: int = 200) -> str:
    if type(lines) is not int or not 1 <= lines <= 2000:
        raise ValueError("Log tail must be between 1 and 2000 lines.")
    if not path.exists():
        return ""
    if not path.is_file() or path.resolve() != path:
        raise ValueError("Unsafe Work log.")
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 1024 * 1024))
        return "\n".join(
            stream.read(1024 * 1024).decode("utf-8", errors="replace").splitlines()[-lines:]
        )


def projection(
    root: Path,
    result: Mapping[str, Any],
    *,
    view: str = "overview",
    run_id: str | None = None,
    attempt: int | None = None,
    tail: int = 200,
    points: int = 200,
) -> dict[str, Any]:
    """Project one Execution or one exact Attempt; no consumer imports, refits or mutations."""
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Work projection requires an owned non-symbolic Execution.")
    if view not in {"overview", "attempt", "metrics", "logs", "outputs", "provenance", "html"}:
        raise ValueError("Unknown Work result view.")
    origin = read_mapping(owned(root, "execution.json"))
    revision_path = owned(root, "current-code.json")
    revision = read_mapping(revision_path) if revision_path.exists() else origin
    if result.get("execution_id") != origin.get("execution_id") or result.get(
        "scientific_fingerprint"
    ) != revision.get("scientific_fingerprint"):
        raise ValueError("Work result differs from its owned Execution identity.")
    rows = result.get("runs", ())
    if not isinstance(rows, list | tuple) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Invalid Work Run records.")
    configuration = read_mapping(owned(root, "configuration.json"))
    header: dict[str, Any] = {
        "work_result_view_version": 1,
        "name": result.get("name"),
        "execution_id": result.get("execution_id"),
        "scientific_fingerprint": result.get("scientific_fingerprint"),
        "status": result.get("status"),
        "created_at_utc": origin.get("created_at_utc"),
        "cluster": origin.get("cluster"),
        "study_expected": is_study(configuration),
        "lifecycle": result.get("lifecycle"),
        "summary": result.get("summary", {}),
        "attempts": [
            {
                key: row.get(key)
                for key in (
                    "name",
                    "run_id",
                    "attempt_id",
                    "attempt_number",
                    "status",
                    "seed",
                    "started_at_utc",
                    "finished_at_utc",
                    "duration_seconds",
                    "work_class",
                )
            }
            for row in rows
        ],
    }
    if view == "overview":
        header["configuration"] = configuration
        publication = owned(root, "products.json")
        header["products"] = read_mapping(publication) if publication.exists() else None
        return header
    eligible = [row for row in rows if run_id is None or row["run_id"] == run_id]
    if run_id is None and len({row["run_id"] for row in eligible}) > 1:
        raise ValueError("Select a Run explicitly; evidence from different Runs is never merged.")
    if attempt is not None:
        eligible = [row for row in eligible if row["attempt_number"] == attempt]
    if not eligible:
        raise KeyError("No persisted evidence for the selected Run/Attempt yet.")
    selected = max(eligible, key=lambda row: row["attempt_number"])
    directory = attempt_path(root, selected)
    native_path = owned(directory, "result.json")
    if native_path.exists():
        native = read_mapping(native_path)
        for field in (
            "execution_id",
            "run_id",
            "attempt_id",
            "attempt_number",
            "seed",
            "scientific_fingerprint",
            "parameters",
            "outputs",
            "artifacts",
            "metrics",
            "result",
            "status",
        ):
            if selected.get(field) != native.get(field):
                raise ValueError(f"Work Attempt evidence disagrees with its aggregate: {field}.")
    selected_fields = (
        "name",
        "execution_id",
        "run_id",
        "attempt_id",
        "attempt_number",
        "seed",
        "status",
        "scientific_fingerprint",
        "work_class",
        "started_at_utc",
        "finished_at_utc",
        "duration_seconds",
        "failure",
    )
    header["selected"] = (
        dict(selected) if view == "attempt" else {key: selected.get(key) for key in selected_fields}
    )
    header["selected"]["run_dir"] = str(directory)
    if view in {"metrics", "attempt"}:
        header.update(curves(directory, points=points))
    if view in {"logs", "attempt"}:
        header["log"] = log_tail(
            owned(directory, str(selected.get("logs") or "work.log")), lines=tail
        )
    if view in {"outputs", "attempt"}:
        header["outputs"] = dict(selected.get("outputs") or {})
        header["result"] = selected.get("result")
        header["datasets"] = dict(selected.get("datasets") or {})
        artifacts = []
        for artifact in selected.get("artifacts", ()):
            path = owned(directory, str(artifact.get("path", "")))
            artifacts.append(
                {
                    **artifact,
                    "location": str(path),
                    "availability": "retained" if path.exists() else "not_retained",
                }
            )
        header["artifacts"] = artifacts
        checkpoint_root = owned(root, f"runs/{selected['run_id']}/checkpoints")
        header["checkpoints"] = (
            [
                {
                    "name": path.relative_to(checkpoint_root).as_posix(),
                    "location": str(path),
                    "size_bytes": path.stat().st_size,
                    "kind": "checkpoint",
                    "availability": "retained",
                    "scope": (
                        "Current logical Run state shared across Attempts; "
                        "not a historical Attempt snapshot"
                    ),
                }
                for path in sorted(checkpoint_root.rglob("*"))
                if path.is_file() and path.resolve() == path
            ]
            if checkpoint_root.exists()
            else []
        )
    if view in {"provenance", "attempt"}:
        header["provenance"] = {
            "execution": origin,
            "inputs": selected.get("inputs", ()),
            "parameters": selected.get("parameters", {}),
            "resources": selected.get("resources", origin.get("resources", {})),
        }
        environment = owned(
            directory, str(selected.get("environment_manifest") or "environment.json")
        )
        header["provenance"]["environment"] = (
            read_mapping(environment) if environment.exists() else None
        )
        heartbeat = owned(directory, "resource-heartbeat.json")
        header["resource_observation"] = read_mapping(heartbeat) if heartbeat.exists() else None
    if view == "html":
        from lambdaforge.study_projection import read_html_sections

        relocated = {**selected, "run_dir": str(directory)}
        header["sections"] = read_html_sections([relocated], root, skip_unretained=True)
    return header


def live_result(
    roots: list[Path], job_id: str, *, status: str = "running"
) -> tuple[Path, dict[str, Any]]:
    """Read exact Job-attested live records, never guess from a name or a newest folder."""
    matches = []
    for base in roots:
        if base.resolve() != base:
            raise ValueError("Live Work evidence owner cannot be symbolic.")
        for manifest in base.glob("*/execution-*/execution.json"):
            if manifest.resolve() != manifest:
                raise ValueError("Live Execution manifest cannot be symbolic.")
            origin = read_mapping(manifest)
            if origin.get("job_id") == job_id:
                matches.append((manifest.parent, origin))
    if len(matches) != 1:
        raise ValueError("No unique Job-attested live Execution yet; Job logs remain available.")
    root, origin = matches[0]
    rows = []
    for request in sorted(root.glob("runs/run-*/attempts/attempt-*/request.json")):
        row = read_mapping(request)
        directory = attempt_path(root, row)
        if request.parent != directory:
            raise ValueError("Live Attempt request disagrees with its owner.")
        result_path = owned(directory, "result.json")
        rows.append(
            read_mapping(result_path)
            if result_path.exists()
            else {
                **row,
                "status": "interrupted"
                if status in {"failed", "cancelled", "timeout"}
                else "running",
                "name": origin.get("name"),
                "scientific_fingerprint": origin.get("scientific_fingerprint"),
            }
        )
    return root, {
        "execution_result_version": 1,
        "execution_id": origin["execution_id"],
        "name": origin.get("name"),
        "scientific_fingerprint": origin["scientific_fingerprint"],
        "status": status,
        "runs": rows,
        "summary": {},
        "live": True,
    }


def preview_artifact(
    detail: Mapping[str, Any], name: str, *, byte_limit: int = 64 * 1024
) -> dict[str, Any]:
    """Explicit small-file access only, never execute HTML or sample partial large bytes."""
    if type(byte_limit) is not int or not 1 <= byte_limit <= 64 * 1024:
        raise ValueError("Output preview is limited to 64 KiB.")
    descriptors = [row for row in detail.get("artifacts", ()) if row.get("name") == name]
    if len(descriptors) != 1:
        raise KeyError("Select a unique registered artifact from this exact Run/Attempt.")
    artifact = descriptors[0]
    directory = Path(str(detail["selected"]["run_dir"]))
    path = owned(directory, str(artifact["path"]))
    answer = {"artifact": dict(artifact), "verified": False, "content": None}
    if not path.exists():
        return {**answer, "notice": "Artifact is not retained; no bytes downloaded."}
    if not path.is_file() or artifact.get("size_bytes", byte_limit + 1) > byte_limit:
        return {
            **answer,
            "notice": (
                "Directory or large artifact: metadata only. Export explicitly to access its bytes."
            ),
        }
    with path.open("rb") as stream:
        content = stream.read(byte_limit + 1)
    checksum = hashlib.sha256(path.name.encode("utf-8") + b"\0" + content).hexdigest()
    if len(content) > byte_limit or (len(content), checksum) != (
        artifact.get("size_bytes"),
        artifact.get("sha256"),
    ):
        raise ValueError("Output preview content differs from its finalized checksum.")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {
            **answer,
            "verified": True,
            "notice": "Verified binary file; use an explicit export to access its bytes.",
        }
    return {
        **answer,
        "verified": True,
        "content": text,
        "notice": "Verified UTF-8 preview; HTML/scripts are shown as text, never executed.",
    }


def page(value: Mapping[str, Any], offset: int = 0) -> dict[str, Any]:
    """The existing 512 KiB projection wire contract, with no aggregate Study-size ceiling."""
    import base64

    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    if type(offset) is not int or not 0 <= offset <= len(raw):
        raise ValueError("Invalid Work projection page offset.")
    chunk = raw[offset : offset + 512 * 1024]
    return {
        "fingerprint": hashlib.sha256(raw).hexdigest(),
        "offset": offset,
        "next_offset": offset + len(chunk),
        "total_bytes": len(raw),
        "data": base64.b64encode(chunk).decode(),
        "eof": offset + len(chunk) == len(raw),
    }


def compact_index(result: Mapping[str, Any], configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Small derived catalog cells; never an alternative scientific result authority."""
    runs = result.get("runs", ())
    latest: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        previous = latest.get(run["run_id"])
        if previous is None or run["attempt_number"] > previous["attempt_number"]:
            latest[run["run_id"]] = run
    starts = [str(row["started_at_utc"]) for row in runs if row.get("started_at_utc")]
    finishes = [str(row["finished_at_utc"]) for row in runs if row.get("finished_at_utc")]
    return {
        "result_index_version": 1,
        "name": result.get("name"),
        "execution_id": result.get("execution_id"),
        "scientific_fingerprint": result.get("scientific_fingerprint"),
        "status": result.get("status"),
        "study_expected": is_study(configuration),
        "run_count": len(latest),
        "attempt_count": len(runs),
        "started_at_utc": min(starts) if starts else None,
        "finished_at_utc": max(finishes) if finishes else None,
        "duration_seconds": sum(float(row.get("duration_seconds") or 0) for row in runs),
        "states": {
            state: sum(row.get("status") == state for row in latest.values())
            for state in ("succeeded", "failed", "pruned", "cancelled", "interrupted")
        },
        "objective": result.get("summary", {}).get("objective"),
    }
