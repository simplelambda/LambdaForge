"""Atomic local extension revision/handoff journal, never a scientific execution engine."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.extension_design import plain
from lambdaforge.work.models import atomic_json
from lambdaforge.work.recovery import read_owned_json, validate_execution

_SERIES = re.compile(r"series-[0-9a-f]{24}\Z")
_REQUEST = re.compile(r"extension-[0-9a-f]{32}\Z")
_MAX_BYTES = 32 * 1024**2


def _digest(value: Mapping[str, Any]) -> str:
    return ScientificIdentity.from_payload(value).digest


class StudySeriesStore:
    """Owned metadata only, with immutable revisions and an atomic append-only history.

    Methods are internal building blocks, not a substitute for service apply validation.
    A reservation is explicitly not a Job, an Execution or completed scientific evidence.
    One atomic envelope avoids inconsistent separate history/revision commit points.
    """

    def __init__(self, project: ProjectContext) -> None:
        self.project = project
        self.root = project.root / ".lambdaforge" / "study-series"
        self.results_root = project.root / ".lambdaforge" / "runs"

    def _directory(self, series_id: str) -> Path:
        if not _SERIES.fullmatch(series_id):
            raise ValueError("Invalid exact Study series ID.")
        path = self.root / series_id
        if path.resolve() != path:
            raise ValueError("Study series storage cannot contain symbolic paths.")
        return path

    def read(self, series_id: str) -> dict[str, Any]:
        """Read bounded checksummed metadata; no lock creation, artifacts or mutations."""
        path = self._directory(series_id) / "series.json"
        if path.stat().st_size > _MAX_BYTES:
            raise ValueError("Study series journal exceeds its bounded metadata contract.")
        value = read_owned_json(path)
        checksum = value.pop("checksum", None)
        if (
            value.get("study_series_version") != 1
            or value.get("study_series_id") != series_id
            or value.get("project_id") != self.project.project_id
            or checksum != _digest(value)
        ):
            raise ValueError("Corrupt or differently owned Study series journal.")
        revisions, history = value.get("revisions"), value.get("history")
        operations = value.get("operations")
        if (
            not isinstance(revisions, list)
            or not revisions
            or not isinstance(history, list)
            or not isinstance(operations, list)
        ):
            raise ValueError("Study series authoritative revision/history records are missing.")
        for number, revision in enumerate(revisions):
            if not isinstance(revision, dict) or revision.get("number") != number:
                raise ValueError("Study series immutable revision order is corrupt.")
            if revision.get("revision_hash") != _digest(
                {key: item for key, item in revision.items() if key != "revision_hash"}
            ):
                raise ValueError("Study series revision checksum differs.")
        previous = None
        for number, event in enumerate(history):
            if (
                not isinstance(event, dict)
                or event.get("sequence") != number
                or event.get("previous_hash") != previous
                or event.get("event_hash")
                != _digest({key: item for key, item in event.items() if key != "event_hash"})
            ):
                raise ValueError("Study series append-only history chain is corrupt.")
            previous = event["event_hash"]
        if value.get("current_revision") != len(revisions) - 1:
            raise ValueError("Study series current revision differs from its history.")
        if len(operations) != len(revisions) - 1 or any(
            not isinstance(item, dict)
            or item.get("revision") != number
            or not _REQUEST.fullmatch(str(item.get("request_id", "")))
            or item.get("state")
            not in {"reserved", "dispatching", "submitted", "succeeded", "failed"}
            or (
                item.get("state") == "reserved"
                and (item.get("job_id") is not None or item.get("execution_id") is not None)
            )
            or (
                item.get("state") != "reserved"
                and not re.fullmatch(r"job-[A-Za-z0-9_-]+", str(item.get("job_id", "")))
            )
            for number, item in enumerate(operations, 1)
        ):
            raise ValueError("Study series reservation records are corrupt or unsupported.")
        return {**value, "checksum": checksum}

    def reserve(
        self,
        plan: Mapping[str, Any],
        *,
        request_id: str,
        expected_revision: int = 0,
        verify_remote: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Internally reserve one revision, never enqueue or invent native ownership.

        Repeated request/plan returns the same reservation. A second pending revision cannot
        supersede it. The source controller lock closes the native deletion/reservation race.
        Local Job references are retained in the same journal for native history/GC protection.
        """
        if not _REQUEST.fullmatch(request_id):
            raise ValueError("Extension request requires an exact extension-UUID identifier.")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValueError("Expected Study revision must be an explicit integer.")
        if (
            plan.get("study_extension_plan_version") != 1
            or plan.get("project", {}).get("project_id") != self.project.project_id
            or plan.get("plan_hash")
            != _digest({key: item for key, item in plan.items() if key != "plan_hash"})
        ):
            raise ValueError("Extension reservation requires an intact, owned native plan.")
        if plan["comparison"]["counts"]["new"] == 0:
            return {"state": "no-op", "request_id": None, "created": False}
        refs = plan["reused_evidence"]
        if not refs:
            raise ValueError("Extension reservation lacks authoritative origin references.")
        remote = plan.get("placement") == "remote"
        execution = Path(
            str(plan.get("origin_execution_dir") or Path(refs[0]["result_path"]).parents[4])
        )
        if not remote and (
            execution.resolve() != execution
            or execution.parent.parent != self.results_root
            or execution.name != plan["origin_execution_id"]
        ):
            raise ValueError("Extension reservation escaped the owned local Execution root.")
        if remote and verify_remote is None:
            raise ValueError("Remote extension requires native host evidence revalidation.")
        directory = self._directory(str(plan["study_series_id"]))
        with (
            nullcontext()
            if remote
            else CrossProcessFileLock(
                execution / ".controller.lock",
                shared=True,
                create=False,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )
        ):
            if remote:
                assert verify_remote is not None
                verify_remote()
                manifest = plan["origin_manifest"]
            else:
                manifest = validate_execution(execution)
            if manifest.get("job_id") != plan.get("origin_job_id"):
                raise ValueError("Origin Job ownership differs from its reviewed plan.")
            expected_series = (
                "series-"
                + _digest(
                    {
                        "series_identity_version": 1,
                        "project_id": self.project.project_id,
                        "origin_execution": manifest["execution_id"],
                        "origin_scientific_identity": manifest["scientific_fingerprint"],
                    }
                ).removeprefix("sha256:")[:24]
            )
            if directory.name != expected_series:
                raise ValueError("Series identity differs from its exact native origin.")
            if not remote and (
                hashlib.sha256((execution / "execution.json").read_bytes()).hexdigest()
                != plan["origin_manifest_sha256"]
            ):
                raise ValueError("Original Execution changed after extension preview.")
            for ref in refs:
                if remote:
                    continue  # Exact host-owned references were checked by the native reader.
                result = Path(str(ref["result_path"]))
                owner = result.parents[4]
                if (
                    result.resolve() != result
                    or owner.parent.parent != self.results_root
                    or owner.name != ref["execution_id"]
                ):
                    raise ValueError("Historical result reference is outside its owned Execution.")
                if result.relative_to(owner).parts != (
                    "runs",
                    ref["run_id"],
                    "attempts",
                    ref["attempt_id"],
                    "result.json",
                ):
                    raise ValueError(
                        "Historical reference does not name its native Attempt result."
                    )
                if result.stat().st_size > _MAX_BYTES:
                    raise ValueError("Historical result exceeds the bounded metadata contract.")
                if hashlib.sha256(result.read_bytes()).hexdigest() != ref["result_sha256"]:
                    raise ValueError("Historical Attempt changed after extension preview.")
            lock_path = self.root / ".locks" / (directory.name + ".lock")
            if lock_path.resolve() != lock_path:
                raise ValueError("Study series lock storage cannot contain symbolic paths.")
            with CrossProcessFileLock(
                lock_path,
                shared=False,
                timeout_seconds=5,
                poll_interval_seconds=0.01,
            ):
                try:
                    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
                except FileExistsError:
                    created = False
                else:
                    created = True
                path = directory / "series.json"
                if path.is_symlink():
                    raise ValueError("Study series authoritative journal is symbolic.")
                if path.exists():
                    current = self.read(directory.name)
                    operations = current["operations"]
                    for operation in operations:
                        if operation["request_id"] == request_id:
                            if operation["plan_hash"] != plan["plan_hash"]:
                                raise ValueError(
                                    "Extension request ID was reused for another plan."
                                )
                            return {**operation, "created": False}
                        if operation["plan_hash"] == plan["plan_hash"]:
                            return {**operation, "created": False}
                    if any(item["state"] != "succeeded" for item in operations):
                        raise ValueError("A pending Study extension already owns this series.")
                else:
                    if not created:
                        raise ValueError(
                            "Study series journal is missing; reconcile its reservation, "
                            "not a silent restart."
                        )
                    origin = {
                        "number": 0,
                        "execution_id": plan["origin_execution_id"],
                        "job_id": plan.get("origin_job_id"),
                        "manifest_sha256": plan["origin_manifest_sha256"],
                        "runtime_fingerprint": plan["runtime_compatibility"]["fingerprint"],
                        "target": plan["target"],
                        "placement": plan.get("placement", "local"),
                        "configuration": plain(plan.get("origin_configuration")),
                        "analysis_semantics": plain(plan.get("origin_analysis_semantics", {})),
                        "evidence": plain(refs),
                    }
                    origin["revision_hash"] = _digest(origin)
                    current = {
                        "study_series_version": 1,
                        "study_series_id": directory.name,
                        "project_id": self.project.project_id,
                        "current_revision": 0,
                        "revisions": [origin],
                        "operations": [],
                        "history": [],
                    }
                if current["current_revision"] != expected_revision:
                    raise ValueError("Study series revision advanced; inspect a new plan.")
                number = expected_revision + 1
                revision = {
                    "number": number,
                    "request_id": request_id,
                    "plan_hash": plan["plan_hash"],
                    "reviewed_plan": plain(plan),
                    "configuration": plain(plan["comparison"]["proposed_configuration"]),
                    "delta_configuration": plain(plan["comparison"]["delta_configuration"]),
                    "proposed_yaml": plan["proposed_yaml"],
                    "proposed_yaml_sha256": plan["proposed_yaml_sha256"],
                    "code_compatibility": plain(plan["code_compatibility"]),
                    "runtime_compatibility": plain(plan["runtime_compatibility"]),
                    "analysis_semantics": plain(plan.get("delta_analysis_semantics", {})),
                    "budget": plain(plan["budget"]),
                    "target": plan["target"],
                    "new_cells": plain(plan["comparison"]["new_cells"]),
                    "required_cells": plain(
                        plan["comparison"]["reused_cells"] + plan["comparison"]["new_cells"]
                    ),
                    "interpretation": plan["comparison"]["interpretation"],
                }
                revision["revision_hash"] = _digest(revision)
                operation = {
                    "request_id": request_id,
                    "revision": number,
                    "plan_hash": plan["plan_hash"],
                    "state": "reserved",
                    "job_id": None,
                    "execution_id": None,
                }
                history = current["history"]
                event = {
                    "sequence": len(history),
                    "previous_hash": history[-1]["event_hash"] if history else None,
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "action": "reserve-extension",
                    **operation,
                }
                event["event_hash"] = _digest(event)
                current["revisions"].append(revision)
                current["operations"].append(operation)
                history.append(event)
                current["current_revision"] = number
                current.pop("checksum", None)
                current["checksum"] = _digest(current)
                if len(json.dumps(current, allow_nan=False).encode()) > _MAX_BYTES:
                    raise ValueError("Study series reservation exceeds its metadata bound.")
                atomic_json(path, current)
                return {**operation, "created": True}

    def transition(
        self,
        series: str,
        request_id: str,
        *,
        state: str,
        facts: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Commit operational handoff facts without rewriting an immutable scientific revision."""
        allowed = {
            "reserved": {"dispatching"},
            "dispatching": {"submitted", "failed"},
            "submitted": {"succeeded", "failed"},
            "failed": {"submitted"},  # Only an exact native retry Job may repair the delta.
        }
        lock = self.root / ".locks" / (series + ".lock")
        if lock.resolve() != lock:
            raise ValueError("Unsafe Study series lock path.")
        with CrossProcessFileLock(
            lock, shared=False, timeout_seconds=5, poll_interval_seconds=0.01
        ):
            value = self.read(series)
            matches = [item for item in value["operations"] if item["request_id"] == request_id]
            if len(matches) != 1:
                raise ValueError("Unknown exact extension request.")
            operation = matches[0]
            if set(facts) - {
                "job_id",
                "execution_id",
                "snapshot",
                "snapshot_sha256",
                "source_yaml",
                "error",
                "recovery_jobs",
                "evidence",
                "spent_runs",
                "elapsed_seconds",
                "manifest_sha256",
            }:
                raise ValueError("Unknown extension handoff fields.")
            recovery = facts.get("recovery_jobs")
            if recovery is not None and (
                not isinstance(recovery, list)
                or recovery[: len(operation.get("recovery_jobs", []))]
                != operation.get("recovery_jobs", [])
                or any(not re.fullmatch(r"job-[A-Za-z0-9_-]+", str(job)) for job in recovery)
            ):
                raise ValueError("Recovery Job history must be an append-only native lineage.")
            if any(
                operation.get(key) not in (None, item)
                for key, item in facts.items()
                if key != "recovery_jobs"
            ):
                raise ValueError("Extension handoff cannot replace previously committed ownership.")
            if state == operation["state"]:
                if all(operation.get(key) == item for key, item in facts.items()):
                    return dict(operation)
            if state != operation["state"] and state not in allowed.get(operation["state"], set()):
                raise ValueError("Invalid native extension operation transition.")
            operation.update(plain(facts), state=state)
            event = {
                "sequence": len(value["history"]),
                "previous_hash": value["history"][-1]["event_hash"],
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "action": "extension-" + state,
                **operation,
            }
            event["event_hash"] = _digest(event)
            value["history"].append(event)
            value.pop("checksum", None)
            value["checksum"] = _digest(value)
            if len(json.dumps(value, allow_nan=False).encode()) > _MAX_BYTES:
                raise ValueError("Study series operation exceeds its metadata bound.")
            atomic_json(self._directory(series) / "series.json", value)
            return dict(operation)

    @staticmethod
    def execution_dependents(results_root: Path, execution_id: str) -> tuple[str, ...]:
        """Fail closed on corrupt journals, without importing or reading scientific payloads."""
        root = results_root.parent / "study-series"
        if not root.exists():
            if root.is_symlink():
                raise ValueError("Study series dependency root is symbolic.")
            return ()
        if root.resolve() != root or not root.is_dir():
            raise ValueError("Study series dependency root is unsafe.")
        project = ProjectContext.discover(results_root.parent.parent)
        store = StudySeriesStore(project)
        if store.root != root:
            raise ValueError("Study series dependencies belong to another project root.")
        retained = []
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError("Study series catalog contains an unsafe entry.")
            if directory.name == ".locks":
                continue
            if not (directory / "series.json").exists():
                raise ValueError("Study series journal is missing; reconcile ownership first.")
            value = store.read(directory.name)
            if any(
                row.get("execution_id") == execution_id
                for row in (*value["revisions"], *value["operations"])
            ):
                retained.append(directory.name)
        return tuple(retained)

    def snapshots(self) -> tuple[dict[str, str], ...]:
        """Read only attested generated YAML descriptors, never arbitrary consumer paths."""
        if not self.root.exists():
            return ()
        if self.root.resolve() != self.root:
            raise ValueError("Unsafe Study series root.")
        rows = []
        for directory in sorted(self.root.iterdir()):
            if directory.name == ".locks":
                continue
            value = self.read(directory.name)
            for operation in value["operations"]:
                if operation.get("snapshot"):
                    rows.append(
                        {"path": operation["snapshot"], "sha256": operation["snapshot_sha256"]}
                    )
        return tuple(rows)

    def job_references(self) -> dict[str, tuple[str, ...]]:
        """Protect physical Job history and storage, even for a terminal extension."""
        if not self.root.exists():
            return {}
        if self.root.resolve() != self.root:
            raise ValueError("Unsafe Study series root.")
        refs: dict[str, list[str]] = {}
        for directory in sorted(self.root.iterdir()):
            if directory.name == ".locks":
                continue
            value = self.read(directory.name)
            for row in (*value["revisions"], *value["operations"]):
                if row.get("job_id"):
                    refs.setdefault(str(row["job_id"]), []).append(directory.name)
                for job_id in row.get("recovery_jobs", []):
                    refs.setdefault(job_id, []).append(directory.name)
        return {key: tuple(sorted(set(values))) for key, values in refs.items()}
