"""High-level read and safe deletion operations for semantic research work."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.ResearchWork import ResearchWork, aggregate_research_work
from lambdaforge.controlplane.StorageService import StorageService


class WorkService:
    """Resolve human work names while retaining JobStore as the operational authority."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        *,
        jobs: JobService | None = None,
        storage: StorageService | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.jobs = jobs or JobService(self.catalog)
        self.storage = storage or StorageService(self.catalog)

    def show(self, selector: str) -> ResearchWork:
        """Resolve one Work by ID, exact name, revision or underlying job selector."""
        works = aggregate_research_work(self.jobs.list(refresh=False))
        matches = tuple(
            work
            for work in works
            if selector in {work.work_id, work.name, work.scientific_revision, work.primary_job_id}
            or selector in work.job_ids
        )
        if not matches:
            try:
                job_id = self.jobs.resolve_selector(selector)
            except Exception as error:
                raise KeyError(f"Unknown work {selector!r}.") from error
            matches = tuple(work for work in works if job_id in work.job_ids)
        if len(matches) != 1:
            raise ValueError(
                f"Work selector {selector!r} is ambiguous; use a work ID or scientific revision."
            )
        return matches[0]

    def cancel(self, selector: str) -> dict[str, Any]:
        """Cancel every active scheduler Job belonging to one semantic Work."""
        work = self.show(selector)
        cancelled: list[dict[str, Any]] = []
        reconciled: list[dict[str, Any]] = []
        terminal: list[dict[str, str]] = []
        failures: list[dict[str, str]] = []
        for job_id in work.job_ids:
            try:
                record = self.jobs.get(job_id)
                reconcilable = (
                    record.state.value == "cancelled"
                    and record.scheduler == "local"
                    and record.scheduler_id is not None
                )
                if record.state.terminal and not reconcilable:
                    terminal.append({"job_id": job_id, "state": record.state.value})
                    continue
                stopped = self.jobs.cancel(job_id)
                selected = {"job_id": stopped.job_id, "state": stopped.state.value}
                (reconciled if reconcilable else cancelled).append(selected)
            except Exception as error:
                failures.append(
                    {
                        "job_id": job_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        if failures:
            details = "; ".join(
                f"{failure['job_id']}: {failure['error']}" for failure in failures
            )
            raise RuntimeError(
                f"Work cancellation was incomplete after attempting every active Job: {details}"
            )
        return {
            "work_id": work.work_id,
            "name": work.name,
            "cancelled_jobs": cancelled,
            "reconciled_cancelled_jobs": reconciled,
            "already_terminal": terminal,
            "status": (
                "cancelled" if cancelled else "reconciled" if reconciled else "already-terminal"
            ),
        }

    def delete(self, selector: str, *, apply: bool = False) -> dict[str, Any]:
        """Preview or delete only terminal attempts and their exact owned workspaces."""
        try:
            work = self.show(selector)
        except KeyError:
            receipt = self._deletion_receipt(selector)
            if receipt is None:
                raise
            return {
                **receipt,
                "workspaces": [],
                "applied": apply,
                "already_deleted": True,
            }
        records = tuple(self.jobs.get(job_id) for job_id in work.job_ids)
        active = tuple(record.job_id for record in records if not record.state.terminal)
        if active:
            raise ValueError(f"Cannot delete active work; cancel it first: {active}.")
        workspace_plans = [
            self.storage.delete_job(
                record.cluster,
                record.job_id,
                apply=apply,
                local_run_root=(
                    self.jobs.job_root(record)
                    if self.catalog.get(record.cluster).transport == "local"
                    else None
                ),
            )
            for record in records
        ]
        if apply:
            self._write_deletion_receipt(work.to_dict(), tuple(record.job_id for record in records))
            for record in records:
                self.jobs.delete(record.job_id)
        return {
            "work": work.to_dict(),
            "job_records": [record.job_id for record in records],
            "workspaces": workspace_plans,
            "applied": apply,
            "already_deleted": False,
            "preserved": [
                "published datasets",
                "shared caches and environments",
                "other Work and job records",
            ],
        }

    def delete_job(self, selector: str, *, apply: bool = False) -> dict[str, Any]:
        """Delete one terminal Job's exact workspace and local history record."""
        job_id = self.jobs.resolve_selector(selector)
        record = self.jobs.get(job_id)
        if not record.state.terminal:
            raise ValueError(f"Cannot delete active job {job_id}; cancel it first.")
        local_root = (
            self.jobs.job_root(record)
            if self.catalog.get(record.cluster).transport == "local"
            else None
        )
        workspace = self.storage.delete_job(
            record.cluster,
            record.job_id,
            apply=apply,
            local_run_root=local_root,
        )
        if apply:
            self.jobs.delete(record.job_id)
        return {
            "job_id": record.job_id,
            "cluster": record.cluster,
            "state": record.state.value,
            "workspace": workspace,
            "applied": apply,
            "preserved": ["published datasets", "shared caches and environments", "other Jobs"],
        }

    def clear_history(self, *, apply: bool = False) -> dict[str, Any]:
        """Preview or remove every terminal Job while preserving all active work."""
        records = self.jobs.list(refresh=True)
        terminal = tuple(record for record in records if record.state.terminal)
        active = tuple(record.job_id for record in records if not record.state.terminal)
        deleted: list[str] = []
        failures: list[dict[str, str]] = []
        plans: list[dict[str, Any]] = []
        for record in terminal:
            try:
                result = self.delete_job(record.job_id, apply=apply)
            except Exception as error:
                failures.append(
                    {
                        "job_id": record.job_id,
                        "error": f"{error.__class__.__name__}: {error}",
                    }
                )
                continue
            plans.append(result)
            if apply:
                deleted.append(record.job_id)
        return {
            "applied": apply,
            "terminal_jobs": [record.job_id for record in terminal],
            "deleted_jobs": deleted,
            "active_jobs_preserved": list(active),
            "failures": failures,
            "jobs": plans,
        }

    @property
    def _deletion_root(self) -> Path:
        return self.jobs.store.root.parent / "work-deletions"

    def _write_deletion_receipt(self, work: dict[str, Any], job_ids: tuple[str, ...]) -> None:
        """Persist only enough terminal metadata to make deletion retries convergent."""
        root = self._deletion_root
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{work['work_id']}.json"
        payload = {
            "receipt_version": 1,
            "deleted_at_utc": datetime.now(timezone.utc).isoformat(),
            "work": work,
            "job_records": list(job_ids),
            "preserved": [
                "published datasets",
                "shared caches and environments",
                "other Work and job records",
            ],
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _deletion_receipt(self, selector: str) -> dict[str, Any] | None:
        root = self._deletion_root
        if not root.is_dir() or root.is_symlink():
            return None
        matches: list[dict[str, Any]] = []
        for path in root.glob("work-*.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            work = value.get("work", {}) if isinstance(value, dict) else {}
            jobs = value.get("job_records", ()) if isinstance(value, dict) else ()
            if isinstance(work, dict) and (
                selector
                in {
                    work.get("work_id"),
                    work.get("name"),
                    work.get("scientific_revision"),
                    work.get("primary_job_id"),
                }
                or selector in jobs
            ):
                matches.append(value)
        if len(matches) > 1:
            raise ValueError(f"Deleted Work selector {selector!r} is ambiguous; use its Work ID.")
        return matches[0] if matches else None
