"""Read-only synthesis of native operational/scientific catalogues, never new state."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.OperationalHealth import OperationalHealth
from lambdaforge.controlplane.WorkService import WorkService


class InspectionService:
    """Exact selector resolution and bounded presentation shared by CLI and future UI clients."""

    def __init__(self, catalog: ClusterCatalog, *, jobs: JobService | None = None) -> None:
        self.catalog = catalog
        self.jobs = jobs or JobService(catalog)

    def inspect(
        self, selector: str, *, kind: str = "auto", refresh: bool = False
    ) -> dict[str, Any]:
        from lambdaforge.agents.contracts import redact
        from lambdaforge.work.ResultStore import ResultStore

        if kind not in {"auto", "work", "study", "execution", "job"}:
            raise ValueError("Inspection supports auto/work/study/execution/job.")
        effective = kind
        if kind == "auto":
            effective = (
                "job"
                if selector.startswith("job-")
                else "execution"
                if selector.startswith("execution-")
                else "work"
                if selector.startswith("work-")
                else "auto"
            )
        if effective == "job":
            job_id = self.jobs.resolve_selector(selector)
            record = self.jobs.get(job_id, refresh=refresh, include_study=False)
            health = OperationalHealth.describe(record)
            operational = {
                key: health.get(key)
                for key in (
                    "health",
                    "reason",
                    "freshness",
                    "observation_age_seconds",
                    "execution_state",
                    "last_success_at_utc",
                    "last_confirmed_at_utc",
                    "diagnostic",
                    "last_error",
                )
            }
            saved = record.metadata.get("remote_state", {})
            progress = saved.get("progress", {}) if isinstance(saved, Mapping) else {}
            return redact(
                {
                    "kind": "job",
                    "job_id": job_id,
                    "name": record.metadata.get("name"),
                    "cluster": record.cluster,
                    "state": record.state.value,
                    "execution_id": record.metadata.get("execution_id"),
                    "operational": operational,
                    "progress": {
                        key: progress[key]
                        for key in ("completed", "total", "unit", "epoch")
                        if key in progress
                    }
                    if isinstance(progress, Mapping)
                    else {},
                    "failure": record.metadata.get("failure"),
                    "refresh_requested": refresh,
                }
            )
        project = self.catalog.project
        results = ResultStore(project.root / ".lambdaforge/runs" if project else None)
        candidates: list[dict[str, Any]] = []
        if effective in {"auto", "execution"}:
            candidates.extend(
                {"kind": "execution", **row}
                for row in results.catalog()
                if selector
                in {row.get("execution_id"), row.get("name"), row.get("scientific_fingerprint")}
            )
        if effective in {"auto", "work", "study"}:
            from lambdaforge.controlplane.ResearchWork import aggregate_research_work

            for work in aggregate_research_work(self.jobs.list(refresh=False)):
                if selector not in {work.work_id, work.name, work.scientific_revision}:
                    continue
                if effective == "study" and not work.study_expected:
                    continue
                candidates.append(
                    {
                        **work.to_dict(study_detail="overview"),
                        "kind": "study" if work.study_expected else "work",
                    }
                )
        if effective == "auto":
            # Names are not identities. Even related native Work/Job/Execution records need
            # an explicit kind rather than silently selecting whichever catalogue ran first.
            try:
                job_id = self.jobs.resolve_selector(selector)
            except KeyError:
                pass
            else:
                candidates.append({"kind": "job", "job_id": job_id})
        if not candidates:
            raise KeyError(f"No {kind} evidence for {selector!r}; use an exact ID or --kind.")
        if len(candidates) != 1:
            choices = "; ".join(
                f"{row['kind']}:"
                f"{row.get('execution_id') or row.get('work_id') or row.get('job_id')}"
                for row in candidates[:20]
            )
            raise ValueError(
                f"Ambiguous selector {selector!r}: {choices}. Specify --kind or exact ID."
            )
        value = candidates[0]
        if value["kind"] == "job":
            return self.inspect(value["job_id"], kind="job", refresh=refresh)
        value.pop("_manifest_path", None)
        if refresh and value["kind"] in {"work", "study"}:
            for job_id in value["job_ids"]:
                self.jobs.get(job_id, refresh=True, include_study=False)
            work = WorkService(self.catalog, jobs=self.jobs).show(value["work_id"])
            value = {
                **work.to_dict(study_detail="overview"),
                "kind": "study" if work.study_expected else "work",
            }
        return redact(value)

    def diagnose(
        self, selector: str, *, kind: str = "auto", refresh: bool = False
    ) -> dict[str, Any]:
        """Native diagnostics and persisted history, with uncertainty and recovery kept honest."""
        value = self.inspect(selector, kind=kind, refresh=refresh)
        job_id = value.get("job_id") or value.get("primary_job_id")
        job = self.inspect(job_id, kind="job", refresh=False) if job_id else None
        events = []
        if job_id:
            events = self.jobs.store.event_page(job_id, offset=0, limit=10)["items"]
        from lambdaforge.agents.contracts import redact

        quoted = shlex.quote(str(job_id or value.get("execution_id") or selector))
        return redact(
            {
                "entity": value,
                "job": job,
                "recent_history": events,
                "diagnostic": (job or {}).get("operational", {}).get("diagnostic"),
                "confirmed_failure": (job or {}).get("failure"),
                "possible_causes": [],
                "recovery": {
                    "eligibility": (
                        "not inspected; state alone cannot prove ownership/checkpoint safety"
                    ),
                    "preview_command": f"lf retry {quoted} --dry-run --json" if job_id else None,
                    "automatic_retry": False,
                },
                "follow_up": [f"lf logs {quoted} --tail 100"] if job_id else [],
                "science_mutated": False,
            }
        )
