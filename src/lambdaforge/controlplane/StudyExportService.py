"""Portable, provider-neutral export of one completed scientific Study."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.WorkService import WorkService
from lambdaforge.work.ResultStore import ResultStore


class StudyExportService:
    """Download exact Study evidence and produce the normal portable result package."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        *,
        jobs: JobService | None = None,
        works: WorkService | None = None,
        factory: ControlPlaneFactory | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.jobs = jobs or JobService(self.catalog, factory=self.factory)
        self.works = works or WorkService(self.catalog, jobs=self.jobs)

    def export(self, selector: str, destination: str | Path) -> dict[str, Any]:
        """Export the newest successful Attempt of one unambiguous semantic Work."""
        work = self.works.show(selector)
        records = [self.jobs.get(job_id, refresh=False) for job_id in work.job_ids]
        successful = [record for record in records if record.state is JobState.SUCCEEDED]
        if not successful:
            states = ", ".join(f"{record.job_id}={record.state.value}" for record in records)
            raise ValueError(
                "A final export requires a succeeded Attempt. Current Attempt states: " + states
            )
        record = max(successful, key=lambda value: value.created_at_utc)
        expected_execution = self._expected_execution(record)
        profile = self.catalog.get(record.cluster)
        transport = self.factory.transport(profile)
        remote_archive = str(
            PurePosixPath(record.work_dir).parent / f".lambdaforge-export-{uuid4().hex}.tar"
        )
        with tempfile.TemporaryDirectory(prefix="lambdaforge-export-download-") as raw_temp:
            temporary = Path(raw_temp)
            local_archive = temporary / "evidence.tar"
            try:
                created = transport.run(
                    (
                        profile.python,
                        "-c",
                        _ARCHIVE_SCRIPT,
                        str(PurePosixPath(record.work_dir).parent),
                        remote_archive,
                        expected_execution or "",
                        record.job_id,
                    ),
                    timeout=3600.0,
                )
                if created.returncode != 0:
                    detail = created.stderr.strip() or created.stdout.strip()
                    raise RuntimeError(f"Could not assemble Study evidence: {detail[-2000:]}")
                transport.get(remote_archive, local_archive)
            except BaseException:
                try:
                    transport.run(("rm", "-f", remote_archive), timeout=30.0)
                except Exception:
                    pass
                raise
            cleaned = transport.run(("rm", "-f", remote_archive), timeout=30.0)
            if cleaned.returncode != 0:
                detail = cleaned.stderr.strip() or cleaned.stdout.strip()
                raise RuntimeError(
                    "Study evidence was downloaded but its temporary provider archive could not "
                    f"be removed: {detail[-1000:]}"
                )

            downloaded = temporary / "downloaded"
            downloaded.mkdir()
            _extract_safe(local_archive, downloaded)
            run_root = downloaded / "runs"
            store = ResultStore(run_root)
            execution_selector = self._execution_selector(record, store, expected_execution)
            supplementary = {
                name: path
                for name, path in {
                    "control-plane": downloaded / "control-plane",
                    "study": downloaded / "study",
                    "published-artifacts": downloaded / "published-artifacts",
                    "submitted-configuration": downloaded / "configuration",
                }.items()
                if path.exists()
            }
            result = store.export(
                execution_selector,
                destination,
                supplementary=supplementary,
                copy_published=False,
            )
        return {
            **result,
            "work_id": work.work_id,
            "job_id": record.job_id,
            "cluster": record.cluster,
        }

    def _expected_execution(self, record: JobRecord) -> str | None:
        """Read the bounded Study index once when it can identify the exact Execution."""
        try:
            study = self.jobs.study(record.job_id)
        except (OSError, RuntimeError, ValueError, KeyError):
            return None
        expected = study.get("execution_id") if isinstance(study, Mapping) else None
        return expected if isinstance(expected, str) and expected else None

    @staticmethod
    def _execution_selector(record: JobRecord, store: ResultStore, expected: str | None) -> str:
        """Resolve the archived Execution without guessing between historical copies."""
        records = store.list()
        if expected is not None and any(value.get("execution_id") == expected for value in records):
            return expected
        matching = [
            value
            for value in records
            if value.get("status") == "succeeded"
            and any(
                isinstance(run, Mapping) and run.get("job_id") == record.job_id
                for run in value.get("runs", ())
            )
        ]
        if len(matching) != 1:
            matching = [value for value in records if value.get("status") == "succeeded"]
        if len(matching) != 1:
            raise RuntimeError(
                "Downloaded evidence does not identify exactly one succeeded Execution; "
                f"found {len(matching)}."
            )
        return str(matching[0]["execution_id"])


def _extract_safe(archive: Path, destination: Path) -> None:
    """Extract only regular files/directories below the destination."""
    root = destination.resolve()
    with tarfile.open(archive, "r") as package:
        for member in package.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in Study export archive: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsafe non-regular entry in Study export archive: {member.name}")
            target = (root / path).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Study export entry escaped its destination: {member.name}")
        for member in package.getmembers():
            target = root / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported entry in Study export archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = package.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read Study export member: {member.name}")
            with source, target.open("wb") as destination_stream:
                shutil.copyfileobj(source, destination_stream, length=1024 * 1024)


_ARCHIVE_SCRIPT = r"""
import json,os,sys,tarfile
from pathlib import Path

authored_job=Path(sys.argv[1]); job=authored_job.resolve(); output=Path(sys.argv[2])
expected=sys.argv[3]; job_id=sys.argv[4]
if (authored_job.is_symlink() or authored_job.absolute() != job
    or job.name.startswith("job-") is False or not job.is_dir()):
 raise SystemExit("invalid owned Job root")
work=job/"work"; runs=work/".lambdaforge"/"runs"
if not runs.is_dir() or runs.is_symlink():
 raise SystemExit("persisted Execution evidence is missing")

def check(path):
 if path.is_symlink(): raise SystemExit("symlinked evidence is not exportable: "+str(path))
 if path.absolute() != path.resolve():
  raise SystemExit("evidence below a symlinked path is not exportable: "+str(path))
 if path.is_file(): return
 if not path.is_dir(): raise SystemExit("special evidence is not exportable: "+str(path))
 for root,dirs,files in os.walk(path,followlinks=False):
  base=Path(root)
  for name in dirs+files:
   child=base/name
   if child.is_symlink(): raise SystemExit("symlinked evidence is not exportable: "+str(child))
   if not child.is_dir() and not child.is_file():
    raise SystemExit("special evidence is not exportable: "+str(child))

entries=[]
for child in sorted(job.iterdir(),key=lambda value:value.name):
 if (child.name in {"work","study"} or child == output
     or child.name.startswith(".lambdaforge-export-")):
  continue
 if child.is_file() and not child.is_symlink(): entries.append((child,"control-plane/"+child.name))
if (job/"study").exists(): entries.append((job/"study","study"))
if (work/"config.yaml").is_file(): entries.append((work/"config.yaml","configuration/config.yaml"))

manifests=[]
for candidate in sorted(runs.glob("*/execution-*/result.json")):
 try: value=json.loads(candidate.read_text(encoding="utf-8"))
 except Exception: continue
 if expected and value.get("execution_id")==expected:
  manifests=[candidate]; break
 if (not expected and value.get("status")=="succeeded"
     and any(isinstance(run,dict) and run.get("job_id")==job_id for run in value.get("runs",[]))):
  manifests.append(candidate)
if len(manifests)!=1:
 raise SystemExit(f"could not identify one succeeded Execution (found {len(manifests)})")
execution=manifests[0].parent
entries.append((execution,f"runs/{execution.parent.name}/{execution.name}"))

seen=set()
for manifest in manifests:
 try: result=json.loads(manifest.read_text(encoding="utf-8"))
 except Exception: continue
 for run_index,run in enumerate(result.get("runs",[])):
  if not isinstance(run,dict): continue
  for artifact_index,artifact in enumerate(run.get("artifacts",[])):
   if not isinstance(artifact,dict): continue
   metadata=artifact.get("metadata") if isinstance(artifact.get("metadata"),dict) else {}
   raw=artifact.get("published_path") or metadata.get("published_to")
   if not isinstance(raw,str) or not raw: continue
   path=Path(raw)
   if not path.is_absolute() or not path.exists() or path in seen: continue
   try: path.resolve().relative_to(job)
   except ValueError: pass
   else: continue
   seen.add(path)
   label=str(artifact.get("name") or path.name or "artifact")
   label="".join(c if c.isalnum() or c in "._-" else "-" for c in label).strip(".-") or "artifact"
   entries.append((path,f"published-artifacts/run-{run_index:05d}/{artifact_index:04d}-{label}"))

with tarfile.open(output,"w",format=tarfile.PAX_FORMAT) as archive:
 for source,name in entries:
  check(source); archive.add(source,arcname=name,recursive=True)
print(json.dumps({"archive":str(output),"entries":len(entries)}))
"""


__all__ = ["StudyExportService"]
