"""Portable, provider-neutral export of one scientific Study in any lifecycle state."""

from __future__ import annotations

import errno
import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.jobs import JobRecord
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.WorkService import WorkService
from lambdaforge.work.models import atomic_json
from lambdaforge.work.ResultStore import ResultStore

ExportProgress = Callable[[Mapping[str, Any]], None]


class StudyExportService:
    """Download an exact bounded Study snapshot and produce a portable package."""

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

    def export(
        self,
        selector: str,
        destination: str | Path,
        *,
        profile: str = "default",
        progress: ExportProgress | None = None,
    ) -> dict[str, Any]:
        """Export the newest Attempt of one unambiguous semantic Work."""
        if profile not in {"default", "full"}:
            raise ValueError("Study export profile must be default or full.")
        started = time.monotonic()
        _emit_progress(progress, started, "resolving", "Resolving the newest Study Attempt.")
        work = self.works.show(selector)
        records = [self.jobs.get(job_id, refresh=False) for job_id in work.job_ids]
        if not records:
            raise ValueError(f"Work {selector!r} has no Attempt to export.")
        record = max(records, key=lambda value: value.created_at_utc)
        if record.metadata.get("fleet"):
            raise ValueError(
                "Distributed Fleet artifact export is not integrated yet; refusing a "
                "coordinator-only package that would omit member evidence."
            )
        # Refresh only the selected Attempt: its captured state must be current, while probing
        # every historical retry would add unnecessary provider traffic and failure modes.
        record = self.jobs.get(record.job_id, refresh=True)
        recovery_arguments: tuple[str, ...] = ()
        recovery = record.metadata.get("recovery_execution_dir")
        owner_id = record.metadata.get("recovery_owner_job")
        if isinstance(recovery, str) and isinstance(owner_id, str):
            owner = self.jobs.get(owner_id, refresh=False)
            if owner.cluster != record.cluster or owner_id not in record.metadata.get(
                "recovery_dependencies", ()
            ):
                raise ValueError("Invalid Study recovery evidence owner.")
            owner_root = owner.work_dir
            if self.catalog.get(owner.cluster).transport == "local":
                from lambdaforge.work.runner import WorkRunner

                source = Path(str(owner.metadata.get("source_config_path") or owner.config_path))
                owner_root = str(WorkRunner._project_root(source.parent) / ".lambdaforge" / "runs")
            recovery_arguments = (recovery, owner_root)
        expected_execution = self._expected_execution(record)
        export_profile = profile
        cluster_profile = self.catalog.get(record.cluster)
        assert cluster_profile.storage is not None
        product_root = cluster_profile.storage.product_root
        if cluster_profile.transport == "local":
            from lambdaforge.products.registry import ProductRegistry

            product_root = str(ProductRegistry().root)
        transport = self.factory.transport(cluster_profile)
        remote_archive = str(
            PurePosixPath(record.work_dir).parent / f".lambdaforge-export-{uuid4().hex}.zip"
        )
        local_parent = _safe_destination_parent(destination)
        with tempfile.TemporaryDirectory(
            prefix=".lambdaforge-export-transfer-", dir=local_parent
        ) as raw_temp:
            temporary = Path(raw_temp)
            local_archive = temporary / "evidence.zip"
            try:
                _emit_progress(
                    progress,
                    started,
                    "compressing",
                    f"Compressing persisted evidence on {record.cluster}.",
                )
                created = transport.run(
                    (
                        cluster_profile.python,
                        "-c",
                        _ARCHIVE_SCRIPT,
                        str(PurePosixPath(record.work_dir).parent),
                        remote_archive,
                        expected_execution or "-",
                        record.job_id,
                        export_profile,
                        *recovery_arguments,
                        "--product-root",
                        product_root,
                    ),
                    timeout=3600.0,
                )
                if created.returncode != 0:
                    detail = created.stderr.strip() or created.stdout.strip()
                    raise RuntimeError(f"Could not assemble Study evidence: {detail[-2000:]}")
                archive_size, extracted_size = _archive_metadata(created.stdout)
                if archive_size is not None and extracted_size is not None:
                    required = archive_size + extracted_size
                    available = shutil.disk_usage(local_parent).free
                    if required > available:
                        raise OSError(
                            errno.ENOSPC,
                            "The selected export filesystem does not have enough free space for "
                            f"the compressed archive plus extracted evidence: required at least "
                            f"{required} bytes, available {available} bytes below {local_parent}.",
                        )
                _emit_progress(
                    progress,
                    started,
                    "downloading",
                    "Downloading the compressed evidence archive.",
                    bytes_total=archive_size,
                )
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
            _emit_progress(
                progress,
                started,
                "extracting",
                "Verifying and extracting the compressed archive beside the destination.",
                bytes_done=local_archive.stat().st_size,
                bytes_total=local_archive.stat().st_size,
            )
            _extract_safe(local_archive, downloaded)
            profile_record = _read_optional_mapping(downloaded / "export-profile.json")
            run_root = downloaded / "runs"
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
            store = ResultStore(run_root)
            _emit_progress(
                progress,
                started,
                "finalizing",
                "Building the atomic package and checksum inventory.",
            )
            if _has_execution_snapshot(run_root):
                execution_selector = self._execution_selector(record, store, expected_execution)
                result = store.export(
                    execution_selector,
                    destination,
                    supplementary=supplementary,
                    copy_published=False,
                    captured_status=record.state.value,
                    link_evidence=True,
                    profile=export_profile,
                    omissions=tuple(
                        value
                        for value in profile_record.get("omitted", ())
                        if isinstance(value, Mapping)
                    ),
                    product_root=downloaded / "product-registry",
                )
            else:
                result = _export_pre_execution_snapshot(
                    destination,
                    name=work.name,
                    record=record,
                    supplementary=supplementary,
                    profile=export_profile,
                )
        output = {
            **result,
            "work_id": work.work_id,
            "job_id": record.job_id,
            "cluster": record.cluster,
            "attempt_state": record.state.value,
            "profile": export_profile,
        }
        _emit_progress(
            progress,
            started,
            "complete",
            f"Export complete: {output['path']}",
            terminal=True,
        )
        return output

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
        planned: list[Mapping[str, Any]] = []
        for path in sorted(store.root.glob("*/execution-*/execution.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(value, Mapping):
                planned.append(value)
        if expected is not None and any(value.get("execution_id") == expected for value in planned):
            return expected
        records = store.list()
        if expected is not None and any(value.get("execution_id") == expected for value in records):
            return expected
        matching = [
            value
            for value in records
            if any(
                isinstance(run, Mapping) and run.get("job_id") == record.job_id
                for run in value.get("runs", ())
            )
        ]
        if len(matching) != 1:
            matching = list(records)
        if len(matching) == 1:
            return str(matching[0]["execution_id"])
        if len(planned) == 1:
            selected = planned[0].get("execution_id")
            if isinstance(selected, str) and selected:
                return selected
        if len(matching) != 1:
            raise RuntimeError(
                "Downloaded evidence does not identify exactly one Execution snapshot; "
                f"found {len(planned)} planned and {len(matching)} finalized candidates."
            )
        raise AssertionError("unreachable")


def _emit_progress(
    callback: ExportProgress | None,
    started: float,
    phase: str,
    message: str,
    *,
    bytes_done: int | None = None,
    bytes_total: int | None = None,
    terminal: bool = False,
) -> None:
    """Publish bounded factual progress without making export depend on presentation."""
    if callback is None:
        return
    callback(
        {
            "phase": phase,
            "message": message,
            "elapsed_seconds": max(time.monotonic() - started, 0.0),
            "bytes_done": bytes_done,
            "bytes_total": bytes_total,
            "terminal": terminal,
        }
    )


def _archive_metadata(output: str) -> tuple[int | None, int | None]:
    """Read compressed and extracted sizes emitted by the remote packager."""
    try:
        value = json.loads(output.strip().splitlines()[-1])
    except (IndexError, TypeError, ValueError):
        return None, None
    if not isinstance(value, Mapping):
        return None, None

    def size(name: str) -> int | None:
        item = value.get(name)
        return (
            int(item)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0
            else None
        )

    return size("size_bytes"), size("uncompressed_bytes")


def _read_optional_mapping(path: Path) -> Mapping[str, Any]:
    """Read optional packager metadata without making old archives incompatible."""
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _safe_destination_parent(destination: str | Path) -> Path:
    """Create and verify the user-owned local parent before allocating temporary bytes."""
    authored = Path(destination).expanduser()
    if authored.is_symlink():
        raise ValueError(f"Export destination cannot be a symlink: {authored}")
    parent = authored.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"Export destination is not a safe directory: {parent}")
    return parent


def _has_execution_snapshot(root: Path) -> bool:
    """Return whether the archive contains planned or finalized Execution evidence."""
    return any(root.glob("*/execution-*/execution.json")) or any(
        root.glob("*/execution-*/result.json")
    )


def _export_pre_execution_snapshot(
    destination: str | Path,
    *,
    name: str,
    record: JobRecord,
    supplementary: Mapping[str, Path],
    profile: str = "default",
) -> dict[str, Any]:
    """Export honest control-plane evidence before an Execution has been planned."""
    captured_at = datetime.now(timezone.utc)
    parent = _safe_destination_parent(destination)
    folder = parent / (
        f"{_portable_name(name)}--{_portable_name(record.job_id)}--snapshot-"
        f"{captured_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    if folder.exists() or folder.is_symlink():
        raise FileExistsError(f"Export destination already exists: {folder}")
    stage = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=parent))
    warnings = [
        "The Study had not persisted an Execution planning manifest at capture time; this "
        "package contains only the available submitted configuration and control-plane evidence."
    ]
    try:
        for label, source in sorted(supplementary.items()):
            _copy_snapshot_tree(source, stage / _portable_name(label))
        (stage / "README.txt").write_text(
            "LambdaForge portable Study snapshot\n"
            "===================================\n\n"
            f"Captured control-plane state: {record.state.value}.\n"
            "This is a point-in-time snapshot, not a final scientific result. No Execution "
            "planning manifest had been published yet, so Run evidence and final analysis are "
            "not available in this capture.\n",
            encoding="utf-8",
        )
        inventory = _snapshot_inventory(stage)
        atomic_json(
            stage / "manifest.json",
            {
                "lambdaforge_export_version": 2,
                "created_at_utc": captured_at.isoformat(),
                "name": name,
                "execution_id": None,
                "job_id": record.job_id,
                "status": record.state.value,
                "attempt_state": record.state.value,
                "execution_status": None,
                "export_kind": "snapshot",
                "profile": profile,
                "finalized": False,
                "inventory": inventory,
                "inventory_scope": "all regular package files except manifest.json itself",
                "file_count": len(inventory),
                "size_bytes": sum(int(item["size_bytes"]) for item in inventory),
                "warnings": warnings,
            },
        )
        stage.replace(folder)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "status": "exported",
        "name": name,
        "execution_id": None,
        "captured_state": record.state.value,
        "execution_status": None,
        "export_kind": "snapshot",
        "finalized": False,
        "path": str(folder),
        "manifest": str(folder / "manifest.json"),
        "analysis_report": None,
        "file_count": len(inventory),
        "size_bytes": sum(int(item["size_bytes"]) for item in inventory),
        "warnings": warnings,
    }


def _portable_name(value: str) -> str:
    selected = "".join(
        character if character.isalnum() or character in "._-" else "-" for character in value
    )
    return selected.strip(".-")[:120] or "study"


def _copy_snapshot_tree(source: Path, destination: Path) -> None:
    """Copy already safely extracted regular evidence without following links."""
    if source.is_symlink() or not source.exists():
        raise ValueError(f"Unsafe Study snapshot evidence: {source}")
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unsafe Study snapshot evidence: {path}")
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        else:
            raise ValueError(f"Unsupported Study snapshot evidence: {path}")


def _snapshot_inventory(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"Portable export contains an unsafe symlink: {path}")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        output.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    return output


def _extract_safe(archive: Path, destination: Path) -> None:
    """Extract only regular ZIP members below the destination."""
    root = destination.resolve()
    with zipfile.ZipFile(archive, "r") as package:
        names: set[str] = set()
        for member in package.infolist():
            path = Path(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or path.as_posix() != member.filename.rstrip("/")
                or path.as_posix() in names
            ):
                raise ValueError(f"Unsafe path in Study export archive: {member.filename}")
            names.add(path.as_posix())
            mode = member.external_attr >> 16
            if mode and (mode & 0o170000) not in {0, 0o040000, 0o100000}:
                raise ValueError(
                    f"Unsafe non-regular entry in Study export archive: {member.filename}"
                )
            target = (root / path).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Study export entry escaped its destination: {member.filename}")
        for member in package.infolist():
            target = root / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member, "r") as source, target.open("wb") as destination_stream:
                shutil.copyfileobj(source, destination_stream, length=1024 * 1024)


_ARCHIVE_SCRIPT = r"""
import hashlib,json,os,re,sys,zipfile
from pathlib import Path

product_root=None
if "--product-root" in sys.argv:
 index=sys.argv.index("--product-root")
 if index+2!=len(sys.argv): raise SystemExit("invalid product-root export arguments")
 product_root=Path(sys.argv[index+1]); del sys.argv[index:]
authored_job=Path(sys.argv[1]); job=authored_job.resolve(); output=Path(sys.argv[2])
expected="" if sys.argv[3]=="-" else sys.argv[3]; job_id=sys.argv[4]; profile=sys.argv[5]
if profile not in {"default","full"}: raise SystemExit("invalid export profile")
if (authored_job.is_symlink() or authored_job.absolute() != job
    or job.name.startswith("job-") is False or not job.is_dir()):
 raise SystemExit("invalid owned Job root")
work=job/"work"; runs=work/".lambdaforge"/"runs"
if runs.exists() and (not runs.is_dir() or runs.is_symlink()):
 raise SystemExit("persisted Execution evidence is unsafe")

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

executions=[] if not runs.is_dir() else [
 candidate for candidate in sorted(runs.glob("*/execution-*"))
 if candidate.is_dir() and not candidate.is_symlink()
]
if len(sys.argv)>6:
 recovered=Path(sys.argv[6]); owner_work=Path(sys.argv[7])
 if (not recovered.is_absolute() or recovered.resolve()!=recovered
     or not recovered.is_relative_to(owner_work)):
  raise SystemExit("invalid recovered Execution owner")
 metadata=json.loads((recovered/"execution.json").read_text())
 if metadata.get("execution_id")!=recovered.name:
  raise SystemExit("recovered Execution identity mismatch")
 executions=[recovered]
selected=[]
if expected:
 for execution in executions:
  metadata=execution/"execution.json"
  try: value=json.loads(metadata.read_text(encoding="utf-8"))
  except Exception: value={}
  if execution.name==expected or value.get("execution_id")==expected:
   selected=[execution]; break
if not selected:
 for execution in executions:
  manifest=execution/"result.json"
  try: value=json.loads(manifest.read_text(encoding="utf-8"))
  except Exception: continue
  if any(isinstance(run,dict) and run.get("job_id")==job_id for run in value.get("runs",[])):
   selected.append(execution)
if not selected and len(executions)==1: selected=executions
if not selected and len(executions)>1:
 raise SystemExit(f"could not identify one Execution snapshot (found {len(executions)})")
if len(selected)>1:
 raise SystemExit(f"could not identify one Execution snapshot (found {len(selected)})")
execution=selected[0] if selected else None
if execution is not None:
 entries.append((execution,f"runs/{execution.parent.name}/{execution.name}"))

# Only products published by this Execution, not all other Studies' catalogs or caches.
# The receiving native ProductBundle service validates schemas, content and origins before sealing.
if execution is not None and (execution/"products.json").exists():
 publication_path=execution/"products.json"; check(publication_path)
 if publication_path.stat().st_size>512*1024:
  raise SystemExit("oversized product publication record")
 publication=json.loads(publication_path.read_text())
 if publication.get("product_publication_version")!=1:
  raise SystemExit("invalid product publication record")
 published=[item for item in publication.get("items",[]) if item.get("status")=="published"]
 request=job/"request.json"
 if request.is_file():
  check(request)
  request_value=json.loads(request.read_text())
  if request_value.get("product_root"): product_root=Path(request_value["product_root"])
 if published and (product_root is None or not product_root.is_absolute()
     or product_root.resolve()!=product_root):
  raise SystemExit("published products require their exact owned project product root")
 seen_products=set()
 for item in published:
  identity=item.get("content_id","")
  if not isinstance(identity,str) or not re.fullmatch(r"sha256:[0-9a-f]{64}",identity):
   raise SystemExit("invalid published product content identity")
  key=identity.split(":",1)[1]; obj=product_root/"objects"/key
  product_name=item.get("name")
  if not isinstance(product_name,str) or not product_name:
   raise SystemExit("invalid published product name")
  alias=hashlib.sha256(product_name.encode("utf-8")).hexdigest()+".json"
  entries.append((product_root/"names"/alias,"product-registry/names/"+alias))
  if key in seen_products: continue
  seen_products.add(key)
  metadata=obj/"manifest.json"; check(metadata)
  if metadata.stat().st_size>512*1024: raise SystemExit("oversized published product manifest")
  value=json.loads(metadata.read_text())
  entries.append((metadata,"product-registry/objects/"+key+"/manifest.json"))
  entries.append((obj/"provenance","product-registry/objects/"+key+"/provenance"))
  for artifact in value.get("artifacts",[]):
   raw=artifact.get("path"); relative=Path(raw) if isinstance(raw,str) else None
   if (relative is None or relative.is_absolute() or ".." in relative.parts
       or "\\" in raw or relative.as_posix()!=raw):
    raise SystemExit("unsafe published product artifact path")
   entries.append((obj/relative,"product-registry/objects/"+key+"/"+raw))

seen=set()
payloads=[]
if execution is not None:
 for manifest in [execution/"result.json",*sorted((execution/"runs").glob("*/result.json"))]:
  try: result=json.loads(manifest.read_text(encoding="utf-8"))
  except Exception: continue
  payloads.extend(result.get("runs",[]) if isinstance(result.get("runs"),list) else [result])
for run_index,run in enumerate(payloads):
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

raw_resource_names={
 "scheduler-trace.jsonl","resource-events.jsonl","observations.jsonl",
 "exploration-evaluations.jsonl","admission-decisions.jsonl","host-history.jsonl"
}
metric_names={"metrics.jsonl","training-metrics.jsonl"}
omitted=[]
resource_summary={"profile":profile,"streams":{}}

def digest_file(path):
 import hashlib
 digest=hashlib.sha256(); size=0
 with path.open("rb") as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b""):
   digest.update(chunk); size+=len(chunk)
 return size,digest.hexdigest()

def jsonl_summary(path):
 count=0; first=None; last=None; reasons={}
 with path.open("r",encoding="utf-8",errors="replace") as stream:
  for line in stream:
   try: item=json.loads(line)
   except Exception: continue
   if not isinstance(item,dict): continue
   count+=1
   if first is None: first=item
   last=item
   reason=item.get("reason") or item.get("decision") or item.get("event")
   if isinstance(reason,str): reasons[reason]=reasons.get(reason,0)+1
 return {"records":count,"first":first,"last":last,"reason_counts":reasons}

def sampled_jsonl(path,limit=4096):
 lines=path.read_bytes().splitlines(keepends=True)
 if len(lines)<=limit: return None,len(lines),len(lines)
 # Deterministic endpoint-preserving uniform sample. The final line is always retained.
 indexes={round(index*(len(lines)-1)/(limit-1)) for index in range(limit)}
 return b"".join(lines[index] for index in sorted(indexes)),len(lines),len(indexes)

def add(archive,source,name):
 check(source)
 if source.is_file():
  if name.startswith("product-registry/"):
   archive.write(source,arcname=name); return
  if profile=="default" and source.name in raw_resource_names:
   size,digest=digest_file(source); summary=jsonl_summary(source)
   omitted.append({"path":name,"reason":"summarized high-frequency resource telemetry",
    "size_bytes":size,"sha256":digest,"records":summary["records"]})
   resource_summary["streams"][name]=summary
   return
  if profile=="default" and source.name in metric_names:
   sampled,total,kept=sampled_jsonl(source)
   if sampled is not None:
    size,digest=digest_file(source)
    omitted.append({"path":name,"reason":"downsampled metric trajectory",
     "size_bytes":size,"sha256":digest,"source_records":total,"exported_records":kept})
    archive.writestr(name,sampled); return
  archive.write(source,arcname=name)
  return
 archive.writestr(name.rstrip("/")+"/",b"")
 for child in sorted(source.rglob("*"),key=lambda value:value.relative_to(source).as_posix()):
  relative=child.relative_to(source).as_posix(); arcname=name.rstrip("/")+"/"+relative
  if child.is_dir(): archive.writestr(arcname.rstrip("/")+"/",b"")
  elif arcname.startswith("product-registry/"): archive.write(child,arcname=arcname)
  elif profile=="default" and child.name in raw_resource_names:
   size,digest=digest_file(child); summary=jsonl_summary(child)
   omitted.append({"path":arcname,"reason":"summarized high-frequency resource telemetry",
    "size_bytes":size,"sha256":digest,"records":summary["records"]})
   resource_summary["streams"][arcname]=summary
  elif profile=="default" and child.name in metric_names:
   sampled,total,kept=sampled_jsonl(child)
   if sampled is None: archive.write(child,arcname=arcname)
   else:
    size,digest=digest_file(child)
    omitted.append({"path":arcname,"reason":"downsampled metric trajectory",
     "size_bytes":size,"sha256":digest,"source_records":total,"exported_records":kept})
    archive.writestr(arcname,sampled)
  else: archive.write(child,arcname=arcname)

with zipfile.ZipFile(
 output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True
) as archive:
 for source,name in entries: add(archive,source,name)
 archive.writestr("export-profile.json",json.dumps({
  "profile":profile,"omitted":omitted,"resource_telemetry_summary":resource_summary
 },sort_keys=True,separators=(",",":")))
 archive.writestr("study/resource-telemetry-summary.json",json.dumps(
  resource_summary,sort_keys=True,separators=(",",":")))
 uncompressed=sum(item.file_size for item in archive.infolist())
print(json.dumps({
 "archive":str(output),"entries":len(entries),"size_bytes":output.stat().st_size,
 "uncompressed_bytes":uncompressed,
 "compression":"zip-deflate-3"
}))
"""


__all__ = ["StudyExportService"]
