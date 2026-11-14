"""Durable Dataset transfers using the existing Job/scheduler lifecycle, not scientific Works."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.OperationalHealth import OperationalHealth, timestamp
from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.data.DatasetMaterializationPlan import DatasetMaterializationPlan
from lambdaforge.data.DatasetPlacement import DatasetPlacement
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json

if TYPE_CHECKING:
    from lambdaforge.data.DatasetService import DatasetService


class DatasetReplicationService:
    """Submit compressed copies and reconnect through native owned Job records."""

    def __init__(self, datasets: DatasetService, jobs: JobService | None = None) -> None:
        self.datasets = datasets
        self.jobs = jobs or JobService(datasets.clusters, factory=datasets.factory)

    def enqueue(self, selector: str, **options: Any) -> DatasetMaterializationPlan:
        """Preview first; acknowledge a durable Job before allowing the caller to disconnect."""
        options.pop("apply", None)
        options.pop("progress", None)
        approved_route = options.pop("approved_route", None)
        approved_diagnostics = options.pop("approved_diagnostics", None)
        site_password = options.get("site_password")
        plan = self.preview(selector, **options)
        if approved_route is not None and plan.transfer_route != approved_route:
            raise ValueError(
                "Transfer route changed after preview; review a new plan before copying."
            )
        if approved_diagnostics is not None:
            plan = replace(
                plan,
                route_diagnostics=tuple(
                    json.loads(SecretRedactor.redact(json.dumps(dict(note))))
                    for note in approved_diagnostics[:4]
                ),
            )
        if plan.relay_consent_required:
            raise ValueError(
                "Controller relay was not approved. Review diagnostics "
                "and choose --route relay explicitly."
            )
        if plan.action != "REPLICATE":
            return self.datasets.replicate(selector, **options, apply=True)
        record = self.datasets.resolve_placement(selector, str(plan.source_cluster)).record
        if record.dataset_id != plan.content_id:
            raise ValueError("Source Dataset identity changed after replication preview.")
        profiles = self._execution_profiles()
        source = profiles[str(plan.source_cluster)]
        direct = plan.transfer_route in {"direct-ssh", "direct-ssh-pull"}
        host = (
            profiles[plan.target_cluster]
            if plan.transfer_route == "direct-ssh-pull"
            else source
            if direct
            else profiles["local"]
        )
        # Only a local relay needs to reconnect using controller credentials after UI exit.
        if not direct:
            for name in (plan.source_cluster, plan.target_cluster):
                profile = self.datasets.clusters.get(str(name))
                if (
                    profile.transport == "ssh"
                    and profile.auth.mode == "password"
                    and not profile.auth.credential
                ):
                    raise ValueError(
                        "Detached replication cannot prompt for passwords. Save a credential "
                        "reference or configure OpenSSH authentication."
                    )
        self.jobs.store.root.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            self.jobs.store.root / ".dataset-replications.lock",
            shared=False,
            timeout_seconds=30,
            poll_interval_seconds=0.05,
        ):
            for previous in self.jobs.list(refresh=False):
                old = previous.metadata.get("dataset_replication", {})
                if (
                    previous.job_type == "dataset-replication"
                    and all(
                        old.get(key) == value
                        for key, value in {
                            "dataset": record.key,
                            "content_id": record.dataset_id,
                            "source": plan.source_cluster,
                            "destination": plan.target_cluster,
                        }.items()
                    )
                    and (
                        not previous.state.terminal
                        or previous.metadata.get("failure_phase") == "submission"
                    )
                ):
                    if previous.metadata.get("failure_phase") == "submission":
                        raise RuntimeError(
                            f"Previous transfer {previous.job_id} failed before acknowledgement. "
                            "Inspect its provider/logs and resolve ownership before submitting "
                            "another copy; lack of acknowledgement is not proof it never started."
                        )
                    # Unknown/offline is not evidence that launching a duplicate is safe.
                    observed = self.jobs.get(previous.job_id)
                    if not observed.state.terminal:
                        return replace(
                            plan,
                            applied=True,
                            job_id=previous.job_id,
                            requires_controller_online=bool(old.get("requires_controller_online")),
                        )
            transport = self.datasets.factory.transport(host)
            python = (
                sys.executable
                if host.transport == "local"
                else self.datasets._python(host.name, transport)
            )
            host = replace(host, python=python, python_runtime=None, environment="existing")
            catalog = ClusterCatalog(
                {
                    **profiles,
                    host.name: host,
                },
                # Profiles above are already project-derived. JobStore retains the project
                # owner; applying ProjectContext again would append projects/ID twice.
            )
            jobs = JobService(catalog, store=self.jobs.store, factory=self.datasets.factory)
            job_id = jobs.new_id()
            assert host.storage is not None
            work = PurePosixPath(host.storage.job_root) / job_id / "work"
            request_path = work / "replication.json"
            request: dict[str, Any] = {
                "dataset": record.key,
                "content_id": record.dataset_id,
                "source": plan.source_cluster,
                "destination": plan.target_cluster,
                "route": plan.transfer_route,
            }
            command: tuple[str, ...]
            if direct:
                target = profiles[plan.target_cluster]
                assert target.storage is not None and target.storage.dataset_root is not None
                target_python = self.datasets._python(
                    target.name, self.datasets.factory.transport(target)
                )
                helper = Path(__file__).with_name("DatasetTransfer.py").read_text(encoding="utf-8")
                receive = {
                    "root": target.storage.dataset_root,
                    "record": record.to_dict(),
                    "cluster": target.name,
                    "registry": str(PurePosixPath(target.storage.state_root) / "datasets.json"),
                    "storage": target.storage.to_dict(),
                    "report_phases": True,
                }
                placement = self.datasets.resolve_placement(
                    selector, str(plan.source_cluster)
                ).placement
                assert placement is not None
                request.update(
                    {
                        "source_root": placement.root,
                        "receiver": list(
                            self.datasets._site_ssh(
                                target,
                                (
                                    *target.command_prefix,
                                    target_python,
                                    "-c",
                                    helper,
                                    "receive",
                                    json.dumps(receive),
                                ),
                            )
                        ),
                    }
                )
                if site_password is not None:
                    if plan.transfer_route != "direct-ssh":
                        raise ValueError(
                            "A destination password authorizes source→destination only."
                        )
                    request["peer_authentication"] = "password"
                    request["peer"] = {
                        "host": target.host,
                        "user": target.user,
                        "port": target.port,
                    }
                    request["receiver"] = list(
                        (
                            *target.command_prefix,
                            target_python,
                            "-c",
                            helper,
                            "receive",
                            json.dumps(receive),
                        )
                    )
                if plan.transfer_route == "direct-ssh-pull":
                    source_python = self.datasets._python(
                        source.name, self.datasets.factory.transport(source)
                    )
                    request.pop("receiver")
                    request["direction"] = "pull"
                    request["receive_options"] = receive
                    request["sender"] = list(
                        self.datasets._site_ssh(
                            source,
                            (
                                *source.command_prefix,
                                source_python,
                                "-c",
                                helper,
                                "pack",
                                placement.root,
                                record.dataset_id,
                            ),
                        )
                    )
                command = (python, "-c", helper, "direct-job", str(request_path))
            else:
                request["registry"] = str(self.datasets.registry.path)
                request["profiles"] = {
                    n: profiles[n].to_dict()
                    for n in {"local", str(plan.source_cluster), plan.target_cluster}
                }
                command = (
                    python,
                    "-m",
                    "lambdaforge.data.DatasetReplicationWorker",
                    str(request_path),
                )
            made = transport.run(("mkdir", "-p", str(work)), timeout=30)
            if made.returncode:
                raise RuntimeError("Cannot stage replication request: " + made.stderr[-1000:])
            with tempfile.TemporaryDirectory(prefix="lf-replication-") as directory:
                path = Path(directory) / "replication.json"
                atomic_write_json(path, request)
                path.chmod(0o600)
                transport.put(path, str(request_path))
            handle = jobs.submit(
                command,
                cluster=host.name,
                resources=ResourceRequest(cpu_cores=1),
                work_dir=str(work),
                job_id=job_id,
                job_type="dataset-replication",
                **({"stdin_data": site_password.encode()} if site_password is not None else {}),
                metadata={
                    "name": f"Replicate {record.key}",
                    "datasets": [record.key],
                    "dataset_ids": [record.dataset_id],
                    "dataset_replication": {
                        "dataset": record.key,
                        "content_id": record.dataset_id,
                        "source": plan.source_cluster,
                        "destination": plan.target_cluster,
                        "route": plan.transfer_route,
                        "requires_controller_online": not direct,
                        "authentication": "password" if site_password is not None else "site",
                        "route_diagnostics": list(plan.route_diagnostics),
                        "data_path": list(plan.data_path),
                        "uncompressed_bytes": plan.estimated_bytes,
                    },
                },
            )
            return replace(
                plan, job_id=handle.job_id, applied=True, requires_controller_online=not direct
            )

    def _execution_profiles(self) -> dict[str, ClusterProfile]:
        """Pin local path owners before staging or serializing a detached command.

        Reuse the native submission anchoring, but do it *before* deriving work/request paths.
        The worker's cwd cannot define the meaning of controller project configuration. SSH
        workspace/storage/interpreter paths belong to the remote host and remain untouched.
        """
        project = (
            self.datasets.clusters.project or self.jobs.store.project or ProjectContext.discover()
        )
        profiles = {}
        for name in self.datasets.clusters.names():
            profile = self.datasets.clusters.get(name)
            if profile.transport == "local":
                profile = JobService._submission_profile(
                    profile, config_path=None, work_dir=project.root
                )
                python = profile.python
                if "/" in python or "\\" in python or python.startswith("~"):
                    executable = Path(python).expanduser()
                    if not executable.is_absolute():
                        executable = project.root / executable
                    python = str(executable.absolute())
                profile = replace(profile, python=python, python_runtime=None)
            profiles[name] = profile
        return profiles

    def preview(self, selector: str, **options: Any) -> DatasetMaterializationPlan:
        """Explain durable placement and host dependency without launching or reserving Jobs."""
        options.pop("apply", None)
        plan = self.datasets.replicate(selector, **options)
        return replace(
            plan,
            requires_controller_online=(
                plan.action == "REPLICATE"
                and plan.transfer_route not in {"direct-ssh", "direct-ssh-pull"}
            ),
        )

    def list(self, selector: str | None = None, *, refresh: bool = True) -> list[dict[str, Any]]:
        """Bounded metadata-only history; no archive, asset or checksum scans."""
        records = [
            r
            for r in self.jobs.list(refresh=False)
            if r.job_type == "dataset-replication"
            and (
                selector is None
                or r.metadata.get("dataset_replication", {}).get("dataset") == selector
            )
        ]
        records.sort(key=lambda record: record.created_at_utc, reverse=True)
        output = []
        for record in records[:20]:
            try:
                needs_observation = refresh and (
                    not record.state.terminal
                    or not record.metadata.get("dataset_replication_outcome")
                    or (
                        record.state.value in {"failed", "timeout", "cancelled"}
                        and not record.metadata.get("dataset_replication_outcome", {}).get(
                            "failure_observed"
                        )
                    )
                )
                output.append(self.status(record.job_id, refresh=needs_observation))
            except Exception as error:
                # An unreachable host is not a failed transfer. Retain native last observation
                # and explicitly report that progress could not be refreshed.
                row = self.status(record.job_id, refresh=False)
                row["observation_error"] = SecretRedactor.redact(f"{type(error).__name__}: {error}")
                if not record.state.terminal:
                    row["state_confirmed"] = False
                    row["display_state"] = f"unconfirmed/stale ({row['state']})"
                output.append(row)
        return output

    def status(self, job_id: str, *, refresh: bool = True) -> dict[str, Any]:
        """Read authoritative native state and bounded transfer progress/result metadata."""
        record = self.jobs.get(job_id, refresh=refresh, include_study=False)
        if record.job_type != "dataset-replication":
            raise ValueError("The selected Job is not a Dataset replication.")
        data = (
            self._metadata(record)
            if refresh
            else dict(record.metadata.get("dataset_replication_outcome") or {})
        )
        descriptor = dict(record.metadata["dataset_replication"])
        result = data.get("result", {})
        if refresh and record.state.value == "succeeded" and not result.get("verified"):
            raise ValueError(
                "Replication Job exited successfully but its verified acknowledgement is missing. "
                "Inspect its logs/destination before considering the replica complete."
            )
        if result:
            if any(
                result.get(key) != descriptor[key]
                for key in ("dataset", "content_id", "destination")
            ):
                raise ValueError("Replication result does not match its pinned Job identity.")
            if result.get("verified"):
                placement = DatasetPlacement.from_mapping(result["placement"])
                if placement.cluster != descriptor["destination"] or not placement.verified:
                    raise ValueError("Replication returned invalid placement evidence.")
                if result.get("root", placement.root) != placement.root:
                    raise ValueError("Replication placement does not match its verified root.")
                # A controller disconnected during a direct transfer learns the destination
                # only now. Never select/relabel a conflicting immutable controller identity.
                try:
                    current = self.datasets.registry.get(descriptor["dataset"])
                    if current.dataset_id != descriptor["content_id"]:
                        raise ValueError(
                            "Controller Dataset identity changed; reconcile explicitly."
                        )
                    if placement not in current.placements:
                        self.datasets.registry.register(
                            replace(current, placements=(*current.placements, placement))
                        )
                except (KeyError, ValueError) as error:
                    data["reconciliation_error"] = str(error)
        failed = record.state.value in {"failed", "timeout", "cancelled"}
        logs_command = f"lf jobs logs {job_id} --tail 100"
        failure = SecretRedactor.redact(
            data.get("stderr_tail") or data.get("error") or record.stderr or ""
        ).strip()
        if failed:
            last_line = failure.splitlines()[-1] if failure else ""
            failure = (
                (
                    last_line
                    if len(last_line) <= 2000
                    else last_line[:900] + " … " + last_line[-1000:]
                )
                if failure
                else (
                    f"Transfer worker {record.state.value}; no stderr diagnostic available. "
                    f"See {logs_command}."
                )
            )
        if refresh and record.state.terminal:
            # Cache bounded terminal metadata for listings; explicit status(refresh=True)
            # still re-reads the host. Never rewrite the Job outcome or provider observation.
            progress = data.get("progress", {})
            self.jobs.store.merge_metadata(
                job_id,
                {
                    "dataset_replication_outcome": {
                        "progress": {
                            key: progress[key][:1024]
                            if isinstance(progress[key], str)
                            else progress[key]
                            for key in (
                                "phase",
                                "message",
                                "compressed_bytes",
                                "updated_at_utc",
                                "last_byte_progress_at_utc",
                            )
                            if key in progress
                        },
                        "result": {
                            key: result[key]
                            for key in (
                                "dataset",
                                "content_id",
                                "destination",
                                "verified",
                                "root",
                                "placement",
                            )
                            if key in result
                        },
                        "error": failure if failed else "",
                        "failure_observed": failed,
                    }
                },
            )
        operational = OperationalHealth.describe(record)
        confirmed = record.state.terminal or (
            operational.get("freshness") == "fresh"
            and not operational.get("last_error")
            and operational.get("reason") == "PROVIDER_CONFIRMED"
        )
        progress = data.get("progress", {})
        last_bytes = timestamp(progress.get("last_byte_progress_at_utc"))
        phase = progress.get("phase", "waiting")
        return {
            **descriptor,
            "job_id": job_id,
            "host": record.cluster,
            "state": record.state.value,
            "state_confirmed": confirmed,
            "display_state": record.state.value
            if confirmed
            else f"unconfirmed/stale ({operational.get('execution_state', record.state.value)})",
            "operational": operational,
            "last_confirmed_at_utc": operational.get("last_confirmed_at_utc"),
            "observation_age_seconds": operational.get("confirmation_age_seconds"),
            "progress": progress,
            "last_byte_progress_at_utc": progress.get("last_byte_progress_at_utc"),
            "byte_progress_age_seconds": max(
                0.0, (datetime.now(timezone.utc) - last_bytes).total_seconds()
            )
            if last_bytes
            else None,
            "activity": "verifying"
            if phase in {"verify-source", "verify-destination", "publication"}
            else "transferring-or-waiting-for-bytes"
            if phase == "transfer"
            else phase,
            "error": failure if failed else "",
            "logs_command": logs_command,
            "reconciliation_error": data.get("reconciliation_error"),
            "placement": result.get("placement")
            if result.get("verified") and not data.get("reconciliation_error")
            else None,
            "created_at_utc": record.created_at_utc,
        }

    def _metadata(self, record: JobRecord) -> dict[str, Any]:
        profile = self.jobs.catalog.get(record.cluster)
        root = PurePosixPath(self.jobs.job_root(record)) / record.job_id
        if PurePosixPath(record.work_dir).parent != root or not root.is_absolute():
            raise ValueError("Unsafe transfer Job metadata root.")
        script = """import json,sys,os,stat
from pathlib import Path
from lambdaforge.work.snapshot import validate_path
root=Path(sys.argv[1]); output={}
validate_path(root)
for name in ('progress','result'):
 p=root/(name+'.json')
 if p.is_symlink(): raise ValueError('Unsafe transfer metadata path')
 if p.is_file():
  with os.fdopen(os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)),'rb') as f:
   if not stat.S_ISREG(os.fstat(f.fileno()).st_mode): raise ValueError('Unsafe metadata file')
   raw=f.read(1048577)
  if len(raw)>1048576: raise ValueError('Transfer metadata exceeds bound')
  output[name]=json.loads(raw)
if sys.argv[2]=='1':
 p=root/'stderr.log'
 if p.is_symlink(): raise ValueError('Unsafe transfer log path')
 if p.is_file():
  with os.fdopen(os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)),'rb') as f:
   if not stat.S_ISREG(os.fstat(f.fileno()).st_mode): raise ValueError('Unsafe transfer log file')
   offset=max(0,os.fstat(f.fileno()).st_size-8192)
   f.seek(offset)
   raw=f.read(8192)
   if offset: raw=raw.partition(bytes([10]))[2]
   output['stderr_tail']=raw.decode('utf-8',errors='replace')
print(json.dumps(output))
"""
        # Failed workers can exit before writing any JSON. Read only a bounded operational
        # stderr tail, never dataset/model bytes. O_NOFOLLOW/ownership checks apply equally.
        failed = record.state.value in {"failed", "timeout", "cancelled"}
        response = self.datasets.factory.transport(profile).run(
            (
                *self.datasets.factory.reader_command(profile),
                "-c",
                script,
                str(root),
                "1" if failed else "0",
            ),
            timeout=20,
        )
        if response.returncode:
            raise RuntimeError("Cannot read transfer metadata: " + response.stderr[-1000:])
        value = json.loads(response.stdout)
        if not isinstance(value, dict):
            raise ValueError("Invalid transfer metadata response.")
        return value
