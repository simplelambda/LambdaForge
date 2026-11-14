"""Target-side execution queue admission through native transport and Job services."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.ExecutionQueue import ExecutionQueue
from lambdaforge.controlplane.SecretRedactor import SecretRedactor


class ExecutionQueueService:
    """Own queue metadata on the execution host, never a laptop-resident dispatcher."""

    def __init__(
        self,
        catalog: ClusterCatalog,
        factory: ControlPlaneFactory | None = None,
        *,
        source_dir: Path | None = None,
    ) -> None:
        self.catalog = catalog
        self.factory = factory or ControlPlaneFactory()
        self.source_dir = source_dir

    def invoke(
        self, cluster: str, operation: str, job_id: str = "", blockers: Sequence[str] = ()
    ) -> dict[str, Any]:
        profile = self.catalog.get(cluster)
        if profile.transport == "local":
            if os.name != "posix":
                raise ValueError("Sequential execution queues require a POSIX process host.")
            from lambdaforge.controlplane.JobService import JobService

            root = (
                self.catalog.project.root if self.catalog.project else self.source_dir or Path.cwd()
            )
            profile = JobService._submission_profile(profile, config_path=None, work_dir=root)
        if profile.scheduler != "local":
            raise ValueError(
                "Sequential execution queues currently support direct process clusters, "
                "not SLURM/Fleet. No fallback submission was made."
            )
        assert profile.storage is not None
        state, jobs = profile.storage.state_root, profile.storage.job_root
        if profile.transport == "local":
            state, jobs = str(Path(state).absolute()), str(Path(jobs).absolute())
        # Ship only the metadata authority; old immutable environments are never modified.
        code = (
            "from __future__ import annotations\n"
            "import hashlib,json,os,re,sys,time\nfrom pathlib import Path\n"
            "from uuid import uuid4\nfrom contextlib import contextmanager\n"
            "from collections.abc import Callable,Iterator,Mapping,Sequence\n"
            "from typing import Any\n" + inspect.getsource(ExecutionQueue) + "\n"
            "queue=ExecutionQueue(sys.argv[1],sys.argv[2],sys.argv[6])\n"
            "op=sys.argv[3]; job=sys.argv[4]\n"
            "if op=='register':\n"
            " blockers=set(json.loads(sys.argv[5]))\n"
            " settled={r['job_id'] for r in queue.snapshot()['entries']\n"
            "          if r['phase'] in {'aborted','released'}}\n"
            " if queue.jobs.is_dir():\n"
            "  for child in queue.jobs.iterdir():\n"
            "   if child.name==job or not child.name.startswith('job-'):continue\n"
            "   if child.name in settled:continue\n"
            "   if child.is_symlink():raise ValueError('Unsafe queue predecessor workspace')\n"
            "   state=child/'state.json'\n"
            "   if not child.is_dir():continue\n"
            "   if state.is_symlink():raise ValueError('Unsafe queue predecessor state')\n"
            "   if state.exists() and state.stat().st_size>1048576:\n"
            "    raise ValueError('Oversized predecessor state; inspect native Job ownership')\n"
            "   value=json.loads(state.read_text()) if state.is_file() else {}\n"
            "   if value and value.get('job_id')!=child.name:\n"
            "    raise ValueError('Queue predecessor ownership mismatch')\n"
            "   placement=value.get('cluster')\n"
            "   if placement is None:\n"
            "    req=child/'request.json'\n"
            "    if req.is_symlink():raise ValueError('Unsafe predecessor request')\n"
            "    if not req.is_file():\n"
            "     if value:raise ValueError('Missing predecessor placement ownership')\n"
            "     continue\n"
            "    if req.stat().st_size>1048576:raise ValueError('Oversized predecessor request')\n"
            "    owner=json.loads(req.read_text())\n"
            "    if owner.get('job_id')!=child.name or not owner.get('cluster'):\n"
            "     raise ValueError('Corrupt predecessor placement ownership')\n"
            "    placement=owner['cluster']\n"
            "   if placement!=queue.placement:continue\n"
            "   if (value.get('state') not in queue.TERMINAL or\n"
            "       (not value.get('scientific_cleanup_complete') and\n"
            "        (value.get('supervisor_identity') or value.get('process_identity')))):\n"
            "    blockers.add(child.name)\n"
            " print(json.dumps(queue.register(job,sorted(blockers))))\n"
            "elif op=='abort':queue.abort_preparation(job);print('{}')\n"
            "elif op=='snapshot':\n"
            " document=queue.snapshot()\n"
            " for row in document['entries']:\n"
            "  if row['phase'] in {'aborted','released'}:continue\n"
            "  directory=queue.jobs/row['job_id'];p=directory/'state.json'\n"
            "  if directory.is_symlink() or p.is_symlink():\n"
            "   raise ValueError('Unsafe queue Job state path')\n"
            "  if not p.is_file():\n"
            "   if row['phase']!='pending':row['last_provider_state']='unknown'\n"
            "   continue\n"
            "  try:\n"
            "   if p.stat().st_size>1048576:raise ValueError('Oversized Job metadata')\n"
            "   value=json.loads(p.read_text())\n"
            "   if not isinstance(value,dict) or value.get('job_id')!=row['job_id']:\n"
            "    raise ValueError('Job metadata ownership mismatch')\n"
            "   row['last_provider_state']=value.get('state','unknown')\n"
            "  except (OSError,ValueError) as error:\n"
            "   row['observation_error']=type(error).__name__\n"
            "   row['last_provider_state']='unavailable'\n"
            " print(json.dumps(document))\n"
            "else:raise ValueError('Unknown execution queue operation')\n"
        )
        transport = self.factory.transport(profile)
        result = transport.run(
            (
                *profile.command_prefix,
                profile.python,
                "-c",
                code,
                state,
                jobs,
                operation,
                job_id or "-",
                json.dumps(list(blockers)),
                cluster,
            ),
            timeout=30,
        )
        if result.returncode:
            diagnostic = SecretRedactor.redact(result.stderr[-4000:])
            raise RuntimeError(f"Execution queue {operation} failed: {diagnostic}")
        return json.loads(result.stdout)

    def register(self, cluster: str, job_id: str, blockers: Sequence[str]) -> dict[str, Any]:
        ticket = self.invoke(cluster, "register", job_id, blockers)
        return {
            "version": 1,
            "job_id": job_id,
            "ticket": ticket,
            "scope": "project/cluster",
            "policy": "FIFO; after any terminal outcome",
        }

    def abort(self, cluster: str, job_id: str) -> None:
        self.invoke(cluster, "abort", job_id)

    def snapshot(self, cluster: str) -> dict[str, Any]:
        return {"cluster": cluster, **self.invoke(cluster, "snapshot")}
