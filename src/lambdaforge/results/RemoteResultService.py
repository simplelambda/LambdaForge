"""Small, allowlisted remote result synchronization service."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.results.ResultSyncResult import ResultSyncResult


class RemoteResultService:
    """Retrieve metadata/metrics/manifests/small plots without copying heavy runs."""

    def __init__(
        self,
        jobs: JobService | None = None,
        catalog: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
        root: str | Path | None = None,
        *,
        max_file_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.jobs = jobs or JobService(self.catalog, factory=self.factory)
        if root is None:
            state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            root = state / "lambdaforge" / "remote-results"
        self.root = Path(root).expanduser().resolve()
        self.max_file_bytes = int(max_file_bytes)

    def sync(self, job_id: str) -> ResultSyncResult:
        """Synchronize only allowlisted small scientific evidence for one job."""
        record = self.jobs.get(job_id)
        profile = self.catalog.get(record.cluster)
        transport = self.factory.transport(profile)
        python = self._python(record.command, profile.python)
        script = (
            "import json, pathlib, sys; "
            "r=pathlib.Path('.').resolve(); m=int(sys.argv[1]); "
            "names={'result.json','metrics.csv','environment.json','config.yaml',"
            "'dataset-artifact.json','preprocessing-manifest.json','summary.json',"
            "'events.jsonl','aggregate.json','aggregate.csv'}; "
            "ext={'.png','.svg','.pdf','.html'}; "
            "out=[{'path':p.relative_to(r).as_posix(),'size':p.stat().st_size} "
            "for p in r.rglob('*') if p.is_file() and not p.is_symlink() "
            "and p.stat().st_size<=m and (p.name in names or "
            "((p.suffix in ext or p.name.endswith('.plot.json')) and 'plots' in p.parts))]; "
            "print(json.dumps(sorted(out,key=lambda x:x['path'])))"
        )
        listing = transport.run(
            (python, "-c", script, str(self.max_file_bytes)), cwd=record.work_dir
        )
        if listing.returncode:
            raise RuntimeError(f"Could not list remote result metadata: {listing.stderr.strip()}")
        payload = json.loads(listing.stdout or "[]")
        if not isinstance(payload, list):
            raise TypeError("Remote result listing did not return a JSON list.")
        destination = self.root / job_id
        files: list[str] = []
        total = 0
        for value in payload:
            if not isinstance(value, dict):
                continue
            relative = PurePosixPath(str(value.get("path", "")))
            size = int(value.get("size", -1))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or size < 0
                or size > self.max_file_bytes
            ):
                raise ValueError(f"Unsafe remote result entry: {value!r}.")
            local = destination / Path(*relative.parts)
            local.parent.mkdir(parents=True, exist_ok=True)
            transport.get(str(PurePosixPath(record.work_dir) / relative), local)
            if local.stat().st_size != size:
                raise OSError(f"Remote result size changed while synchronizing {relative}.")
            files.append(relative.as_posix())
            total += size
        metadata = dict(record.metadata)
        metadata.update(
            {
                "synced_result_root": str(destination),
                "synced_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.jobs.store.write(record.with_updates(metadata=metadata))
        return ResultSyncResult(job_id, str(destination), tuple(files), total)

    @staticmethod
    def _python(command: tuple[str, ...], fallback: str) -> str:
        try:
            module_index = command.index("-m")
        except ValueError:
            return fallback
        return command[module_index - 1] if module_index > 0 else fallback
