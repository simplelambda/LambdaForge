"""Scoped report asset sources; native providers and offline files share one integrity contract."""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

from lambdaforge.work.managed import owned_path
from lambdaforge.work.report_bundle import read_manifest, verified_asset


def reader_prelude(*, include_sections: bool = False) -> str:
    """Ship the current stdlib bundle reader to immutable older workers, never run consumer code."""
    import lambdaforge.work.report_bundle as module

    source = inspect.getsource(module)
    prefix = (
        "import sys, types\n"
        "_report_module=types.ModuleType('lambdaforge.work.report_bundle')\n"
        "sys.modules[_report_module.__name__]=_report_module\n"
        f"exec({source!r}, _report_module.__dict__)\n"
    )
    if include_sections:
        import lambdaforge.study_projection as projection

        prefix += (
            "_section_module=types.ModuleType('lambdaforge.study_projection')\n"
            "sys.modules[_section_module.__name__]=_section_module\n"
            f"exec({inspect.getsource(projection)!r}, _section_module.__dict__)\n"
        )
    return prefix


def selected_asset(source: Mapping[str, Any], identifier: str) -> tuple[Path, dict[str, Any]]:
    """Revalidate native Run artifact attestation before granting one declared member."""
    owner = Path(str(source["owner"]))
    if not owner.is_absolute() or owner.resolve() != owner:
        raise ValueError("Unsafe report owner root.")
    run_dir = owned_path(owner, str(source["run_dir"]), must_exist=True)
    record = owned_path(run_dir, "result.json", must_exist=True)
    if not record.is_file() or record.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Unsafe report Run metadata.")
    result = json.loads(record.read_text())
    matches = [
        a
        for a in result.get("artifacts", ())
        if isinstance(a, dict)
        and a.get("name") == source.get("artifact")
        and a.get("role") == "html-bundle"
    ]
    if len(matches) != 1:
        raise ValueError("Report bundle is not an unambiguous finalized artifact of this Run.")
    artifact = matches[0]
    attestation = artifact.get("metadata", {}).get("report_bundle")
    if attestation != source.get("attestation") or artifact.get("path") != source.get("path"):
        raise ValueError("Report bundle owner/attestation changed; regenerate presentation.")
    root = owned_path(run_dir, artifact["path"], must_exist=True)
    manifest = read_manifest(root, attestation)
    selected = [r for r in manifest["assets"] if r["id"] == identifier]
    if len(selected) != 1:
        raise ValueError(
            "Asset is not declared in this report bundle; paths/cross-Run reads are forbidden."
        )
    return verified_asset(root, selected[0]), selected[0]


@contextmanager
def local_asset(source: Mapping[str, Any], identifier: str) -> Iterator[BinaryIO]:
    path, row = selected_asset(source, identifier)
    with path.open("rb") as stream:
        # Bytes are verified again by the browser; no cache of whole arrays survives this request.
        if path.stat().st_size != row["size_bytes"]:
            raise ValueError("Report asset changed while opening.")
        yield stream


def remote_asset(
    jobs: Any,
    job_id: str,
    source: Mapping[str, Any],
    identifier: str,
    destination: Path,
    *,
    cancel: Callable[[], bool] = lambda: False,
    timeout_seconds: float = 120,
    expected_size: int | None = None,
) -> None:
    """Selected compressed binary transfer through the native Transport, outside HPO/heartbeat."""
    import sys

    from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory

    record = jobs.get(job_id, refresh=False, include_study=False)
    profile = jobs.catalog.get(record.cluster)
    transport = jobs.factory.transport(profile)
    # The owner must belong to this Job's native workspace or verified recovery placement.
    owner = str(source["owner"])
    allowed = str(Path(record.work_dir).parent)
    if not Path(owner).is_relative_to(allowed):
        recovery = record.metadata.get("recovery_execution_dir")
        if not isinstance(recovery, str) or not Path(owner).is_relative_to(recovery):
            raise ValueError("Report assets are outside the Job's recorded ownership.")
        # Reuse the native recovery-owner/dependency/placement authority, not merely
        # a path string saved on a new Job. It checks the original owner and cluster.
        if jobs._owned_study_path(record, source.get("run_dir")) is None:
            raise ValueError("Report recovery assets lack verified native ownership.")
    script = reader_prelude() + (
        "from lambdaforge.work.report_bundle import read_manifest,verified_asset\n"
        "from lambdaforge.work.managed import owned_path\n"
        "from pathlib import Path\nimport json,gzip,shutil\n"
        "source=json.loads(sys.argv[1]); identifier=sys.argv[2]\n"
        "owner=Path(source['owner'])\n"
        "if not owner.is_absolute() or owner.resolve()!=owner: raise ValueError('Unsafe owner')\n"
        "run=owned_path(owner,source['run_dir'],must_exist=True)\n"
        "result=json.loads(owned_path(run,'result.json',must_exist=True).read_text())\n"
        "matches=[a for a in result.get('artifacts',[]) if a.get('name')==source['artifact'] "
        "and a.get('role')=='html-bundle']\n"
        "if len(matches)!=1: raise ValueError('Unowned/ambiguous report bundle')\n"
        "a=matches[0]\n"
        "if a['path']!=source['path'] or a['metadata']['report_bundle']!=source['attestation']: "
        "raise ValueError('Changed report attestation')\n"
        "root=owned_path(run,a['path'],must_exist=True)\n"
        "manifest=read_manifest(root,source['attestation'])\n"
        "rows=[r for r in manifest['assets'] if r['id']==identifier]\n"
        "if len(rows)!=1: raise ValueError('Undeclared report asset')\n"
        "path=verified_asset(root,rows[0])\n"
        "with path.open('rb') as data,"
        "gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as out: "
        "shutil.copyfileobj(data,out,1024*1024)\n"
    )
    python = sys.executable if profile.transport == "local" else profile.python
    for n in range(1, len(record.command) - 1):
        if record.command[n : n + 2] == ("-m", "lambdaforge"):
            python = record.command[n - 1]
            break
    command = (
        *ControlPlaneFactory.reader_command(profile, preferred=python),
        "-c",
        script,
        json.dumps(dict(source)),
        identifier,
    )
    import gzip
    import threading
    import time

    with transport.stream(command) as process:
        done, interrupted = threading.Event(), threading.Event()
        deadline = time.monotonic() + timeout_seconds

        def watch() -> None:
            while not done.wait(0.1):
                if cancel() or time.monotonic() > deadline:
                    interrupted.set()
                    if hasattr(process.owner, "terminate"):
                        process.owner.terminate()
                    else:
                        process.owner.close()
                    return

        watcher = threading.Thread(target=watch, daemon=True, name="lf-report-read-deadline")
        watcher.start()
        try:
            count = 0
            with gzip.GzipFile(fileobj=process.stdout) as data, destination.open("wb") as out:
                while chunk := data.read(1024 * 1024):
                    if cancel() or interrupted.is_set():
                        raise InterruptedError("Report asset loading cancelled/timed out.")
                    count += len(chunk)
                    if expected_size is not None and count > expected_size:
                        raise ValueError("Report transfer exceeds its attested asset size.")
                    out.write(chunk)
            process.wait()
        finally:
            done.set()
            watcher.join(timeout=1)


def write_sources(output: Path, sections: Any) -> None:
    """Persist disposable deployment references; never copy assets or create result records."""
    import os

    from lambdaforge.work.atomic import atomic_write_json

    bundles = [
        copy.deepcopy(dict(s["bundle"])) for s in sections if isinstance(s.get("bundle"), Mapping)
    ]
    for bundle in bundles:
        source = bundle["source"]
        if not source.get("remote"):
            for field in ("owner", "run_dir"):
                source[field] = os.path.relpath(source[field], output.parent)
            source["relative"] = True
    path = output.with_suffix(output.suffix + ".assets.json")
    if bundles:
        atomic_write_json(path, {"report_assets_version": 1, "bundles": bundles})
    elif path.exists():
        path.unlink()
