"""Explicit verified dependency placement through the native portable import transactions."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.products import ProductBundle, ProductRegistry
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.work import ResultStore


class DependencyMaterialization:
    """No implicit transfer/producer execution. Preview is metadata-only; apply copies bytes.

    The same ProductBundle/StudyImport verifiers own publication on both sides. Compression
    transports an explicit immutable package, never a producer Job/cache path as a contract.
    """

    def __init__(
        self, catalog: ClusterCatalog | None = None, factory: ControlPlaneFactory | None = None
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()

    def materialize(
        self,
        selector: str,
        *,
        cluster: str,
        kind: str = "product",
        apply: bool = False,
        source_root: Path | None = None,
        expected_evidence_id: str | None = None,
    ) -> dict[str, Any]:
        if type(apply) is not bool or kind not in {"product", "result"}:
            raise ValueError(
                "Dependency materialization requires kind=product|result and explicit apply."
            )
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        destination = (
            profile.storage.product_root
            if kind == "product"
            else str(PurePosixPath(profile.storage.state_root) / "results")
        )
        if kind == "product":
            registry = ProductRegistry(source_root)
            product = registry.show(selector)
            frozen = product.content_id
            size = sum(row.size_bytes for row in product.artifacts)
            evidence_id = None
        else:
            store = ResultStore(source_root)
            requirement = store.reference(selector)["result"]
            frozen = requirement["execution"]
            evidence_id = requirement["evidence_id"]
            if expected_evidence_id is not None and evidence_id != expected_evidence_id:
                raise ValueError("Historical dependency changed since materialization preview.")
            size = None  # Not guessed from final metrics or logical descriptors.
        plan = {
            "kind": kind,
            "selector": selector,
            "identity": frozen,
            "evidence_id": evidence_id,
            "cluster": cluster,
            "destination": destination,
            "size_bytes": size,
            "compressed": True,
            "applied": False,
            "will_execute": False,
            "notice": "Apply transfers exact portable bytes; no producer computation is launched.",
        }
        if not apply:
            return plan
        transport = self.factory.transport(profile)
        prefix = self.factory.reader_command(profile)
        probe = """
import json, sys
from pathlib import Path
from lambdaforge.products import ProductRegistry
from lambdaforge.work import ResultStore
try:
    if sys.argv[1] == 'product':
        registry = ProductRegistry(sys.argv[2])
        manifest = registry.root / 'objects' / sys.argv[3].removeprefix('sha256:') / 'manifest.json'
        if not manifest.exists() and not manifest.is_symlink():
            value = None
        else:
            selected = registry.show(sys.argv[3])
            registry.verify(selected.content_id)
            value = {'identity': selected.content_id}
    else:
        store = ResultStore(sys.argv[2])
        value = store.reference(sys.argv[3])['result']
        from lambdaforge.work.StudyImport import _verify
        selected = store.select(sys.argv[3])
        placement = Path(selected['_manifest_path']).parent
        if selected.get('imported'):
            _verify(placement / 'portable')
except (KeyError, FileNotFoundError):
    value = None
print(json.dumps(value))
"""
        response = transport.run((*prefix, "-c", probe, kind, destination, frozen), timeout=60)
        if response.returncode:
            raise RuntimeError(
                "Could not inspect exact dependency destination: " + response.stderr[-1500:]
            )
        existing = json.loads(response.stdout)
        if existing is not None:
            if kind == "result" and existing.get("evidence_id") != evidence_id:
                raise ValueError(
                    "Destination has different evidence for this Execution; refusing replacement."
                )
            return {
                **plan,
                "applied": True,
                "reused": True,
                "transfer_bytes": 0,
                "registration": existing,
            }
        remote: PurePosixPath | None = None
        scratch = ProjectContext.discover().root / ".lambdaforge/cache/dependency-transfers"
        if scratch.resolve() != scratch:
            raise ValueError("Dependency transfer scratch cannot contain symbolic paths.")
        scratch.mkdir(parents=True, exist_ok=True)
        if size is not None and shutil.disk_usage(scratch).free < 2 * size:
            raise OSError(
                28, "Insufficient space for the explicit dependency package and ZIP", str(scratch)
            )
        with tempfile.TemporaryDirectory(prefix="transfer-", dir=scratch) as temporary:
            root = Path(temporary)
            if kind == "product":
                package = root / "package"
                ProductBundle.export(registry, frozen, package, apply=True)
            else:
                if store.reference(frozen)["result"] != requirement:
                    raise ValueError("Historical evidence changed before materialization.")
                package = Path(store.export(frozen, root / "exports")["path"])
                if store.reference(frozen)["result"] != requirement:
                    raise ValueError("Historical evidence changed while preparing its package.")
            archive = root / "dependency.zip"
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True
            ) as bundle:
                for file in sorted(package.rglob("*")):
                    if file.resolve() != file:
                        raise ValueError("Dependency package contains a symbolic path.")
                    if file.is_file():
                        bundle.write(file, file.relative_to(package).as_posix())
            prepare = """
import pathlib, tempfile, sys
root = pathlib.Path(sys.argv[1])
if not root.is_absolute() or root.resolve() != root or root == root.parent:
    raise ValueError('unsafe materialization temporary owner')
root.mkdir(parents=True, exist_ok=True)
print(tempfile.mkdtemp(prefix='.dependency-import-', dir=root))
"""
            created = transport.run(
                (*prefix, "-c", prepare, profile.storage.cache_root), timeout=30
            )
            if created.returncode:
                raise RuntimeError(
                    f"Could not prepare dependency placement: {created.stderr[-1000:]}"
                )
            remote = PurePosixPath(created.stdout.strip())
            if (
                not remote.is_absolute()
                or remote.parent != PurePosixPath(profile.storage.cache_root)
                or not remote.name.startswith(".dependency-import-")
                or ".." in remote.parts
            ):
                raise ValueError("Provider returned an unsafe dependency transaction root.")
            try:
                transport.put(archive, str(remote / "dependency.zip"))
                script = """
import json, pathlib, sys
from lambdaforge.controlplane.StudyExportService import _extract_safe
from lambdaforge.products import ProductBundle, ProductRegistry
from lambdaforge.work import ResultStore
stage = pathlib.Path(sys.argv[1])
if stage.resolve() != stage or not stage.name.startswith('.dependency-import-'):
    raise ValueError('unsafe dependency stage')
package = stage / 'package'
_extract_safe(stage / 'dependency.zip', package)
if sys.argv[2] == 'product':
    registry = ProductRegistry(sys.argv[3])
    preview = ProductBundle.import_bundle(registry, package)
    if preview['content_id'] != sys.argv[4]:
        raise ValueError('dependency product changed during materialization')
    result = ProductBundle.import_bundle(registry, package, apply=True)
else:
    from lambdaforge.work.ResultInput import _digest
    store = ResultStore(sys.argv[3])
    preview = store.import_export(package, product_root=sys.argv[5])
    if preview['execution_id'] != sys.argv[4]:
        raise ValueError('dependency execution changed during materialization')
    origin = package / 'execution'
    identity = _digest({'result': json.loads((origin / 'result.json').read_text()),
                        'configuration': json.loads((origin / 'configuration.json').read_text())})
    if identity != sys.argv[6]:
        raise ValueError('dependency evidence changed during materialization')
    result = store.import_export(package, product_root=sys.argv[5], apply=True)
print(json.dumps(result, separators=(',', ':')))
"""
                imported = transport.run(
                    (
                        *prefix,
                        "-c",
                        script,
                        str(remote),
                        kind,
                        destination,
                        frozen,
                        profile.storage.product_root,
                        evidence_id or "-",
                    ),
                    timeout=3600,
                )
                if imported.returncode:
                    raise RuntimeError(
                        f"Could not materialize exact {kind} on {cluster}: "
                        f"{imported.stderr[-1500:]}. "
                        "Ensure the destination Python has the current LambdaForge runtime; "
                        "no producer was executed."
                    )
                response = json.loads(imported.stdout)
                return {
                    **plan,
                    "applied": True,
                    "transfer_bytes": archive.stat().st_size,
                    "registration": response,
                }
            finally:
                active_error = sys.exc_info()[0] is not None
                cleanup = """
import pathlib, shutil, sys
path, owner = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if (path.resolve() != path or path.parent != owner
    or not path.name.startswith('.dependency-import-')):
    raise ValueError('unsafe dependency cleanup target')
if path.exists(): shutil.rmtree(path)
"""
                try:
                    cleaned = transport.run(
                        (*prefix, "-c", cleanup, str(remote), profile.storage.cache_root),
                        timeout=30,
                    )
                    if cleaned.returncode and not active_error:
                        raise RuntimeError(f"Dependency was imported, but cleanup failed: {remote}")
                except Exception:
                    if not active_error:
                        raise
