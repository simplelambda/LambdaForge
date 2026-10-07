"""Read and safely remove local Work Execution result envelopes."""

from __future__ import annotations

import errno
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping, MutableSequence, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import TYPE_CHECKING, Any

from lambdaforge.analysis.Report import write_html
from lambdaforge.analysis.StudyAnalysis import StudyAnalysis
from lambdaforge.hpo.ResourceReplay import ReplayPolicyName, ResourceSchedulerReplay
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.work.config import WorkConfig
from lambdaforge.work.failure import render_scientific_failures, scientific_failures
from lambdaforge.work.models import atomic_json
from lambdaforge.work.state import StudyState

if TYPE_CHECKING:
    from lambdaforge.products.models import StudyProduct


class ResultStore:
    """Treat result manifests as authority while bounding deletion to one run root."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("LAMBDAFORGE_RUN_ROOT")
        self.root = Path(
            configured or ProjectContext.discover().root / ".lambdaforge" / "runs"
        ).resolve()

    def list(self) -> tuple[dict[str, Any], ...]:
        """Return valid current Execution envelopes without guessing partial state."""
        records: list[dict[str, Any]] = []
        if not self.root.is_dir() or self.root.is_symlink():
            return ()
        for path in sorted(self.root.glob("*/execution-*/result.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue  # A concurrent confirmed deletion can remove an inventory entry.
            except (OSError, ValueError, TypeError) as error:
                raise RuntimeError(f"Corrupt Work result manifest: {path}") from error
            if not isinstance(value, dict) or value.get("execution_result_version") != 1:
                raise RuntimeError(f"Unsupported Work result manifest: {path}")
            if "lifecycle" not in value:
                value["lifecycle"] = StudyState.from_execution(
                    value.get("summary", {}),
                    value.get("runs", ()),
                    status=str(value.get("status", "unknown")),
                ).to_dict()
            value["_manifest_path"] = str(path)
            if (path.parent / "import.json").exists() or (path.parent / "import.json").is_symlink():
                from lambdaforge.work.StudyImport import relative_run_path

                receipt = _read_mapping(path.parent / "import.json")
                if (
                    receipt.get("study_import_version") != 1
                    or receipt.get("execution_id") != value.get("execution_id")
                    or receipt.get("scientific_fingerprint") != value.get("scientific_fingerprint")
                    or receipt.get("evidence_root") != "portable/execution"
                    or (path.parent / "import.json").resolve() != path.parent / "import.json"
                ):
                    raise ValueError("Corrupt Study import ownership receipt.")
                value["imported"] = dict(receipt)
                for run in value.get("runs", ()):
                    run["run_dir"] = str(
                        path.parent / "portable/execution" / relative_run_path(run)
                    )
            records.append(value)
        return tuple(records)

    def select(self, selector: str) -> dict[str, Any]:
        """Resolve exact name/Execution/fingerprint and refuse ambiguous names."""
        matches = tuple(
            value
            for value in self.list()
            if selector
            in {
                value.get("name"),
                value.get("execution_id"),
                value.get("scientific_fingerprint"),
            }
        )
        if not matches:
            receipt = self._receipt(selector)
            if receipt is not None:
                return {**receipt, "already_deleted": True}
            raise KeyError(f"Unknown local Work Execution {selector!r}.")
        if len(matches) != 1:
            raise ValueError(
                f"Work selector {selector!r} identifies {len(matches)} local Executions; "
                "use an Execution ID."
            )
        return dict(matches[0])

    def execution_directory(self, selector: str) -> Path:
        """Resolve one live local owned Execution for explicit post-Study operations."""
        selected = self.select(selector)
        if selected.get("already_deleted"):
            raise ValueError(f"Work Execution {selector!r} was already deleted.")
        manifest = Path(str(selected["_manifest_path"])).absolute()
        if manifest.resolve() != manifest:
            raise ValueError("Execution manifest cannot be symbolic.")
        return self._evidence_dir(manifest)

    def import_export(
        self, source: str | Path, *, product_root: str | Path | None = None, apply: bool = False
    ) -> dict[str, Any]:
        """Verify/register a portable Study without executing or modifying original evidence."""
        from lambdaforge.products.registry import ProductRegistry
        from lambdaforge.work.StudyImport import StudyImport

        return StudyImport.inspect(source, self.root, ProductRegistry(product_root), apply=apply)

    def _evidence_dir(self, manifest: Path) -> Path:
        root = self._execution_dir(manifest)
        if (root / "import.json").exists():
            evidence = root / "portable" / "execution"
            if evidence.resolve() != evidence or not evidence.is_dir():
                raise ValueError("Imported Study evidence root is missing or symbolic.")
            return evidence
        return root

    def decision(self, selector: str, *, name: str, contract: str) -> StudyProduct:
        """Seal native selection/valid cached conclusions without computing or publishing them."""
        from lambdaforge.products.decision import build_study_decision

        selected = self.select(selector)
        root = self.execution_directory(selector)
        definition = _analysis_definition(_read_mapping(root / "configuration.json"))
        path = root / "analysis.json"
        if path.is_symlink():
            raise ValueError("Study Analysis must be an owned non-symbolic regular file.")
        analysis = _read_mapping(path) if path.exists() else None
        return build_study_decision(
            selected,
            name=name,
            contract=contract,
            analysis=analysis,
            authored_space=StudyAnalysis.authored_space(definition),
        )

    def finalize_products(
        self, selector: str, *, product_root: str | Path | None = None, apply: bool = False
    ) -> dict[str, Any]:
        """Retry only declared post-Study publication from owned immutable evidence."""
        from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

        root = self.execution_directory(selector)
        if self.select(selector).get("imported"):
            raise ValueError(
                "Imported evidence cannot run producer finalization; import its sealed products."
            )
        if apply:
            with CrossProcessFileLock(
                root / ".controller.lock",
                shared=False,
                timeout_seconds=5,
                poll_interval_seconds=0.05,
            ):
                return self._finalize_products(root, product_root=product_root, apply=True)
        return self._finalize_products(root, product_root=product_root, apply=False)

    def product_status(self, selector: str) -> dict[str, Any]:
        """Read bounded publication feedback without hashing weights or recomputing selection."""
        from lambdaforge.products.registry import ProductRegistry

        root = self.execution_directory(selector)
        path = root / "products.json"
        if not path.exists() and not path.is_symlink():
            return {"execution_id": root.name, "status": "not_declared", "items": []}
        value = ProductRegistry._read(path)
        if (
            type(value.get("product_publication_version")) is not int
            or value["product_publication_version"] != 1
            or value.get("execution_id") != self.select(selector)["execution_id"]
            or value.get("status") not in {"pending", "published", "failed"}
            or not isinstance(value.get("items"), list)
            or len(value["items"]) > 128
            or any(
                not isinstance(item, Mapping)
                or not isinstance(item.get("name"), str)
                or item.get("status") not in {"pending", "published", "failed"}
                for item in value["items"]
            )
        ):
            raise ValueError("Corrupt native product publication feedback.")
        return {**value, "record_path": str(path)}

    @staticmethod
    def _finalize_products(
        root: Path, *, product_root: str | Path | None, apply: bool
    ) -> dict[str, Any]:
        from lambdaforge.products.publication import publish_declared_products
        from lambdaforge.products.registry import ProductRegistry

        configuration = _read_mapping(root / "configuration.json")
        if "products" not in configuration:
            raise ValueError(
                "This Execution declares no products; use products decide/select explicitly."
            )
        analysis_path = root / "analysis.json"
        if analysis_path.resolve() != analysis_path:
            raise ValueError("Study Analysis cannot be symbolic.")
        # Read the exact persisted envelope, not observer-added display projections.
        return publish_declared_products(
            configuration,
            _read_mapping(root / "result.json"),
            root,
            ProductRegistry(product_root),
            analysis=_read_mapping(analysis_path) if analysis_path.exists() else None,
            authored_space=StudyAnalysis.authored_space(_analysis_definition(configuration)),
            apply=apply,
        )

    def delete(self, selector: str, *, apply: bool = False) -> dict[str, Any]:
        """Preview/apply exact-root deletion while preserving shared/durable-independent data."""
        selected = self.select(selector)
        if selected.get("already_deleted"):
            return {**selected, "applied": apply}
        manifest = Path(str(selected.pop("_manifest_path"))).absolute()
        if manifest.resolve() != manifest:
            raise ValueError("Deletion requires a non-symbolic owned Execution path.")
        execution_dir = self._execution_dir(manifest)
        self._validate_deletion(selected, execution_dir)
        payload = {
            "work": selected,
            "execution_dir": str(execution_dir),
            "applied": apply,
            "already_deleted": False,
            "will_remove": ["result envelopes", "Attempts", "owned artifacts", "checkpoints"],
            "preserved": [
                "published datasets",
                "published products",
                "shared environments",
                "reconstructible cache",
                "configured result root",
            ],
        }
        if apply:
            from contextlib import nullcontext

            from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

            # The catalog lock serializes import/delete; the native controller lock prevents
            # deleting during execution, recovery or publication. Preview acquires neither.
            with CrossProcessFileLock(
                self.root / ".study-import.lock",
                shared=False,
                timeout_seconds=30,
                poll_interval_seconds=0.02,
            ):
                current = self.select(str(selected["execution_id"]))
                if current.get("already_deleted"):
                    return {**current, "applied": True}
                controller = (
                    nullcontext()
                    if current.get("imported")
                    else CrossProcessFileLock(
                        execution_dir / ".controller.lock",
                        shared=False,
                        timeout_seconds=0.1,
                        poll_interval_seconds=0.01,
                    )
                )
                with controller:
                    current = self.select(str(selected["execution_id"]))
                    self._validate_deletion(current, execution_dir)
                    payload["work"] = {
                        key: value for key, value in current.items() if key != "_manifest_path"
                    }
                    self._write_receipt(payload)
                    shutil.rmtree(execution_dir)
                payload["removed_empty_work_directory"] = self._prune_work_parent(execution_dir)
        return payload

    @staticmethod
    def _validate_deletion(selected: Mapping[str, Any], directory: Path) -> None:
        """Require persisted ownership and terminal state, not a guessed observer status."""
        if selected.get("execution_id") != directory.name:
            raise ValueError("Execution deletion identity differs from its owned directory.")
        if selected.get("imported"):
            return  # An imported running snapshot is evidence, not a local running controller.
        if selected.get("status") not in {
            "succeeded",
            "failed",
            "cancelled",
            "interrupted",
            "completed_with_failures",
        }:
            raise ValueError(
                "Cannot delete active or unverifiable Execution evidence; cancel/reconcile first."
            )
        origin = directory / "execution.json"
        if origin.resolve() != origin or not origin.is_file():
            raise ValueError("Deletion requires a regular persisted Execution ownership record.")
        value = _read_mapping(origin)
        if (
            value.get("execution_id") != directory.name
            or value.get("scientific_fingerprint") != selected.get("scientific_fingerprint")
            or value.get("ownership", {}).get("execution_dir") != "owned"
        ):
            raise ValueError("Execution deletion ownership/identity evidence is inconsistent.")

    def _prune_work_parent(self, execution_dir: Path) -> bool:
        """Remove only the now-empty owned Work parent; never the configured root or links.

        Use a root descriptor so swapping the child for a symlink cannot redirect deletion.
        Platforms without safe descriptor-relative directory operations retain the empty parent.
        """
        if os.rmdir not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
            return False
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                os.rmdir(execution_dir.parent.name, dir_fd=descriptor)
            except OSError as error:
                if error.errno in {errno.ENOTEMPTY, errno.EEXIST, errno.ENOENT, errno.ENOTDIR}:
                    return False
                raise
            return True
        finally:
            os.close(descriptor)

    def source(self, selector: str) -> Path:
        """Return the verified authored YAML path recorded for one local Execution."""
        selected = self.select(selector)
        if selected.get("already_deleted"):
            raise ValueError(f"Work Execution {selector!r} was already deleted.")
        manifest = Path(str(selected["_manifest_path"])).resolve()
        execution_dir = self._evidence_dir(manifest)
        if selected.get("imported"):
            source = execution_dir / "authored.yaml"
            if source.resolve() != source or not source.is_file():
                raise FileNotFoundError("Imported Study has no authored YAML snapshot.")
            return source
        execution_manifest = execution_dir / "execution.json"
        try:
            value = json.loads(execution_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"Corrupt Work execution manifest: {execution_manifest}") from error
        source = Path(str(value.get("source", ""))).expanduser().resolve()
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Recorded Work YAML is unavailable: {source}")
        return source

    def configuration(self, selector: str) -> WorkConfig:
        """Reconstruct the immutable submitted Work config for an exact local retry."""
        selected = self.select(selector)
        if selected.get("imported"):
            raise ValueError(
                "Imported Study evidence is read-only; recovery requires its original "
                "owned Execution."
            )
        if selected.get("already_deleted"):
            raise ValueError(f"Work Execution {selector!r} was already deleted.")
        execution_dir = self._execution_dir(Path(str(selected["_manifest_path"])).resolve())
        configuration_path = execution_dir / "configuration.json"
        execution_path = execution_dir / "execution.json"
        try:
            raw = json.loads(configuration_path.read_text(encoding="utf-8"))
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"Corrupt Work retry metadata below: {execution_dir}") from error
        if not isinstance(raw, Mapping) or not isinstance(execution, Mapping):
            raise RuntimeError(f"Corrupt Work retry metadata below: {execution_dir}")
        source = Path(str(execution.get("source", ""))).expanduser().resolve()
        return WorkConfig.from_mapping(raw, source=source)

    def log_report(
        self,
        selector: str,
        *,
        tail: int | None = None,
        include_traceback: bool = False,
    ) -> dict[str, Any]:
        """Return bounded local logs and the authoritative structured failure."""
        selected = self.select(selector)
        if selected.get("already_deleted"):
            raise ValueError(f"Work Execution {selector!r} was already deleted.")
        manifest = Path(str(selected["_manifest_path"])).resolve()
        execution_dir = self._evidence_dir(manifest)
        chunks: list[str] = []
        for run in selected.get("runs", ()):
            if not isinstance(run, Mapping):
                raise RuntimeError(f"Corrupt Run entry in {execution_dir / 'result.json'}")
            run_dir = Path(str(run.get("run_dir", ""))).resolve()
            if not run_dir.is_relative_to(execution_dir) or run_dir.is_symlink():
                raise ValueError(f"Unsafe recorded Work log root: {run_dir}")
            log_name = Path(str(run.get("logs", "work.log")))
            if log_name.is_absolute() or ".." in log_name.parts:
                raise ValueError(f"Unsafe recorded Work log path: {log_name}")
            log_path = (run_dir / log_name).resolve()
            if not log_path.is_relative_to(run_dir) or log_path.is_symlink():
                raise ValueError(f"Unsafe recorded Work log path: {log_path}")
            if log_path.is_file():
                chunks.append(log_path.read_text(encoding="utf-8", errors="replace"))
        text = "".join(chunks)
        if tail is not None and tail < 0:
            raise ValueError("Log tail must be a non-negative integer.")
        captured = (
            text
            if tail is None
            else "".join(text.splitlines(keepends=True)[-tail:])
            if tail
            else ""
        )
        failures = scientific_failures(selected, result_path=str(manifest))
        section = render_scientific_failures(
            failures,
            existing_output=captured,
            include_traceback=include_traceback,
        )
        rendered = captured.rstrip()
        if section:
            rendered = f"{rendered}\n\n{section}" if rendered else section
        if rendered:
            rendered += "\n"
        return {
            "execution_id": selected.get("execution_id"),
            "name": selected.get("name"),
            "status": selected.get("status"),
            "text": rendered,
            "failure": failures[0] if failures else None,
            "failures": list(failures),
            "result_path": str(manifest),
        }

    def logs(
        self,
        selector: str,
        *,
        tail: int | None = None,
        include_traceback: bool = False,
    ) -> str:
        """Read captured logs and append persisted terminal failure evidence."""
        return str(
            self.log_report(
                selector,
                tail=tail,
                include_traceback=include_traceback,
            )["text"]
        )

    def compare(
        self,
        selectors: tuple[str, ...],
        *,
        metric: str | None = None,
        mode: str = "max",
    ) -> dict[str, Any]:
        """Compare exact scalar Run metrics across unambiguous local Executions."""
        if len(selectors) < 2:
            raise ValueError("Result comparison requires at least two Execution selectors.")
        if mode not in {"min", "max"}:
            raise ValueError("Result comparison mode must be min or max.")
        selected = [self.select(selector) for selector in selectors]
        execution_ids = [str(value["execution_id"]) for value in selected]
        if len(execution_ids) != len(set(execution_ids)):
            raise ValueError("Result comparison selectors must identify distinct Executions.")
        metric_names = sorted(
            {
                str(name)
                for value in selected
                for run in value.get("runs", ())
                if isinstance(run, Mapping) and run.get("status") == "succeeded"
                for name in (
                    run.get("metrics", {}).keys() if isinstance(run.get("metrics"), Mapping) else ()
                )
            }
        )
        if metric is not None:
            if metric not in metric_names:
                raise KeyError(f"Metric {metric!r} is absent from the selected Executions.")
            metric_names = [metric]
        comparisons: dict[str, list[dict[str, Any]]] = {}
        for name in metric_names:
            rows: list[dict[str, Any]] = []
            for value in selected:
                observations = [
                    float(run["metrics"][name])
                    for run in value.get("runs", ())
                    if isinstance(run, Mapping)
                    and run.get("status") == "succeeded"
                    and isinstance(run.get("metrics"), Mapping)
                    and name in run["metrics"]
                ]
                rows.append(
                    {
                        "name": value.get("name"),
                        "execution_id": value.get("execution_id"),
                        "count": len(observations),
                        "mean": fmean(observations) if observations else None,
                        "minimum": min(observations) if observations else None,
                        "maximum": max(observations) if observations else None,
                        "standard_deviation": stdev(observations)
                        if len(observations) >= 2
                        else None,
                        "standard_error": stdev(observations) / len(observations) ** 0.5
                        if len(observations) >= 2
                        else None,
                    }
                )
            baseline = rows[0].get("mean") if rows else None
            for row in rows:
                mean = row.get("mean")
                row["difference_from_first"] = (
                    float(mean) - float(baseline)
                    if isinstance(mean, int | float) and isinstance(baseline, int | float)
                    else None
                )
                row["relative_difference_from_first"] = (
                    (float(mean) - float(baseline)) / abs(float(baseline))
                    if isinstance(mean, int | float)
                    and isinstance(baseline, int | float)
                    and float(baseline) != 0
                    else None
                )
            comparisons[name] = rows
        ranking = None
        if metric is not None:
            ranked = [row for row in comparisons[metric] if row["mean"] is not None]
            ranking = sorted(
                ranked,
                key=lambda row: float(row["mean"]),
                reverse=mode == "max",
            )
        return {
            "executions": execution_ids,
            "metrics": comparisons,
            "ranking": ranking,
            "objective": {"metric": metric, "mode": mode} if metric else None,
            "resources": {
                str(value["execution_id"]): _execution_resources(value) for value in selected
            },
            "warnings": [
                f"Metric {name!r} is unavailable in one or more selected Executions."
                for name, rows in comparisons.items()
                if any(row["count"] == 0 for row in rows)
            ],
            "study_analysis": {
                str(value["execution_id"]): self.analysis_summary(str(value["execution_id"]))
                for value in selected
            },
        }

    def analysis(self, selector: str, *, recompute: bool = False) -> dict[str, Any]:
        """Compute or reuse the versioned analysis beside one exact Execution."""
        selected = self.select(selector)
        if selected.get("already_deleted"):
            raise ValueError(f"Work Execution {selector!r} was already deleted.")
        manifest = Path(str(selected["_manifest_path"])).resolve()
        execution_dir = self._evidence_dir(manifest)
        if selected.get("imported"):
            if recompute:
                raise ValueError(
                    "Imported evidence cannot refit or rewrite its persisted Analysis."
                )
            for candidate in (
                execution_dir / "analysis.json",
                execution_dir.parent / "reports" / "study-analysis.json",
            ):
                if candidate.is_file() and candidate.resolve() == candidate:
                    return _read_mapping(candidate)
            raise ValueError("No persisted Analysis was included in this Study export.")
        configuration = _read_mapping(execution_dir / "configuration.json")
        definition = _analysis_definition(configuration)
        objective = definition.get("objective")
        objective = objective if isinstance(objective, Mapping) else None
        authored = StudyAnalysis.authored_space(definition)
        semantics_path = execution_dir / "analysis-semantics.json"
        if semantics_path.is_file() and not semantics_path.is_symlink():
            frozen = _read_mapping(semantics_path)
            selected["analysis_semantics"] = frozen.get(
                str(definition.get("name", selected.get("name"))), {}
            )
        return StudyAnalysis.persist(
            selected,
            execution_dir / "analysis.json",
            objective=objective,
            authored_space=authored,
            recompute=recompute,
        )

    def analysis_summary(self, selector: str) -> dict[str, Any] | None:
        """Read a valid persisted summary without triggering expensive analysis."""
        selected = self.select(selector)
        manifest = Path(str(selected["_manifest_path"])).resolve()
        path = self._evidence_dir(manifest) / "analysis.json"
        if not path.is_file() or path.is_symlink():
            return None
        value = _read_mapping(path)
        return {
            "analysis_version": value.get("analysis_version"),
            "status": value.get("source", {}).get("status"),
            "winner": value.get("winner"),
            "summary": value.get("summary"),
            "findings": value.get("findings", []),
            "path": str(path),
        }

    def resource_replay(
        self, selector: str, *, policy: ReplayPolicyName = "ari-v3.1"
    ) -> dict[str, Any]:
        """Replay one Study's canonical resource trace without parsing human logs."""
        selected = self.select(selector)
        if selected.get("already_deleted"):
            raise ValueError(f"Work Execution {selector!r} was already deleted.")
        execution_dir = self._evidence_dir(Path(str(selected["_manifest_path"])).resolve())
        return ResourceSchedulerReplay.from_execution(execution_dir).replay(policy)

    def report(
        self,
        selector: str,
        output: str | Path,
        *,
        recompute: bool = False,
    ) -> Path:
        """Export the same analysis consumed by CLI/TUI to one offline HTML file."""
        from lambdaforge.study_projection import read_html_sections

        selected = self.select(selector)
        execution_dir = self._evidence_dir(Path(str(selected["_manifest_path"])).resolve())
        return write_html(
            self.analysis(selector, recompute=recompute),
            output,
            sections=read_html_sections(selected.get("runs", ()), execution_dir),
        )

    def export(
        self,
        selector: str,
        destination: str | Path,
        *,
        supplementary: Mapping[str, str | Path] | None = None,
        copy_published: bool = True,
        captured_status: str | None = None,
        link_evidence: bool = False,
        profile: str = "full",
        omissions: Sequence[Mapping[str, Any]] = (),
        product_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Create one atomic portable package for a final Execution or live snapshot.

        ``destination`` is a parent directory.  LambdaForge creates a uniquely named
        child and never overwrites an earlier export.  Supplementary roots are used by
        the control plane for bounded Job/Study evidence downloaded from a provider.
        """
        if profile not in {"default", "full"}:
            raise ValueError("Export profile must be default or full.")
        try:
            selected = self.select(selector)
        except KeyError:
            selected = self._select_execution_snapshot(selector, captured_status)
        if selected.get("already_deleted"):
            raise ValueError(f"Work Execution {selector!r} was already deleted.")
        if selected.get("imported"):
            raise ValueError(
                "This is already portable imported evidence. Copy its verified portable/ package "
                "rather than generating a new producer provenance record."
            )
        manifest = Path(str(selected["_manifest_path"])).resolve()
        execution_dir = self._execution_dir(manifest)
        execution_id = str(selected.get("execution_id") or execution_dir.name)
        name = str(selected.get("name") or "study")
        execution_status = str(selected.get("status") or "unknown")
        status = str(captured_status or execution_status)
        finalized = (execution_dir / "result.json").is_file()
        export_kind = "final" if finalized and status == "succeeded" else "snapshot"
        captured_at = datetime.now(timezone.utc)
        authored_parent = Path(destination).expanduser()
        if authored_parent.is_symlink():
            raise ValueError(f"Export destination cannot be a symlink: {authored_parent}")
        parent = authored_parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"Export destination is not a safe directory: {parent}")
        folder_name = f"{_portable_name(name)}--{_portable_name(execution_id)}"
        if export_kind == "snapshot":
            folder_name += f"--snapshot-{captured_at.strftime('%Y%m%dT%H%M%S%fZ')}"
        folder = parent / folder_name
        if folder.exists() or folder.is_symlink():
            raise FileExistsError(
                f"Export destination already exists: {folder}. Choose another directory or "
                "move the previous export first."
            )

        warnings: list[str] = []
        # Never invent a final conclusion for a live snapshot. Terminal evidence may still have
        # no applicable Study Analysis (for example an ordinary one-Run Work).
        analysis = None
        if finalized:
            try:
                analysis = self.analysis(execution_id)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                warnings.append(f"Study Analysis was not applicable: {error}")
        else:
            warnings.append(
                "This is a non-final Study snapshot; active or missing Runs and conclusions may "
                "change after capture."
            )
        stage = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=parent))
        try:
            _copy_evidence_tree(
                execution_dir,
                stage / "execution",
                hardlink=link_evidence,
            )
            source: Path | None = None
            source_error: Exception | None = None
            try:
                source = self.source(execution_id)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                source_error = error
            authored_snapshot = execution_dir / "authored.yaml"
            if (
                source is None
                and authored_snapshot.is_file()
                and not authored_snapshot.is_symlink()
            ):
                source = authored_snapshot
            if source is None and source_error is not None:
                warnings.append(
                    "Authored YAML snapshot unavailable: "
                    f"{type(source_error).__name__}: {source_error}"
                )
            if source is not None:
                _copy_evidence_tree(source, stage / "configuration" / source.name)

            for label, raw_source in sorted((supplementary or {}).items()):
                safe_label = _portable_name(str(label))
                evidence_source = Path(raw_source).expanduser().resolve()
                if not evidence_source.exists():
                    warnings.append(f"Supplementary evidence {label!r} was unavailable.")
                    continue
                _copy_evidence_tree(
                    evidence_source,
                    stage / safe_label,
                    hardlink=link_evidence,
                )

            published = (
                self._copy_published_artifacts(selected, execution_dir, stage, warnings)
                if copy_published
                else sum(1 for path in (stage / "published-artifacts").rglob("*") if path.is_file())
                if (stage / "published-artifacts").is_dir()
                else 0
            )
            reports = stage / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            if analysis is not None:
                atomic_json(reports / "study-analysis.json", analysis)
                try:
                    from lambdaforge.study_projection import read_html_sections

                    write_html(
                        analysis,
                        reports / "study-analysis.html",
                        sections=read_html_sections(
                            selected.get("runs", ()), execution_dir, relocate=True
                        ),
                    )
                except RuntimeError as error:
                    _write_basic_analysis_html(analysis, reports / "study-analysis.html")
                    warnings.append(
                        "Plotly Study Analysis was unavailable; a structured self-contained HTML "
                        f"fallback was generated instead: {error}"
                    )
            try:
                atomic_json(
                    reports / "resource-replay.json",
                    ResourceSchedulerReplay.from_execution(execution_dir).replay("recorded"),
                )
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                warnings.append(
                    f"Recorded resource replay was unavailable: {type(error).__name__}: {error}"
                )

            report_hint = (
                "Open reports/study-analysis.html for the interactive scientific report.\n"
                if (reports / "study-analysis.html").is_file()
                else "No finalized Study Analysis report is included in this package.\n"
            )
            state_notice = (
                "This is a point-in-time snapshot, not a final scientific result. "
                f"Captured control-plane state: {status}.\n\n"
                if export_kind == "snapshot"
                else "This package contains a finalized successful Execution.\n\n"
            )
            (stage / "README.txt").write_text(
                "LambdaForge portable experiment export\n"
                "======================================\n\n"
                + state_notice
                + report_hint
                + "execution/ contains the immutable result envelope, Run logs, scalar evidence, "
                "controller decisions, checkpoints and retained artifacts.\n"
                "study/ and control-plane/ contain provider telemetry when this export came "
                "from a managed Job. published-artifacts/ contains explicit finalized outputs "
                "that lived outside the Execution root.\n\n"
                "Shared datasets, environments, caches and the staged project bundle are "
                "referenced by provenance but intentionally not duplicated.\n",
                encoding="utf-8",
            )
            inventory = _inventory(stage)
            products = self._export_products(execution_dir, stage, product_root)
            if products:
                inventory = _inventory(stage)
            export_manifest = {
                "lambdaforge_export_version": 2,
                "created_at_utc": captured_at.isoformat(),
                "name": name,
                "execution_id": execution_id,
                "scientific_fingerprint": selected.get("scientific_fingerprint"),
                "status": status,
                "attempt_state": status,
                "execution_status": execution_status if finalized else None,
                "lifecycle": selected.get("lifecycle"),
                "export_kind": export_kind,
                "profile": profile,
                "finalized": finalized,
                "source_result": {
                    "name": selected.get("name"),
                    "execution_id": selected.get("execution_id"),
                    "scientific_fingerprint": selected.get("scientific_fingerprint"),
                    "status": execution_status if finalized else None,
                    "run_count": len(selected.get("runs", ())),
                },
                "inventory": inventory,
                "inventory_scope": "all regular package files except manifest.json itself",
                "file_count": len(inventory),
                "size_bytes": sum(int(item["size_bytes"]) for item in inventory),
                "published_artifact_files": published,
                "products": products,
                "warnings": warnings,
                "omitted_source_files": [dict(value) for value in omissions],
                "excluded_shared_state": [
                    "managed datasets",
                    "managed environments",
                    "reconstructible caches",
                    "staged project bundle",
                ],
            }
            atomic_json(stage / "manifest.json", export_manifest)
            stage.replace(folder)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return {
            "status": "exported",
            "name": name,
            "execution_id": execution_id,
            "captured_state": status,
            "execution_status": execution_status if finalized else None,
            "export_kind": export_kind,
            "finalized": finalized,
            "path": str(folder),
            "manifest": str(folder / "manifest.json"),
            "analysis_report": (
                str(folder / "reports" / "study-analysis.html")
                if (folder / "reports" / "study-analysis.html").is_file()
                else None
            ),
            "file_count": len(inventory),
            "size_bytes": sum(int(item["size_bytes"]) for item in inventory),
            "warnings": warnings,
            "profile": profile,
        }

    @staticmethod
    def _export_products(
        execution_dir: Path, stage: Path, product_root: str | Path | None
    ) -> Sequence[dict[str, Any]]:
        """Seal actually published products, never rerun selection or silently omit their bytes."""
        from lambdaforge.products.bundle import ProductBundle
        from lambdaforge.products.registry import ProductRegistry

        path = execution_dir / "products.json"
        if not path.exists() and not path.is_symlink():
            return []
        record = ProductRegistry._read(path)
        if record.get("product_publication_version") != 1:
            raise ValueError("Unsupported product publication record during Study export.")
        registry = ProductRegistry(product_root)
        output = []
        seen: set[str] = set()
        for item in record.get("items", ()):
            if item.get("status") != "published":
                continue
            product = registry.show(item["name"])
            if product.content_id != item["content_id"] or product.name in seen:
                raise ValueError("Study published product identity differs or is duplicated.")
            seen.add(product.name)
            relative = "products/" + hashlib.sha256(product.name.encode("utf-8")).hexdigest()
            ProductBundle.export(registry, product.name, stage / relative, apply=True)
            output.append(
                {"path": relative, "name": product.name, "content_id": product.content_id}
            )
        return output

    def _select_execution_snapshot(
        self, selector: str, captured_status: str | None
    ) -> dict[str, Any]:
        """Resolve a non-final Execution from its immutable planning manifest."""
        matches: list[tuple[Path, Mapping[str, Any]]] = []
        if self.root.is_dir() and not self.root.is_symlink():
            for path in sorted(self.root.glob("*/execution-*/execution.json")):
                if path.is_symlink() or not path.is_file():
                    continue
                value = _read_mapping(path)
                if selector in {
                    value.get("name"),
                    value.get("execution_id"),
                    value.get("scientific_fingerprint"),
                }:
                    matches.append((path, value))
        if not matches:
            raise KeyError(f"Unknown local Work Execution {selector!r}.")
        if len(matches) != 1:
            raise ValueError(
                f"Work selector {selector!r} identifies {len(matches)} Execution snapshots; "
                "use an Execution ID."
            )
        planning_path, planning = matches[0]
        execution_dir = planning_path.parent
        runs: list[Mapping[str, Any]] = []
        for path in sorted((execution_dir / "runs").glob("*/result.json")):
            if path.is_symlink() or not path.is_file():
                continue
            value = _read_mapping(path)
            if value.get("execution_id") == planning.get("execution_id"):
                runs.append(value)
        return {
            "execution_result_version": 1,
            "name": planning.get("name"),
            "execution_id": planning.get("execution_id"),
            "scientific_fingerprint": planning.get("scientific_fingerprint"),
            "status": captured_status or "unknown",
            "runs": runs,
            # ``_execution_dir`` validates ownership from the path shape; the terminal envelope
            # deliberately does not exist yet.
            "_manifest_path": str(execution_dir / "result.json"),
        }

    @staticmethod
    def _copy_published_artifacts(
        selected: Mapping[str, Any],
        execution_dir: Path,
        stage: Path,
        warnings: MutableSequence[str],
    ) -> int:
        """Copy explicit finalized outputs outside the owned Execution once."""
        copied = 0
        seen: set[Path] = set()
        for run_index, run in enumerate(selected.get("runs", ())):
            if not isinstance(run, Mapping):
                continue
            artifacts = run.get("artifacts", ())
            if not isinstance(artifacts, list | tuple):
                continue
            for artifact_index, artifact in enumerate(artifacts):
                if not isinstance(artifact, Mapping):
                    continue
                metadata = artifact.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                raw = artifact.get("published_path") or metadata.get("published_to")
                if not isinstance(raw, str) or not raw:
                    continue
                authored_path = Path(raw).expanduser()
                if authored_path.is_symlink():
                    warnings.append(f"Published artifact is a symlink and was not copied: {raw}")
                    continue
                path = authored_path.resolve()
                if path in seen or path.is_relative_to(execution_dir):
                    continue
                seen.add(path)
                if not path.exists():
                    warnings.append(f"Published artifact is no longer available: {path}")
                    continue
                name = _portable_name(str(artifact.get("name") or path.name or "artifact"))
                target = (
                    stage
                    / "published-artifacts"
                    / f"run-{run_index:05d}"
                    / (f"{artifact_index:04d}-{name}")
                )
                _copy_evidence_tree(path, target)
                copied += (
                    1
                    if target.is_file()
                    else sum(1 for item in target.rglob("*") if item.is_file())
                )
        return copied

    @property
    def _receipt_root(self) -> Path:
        return self.root.parent / "deletions"

    def _write_receipt(self, payload: Mapping[str, Any]) -> None:
        execution_id = str(payload["work"]["execution_id"])
        atomic_json(self._receipt_root / f"{execution_id}.json", payload)

    def _receipt(self, selector: str) -> dict[str, Any] | None:
        if not self._receipt_root.is_dir() or self._receipt_root.is_symlink():
            return None
        matches = []
        for path in self._receipt_root.glob("execution-*.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as error:
                raise RuntimeError(f"Corrupt Work deletion receipt: {path}") from error
            work = value.get("work", {}) if isinstance(value, dict) else {}
            if isinstance(work, dict) and selector in {
                work.get("name"),
                work.get("execution_id"),
                work.get("scientific_fingerprint"),
            }:
                matches.append(value)
        if len(matches) > 1:
            raise ValueError(f"Deleted Work selector {selector!r} is ambiguous.")
        return matches[0] if matches else None

    def _execution_dir(self, manifest: Path) -> Path:
        execution_dir = manifest.parent
        if (
            execution_dir.is_symlink()
            or not execution_dir.is_relative_to(self.root)
            or execution_dir.parent.parent != self.root
            or not execution_dir.name.startswith("execution-")
        ):
            raise ValueError(f"Unsafe Work Execution ownership path: {execution_dir}")
        return execution_dir


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"Corrupt JSON metadata: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _portable_name(value: str) -> str:
    """Return a readable filesystem component without trusting scientific names."""
    normalized = unicodedata.normalize("NFKC", value).strip()
    selected = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized).strip(".-")
    return (selected or "experiment")[:120]


def _copy_evidence_tree(source: Path, destination: Path, *, hardlink: bool = False) -> None:
    """Copy regular evidence without following links or accepting special files."""
    authored = source.expanduser()
    if authored.is_symlink():
        raise ValueError(f"Refusing symlinked export evidence: {authored}")
    if authored.absolute() != authored.resolve():
        raise ValueError(f"Refusing evidence below a symlinked path: {authored}")
    source = authored.resolve()
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_evidence_file(source, destination, hardlink=hardlink)
        return
    if not source.is_dir():
        raise ValueError(f"Export evidence is not a regular file or directory: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    for root, directories, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        safe_directories: list[str] = []
        for name in sorted(directories):
            candidate = root_path / name
            if candidate.is_symlink():
                raise ValueError(f"Refusing symlinked export evidence: {candidate}")
            if not candidate.is_dir():
                raise ValueError(f"Refusing special export evidence: {candidate}")
            (destination / relative / name).mkdir(parents=True, exist_ok=True)
            safe_directories.append(name)
        directories[:] = safe_directories
        for name in sorted(files):
            candidate = root_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"Refusing non-regular export evidence: {candidate}")
            target = destination / relative / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_evidence_file(candidate, target, hardlink=hardlink)


def _copy_evidence_file(source: Path, destination: Path, *, hardlink: bool) -> None:
    """Materialize one verified file, reusing bytes only for owned temporary evidence."""
    if hardlink:
        try:
            os.link(source, destination, follow_symlinks=False)
            return
        except OSError as error:
            if error.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                errno.EMLINK,
                errno.EOPNOTSUPP,
            }:
                raise
    shutil.copy2(source, destination)


def _inventory(root: Path) -> list[dict[str, Any]]:
    """Hash every exported regular file using portable relative paths."""
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


def _write_basic_analysis_html(analysis: Mapping[str, Any], output: Path) -> None:
    """Keep exports browsable when the optional Plotly renderer is not installed."""
    source = analysis.get("source")
    source = source if isinstance(source, Mapping) else {}
    summary = analysis.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>LambdaForge Study Analysis</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:auto;padding:2rem;
background:#111827;color:#e5e7eb}}
h1,h2{{color:#7dd3fc}}.cards{{display:flex;gap:1rem;flex-wrap:wrap}}
.card{{background:#1f2937;padding:1rem;border-radius:.6rem;min-width:12rem}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1220;padding:1rem;
border-radius:.6rem;border:1px solid #334155}}
</style></head><body><h1>LambdaForge Study Analysis</h1>
<p>This portable fallback exposes the exact persisted analysis. Install
<code>lambdaforge[analysis-report]</code> before exporting for the full interactive Plotly
dashboard.</p>
<div class="cards"><div class="card"><strong>Status</strong><br>{status}</div>
<div class="card"><strong>Complete candidates</strong><br>{candidates}</div>
<div class="card"><strong>Evidence fingerprint</strong><br>{fingerprint}</div></div>
<h2>Persisted analysis JSON</h2><pre>{payload}</pre></body></html>
""".format(
        status=html.escape(str(source.get("status", "unknown"))),
        candidates=html.escape(str(summary.get("complete_candidate_count", "unavailable"))),
        fingerprint=html.escape(str(source.get("evidence_fingerprint", "unavailable"))),
        payload=html.escape(json.dumps(analysis, indent=2, ensure_ascii=False)),
    )
    output.write_text(document, encoding="utf-8")


def _analysis_definition(configuration: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(configuration.get("objective"), Mapping):
        return dict(configuration)
    for level in configuration.get("steps", ()):
        values = (
            level.get("parallel", ())
            if isinstance(level, Mapping) and "parallel" in level
            else (level,)
        )
        for value in values:
            if isinstance(value, Mapping) and isinstance(value.get("objective"), Mapping):
                return dict(value)
    return dict(configuration)


def _execution_resources(execution: Mapping[str, Any]) -> dict[str, Any]:
    runs = [value for value in execution.get("runs", ()) if isinstance(value, Mapping)]
    durations = [
        float(value["duration_seconds"])
        for value in runs
        if isinstance(value.get("duration_seconds"), int | float)
    ]
    gpu_seconds = [
        float(value["duration_seconds"])
        for value in runs
        if isinstance(value.get("duration_seconds"), int | float)
        and (value.get("gpu_index") is not None or value.get("gpu_token") is not None)
    ]
    peak_vram = [
        float(value["peak_vram"])
        for value in runs
        if isinstance(value.get("peak_vram"), int | float)
    ]
    return {
        "wall_seconds_sum": sum(durations),
        "gpu_seconds_sum": sum(gpu_seconds),
        "peak_vram": max(peak_vram, default=None),
        "run_count": len(runs),
    }
