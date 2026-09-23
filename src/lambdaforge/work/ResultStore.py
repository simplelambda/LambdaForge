"""Read and safely remove local Work Execution result envelopes."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping, MutableSequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from lambdaforge.analysis.Report import write_html
from lambdaforge.analysis.StudyAnalysis import StudyAnalysis
from lambdaforge.hpo.ResourceReplay import ReplayPolicyName, ResourceSchedulerReplay
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.work.config import WorkConfig
from lambdaforge.work.failure import render_scientific_failures, scientific_failures
from lambdaforge.work.models import atomic_json


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
            except (OSError, ValueError, TypeError) as error:
                raise RuntimeError(f"Corrupt Work result manifest: {path}") from error
            if not isinstance(value, dict) or value.get("execution_result_version") != 1:
                raise RuntimeError(f"Unsupported Work result manifest: {path}")
            value["_manifest_path"] = str(path)
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

    def delete(self, selector: str, *, apply: bool = False) -> dict[str, Any]:
        """Preview/apply exact-root deletion while preserving shared/durable-independent data."""
        selected = self.select(selector)
        if selected.get("already_deleted"):
            return {**selected, "applied": apply}
        manifest = Path(str(selected.pop("_manifest_path"))).resolve()
        execution_dir = self._execution_dir(manifest)
        payload = {
            "work": selected,
            "execution_dir": str(execution_dir),
            "applied": apply,
            "already_deleted": False,
            "will_remove": ["result envelopes", "Attempts", "owned artifacts", "checkpoints"],
            "preserved": ["published datasets", "shared environments", "reconstructible cache"],
        }
        if apply:
            self._write_receipt(payload)
            if execution_dir.exists():
                shutil.rmtree(execution_dir)
        return payload

    def source(self, selector: str) -> Path:
        """Return the verified authored YAML path recorded for one local Execution."""
        selected = self.select(selector)
        if selected.get("already_deleted"):
            raise ValueError(f"Work Execution {selector!r} was already deleted.")
        manifest = Path(str(selected["_manifest_path"])).resolve()
        execution_dir = self._execution_dir(manifest)
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
        execution_dir = self._execution_dir(manifest)
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
        execution_dir = self._execution_dir(manifest)
        configuration = _read_mapping(execution_dir / "configuration.json")
        definition = _analysis_definition(configuration)
        objective = definition.get("objective")
        objective = objective if isinstance(objective, Mapping) else None
        authored = StudyAnalysis.authored_space(definition)
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
        path = self._execution_dir(manifest) / "analysis.json"
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
        execution_dir = self._execution_dir(Path(str(selected["_manifest_path"])).resolve())
        return ResourceSchedulerReplay.from_execution(execution_dir).replay(policy)

    def report(
        self,
        selector: str,
        output: str | Path,
        *,
        recompute: bool = False,
    ) -> Path:
        """Export the same analysis consumed by CLI/TUI to one offline HTML file."""
        return write_html(self.analysis(selector, recompute=recompute), output)

    def export(
        self,
        selector: str,
        destination: str | Path,
        *,
        supplementary: Mapping[str, str | Path] | None = None,
        copy_published: bool = True,
    ) -> dict[str, Any]:
        """Create one atomic, portable archive directory for a successful Execution.

        ``destination`` is a parent directory.  LambdaForge creates a uniquely named
        child and never overwrites an earlier export.  Supplementary roots are used by
        the control plane for bounded Job/Study evidence downloaded from a provider.
        """
        selected = self.select(selector)
        if selected.get("already_deleted"):
            raise ValueError(f"Work Execution {selector!r} was already deleted.")
        if selected.get("status") != "succeeded":
            raise ValueError(
                "Only a succeeded Work Execution can be exported as final scientific evidence; "
                f"{selector!r} is {selected.get('status', 'unknown')!r}."
            )
        manifest = Path(str(selected["_manifest_path"])).resolve()
        execution_dir = self._execution_dir(manifest)
        execution_id = str(selected.get("execution_id") or execution_dir.name)
        name = str(selected.get("name") or "study")
        authored_parent = Path(destination).expanduser()
        if authored_parent.is_symlink():
            raise ValueError(f"Export destination cannot be a symlink: {authored_parent}")
        parent = authored_parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"Export destination is not a safe directory: {parent}")
        folder = parent / f"{_portable_name(name)}--{_portable_name(execution_id)}"
        if folder.exists() or folder.is_symlink():
            raise FileExistsError(
                f"Export destination already exists: {folder}. Choose another directory or "
                "move the previous export first."
            )

        warnings: list[str] = []
        # Persist analysis before copying so the raw evidence and rendered report share
        # exactly one analysis fingerprint. Ordinary one-Run Work has no HPO objective,
        # so its complete evidence remains exportable without inventing an analysis.
        try:
            analysis = self.analysis(execution_id)
        except ValueError as error:
            analysis = None
            warnings.append(f"Study Analysis was not applicable: {error}")
        stage = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=parent))
        try:
            _copy_evidence_tree(execution_dir, stage / "execution")
            source = None
            try:
                source = self.source(execution_id)
            except (OSError, RuntimeError, ValueError) as error:
                warnings.append(
                    f"Authored YAML snapshot unavailable: {type(error).__name__}: {error}"
                )
            if source is not None:
                _copy_evidence_tree(source, stage / "configuration" / source.name)

            for label, raw_source in sorted((supplementary or {}).items()):
                safe_label = _portable_name(str(label))
                evidence_source = Path(raw_source).expanduser().resolve()
                if not evidence_source.exists():
                    warnings.append(f"Supplementary evidence {label!r} was unavailable.")
                    continue
                _copy_evidence_tree(evidence_source, stage / safe_label)

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
                    write_html(analysis, reports / "study-analysis.html")
                except RuntimeError as error:
                    _write_basic_analysis_html(analysis, reports / "study-analysis.html")
                    warnings.append(
                        "Plotly Study Analysis was unavailable; a structured self-contained HTML "
                        f"fallback was generated instead: {error}"
                    )
            try:
                atomic_json(
                    reports / "resource-replay.json",
                    self.resource_replay(execution_id, policy="recorded"),
                )
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                warnings.append(
                    f"Recorded resource replay was unavailable: {type(error).__name__}: {error}"
                )

            report_hint = (
                "Open reports/study-analysis.html for the interactive scientific report.\n"
                if (reports / "study-analysis.html").is_file()
                else "Study analysis remains available as reports/study-analysis.json.\n"
            )
            (stage / "README.txt").write_text(
                "LambdaForge portable experiment export\n"
                "======================================\n\n"
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
            export_manifest = {
                "lambdaforge_export_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "name": name,
                "execution_id": execution_id,
                "scientific_fingerprint": selected.get("scientific_fingerprint"),
                "status": selected.get("status"),
                "source_result": {
                    "name": selected.get("name"),
                    "execution_id": selected.get("execution_id"),
                    "scientific_fingerprint": selected.get("scientific_fingerprint"),
                    "status": selected.get("status"),
                    "run_count": len(selected.get("runs", ())),
                },
                "inventory": inventory,
                "inventory_scope": "all regular package files except manifest.json itself",
                "file_count": len(inventory),
                "size_bytes": sum(int(item["size_bytes"]) for item in inventory),
                "published_artifact_files": published,
                "warnings": warnings,
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


def _copy_evidence_tree(source: Path, destination: Path) -> None:
    """Copy regular evidence without following links or accepting special files."""
    authored = source.expanduser()
    if authored.is_symlink():
        raise ValueError(f"Refusing symlinked export evidence: {authored}")
    if authored.absolute() != authored.resolve():
        raise ValueError(f"Refusing evidence below a symlinked path: {authored}")
    source = authored.resolve()
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
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
            shutil.copy2(candidate, target)


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
