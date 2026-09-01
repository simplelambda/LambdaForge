"""Read and safely remove local Work Execution result envelopes."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from lambdaforge.analysis.Report import write_html
from lambdaforge.analysis.StudyAnalysis import StudyAnalysis
from lambdaforge.work.config import WorkConfig
from lambdaforge.work.failure import render_scientific_failures, scientific_failures
from lambdaforge.work.models import atomic_json


class ResultStore:
    """Treat result manifests as authority while bounding deletion to one run root."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("LAMBDAFORGE_RUN_ROOT")
        self.root = Path(configured or Path.cwd() / ".lambdaforge" / "runs").resolve()

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

    def report(
        self,
        selector: str,
        output: str | Path,
        *,
        recompute: bool = False,
    ) -> Path:
        """Export the same analysis consumed by CLI/TUI to one offline HTML file."""
        return write_html(self.analysis(selector, recompute=recompute), output)

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
