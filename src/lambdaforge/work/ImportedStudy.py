"""Local, read-only Console projections of verified portable Study evidence.

Indexes are derived during the import transaction, not by repeatedly loading a large result
on root-screen refresh. Original scientific records stay byte-for-byte in portable/.
"""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.study_projection import (
    analysis_panel,
    panel_detail,
    projection_page,
    study_table,
    trial_detail,
)
from lambdaforge.work.models import atomic_json


def import_index(
    manifest: Mapping[str, Any], result: Mapping[str, Any], *, package: Path
) -> dict[str, Any]:
    """Small local-placement index; no metrics history or Run evidence belongs here."""
    summary = result.get("summary") or {}
    from lambdaforge.work.StudyImport import _mapping

    def parameter_study(definition: Mapping[str, Any]) -> bool:
        if "search" in definition or "sweep" in definition or "replicates" in definition:
            return True
        seeds = definition.get("seeds")
        if isinstance(seeds, list) and len(seeds) > 1:
            return True
        for step in definition.get("steps", ()):
            if isinstance(step, Mapping):
                children = step.get("parallel", [step])
                if any(parameter_study(child) for child in children if isinstance(child, Mapping)):
                    return True
        return False

    expected = bool(summary.get("study_design")) or parameter_study(
        _mapping(package / "execution/configuration.json")
    )
    states = Counter(str(run.get("status", "unknown")) for run in result.get("runs", ()))
    counts = {
        "candidates": len(summary.get("candidates", ())),
        "completed_runs": states["succeeded"],
        "pruned_runs": states["pruned"],
        "active_runs": states["running"],
        "queued_runs": states["pending"],
    }
    return {
        "import_index_version": 1,
        "work_id": "import:" + str(manifest["execution_id"]),
        "execution_id": manifest["execution_id"],
        "name": manifest["name"],
        "state": manifest["status"],
        "cluster": "imported · local snapshot",
        "study_expected": expected,
        "attempts": len(result.get("runs", ())),
        "imported": True,
        "study_selector": "import:" + str(manifest["execution_id"]),
        "study": {
            "detail_level": "overview",
            "counts": counts,
            "objective": summary.get("objective") or {},
        }
        if expected
        else None,
    }


def write_import_views(
    stage: Path, result: Mapping[str, Any], *, package: Path | None = None
) -> None:
    """Derive hierarchical read models once while the verified import is unpublished."""
    from lambdaforge.work.result_projection import is_study
    from lambdaforge.work.ResultStore import _portable_name
    from lambdaforge.work.StudyImport import _mapping

    configuration = _mapping((package or stage / "portable") / "execution/configuration.json")
    if not is_study(configuration):
        (stage / "import-view").mkdir(parents=True, exist_ok=True)
        return

    source = (package or stage / "portable") / "study"
    target = stage / "import-view"
    summary = result.get("summary") or {}
    # Native exports retain the scientific Study snapshot; prefer it to reconstructing a
    # presentation from results. Local exports without provider telemetry still have Runs.
    path = source / "summary.json"
    package_root = package or stage / "portable"
    relocated_artifacts: dict[str, dict[str, str]] = {}
    published_paths: dict[str, str] = {}
    for run_index, native in enumerate(result.get("runs", ())):
        for artifact_index, artifact in enumerate(native.get("artifacts", ())):
            metadata = artifact.get("metadata") or {}
            published = artifact.get("published_path") or metadata.get("published_to")
            if not isinstance(published, str) or not published:
                continue
            name = _portable_name(str(artifact.get("name") or Path(published).name or "artifact"))
            relative = f"published-artifacts/run-{run_index:05d}/{artifact_index:04d}-{name}"
            if (package_root / relative).exists():
                published_paths[published] = relative
            if published in published_paths:
                relocated_artifacts.setdefault(native["run_dir"], {})[str(artifact["name"])] = (
                    published_paths[published]
                )
    if path.is_file():
        study = _mapping(path, verified_aggregate=True)
    else:
        candidates: dict[int, dict[str, Any]] = {}
        for native in result.get("runs", ()):
            trial = int((native.get("trial") or {}).get("index") or 1)
            candidate = candidates.setdefault(
                trial,
                {
                    "trial": trial,
                    "parameters": native.get("parameters") or {},
                    "runs": [],
                },
            )
            seed = native.get("seed")
            token = "none" if seed is None else str(seed).replace("-", "n")
            objective = native.get("objective_observation") or {}
            metric = (summary.get("objective") or {}).get("metric")
            best = objective.get("best", (native.get("metrics") or {}).get(metric))
            directory = PurePosixPath(native["run_dir"])
            observed = {
                "key": f"trial-{trial:05d}-seed-{token}",
                "trial": trial,
                "seed": seed,
                "state": native.get("status"),
                "run_dir": native.get("run_dir"),
                "log_path": str(directory / str(native.get("logs") or "work.log")),
                "metrics_path": str(directory / "metrics.jsonl"),
                "training_metrics_path": str(directory / "training-metrics.jsonl"),
                "best_observed_objective": best,
                "best_objective": best,
                "current_observed_objective": objective.get("current"),
                "best_step": objective.get("best_step"),
                "latest_step": objective.get("current_step"),
                "final_objective": best if native.get("status") == "succeeded" else None,
                "phase": native.get("study_phase"),
                "gpu_index": native.get("gpu_index"),
                "gpu_token": native.get("gpu_token"),
                "failure": native.get("failure"),
                "duration_seconds": native.get("duration_seconds"),
            }
            # Multiple Attempts remain in portable evidence; show the latest logical Run.
            candidate["runs"] = [r for r in candidate["runs"] if r["key"] != observed["key"]]
            candidate["runs"].append(observed)
        for entry in summary.get("candidates", ()):
            if isinstance(entry, Mapping) and entry.get("trial") in candidates:
                candidate = candidates[entry["trial"]]
                candidate.update({k: v for k, v in entry.items() if k != "runs"})
                candidate["selection_objective"] = entry.get("selection_score")
                candidate["state"] = (
                    "succeeded"
                    if all(r["state"] == "succeeded" for r in candidate["runs"])
                    else "incomplete"
                )
        study = {
            "name": result["name"],
            "execution_id": result["execution_id"],
            "objective": summary.get("objective") or {},
            "candidates": list(candidates.values()),
            "status": result["status"],
            "counts": {
                "candidates": len(candidates),
                "completed_runs": summary.get("completed_runs", 0),
            },
        }
    atomic_json(target / "interactive.json", study_table(study))
    atomic_json(target / "hpo.json", panel_detail(study, "hpo"))
    atomic_json(target / "resources.json", panel_detail(study, "resources"))
    atomic_json(
        target / "index.json",
        {
            "candidates": [
                {"trial": c["trial"], "parameters": c.get("parameters", {})}
                for c in study.get("candidates", ())
            ],
        },
    )
    for candidate in study.get("candidates", ()):
        trial = candidate.get("trial")
        if type(trial) is not int or trial < 1:
            raise ValueError("Imported Study has an invalid Trial identity.")
        atomic_json(target / "trials" / f"trial-{trial:05d}.json", trial_detail(candidate))
        for run in candidate.get("runs", ()):
            key = str(run.get("key", ""))
            if not re.fullmatch(r"trial-\d{5}-seed-(?:none|n?\d+)", key):
                raise ValueError("Imported Study has an invalid Run selector.")
            atomic_json(
                target / "runs" / f"{key}.json",
                {
                    "parameters": candidate.get("parameters") or {},
                    "run": run,
                    "objective": study.get("objective") or {},
                    "imported_artifacts": relocated_artifacts.get(run.get("run_dir", ""), {}),
                },
            )


class ImportedStudy:
    """Observe one imported archive, never an executable Job or recovery target."""

    def __init__(self, root: Path, selector: str) -> None:
        from lambdaforge.work.StudyImport import _mapping

        execution = selector.removeprefix("import:")
        if not re.fullmatch(r"execution-[A-Za-z0-9_-]+", execution):
            raise ValueError("Invalid imported Execution selector.")
        matches = list(root.glob(f"*/{execution}/import.json"))
        if len(matches) != 1:
            raise ValueError("Imported Execution is missing or ambiguous in this project.")
        receipt = _mapping(matches[0])
        if receipt.get("execution_id") != execution or receipt.get("will_execute") is not False:
            raise ValueError("Invalid imported Study ownership receipt.")
        if (
            receipt.get("study_import_version") != 1
            or receipt.get("evidence_root") != "portable/execution"
        ):
            raise ValueError("Invalid imported Study placement contract.")
        self.directory = matches[0].parent
        self.portable = self.directory / "portable"
        if self.portable.resolve() != self.portable or not self.portable.is_dir():
            raise ValueError("Imported evidence root is missing or symbolic.")
        self.selector = selector

    @staticmethod
    def rows(root: Path, *, include_works: bool = False) -> list[dict[str, Any]]:
        from lambdaforge.work.StudyImport import _mapping

        rows = []
        for path in sorted(root.glob("*/execution-*/import-study.json")):
            value = _mapping(path)
            if (
                value.get("import_index_version") != 1
                or value.get("execution_id") != path.parent.name
            ):
                raise ValueError("Invalid imported Study collection index.")
            ImportedStudy(root, value["work_id"])
            if (
                value["work_id"] != "import:" + path.parent.name
                or value.get("imported") is not True
            ):
                raise ValueError("Imported Study index contains an invalid observer identity.")
            if value.get("study_expected") or include_works:
                rows.append(value)  # Ordinary imported Works remain in Results, not fake Studies.
        return rows

    def view(self, view: str, **options: Any) -> dict[str, Any]:
        """Use the native bounded projection protocol on the local placement."""
        # Run files already contain the selected detail, not the host's single-Run format.
        if view == "run":
            from lambdaforge.work.StudyImport import _mapping

            key = str(options.get("run", ""))
            if not re.fullmatch(r"trial-\d{5}-seed-(?:none|n?\d+)", key):
                raise ValueError("Invalid imported Run selector.")
            return _mapping(self.directory / "import-view/runs" / f"{key}.json")
        request = {"view": view, **options}
        data = bytearray()
        while True:
            page = projection_page(str(self.directory / "import-view"), request)
            if page.get("missing"):
                raise ValueError(f"No persisted {view} evidence in this imported snapshot.")
            if page.get("restart"):
                raise ValueError(
                    "Imported presentation changed during observation; retry the view."
                )
            data.extend(base64.b64decode(page["data"]))
            if page["eof"]:
                return json.loads(data)
            request.update(offset=page["next_offset"], fingerprint=page["fingerprint"])

    def analysis(self, *, full: bool = False) -> dict[str, Any]:
        from lambdaforge.work.StudyImport import _mapping

        for path in (
            self.portable / "execution/analysis.json",
            self.portable / "reports/study-analysis.json",
            self.portable / "study/analysis-read-cache.json",
        ):
            if path.is_file():
                value = _mapping(path, verified_aggregate=True)
                if path.name == "analysis-read-cache.json":
                    value = value["value"]
                return value if full else analysis_panel(value)
        raise ValueError(
            "No persisted Analysis in this export; imports never refit scientific evidence."
        )

    def actions(self) -> tuple[dict[str, Any], ...]:
        paths = (
            self.portable / "study/controller-history.jsonl",
            self.portable / "execution/hpo-control/decisions.jsonl",
        )
        for path in paths:
            if path.is_file():
                if path.resolve() != path:
                    raise ValueError("Symbolic imported decision history is unsafe.")
                with path.open(encoding="utf-8") as stream:
                    actions: list[dict[str, Any]] = []
                    while True:
                        line = stream.readline(512 * 1024 + 1)
                        if not line:
                            return tuple(actions)
                        if len(line.encode("utf-8")) > 512 * 1024:
                            raise ValueError("Imported decision exceeds the history page bound.")
                        if line.strip():
                            actions.append(json.loads(line))
        return ()

    def run(self, key: str, *, tail: int, curve_points: int) -> dict[str, Any]:
        from lambdaforge.controlplane.JobService import JobService
        from lambdaforge.controlplane.LocalTransport import LocalTransport
        from lambdaforge.work.StudyImport import _relative, relative_run_path

        if tail < 1 or not 10 <= curve_points <= 500:
            raise ValueError("Run tail must be positive and curve_points between 10 and 500.")
        detail = self.view("run", run=key)
        selected = detail["run"]
        parts = PurePosixPath(selected["run_dir"]).parts
        if len(parts) < 4:
            raise ValueError("Imported Run has no safe native Attempt path.")
        native = {**selected, "run_id": parts[-3]}
        run_dir = self.portable / "execution" / relative_run_path(native)
        if run_dir.resolve() != run_dir:
            raise ValueError("Imported Run placement is symbolic.")
        result_path = run_dir / "result.json"
        if result_path.resolve() != result_path:
            raise ValueError("Imported Run result is symbolic.")

        def relocate(value: Any, default: str) -> PurePosixPath | None:
            relative = PurePosixPath(default)
            if value:
                original = PurePosixPath(str(value))
                try:
                    relative = original.relative_to(PurePosixPath(selected["run_dir"]))
                except ValueError as error:
                    raise ValueError("Imported Run evidence escaped its owned Attempt.") from error
            if ".." in relative.parts:
                raise ValueError("Unsafe imported Run evidence path.")
            path = run_dir / relative
            if path.resolve() != path:
                raise ValueError("Symbolic imported Run evidence is unsafe.")
            return PurePosixPath(path) if path.is_file() else None

        log = relocate(selected.get("log_path"), "work.log")
        metrics = tuple(
            path
            for path in (
                relocate(selected.get("metrics_path"), "metrics.jsonl"),
                relocate(selected.get("training_metrics_path"), "training-metrics.jsonl"),
            )
            if path is not None
        )
        dashboard = JobService._study_run_dashboard(
            detail,
            LocalTransport(),
            job_id=self.selector,
            cluster="imported snapshot",
            run_key=key,
            log_path=log,
            run_dir=PurePosixPath(run_dir),
            metric_paths=metrics,
            tail=tail,
            curve_points=curve_points,
        )
        dashboard["paths"]["run_dir"] = str(run_dir)
        dashboard["placement"] = None
        for artifact in dashboard["artifacts"]:
            relative = detail.get("imported_artifacts", {}).get(artifact["name"])
            if relative:
                path = self.portable / _relative(relative)
                if path.resolve() != path or not path.exists():
                    raise ValueError("Imported published artifact placement is missing or unsafe.")
                artifact["path"] = str(path)
            elif artifact["retention"] == "published-only":
                artifact["path"] = None  # Preserve origin, never present a remote path as local.
        return dashboard
