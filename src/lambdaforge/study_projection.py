"""Small, transport-safe projections of live Study telemetry.

The authoritative ``summary.json`` is intentionally rich.  Interactive clients must not
download it merely to draw a trial table: large studies can contain many paths, failures and
resource diagnostics per Run.  These helpers keep one stable, bounded read model next to the
authoritative evidence without becoming another result store.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_CANDIDATE_FIELDS = (
    "trial",
    "parameters",
    "state",
    "selection_objective",
    "selection_seed_count",
    "selection_standard_error",
    "current_objective",
    "best_objective",
    "partially_censored",
    "pareto_optimal",
    "cost",
    "feasibility",
    "confirmation_status",
    "diagnostic_metrics",
)

_RUN_FIELDS = (
    "key",
    "seed",
    "phase",
    "purpose",
    "target_questions",
    "fidelity",
    "state",
    "current_observed_objective",
    "best_observed_objective",
    "final_objective",
    "objective_status",
    "objective_censoring",
    "latest_step",
    "best_step",
    "best_objective",
    "duration_seconds",
    "gpu_index",
    "gpu_token",
    "termination_type",
    "prune_reason",
    "scientific_continuation",
    "evidence_requirement",
    # Small owned references keep selected-Run detail lazy: the interactive index names the
    # evidence, while JobService reads only the log/scalar file the user opens.  Contents and
    # bulky failure diagnostics never belong in this projection.
    "run_dir",
    "log_path",
    "metrics_path",
    "training_metrics_path",
)


def interactive_study(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project telemetry onto the complete but compact interactive Study index."""
    if (
        value.get("detail_level") == "interactive"
        and int(value.get("interactive_projection_version", 0) or 0) >= 2
    ):
        return copy.deepcopy(dict(value))
    candidates: list[dict[str, Any]] = []
    raw_candidates = value.get("candidates", ())
    if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, str | bytes):
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, Mapping):
                continue
            candidate = {
                field: copy.deepcopy(raw_candidate[field])
                for field in _CANDIDATE_FIELDS
                if field in raw_candidate
            }
            runs: list[dict[str, Any]] = []
            raw_runs = raw_candidate.get("runs", ())
            if isinstance(raw_runs, Sequence) and not isinstance(raw_runs, str | bytes):
                for raw_run in raw_runs:
                    if isinstance(raw_run, Mapping):
                        runs.append(
                            {
                                field: copy.deepcopy(raw_run[field])
                                for field in _RUN_FIELDS
                                if field in raw_run
                            }
                        )
            candidate["runs"] = runs
            candidates.append(candidate)

    controller = value.get("controller")
    controller = controller if isinstance(controller, Mapping) else {}
    admission = value.get("admission")
    admission = admission if isinstance(admission, Mapping) else {}
    projected_controller = {
        field: copy.deepcopy(controller[field])
        for field in (
            "last",
            "recent",
            "history_count",
            "surrogate_belief",
            "scheduler",
            "initialization",
            "controller_telemetry_version",
            "scientific_configuration_required",
        )
        if field in controller
    }
    return {
        field: copy.deepcopy(value[field])
        for field in (
            "study_telemetry_version",
            "name",
            "execution_id",
            "strategy",
            "design",
            "objective",
            "planned_runs",
            "planned_candidates",
            "status",
            "design_status",
            "scientific_status",
            "finish_reason",
            "required_runs",
            "required_completed",
            "required_pruned",
            "required_failed",
            "required_missing",
            "evidence_completion_fraction",
            "attempted_completion_fraction",
            "counts",
            "cost",
            "initial_design",
            "coverage_state",
            "hpo_analysis",
            "surrogate_belief",
            "finished",
            "created_at_utc",
            "updated_at_utc",
        )
        if field in value
    } | {
        "detail_level": "interactive",
        "interactive_projection_version": 2,
        "candidates": candidates,
        "controller": projected_controller,
        # Admission history is diagnostic history.  The Study workspace needs only the latest
        # device/readiness explanation; the durable resource trace remains authoritative.
        "admission": {
            "current": copy.deepcopy(admission.get("current")),
            "updated_at_utc": admission.get("updated_at_utc"),
        },
    }


__all__ = [
    "interactive_study",
    "study_table",
    "trial_detail",
    "panel_detail",
    "analysis_panel",
    "projection_page",
]


def study_table(value: Mapping[str, Any]) -> dict[str, Any]:
    """Trial table only: no seeds, parameters, curves, diagnostics or model state."""
    fields = (
        "study_telemetry_version",
        "name",
        "execution_id",
        "strategy",
        "objective",
        "planned_runs",
        "planned_candidates",
        "status",
        "design_status",
        "lifecycle",
        "scientific_status",
        "finish_reason",
        "required_runs",
        "required_completed",
        "required_pruned",
        "required_failed",
        "required_missing",
        "evidence_completion_fraction",
        "attempted_completion_fraction",
        "counts",
        "cost",
        "run_states",
        "leader_parameters",
        "finished",
        "updated_at_utc",
    )
    rows = []
    states: Counter[str] = Counter()
    leader: Mapping[str, Any] | None = None
    mode = str((value.get("objective") or {}).get("mode", "max"))
    for candidate in value.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        row = {
            key: copy.deepcopy(candidate[key])
            for key in (
                "trial",
                "state",
                "selection_objective",
                "selection_seed_count",
                "selection_standard_error",
                "current_objective",
                "best_objective",
                "best_seed",
                "best_step",
                "seed_count",
                "partially_censored",
                "pareto_optimal",
                "latest_step",
                "gpu_indices",
            )
            if key in candidate
        }
        runs = [run for run in candidate.get("runs", ()) if isinstance(run, Mapping)]
        if runs:
            row["seed_count"] = len(runs)
            states.update(str(run.get("state", "unknown")) for run in runs)
            row["latest_step"] = max((int(run.get("latest_step") or 0) for run in runs), default=0)
            row["gpu_indices"] = sorted(
                {
                    int(run["gpu_index"])
                    for run in runs
                    if run.get("state") == "running" and run.get("gpu_index") is not None
                }
            )
        rows.append(row)
        score = candidate.get("selection_objective")
        if isinstance(score, int | float) and not isinstance(score, bool):
            if leader is None or (
                score < leader["selection_objective"]
                if mode == "min"
                else score > leader["selection_objective"]
            ):
                leader = candidate
    projected = {
        "interactive_projection_version": 3,
        "detail_level": "interactive",
        **{key: copy.deepcopy(value[key]) for key in fields if key in value},
        "candidates": rows,
    }
    if states:
        projected["run_states"] = dict(states)
    if leader is not None and isinstance(leader.get("parameters"), Mapping):
        # One leader preview is visible in Study Overview; no other candidate parameters travel.
        projected["leader_parameters"] = copy.deepcopy(leader["parameters"])
    return projected


def trial_detail(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """One candidate's seed table and diagnostics, without any scalar history."""
    return {key: copy.deepcopy(candidate[key]) for key in _CANDIDATE_FIELDS if key in candidate} | {
        "detail_level": "trial",
        "runs": [
            {key: copy.deepcopy(run[key]) for key in _RUN_FIELDS if key in run}
            for run in candidate.get("runs", ())
            if isinstance(run, Mapping)
        ],
    }


def panel_detail(value: Mapping[str, Any], view: str) -> dict[str, Any]:
    if view == "resources":
        admission = value.get("admission") or {}
        return {
            "admission": {
                key: copy.deepcopy(admission[key])
                for key in ("current", "updated_at_utc")
                if key in admission
            }
        }
    fields = (
        "hpo_analysis",
        "controller",
        "surrogate_belief",
        "initial_design",
        "coverage_state",
    )
    panel = {key: copy.deepcopy(value[key]) for key in fields if key in value}
    design = value.get("design")
    if isinstance(design, Mapping):
        # The HPO plan card displays policy/counts, not every required (candidate, seed).
        panel["design"] = {
            key: copy.deepcopy(design[key])
            for key in ("type", "goal", "replication", "seed_source", "policy_version")
            if key in design
        }
        evidence = design.get("evidence")
        if isinstance(evidence, Mapping):
            panel["design"]["evidence"] = {
                key: copy.deepcopy(evidence[key])
                for key in ("required_run_count", "optional_run_count")
                if key in evidence
            }
    return panel


def analysis_panel(value: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate console analysis, without per-Run evidence or the HTML research catalogue."""
    fields = (
        "analysis_version",
        "source",
        "status",
        "generated_at_utc",
        "objective",
        "summary",
        "search_space",
        "search_space_source",
        "winner",
        "seed_analysis",
        "surrogate",
        "parameter_importance",
        "top_region_importance",
        "response_curves",
        "interactions",
        "coverage",
        "resources",
        "pareto",
        "findings",
        "scientific_understanding",
        "parameter_conclusions",
        "interaction_conclusions",
        "practical_optimal_region",
        "live_hpo",
    )
    candidate_fields = (
        "trial",
        "parameters",
        "state",
        "mean",
        "standard_error",
        "n",
        "objective_components",
        "diagnostic_metrics",
        "resource_cost",
    )
    return {key: copy.deepcopy(value[key]) for key in fields if key in value} | {
        "candidates": [
            {key: copy.deepcopy(candidate[key]) for key in candidate_fields if key in candidate}
            for candidate in value.get("candidates", ())
            if isinstance(candidate, Mapping)
        ]
    }


def _cached_analysis(directory: Path, value: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    """One disposable, host-local post-hoc cache, never evaluated in the HPO heartbeat."""
    from lambdaforge.analysis.StudyAnalysis import StudyAnalysis
    from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

    path = directory / "analysis-read-cache.json"
    lock = directory / "analysis-read-cache.lock"
    if path.is_symlink() or lock.is_symlink():
        raise ValueError("Symlinked analysis cache is unsafe.")
    with CrossProcessFileLock(lock, shared=False, timeout_seconds=100.0, poll_interval_seconds=0.1):
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                return dict(cached["value"])
        design = StudyAnalysis._study_design(value)
        # The reader also serves immutable older workers. Pass sweep geometry explicitly so
        # their compute() cannot re-infer conditional domains from incomplete live evidence.
        authored_space = design.get("space") if design.get("type") == "sweep" else None
        analysis = StudyAnalysis.compute(
            value,
            objective=value.get("objective"),
            authored_space=authored_space if isinstance(authored_space, Mapping) else None,
            status="provisional",
            provisional_bootstrap_replicates=200,
        )
        if isinstance(value.get("hpo_analysis"), Mapping):
            analysis["live_hpo"] = value["hpo_analysis"]
        descriptor, temporary = tempfile.mkstemp(prefix=".analysis-read-", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
                json.dump(
                    {"fingerprint": fingerprint, "value": analysis}, sink, separators=(",", ":")
                )
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return analysis


def projection_page(root: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Host-side projection and byte paging; usable on historical workers with stdlib only.

    A changing snapshot restarts the read, never combines pages from different generations.
    Only explicitly requested provisional analysis uses a disposable host-local cache.
    No scientific/runtime state is mutated by this reader.
    """
    directory = Path(root)
    view = str(request.get("view", "interactive"))
    if view not in {
        "interactive",
        "trial",
        "run",
        "hpo",
        "resources",
        "analysis",
        "analysis-report",
        "report-sections",
    }:
        raise ValueError("Unknown Study read view.")
    trial = int(request.get("trial", 0))
    if view in {"trial", "run"} and trial < 1:
        raise ValueError("Study Trial must be a positive integer.")
    selector = str(request.get("run", ""))
    path = directory / "summary.json"
    preferred = {
        "interactive": directory / "interactive.json",
        "trial": directory / "trials" / f"trial-{trial:05d}.json",
        "run": directory / "runs" / f"{selector}.json",
        "hpo": directory / "hpo.json",
        "resources": directory / "resources.json",
        "analysis": directory / "analysis-panel.json",
        "analysis-report": directory / "analysis.json",
        "report-sections": directory / "summary.json",
    }[view]
    if view == "run":
        import re

        if re.fullmatch(r"trial-\d{5}-seed-(?:none|n?\d+)", selector) is None:
            raise ValueError("Invalid Study Run selector.")
    if preferred.is_symlink():
        raise ValueError("Symlinked Study read model is unsafe.")
    if preferred.is_file():
        path = preferred
    elif view == "analysis" and (directory / "analysis.json").is_file():
        path = directory / "analysis.json"
    if not path.is_file():
        return {"missing": True}
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Symlinked Study read model is unsafe.")
    before = path.stat()
    query = str(request.get("query", "")).strip().lower()
    if len(query) > 256:
        raise ValueError("Study filter is too long.")
    parameter_index = directory / "index.json"
    observation_path = directory / "observations" / f"{selector}.json"
    extra = ""
    if query and parameter_index.is_file():
        if parameter_index.is_symlink():
            raise ValueError("Symlinked Study index is unsafe.")
        indexed_stat = parameter_index.stat()
        extra = f":{indexed_stat.st_ino}:{indexed_stat.st_mtime_ns}:{indexed_stat.st_size}"
    if view == "run" and observation_path.is_file():
        if observation_path.is_symlink() or observation_path.parent.is_symlink():
            raise ValueError("Symlinked Study observation is unsafe.")
        observed_stat = observation_path.stat()
        extra += f":{observed_stat.st_ino}:{observed_stat.st_mtime_ns}:{observed_stat.st_size}"
    fingerprint = hashlib.sha256(
        f"{path}:{before.st_dev}:{before.st_ino}:{before.st_mtime_ns}:{before.st_size}{extra}".encode()
    ).hexdigest()
    known = request.get("fingerprint")
    if view.startswith("analysis") and path.name == "summary.json" and request.get("offset", 0):
        cache = directory / "analysis-read-cache.json"
        if cache.is_file() and not cache.is_symlink():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == known:
                projected = (
                    analysis_panel(cached["value"]) if view == "analysis" else cached["value"]
                )
                return _encoded_page(projected, str(known), int(request["offset"]))
    if request.get("offset", 0) == 0 and known == fingerprint:
        return {"unchanged": True, "fingerprint": fingerprint}
    if request.get("offset", 0) and known != fingerprint:
        return {"restart": True}
    if view == "report-sections":
        # HTML is immutable finalized evidence. Materialize one disposable report-only byte
        # cache, then stream pages without rereading/encoding every document per page.
        return _section_page(directory, path, fingerprint, int(request.get("offset", 0)))
    # These view-specific files are already projected by the execution host. Read only the
    # requested byte range: reparsing/re-encoding a large final analysis per page is quadratic.
    if path == preferred and view in {"trial", "hpo", "resources", "analysis", "analysis-report"}:
        return _file_page(path, fingerprint, int(request.get("offset", 0)), before)
    if path == preferred and view == "interactive" and not query:
        import re

        # Native v3 writes its version first. Historical sorted/rich indexes still go through
        # the host projection below; modern large tables are streamed without repeated parsing.
        with path.open("rb") as stream:
            prefix = stream.read(128)
        if re.match(rb'^\{\s*"interactive_projection_version"\s*:\s*3\s*,', prefix):
            return _file_page(path, fingerprint, int(request.get("offset", 0)), before)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid Study read model.")
    if view == "interactive":
        raw_candidates = value.get("candidates", ())
        value = study_table(value)
        if query:
            # Search parameters on their owner, without downloading them for every Trial.
            if parameter_index.is_file():
                raw_candidates = json.loads(parameter_index.read_text(encoding="utf-8")).get(
                    "candidates", ()
                )
            searched_parameters = {
                c.get("trial"): c.get("parameters", {})
                for c in raw_candidates
                if isinstance(c, Mapping)
            }
            value["candidates"] = [
                c
                for c in value["candidates"]
                if query
                in " ".join(
                    (
                        str(c.get("trial", "")),
                        str(c.get("state", "")),
                        json.dumps(searched_parameters.get(c.get("trial"), {}), sort_keys=True),
                    )
                ).lower()
            ]
    elif view in {"trial", "run"}:
        if path.name == "summary.json":
            candidate = next(
                (
                    c
                    for c in value.get("candidates", ())
                    if isinstance(c, Mapping) and c.get("trial") == trial
                ),
                None,
            )
            if candidate is None:
                return {"missing": True}
            objective = value.get("objective", {})
            value = trial_detail(candidate)
            if view == "run":
                selected = next(
                    (r for r in candidate.get("runs", ()) if r.get("key") == selector), None
                )
                if selected is None:
                    return {"missing": True}
                value = {
                    "parameters": candidate.get("parameters", {}),
                    "run": selected,
                    "objective": objective,
                }
        elif view == "run":
            if observation_path.is_file():
                observed = json.loads(observation_path.read_text(encoding="utf-8"))
                if not isinstance(observed, Mapping):
                    raise ValueError("Invalid Study observation.")
                value.update(observed)
            candidate_path = directory / "trials" / f"trial-{trial:05d}.json"
            if candidate_path.is_symlink() or candidate_path.parent.is_symlink():
                raise ValueError("Symlinked Study Trial metadata is unsafe.")
            parameters: dict[str, Any] = {}
            if candidate_path.is_file() and not candidate_path.is_symlink():
                parameters = json.loads(candidate_path.read_text(encoding="utf-8")).get(
                    "parameters", {}
                )
            else:
                index = directory / "index.json"
                if index.is_file() and not index.is_symlink():
                    indexed = json.loads(index.read_text(encoding="utf-8"))
                    parameters = next(
                        (
                            c.get("parameters", {})
                            for c in indexed.get("candidates", ())
                            if c.get("trial") == trial
                        ),
                        {},
                    )
            header = directory / "overview.json"
            if not header.is_file():
                header = directory / "index.json"
            if not header.is_file():
                header = directory / "summary.json"
            objective = {}
            if header.is_file() and not header.is_symlink():
                objective = json.loads(header.read_text(encoding="utf-8")).get("objective", {})
            value = {"parameters": parameters, "run": value, "objective": objective}
    elif view in {"hpo", "resources"}:
        value = panel_detail(value, view)
    elif view.startswith("analysis"):
        if path.name == "summary.json":
            value = _cached_analysis(directory, value, fingerprint)
        if view == "analysis":
            value = analysis_panel(value)
    after = path.stat()
    if not view.startswith("analysis") and (before.st_ino, before.st_mtime_ns, before.st_size) != (
        after.st_ino,
        after.st_mtime_ns,
        after.st_size,
    ):
        return {"restart": True}
    return _encoded_page(value, fingerprint, int(request.get("offset", 0)))


def _section_page(directory: Path, source: Path, generation: str, offset: int) -> dict[str, Any]:
    from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

    cache = directory / "report-sections-read-cache.bin"
    lock = directory / "report-sections-read-cache.lock"
    if cache.is_symlink() or lock.is_symlink():
        raise ValueError("Symlinked HTML section cache is unsafe.")
    header = generation.encode("ascii") + b"\n"
    with CrossProcessFileLock(lock, shared=False, timeout_seconds=100.0, poll_interval_seconds=0.1):
        valid = False
        if cache.is_file():
            with cache.open("rb") as stream:
                valid = stream.read(len(header)) == header
        if not valid:
            value = json.loads(source.read_text(encoding="utf-8"))
            runs = [
                dict(run, trial=candidate.get("trial"))
                for candidate in value.get("candidates", ())
                if isinstance(candidate, Mapping)
                for run in candidate.get("runs", ())
                if isinstance(run, Mapping)
            ]
            sections = read_html_sections(runs, directory.parent)
            encoded = json.dumps({"sections": sections}, separators=(",", ":")).encode("utf-8")
            descriptor, temporary = tempfile.mkstemp(prefix=".report-sections-", dir=directory)
            try:
                with os.fdopen(descriptor, "wb") as sink:
                    sink.write(header + encoded)
                os.replace(temporary, cache)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        before = cache.stat()
        return _file_page(cache, generation, offset, before, base_offset=len(header))


def read_html_sections(
    runs: Sequence[Any], owned_root: Path, *, relocate: bool = False
) -> list[dict[str, str]]:
    """Explicit report-only read of finalized, checksummed project HTML artifacts.

    No directory walking, asset inference or consumer code execution. The reader is
    stdlib-only so the current controller can serve immutable historical runtimes.
    """
    root = owned_root.resolve()
    sections: list[dict[str, str]] = []
    total = 0
    for run in runs:
        if not isinstance(run, Mapping) or not run.get("run_dir"):
            continue
        if "artifacts" in run and not any(
            isinstance(artifact, Mapping) and artifact.get("role") == "html-section"
            for artifact in run.get("artifacts", ())
        ):
            continue
        raw_dir = Path(str(run["run_dir"]))
        run_dir = raw_dir if raw_dir.is_absolute() else root / raw_dir
        if relocate and not run_dir.is_relative_to(root):
            # Portable provider archives retain original paths in immutable result envelopes.
            # Reconnect only the exact recorded native Run/Attempt under the verified local
            # Execution; never follow the stale remote absolute path or infer by filename.
            import re

            run_id, attempt_id = run.get("run_id"), run.get("attempt_id")
            if (
                run.get("execution_id") != root.name
                or not isinstance(run_id, str)
                or re.fullmatch(r"run-[A-Za-z0-9-]+", run_id) is None
                or not isinstance(attempt_id, str)
                or re.fullmatch(r"attempt-\d{4}", attempt_id) is None
                or raw_dir.parts[-4:] != ("runs", run_id, "attempts", attempt_id)
            ):
                raise ValueError("Cannot reconnect project HTML to its exact exported Run/Attempt.")
            run_dir = root / "runs" / run_id / "attempts" / attempt_id
        if any(parent.is_symlink() for parent in (run_dir, *run_dir.parents)):
            raise ValueError("Symlinked HTML section Run is unsafe.")
        run_dir = run_dir.resolve()
        if not run_dir.is_relative_to(root):
            raise ValueError("HTML section Run is outside its owned Execution/Job.")
        result: Mapping[str, Any] = run
        if "artifacts" not in result:
            path = run_dir / "result.json"
            if not path.exists():
                continue
            if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Unsafe or oversized HTML section result metadata.")
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, Mapping):
                raise ValueError("Invalid HTML section result metadata.")
        for artifact in result.get("artifacts", ()):
            if not isinstance(artifact, Mapping) or artifact.get("role") != "html-section":
                continue
            metadata = artifact.get("metadata") or {}
            descriptor = metadata.get("html_section") if isinstance(metadata, Mapping) else None
            if not isinstance(descriptor, Mapping):
                raise ValueError("HTML section declaration is missing.")
            name, title = descriptor.get("name"), descriptor.get("title")
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
                raise ValueError("Invalid HTML section name.")
            if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120:
                raise ValueError("Invalid HTML section title.")
            relative = Path(str(artifact.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("Unsafe HTML section artifact path.")
            path = run_dir / relative
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                raise ValueError("Symlinked HTML section artifact is unsafe.")
            if not path.is_file():
                raise ValueError(f"HTML section artifact is missing: {artifact.get('name')}")
            size = path.stat().st_size
            if size > 16 * 1024 * 1024 or total + size > 64 * 1024 * 1024:
                raise ValueError("Project HTML exceeds the 16 MiB/document or 64 MiB/report limit.")
            with path.open("rb") as stream:
                content = stream.read(16 * 1024 * 1024 + 1)
            # ManagedOutput's persisted file identity includes its basename and a delimiter.
            digest = hashlib.sha256(path.name.encode("utf-8") + b"\0" + content).hexdigest()
            if len(content) != artifact.get("size_bytes") or digest != artifact.get("sha256"):
                raise ValueError("HTML section artifact no longer matches its finalized checksum.")
            total += len(content)
            trial = run.get("trial")
            trial = trial.get("index") if isinstance(trial, Mapping) else trial
            label = (
                f"Trial {trial or '—'} · seed {run.get('seed', '—')} · {artifact.get('name', name)}"
            )
            sections.append(
                {"name": name, "title": title, "label": label, "html": content.decode("utf-8")}
            )
    return sections


def _encoded_page(value: Mapping[str, Any], fingerprint: str, offset: int) -> dict[str, Any]:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if not 0 <= offset <= len(encoded):
        raise ValueError("Invalid Study page offset.")
    chunk = encoded[offset : offset + 512 * 1024]
    return {
        "fingerprint": fingerprint,
        "offset": offset,
        "next_offset": offset + len(chunk),
        "eof": offset + len(chunk) == len(encoded),
        "data": base64.b64encode(chunk).decode("ascii"),
    }


def _file_page(
    path: Path, fingerprint: str, offset: int, before: os.stat_result, *, base_offset: int = 0
) -> dict[str, Any]:
    size = before.st_size - base_offset
    if not 0 <= offset <= size:
        raise ValueError("Invalid Study page offset.")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_ino, opened.st_mtime_ns, opened.st_size) != (
            before.st_ino,
            before.st_mtime_ns,
            before.st_size,
        ):
            return {"restart": True}
        stream.seek(offset + base_offset)
        chunk = stream.read(512 * 1024)
        after = os.fstat(stream.fileno())
    if (after.st_mtime_ns, after.st_size) != (before.st_mtime_ns, before.st_size):
        return {"restart": True}
    return {
        "fingerprint": fingerprint,
        "offset": offset,
        "next_offset": offset + len(chunk),
        "eof": offset + len(chunk) == size,
        "data": base64.b64encode(chunk).decode("ascii"),
    }
