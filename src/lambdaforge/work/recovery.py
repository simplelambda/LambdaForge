"""Small, fail-closed recovery contract for an existing scientific execution."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lambdaforge.work.models import WorkResult


def read_owned_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError(f"Recovery requires a regular persisted file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Corrupt recovery document: {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Invalid recovery document: {path}")
    return value


def validate_execution(path: Path, *, seed_stream: Any = None) -> dict[str, Any]:
    """Validate an exact execution directory, never a guessed latest sibling."""
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError("Recovery requires an absolute, non-symlinked execution directory.")
    if (path / "import.json").exists() or (path / "import.json").is_symlink():
        raise ValueError(
            "Imported Study evidence is read-only; recover the original owned Execution."
        )
    manifest = read_owned_json(path / "execution.json")
    if manifest.get("execution_id") != path.name or not path.name.startswith("execution-"):
        raise ValueError("Recovery execution identity does not match its owned directory.")
    if manifest.get("ownership", {}).get("execution_dir") != "owned":
        raise ValueError("Recovery execution has no framework ownership evidence.")
    history = path / "recovery-history.jsonl"
    if history.is_symlink() or history.resolve() != history:
        raise ValueError("Recovery history is outside its owned execution.")
    designs = manifest.get("study_designs", [])
    if len(designs) != 1 or not isinstance(designs[0], Mapping):
        raise ValueError("Recovery requires exactly one persisted Study design.")
    if designs[0].get("type") in {"repeated", "sweep"}:
        fixed_inventory(path, manifest=manifest, seed_stream=seed_stream)
        return manifest
    if designs[0].get("type") != "adaptive":
        raise ValueError("Unsupported persisted Study design for recovery.")
    state = read_owned_json(path / "hpo-control" / "state.json")
    if state.get("execution_id") != path.name or int(state.get("state_version", 0)) < 4:
        raise ValueError("Recovery requires a compatible persisted HPO controller state.")
    for run in state.get("runs", ()):
        if not isinstance(run, Mapping):
            raise ValueError("Recovery Run inventory is corrupt.")
        result_path = Path(str(run.get("result_path", "")))
        if not result_path.is_relative_to(path / "runs") or result_path.resolve() != result_path:
            raise ValueError("Recovery Run result is outside its owned execution.")
        result = read_owned_json(result_path)
        if result.get("execution_id") != path.name:
            raise ValueError("Recovery Run belongs to another execution.")
    return manifest


def fixed_requirements(
    design: Mapping[str, Any], blocks: Mapping[str, Any] | None = None, *, seed_stream: Any = None
) -> list[dict[str, Any]]:
    """Project authored and durably created blocks without rewriting scientific design."""
    requirements = design.get("evidence", {}).get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("Persisted fixed evidence requirements are missing or corrupt.")
    requirements = [dict(value) for value in requirements]
    # Paired sweeps may have committed additional shared-seed blocks. Restore the exact
    # persisted stream coordinates, never replace them with freshly selected seeds.
    if design.get("replication") == "auto-blocks":
        if seed_stream is None:
            from lambdaforge.reproducibility.SeedProvider import ProjectSeedStream

            seed_stream = ProjectSeedStream
        stream = seed_stream.from_mapping(design.get("seed_source") or {})
        from lambdaforge.work.sweep_blocks import sweep_block_inventory

        if blocks is None:
            raise ValueError("Persisted sweep block inventory is missing.")
        ordinals, _committed = sweep_block_inventory(blocks)
        candidates = sorted({int(value["candidate"]) for value in requirements})
        initial = {value.get("seed") for value in requirements}
        resolved_seeds = design.get("seed_source", {}).get("resolved", ())
        if not isinstance(resolved_seeds, list | tuple):
            raise ValueError("Initial sweep seed acquisition coordinates are missing or corrupt.")
        for requirement in requirements:
            matching = [
                dict(value)
                for value in resolved_seeds
                if isinstance(value, Mapping) and value.get("value") == requirement.get("seed")
            ]
            if len(matching) != 1 or (
                requirement.get("seed_metadata") is not None
                and requirement["seed_metadata"] != matching[0]
            ):
                raise ValueError(
                    "Initial sweep seed order is missing or conflicts with its design."
                )
            # This is a read projection of the authored resolved coordinate, not a rewrite.
            requirement["seed_metadata"] = matching[0]
        for ordinal in ordinals:
            identity = stream.at(ordinal)
            if identity.value in initial:
                continue
            requirements.extend(
                {
                    "candidate": trial,
                    "seed": identity.value,
                    "seed_metadata": identity.to_dict(),
                    "key": f"candidate-{trial}:seed-{identity.value}:search:fidelity-none",
                    "phase": "search",
                    "required": True,
                    "kind": "SHARED_SEED",
                    "fidelity": None,
                }
                for trial in candidates
            )
    required = {(int(value["candidate"]), value.get("seed")) for value in requirements}
    if len(required) != len(requirements):
        raise ValueError("Persisted fixed evidence identities are duplicated.")
    if design.get("replication") == "auto-blocks":
        for requirement in requirements:
            metadata = requirement.get("seed_metadata")
            seed_ordinal = metadata.get("ordinal") if isinstance(metadata, Mapping) else None
            if (
                not isinstance(seed_ordinal, int)
                or isinstance(seed_ordinal, bool)
                or seed_ordinal < 0
                or metadata != stream.at(seed_ordinal).to_dict()
                or requirement.get("seed") != stream.at(seed_ordinal).value
            ):
                raise ValueError("Persisted sweep seed acquisition metadata is missing or corrupt.")
    return requirements


def fixed_inventory(
    path: Path, *, manifest: Mapping[str, Any] | None = None, seed_stream: Any = None
) -> dict[str, Any]:
    """Read only owned design/Attempt metadata, never artifact or checkpoint bytes.

    Terminal execution results are a reference index, not the authority for latest outcomes:
    a controller can die after a worker publishes its next Attempt but before aggregation.
    """
    manifest = manifest or read_owned_json(path / "execution.json")
    design = manifest["study_designs"][0]
    if design.get("type") not in {"repeated", "sweep"}:
        raise ValueError("Fixed recovery requires a repeated or sweep design.")
    read_owned_json(path / "configuration.json")
    resolved = read_owned_json(path / "resolved-configuration.json")
    levels = resolved.get("levels", [])
    if len(levels) != 1 or len(levels[0]) != 1 or levels[0][0].get("design") != design:
        raise ValueError("Persisted Study design and resolved configuration disagree.")
    blocks = (
        read_owned_json(path / "hpo-control" / "sweep-blocks.json")
        if design.get("replication") == "auto-blocks"
        else None
    )
    requirements = fixed_requirements(design, blocks, seed_stream=seed_stream)
    required = {(int(value["candidate"]), value.get("seed")) for value in requirements}
    requirement_metadata = {
        (int(value["candidate"]), value.get("seed")): value.get("seed_metadata")
        for value in requirements
    }
    attempts: list[dict[str, Any]] = []
    latest: dict[tuple[int, int | None], dict[str, Any]] = {}
    start = datetime.fromisoformat(str(manifest["created_at_utc"]))
    finish = start
    # Older workers published per-cell checkpoint ownership before entering user code,
    # but did not yet write request.json. Recover that exact cell, never guess a seed.
    interrupted_cells: dict[str, tuple[int, int | None]] = {}
    for record in (path / "hpo-control").glob("trial-*-seed-*.checkpoint.json"):
        match = re.fullmatch(r"trial-(\d+)-seed-(none|-?\d+)\.checkpoint\.json", record.name)
        if match is None:
            raise ValueError("Invalid fixed checkpoint ownership record.")
        checkpoint = read_owned_json(record)
        directory = Path(str(checkpoint.get("run_dir", "")))
        if not directory.is_relative_to(path / "runs") or directory.resolve() != directory:
            raise ValueError("Recovery checkpoint ownership is outside the execution.")
        interrupted_cells[str(directory)] = (
            int(match[1]),
            None if match[2] == "none" else int(match[2]),
        )
    runs_root = path / "runs"
    if runs_root.is_symlink() or runs_root.resolve() != runs_root or not runs_root.is_dir():
        raise ValueError("Recovery requires a regular owned Run inventory.")
    for run_root in sorted(runs_root.iterdir()):
        if not re.fullmatch(r"run-[0-9a-f]{20}", run_root.name) or run_root.is_symlink():
            raise ValueError(f"Unsafe recovery Run directory: {run_root}")
        attempt_root = run_root / "attempts"
        if not attempt_root.is_dir() or attempt_root.resolve() != attempt_root:
            raise ValueError(f"Recovery Attempt inventory is missing or unsafe: {attempt_root}")
        for directory in sorted(
            attempt_root.iterdir(),
            key=lambda item: int(item.name[8:]) if item.name[8:].isdigit() else -1,
        ):
            if (
                not re.fullmatch(r"attempt-\d{4,}", directory.name)
                or directory.resolve() != directory
            ):
                raise ValueError(f"Unsafe recovery Attempt directory: {directory}")
            number = int(directory.name.removeprefix("attempt-"))
            result_path = directory / "result.json"
            if result_path.exists() or result_path.is_symlink():
                result = read_owned_json(result_path)
                if (
                    result.get("execution_id") != path.name
                    or result.get("run_id") != run_root.name
                    or result.get("attempt_number") != number
                    or result.get("attempt_id") != directory.name
                    or result.get("run_dir") != str(directory)
                    or result.get("status")
                    not in {"succeeded", "failed", "cancelled", "timeout", "interrupted"}
                ):
                    raise ValueError(f"Recovery Attempt identity/status mismatch: {result_path}")
                trial = result.get("trial") or {"index": 1}
                key = (int(trial["index"]), result.get("seed"))
                if key not in required:
                    raise ValueError("Recovery Run is not part of the persisted evidence design.")
                if design.get("replication") == "auto-blocks":
                    if result.get("seed_metadata") != requirement_metadata[key]:
                        raise ValueError("Recovery Run conflicts with its sweep seed ordinal.")
                previous = latest.get(key)
                if previous and previous["run_id"] != result["run_id"]:
                    raise ValueError("Multiple Run identities claim the same fixed evidence cell.")
                latest[key] = result
                finish = max(finish, datetime.fromisoformat(result["finished_at_utc"]))
                attempts.append(result)
            else:
                heartbeat = read_owned_json(directory / "resource-heartbeat.json")
                if (
                    heartbeat.get("run_id") != run_root.name
                    or heartbeat.get("attempt_id") != directory.name
                ):
                    raise ValueError("Interrupted Attempt ownership evidence is corrupt.")
                finish = max(finish, datetime.fromisoformat(heartbeat["updated_at_utc"]))
                request_path = directory / "request.json"
                request = (
                    read_owned_json(request_path)
                    if request_path.exists() or request_path.is_symlink()
                    else {}
                )
                if request and (
                    request.get("execution_id") != path.name
                    or request.get("run_id") != run_root.name
                    or request.get("attempt_number") != number
                    or request.get("attempt_id") != directory.name
                    or request.get("run_dir") != str(directory)
                ):
                    raise ValueError("Interrupted Attempt request identity is corrupt.")
                interrupted_key = (
                    (int(request["trial"]["index"]), request.get("seed"))
                    if request
                    else interrupted_cells.get(str(directory))
                )
                interrupted = {
                    **request,
                    "run_id": run_root.name,
                    "attempt_number": number,
                    "status": "interrupted",
                    "run_dir": str(directory),
                }
                attempts.append(interrupted)
                if interrupted_key is not None:
                    if interrupted_key not in required:
                        raise ValueError("Interrupted Run is outside the fixed evidence design.")
                    previous = latest.get(interrupted_key)
                    if previous and previous["run_id"] != run_root.name:
                        raise ValueError("Multiple Run identities claim a fixed evidence cell.")
                    latest[interrupted_key] = {**(previous or {}), **interrupted}
                elif not any(value["run_id"] == run_root.name for value in latest.values()):
                    raise ValueError(
                        "Interrupted Run has no scientific request or checkpoint ownership "
                        "record; cannot identify its seed safely."
                    )
    aggregate = path / "result.json"
    if aggregate.exists() or aggregate.is_symlink():
        result = read_owned_json(aggregate)
        if result.get("execution_id") != path.name:
            raise ValueError("Recovery aggregate belongs to another execution.")
        for reference in result.get("runs", []):
            directory = Path(str(reference["run_dir"]))
            if not directory.is_relative_to(runs_root) or directory.resolve() != directory:
                raise ValueError("Recovery Run result is outside its owned execution.")
            read_owned_json(directory / "result.json")
    # A later interrupted Attempt supersedes an earlier failure, never silently hides it.
    interrupted = {value["run_id"]: value for value in attempts if value["status"] == "interrupted"}
    for key, value in tuple(latest.items()):
        tail = interrupted.get(value["run_id"])
        if tail and tail["attempt_number"] > value["attempt_number"]:
            latest[key] = {**value, **tail}
    elapsed = max(0.0, (finish - start).total_seconds())
    budget_path = path / "hpo-control" / "fixed-recovery-state.json"
    if budget_path.exists() or budget_path.is_symlink():
        budget = read_owned_json(budget_path)
        if budget.get("execution_id") != path.name or budget.get("state_version") != 1:
            raise ValueError("Persisted fixed Study budget identity is corrupt.")
        segment = datetime.fromisoformat(budget["segment_started_at_utc"])
        elapsed = max(
            float(budget["elapsed_seconds"]),
            float(budget["prior_elapsed_seconds"]) + max(0.0, (finish - segment).total_seconds()),
        )
        if any(
            not math.isfinite(float(budget[field])) or float(budget[field]) < 0
            for field in ("elapsed_seconds", "prior_elapsed_seconds")
        ):
            raise ValueError("Persisted fixed Study spent time is corrupt.")
    return {
        "design": design,
        "requirements": requirements,
        "attempts": attempts,
        "latest": latest,
        "spent_runs": len(attempts),
        "elapsed_seconds": elapsed,
        "sweep_blocks": blocks if design.get("replication") == "auto-blocks" else None,
    }


def fixed_recovery_preview(path: Path, *, seed_stream: Any = None) -> dict[str, Any]:
    """Compact metadata-only recovery plan, usable on older immutable worker hosts."""
    manifest = validate_execution(path, seed_stream=seed_stream)
    inventory = fixed_inventory(path, manifest=manifest, seed_stream=seed_stream)
    latest = inventory["latest"]
    rows: list[dict[str, Any]] = []
    for requirement in inventory["requirements"]:
        key = (int(requirement["candidate"]), requirement.get("seed"))
        result = latest.get(key)
        rows.append(
            {
                "trial": key[0],
                "seed": key[1],
                "action": "reuse"
                if result and result["status"] == "succeeded"
                else "retry"
                if result
                else "pending",
                "run_id": result.get("run_id") if result else None,
                "attempt": result.get("attempt_number") if result else None,
            }
        )
    return {
        "resumable": True,
        "strategy": inventory["design"]["type"],
        "execution_dir": str(path),
        "execution_id": path.name,
        "reuse_runs": sum(row["action"] == "reuse" for row in rows),
        "retry_runs": sum(row["action"] == "retry" for row in rows),
        "pending_runs": sum(row["action"] == "pending" for row in rows),
        "spent_attempts": inventory["spent_runs"],
        "spent_time_seconds": inventory["elapsed_seconds"],
        "runs": rows,
        "checkpoint_policy": (
            "Compatible application checkpoints may be restored; "
            "epoch continuation requires Work support."
        ),
    }


def latest_outcomes(outcomes: Sequence[WorkResult]) -> tuple[WorkResult, ...]:
    """Resolve logical Run state while retaining physical Attempts in controller history."""
    latest: dict[tuple[Any, ...], WorkResult] = {}
    for result in outcomes:
        key = (
            result.name,
            int((result.trial or {"index": 0})["index"]),
            result.seed,
            result.study_phase,
            int((result.fidelity or {}).get("target", 0)),
        )
        previous = latest.get(key)
        if previous is None or result.attempt_number > previous.attempt_number:
            latest[key] = result
    return tuple(latest.values())
