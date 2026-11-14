"""Small, fail-closed recovery contract for an existing scientific execution."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lambdaforge.work.config import WorkConfig
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


def restore_seed_context(config: WorkConfig, manifest: Mapping[str, Any]) -> WorkConfig:
    """Re-resolve authored seeds in the validated original Execution's namespace.

    A new Job directory is not a new seed stream. Reparse rather than overwrite the design:
    changed seeds, replicate counts or policies must still fail the scientific identity checks.
    Inputs remain anchored to the newly staged YAML; no persisted scientific record is changed.
    """
    from lambdaforge.work.config import WorkConfig

    bindings = manifest.get("recovery_file_bindings", {})
    config = replace(config, recovery_file_bindings=bindings)

    if sequential_execution(manifest):
        levels = manifest["resolved_configuration"]["levels"]
        if len(config.levels) != len(levels) or any(
            len(level.runs) != 1 or level.runs[0].study_expected or level.runs[0].seeds != (None,)
            for level in config.levels
        ):
            raise ValueError("Sequential recovery requires the original unseeded step topology.")
        return replace(config, name=manifest["name"])
    if single_execution(manifest):
        if (
            "run" not in config.raw
            or len(config.levels) != 1
            or len(config.levels[0].runs) != 1
            or config.levels[0].runs[0].study_expected
            or config.levels[0].runs[0].seeds != (None,)
        ):
            raise ValueError("Single Work recovery requires the original unseeded single design.")
        # Labels are not scientific parameters. Staging historically replaced an implicit
        # authored filename with config.yaml. Normalize only the display namespace used by
        # the historical fingerprint; inputs still resolve against the selected source.
        level = config.levels[0]
        name = manifest["resolved_configuration"]["levels"][0][0]["name"]
        return replace(
            config,
            name=manifest["name"],
            levels=(replace(level, runs=(replace(level.runs[0], name=name),)),),
        )
    design = manifest["study_designs"][0]
    seed_source = design.get("seed_source")
    if not isinstance(seed_source, Mapping) or not isinstance(seed_source.get("namespace"), str):
        raise ValueError("Persisted Study seed context is missing or corrupt; recovery refused.")
    restored = WorkConfig.from_mapping(
        config.raw, source=config.source, seed_namespace=seed_source["namespace"]
    )
    current = restored.levels[0].runs[0].study_design
    if current is None or current.to_dict()["seed_source"] != dict(seed_source):
        raise ValueError(
            "Study recovery requires unchanged configuration, inputs and seed policy. "
            "The authored seeds/replicates or persisted seed coordinates differ."
        )
    return replace(restored, recovery_file_bindings=bindings)


def validate_recovery_configuration(config: WorkConfig, manifest: Mapping[str, Any]) -> str:
    """One semantic recovery check for previews and the actual native worker.

    Paths/comments/order are not identity. Exact typed input identities, seed coordinates,
    normalized design/objective/budgets and product contracts remain authoritative. The origin
    code identity deliberately holds code constant here; acknowledging code is a separate check.
    """
    from lambdaforge.work.runner import WorkRunner

    origin = manifest.get("code_identity")
    if not isinstance(origin, Mapping) or not manifest.get("scientific_fingerprint"):
        raise ValueError(
            "Original scientific configuration identity is missing; cannot verify recovery."
        )
    identity = WorkRunner._study_identity(config, code_identity=origin)
    if identity != manifest["scientific_fingerprint"]:
        raise ValueError(
            "Study recovery requires unchanged configuration, inputs and seed policy. "
            "The selected YAML changes scientific identity or its exact inputs cannot be matched."
        )
    if config.raw.get("products") != manifest.get("expected_products"):
        raise ValueError("Study recovery requires unchanged product publication contracts.")
    if sequential_execution(manifest):
        original_levels = manifest["resolved_configuration"]["levels"]
        if config.resources.to_dict() != manifest.get("resources") or any(
            level.runs[0].resources.to_dict() != original[0].get("resources")
            for level, original in zip(config.levels, original_levels, strict=True)
        ):
            raise ValueError(
                "Sequential recovery requires unchanged root and step resource requests."
            )
    if single_execution(manifest):
        original = manifest.get("resources")
        if not isinstance(original, Mapping):
            raise ValueError("Original single Work resource evidence is missing.")
        if any(
            original.get(key) != value
            for key, value in config.resources.to_dict().items()
            if key not in {"cpu_cores", "ram_bytes", "runtime_seconds"}
        ):
            raise ValueError(
                "Single Work recovery only supports CPU/RAM/wall-time resource changes."
            )
    return identity


def single_execution(manifest: Mapping[str, Any]) -> bool:
    """Recognize an authoritative native single Work, never infer one from missing HPO."""
    resolved = manifest.get("resolved_configuration", {})
    if not isinstance(resolved, Mapping):
        raise ValueError("Corrupt persisted resolved Work configuration.")
    levels = resolved.get("levels", [])
    if not isinstance(levels, list) or any(not isinstance(level, list) for level in levels):
        raise ValueError("Corrupt persisted Work levels.")
    if any(not isinstance(value, Mapping) for level in levels for value in level):
        raise ValueError("Corrupt persisted Work definition.")
    return bool(
        manifest.get("study_designs") == []
        and manifest.get("planned_runs") == 1
        and len(levels) == 1
        and len(levels[0]) == 1
        and levels[0][0].get("study_type") == "single"
        and levels[0][0].get("design") == {"type": "single", "seed_source": None}
        and levels[0][0].get("search") is None
    )


def sequential_execution(manifest: Mapping[str, Any]) -> bool:
    """Recognize only persisted sequential, single-Run steps; never guess missing HPO."""
    levels = manifest.get("resolved_configuration", {}).get("levels", [])
    return bool(
        isinstance(levels, list)
        and len(levels) > 1
        and manifest.get("study_designs") == []
        and manifest.get("planned_runs") == len(levels)
        and all(
            isinstance(level, list)
            and len(level) == 1
            and isinstance(level[0], Mapping)
            and level[0].get("study_type") == "single"
            and level[0].get("design") == {"type": "single", "seed_source": None}
            and level[0].get("search") is None
            for level in levels
        )
    )


def sequential_inventory(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect owned child Attempts without reading scientific or checkpoint payload bytes."""
    definitions = [level[0] for level in manifest["resolved_configuration"]["levels"]]
    authored = read_owned_json(path / "configuration.json").get("steps", [])
    if len(authored) != len(definitions) or any(
        not isinstance(step, Mapping)
        or not isinstance(step.get("run"), str)
        or step.get("name", f"step-{index}") != definition["name"]
        for index, (step, definition) in enumerate(
            zip(authored, definitions),  # noqa: B905 - old host Python
            1,
        )
    ):
        raise ValueError("Missing or corrupt authoritative sequential configuration.")
    definitions = [
        {**definition, "work_class": step["run"]}
        for step, definition in zip(authored, definitions)  # noqa: B905 - old host Python
    ]
    names = {value["name"]: value for value in definitions}
    if len(names) != len(definitions):
        raise ValueError("Ambiguous persisted step names.")
    latest: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    root = path / "runs"
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Missing or unsafe composed Run inventory.")
    for run in sorted(root.iterdir()):
        if not re.fullmatch(r"run-[0-9a-f]{20}", run.name) or run.resolve() != run:
            raise ValueError("Unsafe composed Run directory.")
        directory = run / "attempts"
        if directory.resolve() != directory or not directory.is_dir():
            raise ValueError("Missing or unsafe composed Attempt inventory.")
        for attempt in sorted(directory.iterdir()):
            if not re.fullmatch(r"attempt-\d{4,}", attempt.name) or attempt.resolve() != attempt:
                raise ValueError("Unsafe composed Attempt directory.")
            result = attempt / "result.json"
            value = read_owned_json(result if result.exists() else attempt / "request.json")
            number = int(attempt.name.removeprefix("attempt-"))
            if (
                value.get("execution_id") != path.name
                or value.get("run_id") != run.name
                or value.get("attempt_id") != attempt.name
                or value.get("attempt_number") != number
                or value.get("run_dir") != str(attempt)
            ):
                raise ValueError("Composed Attempt ownership mismatch.")
            name = value.get("name")
            if (
                not isinstance(name, str)
                or name not in names
                or value.get("work_class") != names[name]["work_class"]
            ):
                raise ValueError("Composed Attempt is outside the persisted step design.")
            if not result.exists():
                heartbeat = read_owned_json(attempt / "resource-heartbeat.json")
                if (
                    heartbeat.get("run_id") != run.name
                    or heartbeat.get("attempt_id") != attempt.name
                ):
                    raise ValueError("Interrupted composed Attempt ownership mismatch.")
                # An older first Attempt may not record a start time. Conservatively charge
                # from the Execution start, never pretend its interrupted cost was zero.
                elapsed = max(
                    0.0,
                    (
                        datetime.fromisoformat(heartbeat["updated_at_utc"])
                        - datetime.fromisoformat(
                            value.get("started_at_utc", manifest["created_at_utc"])
                        )
                    ).total_seconds(),
                )
                value = {**value, "status": "interrupted", "duration_seconds": elapsed}
            if value.get("status") not in {
                "succeeded",
                "failed",
                "interrupted",
                "timeout",
                "cancelled",
            }:
                raise ValueError("Invalid composed Attempt status.")
            for artifact in value.get("artifacts", []):
                if value["status"] != "succeeded":
                    continue
                target = attempt / artifact["path"]
                if (
                    not target.is_relative_to(attempt)
                    or target.resolve() != target
                    or not target.exists()
                    or not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", "")))
                ):
                    raise ValueError("Missing or unsafe persisted predecessor output.")
            previous = latest.get(name)
            if previous and previous["run_id"] != run.name:
                raise ValueError("Multiple logical Runs claim the same sequential step.")
            if previous is None or previous["attempt_number"] < number:
                latest[name] = value
            attempts.append(value)
    blocked = False
    rows = []
    for definition in definitions:
        name = definition["name"]
        value = latest.get(name, {})
        if blocked and value:
            raise ValueError("Downstream Attempt exists after an incomplete predecessor.")
        action = (
            "reuse" if value and value["status"] == "succeeded" else "retry" if value else "pending"
        )
        blocked |= action != "reuse"
        checkpoint = checkpoint_collection_preview(path, value["run_id"]) if value else None
        rows.append(
            {
                "step": name,
                "action": action,
                "run_id": value["run_id"] if value else None,
                "status": value["status"] if value else "pending",
                "next_attempt": value["attempt_number"] + 1 if action == "retry" else None,
                "checkpoint_collection": checkpoint,
                "checkpoint": checkpoint_recovery_preview(path, value["run_id"]) if value else None,
            }
        )
    return {"latest": latest, "attempts": attempts, "runs": rows}


def sequential_recovery_preview(path: Path, *, controller_lock: Any = None) -> dict[str, Any]:
    """Read-only step plan using the same owned inventory as the native worker."""
    manifest = validate_execution(path, controller_lock=controller_lock)
    inventory = sequential_inventory(path, manifest)
    rows = inventory["runs"]
    return {
        "resumable": True,
        "strategy": "sequential",
        "execution_dir": str(path),
        "execution_id": path.name,
        "runs": rows,
        "reuse_runs": sum(row["action"] == "reuse" for row in rows),
        "preserved_completed_runs": sum(row["action"] == "reuse" for row in rows),
        "retry_runs": sum(row["action"] == "retry" for row in rows),
        "pending_runs": sum(row["action"] == "pending" for row in rows),
        "spent_attempts": len(inventory["attempts"]),
        "spent_run_seconds": sum(
            float(a.get("duration_seconds", 0)) for a in inventory["attempts"]
        ),
        "checkpoint_policy": (
            "Original Run ownership; hashes and application compatibility checked on access."
        ),
        "output_validation": (
            "Owned artifact metadata checked; exact content checked by worker before reuse."
        ),
    }


def validate_execution(
    path: Path, *, seed_stream: Any = None, controller_lock: Any = None
) -> dict[str, Any]:
    """Validate an exact execution directory, never a guessed latest sibling."""
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError("Recovery requires an absolute, non-symlinked execution directory.")
    if (path / "import.json").exists() or (path / "import.json").is_symlink():
        raise ValueError(
            "Imported Study evidence is read-only; recover the original owned Execution."
        )
    manifest = read_owned_json(path / "execution.json")
    from lambdaforge.work.staged_input_identity import original_file_bindings

    manifest = {**manifest, "recovery_file_bindings": original_file_bindings(path)}
    if manifest.get("execution_id") != path.name or not path.name.startswith("execution-"):
        raise ValueError("Recovery execution identity does not match its owned directory.")
    if manifest.get("ownership", {}).get("execution_dir") != "owned":
        raise ValueError("Recovery execution has no framework ownership evidence.")
    history = path / "recovery-history.jsonl"
    if history.is_symlink() or history.resolve() != history:
        raise ValueError("Recovery history is outside its owned execution.")
    designs = manifest.get("study_designs", [])
    if single_execution(manifest) or sequential_execution(manifest):
        lock_path = path / ".controller.lock"
        if lock_path.exists() or lock_path.is_symlink():
            if (
                lock_path.is_symlink()
                or lock_path.resolve() != lock_path
                or not lock_path.is_file()
            ):
                raise ValueError("Unsafe recovered controller lock.")
            # The same native cross-platform lock, with creation/writes disabled.
            # Metadata transport probes inject that exact class, not a second protocol.
            if controller_lock is None:
                from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

                controller_lock = CrossProcessFileLock
            with controller_lock(
                lock_path,
                shared=False,
                timeout_seconds=0.1,
                poll_interval_seconds=0.01,
                create=False,
            ):
                pass
        if sequential_execution(manifest):
            inventory = sequential_inventory(path, manifest)
            for level, row in zip(  # noqa: B905 - old host Python
                manifest["resolved_configuration"]["levels"], inventory["runs"]
            ):
                policy = level[0].get("execution", {})
                attempts = [a for a in inventory["attempts"] if a["name"] == row["step"]]
                elapsed = sum(float(a.get("duration_seconds", 0)) for a in attempts)
                if row["action"] != "reuse" and (
                    (policy.get("max_runs") is not None and len(attempts) >= policy["max_runs"])
                    or (policy.get("max_time") is not None and elapsed >= policy["max_time"])
                ):
                    raise ValueError(
                        f"Recovery budget exhausted for step {row['step']}; costs are not refunded."
                    )
            return manifest
        inventory = fixed_inventory(path, manifest=manifest)
        if inventory["latest"][(1, None)]["status"] != "succeeded":
            policy = manifest["resolved_configuration"]["levels"][0][0].get("execution", {})
            max_runs, max_time = policy.get("max_runs"), policy.get("max_time")
            if (max_runs is not None and inventory["spent_runs"] >= int(max_runs)) or (
                max_time is not None and inventory["elapsed_seconds"] >= float(max_time)
            ):
                raise ValueError(
                    "Single Work recovery budget is exhausted; "
                    "previous Attempt costs are not refunded."
                )
        return manifest
    if len(designs) != 1 or not isinstance(designs[0], Mapping):
        raise ValueError(
            "Unsupported recovery topology: sequential single Works or one Study only; "
            "parallel/composed Studies cannot be recovered."
        )
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
    single = single_execution(manifest)
    design = {"type": "single"} if single else manifest["study_designs"][0]
    if design.get("type") not in {"single", "repeated", "sweep"}:
        raise ValueError("Fixed recovery requires a repeated or sweep design.")
    read_owned_json(path / "configuration.json")
    resolved = read_owned_json(path / "resolved-configuration.json")
    levels = resolved.get("levels", [])
    expected_design = {"type": "single", "seed_source": None} if single else design
    if (
        len(levels) != 1
        or len(levels[0]) != 1
        or levels[0][0].get("design") != expected_design
        or (single and resolved != manifest["resolved_configuration"])
    ):
        raise ValueError("Persisted Study design and resolved configuration disagree.")
    blocks = (
        read_owned_json(path / "hpo-control" / "sweep-blocks.json")
        if design.get("replication") == "auto-blocks"
        else None
    )
    requirements: list[dict[str, Any]] = (
        [{"candidate": 1, "seed": None, "required": True}]
        if single
        else fixed_requirements(design, blocks, seed_stream=seed_stream)
    )
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
    if single and (len(latest) != 1 or len({v["run_id"] for v in attempts}) != 1):
        raise ValueError("Single Work recovery requires exactly one authoritative logical Run.")
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


def fixed_recovery_preview(
    path: Path, *, seed_stream: Any = None, controller_lock: Any = None
) -> dict[str, Any]:
    """Compact metadata-only recovery plan, usable on older immutable worker hosts."""
    manifest = validate_execution(path, seed_stream=seed_stream, controller_lock=controller_lock)
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
                "status": result.get("status") if result else "pending",
                "next_attempt": int(result["attempt_number"]) + 1
                if result and result["status"] != "succeeded"
                else None,
                "checkpoint": checkpoint_recovery_preview(path, str(result["run_id"]))
                if result
                else None,
            }
        )
    if inventory["design"]["type"] == "single":
        for row in rows:
            row["checkpoint_collection"] = checkpoint_collection_preview(path, row["run_id"])
    return {
        "resumable": True,
        "strategy": inventory["design"]["type"],
        "work_status": latest[(1, None)]["status"]
        if inventory["design"]["type"] == "single"
        else None,
        "execution_dir": str(path),
        "execution_id": path.name,
        "reuse_runs": sum(row["action"] == "reuse" for row in rows),
        "preserved_completed_runs": sum(row["action"] == "reuse" for row in rows),
        "retry_runs": sum(row["action"] == "retry" for row in rows),
        "pending_runs": sum(row["action"] == "pending" for row in rows),
        "spent_attempts": inventory["spent_runs"],
        "spent_time_seconds": inventory["elapsed_seconds"],
        "runs": rows,
        "checkpoint_policy": (
            "Compatible application checkpoints may be restored; "
            "epoch continuation requires Work support."
        ),
        "missing_work": (
            "Work.run is re-entered with its original managed checkpoint collection; "
            "the Work validates/reuses complete producer files and computes missing producers. "
            "Producer counts cannot be inferred from opaque checkpoint keys."
            if inventory["design"]["type"] == "single"
            else None
        ),
    }


def checkpoint_collection_preview(execution: Path, run_id: str) -> dict[str, Any]:
    """Inspect all managed file receipts without hashing/downloading their payloads."""
    checkpoint_recovery_preview(execution, run_id)  # canonical root/last-descriptor checks
    root = execution / "runs" / run_id / "checkpoints"
    records = root / ".managed" / "records"
    count = total = invalid = 0
    if root.exists():
        for item in root.rglob("*"):
            if item.is_symlink() or item.resolve() != item:
                raise ValueError("Unsafe file in recovered checkpoint collection.")
    for path in records.glob("*.json"):
        value = read_owned_json(path)
        key = value.get("key")
        if (
            not isinstance(key, str)
            or not key
            or Path(key).is_absolute()
            or ".." in Path(key).parts
            or "\\" in key
            or path.name != hashlib.sha256(key.encode()).hexdigest() + ".json"
            or value.get("managed_file_version") != 1
            or value.get("scope") != "checkpoint"
        ):
            raise ValueError("Corrupt checkpoint collection registration.")
        owner = value.get("owner")
        if (
            not isinstance(owner, Mapping)
            or owner.get("execution_id") != execution.name
            or owner.get("run_id") != run_id
        ):
            raise ValueError("Checkpoint collection belongs to another Execution/Run.")
        size = value.get("size_bytes")
        if (
            type(size) is not int
            or size < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))
        ):
            raise ValueError("Corrupt checkpoint content receipt.")
        target = root / key
        if target.resolve() != target or not target.is_relative_to(root):
            raise ValueError("Checkpoint key escapes its owner.")
        count += 1
        total += size
        invalid += not target.is_file() or target.stat().st_size != size
    maps = root / "maps"
    map_counts = (
        {
            directory.name: sum(1 for _ in directory.glob("*.json"))
            for directory in maps.iterdir()
            if directory.is_dir()
        }
        if maps.is_dir()
        else {}
    )
    return {
        "root": str(root),
        "registered_files": count,
        "registered_bytes": total,
        "missing_or_size_invalid": invalid,
        "payload_hashes": "not_checked",
        "map_record_files": map_counts,
        "map_validation": "dependencies and application validators checked on restoration",
        "owner": {"execution_id": execution.name, "run_id": run_id},
        "restore_policy": (
            "managed hash and application validation on access; invalid files require rebuild"
        ),
    }


def checkpoint_recovery_preview(execution: Path, run_id: str) -> dict[str, Any]:
    """Inspect the last committed checkpoint descriptor, without reading payload bytes.

    Registration/size agreement is not hash verification or application compatibility. Old
    advanced path-written state remains visible as unregistered, never guessed as resumable.
    Kept stdlib-only because native recovery probes execute it on older immutable hosts.
    """
    if not re.fullmatch(r"run-[0-9a-f]{20}", run_id):
        raise ValueError("Unsafe recovery checkpoint Run identity.")
    root = execution / "runs" / run_id / "checkpoints"
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("Unsafe recovery checkpoint root.")
    result: dict[str, Any] = {
        "registered": 0,
        "verified": 0,
        "last": None,
        "continuation": "application_support_and_validation_required",
        "storage_durability": "unspecified",
        "unregistered_payloads_possible": False,
    }
    if not root.exists():
        return result
    if not root.is_dir():
        raise ValueError("Recovery checkpoint root is not a directory.")
    metadata = root / ".managed" / "records"
    if metadata.resolve() != metadata:
        raise ValueError("Unsafe recovery checkpoint metadata root.")
    result["registered"] = sum(1 for _ in metadata.glob("*.json"))
    result["unregistered_payloads_possible"] = (
        any(not item.name.startswith(".") for item in root.iterdir()) and result["registered"] == 0
    )
    policy_path = root / ".storage-policy.json"
    if not policy_path.exists() and not policy_path.is_symlink():
        return result
    policy = read_owned_json(policy_path)
    if policy.get("storage_policy_version") != 1:
        raise ValueError("Invalid checkpoint policy identity.")
    configured = policy.get("checkpoint_policy", {})
    if not isinstance(configured, Mapping):
        raise ValueError("Invalid checkpoint durability policy.")
    result["storage_durability"] = configured.get("storage_durability", "unspecified")
    last = policy.get("last_checkpoint")
    if last is None:
        return result
    if not isinstance(last, Mapping):
        raise ValueError("Invalid last checkpoint descriptor.")
    key = last.get("key")
    if not isinstance(key, str) or not key or Path(key).is_absolute() or ".." in Path(key).parts:
        raise ValueError("Unsafe checkpoint key in recovery descriptor.")
    target = root / key
    if target.resolve() != target or not target.is_relative_to(root):
        raise ValueError("Checkpoint descriptor escapes its owned Run.")
    record_path = metadata / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    record = read_owned_json(record_path)
    if record.get("scope") != "checkpoint" or record.get("key") != key:
        raise ValueError("Checkpoint descriptor conflicts with managed identity.")
    owner = record.get("owner")
    compatible_owner = (
        isinstance(owner, Mapping)
        and owner.get("execution_id") == execution.name
        and owner.get("run_id") == run_id
    )
    size_matches = target.is_file() and target.stat().st_size == record.get("size_bytes")
    result["last"] = {
        "key": key,
        "path": str(target),
        "owner": owner,
        "written_at_utc": record.get("written_at_utc"),
        "step": record.get("step"),
        "sha256": record.get("sha256"),
        "size_bytes": record.get("size_bytes"),
        "integrity": "not_checked" if size_matches else "invalid",
        "ownership": "recorded" if compatible_owner else "unverified",
        "compatibility": "application_validation_required",
    }
    return result


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
