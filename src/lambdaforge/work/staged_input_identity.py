"""Versioned role-bound reconstruction of historical bounded-file identities.

Only original owned bundle receipts may choose a historical filename. New bytes must still
match the original raw digest; directory membership retains its ordinary identity semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RECOVERY_BINDING_CONTRACT = "bounded-role-chain-v2"


def file_digests(path: Path, logical_name: str) -> tuple[str, str, int]:
    """Stream bytes once for raw transport and historical filename-bound identity."""
    if path.resolve() != path or not path.is_file() or path.is_symlink():
        raise ValueError(f"Bounded input must be a regular non-symlink file: {path}")
    if Path(logical_name).name != logical_name or logical_name in {"", ".", ".."}:
        raise ValueError("Unsafe historical input basename.")
    raw = hashlib.sha256()
    legacy = hashlib.sha256(logical_name.encode() + b"\0")
    before = path.stat()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            raw.update(block)
            legacy.update(block)
            size += len(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Bounded input changed during verification.")
    return "sha256:" + raw.hexdigest(), "sha256:" + legacy.hexdigest(), size


def bound_file_identity(path: Path, binding: Mapping[str, Any], role: str) -> tuple[str, int]:
    """Verify bytes, then reconstruct an immutable v1 identity, never replace it."""
    if binding.get("binding_version") != "bounded-role-v1":
        raise ValueError("Unsupported bounded input recovery binding version.")
    logical = str(binding.get("logical_name", ""))
    raw, legacy, size = file_digests(path, logical)
    if (
        raw != binding.get("raw_sha256")
        or size != binding.get("size_bytes")
        or legacy != binding.get("historical_sha256")
    ):
        raise ValueError(
            f"Recovery input role {role!r}, logical name {logical!r}: exact bytes differ; "
            f"expected raw {binding.get('raw_sha256')} ({binding.get('size_bytes')} bytes), "
            f"observed raw {raw} ({size} bytes)."
        )
    return legacy, size


def _definition_parameters(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    selected = (
        config
        if "run" in config
        else next((step for step in config.get("steps", []) if step.get("name") == name), {})
    )
    return selected.get("with", {})


def _role_value(parameters: Mapping[str, Any], role: str) -> Any:
    value: Any = parameters
    for component in role.split("."):
        value = value.get(component, {}) if isinstance(value, Mapping) else {}
    return value


def _later_attempt_bindings(
    execution: Path,
    source: Path,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    expected: Mapping[str, Any],
    origin_result: Mapping[str, Any],
) -> None:
    """Check a later native staging owner and durable recovery lineage, never origin paths."""
    from lambdaforge.work.recovery import read_owned_json

    owner = result.get("job_id")
    origin = manifest.get("job_id")
    if not isinstance(owner, str) or not re.fullmatch(r"job-[a-zA-Z0-9-]+", owner):
        raise ValueError("Later Attempt has no valid recorded Job owner.")
    work = source.parent
    if owner != origin:
        if work.name != "work" or work.parent.name != origin:
            raise ValueError("Cannot establish the original Job workspace for recovery lineage.")
        job = work.parent.parent / owner
        origin_request = read_owned_json(work.parent / "request.json")
        request = read_owned_json(job / "request.json")
        work = job / "work"
        command = request.get("command", [])
        if (
            request.get("job_id") != owner
            or request.get("work_dir") != str(work)
            or not isinstance(command, list)
            or command.count("--resume-execution") != 1
            or str(work / "config.yaml") not in command
        ):
            raise ValueError("Later Job request does not establish owned recovery staging.")
        if origin_request.get("job_id") != origin or request.get("cluster") != origin_request.get(
            "cluster"
        ):
            raise ValueError(
                "Later recovery Job does not share the authoritative origin placement."
            )
        position = command.index("--resume-execution")
        if position + 1 >= len(command) or command[position + 1] != str(execution):
            raise ValueError("Later Job request targets a different recovery Execution.")
        history = execution / "recovery-history.jsonl"
        if history.resolve() != history or not history.is_file():
            raise ValueError("Recovery lineage audit is missing or unsafe.")
        revisions = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        if not any(
            isinstance(revision, dict)
            and revision.get("job_id") == owner
            and revision.get("compatible_evidence_reuse") is True
            for revision in revisions
        ):
            raise ValueError("Later Job has no accepted native recovery audit for this Execution.")
    bundle = read_owned_json(work / "manifest.json")
    if bundle.get("bundle_version") not in {3, 4}:
        raise ValueError("Unsupported later bounded bundle receipt version.")
    parameters = _definition_parameters(bundle.get("config", {}), str(result["name"]))
    staged = dict(bundle.get("staged", []))
    inputs = result.get("inputs", [])
    original_inputs = origin_result.get("inputs", [])
    if (
        not isinstance(inputs, list)
        or any(not isinstance(value, dict) for value in inputs)
        or len({value.get("name") for value in inputs}) != len(inputs)
        or {value.get("name") for value in inputs}
        != {value.get("name") for value in original_inputs}
    ):
        raise ValueError("Later Attempt has missing, extra or duplicated scientific input roles.")
    identity_parameters = json.loads(json.dumps(result.get("parameters", {})))
    for role, binding in expected.items():
        try:
            matches = [value for value in inputs if value.get("name") == role]
            if len(matches) != 1:
                raise ValueError("Missing or duplicate later Attempt input role.")
            value = matches[0]
            logical = value.get("logical_source")
            if (
                value.get("kind") != "file"
                or not isinstance(logical, str)
                or logical not in staged
                or _role_value(parameters, role) != {"file": logical}
            ):
                raise ValueError("Later role conflicts with its own staged bundle/configuration.")
            path = work / logical
            if (
                Path(logical).is_absolute()
                or ".." in Path(logical).parts
                or path.resolve() != path
                or not path.is_relative_to(work)
                or value.get("path") != str(path)
            ):
                raise ValueError(
                    "Later input path escapes or disagrees with its own staging owner."
                )
            raw, local_digest, size = file_digests(path, path.name)
            if raw != staged[logical]:
                raise ValueError(
                    f"Later bundle raw digest mismatch: {raw}; expected {staged[logical]}."
                )
            bound_file_identity(path, binding, role)
            if (
                value.get("sha256") not in {binding["historical_sha256"], local_digest}
                or value.get("size_bytes") != size
            ):
                raise ValueError("Later input historical fingerprint/size receipt is incompatible.")
            recorded = _role_value(identity_parameters, role)
            if (
                not isinstance(recorded, dict)
                or set(recorded) != {"file"}
                or not isinstance(recorded["file"], dict)
                or recorded["file"].get("sha256")
                not in {binding["historical_sha256"], local_digest}
                or recorded["file"].get("size_bytes") != size
            ):
                raise ValueError("Later role parameters conflict with their exact input evidence.")
            recorded["file"] = {"sha256": binding["historical_sha256"], "size_bytes": size}
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise ValueError(
                f"Attempt {result.get('attempt_id')}, role {role!r}, origin owner {origin!r}, "
                f"actual owner {owner!r}: {error} Inspect that Job's request/bundle and recovery "
                "audit; do not change scientific YAML or delete Attempts to bypass ownership."
            ) from error
    if identity_parameters != origin_result.get("parameters", {}):
        raise ValueError(
            "Later Attempt scientific parameters or non-bounded inputs differ from origin."
        )


def original_file_bindings(execution: Path) -> dict[str, dict[str, Any]]:
    """Read original role/config/bundle/Attempt bindings and verify bounded source bytes.

    No prefix guessing or basename matching across roles. Unstaged executions return no
    compatibility bindings. Stdlib-only so previews can ship this authority to older hosts.
    """
    from lambdaforge.work.recovery import read_owned_json

    manifest = read_owned_json(execution / "execution.json")
    source = Path(str(manifest.get("source", "")))
    bundle_path = source.parent / "manifest.json"
    if not bundle_path.exists():
        return {}
    if not source.is_absolute() or not execution.is_relative_to(
        source.parent / ".lambdaforge/runs"
    ):
        raise ValueError("Original recovery source is outside the owned staged workspace.")
    bundle = read_owned_json(bundle_path)
    if bundle.get("bundle_version") not in {3, 4}:
        raise ValueError("Unsupported original bounded bundle receipt version.")
    config = read_owned_json(execution / "configuration.json")
    staged = dict(bundle.get("staged", []))
    bindings: dict[str, dict[str, Any]] = {}
    origins: dict[str, Mapping[str, Any]] = {}
    later: list[Mapping[str, Any]] = []
    for result_path in sorted((execution / "runs").glob("*/attempts/*/result.json")):
        result = read_owned_json(result_path)
        attempt = result_path.parent
        run = attempt.parent.parent
        if (
            result.get("execution_id") != execution.name
            or result.get("run_id") != run.name
            or result.get("attempt_id") != attempt.name
            or result.get("run_dir") != str(attempt)
            or result.get("attempt_number") != int(attempt.name.removeprefix("attempt-"))
        ):
            raise ValueError(f"Foreign/corrupt Attempt ownership in input bindings: {attempt}.")
        if result["attempt_number"] != 1:
            later.append(result)
            continue
        if result.get("job_id") != manifest.get("job_id"):
            raise ValueError(f"Origin Attempt {attempt.name} does not belong to origin Job.")
        origins[run.name] = result
        definition = result["name"]
        authored = (
            config
            if "run" in config
            else next(
                (step for step in config.get("steps", []) if step.get("name") == definition), {}
            )
        )
        parameters = authored.get("with", {})
        bundled = bundle.get("config", {})
        bundled = (
            bundled
            if "run" in bundled
            else next(
                (step for step in bundled.get("steps", []) if step.get("name") == definition), {}
            )
        )
        if bundled.get("with", {}) != parameters:
            raise ValueError("Original bounded bundle/configuration role bindings disagree.")
        bound: dict[str, Any] = {}
        for input_record in result.get("inputs", []):
            role = input_record["name"]
            logical_source = input_record.get("logical_source")
            configured_role = _role_value(parameters, role)
            if (
                isinstance(configured_role, Mapping)
                and set(configured_role) == {"file"}
                and configured_role["file"] in staged
                and (
                    input_record.get("kind") != "file" or logical_source != configured_role["file"]
                )
            ):
                raise ValueError(
                    f"Origin Attempt {result['attempt_id']}, role {role!r}: "
                    "input receipt disagrees with its authoritative bundle role."
                )
            if input_record.get("kind") != "file" or logical_source not in staged:
                continue
            configured: Any = parameters
            # Native input names use dotted paths; unsupported ambiguous schemas fail closed.
            for component in role.split("."):
                configured = (
                    configured.get(component, {}) if isinstance(configured, Mapping) else {}
                )
            if configured != {"file": logical_source}:
                raise ValueError(
                    f"Original bounded input role {role!r} conflicts with configuration."
                )
            original = source.parent / str(logical_source)
            if not original.is_relative_to(source.parent) or original.resolve() != original:
                raise ValueError("Original bounded input escaped its bundle.")
            if not original.is_file():
                if original.is_dir():
                    continue  # Directory relative member names remain scientific content.
                raise ValueError(f"Original bounded input for role {role!r} is missing.")
            if input_record.get("path") != str(original):
                raise ValueError(f"Original bounded input role {role!r} has mismatched ownership.")
            raw, legacy, size = file_digests(original, original.name)
            if (
                raw != staged[logical_source]
                or legacy != input_record.get("sha256")
                or size != input_record.get("size_bytes")
            ):
                raise ValueError(
                    f"Corrupt original bounded input role {role!r}, logical {original.name!r}: "
                    f"raw {raw}; bundle expects {staged[logical_source]}."
                )
            bound[role] = {
                "binding_version": "bounded-role-v1",
                "logical_name": original.name,
                "raw_sha256": raw,
                "historical_sha256": legacy,
                "size_bytes": size,
            }
        if definition in bindings and bindings[definition] != bound:
            raise ValueError("Conflicting authoritative origin bindings for one Work definition.")
        bindings[definition] = bound
    for later_result in later:
        origin = origins.get(str(later_result["run_id"]))
        try:
            if origin is None or any(
                later_result.get(key) != origin.get(key) for key in ("name", "work_class", "seed")
            ):
                raise ValueError("Missing origin Attempt or incompatible later Run definition.")
            _later_attempt_bindings(
                execution, source, manifest, later_result, bindings[str(origin["name"])], origin
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise ValueError(
                f"Recovery binding rejected for Attempt {later_result.get('attempt_id')}; "
                f"origin owner {manifest.get('job_id')!r}, "
                f"actual owner {later_result.get('job_id')!r}: "
                f"{error} Retain evidence; inspect the named owner's bundle/request and audit."
            ) from error
    return bindings
