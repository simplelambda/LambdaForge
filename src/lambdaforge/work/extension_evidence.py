"""Owned, metadata-only evidence inspection shared by local and remote extensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_extension_configuration(directory: Path) -> dict[str, Any]:
    """Preserve frozen authoring order without reading the researcher's edited YAML.

    Atomic JSON sorts mapping keys, but authored dimension order determines native Trial
    ordinals. Use the Execution-owned snapshot only if its exact JSON values equal the
    frozen configuration. The caller must still verify the original scientific fingerprint
    and authoritative Attempts. Missing order evidence never waives those checks.
    """
    import json
    from collections.abc import Mapping

    import yaml

    from lambdaforge.work.recovery import read_owned_json

    raw = read_owned_json(directory / "configuration.json")
    snapshot = directory / "authored.yaml"
    if not snapshot.exists() and not snapshot.is_symlink():
        return raw
    if (
        snapshot.is_symlink() or snapshot.resolve() != snapshot
        or not snapshot.is_file() or snapshot.stat().st_size > 1024 * 1024
    ):
        raise ValueError("Original Study authoring snapshot is unsafe or oversized.")
    try:
        authored = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if not isinstance(authored, Mapping) or json.dumps(
            authored, sort_keys=True, allow_nan=False
        ) != json.dumps(raw, sort_keys=True, allow_nan=False):
            raise ValueError("Original Study authoring snapshot differs from frozen configuration.")
    except (yaml.YAMLError, UnicodeError, TypeError) as error:
        raise ValueError("Original Study authoring snapshot is corrupt.") from error
    return dict(authored)


def inspect_extension_execution(directory: str, job_id: str | None = None) -> dict[str, Any]:
    """Validate native fixed evidence on its host, without copying artifact bytes.

    This function is also shipped as source to an existing scientific interpreter. Keep
    imports inside it: historical environments need not contain the extension service.
    """
    import hashlib
    import json
    from collections.abc import Mapping

    from lambdaforge.analysis.Evidence import normalize_evidence
    from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
    from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
    from lambdaforge.work.config import WorkConfig
    from lambdaforge.work.recovery import (
        fixed_inventory,
        read_owned_json,
        restore_seed_context,
        validate_execution,
        validate_recovery_configuration,
    )
    from lambdaforge.work.runner import _identity_values, _resolve_inputs, _validate_fixed_recovery

    root = Path(directory)
    with CrossProcessFileLock(
        root / ".controller.lock",
        shared=True,
        create=False,
        timeout_seconds=0.1,
        poll_interval_seconds=0.01,
    ):
        manifest = validate_execution(root)
        if manifest.get("job_id") != job_id:
            raise ValueError("Execution does not belong to its recorded native Job.")
        source = Path(str(manifest["source"]))
        raw = read_extension_configuration(root)
        config = restore_seed_context(WorkConfig.from_mapping(raw, source=source), manifest)
        if len(config.levels) != 1 or len(config.levels[0].runs) != 1:
            raise ValueError("Extension requires one native fixed Work.")
        definition = config.levels[0].runs[0]
        if definition.objective is None:
            raise ValueError("Extension requires an explicit scientific objective.")
        design = definition.study_design
        if design is None or design.kind != "sweep" or design.replication != "fixed":
            raise ValueError("Extension requires a fixed explicit-seed sweep.")
        validate_recovery_configuration(config, manifest)
        _validate_fixed_recovery(definition, source, root, manifest)
        inventory = fixed_inventory(root, manifest=manifest)
        aggregate = read_owned_json(root / "result.json")
        if aggregate.get("status") != "succeeded":
            raise ValueError("Complete required evidence with native retry before extending.")
        attested = {(run["run_id"], run["attempt_id"]): run for run in aggregate["runs"]}
        expected = {}
        for trial, variant in enumerate(definition.variants, 1):
            _, inputs, normalized = _resolve_inputs(
                {**definition.parameters, **variant}, source.parent
            )
            for seed in definition.seeds:
                expected[trial, seed] = ScientificIdentity.from_payload(
                    {
                        "identity_version": 1,
                        "work_class": definition.work_class,
                        "code": manifest["code_identity"],
                        "parameters": normalized,
                        "inputs": [value.identity_dict() for value in inputs.values()],
                        "seed": seed,
                        "trial_parameters": dict(variant),
                    }
                ).digest
        evidence = []
        environments = []
        for key, run in inventory["latest"].items():
            if (
                run.get("status") != "succeeded"
                or run.get("pruned")
                or run.get("scientific_fingerprint") != expected.get(key)
            ):
                raise ValueError(
                    "Required historical Run is incomplete or incompatible; use native retry."
                )
            saved = attested.get((run["run_id"], run["attempt_id"]))
            fields = (
                "status",
                "scientific_fingerprint",
                "metrics",
                "parameters",
                "inputs",
                "seed_metadata",
                "objective_observation",
            )
            if not isinstance(saved, Mapping) or any(saved.get(f) != run.get(f) for f in fields):
                raise ValueError("Aggregate and authoritative Attempt evidence disagree.")
            _, observations = normalize_evidence({"runs": [run]}, definition.objective)
            if not observations or observations[0].get("final_objective") is None:
                raise ValueError("Historical Run lacks the declared final objective evidence.")
            result = Path(run["run_dir"]) / "result.json"
            relative = Path(str(run.get("environment_manifest", "environment.json")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe historical environment reference.")
            env_path = Path(run["run_dir"]) / relative
            if env_path.stat().st_size > 256 * 1024:
                raise ValueError("Environment metadata is oversized.")
            environments.append(read_owned_json(env_path))
            # Only scalars and provenance are transferred; not histories, artifact manifests,
            # result payloads, models or checkpoints. Original Attempt remains authoritative.
            evidence.append(
                {
                    **{f: run.get(f) for f in fields},
                    "execution_id": run["execution_id"],
                    "run_id": run["run_id"],
                    "attempt_id": run["attempt_id"],
                    "seed": run["seed"],
                    "trial": run["trial"],
                    "duration_seconds": run.get("duration_seconds"),
                    "result_path": str(result),
                    "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                }
            )
        if len(evidence) != len(expected):
            raise ValueError("Required fixed Study evidence is incomplete; use native retry.")
        # Serializing here also rejects non-finite/corrupt metadata before paging it.
        value = {
            "manifest": manifest,
            "configuration": raw,
            # Wire paging canonicalizes JSON keys too. This string preserves authored order
            # across the metadata transport; its values still match configuration exactly.
            "configuration_ordered_json": json.dumps(raw, allow_nan=False),
            "analysis_semantics": read_owned_json(root / "analysis-semantics.json").get(
                definition.name, {}
            ),
            "execution_dir": str(root),
            "manifest_sha256": hashlib.sha256((root / "execution.json").read_bytes()).hexdigest(),
            "fixed_identities": _identity_values(definition.parameters, source.parent),
            "evidence": evidence,
            "environments": environments,
            "spent_runs": inventory["spent_runs"],
            "elapsed_seconds": inventory["elapsed_seconds"],
        }
        value["environments"] = [
            json.loads(item)
            for item in sorted(
                {
                    json.dumps(
                        {
                            key: env.get(key)
                            for key in (
                                "lambdaforge_version",
                                "python",
                                "platform",
                                "packages",
                                "torch",
                                "plugins",
                                "external_tools",
                            )
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    for env in environments
                }
            )
        ]
        return json.loads(json.dumps(value, allow_nan=False))
