"""Native post-Study publication: sealed evidence first, durable products before compaction.

Publication has its own recoverable operational record. It never launches Runs, reranks HPO,
alters scientific results or turns a failed publication into a failed training observation.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.products.decision import build_study_decision
from lambdaforge.products.models import ProductArtifact, ProductContract, StudyProduct, _text
from lambdaforge.products.registry import ProductRegistry, _encoded
from lambdaforge.products.selection import SelectionPolicy, select_models
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.models import atomic_json
from lambdaforge.work.recovery import read_owned_json


@dataclass(frozen=True, slots=True)
class ProductPublication:
    """One explicit immutable name/contract and native decision or snapshot selection policy."""

    name: str
    kind: str
    contract: str
    selection: SelectionPolicy | None = None
    outputs: tuple[str, ...] = ()
    scientific_meaning: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, value: Any) -> ProductPublication:
        _text(name, field="publication name")
        if (
            not isinstance(value, Mapping)
            or set(value) - {"kind", "contract", "select", "outputs", "scientific_meaning"}
            or not {"kind", "contract"}.issubset(value)
            or value["kind"]
            not in {
                "StudyDecision",
                "ModelSet",
                "ScientificReport",
                "AnalysisResult",
                "ModelArtifact",
                "Selection",
            }
        ):
            raise ValueError(
                "products entries require a supported native product kind and contract."
            )
        ProductContract(value["contract"], ("declaration",))
        if value["kind"] not in {"StudyDecision", "ModelSet"}:
            outputs, meaning = value.get("outputs"), value.get("scientific_meaning")
            if (
                "select" in value
                or not isinstance(outputs, list)
                or not outputs
                or len(outputs) > 128
                or any(not isinstance(item, str) or not item for item in outputs)
                or len(set(outputs)) != len(outputs)
                or not isinstance(meaning, Mapping)
                or not meaning
            ):
                raise ValueError(
                    "Work products require explicit unique outputs and scientific_meaning; "
                    "select is not allowed."
                )
            contract = ProductContract(value["contract"], tuple(meaning))
            meaning = contract.validate(meaning)
            return cls(
                name,
                value["kind"],
                value["contract"],
                outputs=tuple(outputs),
                scientific_meaning=meaning,
            )
        if "outputs" in value or "scientific_meaning" in value:
            raise ValueError(
                "StudyDecision/ModelSet derive meaning from native evidence, "
                "not Work output declarations."
            )
        if value["kind"] == "ModelSet":
            if "select" not in value:
                raise ValueError("ModelSet publication requires an explicit select policy.")
            selection = SelectionPolicy.from_mapping(value["select"])
        else:
            if "select" in value:
                raise ValueError("StudyDecision preserves native selection; select is not allowed.")
            selection = None
        return cls(name, value["kind"], value["contract"], selection)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind, "contract": self.contract}
        if self.selection is not None:
            value["select"] = self.selection.to_dict()
        if self.outputs:
            value["outputs"] = list(self.outputs)
            value["scientific_meaning"] = dict(self.scientific_meaning)
        return value


def publication_declarations(value: Any) -> tuple[ProductPublication, ...]:
    """Normalize a bounded public declaration once; never infer a project's product semantics."""
    if not isinstance(value, Mapping) or not value or len(value) > 128:
        raise ValueError("products must be a non-empty mapping of at most 128 named publications.")
    return tuple(ProductPublication.from_mapping(name, entry) for name, entry in value.items())


def pending_publications(configuration: Mapping[str, Any], execution_id: str) -> dict[str, Any]:
    """Record declared product obligations, independently of current Run status."""
    declarations = publication_declarations(configuration["products"])
    return {
        "product_publication_version": 1,
        "execution_id": execution_id,
        "status": "pending",
        "items": [
            {"name": item.name, **item.to_dict(), "status": "pending"} for item in declarations
        ],
    }


def publish_declared_products(
    configuration: Mapping[str, Any],
    source: Mapping[str, Any],
    execution_dir: Path,
    registry: ProductRegistry,
    *,
    analysis: Mapping[str, Any] | None = None,
    authored_space: Mapping[str, Any] | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Preview/apply native declarations from finalized evidence, retaining publication failures.

    Every committed product is independent of its producer. Successfully recorded publication is
    reusable even after unselected/old Attempt files were compacted. Failed products can be retried
    without recalculating training. A read-only preview never records failure or acquires a lock.
    """
    declarations = publication_declarations(configuration["products"])
    root = execution_dir.expanduser().absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Product publication requires a regular owned Execution directory.")
    if (
        type(source.get("execution_result_version")) is not int
        or source["execution_result_version"] != 1
        or source.get("status")
        not in {"succeeded", "failed", "completed_with_failures", "cancelled", "interrupted"}
        or root.name != source.get("execution_id")
    ):
        raise ValueError("Product publication requires finalized native Execution evidence.")
    state_path = root / "products.json"
    if state_path.resolve() != state_path:
        raise ValueError("Product publication record cannot be symbolic.")
    origin = read_owned_json(root / "execution.json")
    persisted = read_owned_json(root / "result.json")
    revision_path = root / "current-code.json"
    revision = read_owned_json(revision_path) if revision_path.exists() else origin
    if (
        origin.get("execution_id") != source["execution_id"]
        or revision.get("scientific_fingerprint") != source.get("scientific_fingerprint")
        or _encoded(persisted) != _encoded(source)
        or _encoded(read_owned_json(root / "configuration.json").get("products"))
        != _encoded(configuration["products"])
    ):
        raise ValueError("Publication requires the exact owned persisted result and declaration.")
    lock = CrossProcessFileLock(
        root / ".products.lock", shared=False, timeout_seconds=5, poll_interval_seconds=0.05
    )
    if apply:
        with lock:
            return _publish(
                declarations,
                source,
                root,
                registry,
                state_path,
                analysis=analysis,
                authored_space=authored_space,
                apply=True,
            )
    return _publish(
        declarations,
        source,
        root,
        registry,
        state_path,
        analysis=analysis,
        authored_space=authored_space,
        apply=False,
    )


def _publish(
    declarations: tuple[ProductPublication, ...],
    source: Mapping[str, Any],
    root: Path,
    registry: ProductRegistry,
    state_path: Path,
    *,
    analysis: Mapping[str, Any] | None,
    authored_space: Mapping[str, Any] | None,
    apply: bool,
) -> dict[str, Any]:
    # Private presentation keys added by ResultStore are not persisted producer evidence.
    source_digest = hashlib.sha256(
        _encoded({key: value for key, value in source.items() if not key.startswith("_")})
    ).hexdigest()
    previous: dict[str, Mapping[str, Any]] = {}
    if state_path.exists():
        state = registry._read(state_path)
        if (
            type(state.get("product_publication_version")) is not int
            or state["product_publication_version"] != 1
            or state.get("execution_id") != source["execution_id"]
            or not isinstance(state.get("items"), list)
            or any(not isinstance(item, Mapping) for item in state["items"])
        ):
            raise ValueError("Corrupt native product publication record; no Study is restarted.")
        if any(
            not isinstance(item.get("name"), str)
            or item.get("status") not in {"pending", "published", "failed"}
            for item in state["items"]
        ):
            raise ValueError("Product publication record has invalid item identities or states.")
        previous = {item["name"]: item for item in state["items"]}
        if len(previous) != len(state["items"]):
            raise ValueError("Product publication record has duplicate names.")
    items = []
    for declaration in declarations:
        policy = hashlib.sha256(_encoded(declaration.to_dict())).hexdigest()
        prior = previous.get(declaration.name, {})
        record: dict[str, Any] = {
            "name": declaration.name,
            "kind": declaration.kind,
            "contract": declaration.contract,
            "policy_fingerprint": policy,
            "source_digest": source_digest,
        }
        try:
            if (
                prior.get("status") == "published"
                and prior.get("policy_fingerprint") == policy
                and prior.get("source_digest") == source_digest
            ):
                product = registry.resolve(declaration.name, contract=declaration.contract)
                if product.content_id != prior.get("content_id"):
                    raise ValueError("Published product binding differs from its native receipt.")
                if apply:
                    registry.verify(product.content_id)
                plan = {"applied": apply, "reused": True}
            else:
                files: dict[str, Path] = {}
                if declaration.outputs:
                    product, files = _work_product(declaration, source, root)
                elif declaration.selection is not None:
                    selected = select_models(
                        source,
                        root,
                        declaration.selection,
                        name=declaration.name,
                        contract=declaration.contract,
                    )
                    product, files = selected.product, dict(selected.sources)
                else:
                    product = build_study_decision(
                        source,
                        name=declaration.name,
                        contract=declaration.contract,
                        analysis=analysis,
                        authored_space=authored_space,
                    )
                plan = registry.publish(product, files=files, apply=apply)
            record.update(
                {
                    "status": "published" if apply else "ready",
                    "content_id": product.content_id,
                    "scientific_id": product.scientific_id,
                    "reused": bool(plan.get("reused", False)),
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error)[:2048],
                        "phase": "product-publication",
                    },
                    "recovery": "lf products finalize EXECUTION --apply",
                }
            )
        items.append(record)
    result = {
        "product_publication_version": 1,
        "execution_id": source["execution_id"],
        "status": "failed"
        if any(item["status"] == "failed" for item in items)
        else "published"
        if apply
        else "ready",
        "applied": apply,
        "items": items,
    }
    if apply:
        if len(_encoded(result)) > 512 * 1024:
            raise ValueError("Product publication receipt exceeds its bounded metadata contract.")
        history = root / "product-publication-history.jsonl"
        if history.resolve() != history or (history.exists() and not history.is_file()):
            raise ValueError("Product publication history cannot be symbolic or non-regular.")
        with history.open("a", encoding="utf-8") as stream:
            for item in items:
                if item.get("reused"):
                    continue
                stream.write(
                    _encoded(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "execution_id": source["execution_id"],
                            **item,
                        }
                    ).decode()
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(state_path, result)
    return result


def _work_product(
    declaration: ProductPublication,
    source: Mapping[str, Any],
    root: Path,
) -> tuple[StudyProduct, dict[str, Path]]:
    """Promote explicitly selected native outputs; no implicit scientific interpretation."""
    from lambdaforge.work.managed import fingerprint
    from lambdaforge.work.result_projection import attempt_path, owned

    latest: dict[str, Mapping[str, Any]] = {}
    for run in source.get("runs", ()):
        previous = latest.get(run["run_id"])
        if previous is None or run["attempt_number"] > previous["attempt_number"]:
            latest[run["run_id"]] = run
    if len(latest) != 1:
        raise ValueError(
            "Work output promotion requires one logical Run; "
            "use explicit Study model selection for multiple Runs."
        )
    run = next(iter(latest.values()))
    if run.get("status") != "succeeded":
        raise ValueError(
            "Work products require a successful latest Attempt; partial evidence is not promoted."
        )
    directory = attempt_path(root, run)
    values, artifacts, files = {}, [], {}
    registered = {item["name"]: item for item in run.get("artifacts", ())}
    for name in declaration.outputs:
        if name in run.get("outputs", {}):
            values[name] = run["outputs"][name]
            continue
        if name not in registered:
            raise ValueError(f"Declared product output {name!r} was not registered by the Work.")
        artifact = registered[name]
        path = owned(directory, artifact["path"])
        if not path.is_file():
            raise ValueError(
                f"Work product output {name!r} must be a retained regular file. "
                "Register directory members explicitly."
            )
        before = path.stat()
        if fingerprint(path) != (artifact["sha256"], artifact["size_bytes"]):
            raise ValueError(f"Work product output checksum differs: {name}.")
        target = f"artifacts/{name}/{path.name}"
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Work product output changed during byte promotion: {name}.")
        artifacts.append(
            ProductArtifact(
                name, target, checksum.hexdigest(), path.stat().st_size, artifact["role"]
            )
        )
        files[name] = path
    product = StudyProduct(
        declaration.name,
        declaration.kind,
        ProductContract(declaration.contract, tuple(declaration.scientific_meaning)),
        {"outputs": values, "source_run": run["run_id"], "source_attempt": run["attempt_id"]},
        declaration.scientific_meaning,
        {
            "execution_id": source["execution_id"],
            "config_fingerprint": source["scientific_fingerprint"],
            "evidence_fingerprint": "sha256:" + hashlib.sha256(_encoded(source)).hexdigest(),
            "source_status": source["status"],
        },
        tuple(artifacts),
    )
    return product, files


def finalize_native_products(
    configuration: Mapping[str, Any],
    source: Mapping[str, Any],
    execution_dir: Path,
    registry: ProductRegistry,
) -> None:
    """Existing runner's postprocessing boundary; failure cannot erase scientific evidence."""
    from lambdaforge.analysis.StudyAnalysis import StudyAnalysis

    try:
        analysis_path = execution_dir / "analysis.json"
        result = publish_declared_products(
            configuration,
            source,
            execution_dir,
            registry,
            analysis=read_owned_json(analysis_path) if analysis_path.exists() else None,
            authored_space=StudyAnalysis.authored_space(configuration),
            apply=True,
        )
        for item in result["items"]:
            if item["status"] == "failed":
                print(
                    f"[products] Publication failed for {item['name']}: "
                    f"{item['failure']['message']}. Training evidence retained; "
                    "use lf products finalize EXECUTION --apply.",
                    file=sys.stderr,
                    flush=True,
                )
    except Exception as error:
        print(
            f"[products] Publication unavailable: {type(error).__name__}: {error}. "
            "Training evidence retained; inspect products.json before retrying publication.",
            file=sys.stderr,
            flush=True,
        )
