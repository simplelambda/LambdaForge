"""Explicit whole-dataset comparison evidence in the existing immutable product catalog.

Certificates are project verifier assertions, not signatures, byte equality, transitive aliases
or permission to skip verification of a dataset's exact published bytes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.data.DatasetArtifact import DatasetArtifact
from lambdaforge.data.DatasetComparison import DatasetComparison, DatasetComparisonContext
from lambdaforge.products.models import ProductContract, StudyProduct
from lambdaforge.products.registry import ProductRegistry

_CONTRACT = ProductContract(
    "lambdaforge/dataset-equivalence:v1",
    ("reference_content_id", "candidate_content_id", "scientific_contract", "verifier", "policy"),
)


@dataclass(frozen=True, slots=True)
class DatasetEquivalenceCertificate:
    """Sealed, directional comparison for two exact representations of one declared science.

    Build executes the native integrity checks and explicitly supplied trusted verifier. Publishing
    is separate and preview-first. The complete comparison evidence is bounded by the catalog's
    metadata contract; it is never truncated to invent an approval for an unchecked subset.
    """

    product: StudyProduct

    def __post_init__(self) -> None:
        product = self.product
        if (
            not isinstance(product, StudyProduct)
            or product.kind != "ScientificReport"
            or product.contract != _CONTRACT
            or product.artifacts
        ):
            raise ValueError("Not a native DatasetEquivalenceCertificate product contract.")
        timestamp = product.producer.get("dataset_comparison_completed_at_utc")
        try:
            captured = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else None
        except ValueError as error:
            raise ValueError("Certificate comparison timestamp is invalid.") from error
        if captured is None or captured.tzinfo is None:
            raise ValueError("Certificate requires timezone-aware comparison provenance.")
        meaning, payload = product.scientific_meaning, product.payload
        for field in ("reference_content_id", "candidate_content_id"):
            if not isinstance(meaning[field], str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", meaning[field]
            ):
                raise ValueError("Certificate requires exact dataset content identities.")
        scientific_id = DatasetArtifact.scientific_contract_id(meaning["scientific_contract"])
        evidence = payload.get("evidence")
        if (
            type(payload.get("dataset_equivalence_certificate_version")) is not int
            or payload["dataset_equivalence_certificate_version"] != 1
            or payload.get("result") != "scientifically_equivalent"
            or not isinstance(evidence, Mapping)
            or type(evidence.get("comparison_version")) is not int
            or evidence["comparison_version"] != 1
            or evidence.get("status") not in {"exact", "equivalent"}
            or evidence.get("scientifically_equivalent") is not True
            or evidence.get("registration_changed") is not False
            or type(evidence.get("byte_equal")) is not bool
            or evidence["byte_equal"]
            != (meaning["reference_content_id"] == meaning["candidate_content_id"])
            or evidence.get("policy") != meaning["policy"]
            or evidence.get("verifier") != meaning["verifier"]
        ):
            raise ValueError("Certificate is not an approved complete native comparison.")
        for side, field in (
            ("left", "reference_content_id"),
            ("right", "candidate_content_id"),
        ):
            record = evidence.get(side)
            if (
                not isinstance(record, Mapping)
                or record.get("content_id") != meaning[field]
                or record.get("comparison_scientific_id") != scientific_id
                or "root" in record
            ):
                raise ValueError("Certificate evidence differs from its scientific contract.")
        if evidence["status"] == "equivalent" and (
            evidence["byte_equal"] or not meaning["verifier"] or not meaning["policy"]
        ):
            raise ValueError("Different bytes require an explicit verifier and variable policy.")

    @classmethod
    def build(
        cls,
        reference: str | Path,
        candidate: str | Path,
        *,
        name: str,
        scientific_contract: Mapping[str, Any],
        producer: Mapping[str, Any],
        verifier: Callable[[DatasetComparisonContext], Mapping[str, Any]] | None = None,
        verifier_id: str | None = None,
        policy: Mapping[str, Any] | None = None,
        scientific_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> DatasetEquivalenceCertificate:
        """Compare all members first; never seal an arbitrary caller-supplied approval report.

        Producer records the real comparison operation/Execution and provenance. For historical
        datasets lacking declarations, scientific_contracts must explicitly bind each content ID
        to the project assertion, exactly as in DatasetComparison; no identity is inferred.
        """
        DatasetArtifact.scientific_contract_id(scientific_contract)
        report = DatasetComparison.compare(
            reference,
            candidate,
            verifier=verifier,
            verifier_id=verifier_id,
            policy=policy,
            scientific_contracts=scientific_contracts,
        )
        if report.get("scientifically_equivalent") is not True:
            raise ValueError(
                f"Cannot certify Dataset comparison ({report['status']}): "
                f"{report.get('reason', report.get('errors', 'verification did not approve'))}"
            )
        # Machine paths are provenance. They must not change the sealed scientific meaning
        # or the canonical comparison content when verified evidence moves to another host.
        roots = {side: report[side]["root"] for side in ("left", "right")}
        for side in roots:
            report[side] = {key: value for key, value in report[side].items() if key != "root"}
        return cls(
            StudyProduct(
                name=name,
                kind="ScientificReport",
                contract=_CONTRACT,
                scientific_meaning={
                    "reference_content_id": report["left"]["content_id"],
                    "candidate_content_id": report["right"]["content_id"],
                    "scientific_contract": dict(scientific_contract),
                    "verifier": report["verifier"],
                    "policy": report["policy"],
                },
                payload={
                    "dataset_equivalence_certificate_version": 1,
                    "result": "scientifically_equivalent",
                    "evidence": report,
                },
                producer={
                    **dict(producer),
                    "dataset_comparison_roots": roots,
                    "dataset_comparison_completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        )

    @classmethod
    def load(cls, registry: ProductRegistry, selector: str) -> DatasetEquivalenceCertificate:
        """Read bounded sealed evidence; do not execute its recorded verifier code."""
        return cls(registry.resolve(selector, contract=_CONTRACT))

    def publish(self, registry: ProductRegistry, *, apply: bool = False) -> dict[str, Any]:
        """Use the existing immutable writer, lock and idempotent publication authority."""
        return registry.publish(self.product, apply=apply)

    def accepts(
        self,
        *,
        reference_content_id: str,
        candidate_content_id: str,
        scientific_contract: Mapping[str, Any],
    ) -> bool:
        """Explicit exact pair/contract query; never search aliases or infer transitivity.

        This checks recorded scientific approval only. The consumer must separately verify current
        candidate bytes against candidate_content_id with DatasetOperations.verify. Work's ordinary
        typed dataset input still requires its exact content. This does not change resolution.
        """
        meaning = self.product.scientific_meaning
        return (
            meaning["reference_content_id"] == reference_content_id
            and meaning["candidate_content_id"] == candidate_content_id
            and DatasetArtifact.scientific_contract_id(scientific_contract)
            == DatasetArtifact.scientific_contract_id(meaning["scientific_contract"])
        )
