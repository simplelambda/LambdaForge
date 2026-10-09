"""Thin native boundary over the immutable product catalog; never launches scientific work."""

from __future__ import annotations

import json
from typing import Any

from lambdaforge.products import (
    ProductBundle,
    ProductRegistry,
    StudyProduct,
)


def _summary(product: StudyProduct) -> str:
    return (
        f"{product.name} · {product.kind}\n"
        f"  Contract: {product.contract.identifier}\n"
        f"  Content: {product.content_id}\n"
        f"  Scientific meaning: {product.scientific_id}\n"
        f"  Producer: {product.producer['execution_id']}\n"
        f"  Artifacts: {len(product.artifacts)} · "
        f"{sum(item.size_bytes for item in product.artifacts)} bytes"
    )


def run_product_command(arguments: Any) -> int:
    """Resolve/verify products locally; publishing is explicit preview/apply and byte-owning."""
    registry = ProductRegistry(arguments.root)
    operation = arguments.product_command
    payload: Any
    if operation == "materialize":
        from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
        from lambdaforge.controlplane.DependencyMaterialization import DependencyMaterialization

        payload = DependencyMaterialization(ClusterCatalog.load(arguments.clusters)).materialize(
            arguments.selector,
            cluster=arguments.on,
            apply=arguments.apply,
            source_root=arguments.root,
        )
        human = (
            f"{'Materialized' if payload['applied'] else 'Would materialize'} exact product "
            f"{payload['identity']} on {payload['cluster']} at {payload['destination']}. "
            "No producer computation is launched."
        )
    elif operation == "list":
        products = registry.list(offset=arguments.offset, limit=arguments.limit)
        payload = {
            "items": [product.to_dict() for product in products],
            "offset": arguments.offset,
            "limit": arguments.limit,
            "root": str(registry.root),
        }
        human = "\n\n".join(_summary(product) for product in products) or "No published products."
    elif operation == "show":
        product = registry.show(arguments.selector)
        payload = {**product.to_dict(), "placement": str(registry.root)}
        human = _summary(product) + f"\n  Placement: {registry.root}"
    elif operation == "verify":
        payload = registry.verify(arguments.selector)
        human = (
            f"Verified {payload['name']} · {payload['artifacts']} artifacts · "
            f"{payload['size_bytes']} bytes"
        )
    elif operation in {"provenance", "consumers"}:
        records = (registry.provenance if operation == "provenance" else registry.consumers)(
            arguments.selector, offset=arguments.offset, limit=arguments.limit
        )
        payload = {"items": list(records), "offset": arguments.offset, "limit": arguments.limit}
        human = (
            "\n".join(
                f"{record['execution_id']} · evidence {record['evidence_fingerprint']}"
                if operation == "provenance"
                else f"{record['consumer']['execution_id']} · {record['consumer']['run_id']} · "
                f"{record['consumer']['attempt_id']} · input {record['consumer']['input']}"
                for record in records
            )
            or f"No {operation} records."
        )
    elif operation in {"finalize", "status"}:
        from lambdaforge.work import ResultStore

        store = ResultStore(arguments.results_root)
        payload = (
            store.product_status(arguments.selector)
            if operation == "status"
            else store.finalize_products(
                arguments.selector, product_root=arguments.root, apply=arguments.apply
            )
        )
        human = "\n".join(
            f"{item['name']} · {item['status']}"
            + (f" · {item['failure']['message']}" if "failure" in item else "")
            for item in payload["items"]
        )
        if operation == "status" and not payload["items"]:
            human = "No declared product publications."
        if operation == "finalize" and not arguments.apply:
            human += "\nPreview only; add --apply. No training is launched."
    elif operation == "decide":
        from lambdaforge.work import ResultStore

        product = ResultStore(arguments.results_root).decision(
            arguments.selector, name=arguments.name, contract=arguments.contract
        )
        plan = registry.publish(product, apply=arguments.apply)
        payload = {**plan, "decision": dict(product.payload)}
        human = (
            f"{'Published' if plan['applied'] else 'Would publish'} StudyDecision {plan['name']}\n"
            f"Selection: {product.payload['status']}\n"
            f"Candidate: {product.payload['preferred_candidate']}\n"
            f"Scientific questions: {product.payload['scientific_status']}\n"
            "Native selection preserved; no HPO recomputation or training is launched."
        )
        if not plan["applied"]:
            human += "\nPreview only; add --apply after reviewing the decision."
    elif operation == "select":
        import yaml

        from lambdaforge.products.selection import SelectionPolicy, select_models
        from lambdaforge.work import ResultStore

        policy_path = arguments.policy.expanduser().absolute()
        if (
            policy_path.resolve() != policy_path
            or not policy_path.is_file()
            or policy_path.stat().st_size > 512 * 1024
        ):
            raise ValueError("Selection policy requires a bounded regular non-symbolic YAML file.")
        policy = SelectionPolicy.from_mapping(
            yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        )
        store = ResultStore(arguments.results_root)
        selection = select_models(
            store.select(arguments.selector),
            store.execution_directory(arguments.selector),
            policy,
            name=arguments.name,
            contract=arguments.contract,
        )
        plan = registry.publish(
            selection.product, files=dict(selection.sources), apply=arguments.apply
        )
        payload = {**plan, "selection": dict(selection.product.payload), "policy": policy.to_dict()}
        human = (
            f"{'Published' if plan['applied'] else 'Would publish'} ModelSet {plan['name']} · "
            f"{len(selection.sources)} selected models\n"
            f"Ranking: snapshot {policy.rank_by} ({policy.mode})\n"
            "Only explicitly scored model snapshots are eligible; no HPO decision is changed."
        )
        if not plan["applied"]:
            human += "\nPreview only; add --apply after reviewing the selection."
    elif operation in {"export", "import"}:
        if operation == "export":
            payload = ProductBundle.export(
                registry, arguments.selector, arguments.output, apply=arguments.apply
            )
        else:
            payload = ProductBundle.import_bundle(registry, arguments.source, apply=arguments.apply)
        human = (
            f"{'Applied' if payload['applied'] else 'Preview'} product {operation} · "
            f"{payload['name']}\nContent: {payload['content_id']}\n"
            "No Study execution is launched."
        )
        if not payload["applied"]:
            human += "\nAdd --apply after reviewing this plan."
    elif operation == "publish":
        path = arguments.manifest.expanduser().absolute()
        product = registry.read_manifest(path)
        files = {}
        for raw in arguments.file:
            name, separator, value = raw.partition("=")
            if not separator or not name or not value or name in files:
                raise ValueError("--file requires one unique artifact NAME=PATH per source.")
            files[name] = value
        payload = registry.publish(product, files=files, apply=arguments.apply)
        human = (
            f"{'Published' if payload['applied'] else 'Would publish'} {payload['name']} · "
            f"{payload['size_bytes']} bytes\nContract: {payload['contract']}\n"
            f"Content: {payload['content_id']}\n{payload['verification']}"
        )
        if not payload["applied"]:
            human += "\nPreview only; add --apply to promote independent durable bytes."
    else:
        raise ValueError(f"Unknown product operation {operation!r}.")
    print(json.dumps(payload, indent=2, ensure_ascii=False) if arguments.json else human)
    return 4 if isinstance(payload, dict) and payload.get("status") == "failed" else 0
