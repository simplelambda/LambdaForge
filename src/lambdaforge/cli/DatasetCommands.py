"""Thin dataset CLI adapter; dataset behavior remains in data services."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from typing import Any

import yaml

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.data.DatasetService import DatasetService
from lambdaforge.data.errors import DirectTransferUnavailableError


class DatasetCommands:
    """Parse/render dataset commands while delegating lifecycle decisions to services."""

    @classmethod
    def run(cls, arguments: argparse.Namespace) -> int:
        service = DatasetService(clusters=ClusterCatalog.load(arguments.clusters))
        operation = arguments.dataset_command
        if operation == "trust-ssh":
            payload: Any = service.replication_ssh_trust(
                arguments.source,
                arguments.destination,
                apply=arguments.apply,
                fingerprints=arguments.fingerprint,
                expected_file=arguments.known_hosts_file,
            )
        elif operation == "list":
            payload = [
                value.to_dict()
                for value in service.list(cluster=arguments.on, all_clusters=arguments.all)
            ]
        elif operation == "preflight":
            payload = service.publication_preflight(
                arguments.dataset, cluster=arguments.on, intent=arguments.intent
            )
        elif operation == "compare":
            policy = json.loads(arguments.policy.read_text()) if arguments.policy else None
            payload = service.compare_reconstruction(
                arguments.left,
                arguments.right,
                cluster=arguments.on,
                verifier=arguments.verifier,
                policy=policy,
                scientific_contracts=json.loads(arguments.contracts.read_text())
                if arguments.contracts
                else None,
            )
            if arguments.output:
                from lambdaforge.work.models import atomic_json

                if arguments.output.exists():
                    raise FileExistsError(
                        "Comparison report already exists; choose a fresh output path."
                    )
                atomic_json(arguments.output, payload)
        elif operation == "publish-candidate":
            payload = service.publish_candidate(
                arguments.root,
                cluster=arguments.on,
                version=arguments.version,
                apply=arguments.apply,
            )
        elif operation == "show":
            payload = service.describe(arguments.dataset, cluster=arguments.on)
        elif operation == "add":
            payload = service.add(
                arguments.manifest, cluster=arguments.on, root=arguments.root
            ).to_dict()
        elif operation == "locations":
            payload = [value.to_dict() for value in service.locations(arguments.dataset)]
        elif operation == "stats":
            schema = (
                yaml.safe_load(arguments.schema.read_text(encoding="utf-8"))
                if arguments.schema
                else None
            )
            payload = service.stats(arguments.dataset, cluster=arguments.on, schema=schema)
        elif operation == "members":
            partitions = dict(cls._partition(value) for value in arguments.partition)
            payload = service.members(
                arguments.dataset,
                cluster=arguments.on,
                partitions=partitions,
                offset=arguments.offset,
                limit=arguments.limit,
            )
        elif operation == "member":
            payload = service.member(arguments.dataset, arguments.member_id, cluster=arguments.on)
        elif operation == "diff":
            payload = service.diff(arguments.left, arguments.right, cluster=arguments.on)
        elif operation == "verify":
            payload = service.verify(arguments.dataset, cluster=arguments.on)
        elif operation == "lineage":
            payload = service.lineage(arguments.dataset)
        elif operation == "remove":
            if arguments.content_id:
                payload = service.retire(
                    arguments.dataset,
                    cluster=arguments.on,
                    content_id=arguments.content_id,
                    apply=arguments.apply,
                )
            else:
                removed = service.remove(arguments.dataset, cluster=arguments.on)
                payload = {"removed": True, "remaining": removed.to_dict() if removed else None}
        elif operation == "adopt":
            payload = service.adopt(
                arguments.dataset,
                cluster=arguments.on,
                content_id=arguments.content_id,
                apply=arguments.apply,
            )
        elif operation == "reconcile":
            payload = service.reconcile(
                arguments.dataset, cluster=arguments.on, apply=arguments.apply
            )
        elif operation == "delete":
            payload = service.delete(
                arguments.dataset,
                cluster=arguments.on,
                apply=arguments.apply,
                content_id=arguments.content_id,
            ).to_dict()
            if arguments.content_id:
                payload["explicit_content_id"] = arguments.content_id
        elif operation == "transfers":
            payload = service.replications(arguments.dataset)
        elif operation == "transfer-status":
            payload = service.replication_status(arguments.job_id)
        elif operation == "materialize":
            payload = service.materialize(
                arguments.dataset,
                cluster=arguments.on,
                strategy=arguments.strategy,
                apply=arguments.apply,
            ).to_dict()
        else:
            options = dict(
                source=arguments.source,
                destination=arguments.destination,
                route=getattr(arguments, "route", "auto"),
            )

            def ask_password() -> None:
                import getpass
                import warnings

                if arguments.json or not sys.stdin.isatty():
                    raise ValueError(
                        "Password entry needs an interactive terminal; "
                        "no secret CLI arguments or JSON prompts are supported."
                    )
                if options["route"] == "relay" or "local" in {
                    arguments.source,
                    arguments.destination,
                }:
                    raise ValueError(
                        "Destination-password authentication requires "
                        "two remote clusters and Direct."
                    )
                peer = service.clusters.get(arguments.destination)
                print(
                    f"Direct {arguments.source} → {arguments.destination}: "
                    "the destination password "
                    "will be delivered over encrypted SSH to the source, kept only in memory, "
                    "and used for this transfer. It will not be saved. Ctrl-C cancels.",
                    file=sys.stderr,
                )
                options["route"] = "direct"
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", getpass.GetPassWarning)
                        options["site_password"] = getpass.getpass(
                            f"Password for {peer.user}@{peer.host}: "
                        )
                except getpass.GetPassWarning:
                    raise ValueError(
                        "Terminal cannot hide password input; use the LF password dialog."
                    ) from None

            if getattr(arguments, "ask_password", False):
                ask_password()
            try:
                plan = service.replication_preview(arguments.dataset, **options)
            except DirectTransferUnavailableError as error:
                if (
                    "site_password" in options
                    or arguments.json
                    or not sys.stdin.isatty()
                    or not any(n.get("category") == "authentication" for n in error.diagnostics)
                ):
                    raise
                ask_password()
                plan = service.replication_preview(arguments.dataset, **options)
            if (
                plan.relay_consent_required
                and "site_password" not in options
                and not arguments.json
                and sys.stdin.isatty()
                and any(n.get("category") == "authentication" for n in plan.route_diagnostics)
            ):
                ask_password()
                plan = service.replication_preview(arguments.dataset, **options)
            if arguments.apply:
                if plan.relay_consent_required:
                    cls._human("replicate", plan.to_dict()) if not arguments.json else None
                    if arguments.json or not sys.stdin.isatty():
                        raise ValueError(
                            "No usable direct route was confirmed. Controller relay requires "
                            "explicit acceptance: inspect the preview, "
                            "then use --route relay --apply."
                        )
                    answer = input(
                        "Accept source → controller PC → destination relay? "
                        "Keep this PC powered and connected; closing LF is safe. [y/N] "
                    )
                    if answer.strip().lower() not in {"y", "yes"}:
                        raise ValueError("Relay not approved; no transfer started.")
                    options["route"] = "relay"
                payload = service.enqueue_replication(
                    arguments.dataset,
                    **options,
                    approved_route=plan.transfer_route,
                    approved_diagnostics=plan.route_diagnostics,
                ).to_dict()
            else:
                payload = plan.to_dict()
        if getattr(arguments, "json", False):
            if isinstance(payload, dict) and getattr(arguments, "default_cluster_source", None):
                payload.setdefault("target_source", arguments.default_cluster_source)
            print(json.dumps(payload, indent=2))
        else:
            cls._human(
                operation,
                payload,
                default_source=getattr(arguments, "default_cluster_source", None),
                verbose=getattr(arguments, "verbose", False),
            )
        for warning in service.discovery_warnings:
            print(f"Warning: dataset discovery degraded: {warning}", file=sys.stderr)
        return (
            2
            if (operation == "preflight" and not payload["allowed"])
            or (
                operation in {"adopt", "remove", "delete"}
                and isinstance(payload, dict)
                and payload.get("safe") is False
                and getattr(arguments, "content_id", None)
            )
            else 1
            if operation == "compare"
            and payload["status"] in {"invalid", "different", "unresolved"}
            else 0
        )

    @staticmethod
    def _replication_progress(value: Mapping[str, Any]) -> None:
        message = str(value.get("message", value.get("phase", "replication")))
        transferred = value.get("compressed_bytes")
        if isinstance(transferred, int):
            message += f" · {transferred / 1048576:.1f} MiB compressed"
        print(message, file=sys.stderr, flush=True)

    @staticmethod
    def _partition(value: str) -> tuple[str, str]:
        key, separator, selected = value.partition("=")
        if not separator or not key or not selected:
            raise ValueError("--partition requires NAME=VALUE.")
        return key, selected

    @staticmethod
    def _human(
        operation: str,
        payload: Any,
        *,
        default_source: str | None = None,
        verbose: bool = False,
    ) -> None:
        if operation == "trust-ssh":
            print(f"SSH host trust: {payload['initiator']} → {payload['peer']}")
            print(f"Peer: {payload['host']}:{payload['port']} · state: {payload['state']}")
            print(f"Trust file on {payload['initiator']}: {payload['known_hosts_file']}")
            for key in payload["keys"]:
                print(f"  {key['type']}  {key['fingerprint']}")
            print(payload["notice"])
            print(
                "Host trust recorded; no copy started."
                if payload.get("applied")
                else "No key changed. Apply requires --known-hosts-file and every --fingerprint "
                "from this preview, after independent verification."
            )
            return
        if operation in {"transfers", "transfer-status"}:
            for row in payload if isinstance(payload, list) else [payload]:
                progress = row.get("progress", {})
                print(
                    f"{row['job_id']} · {row.get('display_state', row['state'])} · {row['dataset']}"
                )
                print(f"  {row['source']} → {row['destination']} · host {row['host']}")
                print(
                    f"  {progress.get('message') or progress.get('phase') or 'Waiting for worker'}"
                )
                print(
                    f"  Last confirmed: {row.get('last_confirmed_at_utc') or 'not observed'} "
                    f"· age {row.get('observation_age_seconds')} s"
                )
                print(
                    "  Last byte advance: "
                    f"{row.get('last_byte_progress_at_utc') or 'not observed'} "
                    f"· compressed bytes {progress.get('compressed_bytes', 0)} "
                    f"· original bytes {row.get('uncompressed_bytes', 'unknown')}"
                )
                if row.get("reconciliation_error"):
                    print(f"  Registry reconciliation: {row['reconciliation_error']}")
                if row.get("observation_error"):
                    print(f"  Observation unavailable: {row['observation_error']}")
                if row.get("error"):
                    print(f"  Error: {row['error']}")
                print(f"  Logs: lf jobs logs {row['job_id']} --tail 100")
            return
        if operation == "replicate" and payload.get("action") == "REPLICATE":
            print(
                f"Dataset: {payload['dataset']} · original/uncompressed size: "
                f"{payload.get('estimated_bytes')} bytes"
            )
            print("Data path: " + " → ".join(payload.get("data_path", ())))
            print(
                f"Execution host: {payload.get('execution_host')}; "
                f"route: {payload.get('transfer_route')}"
            )
            for note in payload.get("route_diagnostics", ()):
                print(f"  {note.get('direction')}: {note.get('category')} · {note.get('message')}")
            if payload.get("requires_controller_online"):
                print(
                    "Keep this PC powered and connected. Closing LF is safe; "
                    "suspend/shutdown/network loss is not."
                )
            if payload.get("relay_consent_required"):
                print("Relay has NOT been approved. Explicit choice: --route relay --apply.")
            if payload.get("job_id"):
                print(
                    f"Job: {payload['job_id']} · inspect: "
                    f"lf datasets transfer-status {payload['job_id']}"
                )
            return
        if operation == "adopt" or operation == "remove" and "content_id" in payload:
            print(
                "CHOOSE DATASET REFERENCE" if operation == "adopt" else "REMOVE EXACT REGISTRATION"
            )
            print(f"Dataset: {payload['dataset']} · cluster: {payload['cluster']}")
            print(f"Content: {payload['content_id']}")
            print(f"Root: {payload.get('root') or 'not materialized'}")
            if operation == "adopt":
                print(f"Previous reference: {payload.get('previous_content_id') or 'none'}")
            print(payload["notice"])
            print(f"Safe: {payload['safe']} · applied: {payload['applied']}")
            if not payload["safe"]:
                print(f"Reason: {payload.get('reason') or 'active consumers or unsafe target'}")
                if payload.get("active_consumers"):
                    print("Active consumers: " + ", ".join(payload["active_consumers"]))
            if payload.get("history"):
                print(f"Archived declaration: {payload['history']}")
            if payload["safe"] and not payload["applied"]:
                print(
                    f"Apply: lf datasets {operation} {payload['dataset']} "
                    f"--on {payload['cluster']} --content-id {payload['content_id']} --apply"
                )
            return
        if operation == "preflight":
            print(f"Dataset: {payload['dataset']} · {payload['intent']}")
            print(f"Preflight: {'ALLOWED' if payload['allowed'] else 'BLOCKED'}")
            print(payload["reason"])
            for placement in payload["locations"]:
                print(f"  {placement['cluster']} · {placement['content_id']}")
                print(f"    {placement['root']}")
            if payload.get("next"):
                print(f"Next: {payload['next']}")
            return
        if operation == "compare":
            print(f"Dataset comparison: {payload['status'].upper()}")
            for side in ("left", "right"):
                print(f"{side.title()}: {payload[side]['root']}")
                print(f"  Content: {payload[side]['content_id']}")
            print(f"Exact byte equality: {payload['byte_equal']}")
            print(f"Scientific equivalence: {payload['scientifically_equivalent']}")
            if payload.get("reason"):
                print(payload["reason"])
            print(f"Changed members checked: {len(payload['members'])}")
            if payload.get("errors"):
                for group in payload["errors"]:
                    for error in group[:5]:
                        print(f"  {error}")
            print("No registered content was changed. Use --json for complete evidence.")
            return
        if operation == "publish-candidate":
            print(f"Dataset: {payload['dataset']}")
            print(f"Content: {payload['content_id']}")
            print(f"Saved candidate: {payload['candidate']}")
            print(f"Destination: {payload['destination']}")
            print("Published." if payload["applied"] else "Preview only; add --apply to publish.")
            print("Consumer computation and previous Attempt history are unchanged.")
            return
        if operation == "list":
            print("DATASET  MEMBERS  PLACEMENTS  CONTENT")
            for record in payload:
                print(
                    f"{record['name']}@{record['version']}  {record['sample_count']}  "
                    f"{','.join(item['cluster'] for item in record['placements']) or '-'}  "
                    f"{record['dataset_id']}"
                )
            return
        if operation == "members":
            print("MEMBER  PARTITIONS  TARGETS  ASSETS")
            for member in payload["members"]:
                print(
                    f"{member['id']}  {member['partitions']}  {member['targets']}  "
                    f"{','.join(member['assets']) or '-'}"
                )
            print(f"Returned {payload['returned']} (offset={payload['offset']}).")
            return
        if operation == "show":
            print(f"Dataset: {payload['name']}@{payload['version']}")
            print(f"Content: {payload['dataset_id']}")
            print(f"Build: {payload.get('build_id') or 'unknown'}")
            print(f"Published: {payload.get('created_at_utc') or 'unknown'}")
            print(f"Members: {payload['sample_count']}")
            print(f"Partitions: {payload.get('partitions', {})}")
            print(f"Target schema: {payload.get('target_schema') or {}}")
            print(f"Global assets: {', '.join(payload.get('global_assets', {})) or 'none'}")
            print(f"Producer: {payload.get('producer') or {}}")
            print(f"Lineage: {', '.join(payload.get('lineage', ())) or 'none'}")
            print("Placements:")
            consistency = payload.get("placement_consistency", {})
            if not payload["placements"]:
                print("  none")
            for placement in payload["placements"]:
                availability = str(
                    consistency.get("states", {}).get(
                        placement["cluster"],
                        consistency.get("state")
                        if consistency.get("cluster") == placement["cluster"]
                        else "cached",
                    )
                )
                print(f"  {placement['cluster']}  {availability.upper()}")
                print(f"    {placement['root']}")
            if consistency.get("state"):
                print(
                    f"Target state: {str(consistency['state']).upper()} "
                    f"on {consistency.get('cluster')}"
                )
                if consistency.get("reason"):
                    print(f"  {consistency['reason']}")
            return
        if operation == "delete":
            print("DELETE DATASET PLACEMENT")
            print(f"Dataset: {payload['dataset']}")
            print(f"Dataset ID: {payload.get('dataset_id')}")
            print(f"Cluster: {payload['cluster']}")
            print(f"Root: {payload.get('root') or 'not observed'}")
            file_count = payload.get("file_count")
            size_bytes = payload.get("size_bytes")
            print(f"Files: {file_count if file_count is not None else 'unknown'}")
            print(f"Size: {size_bytes if size_bytes is not None else 'unknown'} bytes")
            print(f"Placement state: {str(payload.get('placement_state', 'unknown')).upper()}")
            verification = (
                "VALID"
                if payload.get("placement_state") == "available"
                else "NOT APPLICABLE"
                if payload.get("action") == "CLEAN_STALE_REGISTRATION"
                else "UNCONFIRMED"
            )
            print(f"Verification: {verification}")
            consumers = payload.get("active_consumers", [])
            print(f"Active consumers: {', '.join(consumers) if consumers else 'none'}")
            print(f"Action: {payload.get('action')}")
            print(f"Safe: {payload.get('safe')}")
            if payload.get("reasons"):
                print("Reasons:")
                for reason in payload["reasons"]:
                    print(f"  - {reason}")
            print("Other identities and scientific Run evidence: PRESERVED")
            print(f"Applied: {payload.get('applied')}")
            if payload.get("safe") and not payload.get("applied"):
                print(
                    "Apply: lf datasets delete "
                    f"{payload['dataset']} --on {payload['cluster']} "
                    + (
                        f"--content-id {payload['explicit_content_id']} "
                        if payload.get("explicit_content_id")
                        else ""
                    )
                    + "--apply"
                )
            return
        if operation == "reconcile":
            print("DATASET RECONCILIATION")
            print(f"Dataset: {payload['dataset']}")
            print(f"Dataset ID: {payload['dataset_id']}")
            print(f"Cluster: {payload['cluster']}")
            print(f"State: {str(payload['state']).upper()}")
            print(f"Proposed action: {payload['action']}")
            print(f"Reason: {payload['reason']}")
            if payload.get("why_safe"):
                print(f"Why safe: {payload['why_safe']}")
            print(f"Applied: {payload['applied']}")
            if payload.get("safe") and not payload.get("applied") and payload["action"] != "NOOP":
                print(
                    "Apply: lf datasets reconcile "
                    f"{payload['dataset']} --on {payload['cluster']} --apply"
                )
            return
        print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).rstrip())
