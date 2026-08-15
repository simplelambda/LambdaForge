"""Cluster catalog, credential, diagnostics and bootstrap CLI commands."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import yaml

from lambdaforge.cli.common import (
    delete_dotted,
    keyring_reference,
    print_resources,
    print_storage,
    set_dotted,
)
from lambdaforge.controlplane.ClusterAuthentication import ClusterAuthentication
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterService import ClusterService
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.CredentialService import CredentialService
from lambdaforge.controlplane.Doctor import Doctor
from lambdaforge.controlplane.python_runtime import PythonRuntimePolicy
from lambdaforge.controlplane.ResourceService import ResourceService
from lambdaforge.controlplane.SshConnectionPolicy import SshConnectionPolicy
from lambdaforge.controlplane.StorageService import StorageService
from lambdaforge.controlplane.TorchInstallationPolicy import TorchInstallationPolicy


def run_cluster_command(arguments: argparse.Namespace) -> int:
    """Execute one parsed ``clusters`` action through control-plane services."""
    cluster_catalog = ClusterCatalog.load(arguments.catalog)
    if arguments.cluster_command == "add":
        destination = arguments.catalog or (
            ClusterCatalog.user_path()
            if arguments.scope == "user"
            else ClusterCatalog.project_path()
        )
        credential = arguments.credential
        credentials = CredentialService()
        if arguments.store_password:
            if arguments.auth != "password":
                raise ValueError("--store-password requires --auth password.")
            credential = credential or keyring_reference(
                arguments.name, arguments.user, arguments.host
            )
            if not credential.startswith("keyring:"):
                raise ValueError("--store-password requires a keyring: reference.")
        runtime_policy = PythonRuntimePolicy(
            arguments.python_strategy
            or ("auto" if arguments.environment == "managed" else "existing"),
            arguments.python,
            arguments.python_version,
            not arguments.no_managed_python,
        )
        cluster_profile = ClusterProfile(
            arguments.name,
            transport="ssh" if arguments.host else "local",
            scheduler=arguments.scheduler,
            host=arguments.host,
            user=arguments.user,
            port=arguments.port,
            auth=ClusterAuthentication(arguments.auth, credential),
            known_hosts=(str(arguments.known_hosts) if arguments.known_hosts else None),
            ssh_timeout=arguments.ssh_timeout,
            connection=SshConnectionPolicy(
                connect_timeout=arguments.connect_timeout,
                auth_timeout=arguments.auth_timeout,
                banner_timeout=arguments.banner_timeout,
                keepalive=arguments.keepalive_interval,
                multiplex=not arguments.no_multiplex,
                persist=arguments.control_persist,
                command_timeout=arguments.command_timeout,
            ),
            workspace=arguments.workspace,
            storage=ClusterStoragePolicy.from_mapping(
                {
                    key: value
                    for key, value in {
                        "state_root": arguments.state_root,
                        "cache_root": arguments.cache_root,
                        "run_root": arguments.run_root,
                        "dataset_root": arguments.dataset_root,
                        "cache_max_size": arguments.cache_max_size,
                        "cache_max_age": arguments.cache_max_age,
                    }.items()
                    if value is not None
                },
                workspace=arguments.workspace,
            ),
            python=arguments.python,
            python_runtime=runtime_policy,
            environment=arguments.environment,
            wheelhouse=(str(arguments.wheelhouse) if arguments.wheelhouse else None),
            pytorch=TorchInstallationPolicy(
                arguments.torch_channel,
                (True if arguments.require_cuda else False if arguments.no_require_cuda else None),
            ),
            project_module=arguments.project_module,
            data_environment=arguments.data_environment,
        )
        if arguments.store_password:
            secret = credentials.interactive.get(
                "interactive",
                prompt=f"Password for {arguments.user or ''}@{arguments.host}: ",
            )
            credentials.store(credential or "", secret)
            del secret
        print(ClusterCatalog.add(destination, cluster_profile))
        return 0
    if arguments.cluster_command == "list":
        cluster_payload = [cluster_catalog.inspect(name) for name in cluster_catalog.names()]
        if arguments.json:
            print(json.dumps(cluster_payload, indent=2))
        else:
            for inspected in cluster_payload:
                profile_payload = inspected["profile"]
                print(
                    f"{profile_payload['name']:<16} "
                    f"{profile_payload['transport']:<6} "
                    f"{profile_payload['scheduler']:<6} "
                    f"{profile_payload['host'] or '-'} "
                    f"[{inspected['source']}]"
                )
        return 0
    if arguments.cluster_command == "resources":
        cluster_resource = ResourceService(cluster_catalog).get(arguments.name)
        value = cluster_resource.to_dict()
        if arguments.json:
            print(json.dumps(value, indent=2))
        else:
            print_resources(value)
        return 0 if cluster_resource.online else 1
    if arguments.cluster_command == "storage":
        cluster_storage = StorageService(cluster_catalog).status(arguments.name)
        value = cluster_storage.to_dict()
        if arguments.json:
            print(json.dumps(value, indent=2))
        else:
            print_storage([value])
        return 0 if cluster_storage.online else 1
    cluster_profile = cluster_catalog.get(arguments.name)
    if arguments.cluster_command in {"set", "unset", "remove"}:
        source = cluster_catalog.source(cluster_profile.name)
        if source is None:
            raise ValueError("The built-in local profile cannot be modified.")
        if arguments.cluster_command == "remove":
            ClusterCatalog.remove(source, cluster_profile.name)
            print(source)
            return 0
        descriptor = cluster_profile.to_dict(include_defaults=False)
        descriptor.pop("name", None)
        if arguments.cluster_command == "set":
            if arguments.key.startswith("python.") and isinstance(descriptor.get("python"), str):
                descriptor["python"] = PythonRuntimePolicy(
                    "existing", str(descriptor["python"])
                ).to_dict()
            set_dotted(
                descriptor,
                arguments.key,
                yaml.safe_load(arguments.value),
            )
        else:
            delete_dotted(descriptor, arguments.key)
        ClusterCatalog.add(
            source,
            ClusterProfile.from_mapping(cluster_profile.name, descriptor),
        )
        print(source)
        return 0
    if arguments.cluster_command == "show":
        print(json.dumps(cluster_profile.to_dict(), indent=2))
        return 0
    if arguments.cluster_command == "inspect":
        print(json.dumps(cluster_catalog.inspect(arguments.name), indent=2))
        return 0
    if arguments.cluster_command == "export":
        cluster_export_payload = cluster_profile.to_dict()
        cluster_export_payload.pop("name", None)
        exported = {"clusters": {cluster_profile.name: cluster_export_payload}}
        if arguments.output is None:
            print(yaml.safe_dump(exported, sort_keys=False, allow_unicode=True), end="")
        else:
            ClusterCatalog.export(arguments.output, cluster_profile)
            print(arguments.output.expanduser().resolve())
        return 0
    if arguments.cluster_command == "credentials":
        if cluster_profile.name == "local":
            raise ValueError("The built-in local profile has no SSH credentials.")
        source = cluster_catalog.source(cluster_profile.name)
        if source is None:
            raise ValueError("The selected profile has no writable catalog source.")
        credentials = CredentialService()
        reference = cluster_profile.auth.credential or keyring_reference(
            cluster_profile.name, cluster_profile.user, cluster_profile.host
        )
        if arguments.credential_command == "set":
            if not reference.startswith("keyring:"):
                reference = keyring_reference(
                    cluster_profile.name, cluster_profile.user, cluster_profile.host
                )
            secret = credentials.interactive.get(
                "interactive",
                prompt=(f"Password for {cluster_profile.user or ''}@{cluster_profile.host}: "),
            )
            credentials.store(reference, secret)
            del secret
            updated = replace(
                cluster_profile,
                auth=ClusterAuthentication("password", reference),
            )
            ClusterCatalog.add(source, updated)
            print(f"Stored credential reference {reference!r} for {cluster_profile.name!r}.")
            return 0
        if not reference.startswith("keyring:"):
            raise ValueError("Only keyring: credentials can be deleted from storage.")
        credentials.delete(reference)
        updated = replace(cluster_profile, auth=ClusterAuthentication("password", None))
        ClusterCatalog.add(source, updated)
        print(f"Deleted credential for {cluster_profile.name!r}; password mode is now interactive.")
        return 0
    if arguments.cluster_command == "test":
        cluster_report = Doctor(cluster_catalog).check(cluster_profile.name)
        print(
            json.dumps(cluster_report.to_dict(), indent=2)
            if arguments.json
            else cluster_report.summary()
        )
        return cluster_report.exit_code
    bootstrapped = ClusterService(cluster_catalog).bootstrap(
        cluster_profile.name,
        wheelhouse=arguments.wheelhouse,
        dry_run=arguments.dry_run,
    )
    print(
        json.dumps(bootstrapped.to_dict(), indent=2)
        if arguments.json
        else (
            f"Cluster {bootstrapped.cluster!r} "
            f"{'plans' if bootstrapped.planned else 'is ready with'} "
            f"{bootstrapped.environment_id} "
            f"({bootstrap_action(bootstrapped.planned, bootstrapped.reused)}).\n"
            f"Python: {bootstrapped.python}\n"
            f"Runtime: {dict(bootstrapped.runtime or {})}\n"
            f"PyTorch: {dict(bootstrapped.pytorch or {})}"
        )
    )
    return 0


def bootstrap_action(planned: bool, reused: bool) -> str:
    """Render the mutually exclusive bootstrap disposition compactly."""
    if planned:
        return "planned"
    return "reused" if reused else "created"
