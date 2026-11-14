"""Bounded no-copy probes; passwords require explicit scoped in-memory delivery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lambdaforge.controlplane.RemoteCommandTimeout import RemoteCommandTimeout
from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.data.errors import DirectTransferUnavailableError, UnsafeDatasetOperationError

if TYPE_CHECKING:
    from lambdaforge.data.DatasetService import DatasetService


def classify_route_failure(message: str) -> str:
    """Describe observed errors, never infer absence of site connectivity from uncertainty."""
    text = message.lower()
    for category, markers in (
        (
            "host-key",
            (
                "host key verification failed",
                "remote host identification has changed",
                "known_hosts",
                "hostkey",
                "host key for server",
            ),
        ),
        (
            "authentication",
            (
                "permission denied (publickey",
                "authentication failed",
                "no supported authentication",
                "no site key",
            ),
        ),
        ("timeout", ("timed out", "timeout")),
        (
            "network",
            (
                "no route to host",
                "network is unreachable",
                "connection refused",
                "could not resolve hostname",
                "connection reset",
            ),
        ),
        ("permissions", ("permissionerror", "permission denied", "not readable", "not writable")),
        ("tools", ("not found", "no module named", "command not found", "importerror")),
    ):
        if any(marker in text for marker in markers):
            return category
    return "indeterminate"


def select_route(
    service: DatasetService,
    source: str,
    destination: str,
    route: str,
    *,
    source_root: str | None = None,
    target_options: dict[str, Any] | None = None,
    site_password: str | None = None,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Try push, then pull, and retain diagnostics for each attempted direction.

    A probe checks native imports/interpreter plus readonly source/destination permissions and
    owned destination planning. Full checksums/storage admission still run at copy/commit time.
    """
    origin, target = service.clusters.get(source), service.clusters.get(destination)
    if site_password is not None and (
        origin.transport == "local"
        or target.transport == "local"
        or route == "relay"
        or not site_password
        or len(site_password.encode()) > 4096
    ):
        raise ValueError(
            "A transient destination password requires two remote clusters and a direct route."
        )
    if target.storage is None or (target.storage.dataset_root is None and destination != "local"):
        raise ValueError("Configure storage.dataset_root on the destination before replication.")
    if origin.transport == "local" or target.transport == "local":
        selected = "local-stream" if origin.transport == target.transport else "controller-stream"
        if route == "direct" and selected == "controller-stream":
            raise UnsafeDatasetOperationError(
                "Direct site SSH is unavailable for a controller endpoint; "
                "select --route relay explicitly."
            )
        return selected, (
            {
                "direction": "controller",
                "category": "local-endpoint",
                "message": "A selected endpoint is on the controller computer.",
            },
        )
    if route == "relay":
        return "controller-stream", (
            {
                "direction": "controller",
                "category": "explicit-relay",
                "message": "Operator explicitly selected controller relay; "
                "no direct-route claim is made.",
            },
        )

    helper = Path(__file__).with_name("DatasetTransfer.py").read_text(encoding="utf-8")
    diagnostics: list[dict[str, Any]] = []
    directions = (
        (("direct-ssh", origin, target),)
        if site_password is not None
        else (("direct-ssh", origin, target), ("direct-ssh-pull", target, origin))
    )
    for mode, host, peer in directions:
        try:
            host_transport = service.factory.transport(host)
            host_python = service._python(host.name, host_transport)
            peer_python = service._python(peer.name, service.factory.transport(peer))
            host_options: dict[str, Any] = (
                {"role": "source", "root": source_root}
                if host is origin
                else {"role": "destination", **(target_options or {})}
            )
            if site_password is not None:
                host_options["password_support"] = True
            peer_options = (
                {"role": "destination", **(target_options or {})}
                if peer is target
                else {"role": "source", "root": source_root}
            )
            # Pin both interpreters and paths. Run the same no-write helper at both ends;
            # the local-to-host connection is *not* accepted as evidence for peer SSH.
            own = host_transport.run(
                (host_python, "-c", helper, "probe", json.dumps(host_options)), timeout=15
            )
            if own.returncode or own.stdout.strip() != "LF_DATA_TRANSFER_READY":
                raise RuntimeError("Execution endpoint probe failed: " + own.stderr[-1500:])
            peer_command = (
                *peer.command_prefix,
                peer_python,
                "-c",
                helper,
                "probe",
                json.dumps(peer_options),
            )
            if site_password is None:
                result = host_transport.run(service._site_ssh(peer, peer_command), timeout=20)
            else:
                from lambdaforge.controlplane.CommandResult import CommandResult

                options = {
                    "peer": {"host": peer.host, "user": peer.user, "port": peer.port},
                    "command": list(peer_command),
                }
                with host_transport.stream(
                    (host_python, "-c", helper, "password-probe", json.dumps(options))
                ) as stream:
                    stream.stdin.write(site_password.encode())
                    stream.finish_input()
                    output = stream.stdout.read(65537)
                    stream.wait()
                    if len(output) > 65536:
                        raise ValueError("Password route probe exceeded its metadata bound.")
                    result = CommandResult(0, output.decode(), "")
            ready = result.returncode == 0 and result.stdout.strip() == "LF_DATA_TRANSFER_READY"
            message = (
                "Effective read/write transfer prerequisites verified (no bytes copied)."
                if ready
                else (
                    result.stderr[-1500:]
                    or "Unexpected/empty probe response; connectivity is unconfirmed."
                )
            )
            category = "ready" if ready else classify_route_failure(message)
            note = {
                "direction": mode,
                "execution_host": host.name,
                "peer": peer.name,
                "category": category,
                "exit_code": result.returncode,
                "message": SecretRedactor.redact(message),
                **(
                    {"authentication": "password", "credential_retention": "memory-only"}
                    if site_password is not None
                    else {}
                ),
            }
        except Exception as error:
            message = SecretRedactor.redact(
                f"{type(error).__name__}: {error}", (site_password,) if site_password else ()
            )[:1500]
            note = {
                "direction": mode,
                "execution_host": host.name,
                "peer": peer.name,
                "category": "timeout"
                if isinstance(error, (TimeoutError, RemoteCommandTimeout))
                else classify_route_failure(message),
                "exit_code": None,
                "message": message,
            }
            ready = False
        diagnostics.append(note)
        if ready:
            return mode, tuple(diagnostics)
    if route == "direct" or site_password is not None:
        details = "; ".join(
            f"{d['direction']}: {d['category']} ({d['message']})" for d in diagnostics
        )
        raise DirectTransferUnavailableError(
            (
                "Direct password SSH is unavailable or unconfirmed; no copy started. "
                if site_password is not None
                else "Direct site SSH is unavailable or unconfirmed in both directions. "
            )
            + details
            + ". Fingerprint trust is separate from login authorization. Configure approved "
            "host-to-host authentication, or use --ask-password in an interactive terminal / "
            "Direct with password in LF for explicit one-use destination-password delivery "
            "to the source host. The source requires current LF and optional cluster-password "
            "support. Credentials are never persisted; no automatic retry or relay fallback. "
            "Never disable host-key checks or forward private keys. "
            "Use --route relay only to explicitly accept a controller-dependent transfer.",
            tuple(diagnostics),
        )
    return "controller-stream", tuple(diagnostics)
