"""Explicit first-use SSH host trust on an initiating cluster, never automatic key replacement.

This file is also sent to an already authenticated cluster interpreter. ssh-keyscan observations
are NOT authenticated: the operator must verify the displayed fingerprints independently.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _run(command: list[str], *, timeout: float = 12) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if len(result.stdout) > 65536 or len(result.stderr) > 65536:
        raise ValueError("SSH trust response exceeds the metadata limit.")
    return result


def _safe(path: Path) -> None:
    from lambdaforge.work.snapshot import validate_path

    validate_path(path)
    if not path.is_absolute() or not path.is_relative_to(Path.home()):
        raise ValueError("Host trust requires an explicit user-owned known_hosts file under HOME.")
    for candidate in (path, *path.parents):
        if not candidate.is_relative_to(Path.home()):
            break
        if candidate.exists():
            info = candidate.stat()
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise ValueError("Host trust file/directory is not exclusively user-controlled.")
    if path.exists() and (not path.is_file() or path.stat().st_size > 8 * 1024 * 1024):
        raise ValueError("Unsafe or oversized known_hosts file; review with the administrator.")


def inspect(options: dict[str, Any]) -> dict[str, Any]:
    """Read effective SSH configuration and offered keys; never create or modify files."""
    if not hasattr(os, "getuid"):
        raise ValueError("Peer host trust requires POSIX user ownership on the initiating host.")
    host, port = options["host"], options["port"]
    if (
        not isinstance(host, str)
        or not host
        or host.startswith("-")
        or any(c.isspace() for c in host)
        or any(c in host for c in "/,\\\0")
    ):
        raise ValueError("Invalid SSH peer hostname.")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Invalid SSH peer port.")
    user = options.get("user")
    if user and (
        not isinstance(user, str) or user.startswith("-") or any(c.isspace() for c in user)
    ):
        raise ValueError("Invalid SSH peer user.")
    config = _run(["ssh", "-G", "-p", str(port), f"{user}@{host}" if user else host])
    if config.returncode:
        raise ValueError(
            "Cannot inspect initiating host SSH configuration: " + config.stderr[-1000:]
        )
    settings = dict(line.split(None, 1) for line in config.stdout.splitlines() if " " in line)
    # Special site policies need administrator review, not a guessed identity/path expansion.
    if settings.get("hostkeyalias", "none") != "none":
        raise ValueError("HostKeyAlias requires administrator-managed host trust.")
    if any(settings.get(k, "none") != "none" for k in ("proxycommand", "proxyjump")):
        raise ValueError("Proxy SSH routes require administrator-managed host trust.")
    address = settings.get("hostname", host)
    if (
        address.startswith("-")
        or any(c.isspace() for c in address)
        or any(c in address for c in "/,\\\0%")
    ):
        raise ValueError("Invalid effective SSH hostname.")
    port = int(settings.get("port", port))
    if not 1 <= port <= 65535:
        raise ValueError("Invalid effective SSH port.")
    token = address if port == 22 else f"[{address}]:{port}"
    users = shlex.split(settings.get("userknownhostsfile", ""))
    if not users or users[0] == "none":
        raise ValueError("SSH has no writable user known_hosts policy; ask the administrator.")

    def expand(raw: str) -> Path:
        if "%" in raw or "$" in raw:
            raise ValueError("Special known_hosts expansions require administrator-managed trust.")
        return Path(raw).expanduser()

    path = expand(users[0])
    _safe(path)
    found: list[tuple[str, str, bool]] = []
    for raw in [*users, *shlex.split(settings.get("globalknownhostsfile", ""))]:
        if raw == "none":
            continue
        file = expand(raw)
        if not file.exists():
            continue
        if not file.is_file() or file.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Unsafe or oversized existing SSH trust file.")
        lookup = _run(["ssh-keygen", "-F", token, "-f", str(file)])
        if lookup.returncode not in (0, 1):
            raise ValueError("Cannot inspect existing SSH host trust: " + lookup.stderr[-1000:])
        for line in lookup.stdout.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            marked = fields[0].startswith("@")
            if marked:
                fields = fields[1:]
            if len(fields) < 3:
                raise ValueError("Malformed existing host trust; no replacement is allowed.")
            found.append((fields[1], fields[2], marked))
    scan = _run(["ssh-keyscan", "-T", "5", "-p", str(port), "-t", "ed25519,ecdsa,rsa", address])
    keys = {}
    for line in scan.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("Invalid offered SSH host key.")
        kind, encoded = fields[1:]
        blob = base64.b64decode(encoded, validate=True)
        length = int.from_bytes(blob[:4], "big")
        if not 16 <= len(blob) <= 16384 or blob[4 : 4 + length] != kind.encode():
            raise ValueError("Invalid SSH public key encoding.")
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip(
            "="
        )
        keys[kind, encoded] = {"type": kind, "key": encoded, "fingerprint": fingerprint}
    if scan.returncode or not keys:
        raise ValueError("Cannot obtain offered host keys: " + scan.stderr[-1000:])
    # Any pre-existing different/marked identity is a hard refusal, even if another key matches.
    changed = any(marked or (kind, encoded) not in keys for kind, encoded, marked in found)
    state = "changed" if changed else "known" if found else "unknown"
    return {
        "host": address,
        "port": port,
        "lookup_host": token,
        "known_hosts_file": str(path),
        "state": state,
        "can_add": state == "unknown",
        "keys": sorted(keys.values(), key=lambda item: item["fingerprint"]),
        "notice": "Offered fingerprints are unverified network observations. Compare them with "
        "the administrator or another trusted channel before confirming. This only trusts the "
        "server identity; it does not authorize login or start a Dataset copy.",
    }


def approve(options: dict[str, Any]) -> dict[str, Any]:
    """Add only explicitly approved first-use keys, rechecking state under a native lock."""
    from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

    expected = options.get("fingerprints")
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(v, str) for v in expected)
    ):
        raise ValueError("Explicit expected SHA256 fingerprints are required before host trust.")
    preview = inspect(options)
    path = Path(preview["known_hosts_file"])

    def verify(value: dict[str, Any]) -> None:
        if set(expected) != {key["fingerprint"] for key in value["keys"]}:
            raise ValueError("Offered fingerprints changed since review; refusing host trust.")
        if value["state"] == "changed":
            raise ValueError(
                "Existing host key differs or is revoked/certified; never replacing it."
            )
        if options.get("expected_file") != value["known_hosts_file"]:
            raise ValueError("SSH known_hosts policy changed since preview; review again.")

    verify(preview)
    if preview["state"] == "known":
        return {**preview, "applied": False, "already_trusted": True}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _safe(path)
    with CrossProcessFileLock(
        path.parent / ".lambdaforge-host-trust.lock",
        shared=False,
        timeout_seconds=5,
        poll_interval_seconds=0.05,
    ):
        current = inspect(options)
        verify(current)
        if current["state"] == "known":
            return {**current, "applied": False, "already_trusted": True}
        original = path.read_bytes() if path.exists() else b""
        # Hash host labels without changing any original entries or permissions.
        salt = os.urandom(20)
        host = current["lookup_host"].encode()
        hashed = (
            "|1|"
            + base64.b64encode(salt).decode()
            + "|"
            + base64.b64encode(hmac.new(salt, host, hashlib.sha1).digest()).decode()
        )
        added = "".join(f"{hashed} {k['type']} {k['key']}\n" for k in current["keys"]).encode()
        fd, temporary = tempfile.mkstemp(prefix=".lf-host-trust-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as file:
                os.fchmod(
                    file.fileno(), stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
                )
                file.write(original + (b"\n" if original and not original.endswith(b"\n") else b""))
                file.write(added)
                file.flush()
                os.fsync(file.fileno())
            _safe(path)
            if (path.read_bytes() if path.exists() else b"") != original:
                raise ValueError("SSH trust file changed concurrently; review and retry.")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {**current, "applied": True, "state": "known", "can_add": False}


if __name__ == "__main__":
    options = json.loads(sys.argv[2])
    print(json.dumps(approve(options) if sys.argv[1] == "approve" else inspect(options)))
