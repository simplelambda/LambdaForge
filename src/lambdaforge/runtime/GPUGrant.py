"""Parse authoritative visibility data; never execute shell output or infer grants from activity."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence

_TOKEN = re.compile(r"(?:\d+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9-]+(?:/\d+/\d+)?)\Z")


def parse_visibility(output: str) -> tuple[str, ...] | None:
    """Single CSV line or explicit CUDA_VISIBLE_DEVICES shell assignment.

    Unexpected/ambiguous output is unknown, never proof of revocation. Shell is parsed,
    not evaluated. Only an explicit empty/disabled assignment proves an empty grant.
    """
    if len(output) > 16384:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    assignments = [
        line for line in lines if re.match(r"^(?:export\s+)?CUDA_VISIBLE_DEVICES=", line)
    ]
    if assignments:
        if len(assignments) != 1:
            return None
        try:
            words = shlex.split(assignments[0].removesuffix(";").strip())
        except ValueError:
            return None
        if words[:1] == ["export"]:
            words = words[1:]
        if len(words) != 1 or not words[0].startswith("CUDA_VISIBLE_DEVICES="):
            return None
        value = words[0].split("=", 1)[1]
    elif len(lines) == 1:
        value = lines[0]
    else:
        return None
    if value.lower() in {"", "-1", "none", "void", "nodevfiles"}:
        return ()
    tokens = tuple(part.strip() for part in value.split(","))
    if len(tokens) != len(set(tokens)) or not all(_TOKEN.fullmatch(token) for token in tokens):
        return None
    return tokens


def intersect_visibility(
    initial: Sequence[str],
    reported: Sequence[str],
    *,
    uuid_by_token: Mapping[str, str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> frozenset[str] | None:
    """Translate only the original grant; a different output encoding is not revocation.

    Numeric provider tokens are never equated to NVML indices. Cross-format comparison
    needs pinned native UUID evidence. Unknown/ambiguous mappings return None, not empty.
    """
    if not reported:
        return frozenset()
    pinned, names = uuid_by_token or {}, aliases or {}
    selected: set[str] = set()
    for value in reported:
        if value.isdigit():
            if any(not token.isdigit() for token in initial):
                return None  # A physical ordinal cannot be inferred from a UUID grant.
            if value in initial:
                selected.add(value)
            continue
        matched = []
        for token in initial:
            uuid = pinned.get(token, token if not token.isdigit() else None)
            if uuid is None:
                return None
            if uuid.startswith(value) or value in names.get(uuid, ()):
                matched.append(token)
        if len(matched) > 1:
            return None
        selected.update(matched)
    return frozenset(selected)
