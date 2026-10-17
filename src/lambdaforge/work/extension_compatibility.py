"""Extension-specific compatibility proofs; never alter global CodeIdentity/Run IDs."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from lambdaforge._version import VERSION
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.reproducibility.CodeIdentity import CodeIdentity
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.work.extension_design import exact, plain
from lambdaforge.work.recovery import read_owned_json


def code_review_pin(project: ProjectContext) -> str:
    """Pin preparation inputs, excluding only checksum-attested generated snapshots.

    This is a review TOCTOU guard, not a claim of scientific code equivalence.
    """
    from lambdaforge.work.StudySeriesStore import StudySeriesStore

    ignored = set()
    for snapshot in StudySeriesStore(project).snapshots():
        path = Path(snapshot["path"])
        if (
            path.resolve() != path or not path.is_relative_to(project.root)
            or not path.name.startswith(".lf-extension-") or path.suffix != ".yaml"
            or not path.is_file() or path.stat().st_size > 1024**2
            or hashlib.sha256(path.read_bytes()).hexdigest() != snapshot["sha256"]
        ):
            raise ValueError("Unsafe or altered native extension configuration snapshot.")
        ignored.add(path.resolve())
    digest = hashlib.sha256()
    try:
        root = Path(os.fsdecode(_git(project.root, "rev-parse", "--show-toplevel")).strip())
    except (OSError, subprocess.SubprocessError):
        root = None
    total = 0
    if root is not None:
        if _git(root, "submodule", "status", "--recursive").strip():
            raise ValueError("Extension review cannot pin submodule source trees.")
        digest.update(_git(root, "rev-parse", "HEAD"))
        digest.update(_git(root, "diff", "--binary", "HEAD"))
        for name in sorted(filter(None, _git(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).split(b"\0"))):
            path = root / os.fsdecode(name)
            if path in ignored:
                continue
            if path.resolve() != path or not path.is_file() or path.stat().st_size > 16 * 1024**2:
                raise ValueError(
                    "Untracked review inputs are unsafe/oversized; ignore outputs in Git."
                )
            total += path.stat().st_size
            if total > 64 * 1024**2:
                raise ValueError("Review source metadata exceeds 64 MiB; keep data outside source.")
            digest.update(name)
            digest.update(path.read_bytes())
    else:
        current = CodeIdentity.capture(project.root)
        if current.provider != "distribution" or not current.source_digest:
            raise ValueError("Current executable source cannot be pinned for an extension.")
        digest.update(exact(current.to_dict()).encode())
        files = [project.root / "pyproject.toml"]
        files.extend((project.root / "src").rglob("*"))
        for path in sorted(files):
            if path.is_symlink():
                raise ValueError("Current package sources cannot contain symbolic links.")
            if "__pycache__" in path.parts or path.suffix == ".pyc" or path.is_dir():
                continue
            if path.resolve() != path or not path.is_file() or path.stat().st_size > 16 * 1024**2:
                raise ValueError("Current package sources are unsafe or oversized.")
            total += path.stat().st_size
            if total > 64 * 1024**2:
                raise ValueError("Current package review sources exceed 64 MiB.")
            digest.update(path.relative_to(project.root).as_posix().encode())
            digest.update(path.read_bytes())
    from lambdaforge.controlplane.NativeEnvironment import NativeEnvironmentSpecification

    native = NativeEnvironmentSpecification.discover(project.root)
    if native is not None:
        if (
            native.source.resolve() != native.source
            or not native.source.is_relative_to(project.root)
        ):
            raise ValueError("Native dependency declaration is outside the pinned project.")
        digest.update(native.source.relative_to(project.root).as_posix().encode())
        digest.update(native.source.read_bytes())
    return "sha256:" + digest.hexdigest()


def review_extension_code(
    origin: Mapping[str, Any], project: ProjectContext, selected_yaml: Path, policy: str
) -> dict[str, Any]:
    """Allow a new code cohort, never turn acknowledgement into equivalence."""
    if policy not in {"strict", "revisioned"}:
        raise ValueError("Extension code policy must be strict or revisioned.")
    if policy == "strict":
        return verify_code(origin, project, selected_yaml)
    pin = code_review_pin(project)
    try:
        proof = verify_code(origin, project, selected_yaml)
    except ValueError as error:
        return {
            "kind": "revisioned-current", "origin": plain(origin),
            "current": CodeIdentity.capture(project.root).to_dict(), "review_pin": pin,
            "equivalent": False, "reason": str(error),
            "warning": "New code/runtime cohort: old Runs remain unchanged; "
                       "cross-cohort statistical pooling is disabled.",
        }
    return proof


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments), capture_output=True, timeout=10, check=True
    )
    if len(result.stdout) > 1024**2:
        raise ValueError("Extension Git compatibility evidence exceeds bounded policy.")
    return result.stdout


def verify_code(
    origin: Mapping[str, Any], project: ProjectContext, selected_yaml: Path
) -> dict[str, Any]:
    """Exact code or a clean-origin/same-HEAD/selected-YAML-only Git proof.

    An acknowledgement does not waive this check. An arbitrary dirty origin cannot be
    reconstructed after editing; distribution identity requires actual source digests.
    """
    current = CodeIdentity.capture(project.root).to_dict()
    provider = origin.get("provider")
    if provider not in {"git", "distribution"}:
        raise ValueError(
            "Origin code provenance cannot be verified for extension; use a new Study."
        )
    if provider == "distribution" and not origin.get("source_digest"):
        raise ValueError("Non-Git extension requires the native executable source digest.")
    if exact(origin) == exact(current):
        return {"kind": "exact", "origin": plain(origin), "current": current}
    if (
        provider != "git"
        or current.get("provider") != "git"
        or origin.get("dirty")
        or origin.get("revision") != current.get("revision")
    ):
        raise ValueError(
            "Changed/unverifiable executable code or dirty origin: extension cannot reuse "
            "evidence. Human acknowledgement is not a code-equivalence proof."
        )
    root = Path(os.fsdecode(_git(project.root, "rev-parse", "--show-toplevel")).strip()).resolve()
    if not selected_yaml.is_relative_to(root) or selected_yaml.suffix.lower() not in {
        ".yaml",
        ".yml",
    }:
        raise ValueError("The authorized scientific YAML must belong to the original Git project.")
    changed = {
        os.fsdecode(item)
        for item in (
            _git(root, "diff", "--name-only", "-z", "HEAD")
            + _git(root, "ls-files", "--others", "--exclude-standard", "-z")
        ).split(b"\0")
        if item
    }
    # Only exact, journal-attested non-executable snapshots can be ignored by this
    # extension-specific proof. Global CodeIdentity and native Run fingerprints do not change.
    from lambdaforge.work.StudySeriesStore import StudySeriesStore

    for snapshot in StudySeriesStore(project).snapshots():
        path = Path(snapshot["path"])
        if (
            path.resolve() != path
            or not path.is_relative_to(project.root)
            or path.suffix != ".yaml"
            or not path.name.startswith(".lf-extension-")
            or not path.is_file()
            or path.stat().st_size > 1024**2
        ):
            raise ValueError("Unsafe or missing native extension configuration snapshot.")
        import hashlib

        if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot["sha256"]:
            raise ValueError("Native extension configuration snapshot was altered.")
        changed.discard(path.relative_to(root).as_posix())
    relative = selected_yaml.relative_to(root).as_posix()
    if changed != {relative}:
        raise ValueError(
            "Executable/dependency changes or additional unknown Git changes prevent extension. "
            "Only the selected scientific YAML may differ from the clean origin HEAD."
        )
    if _git(root, "submodule", "status", "--recursive").strip():
        raise ValueError("Submodule code equivalence requires a declared extension proof.")
    return {
        "kind": "authorized-yaml-only",
        "origin": plain(origin),
        "current": current,
        "same_head": origin["revision"],
        "authorized_changed_path": relative,
        "proof": "clean origin; same HEAD; all tracked/untracked changes are selected YAML",
    }


def verify_local_runtime(runs: Sequence[Mapping[str, Any]], execution: Path) -> dict[str, Any]:
    """Read small saved environment metadata, never import Torch, load assets or infer precision.

    This helper accepts same-interpreter CPU evidence only. Managed cluster/GPU attestation
    uses the separate retained-prefix and worker-side numerical contract in extension_runtime.
    """
    references = []
    fingerprints = set()
    for run in runs:
        directory = Path(str(run["run_dir"]))
        if directory.resolve() != directory or not directory.is_relative_to(execution / "runs"):
            raise ValueError("Historical Run is outside the original owned Execution.")
        relative = Path(str(run.get("environment_manifest", "environment.json")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe historical environment metadata path.")
        path = directory / relative
        if path.resolve() != path or path.stat().st_size > 256 * 1024:
            raise ValueError("Unsafe or oversized environment compatibility metadata.")
        environment = read_owned_json(path)
        python = environment.get("python", {})
        if (
            python.get("version") != platform.python_version()
            or python.get("implementation") != platform.python_implementation()
            or Path(str(python.get("executable", ""))).resolve() != Path(sys.executable).resolve()
            or environment.get("lambdaforge_version") != VERSION
        ):
            raise ValueError("Different/unverifiable Python/framework runtime for extension.")
        hardware = environment.get("platform", {})
        if any(
            hardware.get(key) != value
            for key, value in (
                ("system", platform.system()),
                ("machine", platform.machine()),
                ("processor", platform.processor()),
            )
        ):
            raise ValueError("CPU/platform compatibility cannot be verified for extension.")
        packages = environment.get("packages")
        if not isinstance(packages, Mapping) or not packages:
            raise ValueError("Historical dependency provenance is missing.")
        for name, expected in packages.items():
            try:
                installed = version(str(name))
            except PackageNotFoundError:
                installed = None
            if installed != expected:
                raise ValueError(f"Dependency {name!r} changed; do not mix incompatible evidence.")
        if environment.get("plugins") or environment.get("external_tools"):
            raise ValueError("Plugin/external-tool runtime compatibility is not integrated yet.")
        requested = run.get("resources", {}).get("requested", {})
        if requested.get("gpu", requested.get("gpu_count", 0)):
            raise ValueError("GPU/hardware compatibility requires a provider extension contract.")
        fingerprint = ScientificIdentity.from_payload(
            {
                "python": {
                    key: python.get(key) for key in ("version", "implementation", "executable")
                },
                "packages": plain(packages),
                "platform": {key: hardware.get(key) for key in ("system", "machine", "processor")},
                "framework": VERSION,
            }
        ).digest
        fingerprints.add(fingerprint)
        references.append(
            {
                "execution_id": run["execution_id"],
                "run_id": run["run_id"],
                "attempt_id": run["attempt_id"],
                "environment_fingerprint": fingerprint,
            }
        )
    if len(fingerprints) != 1:
        raise ValueError("Historical evidence has mixed/unverifiable runtime identities.")
    return {
        "scope": "same-interpreter CPU; native recorded package/platform metadata",
        "fingerprint": next(iter(fingerprints)),
        "references": references,
    }
