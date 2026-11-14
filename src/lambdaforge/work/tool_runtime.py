"""Public isolated-runtime handles: prepared packages, explicit files, no nested Work runner."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.NativeLibraryPolicy import NativeLibraryPolicy
from lambdaforge.controlplane.TlsTrust import TlsTrust
from lambdaforge.controlplane.ToolRuntimeService import (
    validate_verification,
    verification_code,
    verification_policy,
)
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.tools import ToolResult, ToolService


class IsolatedToolRuntime:
    """Execute a named immutable Python tool with the enclosing Work's grants and logs."""

    def __init__(self, binding: Mapping[str, Any], owner: ToolService) -> None:
        self.binding = dict(binding)
        self.name = str(binding["name"])
        self.python = Path(str(binding["python"])).absolute()
        self.prefix = self.python.parent.parent
        self.cache_root = self.prefix.parent.parent
        self.owner = owner
        self._verified = False
        environment = NativeLibraryPolicy.isolated_environment(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
            environment.pop(key, None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["VIRTUAL_ENV"] = str(self.prefix)
        trust = TlsTrust.from_mapping(binding.get("tls_trust"))
        if trust is not None:
            for assignment in trust.assignments():
                key, value = assignment.split("=", 1)
                environment[key] = value
        path = os.pathsep.join((str(self.python.parent), "/usr/local/bin", "/usr/bin", "/bin"))
        self._runner = ToolService(
            owner._log,
            max_threads=owner._max_threads,
            stop_check=owner._stop_check,
            environment=environment,
            search_path=path,
        )

    @classmethod
    def load(cls, name: str, owner: ToolService) -> IsolatedToolRuntime:
        configured = os.environ.get("LAMBDAFORGE_TOOL_RUNTIMES_PATH")
        if configured:
            source = Path(configured)
        else:
            from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
            from lambdaforge.ProjectContext import ProjectContext

            project = ProjectContext.discover(owner._source_dir or Path.cwd())
            profile = ClusterCatalog.load(project=project).get("local")
            assert profile.storage is not None
            source = Path(profile.storage.state_root) / "tool-runtimes.json"
        cluster = os.environ.get("LAMBDAFORGE_CLUSTER", "local")
        hint = (
            f"Run lf clusters bootstrap {cluster} --project . after declaring tool_runtimes.{name}."
        )
        if source.is_symlink() or not source.is_file() or source.stat().st_size > 1024**2:
            raise RuntimeError(f"Isolated tool runtime {name!r} is not prepared. {hint}")
        document = json.loads(source.read_text())
        if document.get("tool_registry_version") != 1 or name not in document.get("runtimes", {}):
            raise RuntimeError(f"Missing declared isolated tool runtime {name!r}. {hint}")
        binding = document["runtimes"][name]
        if binding.get("name") != name:
            raise ValueError("Isolated tool binding identity is corrupt.")
        if not configured:
            from lambdaforge.controlplane.ToolRuntimeSpecification import ToolRuntimeSpecification

            specs = {spec.name: spec for spec in ToolRuntimeSpecification.discover(project.root)}
            spec = specs.get(name)
            if spec is None or spec.contract_id != binding["contract_id"]:
                raise RuntimeError(f"Tool declaration changed or is missing. {hint}")
            if {k: v["sha256"] for k, v in spec.descriptor()["entrypoints"].items()} != {
                k: v["sha256"] for k, v in binding["entrypoints"].items()
            }:
                raise RuntimeError(f"Tool entrypoint source changed. {hint}")
        instance = cls(binding, owner)
        instance._check_receipt()
        with owner._lock:
            owner._runtime_provenance[name] = {
                "name": f"isolated:{name}",
                "path": str(instance.python),
                "version": str(binding["policy"]["python"]),
                "runtime_identity": str(binding["environment_id"]),
                "resolution_identity": str(binding["resolution_id"]),
                "declaration_identity": str(binding["contract_id"]),
            }
        return instance

    def _check_receipt(self) -> None:
        if self.prefix.parent.name != "environments" or not self.prefix.name.startswith(
            "env-tool-"
        ):
            raise ValueError("Isolated runtime interpreter is outside a native tool environment.")
        if any(path.is_symlink() for path in (self.prefix, *self.prefix.parents)):
            raise ValueError("Isolated runtime prefix must not traverse symlinks.")
        marker = self.prefix / ".lambdaforge-environment.json"
        if marker.is_symlink() or not marker.is_file() or not self.python.is_file():
            raise RuntimeError(
                f"Isolated runtime {self.name!r} is missing; bootstrap it before Work execution."
            )
        record = json.loads(marker.read_text())
        if (
            record.get("layer_kind") != "isolated-tool"
            or record.get("environment_id") != self.binding["environment_id"]
            or record.get("tool_contract") != self.binding["contract_id"]
            or record.get("dependency_resolution_id") != self.binding["resolution_id"]
        ):
            raise ValueError("Isolated runtime receipt does not match its owned binding.")

    @contextmanager
    def _lease(self) -> Iterator[None]:
        environment = CrossProcessFileLock(
            self.cache_root / f".tool-runtime-use-{self.prefix.name}.lock",
            shared=True,
            timeout_seconds=10,
            poll_interval_seconds=0.05,
        )
        code_id = self.binding["entrypoint_bundle_id"]
        if (
            not isinstance(code_id, str)
            or not code_id.startswith("tool-code-")
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in code_id)
        ):
            raise ValueError("Unsafe isolated tool code bundle identity.")
        code = CrossProcessFileLock(
            self.cache_root / f".tool-runtime-use-{code_id}.lock",
            shared=True,
            timeout_seconds=10,
            poll_interval_seconds=0.05,
        )
        with environment, code:
            yield

    def entrypoint(self, name: str) -> Path:
        """Return an explicitly staged and hash-verified small Python entrypoint."""
        value = self.binding["entrypoints"].get(name)
        if value is None:
            raise ValueError(f"Tool {self.name!r} has no declared entrypoint {name!r}.")
        path = Path(value["path"])
        expected = self.cache_root / "bundles" / self.binding["entrypoint_bundle_id"]
        if path.parent != expected or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Tool entrypoint escaped its immutable code bundle.")
        if (
            not path.is_file()
            or path.stat().st_size > 1024**2
            or hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]
        ):
            raise ValueError(
                "Tool entrypoint bytes are missing or changed; bootstrap/re-stage declared code."
            )
        with self.owner._lock:
            self.owner._runtime_provenance[f"{self.name}:{name}"] = {
                "name": f"isolated:{self.name}:{name}",
                "path": str(path),
                "version": value["sha256"],
                "runtime_identity": str(self.binding["environment_id"]),
            }
        return path

    def preflight(self) -> ToolResult:
        """Verify imports/ABI on granted logical GPUs; never install weights or packages."""
        with self._lease():
            self._check_receipt()
            policy = self.binding["policy"]
            cuda = policy.get("torch_cuda", "cpu") != "cpu"
            result = self._runner.run(
                (
                    self.python,
                    "-I",
                    "-c",
                    verification_code(),
                    json.dumps(verification_policy(policy)),
                    str(cuda).lower(),
                ),
                name=f"{self.name}:preflight",
            )
            validate_verification(policy, result.stdout)
            self._verified = True
            return result

    def run(
        self,
        command: Sequence[str | os.PathLike[str]],
        *,
        threads: int | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> ToolResult:
        """Run argv; use ``python`` and declared entrypoints for explicit file exchange."""
        if not command:
            raise ValueError("Tool command cannot be empty.")
        with self._lease():
            self._check_receipt()
            if not self._verified:
                self.preflight()
            values = tuple(os.fspath(value) for value in command)
            if values[0] == "python" or Path(values[0]).absolute() == self.python:
                values = (str(self.python), "-I", *values[1:])
            else:
                candidate = self.python.parent / values[0]
                if Path(values[0]).name != values[0] or not candidate.is_file():
                    raise ValueError(
                        "Tool executables must be named members of the isolated runtime bin."
                    )
                values = (str(candidate), *values[1:])
            return self._runner.run(
                values,
                name=f"{self.name}:{Path(values[0]).name}",
                threads=threads,
                cwd=cwd,
                env=env,
                timeout=timeout,
                check=check,
            )
