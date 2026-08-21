"""Boundary classification, job-failure context and durable diagnostic records."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.diagnostics.models import (
    ErrorCategory,
    ErrorDiagnostic,
    LambdaForgeError,
    RetryDisposition,
    diagnostic,
)
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


@dataclass(frozen=True, slots=True)
class DiagnosticContext:
    """Retain safe invocation context independently of argparse command implementations."""

    arguments: tuple[str, ...]
    operation: str
    cluster: str | None = None
    json_output: bool = False
    debug: bool = False
    verbose: bool = False

    @property
    def command(self) -> str:
        return shlex.join(("lf", *self.arguments))

    @classmethod
    def from_argv(cls, arguments: Sequence[str]) -> DiagnosticContext:
        """Extract only non-secret routing facts before command parsing."""
        values = tuple(str(value) for value in arguments)
        cluster = None
        if "--on" in values:
            index = values.index("--on")
            if index + 1 < len(values):
                cluster = values[index + 1]
        elif len(values) >= 3 and values[0] == "clusters":
            cluster = values[2]
        operation = " ".join(value for value in values[:3] if not value.startswith("-"))
        return cls(
            values,
            operation or "LambdaForge command",
            cluster,
            "--json" in values,
            "--debug" in values,
            "--verbose" in values,
        )


class DiagnosticClassifier:
    """Translate only semantics understood at the operational boundary."""

    _JOB = re.compile(r"\bjob-\d{14}-[a-z0-9]+\b", re.I)

    def classify(self, error: BaseException, context: DiagnosticContext) -> ErrorDiagnostic:
        """Return an actionable diagnosis while leaving unknown failures explicitly internal."""
        if isinstance(error, LambdaForgeError):
            return error.diagnostic
        message = SecretRedactor.redact(error)
        lowered = message.lower()
        job_match = self._JOB.search(message)
        job_id = job_match.group(0) if job_match else None
        commands = self._diagnostic_commands(context, job_id)

        from lambdaforge.controlplane.CudaCompatibilityResolver import (
            NoCompatibleTorchWheelError,
        )
        from lambdaforge.controlplane.python_runtime import NoCompatiblePythonRuntimeError
        from lambdaforge.controlplane.RemoteCommandTimeout import RemoteCommandTimeout
        from lambdaforge.data.errors import (
            AmbiguousDatasetVersionError,
            DatasetRegistryCorruptionError,
            DatasetResolutionError,
            InvalidDatasetBuildError,
            MissingDatasetPlacementError,
            MissingDatasetRecipeError,
            OfflineClusterError,
            UnknownDatasetError,
            UnsafeDatasetOperationError,
        )

        if isinstance(error, NoCompatiblePythonRuntimeError):
            strategy = getattr(error, "strategy", "unknown")
            detected = tuple(getattr(error, "detected", ()))
            required = tuple(getattr(error, "requirements", ()))
            candidates = tuple(getattr(error, "candidates", ()))
            cluster = context.cluster or getattr(error, "cluster", None)
            fixes = [
                "Allow LambdaForge to prepare a compatible user-space Python runtime."
                if strategy == "existing"
                else "Review the incompatible Python constraints or configure an explicit runtime."
            ]
            environment_commands: list[tuple[str, str]] = []
            if cluster:
                environment_commands.extend(
                    (
                        (
                            "Enable automatic runtime provisioning",
                            f"lf clusters set {shlex.quote(cluster)} python.strategy auto",
                        ),
                        ("Bootstrap", f"lf clusters bootstrap {shlex.quote(cluster)}"),
                        ("Diagnose", f"lf doctor --on {shlex.quote(cluster)}"),
                    )
                )
            else:
                environment_commands.extend(commands)
            return diagnostic(
                ErrorCategory.ENVIRONMENT,
                "A compatible Python runtime could not be prepared.",
                "The selected cluster cannot currently run this LambdaForge project.",
                reason=(
                    "The active strategy only permits existing interpreters, so LambdaForge did "
                    "not install a managed runtime."
                    if strategy == "existing"
                    else "No permitted candidate satisfied all declared Python constraints."
                ),
                impact=("Bootstrap stopped before a scientific job was submitted.",),
                fixes=fixes,
                commands=environment_commands,
                context={
                    "cluster": cluster,
                    "detected": detected or "no compatible interpreter",
                    "required": required,
                    "candidates_considered": candidates,
                    "runtime_strategy": strategy,
                },
                operation=context.operation,
            )
        if isinstance(error, NoCompatibleTorchWheelError):
            return diagnostic(
                ErrorCategory.ENVIRONMENT,
                "No compatible official PyTorch build is available.",
                message,
                reason="The Python, platform and selected CUDA channel have no common wheel.",
                impact=("Environment preparation stopped; no scientific job was submitted.",),
                fixes=("Use another compatible Python version or review the configured channel.",),
                commands=commands,
                context={"cluster": context.cluster},
                operation=context.operation,
            )
        if isinstance(error, RemoteCommandTimeout):
            return diagnostic(
                ErrorCategory.CONNECTION,
                "A remote command exceeded its configured deadline.",
                "LambdaForge stopped waiting for the control command.",
                reason=(
                    f"The command timeout was {error.timeout_seconds:g}s. This is separate from "
                    "the SSH connection timeout and the scientific job runtime."
                ),
                impact=(
                    "The remote job state may be unchanged; LambdaForge did not assume failure.",
                ),
                fixes=(
                    "Check the cluster and increase command_timeout only if the probe is slow.",
                ),
                commands=commands,
                context={"cluster": context.cluster, "timeout_seconds": error.timeout_seconds},
                retryable=RetryDisposition.IMMEDIATE,
                operation=context.operation,
                job_id=job_id,
            )
        if isinstance(error, MissingDatasetPlacementError):
            selector = error.selector
            cluster = error.cluster
            return diagnostic(
                ErrorCategory.DATA,
                f"Dataset {selector!r} has no physical copy on {cluster!r}.",
                "The immutable dataset is registered, but it is not available where requested.",
                reason=(
                    "Training and dataset operations require a verified physical copy on the "
                    "selected cluster."
                ),
                impact=("No dataset bytes were guessed, copied or consumed.",),
                fixes=("Materialize the exact registered version on the target cluster.",),
                commands=(
                    (
                        "Materialize dataset",
                        f"lf datasets materialize {shlex.quote(selector)} --on "
                        f"{shlex.quote(cluster)} --apply",
                    ),
                ),
                context={
                    "dataset": selector,
                    "target_cluster": cluster,
                    "available_placements": error.available,
                },
                operation=context.operation,
            )
        if isinstance(error, AmbiguousDatasetVersionError):
            return diagnostic(
                ErrorCategory.DATA,
                f"Dataset {error.name!r} has more than one matching version.",
                "LambdaForge refused to guess which scientific data should be used.",
                reason="Unversioned resolution would make the experiment identity ambiguous.",
                impact=("No dataset was selected and no computation was started.",),
                fixes=("Use one exact name@version selector from the available versions.",),
                commands=tuple(
                    (
                        f"Inspect {version}",
                        f"lf datasets show {shlex.quote(f'{error.name}@{version}')}",
                    )
                    for version in error.versions
                ),
                context={"dataset": error.name, "versions": error.versions},
                operation=context.operation,
            )
        if isinstance(error, UnknownDatasetError):
            return diagnostic(
                ErrorCategory.DATA,
                f"Dataset {error.selector!r} is not registered.",
                "No managed DatasetVersion matches the requested selector.",
                reason="LambdaForge only resolves datasets with auditable identity metadata.",
                impact=("No untracked path was substituted and no computation was started.",),
                fixes=("List known versions or build/add the intended dataset explicitly.",),
                commands=(("List datasets", "lf datasets list"),),
                context={"dataset": error.selector, "known": error.known},
                operation=context.operation,
            )
        if isinstance(error, MissingDatasetRecipeError):
            return diagnostic(
                ErrorCategory.CONFIGURATION,
                f"No DatasetRecipe is available for {error.selector!r}.",
                "LambdaForge cannot build an unregistered version without its source recipe.",
                reason="A kind: dataset YAML recipe defines the reproducible build DAG.",
                impact=("No build job was submitted.",),
                fixes=("Create or discover a kind: dataset recipe, then build it by name.",),
                commands=(("List dataset configs", "lf configs list"),),
                context={"dataset": error.selector, "known_recipes": error.known},
                operation=context.operation,
            )
        if isinstance(error, InvalidDatasetBuildError):
            if "different immutable identity" in lowered:
                return diagnostic(
                    ErrorCategory.OPERATION_REFUSED,
                    "LambdaForge refused to overwrite an immutable dataset version.",
                    message,
                    reason=(
                        "A published name@version permanently identifies one content hash; "
                        "changing its bytes would make earlier results ambiguous."
                    ),
                    impact=("The existing version and registry record were left unchanged.",),
                    fixes=("Publish changed scientific content under a new dataset version.",),
                    commands=(("Inspect existing versions", "lf datasets list"),),
                    context={"cluster": context.cluster},
                    operation=context.operation,
                )
            return self._known_data(error, context, "Dataset build evidence is invalid.", commands)
        if isinstance(error, UnsafeDatasetOperationError):
            return diagnostic(
                ErrorCategory.OPERATION_REFUSED,
                "LambdaForge intentionally refused an unsafe dataset operation.",
                message,
                reason=(
                    "The requested mutation violates a dataset immutability or path-safety rule."
                ),
                impact=("No protected dataset bytes were changed.",),
                fixes=("Review the exact dataset/version and use a safe new version or location.",),
                commands=commands,
                context={"cluster": context.cluster},
                operation=context.operation,
            )
        if isinstance(error, OfflineClusterError):
            return diagnostic(
                ErrorCategory.CONNECTION,
                "The selected cluster is unavailable.",
                message,
                reason="LambdaForge could not complete the transport probe.",
                impact=("No new remote computation was started by this command.",),
                fixes=("Verify network access, SSH configuration and cluster availability.",),
                commands=commands,
                context={"cluster": context.cluster},
                retryable=RetryDisposition.IMMEDIATE,
                operation=context.operation,
            )
        if isinstance(error, DatasetRegistryCorruptionError):
            return diagnostic(
                ErrorCategory.DATA,
                "Dataset registry state is corrupt or unreadable.",
                message,
                reason=(
                    "An existing registry cannot be interpreted safely and must not be treated "
                    "as an empty index."
                ),
                impact=(
                    "No dataset absence was inferred.",
                    "No registry or dataset bytes were modified.",
                ),
                fixes=(
                    "Inspect or restore the registry file before retrying.",
                    "Do not replace it with an empty file if physical placements still exist.",
                ),
                commands=commands,
                context={
                    "cluster": context.cluster,
                    "registry": getattr(error, "path", None),
                },
                operation=context.operation,
            )
        if isinstance(error, DatasetResolutionError):
            return self._known_data(error, context, "Dataset operation failed.", commands)

        overwrite_refused = (
            isinstance(error, FileExistsError)
            or "refusing to overwrite" in lowered
            or "must not overwrite" in lowered
        )
        if overwrite_refused:
            refusal_commands: list[tuple[str, str]] = []
            if (
                "must not overwrite the source" not in lowered
                and context.arguments
                and "--force" not in context.arguments
            ):
                refusal_commands.append(
                    (
                        "Retry only if replacement is intentional",
                        shlex.join(("lf", *context.arguments, "--force")),
                    )
                )
            elif context.arguments:
                preview_arguments: list[str] = []
                skip_next = False
                for value in context.arguments:
                    if skip_next:
                        skip_next = False
                        continue
                    if value == "--output":
                        skip_next = True
                        continue
                    if value != "--force":
                        preview_arguments.append(value)
                refusal_commands.append(
                    (
                        "Preview without writing",
                        shlex.join(("lf", *preview_arguments)),
                    )
                )
            return diagnostic(
                ErrorCategory.OPERATION_REFUSED,
                "LambdaForge intentionally refused to overwrite an existing file.",
                message,
                reason="Implicit replacement could destroy user-authored configuration or output.",
                impact=("The existing file was left unchanged.",),
                fixes=(
                    "Choose another output path, or use --force only after reviewing the target.",
                ),
                commands=refusal_commands,
                context={"cluster": context.cluster},
                operation=context.operation,
            )

        if lowered.startswith("invalid command line:"):
            return diagnostic(
                ErrorCategory.CONFIGURATION,
                "The command-line arguments are invalid.",
                message.removeprefix("Invalid command line: ").strip(),
                reason=(
                    "The requested command or option combination is not part of the CLI grammar."
                ),
                impact=("No project configuration was loaded and no job was submitted.",),
                fixes=("Review the command help and correct the reported argument.",),
                commands=(("Show command help", "lf --help"),),
                context={"operation": context.operation},
                operation=context.operation,
            )

        if isinstance(error, (yaml.YAMLError, json.JSONDecodeError)) or any(
            marker in lowered
            for marker in ("invalid workflow", "invalid dataset recipe", "schema validation")
        ):
            source = next(
                (
                    value
                    for value in context.arguments[1:]
                    if not value.startswith("-")
                    and Path(value).suffix.lower() in {".yaml", ".yml", ".json"}
                ),
                None,
            )
            validation_commands = (
                (("Validate again", shlex.join(("lf", "validate", source))),)
                if source is not None
                else commands
            )
            return diagnostic(
                ErrorCategory.VALIDATION,
                "The configuration does not satisfy its declared contract.",
                message,
                reason="The file could be read, but one or more values are invalid.",
                impact=("Validation stopped before a job was submitted.",),
                fixes=("Correct the reported file/field and validate it again.",),
                commands=validation_commands,
                context={"cluster": context.cluster, "file": source},
                operation=context.operation,
            )
        storage_failure = (
            isinstance(error, PermissionError)
            or any(marker in lowered for marker in ("read-only", "no space", "disk full"))
            or ("quota" in lowered and any(marker in lowered for marker in ("disk", "storage")))
            or (
                "permission" in lowered
                and any(
                    marker in lowered
                    for marker in ("path", "filesystem", "storage", "publish", "dataset")
                )
            )
        )
        if storage_failure:
            return diagnostic(
                ErrorCategory.STORAGE,
                "LambdaForge cannot access the required storage.",
                message,
                reason="The operating system or remote filesystem rejected the requested access.",
                impact=("The operation stopped without publishing incomplete output.",),
                fixes=("Check the reported path, permissions, quota and available capacity.",),
                commands=commands,
                context={"cluster": context.cluster},
                operation=context.operation,
            )
        if any(
            marker in lowered
            for marker in (
                "authentication",
                "host key",
                "credential",
                "permission denied (publickey",
            )
        ):
            authentication_commands: list[tuple[str, str]] = []
            if context.cluster:
                authentication_commands.extend(
                    (
                        (
                            "Configure credential",
                            f"lf clusters credentials set {shlex.quote(context.cluster)}",
                        ),
                        ("Diagnose", f"lf doctor --on {shlex.quote(context.cluster)}"),
                    )
                )
            else:
                authentication_commands.extend(commands)
            return diagnostic(
                ErrorCategory.AUTHENTICATION,
                "Cluster authentication failed.",
                message,
                reason="The configured credential or host-key policy did not authorize access.",
                impact=("No remote command or scientific job was started.",),
                fixes=(
                    "Verify the credential reference and known_hosts entry; no secret is shown.",
                ),
                commands=authentication_commands,
                context={"cluster": context.cluster},
                retryable=RetryDisposition.AFTER_FIX,
                operation=context.operation,
            )
        if any(
            marker in lowered
            for marker in ("ssh", "scp", "sftp", "connection refused", "unreachable", "network")
        ):
            return diagnostic(
                ErrorCategory.CONNECTION,
                "LambdaForge could not complete the cluster connection operation.",
                message,
                reason="The SSH/network transport failed before the requested operation completed.",
                impact=("Remote state was not guessed; inspect it before retrying mutations.",),
                fixes=("Check the host, network, SSH agent/configuration and timeout policy.",),
                commands=commands,
                context={"cluster": context.cluster},
                retryable=RetryDisposition.IMMEDIATE,
                operation=context.operation,
            )
        if any(
            marker in lowered
            for marker in (
                "cuda",
                "pytorch",
                "python runtime",
                "requires-python",
                "micromamba",
                "compatible wheel",
                "package dependency",
                "pip install",
            )
        ):
            return diagnostic(
                ErrorCategory.ENVIRONMENT,
                "The execution environment could not be prepared.",
                message,
                reason="A required Python, package, PyTorch or CUDA compatibility check failed.",
                impact=("No scientific job was submitted with an unverified environment.",),
                fixes=("Run doctor and review the managed environment/bootstrap plan.",),
                commands=commands,
                context={"cluster": context.cluster},
                operation=context.operation,
            )
        if any(
            marker in lowered
            for marker in (
                "out of memory",
                "oom",
                "gpu resources",
                "insufficient memory",
                "qos",
                "quota",
            )
        ) or any(
            marker in lowered
            for marker in ("invalid partition", "account required", "sbatch: error")
        ):
            resource_commands = list(commands)
            if context.cluster:
                resource_commands.insert(
                    0,
                    (
                        "Inspect available resources",
                        f"lf resources --on {shlex.quote(context.cluster)}",
                    ),
                )
            return diagnostic(
                ErrorCategory.RESOURCE,
                "The requested compute resources cannot be satisfied.",
                message,
                reason="The available allocation or scheduler policy cannot meet the request.",
                impact=("The requested computation could not proceed with these resources.",),
                fixes=(
                    "Reduce the request or select a cluster/partition with sufficient capacity.",
                ),
                commands=resource_commands,
                context={"cluster": context.cluster},
                operation=context.operation,
                job_id=job_id,
            )
        if isinstance(error, subprocess.CalledProcessError):
            stderr = SecretRedactor.redact(error.stderr or error.stdout or "")
            command = (
                str(error.cmd)
                if isinstance(error.cmd, str)
                else shlex.join(str(value) for value in error.cmd)
            )
            return diagnostic(
                ErrorCategory.EXECUTION,
                "A required subprocess exited unsuccessfully.",
                stderr.strip() or f"The process exited with status {error.returncode}.",
                reason=(
                    "LambdaForge cannot safely infer a more specific cause from this process "
                    "result. The sanitized command and exit status are retained for diagnosis."
                ),
                impact=("The containing operation did not publish a successful result.",),
                fixes=("Inspect the original stderr and the relevant job or diagnostic record.",),
                commands=commands,
                context={
                    "cluster": context.cluster,
                    "command": SecretRedactor.redact(command),
                    "returncode": error.returncode,
                },
                retryable=RetryDisposition.UNKNOWN,
                operation=context.operation,
                job_id=job_id,
            )
        if isinstance(error, TimeoutError):
            return diagnostic(
                ErrorCategory.CONNECTION if context.cluster else ErrorCategory.EXECUTION,
                "The operation exceeded its deadline.",
                message or "LambdaForge stopped waiting after the configured timeout.",
                reason=(
                    "The underlying operation did not provide enough evidence to classify this "
                    "as a connection, scheduler or scientific runtime timeout more precisely."
                ),
                impact=("The remote state was not guessed; inspect it before retrying.",),
                fixes=("Inspect job/cluster state and review the configured timeout.",),
                commands=commands,
                context={"cluster": context.cluster},
                retryable=RetryDisposition.IMMEDIATE,
                operation=context.operation,
                job_id=job_id,
            )
        if "not supported by this scheduler" in lowered:
            return diagnostic(
                ErrorCategory.OPERATION_REFUSED,
                "The selected scheduler does not support this lifecycle operation.",
                message,
                reason="LambdaForge checked the provider capability before changing job state.",
                impact=("The job and its allocated resources were left unchanged.",),
                fixes=("Use a supported lifecycle command or the cluster scheduler directly.",),
                commands=commands,
                context={"cluster": context.cluster, "job": job_id},
                operation=context.operation,
                job_id=job_id,
            )
        if any(
            marker in lowered
            for marker in (
                "remote bundle cache",
                "job workspace",
                "dataset staging",
                "atomic dataset publish",
                "remote dataset placement",
                "remote dataset registration",
            )
        ):
            return diagnostic(
                ErrorCategory.STORAGE,
                "LambdaForge could not prepare or publish required storage.",
                message,
                reason="A verified staging, workspace, cache or atomic publication step failed.",
                impact=("Incomplete data was not registered as a successful result.",),
                fixes=("Check the target path, permissions, quota and available disk space.",),
                commands=commands,
                context={"cluster": context.cluster},
                operation=context.operation,
                job_id=job_id,
            )
        if "dataset" in lowered and any(
            marker in lowered
            for marker in (
                "replicat",
                "producer input",
                "physical placement",
                "profiler failed",
                "remote dataset",
                "usable producer",
            )
        ):
            return diagnostic(
                ErrorCategory.DATA,
                "A dataset operation could not complete.",
                message,
                reason=(
                    "The dataset provider, placement or verification step did not produce "
                    "publishable evidence."
                ),
                impact=("No unverified dataset placement was registered.",),
                fixes=(
                    "Inspect the dataset placement/build evidence and retry after correcting it.",
                ),
                commands=commands,
                context={"cluster": context.cluster},
                operation=context.operation,
                job_id=job_id,
            )
        if isinstance(error, (FileExistsError, FileNotFoundError, KeyError, ValueError, TypeError)):
            return diagnostic(
                ErrorCategory.CONFIGURATION,
                "LambdaForge cannot perform the requested operation with the supplied values.",
                message,
                reason="A required selector, file, field or option is missing or incompatible.",
                impact=("The command stopped before starting new scientific computation.",),
                fixes=("Correct the reported value or inspect the corresponding configuration.",),
                commands=commands,
                context={"cluster": context.cluster},
                operation=context.operation,
            )
        return diagnostic(
            ErrorCategory.INTERNAL,
            "LambdaForge encountered an unexpected internal failure.",
            "The framework reached a condition that is not classified as normal user error.",
            reason=message or type(error).__name__,
            impact=("LambdaForge stopped the current operation to avoid unsafe assumptions.",),
            fixes=("Retry once with --debug; if it persists, report the diagnostic record.",),
            commands=commands,
            context={"cluster": context.cluster},
            retryable=RetryDisposition.UNKNOWN,
            operation=context.operation,
            job_id=job_id,
            details=("This is probably a LambdaForge bug, not an expected configuration error.",),
        )

    @staticmethod
    def _known_data(
        error: BaseException,
        context: DiagnosticContext,
        title: str,
        commands: Sequence[tuple[str, str]],
    ) -> ErrorDiagnostic:
        return diagnostic(
            ErrorCategory.DATA,
            title,
            SecretRedactor.redact(error),
            reason="LambdaForge could not resolve or verify the requested immutable dataset.",
            impact=("No ambiguous or unverified dataset was used.",),
            fixes=("Use an exact version/location or repair the reported dataset metadata.",),
            commands=commands,
            context={"cluster": context.cluster},
            operation=context.operation,
        )

    @staticmethod
    def _diagnostic_commands(
        context: DiagnosticContext, job_id: str | None
    ) -> tuple[tuple[str, str], ...]:
        commands: list[tuple[str, str]] = []
        if job_id:
            commands.extend(
                (
                    ("View logs", f"lf jobs logs {job_id} --tail 300"),
                    ("Job details", f"lf jobs show {job_id}"),
                )
            )
        elif context.cluster:
            commands.append(("Diagnose cluster", f"lf doctor --on {shlex.quote(context.cluster)}"))
        if context.arguments:
            debug_arguments = tuple(value for value in context.arguments if value != "--debug")
            commands.append(
                ("Retry with internals", shlex.join(("lf", *debug_arguments, "--debug")))
            )
        return tuple(commands)


class DiagnosticRecorder:
    """Persist redacted traceback and invocation evidence atomically on the controller."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            root = state / "lambdaforge" / "logs" / "errors"
        self.root = Path(root).expanduser().resolve()

    def record(
        self,
        value: ErrorDiagnostic,
        error: BaseException,
        context: DiagnosticContext,
    ) -> Path | None:
        """Best-effort persistence must never replace the original user diagnosis."""
        timestamp = datetime.now(timezone.utc)
        name = f"error-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}.json"
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.root.chmod(0o700)
            path = self.root / name
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            traceback_text = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            payload = {
                "diagnostic_record_version": 1,
                "timestamp_utc": timestamp.isoformat(),
                "lambdaforge_version": LambdaForgeVersion.CURRENT,
                "command": context.command,
                "operation": context.operation,
                "cluster": context.cluster,
                "job_id": value.job_id,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback_text,
                "diagnostic": value.to_dict(),
            }
            redacted = self._redact(payload)
            temporary.write_text(
                json.dumps(redacted, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            temporary.chmod(0o600)
            os.replace(temporary, path)
            return path
        except OSError:
            return None

    @classmethod
    def _redact(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): (
                    SecretRedactor.MARKER
                    if SecretRedactor.is_secret_key(str(key))
                    else cls._redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._redact(item) for item in value]
        return SecretRedactor.redact(value) if isinstance(value, str) else value


def job_failure_diagnostic(record: Any, logs: str = "") -> ErrorDiagnostic:
    """Describe a terminal job root cause, derived blocking and preserved work."""
    job_id = str(record.job_id)
    state = str(getattr(record.state, "value", record.state))
    if state == "cancelled":
        return diagnostic(
            ErrorCategory.CANCELLED,
            f"Job {job_id} was cancelled.",
            "Cancellation is a terminal user/provider action, not a framework error.",
            reason="The scheduler reports the job as cancelled.",
            impact=("Completed logs and durable job metadata were retained.",),
            fixes=("Submit or retry only if the computation is still required.",),
            commands=(("Job details", f"lf jobs show {job_id}"),),
            context={"cluster": record.cluster, "state": state},
            retryable=RetryDisposition.AFTER_FIX,
            operation="job status",
            job_id=job_id,
        )
    if record.metadata.get("failure_phase") == "submission":
        raw_category = str(record.metadata.get("failure_category", "execution"))
        try:
            category = ErrorCategory(raw_category)
        except ValueError:
            category = ErrorCategory.EXECUTION
        return diagnostic(
            category,
            f"Job {job_id} was not accepted by the scheduler.",
            str(record.stderr).strip() or "Submission failed before scheduler acknowledgement.",
            reason="The submission/preflight provider failed before a scientific process started.",
            impact=(
                "A durable failed-submission record was retained.",
                "No scheduler acknowledgement or scientific execution was recorded.",
            ),
            fixes=("Correct the reported cluster/environment issue before retrying.",),
            commands=(
                ("Submission record", f"lf jobs logs {job_id} --tail 300"),
                ("Job details", f"lf jobs show {job_id} --json"),
                ("Diagnose cluster", f"lf doctor --on {record.cluster}"),
                ("Retry after fixing", f"lf jobs retry {job_id}"),
            ),
            context={
                "job": job_id,
                "cluster": record.cluster,
                "state": state,
                "phase": "submission",
            },
            operation="job submission",
            job_id=job_id,
        )
    payload = _dataset_build_payload(logs)
    impact: list[str] = []
    details: list[str] = []
    fixes = ["Inspect the logs, fix the root cause, then retry the terminal job."]
    cause = str(record.stderr).strip() or "The remote process exited unsuccessfully."
    title = f"Job {job_id} failed after it was started."
    if payload is not None:
        title = f"Dataset build {payload.get('dataset', record.metadata.get('name', ''))} failed."
        stages = payload.get("stages", {})
        failed = [name for name, item in stages.items() if item.get("status") == "failed"]
        blocked = [name for name, item in stages.items() if item.get("status") == "blocked"]
        completed = [name for name, item in stages.items() if item.get("status") == "ok"]
        if failed:
            root = failed[0]
            error = stages[root].get("error", {})
            cause = f"{error.get('type', 'Error')}: {error.get('message', 'stage failed')}"
            details.append(f"Root cause: {root} FAILED — {cause}")
        details.extend(
            f"{name} BLOCKED by {', '.join(stages[name].get('blocked_by', ())) or 'dependency'}"
            for name in blocked
        )
        if completed:
            impact.append(f"Reusable completed stages preserved: {', '.join(completed)}.")
            fixes.append(
                "It is safe to retry after the fix; verified completed stages may be reused."
            )
        if failed:
            impact.append(f"Failed stage: {failed[0]}.")
        if blocked:
            impact.append(f"Derived blocked stages did not execute: {', '.join(blocked)}.")
    else:
        impact.append(
            "LambdaForge cannot infer whether an external kill, resource limit or project error "
            "caused the exit from the available summary."
        )
    return diagnostic(
        ErrorCategory.RESOURCE if state == "timeout" else ErrorCategory.EXECUTION,
        title,
        cause,
        reason="The job was submitted and project/task code or its allocated process failed.",
        impact=impact,
        fixes=fixes,
        commands=(
            ("View logs", f"lf jobs logs {job_id} --tail 300"),
            ("Job details", f"lf jobs show {job_id} --json"),
            ("Retry after fixing", f"lf jobs retry {job_id}"),
        ),
        context={
            "job": job_id,
            "cluster": record.cluster,
            "state": state,
            "component": record.job_type,
        },
        retryable=RetryDisposition.AFTER_FIX,
        operation="job status",
        job_id=job_id,
        details=details,
    )


def execution_failure_diagnostic(
    *,
    kind: str,
    name: str,
    source: str | Path,
    error: Mapping[str, Any] | str | None = None,
    nodes: Mapping[str, Mapping[str, Any]] | None = None,
    run_dir: str | Path | None = None,
) -> ErrorDiagnostic:
    """Explain a synchronous task/workflow failure and derived blocked branches."""
    context = {"kind": kind, "name": name, "config": str(source), "run_dir": run_dir}
    impact: list[str] = []
    details: list[str] = []
    summary = "Project or task code returned an unsuccessful result."
    if isinstance(error, Mapping):
        summary = f"{error.get('type', 'Error')}: {error.get('message', 'execution failed')}"
    elif error:
        summary = str(error)
    if nodes:
        failed = [key for key, value in nodes.items() if value.get("status") == "failed"]
        blocked = [key for key, value in nodes.items() if value.get("status") == "blocked"]
        completed = [key for key, value in nodes.items() if value.get("status") == "ok"]
        if failed:
            root = failed[0]
            root_error = nodes[root].get("error", {})
            if isinstance(root_error, Mapping):
                summary = (
                    f"{root_error.get('type', 'Error')}: "
                    f"{root_error.get('message', 'stage failed')}"
                )
            details.append(f"Root cause: {root} FAILED — {summary}")
            impact.append(f"Failed component: {root}.")
        for blocked_name in blocked:
            dependencies = nodes[blocked_name].get("blocked_by", ())
            details.append(
                f"{blocked_name} BLOCKED by "
                f"{', '.join(str(value) for value in dependencies) or 'dependency'}"
            )
        if blocked:
            impact.append(f"Blocked components did not execute: {', '.join(blocked)}.")
        if completed:
            impact.append(f"Verified completed work was preserved: {', '.join(completed)}.")
    invocation = DiagnosticContext.from_argv(("run", str(source)))
    category = (
        ErrorCategory.DATA
        if "duplicate preprocessing record" in summary.lower()
        else ErrorCategory.EXECUTION
    )
    return diagnostic(
        category,
        f"{kind.title()} {name!r} failed after execution started.",
        summary,
        reason="The configured project task/component raised or returned a failed result.",
        impact=impact or ("No successful terminal result was published for this execution.",),
        fixes=(
            "Inspect the project/task error, correct its cause, then rerun the same configuration.",
        ),
        commands=(("Retry after fixing", invocation.command),),
        context=context,
        retryable=RetryDisposition.AFTER_FIX,
        operation=f"run {kind}",
        details=details,
    )


def _dataset_build_payload(logs: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", logs):
        try:
            value, _ = decoder.raw_decode(logs[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("kind") == "dataset-build":
            return value
    return None
