"""Per-cluster SLURM dialect descriptor."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lambdaforge.controlplane.SchedulerCommand import SchedulerCommand
from lambdaforge.controlplane.SlurmResourceMapping import SlurmResourceMapping
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class SlurmProfile:
    """Customize SLURM resources, static directives, commands and job scripts."""

    resource_mapping: SlurmResourceMapping = field(default_factory=SlurmResourceMapping)
    directives: Mapping[str, Any] = field(default_factory=dict)
    submit: SchedulerCommand = field(
        default_factory=lambda: SchedulerCommand(
            "sbatch", ("--parsable", "{script}"), r"^(\d+)(?:;.*)?$"
        )
    )
    queue: SchedulerCommand = field(
        default_factory=lambda: SchedulerCommand("squeue", ("-h", "-j", "{job_id}", "-o", "%T"))
    )
    accounting: SchedulerCommand = field(
        default_factory=lambda: SchedulerCommand(
            "sacct", ("-n", "-X", "-j", "{job_id}", "-o", "State", "--parsable2")
        )
    )
    cancel: SchedulerCommand = field(
        default_factory=lambda: SchedulerCommand("scancel", ("{job_id}",))
    )
    info: SchedulerCommand = field(
        default_factory=lambda: SchedulerCommand("sinfo", ("-h", "-p", "{partition}", "-o", "%P"))
    )
    shell: str = "/bin/bash"
    prologue: tuple[str, ...] = ()
    epilogue: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        resource_mapping = self.resource_mapping
        if isinstance(resource_mapping, Mapping):
            resource_mapping = SlurmResourceMapping.from_mapping(resource_mapping)
            object.__setattr__(self, "resource_mapping", resource_mapping)
        if not self.shell.startswith("/") or "\n" in self.shell or "\x00" in self.shell:
            raise ValueError("SLURM job-script shell must be an absolute newline-free path.")
        if any(
            "\n" in line or "\r" in line or "\x00" in line
            for line in (*self.prologue, *self.epilogue)
        ):
            raise ValueError("SLURM prologue/epilogue entries must each be one shell line.")
        normalized: dict[str, Any] = {}
        resource_options = {
            str(rule.get("option"))
            for rule in resource_mapping.rules.values()
            if not rule.get("omit")
        }
        for key, value in self.directives.items():
            name = str(key).removeprefix("--")
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9-]*", name) is None:
                raise ValueError(f"Unsafe static SLURM directive {name!r}.")
            if name in resource_options:
                raise ValueError(
                    f"Static directive --{name} duplicates the resource translation layer."
                )
            values = (
                value
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
                else (value,)
            )
            clean: list[Any] = []
            for item in values:
                if item is False or item is None:
                    continue
                if not isinstance(item, (str, int, float, bool)):
                    raise TypeError(
                        f"Static SLURM directive {name!r} needs scalar or repeated values."
                    )
                if isinstance(item, str) and ("\n" in item or "\r" in item):
                    raise ValueError(f"Unsafe value for static SLURM directive {name!r}.")
                clean.append(item)
            if clean:
                normalized[name] = clean
        object.__setattr__(self, "directives", FrozenJsonMapping(normalized))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None, *, legacy_options: Mapping[str, Any] | None = None
    ) -> SlurmProfile:
        """Parse an explicit dialect while preserving legacy scheduler_options."""
        data = dict(value or {})
        commands = data.get("scheduler_commands", data.get("commands", {}))
        if not isinstance(commands, Mapping):
            raise TypeError("scheduler_commands must be a mapping.")
        job_script = data.get("job_script", {})
        if not isinstance(job_script, Mapping):
            raise TypeError("job_script must be a mapping.")
        directives = dict(legacy_options or {})
        explicit = data.get("scheduler_directives", data.get("directives", {}))
        if not isinstance(explicit, Mapping):
            raise TypeError("scheduler_directives must be a mapping.")
        directives.update(explicit)
        submit_default = SchedulerCommand("sbatch", ("--parsable", "{script}"), r"^(\d+)(?:;.*)?$")
        queue_default = SchedulerCommand("squeue", ("-h", "-j", "{job_id}", "-o", "%T"))
        accounting_default = SchedulerCommand(
            "sacct", ("-n", "-X", "-j", "{job_id}", "-o", "State", "--parsable2")
        )
        cancel_default = SchedulerCommand("scancel", ("{job_id}",))
        info_default = SchedulerCommand("sinfo", ("-h", "-p", "{partition}", "-o", "%P"))
        prologue = cls._lines(job_script.get("prologue", ()), "prologue")
        epilogue = cls._lines(job_script.get("epilogue", ()), "epilogue")
        return cls(
            resource_mapping=SlurmResourceMapping.from_mapping(data.get("resource_mapping")),
            directives=directives,
            submit=SchedulerCommand.from_value(commands.get("submit"), default=submit_default),
            queue=SchedulerCommand.from_value(commands.get("queue"), default=queue_default),
            accounting=SchedulerCommand.from_value(
                commands.get("accounting"), default=accounting_default
            ),
            cancel=SchedulerCommand.from_value(commands.get("cancel"), default=cancel_default),
            info=SchedulerCommand.from_value(commands.get("info"), default=info_default),
            shell=str(job_script.get("shell", "/bin/bash")),
            prologue=prologue,
            epilogue=epilogue,
        )

    def render_directives(self) -> tuple[str, ...]:
        """Render deterministic static flags, values and repeated values."""
        rendered = []
        if "output" not in self.directives:
            rendered.append("--output=lambdaforge-%j.out")
        if "error" not in self.directives:
            rendered.append("--error=lambdaforge-%j.out")
        for name in sorted(self.directives):
            for value in self.directives[name]:
                rendered.append(f"--{name}" if value is True else f"--{name}={value}")
        return tuple(rendered)

    def executables(self) -> tuple[str, ...]:
        """Return every configured scheduler executable once."""
        return tuple(
            dict.fromkeys(
                (
                    self.submit.executable,
                    self.queue.executable,
                    self.accounting.executable,
                    self.cancel.executable,
                    self.info.executable,
                )
            )
        )

    def to_dict(self, *, include_defaults: bool = True) -> dict[str, Any]:
        """Return the complete effective or compact customized SLURM dialect."""
        effective = {
            "resource_mapping": self.resource_mapping.to_dict(),
            "scheduler_directives": {
                key: (values[0] if len(values) == 1 else list(values))
                for key, values in self.directives.items()
            },
            "scheduler_commands": {
                "submit": self.submit.to_dict(),
                "queue": self.queue.to_dict(),
                "accounting": self.accounting.to_dict(),
                "cancel": self.cancel.to_dict(),
                "info": self.info.to_dict(),
            },
            "job_script": {
                "shell": self.shell,
                "prologue": list(self.prologue),
                "epilogue": list(self.epilogue),
            },
        }
        if include_defaults:
            return effective
        defaults = SlurmProfile().to_dict()
        compact: dict[str, Any] = {}
        for key, value in effective.items():
            if value != defaults[key]:
                compact[key] = value
        return compact

    @staticmethod
    def _lines(value: Any, name: str) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            raise TypeError(f"job_script.{name} must be a string or list of strings.")
        return tuple(str(item) for item in value)
