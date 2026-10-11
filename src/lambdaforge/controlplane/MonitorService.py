"""Independent project observer over the native Job catalog, never a scientific runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.OperationalHealth import (
    ObservationPolicy,
    OperationalHealth,
    timestamp,
)
from lambdaforge.controlplane.OperationalIncidents import (
    OperationalIncidents,
    read_document,
    write_document,
)
from lambdaforge.controlplane.OverviewService import OverviewService
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class MonitorService:
    """A single locked read-only provider monitor per project, with native durable observations."""

    def __init__(self, jobs: JobService) -> None:
        self.jobs = jobs
        self.root = jobs.store.root / ".operational" / "monitor"
        self.incidents = OperationalIncidents(jobs.store)

    def config(self, settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Validate and persist explicit operational policy; never alter authored Work YAML."""
        path = self.root / "config.json"
        if settings is None:
            return read_document(path) or {
                "interval_seconds": 60.0,
                "observation": {},
                "smtp": {"enabled": False},
            }
        value = dict(settings)
        if set(value) - {"interval_seconds", "observation", "smtp"}:
            raise ValueError("Unknown monitor configuration fields.")
        interval = value.get("interval_seconds", 60.0)
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval < 10
        ):
            raise ValueError("Monitor interval_seconds must be finite and at least 10 seconds.")
        ObservationPolicy.from_mapping(value.get("observation"))
        smtp = value.get("smtp", {})
        if not isinstance(smtp, Mapping):
            raise TypeError("Monitor smtp must be a mapping.")
        allowed = {
            "enabled",
            "host",
            "port",
            "tls",
            "sender",
            "recipients",
            "username",
            "credential",
            "cooldown_seconds",
        }
        if set(smtp) - allowed:
            raise ValueError("Unknown SMTP fields; passwords must use a credential reference.")
        if type(smtp.get("enabled", False)) is not bool:
            raise ValueError("SMTP enabled must be a boolean.")
        port = smtp.get("port", 587)
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("SMTP port must be an integer from 1 to 65535.")
        if smtp.get("credential") and not str(smtp["credential"]).startswith(("env:", "keyring:")):
            raise ValueError("SMTP credentials must use env: or keyring: references.")
        if smtp.get("enabled"):
            for field in ("host", "sender"):
                if not isinstance(smtp.get(field), str) or not smtp[field].strip():
                    raise ValueError(f"SMTP requires {field}.")
            recipients = smtp.get("recipients")
            if (
                not isinstance(recipients, list)
                or not recipients
                or any(not isinstance(item, str) or not item.strip() for item in recipients)
            ):
                raise ValueError("SMTP recipients must be a nonempty list of addresses.")
            if smtp.get("tls", "starttls") not in {"ssl", "starttls"}:
                raise ValueError("SMTP requires ssl or starttls.")
            if smtp.get("username") and not str(smtp.get("credential", "")).startswith(
                ("env:", "keyring:")
            ):
                raise ValueError(
                    "SMTP authentication requires an env: or keyring: credential reference."
                )
            cooldown = smtp.get("cooldown_seconds", 900)
            if (
                isinstance(cooldown, bool)
                or not isinstance(cooldown, (int, float))
                or not math.isfinite(cooldown)
                or cooldown < 30
            ):
                raise ValueError("SMTP cooldown_seconds must be at least 30 seconds.")
        value.update(
            interval_seconds=float(interval),
            observation=dict(value.get("observation") or {}),
            smtp=dict(smtp),
        )
        with self._lock("config"):
            write_document(path, value)
        return value

    def test_alert(self) -> dict[str, bool]:
        """Explicit delivery test shared by CLI and Console; no incident or science mutation."""
        settings = self.config().get("smtp", {})
        if not settings.get("enabled"):
            raise ValueError("SMTP is disabled; configure an explicit channel first.")
        try:
            OperationalIncidents.send(
                settings,
                {
                    "status": "test",
                    "code": "NOTIFICATION_TEST",
                    "job_id": "none",
                    "incident_id": "lambdaforge-test",
                    "summary": "Explicit operational notification test.",
                    "diagnostic_command": "lf monitor status",
                    "recommendation": "No scientific action required.",
                },
            )
        except Exception as error:
            # SMTP replies are untrusted and can echo authentication material. Keep type,
            # never the server reply or chained authentication exception, even in --debug.
            raise RuntimeError(
                f"SMTP test failed ({type(error).__name__}); "
                "inspect TLS, relay and credential access."
            ) from None
        return {"accepted_by_smtp": True, "scientific_jobs_unchanged": True}

    def status(self) -> dict[str, Any]:
        """Verify monitor process lifetime separately from its durable heartbeat."""
        value = read_document(self.root / "state.json")
        raw = value.get("identity")
        lifetime = (
            ProcessIdentity.from_mapping(raw).storage_status()
            if isinstance(raw, Mapping)
            else "missing"
        )
        heartbeat = timestamp(value.get("heartbeat_at_utc"))
        age = (datetime.now(timezone.utc) - heartbeat).total_seconds() if heartbeat else None
        interval = float(self.config()["interval_seconds"])
        return {
            **value,
            "monitor_version": 1,
            "state_root": str(self.root),
            "process_lifetime": lifetime,
            "heartbeat_age_seconds": age,
            "health": ("degraded" if value.get("failures", 0) else "healthy")
            if lifetime == "alive" and age is not None and 0 <= age < 2 * interval + 60
            else "stale"
            if lifetime == "alive"
            else "stopped"
            if lifetime in {"dead", "reused", "missing"}
            else "unknown",
            "restart_command": "lf monitor start",
            "diagnostic_command": "lf monitor diagnose",
        }

    def start(self, *, clusters: Path | None = None) -> dict[str, Any]:
        """Detach from the terminal; automatic reboot/logout persistence requires an OS service."""
        project = self.jobs.catalog.project
        if project is None:
            raise ValueError("Independent monitoring requires a project-owned JobStore.")
        from lambdaforge.controlplane.JobStore import JobStore

        if self.jobs.store.root != JobStore(project=project).root:
            raise ValueError("Detached monitoring requires the native project JobStore path.")
        with self._lock("start"):
            status = self.status()
            if status["process_lifetime"] in {"alive", "unresolved"}:
                return {
                    **status,
                    "started": False,
                    "reason": "Existing monitor owner; no duplicate launched.",
                }
            (self.root / "stop.json").unlink(missing_ok=True)
            command = self._command(project, clusters)
            options: dict[str, Any] = (
                {"start_new_session": True}
                if os.name == "posix"
                else {
                    "creationflags": getattr(subprocess, "DETACHED_PROCESS", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                }
            )
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "monitor.log").open("ab") as output:
                process = subprocess.Popen(
                    command,
                    cwd=project.root,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=output,
                    close_fds=True,
                    **options,
                )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                result = self.status()
                if (
                    result.get("identity", {}).get("pid") == process.pid
                    and result["process_lifetime"] == "alive"
                ):
                    return {
                        **result,
                        "started": True,
                        "persistence": "detached; use monitor install for managed restart",
                    }
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Monitor launch failed; inspect {self.root / 'monitor.log'}."
                    )
                time.sleep(0.05)
            raise TimeoutError(
                f"Monitor did not acknowledge launch; inspect {self.root / 'monitor.log'}."
            )

    def stop(self) -> dict[str, Any]:
        """Request observer shutdown only; never signal scientific supervisors or Runs."""
        status = self.status()
        with self._lock("start"):
            write_document(
                self.root / "stop.json",
                {"identity": status.get("identity"), "requested_at_utc": self._now()},
            )
        return {**status, "stop_requested": True, "scientific_jobs_unchanged": True}

    def tick(self) -> dict[str, Any]:
        """One bounded metadata-only catalog pass; no artifacts/epochs/HPO fitting."""
        settings = self.config()
        values = OverviewService._jobs(self.jobs)
        policy = ObservationPolicy.from_mapping(settings.get("observation"))
        for record in values:
            self.incidents.observe(record)
        self.incidents.deliver(settings.get("smtp", {}))
        return {
            "observed_at_utc": self._now(),
            "jobs": len(values),
            "health": {
                record.job_id: OperationalHealth.describe(record, policy=policy)
                for record in values
            },
        }

    def run(self) -> None:
        """Foreground service entry point, with a process lock against duplicate observers."""
        with self._lock("owner", timeout=0.2):
            command = tuple(str(item) for item in __import__("psutil").Process().cmdline())
            identity = ProcessIdentity.create(
                os.getpid(),
                os.getpgrp() if os.name == "posix" else os.getpid(),
                command,
                "operational-monitor",
            )
            state: dict[str, Any] = {
                "monitor_version": 1,
                "identity": identity.to_dict(),
                "started_at_utc": self._now(),
                "failures": 0,
            }
            write_document(self.root / "state.json", {**state, "heartbeat_at_utc": self._now()})
            while True:
                stop = read_document(self.root / "stop.json")
                if stop and stop.get("identity") == identity.to_dict():
                    write_document(
                        self.root / "state.json", {**state, "stopped_at_utc": self._now()}
                    )
                    return
                try:
                    result = self.tick()
                except Exception as error:
                    state.update(
                        failures=int(state["failures"]) + 1, last_error_type=type(error).__name__
                    )
                else:
                    state.update(
                        last_success_at_utc=result["observed_at_utc"],
                        jobs=result["jobs"],
                        failures=0,
                    )
                    state.pop("last_error_type", None)
                write_document(self.root / "state.json", {**state, "heartbeat_at_utc": self._now()})
                deadline = time.monotonic() + float(self.config()["interval_seconds"])
                while time.monotonic() < deadline:
                    stop = read_document(self.root / "stop.json")
                    if stop and stop.get("identity") == identity.to_dict():
                        break
                    time.sleep(min(0.5, max(0.01, deadline - time.monotonic())))

    def install(self, *, clusters: Path | None = None, apply: bool = False) -> dict[str, Any]:
        """Preview/apply a native systemd-user service; never enable linger without consent."""
        project = self.jobs.catalog.project
        if project is None or sys.platform != "linux":
            raise ValueError(
                "Automatic service installation requires Linux/systemd-user; "
                "use monitor run with your OS service manager."
            )
        unit = "lambdaforge-monitor-" + project.project_id + ".service"
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd" / "user"
        path = root / unit
        command = " ".join(
            json.dumps(item.replace("%", "%%")) for item in self._command(project, clusters)
        )
        content = (
            "# Managed LambdaForge operational monitor\n[Unit]\n"
            "Description=LambdaForge project observer\n"
            f"[Service]\nExecStart={command}\n"
            f"WorkingDirectory={json.dumps(str(project.root).replace('%', '%%'))}\n"
            "Restart=on-failure\nRestartSec=30\n[Install]\nWantedBy=default.target\n"
        )
        if apply:
            if path.is_symlink() or (
                path.exists()
                and not path.read_text().startswith("# Managed LambdaForge operational monitor\n")
            ):
                raise ValueError("Refusing to overwrite an unrelated systemd unit.")
            root.mkdir(parents=True, exist_ok=True)
            # Use the same safe atomic document writer for structured operational files only;
            # this unit is text and its directory has been explicitly selected above.
            temporary = path.with_suffix(".service.tmp")
            if temporary.exists() or temporary.is_symlink():
                raise ValueError("Unexpected systemd unit staging file.")
            try:
                with temporary.open("x", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            for argv in (
                ("systemctl", "--user", "daemon-reload"),
                ("systemctl", "--user", "enable", "--now", unit),
            ):
                result = subprocess.run(
                    argv, capture_output=True, text=True, timeout=20, check=False
                )
                if result.returncode:
                    raise RuntimeError(
                        "systemd-user activation failed; unit was retained for diagnosis. "
                        + result.stderr[:500]
                    )
        return {
            "unit": str(path),
            "content": content,
            "applied": apply,
            "logout_persistence": (
                "Depends on systemd user lingering; inspect loginctl show-user. "
                "No linger change was made."
            ),
        }

    def _command(self, project: ProjectContext, clusters: Path | None) -> tuple[str, ...]:
        command = (
            sys.executable,
            "-m",
            "lambdaforge.controlplane.MonitorService",
            "--project",
            str(project.root),
        )
        return (*command, "--clusters", str(clusters.resolve())) if clusters else command

    def _lock(self, name: str, *, timeout: float = 10) -> CrossProcessFileLock:
        return CrossProcessFileLock(
            self.root / f"{name}.lock",
            shared=False,
            timeout_seconds=timeout,
            poll_interval_seconds=0.05,
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Native project operational observer")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--clusters", type=Path)
    args = parser.parse_args()
    project = ProjectContext.discover(args.project)
    MonitorService(JobService(ClusterCatalog.load(args.clusters, project=project))).run()


if __name__ == "__main__":
    main()
