"""Host-key-verifying password SSH transport."""

from __future__ import annotations

import shlex
import stat
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.RemoteCommandTimeout import RemoteCommandTimeout
from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.controlplane.Transport import Transport


class PasswordSshTransport(Transport):
    """Use Paramiko for password auth while rejecting unknown host keys."""

    def __init__(
        self,
        host: str,
        *,
        password_provider: Callable[[], str],
        user: str | None = None,
        port: int = 22,
        timeout: float = 15.0,
        auth_timeout: float | None = None,
        banner_timeout: float | None = None,
        keepalive: float = 30.0,
        command_timeout: float | None = None,
        known_hosts: str | Path | None = None,
        paramiko_module: Any | None = None,
    ) -> None:
        if not host.strip() or host.startswith("-") or "\n" in host:
            raise ValueError("SSH host must be a non-empty host name.")
        if timeout <= 0:
            raise ValueError("SSH timeout must be positive.")
        self.host = host
        self.user = user
        self.port = port
        self.timeout = timeout
        self.auth_timeout = auth_timeout if auth_timeout is not None else timeout
        self.banner_timeout = banner_timeout if banner_timeout is not None else timeout
        self.keepalive = keepalive
        self.command_timeout = command_timeout
        self.known_hosts = Path(known_hosts).expanduser() if known_hosts is not None else None
        self._password_provider = password_provider
        self._paramiko_module = paramiko_module
        self._client: Any | None = None

    def __repr__(self) -> str:
        return (
            f"PasswordSshTransport(host={self.host!r}, user={self.user!r}, "
            f"port={self.port!r}, password={SecretRedactor.MARKER!r})"
        )

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Execute one safely quoted argv through the established SSH channel."""
        if not command:
            raise ValueError("SSH commands cannot be empty.")
        remote = shlex.join(tuple(str(item) for item in command))
        if cwd is not None:
            remote = f"cd {shlex.quote(str(cwd))} && exec {remote}"
        try:
            deadline = self.command_timeout if timeout is None else timeout
            _, stdout, stderr = self._connection().exec_command(remote, timeout=deadline)
            output = stdout.read().decode("utf-8", errors="replace")
            errors = stderr.read().decode("utf-8", errors="replace")
            return CommandResult(stdout.channel.recv_exit_status(), output, errors)
        except TimeoutError as error:
            raise RemoteCommandTimeout(remote, float(deadline or 0)) from error
        except Exception as error:
            raise RuntimeError(
                f"Password SSH command failed for {self.host!r}: {SecretRedactor.redact(error)}"
            ) from None

    def put(self, source: str | Path, destination: str | Path) -> None:
        """Upload one file or directory recursively through SFTP."""
        source_path = Path(source).resolve()
        sftp = self._connection().open_sftp()
        try:
            if source_path.is_dir():
                self._mkdirs(sftp, str(destination))
                for item in sorted(source_path.rglob("*")):
                    relative = item.relative_to(source_path).as_posix()
                    remote = str(PurePosixPath(str(destination)) / relative)
                    if item.is_dir():
                        self._mkdirs(sftp, remote)
                    else:
                        self._mkdirs(sftp, str(PurePosixPath(remote).parent))
                        sftp.put(str(item), remote)
            else:
                self._mkdirs(sftp, str(PurePosixPath(str(destination)).parent))
                sftp.put(str(source_path), str(destination))
        except Exception as error:
            raise RuntimeError(f"SFTP upload failed for {self.host!r}: {error}") from None
        finally:
            sftp.close()

    def get(self, source: str | Path, destination: str | Path) -> None:
        """Download one explicit file or directory recursively through SFTP."""
        destination_path = Path(destination).resolve()
        sftp = self._connection().open_sftp()
        try:
            attributes = sftp.stat(str(source))
            if stat.S_ISDIR(attributes.st_mode):
                destination_path.mkdir(parents=True, exist_ok=True)
                self._download_directory(sftp, str(source), destination_path)
            else:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(str(source), str(destination_path))
        except Exception as error:
            raise RuntimeError(f"SFTP download failed for {self.host!r}: {error}") from None
        finally:
            sftp.close()

    def close(self) -> None:
        """Close the reusable SSH connection, if opened."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _connection(self) -> Any:
        if self._client is not None:
            return self._client
        module = self._paramiko_module or self._import_paramiko()
        client = module.SSHClient()
        client.load_system_host_keys()
        if self.known_hosts is not None:
            if not self.known_hosts.is_file():
                raise RuntimeError(
                    f"Configured known_hosts file does not exist: {self.known_hosts}."
                )
            client.load_host_keys(str(self.known_hosts))
        client.set_missing_host_key_policy(module.RejectPolicy())
        password = self._password_provider()
        try:
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                password=password,
                timeout=self.timeout,
                banner_timeout=self.banner_timeout,
                auth_timeout=self.auth_timeout,
                look_for_keys=False,
                allow_agent=False,
            )
        except Exception as error:
            client.close()
            message = SecretRedactor.redact(error, (password,))
            raise RuntimeError(
                f"Password SSH authentication/host-key verification failed for {self.host!r}: "
                f"{message}"
            ) from None
        finally:
            del password
        transport = client.get_transport() if hasattr(client, "get_transport") else None
        if transport is not None and self.keepalive > 0:
            transport.set_keepalive(int(self.keepalive))
        self._client = client
        return client

    @staticmethod
    def _import_paramiko() -> Any:
        try:
            import paramiko  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError(
                "Password SSH requires Paramiko. Install 'lambdaforge[cluster-password]' or "
                "use the default OpenSSH authentication."
            ) from error
        return paramiko

    @classmethod
    def _mkdirs(cls, sftp: Any, destination: str) -> None:
        current = PurePosixPath("/") if destination.startswith("/") else PurePosixPath()
        for part in PurePosixPath(destination).parts:
            if part == "/":
                continue
            current = current / part
            try:
                sftp.stat(str(current))
            except OSError:
                sftp.mkdir(str(current))

    @classmethod
    def _download_directory(cls, sftp: Any, source: str, destination: Path) -> None:
        for item in sftp.listdir_attr(source):
            remote = str(PurePosixPath(source) / item.filename)
            local = destination / item.filename
            if stat.S_ISDIR(item.st_mode):
                local.mkdir(parents=True, exist_ok=True)
                cls._download_directory(sftp, remote, local)
            else:
                sftp.get(remote, str(local))
