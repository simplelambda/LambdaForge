"""Password-only loopback SSH; temporary datasets, credentials and host keys only."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.data.DatasetMaterializationPlan import DatasetMaterializationPlan
from lambdaforge.data.errors import DirectTransferUnavailableError
from tests.datasets.test_compressed_replication import transfer  # noqa: F401
from tests.datasets.test_durable_replication import durable, finish  # noqa: F401


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_prompts_once_without_echoing_or_serializing_the_password(
    monkeypatch, capsys, explicit
):
    import importlib

    module = importlib.import_module("lambdaforge.cli.DatasetCommands")
    secret = "fixture-cli-password"
    calls, prompts = [], []
    plan = DatasetMaterializationPlan(
        "example@1",
        "gpu16",
        "REPLICATE",
        "gpu12",
        transfer_route="direct-ssh",
        execution_host="gpu12",
    )

    class Service:
        discovery_warnings = ()
        clusters = SimpleNamespace(
            get=lambda _: SimpleNamespace(user="fixture", host="peer.invalid")
        )

        def replication_preview(self, selector, **options):
            if "site_password" not in options:
                raise DirectTransferUnavailableError(
                    "No automatic authentication", ({"category": "authentication"},)
                )
            assert options["site_password"] == secret
            return plan

        def enqueue_replication(self, selector, **options):
            assert options["site_password"] == secret
            calls.append(selector)
            return replace(plan, job_id="job-fixture", applied=True)

    monkeypatch.setattr(module.ClusterCatalog, "load", lambda _: None)
    monkeypatch.setattr(module, "DatasetService", lambda **_: Service())
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))

    def prompt(message):
        prompts.append(message)
        return secret

    monkeypatch.setattr("getpass.getpass", prompt)
    args = SimpleNamespace(
        dataset_command="replicate",
        clusters=None,
        dataset="example@1",
        source="gpu12",
        destination="gpu16",
        route="direct",
        apply=True,
        json=False,
        ask_password=explicit,
    )
    assert module.DatasetCommands.run(args) == 0
    assert calls == ["example@1"] and len(prompts) == 1
    output = capsys.readouterr()
    assert secret not in output.out + output.err


def test_cli_never_prompts_for_passwords_in_json_or_unattended_mode(monkeypatch):
    import importlib

    module = importlib.import_module("lambdaforge.cli.DatasetCommands")
    monkeypatch.setattr(module.ClusterCatalog, "load", lambda _: None)
    monkeypatch.setattr(module, "DatasetService", lambda **_: object())
    monkeypatch.setattr("getpass.getpass", lambda _: pytest.fail("An unattended command prompted"))
    args = SimpleNamespace(
        dataset_command="replicate",
        clusters=None,
        dataset="example@1",
        source="gpu12",
        destination="gpu16",
        route="direct",
        apply=True,
        json=True,
        ask_password=True,
    )
    with pytest.raises(ValueError, match="interactive terminal"):
        module.DatasetCommands.run(args)


@pytest.fixture
def password_peer(request, monkeypatch):
    paramiko = pytest.importorskip("paramiko", reason="Uses optional cluster-password dependency")
    value, record, root = request.getfixturevalue("durable")
    service = value.datasets
    service.replicate(record.key, source="local", destination="gpu12", apply=True)
    home = root / "fixture-home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(repository / "src"), str(repository))))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.2)
    port = listener.getsockname()[1]
    key = paramiko.RSAKey.generate(2048)
    (home / ".ssh/known_hosts").write_text(
        f"[127.0.0.1]:{port} {key.get_name()} {key.get_base64()}\n"
    )
    stop = Event()
    errors, commands, threads = [], [], []
    secret = "disposable-fixture-only-password"

    class Server(paramiko.ServerInterface):
        def __init__(self):
            self.ready = Event()
            self.command = None

        def get_allowed_auths(self, username):
            return "password"

        def check_auth_password(self, username, password):
            return (
                paramiko.AUTH_SUCCESSFUL
                if username == "fixture" and password == secret
                else paramiko.AUTH_FAILED
            )

        def check_channel_request(self, kind, chanid):
            return (
                paramiko.OPEN_SUCCEEDED
                if kind == "session"
                else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
            )

        def check_channel_exec_request(self, channel, command):
            self.command = command.decode()
            commands.append(self.command)
            self.ready.set()
            return True

    def handle(connection):
        transport = paramiko.Transport(connection)
        try:
            transport.add_server_key(key)
            server = Server()
            transport.start_server(server=server)
            channel = transport.accept(timeout=15)
            if channel is None:
                return  # An explicitly rejected authentication opens no data channel.
            assert server.ready.wait(10)
            with subprocess.Popen(
                shlex.split(server.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as child:

                def input_bytes():
                    with channel.makefile("rb") as data:
                        while chunk := data.read(65536):
                            child.stdin.write(chunk)
                        child.stdin.close()

                def output_bytes(stream, send):
                    while chunk := stream.read(65536):
                        send(chunk)

                workers = [
                    Thread(target=input_bytes, daemon=True),
                    Thread(target=output_bytes, args=(child.stdout, channel.sendall), daemon=True),
                    Thread(
                        target=output_bytes,
                        args=(child.stderr, channel.sendall_stderr),
                        daemon=True,
                    ),
                ]
                for worker in workers:
                    worker.start()
                child.wait(timeout=30)
                for worker in workers[1:]:
                    worker.join(timeout=5)
                channel.send_exit_status(child.returncode)
                channel.shutdown_write()
        except (EOFError, OSError):
            pass  # A client may close immediately after a rejected password.
        except Exception as error:
            errors.append(error)
        finally:
            transport.close()

    def accept():
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            thread = Thread(target=handle, args=(connection,), daemon=True)
            threads.append(thread)
            thread.start()

    acceptor = Thread(target=accept, daemon=True)
    acceptor.start()
    for name in ("gpu12", "gpu16"):
        service.clusters._profiles[name] = replace(
            service.clusters.get(name), transport="ssh", host="127.0.0.1", user="fixture", port=port
        )
    monkeypatch.setattr(service.factory, "transport", lambda _: LocalTransport())
    yield value, record, root, secret, commands
    stop.set()
    acceptor.join(timeout=2)
    listener.close()
    for thread in threads:
        thread.join(timeout=3)
    assert not errors


def test_password_direct_job_verifies_and_survives_controller_handoff(password_peer):
    value, record, root, secret, commands = password_peer
    plan = value.preview(
        record.key, source="gpu12", destination="gpu16", route="direct", site_password=secret
    )
    assert plan.transfer_route == "direct-ssh" and not plan.requires_controller_online
    assert secret not in json.dumps(plan.to_dict())
    assert not list(value.jobs.store.root.glob("job-*.json"))
    result = value.enqueue(
        record.key,
        source="gpu12",
        destination="gpu16",
        route="direct",
        site_password=secret,
        approved_route=plan.transfer_route,
    )
    assert result.job_id and result.execution_host == "gpu12"
    row = finish(value, result.job_id)
    assert row["state"] == "succeeded", row
    assert row["progress"]["phase"] == "completed"
    assert value.datasets.resolve_placement(record.key, "gpu16").physically_available
    for command in commands:
        assert secret not in command
    for path in root.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path


def test_wrong_password_and_missing_trust_do_not_start_jobs(password_peer):
    value, record, root, secret, _ = password_peer
    with pytest.raises(DirectTransferUnavailableError) as failure:
        value.preview(
            record.key,
            source="gpu12",
            destination="gpu16",
            route="direct",
            site_password="incorrect-fixture-password",
        )
    assert "incorrect-fixture-password" not in str(failure.value)
    assert not list(value.jobs.store.root.glob("job-*.json"))
    (root / "fixture-home/.ssh/known_hosts").write_text("")
    with pytest.raises(DirectTransferUnavailableError):
        value.preview(
            record.key, source="gpu12", destination="gpu16", route="direct", site_password=secret
        )
    assert not value.datasets.resolve_placement(record.key, "gpu16").physically_available


def test_password_rejected_after_preview_fails_only_owned_job_and_redacts_logs(
    password_peer, monkeypatch
):
    from tests.controlplane.test_execution_queue import wait_for

    value, record, root, secret, _ = password_peer
    plan = value.preview(
        record.key, source="gpu12", destination="gpu16", route="direct", site_password=secret
    )
    # Model a credential rotation between a successful preview and the actual peer connection.
    monkeypatch.setattr(value, "preview", lambda *args, **kwargs: plan)
    rejected = "fixture-credential-expired-after-preview"
    result = value.enqueue(
        record.key, source="gpu12", destination="gpu16", route="direct", site_password=rejected
    )
    wait_for(lambda: value.status(result.job_id)["state"] == "failed")
    assert not value.datasets.resolve_placement(record.key, "gpu16").physically_available
    assert "authentication" in value.status(result.job_id)["error"].lower()
    for path in root.rglob("*"):
        if path.is_file():
            assert rejected.encode() not in path.read_bytes(), path
