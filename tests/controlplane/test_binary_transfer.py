"""Bounded binary transport preserves normal SSH policy and sends explicit EOF."""

import io
import shlex
import sys
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane.BinaryCommand import subprocess_stream
from lambdaforge.controlplane.PasswordSshTransport import PasswordSshTransport
from lambdaforge.controlplane.SshTransport import SshTransport


def test_large_stderr_is_drained_and_bounded_without_blocking_binary_output():
    command = (
        sys.executable,
        "-c",
        "import sys; sys.stderr.buffer.write(b'x'*1048576+b'final cause'); "
        "sys.stdout.buffer.write(b'bytes'); sys.exit(3)",
    )
    with subprocess_stream(command) as stream:
        assert stream.stdout.read() == b"bytes"
        with pytest.raises(RuntimeError, match="final cause") as failure:
            stream.wait()
        assert len(str(failure.value)) < 20000


def test_password_binary_channel_is_reused_quoted_and_half_closed():
    events = []
    channel = SimpleNamespace(
        shutdown_write=lambda: events.append("EOF"),
        recv_exit_status=lambda: 0,
        close=lambda: events.append("close"),
    )
    stdin, stdout, stderr = io.BytesIO(), io.BytesIO(b"ack"), io.BytesIO()
    stdin.channel = stdout.channel = channel
    commands = []

    def execute(command):
        commands.append(command)
        return stdin, stdout, stderr

    transport = PasswordSshTransport(
        "example.invalid", password_provider=lambda: pytest.fail("Do not reauthenticate")
    )
    transport._client = SimpleNamespace(exec_command=execute)
    with transport.stream(("python", "path with spaces")) as stream:
        stream.stdin.write(b"compressed")
        stream.finish_input()
        assert stream.stdout.read() == b"ack"
        stream.wait()
    assert commands == [shlex.join(("python", "path with spaces"))]
    assert events == ["EOF", "close"]


def test_openssh_binary_stream_preserves_audited_options(monkeypatch, tmp_path):
    # Import dotted modules explicitly: the package also exposes the class under this name.
    import importlib

    module = importlib.import_module("lambdaforge.controlplane.SshTransport")
    commands = []
    marker = object()
    monkeypatch.setattr(
        module, "subprocess_stream", lambda command: commands.append(command) or marker
    )
    transport = SshTransport(
        "example.invalid",
        user="scientist",
        port=2022,
        options=("-o", "StrictHostKeyChecking=yes"),
        control_root=tmp_path / "ssh",
    )
    assert transport.stream(("python", "path with spaces")) is marker
    assert commands[0] == (
        "ssh",
        *transport.options,
        transport.destination,
        "--",
        shlex.join(("python", "path with spaces")),
    )
