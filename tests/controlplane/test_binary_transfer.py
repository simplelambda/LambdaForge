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
    channel.makefile = lambda mode, size: stdout
    channel.makefile_stderr = lambda mode, size: stderr
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


def test_password_cleanup_eof_cannot_replace_primary_transport_failure():
    closed = []

    class BrokenClose(io.BytesIO):
        def close(self):
            raise EOFError("secondary cleanup EOF")

    channel = SimpleNamespace(close=lambda: closed.append(True), exit_status_ready=lambda: False)
    stdout = io.BytesIO()
    stdout.channel = channel
    channel.makefile = lambda mode, size: stdout
    channel.makefile_stderr = lambda mode, size: io.BytesIO(b"peer diagnostic")
    transport = PasswordSshTransport("peer.invalid", password_provider=lambda: "unused")
    transport._client = SimpleNamespace(
        exec_command=lambda _c: (BrokenClose(), stdout, io.BytesIO(b"peer diagnostic"))
    )
    primary = EOFError("original write failure")
    with pytest.raises(EOFError) as caught:
        with transport.stream(("receiver",)) as stream:
            assert stream.diagnostics()["exit_code"] is None
            raise primary
    assert caught.value is primary
    assert closed == [True]


def test_paramiko_text_default_readline_is_replaced_with_binary_channel_reader():
    header = b'{"content_id":"sha256:abc"}\n'
    payload = bytes(range(256)) * 16
    modes = []
    channel = SimpleNamespace(
        makefile=lambda mode, size: modes.append(mode) or io.BytesIO(header + payload),
        makefile_stderr=lambda mode, size: modes.append(mode) or io.BytesIO(),
        close=lambda: None,
        recv_exit_status=lambda: 0,
    )
    text_stdout = io.StringIO(header.decode())
    text_stdout.channel = channel
    transport = PasswordSshTransport("fixture.invalid", password_provider=lambda: "unused")
    transport._client = SimpleNamespace(
        exec_command=lambda _: (io.BytesIO(), text_stdout, io.BytesIO())
    )
    with transport.stream(("pack",)) as command:
        assert command.stdout.readline(65537) == header
        assert command.stdout.read() == payload
        command.wait()
    assert modes == ["rb", "rb"]


def test_real_paramiko_channel_preserves_header_and_non_utf8_payload():
    paramiko = pytest.importorskip("paramiko")
    payload = b'{"id":"fixture"}\n' + bytes(range(256))

    class Channel(paramiko.Channel):
        def __init__(self):
            super().__init__(1)
            self.data = io.BytesIO(payload)

        def recv(self, size):
            return self.data.read(size)

        def recv_stderr(self, size):
            return b""

        def close(self):
            pass

    channel = Channel()
    transport = PasswordSshTransport("fixture.invalid", password_provider=lambda: "unused")
    transport._client = SimpleNamespace(
        exec_command=lambda _: (io.BytesIO(), channel.makefile("r"), channel.makefile_stderr("r"))
    )
    with transport.stream(("pack",)) as stream:
        assert stream.stdout.readline() == b'{"id":"fixture"}\n'
        assert stream.stdout.read() == bytes(range(256))


def test_binary_finish_input_reports_available_peer_error_without_unbounded_wait():
    from lambdaforge.controlplane.BinaryCommand import BinaryCommand

    class BrokenFlush(io.BytesIO):
        def flush(self):
            raise EOFError("primary flush EOF")

    owner = SimpleNamespace(
        exit_status_ready=lambda: False,
        recv_exit_status=lambda: pytest.fail("Do not wait on lost SSH"),
    )
    stream = BinaryCommand(
        BrokenFlush(), io.BytesIO(), io.BytesIO(b"receiver disk admission denied"), owner
    )
    with pytest.raises(RuntimeError, match="receiver disk admission denied") as caught:
        stream.finish_input()
    assert isinstance(caught.value.__cause__, EOFError)


def test_unknown_peer_exit_minus_one_is_not_reported_as_known_process_failure():
    from lambdaforge.controlplane.BinaryCommand import BinaryCommand

    channel = SimpleNamespace(exit_status_ready=lambda: True, recv_exit_status=lambda: -1)
    stream = BinaryCommand(io.BytesIO(), io.BytesIO(), io.BytesIO(), channel)
    assert stream.diagnostics()["exit_code"] is None


@pytest.mark.parametrize("primary", [None, EOFError("original stream error")])
def test_subprocess_cleanup_reaps_child_even_when_stdin_close_fails(monkeypatch, primary):
    import importlib

    module = importlib.import_module("lambdaforge.controlplane.BinaryCommand")
    events = []

    class BrokenClose(io.BytesIO):
        def close(self):
            raise BrokenPipeError("cleanup pipe error")

    class Process:
        stdin = BrokenClose()
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append("exit")

        def poll(self):
            return None

        def wait(self, *, timeout):
            events.append(("wait", timeout))
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    with pytest.raises(EOFError if primary else BrokenPipeError) as caught:
        with subprocess_stream(("fixture-only",)):
            if primary:
                raise primary
    if primary:
        assert caught.value is primary
    assert events == [("wait", 5), "exit"]
