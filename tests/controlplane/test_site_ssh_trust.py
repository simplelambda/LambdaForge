"""First-use SSH trust uses disposable keys/HOME; never contacts real SSH hosts."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from lambdaforge.controlplane import SiteSshTrust as trust


@pytest.fixture
def site(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    for name in ("key", "changed"):
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / name)],
            check=True,
            capture_output=True,
        )
    pub = (tmp_path / "key.pub").read_text().split()[:2]
    changed = (tmp_path / "changed.pub").read_text().split()[:2]
    path = home / ".ssh" / "known_hosts"
    run = trust._run
    offered = [pub]
    config = {
        "hostname": "peer.invalid",
        "port": "22",
        "userknownhostsfile": str(path),
        "globalknownhostsfile": str(tmp_path / "global_known_hosts"),
    }

    def execute(command, **kwargs):
        if command[:2] == ["ssh", "-G"]:
            return subprocess.CompletedProcess(
                command, 0, "".join(f"{k} {v}\n" for k, v in config.items()), ""
            )
        if command[0] == "ssh-keyscan":
            assert command[-1] == "peer.invalid"  # No real endpoint/network.
            token = "peer.invalid" if config["port"] == "22" else f"[peer.invalid]:{config['port']}"
            return subprocess.CompletedProcess(
                command, 0, "".join(f"{token} {' '.join(k)}\n" for k in offered), ""
            )
        assert command[0] == "ssh-keygen"
        return run(command, **kwargs)

    monkeypatch.setattr(trust, "_run", execute)
    return {
        "home": home,
        "path": path,
        "pub": pub,
        "changed": changed,
        "offered": offered,
        "config": config,
        "options": {"host": "peer.invalid", "port": 22, "user": "user"},
    }


def approval(site):
    result = trust.inspect(site["options"])
    return {
        **site["options"],
        "fingerprints": [key["fingerprint"] for key in result["keys"]],
        "expected_file": result["known_hosts_file"],
    }


def test_preview_readonly_and_fingerprint_matches_openssh(site, tmp_path):
    result = trust.inspect(site["options"])
    assert result["state"] == "unknown" and result["can_add"]
    assert not list(site["home"].iterdir())
    native = subprocess.run(
        ["ssh-keygen", "-lf", str(tmp_path / "key.pub"), "-E", "sha256"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result["keys"][0]["fingerprint"] in native.stdout


def test_explicit_first_use_hashed_atomic_idempotent(site):
    options = approval(site)
    assert trust.approve(options)["applied"]
    before = site["path"].read_bytes()
    assert before.startswith(b"|1|") and b"peer.invalid" not in before
    assert site["path"].stat().st_mode & 0o777 == 0o600
    assert trust.inspect(site["options"])["state"] == "known"
    assert not trust.approve(options)["applied"]
    assert site["path"].read_bytes() == before
    assert not list(site["path"].parent.glob(".lf-host-trust-*"))


@pytest.mark.parametrize("wrong", [[], ["SHA256:wrong"]])
def test_unapproved_or_mismatched_keys_never_write(site, wrong):
    options = {**approval(site), "fingerprints": wrong}
    with pytest.raises(ValueError, match="required|changed"):
        trust.approve(options)
    assert not site["path"].exists()


def test_offer_changed_since_review_never_writes(site):
    options = approval(site)
    site["offered"][:] = [site["changed"]]
    with pytest.raises(ValueError, match="changed since review"):
        trust.approve(options)
    assert not site["path"].exists()


@pytest.mark.parametrize("revoked", [False, True])
def test_existing_changed_or_revoked_keys_never_replaced(site, revoked):
    options = approval(site)
    site["path"].parent.mkdir(mode=0o700)
    key = site["pub"] if revoked else site["changed"]
    text = ("@revoked " if revoked else "") + "peer.invalid " + " ".join(key) + "\n"
    site["path"].write_text(text)
    assert trust.inspect(site["options"])["state"] == "changed"
    with pytest.raises(ValueError, match="never replacing"):
        trust.approve(options)
    assert site["path"].read_text() == text


def test_append_preserves_unrelated_host_and_known_file_permissions(site):
    site["path"].parent.mkdir(mode=0o700)
    text = "other.invalid " + " ".join(site["pub"])  # No trailing newline.
    site["path"].write_text(text)
    site["path"].chmod(0o640)
    trust.approve(approval(site))
    assert site["path"].read_text().startswith(text + "\n")
    assert site["path"].stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize("unsafe", ["symlink", "writable", "outside", "policy-changed"])
def test_unsafe_paths_and_changed_trust_policy_rejected(site, tmp_path, unsafe):
    options = approval(site)
    site["path"].parent.mkdir(mode=0o700)
    if unsafe == "symlink":
        unrelated = tmp_path / "unrelated"
        unrelated.write_text("untouched")
        site["path"].symlink_to(unrelated)
    elif unsafe == "writable":
        site["path"].parent.chmod(0o777)
    elif unsafe == "outside":
        site["config"]["userknownhostsfile"] = str(tmp_path / "not-home")
    else:
        site["config"]["userknownhostsfile"] = str(site["home"] / "another-trust-file")
    with pytest.raises((ValueError, OSError)):
        trust.approve(options)
    if unsafe == "symlink":
        assert unrelated.read_text() == "untouched"


def test_concurrent_approvals_write_one_key_only(site):
    options = approval(site)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: trust.approve(options), range(2)))
    assert sum(r["applied"] for r in results) == 1
    assert len(site["path"].read_text().splitlines()) == 1


def test_nonstandard_port_uses_exact_host_token(site):
    site["config"]["port"] = "2222"
    result = trust.inspect(site["options"])
    assert result["lookup_host"] == "[peer.invalid]:2222"
    trust.approve(approval(site))
    assert trust.inspect(site["options"])["state"] == "known"


def test_service_sends_helper_to_initiator_never_resolves_remote_paths_locally(monkeypatch):
    from types import SimpleNamespace

    from lambdaforge.controlplane.CommandResult import CommandResult
    from lambdaforge.controlplane.SiteSshTrustService import SiteSshTrustService

    calls = []
    profiles = {
        n: SimpleNamespace(name=n, transport="ssh", host=n + ".invalid", port=22, user="user")
        for n in ("source", "peer")
    }

    class Transport:
        def run(self, command, **kwargs):
            import json

            calls.append((command, kwargs))
            assert command[0] == "/remote/python"
            data = json.loads(command[-1])
            assert data["host"] == "peer.invalid"
            return CommandResult(0, '{"known_hosts_file":"/remote/home/.ssh/known_hosts"}', "")

    transport = Transport()
    service = SimpleNamespace(
        clusters=SimpleNamespace(get=profiles.__getitem__),
        factory=SimpleNamespace(
            transport=lambda p: transport if p.name == "source" else pytest.fail()
        ),
        _python=lambda *_: "/remote/python",
    )
    native = SiteSshTrustService(service)
    result = native.inspect("source", "peer")
    assert result["initiator"] == "source"
    with pytest.raises(ValueError, match="explicitly approve"):
        native.inspect("source", "peer", apply=True)
    assert len(calls) == 1


def test_cli_uses_same_native_host_trust(monkeypatch, capsys):
    from types import SimpleNamespace

    module = importlib.import_module("lambdaforge.cli.DatasetCommands")
    calls = []
    service = SimpleNamespace(
        discovery_warnings=(),
        replication_ssh_trust=lambda *args, **kw: (
            calls.append((args, kw))
            or {
                "initiator": "source",
                "peer": "peer",
                "host": "peer.invalid",
                "port": 22,
                "state": "unknown",
                "known_hosts_file": "/remote/.ssh/known_hosts",
                "keys": [],
                "notice": "Verify independently",
            }
        ),
    )
    monkeypatch.setattr(module.ClusterCatalog, "load", lambda _: None)
    monkeypatch.setattr(module, "DatasetService", lambda **_: service)
    args = SimpleNamespace(
        dataset_command="trust-ssh",
        clusters=None,
        source="source",
        destination="peer",
        apply=False,
        fingerprint=[],
        known_hosts_file=None,
        json=False,
    )
    assert module.DatasetCommands.run(args) == 0
    assert calls[0][0] == ("source", "peer") and not calls[0][1]["apply"]
    assert "No key changed" in capsys.readouterr().out


def test_actual_host_helper_execution_in_disposable_home(site, tmp_path, monkeypatch):
    """Exercise serialization and remote-script entrypoint, with only fake network binaries."""
    from types import SimpleNamespace

    from lambdaforge.controlplane.LocalTransport import LocalTransport
    from lambdaforge.controlplane.SiteSshTrustService import SiteSshTrustService

    binary = tmp_path / "bin"
    binary.mkdir()
    ssh = binary / "ssh"
    keyscan = binary / "ssh-keyscan"
    config = "hostname peer.invalid\nport 22\nuserknownhostsfile " + str(site["path"])
    ssh.write_text(f"#!{sys.executable}\nimport sys\nassert sys.argv[1]=='-G'\nprint({config!r})\n")
    keyscan.write_text(
        f"#!{sys.executable}\nimport sys\nassert sys.argv[-1]=='peer.invalid'\n"
        f"print({'peer.invalid ' + ' '.join(site['pub'])!r})\n"
    )
    ssh.chmod(0o700)
    keyscan.chmod(0o700)
    import os

    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    profiles = {
        n: SimpleNamespace(name=n, transport="ssh", host=n + ".invalid", port=22, user="user")
        for n in ("source", "peer")
    }
    native = SiteSshTrustService(
        SimpleNamespace(
            clusters=SimpleNamespace(get=profiles.__getitem__),
            factory=SimpleNamespace(transport=lambda _: LocalTransport()),
            _python=lambda *_: sys.executable,
        )
    )
    preview = native.inspect("source", "peer")
    assert preview["can_add"] and not site["path"].exists()
    result = native.inspect(
        "source",
        "peer",
        apply=True,
        fingerprints=[k["fingerprint"] for k in preview["keys"]],
        expected_file=preview["known_hosts_file"],
    )
    assert result["applied"] and site["path"].exists()
    assert json.loads(json.dumps(result))["state"] == "known"
