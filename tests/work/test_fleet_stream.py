"""Ordered native scalar evidence survives retransmission and receiver interruption."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.work import fleet_stream
from lambdaforge.work.fleet_stream import CHANNELS, ScalarStream, read_chunk


def identity() -> dict[str, Any]:
    return {
        "study_identity": "science",
        "run_key": "sha256:" + "a" * 64,
        "attempt": 1,
        "lease_id": "lease",
        "cluster": "A",
        "shard_id": "b" * 64,
    }


def payload(source: Path, receiver: ScalarStream, *, terminal: bool) -> dict[str, Any]:
    cursors = receiver.cursors()
    return {
        "stream_version": 1,
        "identity": receiver.identity,
        "channels": {
            channel: read_chunk(source / (channel + ".jsonl"), cursors[channel], terminal=terminal)
            for channel in CHANNELS
        },
    }


def test_ordered_bounded_channels_replay_and_complete(tmp_path: Path) -> None:
    source = tmp_path / "native"
    source.mkdir()
    raw = b"".join(
        json.dumps({"name": "score", "value": step / 1000, "step": step}).encode() + b"\n"
        for step in range(3000)
    )
    for channel in CHANNELS:
        (source / (channel + ".jsonl")).write_bytes(raw)
    receiver = ScalarStream(tmp_path / "mirror", identity())
    first = payload(source, receiver, terminal=True)
    assert not receiver.ingest(first)
    assert not receiver.ingest(first)  # exact retransmission, never duplicate evidence
    assert receiver.path("metrics").stat().st_size <= fleet_stream.CHUNK_BYTES
    resumed = ScalarStream(receiver.root, identity())
    for _ in range(20):
        if resumed.ingest(payload(source, resumed, terminal=True)):
            break
    else:
        pytest.fail("Bounded reads did not finish")
    assert all(resumed.path(channel).read_bytes() == raw for channel in CHANNELS)
    assert resumed.ingest(first)  # a delayed old read cannot undo terminal completeness


def test_unfinished_record_is_not_published_and_terminal_truncation_fails(tmp_path: Path) -> None:
    source = tmp_path / "native"
    source.mkdir()
    path = source / "metrics.jsonl"
    path.write_bytes(b'{"name":"score","value":0.5')
    receiver = ScalarStream(tmp_path / "mirror", identity())
    assert not receiver.ingest(payload(source, receiver, terminal=False))
    assert receiver.path("metrics").read_bytes() == b""
    with pytest.raises(ValueError, match="incomplete"):
        payload(source, receiver, terminal=True)
    path.write_bytes(path.read_bytes() + b',"step":1}\n')
    assert receiver.ingest(payload(source, receiver, terminal=True))
    assert json.loads(receiver.path("metrics").read_text())["step"] == 1


def test_receiver_recovers_uncommitted_append_without_guessing_a_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "native"
    source.mkdir()
    (source / "metrics.jsonl").write_bytes(b'{"name":"score","value":0.5,"step":1}\n')
    receiver = ScalarStream(tmp_path / "mirror", identity())
    original = fleet_stream.atomic_write_json
    calls = 0

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated crash before cursor commit")
        return original(*args, **kwargs)

    chunk = payload(source, receiver, terminal=True)
    monkeypatch.setattr(fleet_stream, "atomic_write_json", interrupted)
    with pytest.raises(OSError, match="simulated crash"):
        receiver.ingest(chunk)
    monkeypatch.setattr(fleet_stream, "atomic_write_json", original)
    restored = ScalarStream(receiver.root, identity())
    assert restored.cursors() == dict.fromkeys(CHANNELS, 0)
    assert restored.ingest(chunk)
    assert restored.path("metrics").read_bytes() == (source / "metrics.jsonl").read_bytes()
    (receiver.root / "stream.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        restored.cursors()


@pytest.mark.parametrize("mutation", ["lease", "digest", "gap", "replay", "oversized"])
def test_corrupt_or_foreign_evidence_is_rejected_before_publication(
    tmp_path: Path, mutation: str
) -> None:
    source = tmp_path / "native"
    source.mkdir()
    raw = b'{"name":"score","value":0.5,"step":1}\n'
    (source / "metrics.jsonl").write_bytes(raw)
    receiver = ScalarStream(tmp_path / "mirror", identity())
    chunk = payload(source, receiver, terminal=True)
    if mutation == "lease":
        chunk["identity"] = {**identity(), "lease_id": "foreign"}
    elif mutation == "digest":
        chunk["channels"]["metrics"]["sha256"] = "0" * 64
    elif mutation == "gap":
        metrics = chunk["channels"]["metrics"]
        for field in ("offset", "next_offset", "size"):
            metrics[field] += 1
    elif mutation == "oversized":
        chunk["channels"]["metrics"]["data"] = base64.b64encode(
            b"x" * (fleet_stream.CHUNK_BYTES + 1)
        ).decode()
    else:
        receiver.ingest(chunk)
        receiver.path("metrics").write_bytes(raw.replace(b"0.5", b"0.6"))
    with pytest.raises(ValueError):
        receiver.ingest(chunk)
    if mutation != "replay":
        assert not receiver.path("metrics").exists()


def test_missing_and_symlinked_native_evidence_is_not_followed(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    assert read_chunk(path, 0, terminal=True)["complete"]
    path.symlink_to(tmp_path / "other")
    with pytest.raises(ValueError, match="symlink"):
        read_chunk(path, 0, terminal=True)
    path.unlink()
    path.write_text('{"name":"score","value":NaN}\n')
    with pytest.raises(ValueError):
        read_chunk(path, 0, terminal=True)


def test_native_lightning_chart_annotations_survive_streaming(tmp_path: Path) -> None:
    from lambdaforge.training.callbacks.AdaptiveHpoCallback import AdaptiveHpoCallback

    source = tmp_path / "native"
    source.mkdir()
    callback = AdaptiveHpoCallback(
        metric=None,
        metrics_path=None,
        stop_path=None,
        training_metrics_path=source / "training-metrics.jsonl",
        chart_include=("accuracy",),
        display_names={"accuracy": "Accuracy"},
    )
    callback._write_training({"accuracy": 0.9}, step=1)
    receiver = ScalarStream(tmp_path / "mirror", identity())
    assert receiver.ingest(payload(source, receiver, terminal=True))
    assert (
        receiver.path("training-metrics").read_bytes()
        == (source / "training-metrics.jsonl").read_bytes()
    )
