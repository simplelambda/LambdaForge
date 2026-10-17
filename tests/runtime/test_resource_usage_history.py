"""Compression bounds only operational samples; scientific logs/results stay unchanged."""

import gzip
import json

import pytest

from lambdaforge.runtime.ResourceUsageHistory import record_usage


def test_operational_history_compresses_windows_preserves_peaks_and_exact_recent_samples(tmp_path):
    scientific = tmp_path / "training-metrics.jsonl"
    scientific.write_text("scientific unmodified evidence\n")
    samples = [
        {
            "timestamp_utc": f"sample-{index}",
            "rss_bytes": 1000 if index == 1 else 10,
            "process_count": index,
            "padding": "x" * 100,
        }
        for index in range(40)
    ]
    for sample in samples:
        value = record_usage(tmp_path, sample, limit_bytes=1024, retained_samples=2)
    assert value["sample_count"] == 40
    assert value["lifetime_sampled_peaks"]["rss_bytes"] == 1000
    assert value["lifetime_sampled_peaks"]["process_count"] == 39
    assert value["compacted_windows"] > 4 and not value["complete_timeline"]
    archives = list(tmp_path.glob("usage-window-*.jsonl.gz"))
    assert len(archives) == 4
    assert all(
        json.loads(line) in samples
        for path in archives
        for line in gzip.decompress(path.read_bytes()).splitlines()
    )
    assert (tmp_path / "usage.jsonl").stat().st_size < 1024
    assert scientific.read_text() == "scientific unmodified evidence\n"


def test_unknown_measurement_is_not_peak_zero(tmp_path):
    value = record_usage(tmp_path, {"timestamp_utc": "now", "rss_bytes": None})
    assert "rss_bytes" not in value["lifetime_sampled_peaks"]


def test_symlink_cannot_redirect_or_overwrite_evidence(tmp_path):
    protected = tmp_path / "protected"
    protected.write_text("unchanged")
    (tmp_path / "usage.jsonl").symlink_to(protected)
    with pytest.raises(ValueError, match="non-symbolic"):
        record_usage(tmp_path, {})
    assert protected.read_text() == "unchanged"


def test_legacy_oversized_history_is_not_read_deleted_or_grown(tmp_path):
    history = tmp_path / "usage.jsonl"
    history.write_bytes(b"x" * (8 * 1024**2 + 1))
    original = history.stat().st_size
    with pytest.raises(ValueError, match="bounded read"):
        record_usage(tmp_path, {})
    assert history.stat().st_size == original


def test_interrupted_archive_temporary_does_not_disable_compaction(tmp_path):
    leftover = tmp_path / "usage-window-0.jsonl.tmp"
    leftover.write_text("old crash evidence")
    (tmp_path / "usage.jsonl").write_text("{}\n" * 350)
    value = record_usage(tmp_path, {"rss_bytes": 10}, limit_bytes=1024)
    assert value["compacted_windows"] == 1
    assert leftover.read_text() == "old crash evidence"
    assert (tmp_path / "usage-window-0.jsonl.gz").is_file()
