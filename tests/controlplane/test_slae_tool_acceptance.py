"""Opt-in real CPU import/checkpoint acceptance; never installs or runs extraction."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from lambdaforge.work.tools import ToolService


def test_reviewed_slae_base_checkpoint_cpu(monkeypatch):
    bindings = os.environ.get("LAMBDAFORGE_TEST_TOOL_BINDINGS")
    checkpoint = os.environ.get("LAMBDAFORGE_TEST_SLAE_CHECKPOINT")
    if not bindings or not checkpoint:
        pytest.skip("Requires explicitly prepared disposable SLAE bindings and reviewed checkpoint")
    path = Path(checkpoint)
    expected = "ac5bf9613209fc310145e58f1aa7ddc9111d833a131488fe30ea738d51f7bb99"
    assert path.stat().st_size == 882525854
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(part)
    assert digest.hexdigest() == expected
    monkeypatch.setenv("LAMBDAFORGE_TOOL_RUNTIMES_PATH", bindings)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    class Log:
        def emit(self, message, **kwargs):
            pass

    tool = ToolService(Log(), max_threads=2).runtime("slae_cpu")
    assert tool.binding["policy"]["torch_cuda"] == "cpu"
    result = tool.run(
        [
            "python",
            "-c",
            "import importlib.metadata as m,importlib.util,json,sys,torch;"
            "assert importlib.util.find_spec('lambdaforge') is None;"
            "assert importlib.util.find_spec('wisdom') is None;from SLAE import load_autoencoder;"
            "assert torch.version.cuda is None;model=load_autoencoder(sys.argv[1],device='cpu');"
            "print('ACCEPTANCE '+json.dumps({n:m.version(n) for n in "
            "['SLAE','torch','numpy','esm','transformers','torch-geometric']}))",
            path,
        ],
        timeout=180,
    )
    record = next(
        line[11:] for line in result.stdout.splitlines() if line.startswith("ACCEPTANCE ")
    )
    versions = json.loads(record)
    assert versions["torch"] == "2.6.0+cpu" and versions["transformers"] == "4.48.1"
    assert versions["numpy"] == "1.26.4" and versions["esm"] == "3.2.1"
