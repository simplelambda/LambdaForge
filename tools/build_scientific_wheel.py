"""Build a non-editable wheel from the actual working tree with an identity receipt.

No network, dependency installation, git checkout or consumer mutation. Requires installed
setuptools/wheel. SOURCE_DATE_EPOCH is fixed explicitly; dirty/untracked package files are
included by normal setuptools discovery, never a payload copied from another application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path, *, epoch: int = 315532800) -> dict[str, object]:
    destination = destination.absolute()
    destination.mkdir(parents=True, exist_ok=True)
    # Fresh metadata belongs to this working tree, not its last commit or an editable install.
    from build_operation_catalog import SOURCE
    from build_operation_catalog import build as catalog

    declared = json.loads((SOURCE / "catalog/native.json").read_text())
    if declared != json.loads(json.dumps(catalog())):
        raise ValueError("Regenerate native.json with tools/build_operation_catalog.py first.")
    from build_cluster_schema import build as cluster_schema

    profile_schema = json.loads((SOURCE / "schemas/cluster-profile-v1.json").read_text())
    if profile_schema != cluster_schema():
        raise ValueError(
            "Regenerate cluster-profile-v1.json with tools/build_cluster_schema.py first."
        )
    env = {
        **os.environ,
        "SOURCE_DATE_EPOCH": str(epoch),
        "PYTHONHASHSEED": "0",
        "PIP_NO_INDEX": "1",
    }
    code = f"import setuptools.build_meta as b; b.build_wheel({str(destination)!r})"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)
    wheels = list(destination.glob("lambdaforge-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Use a new output directory containing exactly this wheel.")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        members = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(archive.namelist())
            if not name.endswith("/RECORD")
        }
    content = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()
    receipt: dict[str, object] = {
        "receipt_version": "scientific-wheel-v1",
        "framework_version": declared["framework_version"],
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "source_date_epoch": epoch,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "packaged_content_sha256": content,
        "members": members,
    }
    (destination / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epoch", type=int, default=315532800)
    args = parser.parse_args()
    result = build(args.output, epoch=args.epoch)
    print(json.dumps({k: v for k, v in result.items() if k != "members"}))
