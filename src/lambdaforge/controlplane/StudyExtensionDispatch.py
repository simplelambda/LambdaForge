"""Extension preflight hooks for the ordinary native submission pipeline."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.PreparedWork import PreparedWork
from lambdaforge.work.extension_runtime import (
    check_worker_runtime,
    inspect_retained_runtime,
    runtime_contract,
)


def native_delta_contract(config: Any) -> dict[str, Any]:
    """Placement-independent science; explicit seeds are literal, not a host namespace."""
    import json

    from lambdaforge.work.runner import _identity_values

    if len(config.levels) != 1 or len(config.levels[0].runs) != 1:
        raise ValueError("Prepared extension must contain one native Work.")
    definition = config.levels[0].runs[0]
    if (
        definition.study_design is None or definition.study_design.replication != "fixed"
        or definition.study_design.kind != "sweep"
        or definition.study_design.seed_source.get("kind") != "explicit"
    ):
        raise ValueError("Prepared extension must have an explicit fixed design.")
    design = json.loads(json.dumps(definition.study_design.to_dict()))
    # Only the host-derived namespace differs; exact authored seed values remain pinned.
    design["seed_source"].pop("namespace", None)
    for seed in design["seed_source"].get("resolved", []):
        seed.pop("namespace", None)
    return {
        "name": config.name,
        "work_name": definition.name,
        "work_class": definition.work_class,
        "parameters": _identity_values(definition.parameters, config.source.parent),
        "variants": [dict(value) for value in definition.variants],
        "seeds": list(definition.seeds), "objective": dict(definition.objective or {}),
        "resources": definition.resources.to_dict(),
        "execution": definition.execution_policy.to_dict(),
        "study_design": design,
        "analysis_semantics": dict(definition.analysis_semantics),
        "products": json.loads(json.dumps(config.raw.get("products"))),
    }


def delta_contract_digests(contract: dict[str, Any]) -> dict[str, str]:
    """Exact canonical field commitments, not metadata embedded in SSH command arguments."""
    import hashlib
    import json

    return {
        key: hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        for key, value in contract.items()
    }


def inspect_prepared_delta(source: str, expected: dict[str, str]) -> dict[str, str]:
    """Read-only native validation on the actual execution host, no scientific Run."""
    import hashlib
    from pathlib import Path

    from lambdaforge.work.config import WorkConfig
    from lambdaforge.work.recovery import read_owned_json
    from lambdaforge.work.runner import WorkRunner

    path = Path(source)
    if path.resolve() != path or path.is_symlink():
        raise ValueError("Unsafe prepared extension configuration.")
    config = WorkConfig.from_yaml(path)
    actual = delta_contract_digests(native_delta_contract(config))
    if actual != expected:
        changed = [
            key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
        ]
        raise ValueError("Prepared extension differs from reviewed " + ", ".join(sorted(changed)))
    code = read_owned_json(path.parent / "code-identity.json")
    return {
        "scientific_identity": WorkRunner._study_identity(config, code_identity=code),
        "configuration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(
            (path.parent / "code-identity.json").read_bytes()
        ).hexdigest(),
    }


def prepare_native_delta(prepared: PreparedWork, expected: Mapping[str, Any]) -> dict[str, str]:
    """Use existing transport and provider storage; never copy model/artifact bytes."""
    profile = prepared.profile
    assert profile.storage is not None
    script = (
        "from __future__ import annotations\nfrom typing import Any\nimport json,sys\n"
        + inspect.getsource(native_delta_contract) + "\n"
        + inspect.getsource(delta_contract_digests) + "\n"
        + inspect.getsource(inspect_prepared_delta)
        + "\nprint(json.dumps(inspect_prepared_delta(sys.argv[1],json.loads(sys.argv[2]))))\n"
    )
    response = prepared.transport.run((
        "env", f"LAMBDAFORGE_CLUSTER={profile.name}",
        f"LAMBDAFORGE_DATASET_REGISTRY={profile.storage.state_root}/datasets.json",
        f"LAMBDAFORGE_DATASET_ROOT={profile.storage.dataset_root or ''}",
        f"LAMBDAFORGE_PRODUCT_ROOT={profile.storage.product_root}",
        f"LAMBDAFORGE_RESULT_INPUT_ROOT={profile.storage.state_root}/results",
        f"LAMBDAFORGE_STORAGE_POLICY={json.dumps(profile.storage.to_dict())}",
        prepared.python, "-c", script, prepared.config,
        json.dumps(delta_contract_digests(dict(expected)), separators=(",", ":")),
    ), timeout=60)
    if response.returncode:
        raise ValueError("Cannot attest prepared native extension: " + response.stderr[-1800:])
    value = json.loads(response.stdout)
    if (
        not isinstance(value, dict)
        or set(value) != {"scientific_identity", "configuration_sha256", "code_sha256"}
        or not isinstance(value["scientific_identity"], str)
        or not value["scientific_identity"].startswith("sha256:")
        or any(
            not isinstance(value[key], str) or len(value[key]) != 64
            or any(char not in "0123456789abcdef" for char in value[key])
            for key in ("configuration_sha256", "code_sha256")
        )
    ):
        raise ValueError("Invalid native extension attestation.")
    return value


def _configuration_guard(attestation: Mapping[str, str]) -> str:
    """Indented integrity checks shared by both native dispatch preflights."""
    return (
        " import hashlib\n from pathlib import Path\n"
        " config=Path(sys.argv[1])\n"
        " if config.resolve()!=config or hashlib.sha256(config.read_bytes()).hexdigest()!="
        + repr(attestation["configuration_sha256"]) + ":\n"
        "  raise ValueError('Prepared extension configuration changed')\n"
        " code=config.parent/'code-identity.json'\n"
        " if code.resolve()!=code or hashlib.sha256(code.read_bytes()).hexdigest()!="
        + repr(attestation["code_sha256"]) + ":\n"
        "  raise ValueError('Prepared extension code identity changed')\n"
    )


def current_entrypoint(
    prepared: PreparedWork, attestation: Mapping[str, str], run_arguments: Sequence[str] = ()
) -> Sequence[str]:
    """Integrity preflight followed by the unchanged native CLI/WorkRunner."""
    module = "_lambdaforge_current_extension"
    script = (
        "import os,sys\n"
        "if __name__=='__main__':\n"
        " if os.environ.get('LAMBDAFORGE_EXECUTION_MODE')!='worker':\n"
        "  raise ValueError('Expected native worker mode')\n"
        + _configuration_guard(attestation)
        +
        " from lambdaforge.cli.CommandLineInterface import CommandLineInterface\n"
        " raise SystemExit(CommandLineInterface.main(['run',*sys.argv[1:]]))\n"
    )
    path = str(PurePosixPath(prepared.work_dir) / (module + ".py"))
    # Atomic no-overwrite creation in the already owned Job workspace.
    write = (
        "import os,sys,hashlib\nfrom pathlib import Path\n"
        "p=Path(sys.argv[1]); raw=sys.argv[2].encode()\n"
        "if p.resolve()!=p: raise ValueError('Unsafe extension guard')\n"
        "try: fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)\n"
        "except FileExistsError:\n"
        " if p.read_bytes()!=raw: raise ValueError('Changed extension guard')\n"
        "else:\n"
        " with os.fdopen(fd,'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())\n"
    )
    if not prepared.dry_run:
        staged = prepared.transport.run((prepared.python, "-c", write, path, script), timeout=30)
        if staged.returncode:
            raise ValueError(
                "Could not stage extension integrity preflight: " + staged.stderr[-1000:]
            )
    return (module, prepared.config, *run_arguments)


def retained_profile(profile: ClusterProfile, proof: Mapping[str, Any]) -> ClusterProfile:
    """Narrow only the interpreter to an owned immutable prefix; keep provider/GPU policy."""
    assert profile.storage is not None
    environment_id = str(proof["environment_id"])
    if not environment_id.startswith("env-") or "/" in environment_id or ".." in environment_id:
        raise ValueError("Unsafe retained environment identity.")
    prefix = PurePosixPath(profile.storage.environment_root) / environment_id
    if str(prefix) != proof.get("prefix") or str(prefix / "bin/python") != proof.get("python"):
        raise ValueError("Extension runtime is outside its owned same-placement cache.")
    return replace(
        profile, python=str(proof["python"]), python_runtime=None, environment="existing"
    )


def prepared_entrypoint(
    prepared: PreparedWork,
    proof: Mapping[str, Any],
    run_arguments: Sequence[str] = (),
    *,
    attestation: Mapping[str, str] | None = None,
) -> Sequence[str]:
    """A numerical preflight followed by the unchanged native CLI/WorkRunner.

    The tiny guard is staged by the trusted control plane in this exact Job workspace.
    ControlPlane still adds TLS, site commands, scheduler resources and GPU grants. It is
    not another runner, and contains no scheduling, recovery or scientific decision logic.
    """
    if prepared.python != proof["python"]:
        raise ValueError("Prepared extension interpreter differs from the reviewed runtime.")
    probe = (
        "from __future__ import annotations\nfrom typing import Any\nimport json,sys\n"
        + inspect.getsource(inspect_retained_runtime)
        + "\nprint(json.dumps(inspect_retained_runtime(*json.loads(sys.argv[1]))))\n"
    )
    response = prepared.transport.run(
        (
            prepared.python,
            "-c",
            probe,
            json.dumps(
                [
                    proof["prefix"],
                    proof["environment_id"],
                    [proof["contract"]],
                    bool(proof.get("sparse_sweeps")),
                ]
            ),
        ),
        timeout=30,
    )
    if response.returncode or json.loads(response.stdout) != dict(proof):
        raise ValueError("Retained extension environment changed before native dispatch.")
    name = "_lambdaforge_extension_guard"
    script = (
        "from __future__ import annotations\nfrom typing import Any\nimport json,sys,os\n"
        + inspect.getsource(runtime_contract)
        + "\n"
        + inspect.getsource(check_worker_runtime)
        + "\nif __name__=='__main__':\n"
        " if os.environ.get('LAMBDAFORGE_EXECUTION_MODE')!='worker':\n"
        "  raise ValueError('Extension preflight requires the native worker execution mode')\n"
        + (_configuration_guard(attestation) if attestation is not None else "")
        +
        " check_worker_runtime(json.loads("
        + repr(json.dumps(proof["contract"], sort_keys=True, separators=(",", ":")))
        + "))\n"
        " from lambdaforge.cli.CommandLineInterface import CommandLineInterface\n"
        " raise SystemExit(CommandLineInterface.main(['run',*sys.argv[1:]]))\n"
    )
    path = str(PurePosixPath(prepared.work_dir) / (name + ".py"))
    write = (
        "import os,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[1]); raw=sys.argv[2].encode()\n"
        "if p.parent.resolve()!=p.parent or p.is_symlink(): raise ValueError('Unsafe guard path')\n"
        "try: fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)\n"
        "except FileExistsError:\n"
        " if p.read_bytes()!=raw: raise ValueError('Changed extension guard')\n"
        "else:\n"
        " with os.fdopen(fd,'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())\n"
    )
    if not prepared.dry_run:
        staged = prepared.transport.run((prepared.python, "-c", write, path, script), timeout=30)
        if staged.returncode:
            raise ValueError("Could not stage native extension preflight: " + staged.stderr[-1000:])
    return (name, prepared.config, *run_arguments)
