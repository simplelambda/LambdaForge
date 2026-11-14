"""Provision isolated scientific tools through native runtime and storage boundaries."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion

from lambdaforge.controlplane.CacheBuildLease import CacheBuildLease
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.python_runtime import PythonRuntimePolicy
from lambdaforge.controlplane.PythonRuntimeResolver import PythonRuntimeResolver
from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.controlplane.TlsTrust import TlsTrustResolver
from lambdaforge.controlplane.ToolRuntimeSpecification import ToolRuntimeSpecification
from lambdaforge.controlplane.Transport import Transport


def verification_code() -> str:
    """Probe with stdlib only, before importing scientific packages; never import pip."""
    return """import importlib,importlib.metadata,json,os,sys,pathlib,re
p=json.loads(sys.argv[1]); required=sys.argv[2]=='true'
versions={}
for requirement in p['verification_requirements']:
 name=requirement['name']
 versions[name]=importlib.metadata.version(name)
 if requirement['url']:
  direct=importlib.metadata.distribution(name).read_text('direct_url.json')
  source=json.loads(direct or '{}')
  base,commit=requirement['url'][4:].rsplit('@',1)
  if source.get('url')!=base or source.get('vcs_info',{}).get('commit_id')!=commit:
   raise RuntimeError(f'Pinned tool source mismatch: {name}')
marker=pathlib.Path(sys.executable).parent.parent/'.lambdaforge-environment.json'
if marker.is_file():
 def canonical(name):return re.sub('[-_.]+','-',name).lower()
 inventory=json.loads(marker.read_text())['installed_packages']
 expected={canonical(v['name']):v['version'] for v in inventory}
 actual={canonical(v.metadata['Name']):v.version for v in importlib.metadata.distributions()}
 if expected!=actual:
  raise RuntimeError('Immutable tool package inventory changed; bootstrap/review it')
if 'torch_cuda' in p:
 import torch
 cuda=torch.version.cuda
 expected=None if p['torch_cuda']=='cpu' else p['torch_cuda']
 if cuda!=expected:
  raise RuntimeError(f'Torch CUDA build mismatch: expected {expected}, got {cuda}')
 if required:
  if not os.environ.get('CUDA_VISIBLE_DEVICES','').strip():
   raise RuntimeError('CUDA tool requires inherited provider-owned CUDA_VISIBLE_DEVICES grants')
  if not torch.cuda.is_available():
   raise RuntimeError('Granted CUDA runtime is unavailable; no CPU fallback')
  probe=torch.ones(1,device='cuda'); assert probe.item()==1
for module in p['imports']: importlib.import_module(module)
for source in p['wheel_sources']:
 for name in source['packages']:
  version=importlib.metadata.version(name)
  if not version.endswith('+'+source['build_tag']):
   raise RuntimeError(f'ABI wheel mismatch for {name}: {version}')
print(json.dumps({'imports':p['imports'],'cuda_required':required,'versions':versions}))
"""


def verification_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Parse requirements in LF, not via installer internals in the scientific interpreter."""
    requirements = [Requirement(text) for text in policy["requirements"]]
    return {
        **policy,
        "verification_requirements": [
            {"name": value.name, "url": value.url} for value in requirements
        ],
    }


def validate_verification(policy: Mapping[str, Any], stdout: str) -> None:
    """Apply full PEP 440 pin semantics to the isolated interpreter's metadata report."""
    # The probe writes its report after scientific imports, which may themselves print.
    try:
        report = json.loads(stdout.strip().splitlines()[-1])
        versions = report["versions"]
        if not isinstance(versions, dict):
            raise ValueError("Tool version inventory is not a mapping.")
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Isolated tool verification returned an invalid metadata report."
        ) from error
    for text in policy["requirements"]:
        requirement = Requirement(text)
        version = versions.get(requirement.name)
        try:
            valid = isinstance(version, str) and (
                requirement.url is not None or requirement.specifier.contains(version)
            )
        except InvalidVersion:
            valid = False
        if not valid:
            raise RuntimeError(
                f"Pinned tool package mismatch: {requirement.name} {version}; expected {text}"
            )


class ToolRuntimeService:
    """Install no framework/consumer code into the isolated environment; never run a Work."""

    def __init__(self, resolver: PythonRuntimeResolver | None = None) -> None:
        self.resolver = resolver or PythonRuntimeResolver()

    @staticmethod
    def plan(specs: Sequence[ToolRuntimeSpecification]) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                **spec.descriptor(),
                "status": "declared; exact target resolution follows provisioning",
            }
            for spec in specs
        )

    def prepare(
        self,
        specs: Sequence[ToolRuntimeSpecification],
        profile: ClusterProfile,
        transport: Transport,
        *,
        progress: Callable[[str], None] | None = None,
        retain: ExitStack | None = None,
    ) -> tuple[dict[str, Any], ...]:
        result = []
        for spec in specs:
            if progress:
                progress(
                    f"Preparing isolated tool {spec.name!r}; consumer dependencies remain separate."
                )
            try:
                result.append(self._prepare(spec, profile, transport, retain=retain))
            except (RuntimeError, ValueError) as error:
                raise RuntimeError(
                    f"Isolated tool {spec.name!r} is not ready: {error}. "
                    f"Review its declaration and run lf clusters bootstrap {profile.name} "
                    "--project ."
                ) from error
        return tuple(result)

    def _prepare(
        self,
        spec: ToolRuntimeSpecification,
        profile: ClusterProfile,
        transport: Transport,
        *,
        retain: ExitStack | None = None,
    ) -> dict[str, Any]:
        assert profile.storage is not None
        if profile.transport == "local":
            profile = replace(
                profile,
                storage=replace(
                    profile.storage,
                    state_root=str(Path(profile.storage.state_root).absolute()),
                    cache_root=str(Path(profile.storage.cache_root).absolute()),
                    run_root=str(Path(profile.storage.run_root).absolute()),
                ),
            )
        assert profile.storage is not None
        policy = PythonRuntimePolicy(
            "auto",
            profile.python,
            str(spec.policy["python"]),
            allow_managed_install=profile.runtime_policy.allow_managed_install,
        )
        selected = replace(profile, python_runtime=policy)
        runtime = self.resolver.resolve(
            selected, transport, requirements=(f"=={spec.policy['python']}.*",)
        )
        trust = runtime.tls_trust or TlsTrustResolver().resolve(
            profile, transport, host_python=runtime.executable
        )
        runtime = replace(runtime, tls_trust=trust)
        if runtime.system not in {"Linux", "Darwin"}:
            raise RuntimeError("Managed isolated tools currently require a POSIX execution host.")
        identity_data = {
            "contract": dict(spec.policy),
            "python": runtime.version,
            "runtime": runtime.runtime_id,
            "system": runtime.system,
            "architecture": runtime.architecture,
        }
        identity = (
            "env-tool-"
            + hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()[:24]
        )
        cache = PurePosixPath(profile.storage.cache_root)
        prefix = PurePosixPath(profile.storage.environment_root) / identity
        marker = prefix / ".lambdaforge-environment.json"
        lock = cache / f".environment-build-{identity}.lock"
        made = transport.run(
            ("mkdir", "-p", str(cache), str(prefix.parent), profile.storage.state_root)
        )
        if made.returncode:
            raise RuntimeError(made.stderr)
        CacheBuildLease.acquire(
            transport,
            lock,
            python=profile.python,
            timeout=3600,
            references={"environments": [identity], "runtimes": [runtime.runtime_id]},
        )
        effective = replace(profile, python=runtime.executable)
        layer = {"identity": identity, "contracts": [], "resolution_policy": identity_data}
        with (
            CacheBuildLease.maintain(transport, lock, python=profile.python),
            ExitStack() as staging,
        ):
            frozen = ManagedEnvironmentProvider._dependency_resolution(
                effective, transport, prefix, layer, seal=False
            )
            reused = transport.run(("test", "-f", str(marker))).returncode == 0
            if reused:
                self._checked(transport, (str(prefix / "bin/python"), "-I", "-m", "pip", "check"))
                resolution = ManagedEnvironmentProvider._dependency_resolution(
                    effective, transport, prefix, layer, seal=True
                )
            else:
                temporary = prefix.parent / f".{identity}.tmp-{uuid4().hex}"
                try:
                    self._checked(
                        transport,
                        (runtime.executable, "-I", "-m", "venv", "--symlinks", str(temporary)),
                    )
                    python = str(temporary / "bin/python")
                    base = [
                        "env",
                        "GIT_LFS_SKIP_SMUDGE=1",
                        *trust.assignments(),
                        python,
                        "-I",
                        "-m",
                        "pip",
                        "--disable-pip-version-check",
                        "install",
                        "--cache-dir",
                        str(cache / "pip"),
                        "--only-binary=:all:",
                    ]
                    pin_args: list[str] = []
                    if frozen:
                        pins = temporary / "frozen-constraints.txt"
                        self._write(
                            transport,
                            python,
                            pins,
                            "\n".join(f"{v['name']}=={v['version']}" for v in frozen["packages"])
                            + "\n",
                        )
                        pin_args = ["--constraint", str(pins)]
                    requirements = list(spec.policy["requirements"])
                    torch = next(
                        (
                            r
                            for r in requirements
                            if canonicalize_name(Requirement(r).name) == "torch"
                        ),
                        None,
                    )
                    if torch:
                        self._checked(
                            transport,
                            (*base, *pin_args, "--index-url", spec.policy["torch_index"], torch),
                        )
                    command = [*base, *pin_args, "--index-url", spec.policy["index_url"]]
                    for index in spec.policy["extra_index_urls"]:
                        command.extend(("--extra-index-url", index))
                    for source in spec.policy["wheel_sources"]:
                        command.extend(("--find-links", source["url"]))
                    if spec.policy["source_packages"]:
                        command.append("--no-binary=" + ",".join(spec.policy["source_packages"]))
                    report = temporary / "resolution-report.json"
                    self._checked(
                        transport, (*command, "--dry-run", "--report", str(report), *requirements)
                    )
                    raw = self._checked(
                        transport,
                        (
                            python,
                            "-I",
                            "-c",
                            "import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())",
                            str(report),
                        ),
                    )
                    self.validate_sources(spec.policy, json.loads(raw))
                    self._checked(transport, (*command, "--report", str(report), *requirements))
                    installed_report = self._checked(
                        transport,
                        (
                            python,
                            "-I",
                            "-c",
                            "import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())",
                            str(report),
                        ),
                    )
                    self.validate_sources(spec.policy, json.loads(installed_report))
                    self._checked(transport, (python, "-I", "-m", "pip", "check"))
                    self._verify(spec.policy, transport, python)
                    inventory = json.loads(
                        self._checked(
                            transport, (python, "-I", "-m", "pip", "list", "--format=json")
                        )
                    )
                    self._write(
                        transport,
                        python,
                        temporary / ".lambdaforge-environment.json",
                        json.dumps(
                            {
                                "environment_id": identity,
                                "layer_kind": "isolated-tool",
                                "tool_contract": spec.contract_id,
                                "installed_packages": inventory,
                                "environment_policy": {
                                    "python_runtime": runtime.to_dict(),
                                    "tool_policy": dict(spec.policy),
                                },
                            }
                        ),
                    )
                    resolution = ManagedEnvironmentProvider._dependency_resolution(
                        effective, transport, temporary, layer, seal=True
                    )
                    assert resolution is not None
                    self._checked(
                        transport,
                        (
                            python,
                            "-I",
                            "-c",
                            "import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);"
                            "v=json.loads(p.read_text());"
                            "v['dependency_resolution_id']=sys.argv[2];p.write_text(json.dumps(v))",
                            str(temporary / ".lambdaforge-environment.json"),
                            str(resolution["resolution_id"]),
                        ),
                    )
                    # Python -I uses the final prefix directly; console-script shebangs must too.
                    publish = """import os,pathlib,sys
p=pathlib.Path(sys.argv[1]); final=pathlib.Path(sys.argv[2])
for item in (p/'bin').iterdir():
 if item.is_symlink() or not item.is_file(): continue
 with item.open('rb') as f: first=f.read(512)
 if first.startswith(b'#!') and str(p).encode() in first:
  lines=item.read_bytes().splitlines(keepends=True)
  item.write_bytes(b''.join(lines[:4]).replace(str(p).encode(),str(final).encode())+b''.join(lines[4:]))
os.rename(p,final)
"""
                    self._checked(
                        transport,
                        (runtime.executable, "-I", "-c", publish, str(temporary), str(prefix)),
                    )
                finally:
                    ManagedEnvironmentProvider._cleanup(transport, temporary)
            self._verify(spec.policy, transport, str(prefix / "bin/python"))
            assert resolution is not None
            code_bundle = self._code_id(spec)
            staging.enter_context(
                CacheBuildLease.preparation(
                    transport,
                    cache_root=cache,
                    environment=identity,
                    bundle_id=code_bundle,
                    job_id="tool-stage-" + uuid4().hex,
                    work_dir=str(prefix),
                    python=profile.python,
                    runtime_id=runtime.runtime_id,
                )
            )
            entries, code_bundle = self._stage_entrypoints(spec, profile, transport)
            binding = {
                "name": spec.name,
                "environment_id": identity,
                "python": str(prefix / "bin/python"),
                "resolution_id": resolution["resolution_id"],
                "contract_id": spec.contract_id,
                "policy": dict(spec.policy),
                "entrypoints": entries,
                "entrypoint_bundle_id": code_bundle,
                "reused": reused,
                "cluster": profile.name,
                "tls_trust": trust.to_dict(),
            }
            self._register(profile, transport, binding)
            if retain is not None:
                retain.enter_context(
                    CacheBuildLease.preparation(
                        transport,
                        cache_root=cache,
                        environment=identity,
                        bundle_id=code_bundle,
                        job_id="tool-use-" + uuid4().hex,
                        work_dir=str(prefix),
                        python=profile.python,
                        runtime_id=runtime.runtime_id,
                    )
                )
            return binding

    @staticmethod
    def _checked(transport: Transport, command: Sequence[str]) -> str:
        result = transport.run(tuple(command), timeout=1800)
        if result.returncode:
            diagnostic = SecretRedactor.redact(result.stderr[-12000:] or result.stdout[-12000:])
            raise RuntimeError(
                f"Tool provisioning command failed ({result.returncode}): {diagnostic}"
            )
        return result.stdout

    @classmethod
    def _write(cls, transport: Transport, python: str, path: PurePosixPath, content: str) -> None:
        cls._checked(
            transport,
            (
                python,
                "-I",
                "-c",
                "from pathlib import Path;import sys;Path(sys.argv[1]).write_text(sys.argv[2])",
                str(path),
                content,
            ),
        )

    @classmethod
    def _verify(cls, policy: Mapping[str, Any], transport: Transport, python: str) -> None:
        report = cls._checked(
            transport,
            (
                python,
                "-I",
                "-c",
                verification_code(),
                json.dumps(verification_policy(policy)),
                "false",
            ),
        )
        validate_verification(policy, report)

    @staticmethod
    def validate_sources(policy: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        """Reject unexpected source builds and companion wheels from a different ABI/source."""
        explicit = {
            canonicalize_name(Requirement(v).name): Requirement(v) for v in policy["requirements"]
        }
        allowed_source = set(policy["source_packages"])
        for item in report.get("install", ()):
            name = canonicalize_name(item["metadata"]["name"])
            info = item.get("download_info", {})
            url = unquote(str(info.get("url", "")))
            if "vcs_info" in info:
                requested = explicit.get(name)
                commit = info["vcs_info"].get("commit_id")
                if not requested or not requested.url or requested.url.rsplit("@", 1)[-1] != commit:
                    raise ValueError(f"Unpinned/unexpected VCS tool dependency: {name}")
            elif not urlsplit(url).path.endswith(".whl") and name not in allowed_source:
                raise ValueError(
                    f"No compatible binary wheel for {name}; source compilation is forbidden."
                )
            for source in policy["wheel_sources"]:
                if name in map(canonicalize_name, source["packages"]):
                    origin = urlsplit(source["url"])
                    found = urlsplit(url)
                    if found.netloc != origin.netloc or not found.path.startswith(
                        str(PurePosixPath(origin.path).parent) + "/"
                    ):
                        raise ValueError(
                            f"Unexpected binary wheel origin for {name}: {found.netloc}"
                        )
                    if not item["metadata"]["version"].endswith("+" + source["build_tag"]):
                        raise ValueError(f"Wrong ABI wheel for {name}")

    def _stage_entrypoints(
        self, spec: ToolRuntimeSpecification, profile: ClusterProfile, transport: Transport
    ) -> tuple[dict[str, Any], str]:
        assert profile.storage is not None
        code_id = self._code_id(spec)
        lock = PurePosixPath(profile.storage.cache_root) / f".environment-build-{code_id}.lock"
        CacheBuildLease.acquire(
            transport, lock, python=profile.python, timeout=3600, references={"bundles": [code_id]}
        )
        with CacheBuildLease.maintain(transport, lock, python=profile.python):
            return self._stage_entrypoints_locked(spec, profile, transport)

    def _stage_entrypoints_locked(
        self, spec: ToolRuntimeSpecification, profile: ClusterProfile, transport: Transport
    ) -> tuple[dict[str, Any], str]:
        assert profile.storage is not None
        descriptors = spec.descriptor()["entrypoints"]
        code_id = self._code_id(spec)
        remote = PurePosixPath(profile.storage.bundle_root) / code_id
        cached = transport.run(("test", "-f", str(remote / "manifest.json"))).returncode == 0
        if cached:
            return {
                name: {"path": str(remote / f"{name}.py"), "sha256": value["sha256"]}
                for name, value in descriptors.items()
            }, code_id
        self._checked(transport, ("mkdir", "-p", str(remote.parent)))
        with tempfile.TemporaryDirectory(prefix="lf-tool-code-") as scratch:
            root = Path(scratch)
            for name, path in spec.entrypoints.items():
                if path.is_symlink():
                    raise ValueError("Tool entrypoint changed into a symlink during staging.")
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != descriptors[name]["sha256"]:
                    raise ValueError(
                        "Tool source changed during preparation; retry with a coherent snapshot."
                    )
                (root / f"{name}.py").write_bytes(content)
            (root / "manifest.json").write_text(json.dumps(descriptors))
            temporary = remote.parent / f".{code_id}.tmp-{uuid4().hex}"
            try:
                transport.put(root, str(temporary))
                self._checked(
                    transport,
                    (
                        profile.python,
                        "-c",
                        "import os,sys;os.rename(sys.argv[1],sys.argv[2])",
                        str(temporary),
                        str(remote),
                    ),
                )
            finally:
                ManagedEnvironmentProvider._cleanup(transport, temporary)
        return {
            name: {"path": str(remote / f"{name}.py"), "sha256": value["sha256"]}
            for name, value in descriptors.items()
        }, code_id

    @staticmethod
    def _code_id(spec: ToolRuntimeSpecification) -> str:
        return (
            "tool-code-"
            + hashlib.sha256(
                json.dumps(spec.descriptor()["entrypoints"], sort_keys=True).encode()
            ).hexdigest()[:24]
        )

    @classmethod
    def bind_job(
        cls,
        profile: ClusterProfile,
        transport: Transport,
        job_id: str,
        bindings: Sequence[Mapping[str, Any]],
    ) -> str:
        assert profile.storage is not None
        if not job_id.startswith("job-") or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in job_id
        ):
            raise ValueError("Invalid Job identity for isolated tool bindings.")
        root = profile.storage.job_root
        if profile.transport == "local":
            root = str(Path(root).absolute())
        path = PurePosixPath(root) / job_id / "tool-runtimes.json"
        payload = json.dumps(
            {"tool_registry_version": 1, "runtimes": {v["name"]: dict(v) for v in bindings}}
        )
        cls._checked(
            transport,
            (
                profile.python,
                "-c",
                "import os,pathlib,sys,uuid;p=pathlib.Path(sys.argv[1]); "
                "\nif any(v.is_symlink() for v in (p,*p.parents)): "
                "raise ValueError('Unsafe tool Job binding path')\n"
                "p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.'+uuid.uuid4().hex+'.tmp');"
                "t.write_text(sys.argv[2]);os.replace(t,p)",
                str(path),
                payload,
            ),
        )
        return str(path)

    @classmethod
    def _register(
        cls, profile: ClusterProfile, transport: Transport, binding: Mapping[str, Any]
    ) -> None:
        assert profile.storage is not None
        code = """import fcntl,json,os,pathlib,sys,uuid
state=pathlib.Path(sys.argv[1]);cache=pathlib.Path(sys.argv[2]);binding=json.loads(sys.argv[3])
p=state/'tool-runtimes.json';state.mkdir(parents=True,exist_ok=True)
if any(v.is_symlink() for v in (p,*p.parents,cache,cache/'.gc.lock',state/'.tool-runtimes.lock')):
 raise ValueError('Unsafe tool runtime registry')
with (cache/'.gc.lock').open('a+') as gc, (state/'.tool-runtimes.lock').open('a+') as lock:
 fcntl.flock(gc,fcntl.LOCK_EX);fcntl.flock(lock,fcntl.LOCK_EX)
 existing=json.loads(p.read_text()) if p.exists() else {'tool_registry_version':1,'runtimes':{}}
 if existing.get('tool_registry_version')!=1: raise ValueError('Unknown tool runtime registry')
 existing['runtimes'][binding['name']]=binding
 temporary=p.with_suffix('.'+uuid.uuid4().hex+'.tmp')
 try:
  with temporary.open('x') as f:json.dump(existing,f,sort_keys=True);f.flush();os.fsync(f.fileno())
  os.replace(temporary,p)
 finally:temporary.unlink(missing_ok=True)
"""
        cls._checked(
            transport,
            (
                profile.python,
                "-c",
                code,
                profile.storage.state_root,
                profile.storage.cache_root,
                json.dumps(binding),
            ),
        )
