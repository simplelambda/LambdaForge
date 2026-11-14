"""Portable venv/pip managed environment provider."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.utils import canonicalize_name

from lambdaforge.controlplane.CacheBuildLease import CacheBuildLease
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.DependencyLayer import DependencyLayer
from lambdaforge.controlplane.EnvironmentProvider import EnvironmentProvider
from lambdaforge.controlplane.EnvironmentPublication import (
    PUBLICATION_CONTRACT,
    EnvironmentPublication,
)
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.NativeEnvironment import (
    NativeEnvironmentError,
    NativeEnvironmentPlan,
)
from lambdaforge.controlplane.NativeLibraryPolicy import NativeLibraryPolicy
from lambdaforge.controlplane.PreparedEnvironment import PreparedEnvironment
from lambdaforge.controlplane.TlsTrust import TlsTrust
from lambdaforge.controlplane.TorchInstallationPlan import TorchInstallationPlan
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


class ManagedEnvironmentProvider(EnvironmentProvider):
    """Create immutable pip-only venvs or unified native Conda prefixes."""

    def prepare(
        self,
        profile: ClusterProfile,
        transport: Transport,
        bundle: ExecutionBundle,
        *,
        remote_bundle_dir: str | Path,
    ) -> PreparedEnvironment:
        """Build under an advisory marker so cache GC fails closed during mutation."""
        assert profile.storage is not None
        cache_root = PurePosixPath(profile.storage.cache_root)
        if bundle.environment_id is None:
            raise ValueError("Managed environment bundles require an environment identity.")
        marker = cache_root / f".environment-build-{bundle.environment_id}.lock"
        completion = (
            PurePosixPath(profile.storage.environment_root)
            / bundle.environment_id
            / ".lambdaforge-environment.json"
        )
        created = transport.run(("mkdir", "-p", str(cache_root)))
        if created.returncode:
            raise RuntimeError(f"Could not create managed cache root: {created.stderr.strip()}")
        self._acquire(transport, marker, completion, python=profile.python)
        with CacheBuildLease.maintain(transport, marker, python=profile.python):
            layer = (bundle.environment_policy or {}).get("dependency_layer")
            if isinstance(layer, Mapping):
                return self._prepare_layer(profile, transport, bundle, remote_bundle_dir, layer)
            return self._prepare(
                profile,
                transport,
                bundle,
                remote_bundle_dir=remote_bundle_dir,
            )

    def _prepare_layer(
        self,
        profile: ClusterProfile,
        transport: Transport,
        bundle: ExecutionBundle,
        remote_bundle_dir: str | Path,
        layer: Mapping[str, Any],
    ) -> PreparedEnvironment:
        """Freeze one dependency base, then install exact code in independent small venvs."""
        assert profile.storage is not None and bundle.environment_id is not None
        identity = str(layer["identity"])
        if (
            not identity.startswith("deps-")
            or len(identity) != 29
            or not all(character in "0123456789abcdef" for character in identity[5:])
        ):
            raise ValueError("Invalid dependency-layer identity.")
        root = PurePosixPath(profile.storage.environment_root)
        base = root / identity
        lock = root.parent / f".environment-build-{identity}.lock"
        self._acquire(
            transport, lock, base / ".lambdaforge-environment.json", python=profile.python
        )
        with CacheBuildLease.maintain(transport, lock, python=profile.python):
            policy = dict(bundle.environment_policy or {})
            policy.pop("dependency_layer")
            base_bundle = replace(bundle, environment_id=identity, environment_policy=policy)
            resolution = self._dependency_resolution(profile, transport, base, layer, seal=False)
            if resolution is None and layer.get("legacy_identity"):
                # Migrate the operational publication contract, not the scientific solve.
                # Only read the old receipt; old prefixes may still be live/recovery-owned.
                legacy_id = DependencyLayer.previous_identity(layer)
                legacy_layer = {**layer, "identity": legacy_id}
                resolution = self._dependency_resolution(
                    profile, transport, root / legacy_id, legacy_layer, seal=False
                )
            frozen_packages = resolution["packages"] if resolution else None
            bootstrap_names = tuple(str(value["name"]) for value in layer["contracts"])
            # The base retains its initial bootstrap wheels for installer/provenance checks.
            # They are never upgraded; current execution wheels shadow them in the code venv.
            self._prepare(
                profile,
                transport,
                base_bundle,
                remote_bundle_dir=remote_bundle_dir,
                activate=False,
                framework_check=False,
                dependency_pins=tuple(
                    f"{value['name']}=={value['version']}" for value in (frozen_packages or ())
                ),
                frozen_packages=frozen_packages,
                bootstrap_names=bootstrap_names,
            )
            resolution = self._dependency_resolution(profile, transport, base, layer, seal=True)
            assert resolution is not None
            resolution_id = str(resolution["resolution_id"])
            target = root / bundle.environment_id
            python = target / "bin/python"
            marker = target / ".lambdaforge-environment.json"
            raw_plan = policy.get("pytorch")
            plan = (
                TorchInstallationPlan.from_mapping(raw_plan)
                if isinstance(raw_plan, Mapping)
                else None
            )
            raw_runtime = policy.get("python_runtime", {})
            raw_trust = raw_runtime.get("tls_trust") if isinstance(raw_runtime, Mapping) else None
            trust = TlsTrust.from_mapping(raw_trust if isinstance(raw_trust, Mapping) else None)
            library_policy = (
                NativeLibraryPolicy(str(base)) if policy.get("native_environment") else None
            )
            library_evidence: dict[str, Any] = {}
            if transport.run(("test", "-f", str(marker))).returncode == 0:
                receipt = transport.run(
                    (
                        profile.python,
                        "-c",
                        "import json,sys; from pathlib import Path; "
                        "m=json.loads(Path(sys.argv[1]).read_text()); "
                        "assert m['environment_id']==sys.argv[2] and "
                        "m['dependency_environment']==sys.argv[3] and "
                        "m['dependency_resolution_id']==sys.argv[4] and "
                        "m.get('publication_contract')==sys.argv[5] and "
                        "m.get('installation_prefix')==str(Path(sys.argv[1]).parent)",
                        str(marker),
                        bundle.environment_id,
                        identity,
                        resolution_id,
                        PUBLICATION_CONTRACT,
                    )
                )
                if receipt.returncode:
                    raise RuntimeError(
                        f"Code-layer dependency receipt is corrupt; preserved {target}."
                    )
                checked = self._verify(
                    profile,
                    transport,
                    str(python),
                    plan,
                    trust,
                    library_policy=library_policy,
                    library_evidence=library_evidence,
                )
                if checked.returncode:
                    raise RuntimeError(
                        f"Immutable code layer failed verification; preserved {target}: "
                        f"{checked.stderr}"
                    )
                self._activate(profile, transport, str(python))
                return PreparedEnvironment(
                    bundle.environment_id,
                    str(python),
                    True,
                    resolution_id,
                    library_policy,
                    library_evidence or None,
                )
            code = """import json,pathlib,subprocess,sys,sysconfig,venv
base=pathlib.Path(sys.argv[1]); target=pathlib.Path(sys.argv[2])
venv.EnvBuilder(with_pip=True,symlinks=True).create(target)
python=target/'bin/python'
probe="import sysconfig;print(sysconfig.get_path('purelib'))"
site=subprocess.check_output([str(python),'-c',probe],text=True).strip()
pathlib.Path(site,'lambdaforge-dependencies.pth').write_text(sysconfig.get_path('purelib')+'\\n')
"""
            with EnvironmentPublication.build(transport, profile.python, target):
                created = transport.run(
                    (
                        *(library_policy.launch_prefix(profile.python) if library_policy else ()),
                        str(base / "bin/python"),
                        "-c",
                        code,
                        str(base),
                        str(target),
                    )
                )
                if created.returncode:
                    raise RuntimeError(f"Could not create immutable code layer: {created.stderr}")
                install_python = str(target / "bin/python")
                prefix = (
                    *(() if trust is None else trust.prefix()),
                    *(library_policy.launch_prefix(profile.python) if library_policy else ()),
                )
                installed = transport.run(
                    (
                        *prefix,
                        install_python,
                        "-m",
                        "pip",
                        "install",
                        "--no-index",
                        "--disable-pip-version-check",
                        "--no-deps",
                        "--ignore-installed",
                        *(
                            str(PurePosixPath(str(remote_bundle_dir)) / "packages" / name)
                            for name in bundle.package_names
                        ),
                    )
                )
                if installed.returncode:
                    raise RuntimeError(
                        f"Could not install exact execution wheels: {installed.stderr}"
                    )
                checked = transport.run((*prefix, install_python, "-m", "pip", "check"))
                if checked.returncode:
                    raise RuntimeError(
                        f"Code/dependency contract mismatch: {checked.stdout} {checked.stderr}"
                    )
                checked = self._verify(
                    profile,
                    transport,
                    install_python,
                    plan,
                    trust,
                    library_policy=library_policy,
                    library_evidence=library_evidence,
                )
                if checked.returncode:
                    raise RuntimeError(f"Code-layer verification failed: {checked.stderr}")
                raw_native = policy.get("native_environment")
                native_tools: list[object] = []
                if isinstance(raw_native, Mapping):
                    native = NativeEnvironmentPlan.from_mapping(raw_native)
                    evidence = self._native_tools(transport, str(base / "bin/python"), native)
                    if evidence.returncode:
                        raise NativeEnvironmentError(
                            f"Native base tool verification failed: {evidence.stderr}"
                        )
                    native_tools = json.loads(evidence.stdout)
                payload = {
                    "environment_id": bundle.environment_id,
                    "layer_kind": "execution-code",
                    "dependency_environment": identity,
                    "dependency_resolution_id": resolution_id,
                    "packages": list(bundle.package_names),
                    "environment_policy": dict(bundle.environment_policy or {}),
                    "native_tools": native_tools,
                    "native_library_policy": library_policy.to_dict() if library_policy else None,
                    "native_libraries": library_evidence or None,
                }
                publish = """import pathlib,sys
p=pathlib.Path(sys.argv[1]); base=pathlib.Path(sys.argv[2])
# Link after installation: pip must never write through a link into the base.
for executable in (base/'bin').iterdir():
 destination=p/'bin'/executable.name
 if not executable.is_file() or destination.exists(): continue
 with executable.open('rb') as handle: header=handle.read(512)
 is_python=b'python' in header.splitlines()[0] or b"'''exec" in header
 if header.startswith(b'#!') and is_python:
  import shlex
  destination.write_text('#!/bin/sh\\nexec '+shlex.quote(str(p/'bin/python'))+' '+
                         shlex.quote(str(executable))+' "$@"\\n')
  destination.chmod(0o755)
 else: destination.symlink_to(executable)
"""
                published = transport.run((install_python, "-c", publish, str(target), str(base)))
                if published.returncode:
                    raise RuntimeError(
                        f"Could not publish immutable code layer: {published.stderr}"
                    )
                checked = self._verify(
                    profile,
                    transport,
                    str(python),
                    plan,
                    trust,
                    library_policy=library_policy,
                    library_evidence=library_evidence,
                )
                if checked.returncode:
                    raise RuntimeError(f"Final code-layer verification failed: {checked.stderr}")
                EnvironmentPublication.seal(transport, profile.python, target, payload)
            self._activate(profile, transport, str(python))
            return PreparedEnvironment(
                bundle.environment_id,
                str(python),
                False,
                resolution_id,
                library_policy,
                library_evidence or None,
            )

    @staticmethod
    def _dependency_resolution(
        profile: ClusterProfile,
        transport: Transport,
        base: PurePosixPath,
        layer: Mapping[str, Any],
        *,
        seal: bool,
    ) -> dict[str, Any] | None:
        """Retain a permanent exact dependency resolution outside collectable environment bytes."""
        assert profile.storage is not None
        receipt = (
            PurePosixPath(profile.storage.state_root)
            / "dependency-resolutions"
            / f"{layer['identity']}.json"
        )
        code = """import hashlib,json,os,pathlib,re,sys,uuid
p=pathlib.Path(sys.argv[1]); base=pathlib.Path(sys.argv[2]); layer=json.loads(sys.argv[3])
seal=sys.argv[4]=='true'
if any(v.is_symlink() for v in (p,*p.parents)):
 raise SystemExit('Unsafe dependency resolution receipt')
def checked(value):
 if (not isinstance(value,dict) or value.get('version')!=1
     or value.get('identity')!=layer['identity']):
  raise ValueError('Invalid frozen dependency resolution identity')
 payload={k:v for k,v in value.items() if k!='resolution_id'}
 digest='sha256:'+hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
 if value.get('resolution_id')!=digest: raise ValueError('Corrupt frozen dependency resolution')
 if value.get('policy')!=layer['resolution_policy']:
  raise ValueError('Frozen dependency resolution policy changed')
 for item in value['packages']:
  if not re.fullmatch('[a-z0-9][a-z0-9.-]*',item['name']):
   raise ValueError('Invalid dependency name')
  if not re.fullmatch('[A-Za-z0-9_.+!-]+',item['version']):
   raise ValueError('Invalid dependency version')
 return value
existing=None
if p.exists():
 if p.stat().st_size>1048576: raise ValueError('Oversized dependency resolution receipt')
 existing=checked(json.loads(p.read_text()))
if not seal: print(json.dumps(existing)); raise SystemExit(0)
marker=json.loads((base/'.lambdaforge-environment.json').read_text())
names={v['name'] for v in layer['contracts']}
def canonical(value): return re.sub('[-_.]+','-',value).lower()
packages=sorted([{'name':canonical(v['name']),'version':v['version']}
                 for v in marker['installed_packages'] if canonical(v['name']) not in names],
                key=lambda v:v['name'])
payload={'version':1,'identity':layer['identity'],'policy':layer['resolution_policy'],'packages':packages}
payload['resolution_id']='sha256:'+hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
checked(payload)
if existing is not None:
 if existing!=payload:
  raise ValueError('Frozen dependency resolution changed; refusing silent reuse')
else:
 p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix('.'+uuid.uuid4().hex+'.tmp')
 try:
  with tmp.open('x') as f: json.dump(payload,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,p)
 finally: tmp.unlink(missing_ok=True)
print(json.dumps(payload))
"""
        result = transport.run(
            (
                profile.python,
                "-c",
                code,
                str(receipt),
                str(base),
                json.dumps(layer),
                str(seal).lower(),
            )
        )
        if result.returncode:
            raise RuntimeError(f"Cannot verify frozen dependency resolution: {result.stderr}")
        parsed = json.loads(result.stdout)
        if parsed is not None and not isinstance(parsed, dict):
            raise RuntimeError("Invalid dependency resolution response.")
        return parsed

    def _prepare(
        self,
        profile: ClusterProfile,
        transport: Transport,
        bundle: ExecutionBundle,
        *,
        remote_bundle_dir: str | Path,
        activate: bool = True,
        framework_check: bool = True,
        dependency_pins: tuple[str, ...] = (),
        frozen_packages: Sequence[Mapping[str, str]] | None = None,
        bootstrap_names: tuple[str, ...] = (),
    ) -> PreparedEnvironment:
        """Own an unsealed final prefix through installation and atomic publication."""
        assert profile.storage is not None and bundle.environment_id is not None
        target = PurePosixPath(profile.storage.environment_root) / bundle.environment_id

        def install() -> PreparedEnvironment:
            return self._prepare_prefix(
                profile,
                transport,
                bundle,
                remote_bundle_dir=remote_bundle_dir,
                activate=activate,
                framework_check=framework_check,
                dependency_pins=dependency_pins,
                frozen_packages=frozen_packages,
                bootstrap_names=bootstrap_names,
            )

        if (
            transport.run(("test", "-f", str(target / ".lambdaforge-environment.json"))).returncode
            == 0
        ):
            return install()
        with EnvironmentPublication.build(transport, profile.python, target):
            return install()

    def _prepare_prefix(
        self,
        profile: ClusterProfile,
        transport: Transport,
        bundle: ExecutionBundle,
        *,
        remote_bundle_dir: str | Path,
        activate: bool = True,
        framework_check: bool = True,
        dependency_pins: tuple[str, ...] = (),
        frozen_packages: Sequence[Mapping[str, str]] | None = None,
        bootstrap_names: tuple[str, ...] = (),
    ) -> PreparedEnvironment:
        """Install and verify at the lifetime prefix; never relocate an environment."""
        if bundle.environment_id is None:
            raise ValueError("Managed environment bundles require an environment identity.")
        remote = PurePosixPath(str(remote_bundle_dir))
        assert profile.storage is not None
        environment_root = PurePosixPath(profile.storage.environment_root)
        environment = environment_root / bundle.environment_id
        python = environment / "bin" / "python"
        marker = environment / ".lambdaforge-environment.json"
        policy = dict(bundle.environment_policy or {})
        raw_plan = policy.get("pytorch")
        plan = (
            TorchInstallationPlan.from_mapping(raw_plan) if isinstance(raw_plan, Mapping) else None
        )
        raw_runtime = policy.get("python_runtime")
        raw_trust = raw_runtime.get("tls_trust") if isinstance(raw_runtime, Mapping) else None
        trust = TlsTrust.from_mapping(raw_trust if isinstance(raw_trust, Mapping) else None)
        raw_native = policy.get("native_environment")
        native = (
            NativeEnvironmentPlan.from_mapping(raw_native)
            if isinstance(raw_native, Mapping)
            else None
        )
        library_policy = NativeLibraryPolicy(str(environment)) if native is not None else None
        library_evidence: dict[str, Any] = {}
        process_prefix = (
            *(() if trust is None else trust.prefix()),
            *(library_policy.launch_prefix(profile.python) if library_policy else ()),
        )
        cached = transport.run(("test", "-f", str(marker)))
        if cached.returncode == 0:
            verified = self._verify_reusable(
                profile,
                transport,
                environment,
                str(python),
                plan,
                trust,
                native,
                framework_check=framework_check,
                library_evidence=library_evidence,
            )
            if verified.returncode == 0:
                if activate:
                    self._activate(profile, transport, str(python))
                return PreparedEnvironment(
                    bundle.environment_id,
                    str(python),
                    True,
                    library_policy=library_policy,
                    library_evidence=library_evidence or None,
                )
            # This exact immutable prefix may still serve another Job or recovery.
            # Never rename/remove it as an installation side effect. GC, with its
            # authoritative live/recovery references, is the only eviction authority.
            error_type = NativeEnvironmentError if native is not None else RuntimeError
            raise error_type(
                f"Cached managed environment {bundle.environment_id} failed verification: "
                f"{verified.stderr.strip() or verified.stdout.strip()}. "
                "The prefix was preserved because other Jobs may reference it. "
                "Inspect the environment and preview `lf clean` before removing an unused cache."
            )
        created_root = transport.run(("mkdir", "-p", str(environment_root)))
        if created_root.returncode:
            raise RuntimeError(f"Could not create environment cache: {created_root.stderr.strip()}")
        created = (
            self._create_native(
                profile,
                transport,
                native,
                environment,
                remote,
                trust,
            )
            if native is not None
            else transport.run(
                (
                    *profile.command_prefix,
                    *process_prefix,
                    profile.python,
                    "-m",
                    "venv",
                    str(environment),
                )
            )
        )
        if created.returncode:
            kind = "native Conda prefix" if native is not None else "managed venv"
            raise RuntimeError(f"Could not create the {kind}: {created.stderr.strip()}")
        if native is not None:
            inventory = self._verify_native_inventory(
                profile, transport, environment, native, trust
            )
            if inventory.returncode:
                raise NativeEnvironmentError(
                    "Native package inventory differs from the resolved environment identity: "
                    f"{inventory.stderr.strip()}"
                )
        wheel_dir = remote / "packages"
        wheel_names = tuple(bundle.package_names)
        if not wheel_names:
            raise ValueError("Managed environment bundle contains no exact wheels.")
        constraint: PurePosixPath | None = None
        pin_args: tuple[str, ...] = ()
        if dependency_pins:
            pins_path = environment / ".lambdaforge-dependency-pins.txt"
            written = transport.run(
                (
                    *process_prefix,
                    str(python),
                    "-c",
                    "from pathlib import Path; import sys; "
                    "Path(sys.argv[1]).write_text(sys.argv[2])",
                    str(pins_path),
                    "\n".join(dependency_pins) + "\n",
                )
            )
            if written.returncode:
                raise RuntimeError("Could not stage frozen dependency constraints.")
            pin_args = ("--constraint", str(pins_path))
        if not bundle.offline and plan is not None:
            if plan.version is None or plan.index_url is None:
                raise ValueError("Online managed environments require an exact PyTorch plan.")
            torch_install = transport.run(
                (
                    *process_prefix,
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    *pin_args,
                    "--cache-dir",
                    str(PurePosixPath(profile.storage.cache_root) / "pip"),
                    "--index-url",
                    plan.index_url,
                    f"torch=={plan.version}",
                )
            )
            if torch_install.returncode:
                raise RuntimeError(
                    f"Could not install resolved PyTorch {plan.version} from {plan.channel}: "
                    f"{torch_install.stderr.strip()}"
                )
            constraint = remote / "torch-constraint.txt"
            written = transport.run(
                (
                    *process_prefix,
                    str(python),
                    "-c",
                    "from pathlib import Path; import sys; "
                    "Path(sys.argv[1]).write_text(sys.argv[2])",
                    str(constraint),
                    f"torch=={plan.version}\n",
                )
            )
            if written.returncode:
                raise RuntimeError(
                    f"Could not write the managed PyTorch constraint: {written.stderr.strip()}"
                )
        install = [
            *process_prefix,
            str(python),
            "-m",
            "pip",
            "install",
            *pin_args,
            "--cache-dir",
            str(PurePosixPath(profile.storage.cache_root) / "pip"),
        ]
        if bundle.offline:
            install.extend(("--no-index", "--find-links", str(remote / "wheelhouse")))
        if constraint is not None:
            install.extend(("--constraint", str(constraint)))
        install.extend(str(wheel_dir / name) for name in wheel_names)
        installed = transport.run(tuple(install))
        if installed.returncode:
            hint = (
                "The offline wheelhouse is incomplete for the remote platform."
                if bundle.offline
                else "Dependency installation failed; inspect pip output and cluster connectivity."
            )
            raise RuntimeError(f"{hint} {installed.stderr.strip()}")
        checked = transport.run(
            (
                *process_prefix,
                str(python),
                "-m",
                "pip",
                "check",
            )
        )
        if checked.returncode:
            raise RuntimeError(
                "Managed Python dependencies are inconsistent after installation: "
                f"{checked.stdout.strip() or checked.stderr.strip()}"
            )
        if native is not None:
            inventory = self._verify_native_inventory(
                profile, transport, environment, native, trust
            )
            if inventory.returncode:
                raise NativeEnvironmentError(
                    "Pip installation changed the identity-bearing native inventory: "
                    f"{inventory.stderr.strip()}"
                )
        verified = self._verify(
            profile,
            transport,
            str(python),
            plan,
            trust,
            native,
            framework_check=framework_check,
            library_evidence=library_evidence,
        )
        if verified.returncode:
            raise RuntimeError(
                f"Managed environment verification failed: {verified.stderr.strip()}"
            )
        native_tools: list[object] = []
        if native is not None:
            native_evidence = self._native_tools(transport, str(python), native)
            if native_evidence.returncode:
                raise NativeEnvironmentError(
                    f"Required native executable verification failed: {native_evidence.stderr}"
                )
            try:
                parsed_tools = json.loads(native_evidence.stdout or "[]")
            except json.JSONDecodeError as error:
                raise RuntimeError("Native executable evidence is not valid JSON.") from error
            if not isinstance(parsed_tools, list):
                raise RuntimeError("Native executable evidence has an invalid shape.")
            native_tools = parsed_tools
        installed_inventory = transport.run(
            (*process_prefix, str(python), "-m", "pip", "list", "--format=json")
        )
        if installed_inventory.returncode:
            raise RuntimeError(
                f"Could not record resolved dependency inventory: {installed_inventory.stderr}"
            )
        try:
            inventory_packages = json.loads(installed_inventory.stdout or "[]")
        except ValueError as error:
            raise RuntimeError("Invalid resolved dependency inventory.") from error
        if frozen_packages is not None:
            actual = {
                canonicalize_name(value["name"]): value["version"]
                for value in inventory_packages
                if canonicalize_name(value["name"]) not in bootstrap_names
            }
            expected = {value["name"]: value["version"] for value in frozen_packages}
            if actual != expected:
                raise RuntimeError(
                    "Rebuilt dependencies differ from the frozen resolution; "
                    "no prefix was published."
                )
        payload = {
            "environment_id": bundle.environment_id,
            "packages": list(wheel_names),
            "environment_policy": policy,
            "native_tools": native_tools,
            "native_library_policy": library_policy.to_dict() if library_policy else None,
            "native_libraries": library_evidence or None,
            "installed_packages": inventory_packages,
            "layer_kind": "dependencies"
            if bundle.environment_id.startswith("deps-")
            else "monolithic",
        }
        EnvironmentPublication.seal(transport, profile.python, environment, payload)
        if activate:
            self._activate(profile, transport, str(python))
        return PreparedEnvironment(
            bundle.environment_id,
            str(python),
            False,
            library_policy=library_policy,
            library_evidence=library_evidence or None,
        )

    @staticmethod
    def _verify(
        profile: ClusterProfile,
        transport: Transport,
        python: str,
        plan: TorchInstallationPlan | None,
        trust: TlsTrust | None = None,
        native: NativeEnvironmentPlan | None = None,
        *,
        framework_check: bool = True,
        library_policy: NativeLibraryPolicy | None = None,
        library_evidence: dict[str, Any] | None = None,
    ) -> CommandResult:
        """Verify runtime/grants and both import orders against actual native library origins."""
        require_cuda = bool(plan and plan.require_cuda)
        expected_torch = plan.version if plan is not None else None
        code = (
            "import lambdaforge,ssl,sys,torch\n"
            + (
                f"assert lambdaforge.__version__ == {LambdaForgeVersion.CURRENT!r}\n"
                if framework_check
                else ""
            )
            + f"assert {expected_torch!r} is None or torch.__version__ == {expected_torch!r}\n"
            "assert ssl.create_default_context().get_ca_certs(), 'no trusted TLS CAs'\n"
            f"required={require_cuda!r}\n"
            "available=torch.cuda.is_available()\n"
            "if required and not available: sys.exit(3)\n"
            "if available:\n"
            " for device in range(torch.cuda.device_count()):\n"
            "  with torch.cuda.device(device):\n"
            "   probe=torch.ones(1, device='cuda').mul_(2)\n"
            "   assert probe.item() == 2\n"
            "print(lambdaforge.__version__, torch.__version__, torch.version.cuda, available)\n"
        )
        library_policy = library_policy or (
            NativeLibraryPolicy(str(PurePosixPath(python).parent.parent)) if native else None
        )
        if library_policy:
            code += library_policy.probe_code()
        command = (
            *(() if trust is None else trust.prefix()),
            *(library_policy.launch_prefix(profile.python) if library_policy else ()),
            python,
            "-c",
            code,
        )
        verified = transport.run(profile.gpu_access.wrap(command) if require_cuda else command)
        if verified.returncode:
            if library_policy:
                return CommandResult(
                    verified.returncode,
                    verified.stdout,
                    "Native interoperability verification (Torch then SciPy); "
                    f"selected={library_policy.dependency_prefix}\n{verified.stderr}",
                )
            return verified
        if library_policy:
            try:
                first = library_policy.read_report(verified.stdout)
            except ValueError as error:
                return CommandResult(1, stderr=str(error))
            reverse = (
                *(() if trust is None else trust.prefix()),
                *library_policy.launch_prefix(profile.python),
                python,
                "-c",
                library_policy.probe_code(scipy_first=True),
            )
            second = transport.run(profile.gpu_access.wrap(reverse) if require_cuda else reverse)
            if second.returncode:
                return CommandResult(
                    second.returncode,
                    second.stdout,
                    "Native interoperability verification (SciPy then Torch); "
                    f"selected={library_policy.dependency_prefix}\n{second.stderr}",
                )
            try:
                second_report = library_policy.read_report(second.stdout)
            except ValueError as error:
                return CommandResult(1, stderr=str(error))
            if library_evidence is not None:
                library_evidence.update(torch_first=first, scipy_first=second_report)
        if native is None:
            return verified
        return ManagedEnvironmentProvider._native_tools(transport, python, native)

    @staticmethod
    def _verify_reusable(
        profile: ClusterProfile,
        transport: Transport,
        prefix: PurePosixPath,
        python: str,
        plan: TorchInstallationPlan | None,
        trust: TlsTrust | None,
        native: NativeEnvironmentPlan | None,
        *,
        framework_check: bool = True,
        library_evidence: dict[str, Any] | None = None,
    ) -> CommandResult:
        """Verify runtime behavior and exact native inventory before cache reuse."""
        verified = ManagedEnvironmentProvider._verify(
            profile,
            transport,
            python,
            plan,
            trust,
            native,
            framework_check=framework_check,
            library_evidence=library_evidence,
        )
        if verified.returncode:
            return verified
        inventory_code = """import importlib.metadata,json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'.lambdaforge-environment.json'
record=json.loads(p.read_text())
if (record.get('publication_contract')!=sys.argv[2]
    or record.get('installation_prefix')!=str(p.parent)):
 raise SystemExit('Unverified/relocated managed prefix publication; preserved. '
                  'Bootstrap with current LambdaForge to build the new publication identity.')
expected=record.get('installed_packages')
if expected is not None:
 actual={d.metadata['Name'].lower().replace('_','-'):d.version
         for d in importlib.metadata.distributions()}
 pinned={d['name'].lower().replace('_','-'):d['version'] for d in expected}
 if actual!=pinned:
  raise SystemExit('Resolved dependency inventory changed; immutable prefix preserved')
"""
        loader = NativeLibraryPolicy(str(prefix)).launch_prefix(profile.python) if native else ()
        inventory = transport.run(
            (*loader, python, "-c", inventory_code, str(prefix), PUBLICATION_CONTRACT)
        )
        if inventory.returncode or native is None:
            return inventory
        return ManagedEnvironmentProvider._verify_native_inventory(
            profile, transport, prefix, native, trust
        )

    @staticmethod
    def _create_native(
        profile: ClusterProfile,
        transport: Transport,
        native: NativeEnvironmentPlan,
        temporary: PurePosixPath,
        remote_bundle: PurePosixPath,
        trust: TlsTrust | None,
    ) -> CommandResult:
        """Create the one final prefix from an exact native plan, never through activation."""
        assert profile.storage is not None
        cache_root = PurePosixPath(profile.storage.cache_root)
        manager = (
            cache_root
            / "runtime-managers"
            / f"micromamba-{native.manager_version}-{native.platform}"
        )
        package_root = (
            cache_root / "native-packages" / native.package_cache_id / "packages"
            if native.package_cache_id is not None
            else cache_root / "conda-pkgs"
        )
        command = [
            *profile.command_prefix,
            "env",
            f"MAMBA_ROOT_PREFIX={cache_root / 'micromamba-root'}",
            f"CONDA_PKGS_DIRS={package_root}",
            "CONDARC=/dev/null",
            "MAMBARC=/dev/null",
            *(() if trust is None else trust.assignments()),
            str(manager),
            "create",
            "--yes",
            "--prefix",
            str(temporary),
            "--platform",
            native.platform,
        ]
        if native.offline:
            command.extend(
                (
                    "--offline",
                    "--file",
                    str(remote_bundle / "native" / native.specification_name),
                )
            )
        else:
            command.append("--override-channels")
            for channel in native.channels:
                command.extend(("--channel", channel))
            command.extend(native.exact_packages)
        return transport.run(tuple(command), timeout=900.0)

    @classmethod
    def _verify_native_inventory(
        cls,
        profile: ClusterProfile,
        transport: Transport,
        prefix: PurePosixPath,
        native: NativeEnvironmentPlan,
        trust: TlsTrust | None,
    ) -> CommandResult:
        """Compare regular conda-meta records with the identity-bearing solve."""
        assert profile.storage is not None
        manager = (
            PurePosixPath(profile.storage.cache_root)
            / "runtime-managers"
            / f"micromamba-{native.manager_version}-{native.platform}"
        )
        listed = transport.run(
            (
                *profile.command_prefix,
                *(() if trust is None else trust.prefix()),
                str(manager),
                "list",
                "--json",
                "--prefix",
                str(prefix),
            )
        )
        if listed.returncode:
            return listed
        metadata_code = (
            "# lambdaforge-conda-meta-inventory\n"
            "import json,sys\n"
            "from pathlib import Path\n"
            "records=[]\n"
            "try:\n"
            " for path in sorted(Path(sys.argv[1]).glob('*.json')):\n"
            "  item=json.loads(path.read_text(encoding='utf-8'))\n"
            "  if not isinstance(item,dict): raise TypeError(f'{path.name} is not an object')\n"
            "  records.append(item)\n"
            "except Exception as error:\n"
            " print(f'{type(error).__name__}: {error}',file=sys.stderr)\n"
            " raise SystemExit(2)\n"
            "print(json.dumps(records,sort_keys=True))\n"
        )
        metadata = transport.run(
            (
                *profile.command_prefix,
                *NativeLibraryPolicy(str(prefix)).launch_prefix(profile.python),
                str(prefix / "bin" / "python"),
                "-c",
                metadata_code,
                str(prefix / "conda-meta"),
            )
        )
        if metadata.returncode:
            detail = (metadata.stderr or metadata.stdout).strip().splitlines()
            return CommandResult(
                1,
                stderr=(
                    "Installed Conda metadata could not be read"
                    + (f": {detail[-1][:300]}" if detail else ".")
                ),
            )
        try:
            raw_list = json.loads(listed.stdout)
            raw_metadata = json.loads(metadata.stdout)
            if not isinstance(raw_list, list) or not isinstance(raw_metadata, list):
                raise TypeError("inventory responses must be arrays")
            listed_by_package: dict[tuple[str, str, str], Mapping[str, object]] = {}
            for item in raw_list:
                if not isinstance(item, Mapping):
                    raise TypeError("micromamba list contains a non-object record")
                key = cls._native_package_key(item)
                if key in listed_by_package:
                    raise ValueError(f"micromamba list contains duplicate package {key[0]!r}")
                listed_by_package[key] = item
            actual: set[tuple[str, str, str, str, str]] = set()
            conda_keys: set[tuple[str, str, str]] = set()
            for item in raw_metadata:
                if not isinstance(item, Mapping):
                    raise TypeError("conda-meta contains a non-object record")
                key = cls._native_package_key(item)
                if key in conda_keys:
                    raise ValueError(f"conda-meta contains duplicate package {key[0]!r}")
                conda_keys.add(key)
                subdir = str(item.get("subdir", "")).strip()
                if not subdir:
                    raise ValueError(f"conda-meta lacks subdir for package {key[0]!r}")
                listed_item = listed_by_package.get(key, {})
                channel = str(
                    listed_item.get(
                        "channel",
                        listed_item.get("base_url", item.get("channel", item.get("base_url", ""))),
                    )
                ).rstrip("/")
                if not channel:
                    raise ValueError(f"installed metadata lacks channel for package {key[0]!r}")
                actual.add((*key, channel, subdir))
            unmanaged = [
                key
                for key, item in listed_by_package.items()
                if key not in conda_keys
                and str(item.get("channel", "")).strip().lower() not in {"pypi", "pypi_0"}
            ]
            if unmanaged:
                names = ", ".join(repr(item[0]) for item in sorted(unmanaged)[:5])
                suffix = f" (+{len(unmanaged) - 5} more)" if len(unmanaged) > 5 else ""
                raise ValueError(f"conda-meta is missing for installed package(s) {names}{suffix}")
            expected = {
                (
                    item["name"],
                    item["version"],
                    item["build"],
                    item["channel"].rstrip("/"),
                    item["subdir"],
                )
                for item in native.package_inventory
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            return CommandResult(1, stderr=f"Invalid native package inventory: {error}")
        if actual != expected:
            return CommandResult(
                1,
                stderr=cls._summarize_native_inventory_difference(expected, actual),
            )
        return CommandResult(0, listed.stdout)

    @staticmethod
    def _native_package_key(item: Mapping[str, object]) -> tuple[str, str, str]:
        """Read the stable package key shared by solver, list and conda-meta JSON."""
        name = str(item.get("name", "")).strip()
        version = str(item.get("version", "")).strip()
        build = str(item.get("build_string", item.get("build", ""))).strip()
        if not name or not version or not build:
            raise ValueError("installed package metadata lacks name, version or build")
        return name, version, build

    @staticmethod
    def _summarize_native_inventory_difference(
        expected: set[tuple[str, str, str, str, str]],
        actual: set[tuple[str, str, str, str, str]],
        *,
        limit: int = 6,
    ) -> str:
        """Describe a bounded package-level diff instead of dumping whole environments."""
        fields = ("version", "build", "channel", "subdir")
        expected_by_name = {item[0]: item for item in expected}
        actual_by_name = {item[0]: item for item in actual}
        differences: list[str] = []
        for name in sorted(expected_by_name.keys() | actual_by_name.keys()):
            wanted = expected_by_name.get(name)
            observed = actual_by_name.get(name)
            if wanted is None:
                differences.append(f"unexpected {name!r}")
                continue
            if observed is None:
                differences.append(f"missing {name!r}")
                continue
            changed = [
                f"{field} {wanted[index]!r} -> {observed[index]!r}"
                for index, field in enumerate(fields, start=1)
                if wanted[index] != observed[index]
            ]
            if changed:
                differences.append(f"changed {name!r}: {', '.join(changed)}")
        shown = differences[:limit]
        if len(differences) > limit:
            shown.append(f"+{len(differences) - limit} more")
        return f"{len(differences)} package difference(s): {'; '.join(shown)}"

    @staticmethod
    def _native_tools(
        transport: Transport,
        python: str,
        native: NativeEnvironmentPlan,
    ) -> CommandResult:
        """Verify tools with bounded, tool-specific version policies on every preparation."""
        if not native.required_executables:
            return CommandResult(0, "[]")
        code = (
            "import json,os,re,shlex,subprocess,sys\n"
            "from pathlib import Path\n"
            "prefix=Path(sys.executable).parent.parent\n"
            "records=[]\n"
            "metadata=[]\n"
            "for item in sorted((prefix/'conda-meta').glob('*.json')):\n"
            " try: metadata.append(json.loads(item.read_text()))\n"
            " except Exception: pass\n"
            "for name in json.loads(sys.argv[1]):\n"
            " path=(prefix/'bin'/name).resolve()\n"
            " if not path.is_file() or not os.access(path,os.X_OK):\n"
            "  kind='tool-absent' if not path.is_file() else 'launcher-not-executable'\n"
            "  raise SystemExit(f'missing executable: {name}; diagnostic={kind}')\n"
            " relative=path.relative_to(prefix).as_posix()\n"
            " owner=next((item for item in metadata if relative in item.get('files',[])),None)\n"
            " if owner is None: raise SystemExit(f'no Conda package owns executable: {name}')\n"
            " missing_interpreter=None\n"
            " with path.open('rb') as handle: header=handle.read(8192)\n"
            " if header.startswith(b'#!'):\n"
            "  text=header.decode('utf-8',errors='replace')\n"
            "  first=shlex.split(text.splitlines()[0][2:])\n"
            "  if first and first[0].startswith('/') and not Path(first[0]).is_file():\n"
            "   missing_interpreter=first[0]\n"
            "  pattern=(r'''(?:'([^'\\n]+/bin/python[0-9.]*)'|'''\n"
            "           r'''\"([^\"\\n]+/bin/python[0-9.]*)\"|'''\n"
            "           r'''(/[\\w./+-]+/bin/python[0-9.]*))''')\n"
            "  for match in re.finditer(pattern,'\\n'.join(text.splitlines()[:4])):\n"
            "   candidate=next(value for value in match.groups() if value)\n"
            "   if not Path(candidate).is_file(): missing_interpreter=candidate\n"
            " version=None\n"
            " probes=[]\n"
            " for args in (['--version'],['version']):\n"
            "  probe={'argv':[str(path),*args],'exit_code':None}\n"
            "  try:\n"
            "   result=subprocess.run([str(path),*args],capture_output=True,text=True,\n"
            "                         errors='replace',timeout=10)\n"
            "  except (OSError,subprocess.TimeoutExpired) as error:\n"
            "   def captured(value):\n"
            "    if isinstance(value,bytes): value=value.decode('utf-8',errors='replace')\n"
            "    return (value or '')[:2000]\n"
            "   probe.update(error=type(error).__name__+': '+str(error)[:2000],\n"
            "                diagnostic=('missing-interpreter' if missing_interpreter else\n"
            "                 'probe-timeout' if isinstance(error,subprocess.TimeoutExpired)\n"
            "                 else 'broken-launcher'),interpreter=missing_interpreter,\n"
            "                stdout=captured(getattr(error,'stdout',None)),\n"
            "                stderr=captured(getattr(error,'stderr',None)))\n"
            "   probes.append(probe); continue\n"
            "  output='\\n'.join(\n"
            "   part.strip() for part in (result.stdout,result.stderr) if part).strip()\n"
            "  probe.update(exit_code=result.returncode,stdout=result.stdout[:2000],\n"
            "               stderr=result.stderr[:2000],policy='zero-exit')\n"
            "  accepted=result.returncode==0 and bool(output)\n"
            # APBS 3.4.1 main.c --version prints PACKAGE_STRING and takes VERROR1 (APBSRC=13).
            # Do not generalize this exception to executions, other flags, packages or versions.
            "  if name=='apbs':\n"
            "   banner=(r'(?m)^[ \\t]*APBS[ \\t]+'+re.escape(str(owner.get('version','')))\n"
            "           +r'[ \\t]*\\r?$')\n"
            "   identity=(owner.get('name')=='apbs' and bool(owner.get('version'))\n"
            "             and bool(re.search(banner,output)))\n"
            "   known_status=(args==['--version'] and owner.get('version')=='3.4.1'\n"
            "                 and result.returncode==13)\n"
            "   accepted=identity and (accepted or known_status)\n"
            "   probe['policy']='apbs-banner-zero-or-3.4.1-version-exit-13'\n"
            "   probe['expected_banner']='APBS '+str(owner.get('version',''))\n"
            "  accepted=accepted and missing_interpreter is None\n"
            "  if not accepted:\n"
            "   probe.update(interpreter=missing_interpreter,diagnostic=(\n"
            "    'missing-interpreter' if missing_interpreter else 'broken-launcher'\n"
            "    if result.returncode in (126,127) else 'version-probe-policy'))\n"
            "  probe['accepted']=accepted\n"
            "  probes.append(probe)\n"
            "  if accepted:\n"
            "   version=output[:2000]; break\n"
            " if version is None:\n"
            "  raise SystemExit('could not query executable version: '+name+\n"
            "                   '; probes='+json.dumps(probes,sort_keys=True))\n"
            " records.append({'name':name,'path':str(path),'version':version,"
            "'package':owner.get('name'),'package_version':owner.get('version'),"
            "'build':owner.get('build'),'channel':owner.get('channel'),"
            "'subdir':owner.get('subdir'),'probe':probe})\n"
            "print(json.dumps(records,sort_keys=True))\n"
        )
        selected = NativeLibraryPolicy(str(PurePosixPath(python).parent.parent))
        return transport.run(
            (
                *selected.launch_prefix(python),
                python,
                "-c",
                code,
                json.dumps(native.required_executables),
            )
        )

    @staticmethod
    def _activate(profile: ClusterProfile, transport: Transport, python: str) -> None:
        """Publish the interpreter selected by the latest successful preparation."""
        assert profile.storage is not None
        pointer = PurePosixPath(profile.storage.state_root) / "active-environment"
        result = transport.run(
            (
                python,
                "-c",
                "from pathlib import Path; import sys,os; "
                "Path(sys.argv[1]).write_text(sys.argv[2]); "
                "p=Path(sys.argv[2].strip()).parent.parent/'.lambdaforge-environment.json'; "
                "(os.utime(p,None),os.utime(p.parent,None)) if p.is_file() else None",
                str(pointer),
                f"{python}\n",
            )
        )
        if result.returncode:
            raise RuntimeError(f"Could not publish the active managed environment: {result.stderr}")

    @staticmethod
    def _cleanup(transport: Transport, path: PurePosixPath) -> None:
        """Remove one exact provider-created temporary path, never a broad root."""
        if (
            not path.name.startswith(".env-")
            and ".tmp-" not in path.name
            and ".stale-" not in path.name
        ):
            raise ValueError(f"Refusing to clean an unexpected environment path: {path}")
        transport.run(("rm", "-rf", str(path)))

    @staticmethod
    def _acquire(
        transport: Transport,
        lock: PurePosixPath,
        completion: PurePosixPath,
        *,
        timeout: float = 3600.0,
        python: str = "python3",
    ) -> bool:
        """Wait through native/pip provisioning, not just a short connection timeout.

        Individual installation phases can each take 900 seconds. Contention is ordinary
        preparation, not a failed submission; a missing completion still fails closed after
        the bounded provisioning window, without stealing or deleting another builder's lock.
        """
        # Reuse must also own the marker: seeing a completion receipt alone cannot
        # prevent GC from evicting the prefix during verification/activation.
        del completion
        try:
            CacheBuildLease.acquire(transport, lock, python=python, timeout=timeout)
        except TimeoutError as error:
            raise RuntimeError(f"Timed out waiting for managed environment lock {lock}.") from error
        return True
