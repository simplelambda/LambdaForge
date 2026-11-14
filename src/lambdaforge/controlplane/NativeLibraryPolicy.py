"""Process-scoped loader policy for one selected immutable Conda dependency prefix."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

LIBRARY_CONTRACT = "native-prefix-libraries-v1"
POLICY_VARIABLE = "LAMBDAFORGE_NATIVE_LIBRARY_POLICY"
REPORT_MARKER = "LAMBDAFORGE_NATIVE_LIBRARY_REPORT="

# Run only stdlib in the launcher, then exec: changing os.environ after Torch has
# loaded a SONAME cannot repair the process. No shell expansion or global exports.
_LAUNCH = """import json,os,pathlib,sys
if sys.argv[1]!='native-prefix-libraries-v1': raise SystemExit('Unknown native loader contract')
policy=json.loads(sys.argv[2]); root=pathlib.Path(policy['dependency_prefix'])
if policy.get('contract')!='native-prefix-libraries-v1':
 raise SystemExit('Incompatible selected native loader policy')
library=root/'lib'
if (not root.is_absolute() or '..' in root.parts or not library.is_dir()
    or any(value.is_symlink() for value in (root,*root.parents))
    or not library.resolve().is_relative_to(root.resolve())):
 raise SystemExit('Unsafe/missing selected native library directory: '+str(library))
environment=os.environ.copy()
previous=json.loads(environment.get('LAMBDAFORGE_NATIVE_LIBRARY_POLICY','{}'))
owned=set(previous.get('directories',[]))
inherited=[]
for value in environment.get('LD_LIBRARY_PATH','').split(':'):
 if value.startswith('/') and value not in owned and value!=str(library) and value not in inherited:
  inherited.append(value)
environment['LD_LIBRARY_PATH']=':'.join([str(library),*inherited])
policy['inherited_directories']=inherited
environment['LAMBDAFORGE_NATIVE_LIBRARY_POLICY']=json.dumps(policy,sort_keys=True)
command=sys.argv[3:]
if not command: raise SystemExit('Native library launch requires an executable argv')
os.execvpe(command[0],command,environment)
"""


def native_library_report(
    policy: Mapping[str, Any] | None = None, *, verify: bool = False
) -> dict[str, Any]:
    """Read bounded runtime-library provenance; never import scientific packages here."""
    if policy is None:
        policy = json.loads(os.environ.get("LAMBDAFORGE_NATIVE_LIBRARY_POLICY", "null"))
    families = ("libstdc++", "libgcc_s", "libgomp", "libiomp", "libcuda", "libcudart")
    loaded: dict[str, list[str]] = {}
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            path = fields[5].replace("\\040", " ")
            name = Path(path).name
            for family in families:
                if name.startswith(family + ".") or name.startswith(family + "-"):
                    values = loaded.setdefault(family, [])
                    if path not in values:
                        values.append(path)
    validated = []
    if verify and policy is not None:
        if not maps.is_file():
            raise RuntimeError("Cannot attest selected native libraries without /proc/self/maps")
        root = Path(policy["dependency_prefix"]).resolve()
        for name, family in (("libstdc++.so.6", "libstdc++"), ("libgcc_s.so.1", "libgcc_s")):
            expected = root / "lib" / name
            if not expected.exists():
                continue
            expected = expected.resolve()
            actual = loaded.get(family, [])
            if not expected.is_relative_to(root) or any(
                Path(v).resolve() != expected for v in actual
            ):
                raise RuntimeError(
                    f"Native library origin mismatch: {family}; expected={expected}; "
                    f"loaded={actual}"
                )
            if family == "libstdc++" and not actual:
                raise RuntimeError(f"Selected C++ runtime was not observed loaded: {expected}")
            if actual:
                validated.append(str(expected))
    return {
        "policy": dict(policy) if policy is not None else None,
        "loaded": loaded,
        "validated_libraries": validated,
    }


@dataclass(frozen=True, slots=True)
class NativeLibraryPolicy:
    """Derive the loader root from placement, never from a cluster-specific default."""

    dependency_prefix: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.dependency_prefix)
        if (
            not path.is_absolute()
            or path == PurePosixPath(path.anchor)
            or ".." in path.parts
            or any(c in self.dependency_prefix for c in ("\n", "\r", "\0", ":"))
        ):
            raise ValueError("Native library prefix must be an absolute, safe placement path.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": LIBRARY_CONTRACT,
            "dependency_prefix": self.dependency_prefix,
            "directories": [str(PurePosixPath(self.dependency_prefix) / "lib")],
        }

    def launch_prefix(self, python: str) -> tuple[str, ...]:
        """An argv prefix placed inside the provider GPU wrapper, before target startup."""
        return (
            python,
            "-I",
            "-S",
            "-c",
            _LAUNCH,
            LIBRARY_CONTRACT,
            json.dumps(self.to_dict(), sort_keys=True),
        )

    @classmethod
    def from_command(cls, command: Sequence[str]) -> NativeLibraryPolicy | None:
        """Read the versioned native argv envelope, not shell or human CLI output."""
        for index, value in enumerate(command):
            if value != LIBRARY_CONTRACT or index < 3 or command[index - 2] != "-c":
                continue
            if index + 1 >= len(command):
                raise ValueError("Native library command envelope is incomplete")
            payload = json.loads(command[index + 1])
            policy = cls(str(payload["dependency_prefix"]))
            if payload != policy.to_dict():
                raise ValueError("Native library command envelope has an incompatible policy")
            return policy
        return None

    def wrap(self, command: Sequence[str], *, python: str) -> tuple[str, ...]:
        if not command:
            raise ValueError("Native library policy requires a nonempty command.")
        return (*self.launch_prefix(python), *command)

    def probe_code(self, *, scipy_first: bool = False) -> str:
        """Standalone source also works in a dependency base retaining older LF code."""
        order = (
            "if importlib.util.find_spec('scipy') is not None: import scipy.spatial\nimport torch\n"
            if scipy_first
            else "import torch\n"
            "if importlib.util.find_spec('scipy') is not None: import scipy.spatial\n"
        )
        exercises = (
            "assert torch.tensor([1.,2.],device='cpu').sum().item()==3.\n"
            "if importlib.util.find_spec('scipy') is not None:\n"
            " assert scipy.spatial.KDTree([[0.,0.],[1.,1.]]).query([0.,0.])[1]==0\n"
        )
        return (
            "import importlib.util,json,os\nfrom pathlib import Path\n"
            "from typing import Any\nfrom collections.abc import Mapping\n"
            + inspect.getsource(native_library_report)
            + "\n"
            + order
            + exercises
            + f"print({REPORT_MARKER!r}+"
            "json.dumps(native_library_report(verify=True),sort_keys=True))\n"
        )

    def read_report(self, stdout: str) -> dict[str, Any]:
        """Require a report bound to the selected prefix, not merely a successful import."""
        reports = [
            line[len(REPORT_MARKER) :]
            for line in stdout.splitlines()
            if line.startswith(REPORT_MARKER)
        ]
        if len(reports) != 1:
            raise ValueError("Native library verification receipt is missing or ambiguous")
        report = json.loads(reports[0])
        policy = report.get("policy") if isinstance(report, dict) else None
        if not isinstance(policy, dict) or any(
            policy.get(k) != v for k, v in self.to_dict().items()
        ):
            raise ValueError("Native library verification receipt belongs to another prefix/policy")
        if not isinstance(report.get("loaded"), dict):
            raise ValueError("Native library verification receipt has no loaded-library evidence")
        return report

    @staticmethod
    def isolated_environment(environment: Mapping[str, str]) -> dict[str, str]:
        """Remove only the consumer's injected paths when entering another named runtime."""
        result = dict(environment)
        policy = json.loads(result.pop(POLICY_VARIABLE, "{}"))
        owned = set(policy.get("directories", ()))
        if owned:
            result["LD_LIBRARY_PATH"] = ":".join(
                value
                for value in result.get("LD_LIBRARY_PATH", "").split(":")
                if value and value not in owned
            )
        return result
