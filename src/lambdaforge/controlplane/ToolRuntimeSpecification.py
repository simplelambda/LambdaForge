"""Project-owned isolated tool declarations, never consumer installation hooks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import tomli
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from lambdaforge.ImmutableJson import FrozenJsonMapping


def public_https(value: str) -> str:
    """Package sources are explicit HTTPS locations, without inline credentials."""
    url = urlsplit(value)
    if url.scheme != "https" or not url.netloc or url.username or url.password or url.query:
        raise ValueError("Tool package sources require HTTPS without credentials/query secrets.")
    return value


@dataclass(frozen=True, slots=True)
class ToolRuntimeSpecification:
    """One isolated Python/package contract; entrypoint code has a separate identity."""

    name: str
    project_root: Path
    policy: Mapping[str, Any]
    entrypoints: Mapping[str, Path]

    @classmethod
    def discover(cls, project: Path | None) -> tuple[ToolRuntimeSpecification, ...]:
        if project is None:
            return ()
        path = project / "pyproject.toml"
        if not path.is_file():
            return ()
        with path.open("rb") as stream:
            raw = tomli.load(stream).get("tool", {}).get("lambdaforge", {}).get("tool_runtimes", {})
        if not isinstance(raw, dict):
            raise ValueError("tool.lambdaforge.tool_runtimes must be a named TOML table.")
        return tuple(
            cls.from_mapping(name, values, project) for name, values in sorted(raw.items())
        )

    @classmethod
    def from_mapping(
        cls, name: str, value: Mapping[str, Any], project: Path
    ) -> ToolRuntimeSpecification:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) is None or not isinstance(
            value, Mapping
        ):
            raise ValueError("Tool runtimes require safe names and declarative TOML tables.")
        allowed = {
            "python",
            "requirements",
            "index_url",
            "extra_index_urls",
            "wheel_sources",
            "source_packages",
            "imports",
            "entrypoints",
            "torch_index",
            "torch_cuda",
        }
        if set(value) - allowed:
            raise ValueError(f"Unknown isolated tool options: {sorted(set(value) - allowed)}")
        python = str(value.get("python", ""))
        if re.fullmatch(r"3\.(?:1[0-9]|[2-9][0-9])(?:\.[0-9]+)?", python) is None:
            raise ValueError('Tool python must pin a supported minor/patch, e.g. "3.12".')
        requirements = value.get("requirements", [])
        if (
            not isinstance(requirements, list)
            or not requirements
            or not all(isinstance(v, str) for v in requirements)
        ):
            raise ValueError(
                "Tool requirements must be a nonempty array of pinned PEP 508 requirements."
            )
        by_name: dict[str, Requirement] = {}
        vcs = set()
        for text in requirements:
            requirement = Requirement(text)
            package = canonicalize_name(requirement.name)
            if package in by_name or requirement.marker is not None:
                raise ValueError("Tool top-level requirements must be unique and unconditional.")
            if requirement.url:
                if not re.fullmatch(r"git\+https://[^?#\s]+\.git@[a-fA-F0-9]{40}", requirement.url):
                    raise ValueError(
                        "Tool source requirements need a full immutable HTTPS Git revision."
                    )
                public_https(requirement.url[4:])
                vcs.add(package)
            else:
                pins = list(requirement.specifier)
                if len(pins) != 1 or pins[0].operator != "==" or "*" in pins[0].version:
                    raise ValueError(f"Tool requirement must be exactly pinned: {text}")
            by_name[package] = requirement
        source_packages = value.get("source_packages", [])
        if not isinstance(source_packages, list) or not all(
            isinstance(v, str) for v in source_packages
        ):
            raise ValueError("source_packages must name explicitly pinned source-build exceptions.")
        sources = set(map(canonicalize_name, source_packages)) | vcs
        if not sources <= set(by_name):
            raise ValueError("Every source-build exception must be a pinned top-level requirement.")
        wheel_sources = value.get("wheel_sources", [])
        if not isinstance(wheel_sources, list):
            raise ValueError("wheel_sources must be an array of source-policy tables.")
        for source in wheel_sources:
            if not isinstance(source, Mapping) or set(source) != {"url", "packages", "build_tag"}:
                raise ValueError(
                    "Each wheel source requires url, packages and exact local build_tag."
                )
            public_https(str(source["url"]))
            if (
                not isinstance(source["packages"], list)
                or not source["packages"]
                or not all(isinstance(v, str) for v in source["packages"])
            ):
                raise ValueError("A wheel source must own explicitly pinned packages.")
            if re.fullmatch("[a-z0-9][a-z0-9.]*", str(source["build_tag"])) is None:
                raise ValueError("build_tag must identify the exact wheel local-version build.")
            for package in map(canonicalize_name, source["packages"]):
                if package not in by_name or package in sources:
                    raise ValueError(
                        "Compiled wheel packages must be pinned and cannot permit source builds."
                    )
                pin = next(iter(by_name[package].specifier), None)
                if pin is None or not pin.version.endswith("+" + source["build_tag"]):
                    raise ValueError(
                        "Wheel-source packages must pin the matching +build_tag version."
                    )
        imports = value.get("imports", [])
        if not isinstance(imports, list) or not all(
            isinstance(v, str) and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", v)
            for v in imports
        ):
            raise ValueError("Tool imports must be explicit Python module names.")
        indexes = value.get("extra_index_urls", [])
        if not isinstance(indexes, list) or not all(isinstance(v, str) for v in indexes):
            raise ValueError("extra_index_urls must be an array of public HTTPS URLs.")
        policy = {
            "tool_runtime_version": 1,
            "python": python,
            "requirements": sorted(requirements),
            "index_url": public_https(str(value.get("index_url", "https://pypi.org/simple"))),
            "extra_index_urls": [public_https(v) for v in indexes],
            "wheel_sources": wheel_sources,
            "source_packages": sorted(sources),
            "imports": imports,
        }
        if "torch" in by_name:
            torch = by_name["torch"]
            if torch.url or "torch_index" not in value or "torch_cuda" not in value:
                raise ValueError(
                    "A Torch tool requires an exact wheel version, torch_index and "
                    'torch_cuda ("cpu" or e.g. "12.4").'
                )
            cuda = str(value["torch_cuda"])
            if cuda != "cpu" and not re.fullmatch(r"[0-9]+\.[0-9]+", cuda):
                raise ValueError("torch_cuda must be cpu or an exact CUDA major.minor.")
            policy.update(torch_index=public_https(str(value["torch_index"])), torch_cuda=cuda)
        elif "torch_index" in value or "torch_cuda" in value:
            raise ValueError("Torch source/ABI policy requires a pinned torch requirement.")
        entrypoints = {}
        raw_entries = value.get("entrypoints", {})
        if not isinstance(raw_entries, Mapping) or len(raw_entries) > 32:
            raise ValueError(
                "Tool entrypoints must map names to small project-contained Python files."
            )
        for key, relative in raw_entries.items():
            if (
                not isinstance(key, str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", key) is None
            ):
                raise ValueError("Invalid tool entrypoint name.")
            authored = project / str(relative)
            if (
                Path(str(relative)).is_absolute()
                or ".." in Path(str(relative)).parts
                or any(p.is_symlink() for p in (authored, *authored.parents))
            ):
                raise ValueError(
                    "Tool entrypoints must not traverse symlinks or escape the project."
                )
            if (
                authored.suffix != ".py"
                or not authored.is_file()
                or authored.stat().st_size > 1024**2
            ):
                raise ValueError(
                    "Tool entrypoints must be existing Python files no larger than 1 MiB."
                )
            entrypoints[key] = authored.resolve()
        return cls(
            name, project.resolve(), FrozenJsonMapping(policy), MappingProxyType(entrypoints)
        )

    @property
    def contract_id(self) -> str:
        return (
            "tool-"
            + hashlib.sha256(json.dumps(self.policy, sort_keys=True).encode()).hexdigest()[:24]
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contract_id": self.contract_id,
            "policy": dict(self.policy),
            "entrypoints": {
                key: {
                    "path": str(path.relative_to(self.project_root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for key, path in self.entrypoints.items()
            },
        }
