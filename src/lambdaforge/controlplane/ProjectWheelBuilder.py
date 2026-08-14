"""Build and cache exact pure-Python project wheels for remote deployment."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import Distribution, distribution
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from uuid import uuid4

from lambdaforge.reproducibility.CodeIdentity import CodeIdentity


class ProjectWheelBuilder:
    """Create exact wheels from a source project or an installed distribution."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def build(self, project_root: str | Path) -> Path:
        """Build or reuse a wheel keyed by code identity and packaging declarations."""
        project = Path(project_root).resolve()
        pyproject = project / "pyproject.toml"
        if not pyproject.is_file():
            raise FileNotFoundError(f"Managed deployment requires {pyproject}.")
        identity = CodeIdentity.capture(project).to_dict()
        payload = json.dumps(identity, sort_keys=True).encode("utf-8") + pyproject.read_bytes()
        key = hashlib.sha256(payload).hexdigest()[:24]
        destination = self.root / key
        wheels = tuple(destination.glob("*.whl")) if destination.is_dir() else ()
        if len(wheels) == 1:
            return wheels[0]
        destination.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(destination),
                str(project),
            ),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Could not build exact project wheel: {completed.stderr.strip()}")
        wheels = tuple(destination.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one project wheel in {destination}, found {len(wheels)}.")
        return wheels[0]

    def build_installed(
        self,
        distribution_name: str,
        *,
        source_hint: str | Path | None = None,
    ) -> Path:
        """Build a source checkout or deterministically repack the installed distribution.

        Editable installs expose their source through ``direct_url.json``. A regular wheel install
        has no ``pyproject.toml`` beside ``__file__``; in that case the installed package, metadata
        and LambdaForge shared data are repacked without contacting an index.
        """
        installed = distribution(distribution_name)
        project = self.installed_project_root(installed, source_hint=source_hint)
        if project is not None:
            return self.build(project)
        wheel = self._direct_wheel(installed)
        if wheel is not None:
            return self._cache_wheel(wheel)
        package_name = re.sub(r"[-.]+", "_", distribution_name).lower()
        spec = find_spec(package_name)
        locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
        if len(locations) != 1:
            raise RuntimeError(
                f"Cannot locate one installed package directory for {distribution_name!r}."
            )
        return self._pack_installed(installed, Path(locations[0]))

    @classmethod
    def installed_project_root(
        cls,
        installed: Distribution,
        *,
        source_hint: str | Path | None = None,
    ) -> Path | None:
        """Return a real source root without mistaking ``site-packages`` for a project."""
        if source_hint is not None:
            candidate = Path(source_hint).expanduser().resolve()
            if (candidate / "pyproject.toml").is_file():
                return candidate
        direct = cls._direct_path(installed)
        if direct is not None and direct.is_dir() and (direct / "pyproject.toml").is_file():
            return direct.resolve()
        return None

    @classmethod
    def _direct_wheel(cls, installed: Distribution) -> Path | None:
        direct = cls._direct_path(installed)
        return (
            direct.resolve()
            if direct is not None and direct.is_file() and direct.suffix == ".whl"
            else None
        )

    @staticmethod
    def _direct_path(installed: Distribution) -> Path | None:
        raw = installed.read_text("direct_url.json")
        if raw is None:
            return None
        try:
            url = json.loads(raw).get("url")
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(url, str):
            return None
        parsed = urlparse(url)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            return None
        return Path(url2pathname(unquote(parsed.path))).expanduser()

    def _cache_wheel(self, wheel: Path) -> Path:
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()[:24]
        destination = self.root / f"installed-{digest}" / wheel.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            shutil.copy2(wheel, temporary)
            os.replace(temporary, destination)
        return destination

    def _pack_installed(self, installed: Distribution, package_root: Path) -> Path:
        """Create a deterministic pure-Python wheel from one verified installed package tree."""
        package = package_root.resolve()
        if not package.is_dir() or package.is_symlink():
            raise RuntimeError(f"Installed package directory is missing or unsafe: {package}")
        name = str(installed.metadata["Name"] or package.name)
        normalized_name = re.sub(r"[-.]+", "_", name).lower()
        normalized_version = re.sub(r"[-]+", "_", installed.version)
        dist_info = f"{normalized_name}-{normalized_version}.dist-info"
        data_root = f"{normalized_name}-{normalized_version}.data/data/share/{normalized_name}"
        entries: dict[str, bytes] = {}

        for source in sorted(package.rglob("*")):
            if source.is_symlink():
                raise RuntimeError(f"Refusing to package installed symlink: {source}")
            if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
                continue
            entries[f"{package.name}/{source.relative_to(package).as_posix()}"] = (
                source.read_bytes()
            )

        for filename in ("METADATA", "WHEEL", "entry_points.txt", "top_level.txt"):
            metadata_content = installed.read_text(filename)
            if metadata_content is not None:
                entries[f"{dist_info}/{filename}"] = metadata_content.encode("utf-8")
        if f"{dist_info}/METADATA" not in entries or f"{dist_info}/WHEEL" not in entries:
            raise RuntimeError(f"Installed distribution {name!r} has incomplete wheel metadata.")

        for relative in installed.files or ():
            source = Path(str(installed.locate_file(relative)))
            if not source.is_file() or source.is_symlink():
                continue
            parts = source.resolve().parts
            shared = next(
                (
                    index
                    for index in range(len(parts) - 1)
                    if parts[index] == "share" and parts[index + 1] == normalized_name
                ),
                None,
            )
            if shared is not None:
                tail = Path(*parts[shared + 2 :]).as_posix()
                entries[f"{data_root}/{tail}"] = source.read_bytes()
            relative_text = relative.as_posix()
            marker = ".dist-info/licenses/"
            if marker in relative_text:
                tail = relative_text.split(marker, 1)[1]
                entries[f"{dist_info}/licenses/{tail}"] = source.read_bytes()

        record_path = f"{dist_info}/RECORD"
        record = io.StringIO(newline="")
        writer = csv.writer(record, lineterminator="\n")
        for path, entry_content in sorted(entries.items()):
            digest = base64.urlsafe_b64encode(hashlib.sha256(entry_content).digest()).rstrip(b"=")
            writer.writerow((path, f"sha256={digest.decode('ascii')}", len(entry_content)))
        writer.writerow((record_path, "", ""))
        entries[record_path] = record.getvalue().encode("utf-8")

        identity = hashlib.sha256()
        for path, entry_content in sorted(entries.items()):
            identity.update(path.encode("utf-8") + b"\0" + entry_content + b"\0")
        destination = self.root / f"installed-{identity.hexdigest()[:24]}"
        wheel = destination / f"{normalized_name}-{normalized_version}-py3-none-any.whl"
        if wheel.is_file():
            return wheel
        destination.mkdir(parents=True, exist_ok=True)
        temporary = wheel.with_name(f".{wheel.name}.{uuid4().hex}.tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path, entry_content in sorted(entries.items()):
                    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    archive.writestr(info, entry_content)
            os.replace(temporary, wheel)
        finally:
            temporary.unlink(missing_ok=True)
        return wheel
