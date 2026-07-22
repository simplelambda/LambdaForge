"""Containment and pattern checks for retention filesystem operations."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path, PurePosixPath


class ArtifactPathGuard:
    """Reject traversal, links, reparse points and protected internal paths."""

    _DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
    _RULE_PROTECTED_PREFIXES = frozenset({"aggregate", "checkpoints", ".lambdaforge"})

    @classmethod
    def validate_pattern(cls, pattern: str, *, rule_pattern: bool) -> str:
        """Validate a relative POSIX glob without touching the filesystem."""
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Retention patterns must be non-empty strings.")
        if "\0" in pattern:
            raise ValueError("Retention patterns cannot contain NUL bytes.")
        if "\\" in pattern:
            raise ValueError("Retention patterns must use POSIX '/' separators.")
        if pattern.startswith(("/", "//")) or cls._DRIVE_PATTERN.match(pattern):
            raise ValueError(f"Retention pattern must be relative: {pattern!r}.")
        segments = PurePosixPath(pattern).parts
        if any(segment == ".." for segment in segments):
            raise ValueError(f"Retention pattern cannot traverse parents: {pattern!r}.")
        if rule_pattern and segments:
            prefix = segments[0]
            if prefix in cls._RULE_PROTECTED_PREFIXES:
                raise ValueError(
                    f"Generic retention rules cannot target protected {prefix!r} paths."
                )
        return pattern

    @staticmethod
    def matches(relative_path: str, patterns: tuple[str, ...]) -> bool:
        """Match one normalized relative path against POSIX pathlib globs."""
        candidate = PurePosixPath(relative_path)
        for pattern in patterns:
            if candidate.match(pattern):
                return True
            if pattern.endswith("/**"):
                prefix = pattern[:-3].rstrip("/")
                if relative_path == prefix or relative_path.startswith(f"{prefix}/"):
                    return True
        return False

    @classmethod
    def relative_regular_file(cls, root: Path, path: Path) -> str | None:
        """Return a safe POSIX relative path or None for unsafe entries."""
        try:
            relative = path.relative_to(root)
        except ValueError:
            return None
        if not relative.parts:
            return None
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            try:
                metadata = cursor.lstat()
            except OSError:
                return None
            if stat.S_ISLNK(metadata.st_mode) or cls._is_reparse_point(metadata):
                return None
        try:
            metadata = path.lstat()
        except OSError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            return None
        try:
            resolved_root = root.resolve(strict=True)
            resolved_path = path.resolve(strict=True)
        except OSError:
            return None
        if not resolved_path.is_relative_to(resolved_root):
            return None
        return relative.as_posix()

    @classmethod
    def safe_destination(cls, root: Path, relative_path: str) -> Path:
        """Resolve a destination without accepting linked path components."""
        cls.validate_pattern(relative_path, rule_pattern=False)
        try:
            root_metadata = root.lstat()
        except OSError as error:
            raise ValueError(f"Retention root is unavailable: {root}.") from error
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_ISLNK(root_metadata.st_mode)
            or cls._is_reparse_point(root_metadata)
        ):
            raise ValueError(f"Retention root is not a safe directory: {root}.")
        destination = root.joinpath(*PurePosixPath(relative_path).parts)
        cursor = root
        for part in PurePosixPath(relative_path).parts[:-1]:
            cursor = cursor / part
            try:
                metadata = cursor.lstat()
            except OSError as error:
                raise ValueError(
                    f"Retention destination parent is unavailable: {cursor}."
                ) from error
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or cls._is_reparse_point(metadata)
            ):
                raise ValueError(f"Unsafe retention destination parent: {cursor}.")
        try:
            destination_metadata = destination.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ValueError(f"Retention destination is unavailable: {destination}.") from error
        else:
            if stat.S_ISLNK(destination_metadata.st_mode) or cls._is_reparse_point(
                destination_metadata
            ):
                raise ValueError(f"Unsafe retention destination: {destination}.")
        resolved_root = root.resolve()
        resolved_parent = destination.parent.resolve()
        if not resolved_parent.is_relative_to(resolved_root):
            raise ValueError(f"Retention destination escapes its root: {relative_path!r}.")
        return destination

    @classmethod
    def ensure_directory(cls, root: Path, relative_path: str) -> Path:
        """Create an internal directory one component at a time and reject links."""
        cls.validate_pattern(relative_path, rule_pattern=False)
        try:
            root_metadata = root.lstat()
        except OSError as error:
            raise ValueError(f"Retention root is unavailable: {root}.") from error
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_ISLNK(root_metadata.st_mode)
            or cls._is_reparse_point(root_metadata)
        ):
            raise ValueError(f"Retention root is not a safe directory: {root}.")
        resolved_root = root.resolve(strict=True)
        cursor = root
        for part in PurePosixPath(relative_path).parts:
            cursor = cursor / part
            try:
                cursor.mkdir()
            except FileExistsError:
                pass
            try:
                metadata = cursor.lstat()
            except OSError as error:
                raise ValueError(f"Retention directory is unavailable: {cursor}.") from error
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or cls._is_reparse_point(metadata)
                or not cursor.resolve(strict=True).is_relative_to(resolved_root)
            ):
                raise ValueError(f"Unsafe retention directory: {cursor}.")
        return cursor

    @classmethod
    def validate_regular_tree(cls, root: Path) -> None:
        """Reject links, reparse points and special entries in an internal tree."""
        try:
            root_metadata = root.lstat()
        except OSError as error:
            raise ValueError(f"Retention tree is unavailable: {root}.") from error
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_ISLNK(root_metadata.st_mode)
            or cls._is_reparse_point(root_metadata)
        ):
            raise ValueError(f"Retention tree root is unsafe: {root}.")
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as iterator:
                    entries = tuple(iterator)
            except OSError as error:
                raise ValueError(f"Retention tree is unreadable: {directory}.") from error
            for entry in entries:
                path = Path(entry.path)
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise ValueError(f"Retention tree entry is unreadable: {path}.") from error
                if stat.S_ISLNK(metadata.st_mode) or cls._is_reparse_point(metadata):
                    raise ValueError(f"Retention tree contains an unsafe link: {path}.")
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                elif not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(f"Retention tree contains a special entry: {path}.")

    @staticmethod
    def _is_reparse_point(metadata: os.stat_result) -> bool:
        attributes = getattr(metadata, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return bool(attributes & reparse_flag)
