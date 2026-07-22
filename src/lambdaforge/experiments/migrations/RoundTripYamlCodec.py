"""Safe round-trip YAML loading and rendering for migration previews."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping, MutableMapping, Sequence, Set
from io import StringIO
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


class RoundTripYamlCodec:
    """Preserve YAML presentation data without constructing Python objects."""

    def load_file(self, path: str | Path) -> tuple[MutableMapping[str, Any], str]:
        """Load one UTF-8 mapping document and return its original text."""
        path = Path(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            text = handle.read()
        return self.load_text(text, source=str(path)), text

    def load_text(
        self,
        text: str,
        *,
        source: str = "<string>",
    ) -> MutableMapping[str, Any]:
        """Load exactly one mapping document with duplicate keys rejected."""
        yaml = self._yaml()
        value = yaml.load(text)
        if not isinstance(value, MutableMapping):
            raise TypeError(f"Experiment YAML root must be a mapping: {source}")
        return value

    def dump(
        self,
        config: MutableMapping[str, Any],
        *,
        newline: str = "\n",
    ) -> str:
        """Render one round-trip mapping with a stable final newline."""
        stream = StringIO()
        self._yaml().dump(config, stream)
        text = stream.getvalue()
        if not text.endswith("\n"):
            text += "\n"
        return text if newline == "\n" else text.replace("\n", newline)

    def dump_preview(
        self,
        config: MutableMapping[str, Any],
        *,
        newline: str = "\n",
    ) -> str:
        """Render a semantic preview without constraining programmatic values."""
        try:
            return self.dump(config, newline=newline)
        except Exception:
            projected = self._to_preview_value(config, {})
            if not isinstance(projected, MutableMapping):
                raise TypeError("Preview projection must produce a mutable mapping.") from None
            return self.dump(projected, newline=newline)

    def newline_for(self, text: str) -> str:
        """Return the source's dominant newline convention."""
        return "\r\n" if "\r\n" in text else "\n"

    def to_plain_mapping(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Detach round-trip node types while preserving values and aliases."""
        converted = self._to_plain_value(config, {})
        if not isinstance(converted, dict):
            raise TypeError("Converted experiment configuration must be a dictionary.")
        return converted

    def _to_plain_value(self, value: Any, memo: dict[int, Any]) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return str(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)

        identity = id(value)
        if identity in memo:
            return memo[identity]
        if isinstance(value, Mapping):
            output: dict[Any, Any] = {}
            memo[identity] = output
            for key, item in value.items():
                output[self._to_plain_value(key, memo)] = self._to_plain_value(item, memo)
            return output
        if isinstance(value, tuple):
            sequence_tuple: list[Any] = []
            memo[identity] = sequence_tuple
            sequence_tuple.extend(self._to_plain_value(item, memo) for item in value)
            converted_tuple = tuple(sequence_tuple)
            memo[identity] = converted_tuple
            return converted_tuple
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            sequence: list[Any] = []
            memo[identity] = sequence
            sequence.extend(self._to_plain_value(item, memo) for item in value)
            return sequence
        if isinstance(value, Set):
            collection: set[Any] = set()
            memo[identity] = collection
            collection.update(self._to_plain_value(item, memo) for item in value)
            return collection
        return copy.deepcopy(value)

    def _to_preview_value(self, value: Any, memo: dict[int, Any]) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, os.PathLike):
            return os.fspath(value)

        identity = id(value)
        if identity in memo:
            return memo[identity]
        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            memo[identity] = output
            for key, item in value.items():
                output[str(key)] = self._to_preview_value(item, memo)
            return output
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            sequence: list[Any] = []
            memo[identity] = sequence
            sequence.extend(self._to_preview_value(item, memo) for item in value)
            return sequence
        if isinstance(value, Set):
            return sorted(
                (self._to_preview_value(item, memo) for item in value),
                key=repr,
            )
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            return isoformat()
        value_type = type(value)
        return f"<{value_type.__module__}.{value_type.__qualname__}>"

    def _yaml(self) -> Any:
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.allow_duplicate_keys = False
        yaml.allow_unicode = True
        yaml.width = 4_096
        yaml.indent(mapping=2, sequence=4, offset=2)
        return yaml
