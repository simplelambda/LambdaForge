"""Deterministic non-pickle serializer for common NumPy/PyTorch sample trees."""

from __future__ import annotations

import base64
import io
import json
import math
import zipfile
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from lambdaforge.data.cache.DatasetSerializer import DatasetSerializer


class NumpyDatasetSerializer(DatasetSerializer):
    """Serialize safe scalar/container/array trees without importing user code.

    The archive contains canonical JSON and individually bounded `.npy`
    members written with `allow_pickle=False`. ZIP member sizes are checked
    before any array is materialized.
    """

    FORMAT_VERSION = "1"
    MANIFEST_NAME = "manifest.json"

    def __init__(
        self,
        compressed: bool = False,
        max_arrays: int = 1024,
        max_decoded_bytes: int = 1_073_741_824,
        max_manifest_bytes: int = 16_777_216,
        max_archive_bytes: int = 1_073_741_824,
        max_depth: int = 64,
    ) -> None:
        if not isinstance(compressed, bool):
            raise TypeError("compressed must be a bool.")
        for name, value in (
            ("max_arrays", max_arrays),
            ("max_decoded_bytes", max_decoded_bytes),
            ("max_manifest_bytes", max_manifest_bytes),
            ("max_archive_bytes", max_archive_bytes),
            ("max_depth", max_depth),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.compressed = compressed
        self.max_arrays = max_arrays
        self.max_decoded_bytes = max_decoded_bytes
        self.max_manifest_bytes = max_manifest_bytes
        self.max_archive_bytes = max_archive_bytes
        self.max_depth = max_depth

    @property
    def format_fingerprint(self) -> str:
        """Identify bytes whose encoding semantics can affect cache keys."""
        compression = "deflate" if self.compressed else "stored"
        return f"lambdaforge-numpy-tree-{self.FORMAT_VERSION}:{compression}"

    def dumps(self, value: Any) -> bytes:
        """Encode one supported tree into deterministic ZIP/NPY bytes."""
        arrays: list[np.ndarray[Any, Any]] = []
        node = self._encode_node(value, arrays, depth=0, active=set())
        manifest = json.dumps(
            {"format_version": self.FORMAT_VERSION, "value": node},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(manifest) > self.max_manifest_bytes:
            raise ValueError("Serialized dataset manifest exceeds max_manifest_bytes.")
        output = io.BytesIO()
        compression = zipfile.ZIP_DEFLATED if self.compressed else zipfile.ZIP_STORED
        with zipfile.ZipFile(
            output, mode="w", compression=compression, strict_timestamps=True
        ) as zf:
            self._write_member(zf, self.MANIFEST_NAME, manifest, compression)
            for index, array in enumerate(arrays):
                member = io.BytesIO()
                np.lib.format.write_array(member, array, allow_pickle=False)
                self._write_member(
                    zf,
                    self._array_name(index),
                    member.getvalue(),
                    compression,
                )
        payload = output.getvalue()
        if len(payload) > self.max_archive_bytes:
            raise ValueError("Serialized dataset archive exceeds max_archive_bytes.")
        return payload

    def loads(self, payload: Any) -> Any:
        """Validate archive bounds and reconstruct one independent sample tree."""
        raw = bytes(payload)
        if len(raw) > self.max_archive_bytes:
            raise ValueError("Dataset archive exceeds max_archive_bytes.")
        with zipfile.ZipFile(io.BytesIO(raw), mode="r") as zf:
            infos = zf.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("Dataset archive contains duplicate members.")
            if self.MANIFEST_NAME not in names:
                raise ValueError("Dataset archive is missing its manifest.")
            array_infos = [info for info in infos if info.filename != self.MANIFEST_NAME]
            if len(array_infos) > self.max_arrays:
                raise ValueError("Dataset archive exceeds max_arrays.")
            if any(
                info.filename != self._array_name(index) for index, info in enumerate(array_infos)
            ):
                raise ValueError("Dataset archive contains unexpected or unordered members.")
            manifest_info = infos[names.index(self.MANIFEST_NAME)]
            if manifest_info.file_size > self.max_manifest_bytes:
                raise ValueError("Dataset archive manifest exceeds max_manifest_bytes.")
            total_declared = sum(info.file_size for info in array_infos)
            if total_declared > self.max_decoded_bytes + 4096 * len(array_infos):
                raise ValueError(
                    "Dataset archive declared array bytes exceed the configured limit."
                )
            try:
                manifest = json.loads(zf.read(self.MANIFEST_NAME).decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("Dataset archive manifest is invalid.") from error
            if (
                not isinstance(manifest, dict)
                or manifest.get("format_version") != self.FORMAT_VERSION
                or set(manifest) != {"format_version", "value"}
            ):
                raise ValueError("Dataset archive format version is unsupported.")
            arrays: list[np.ndarray[Any, Any]] = []
            decoded_bytes = 0
            for info in array_infos:
                member = zf.read(info.filename)
                expected_bytes = self._validate_array_member(member)
                decoded_bytes += expected_bytes
                if decoded_bytes > self.max_decoded_bytes:
                    raise ValueError("Decoded arrays exceed max_decoded_bytes.")
                array = np.load(io.BytesIO(member), allow_pickle=False)
                if (
                    not isinstance(array, np.ndarray)
                    or array.dtype.hasobject
                    or array.dtype.fields is not None
                    or int(array.nbytes) != expected_bytes
                ):
                    raise ValueError("Dataset archive contains an unsafe array dtype.")
                arrays.append(np.array(array, copy=True))
        value, used = self._decode_node(manifest["value"], arrays, depth=0)
        if used != set(range(len(arrays))):
            raise ValueError("Dataset archive contains unreferenced array members.")
        return value

    def _encode_node(
        self,
        value: Any,
        arrays: list[np.ndarray[Any, Any]],
        *,
        depth: int,
        active: set[int],
    ) -> dict[str, Any]:
        if depth > self.max_depth:
            raise ValueError("Dataset sample exceeds max_depth.")
        if value is None:
            return {"type": "none"}
        if type(value) is bool:
            return {"type": "bool", "value": value}
        if type(value) is int:
            return {"type": "int", "value": str(value)}
        if type(value) is float:
            return {"type": "float", "value": repr(value)}
        if isinstance(value, str):
            return {"type": "str", "value": value}
        if isinstance(value, bytes):
            return {
                "type": "bytes",
                "value": base64.b64encode(value).decode("ascii"),
            }
        if torch.is_tensor(value):
            if value.is_cuda or value.layout is not torch.strided or value.is_quantized:
                raise TypeError("Only dense, non-quantized CPU tensors can be safely cached.")
            try:
                array = value.detach().contiguous().numpy()
            except (RuntimeError, TypeError) as error:
                raise TypeError(
                    "Tensor dtype cannot be represented safely as a NumPy array."
                ) from error
            index = self._append_array(array, arrays)
            return {"type": "tensor", "array": index}
        if isinstance(value, np.ndarray):
            index = self._append_array(value, arrays)
            return {"type": "ndarray", "array": index}
        if isinstance(value, np.generic):
            index = self._append_array(np.asarray(value), arrays)
            return {"type": "numpy_scalar", "array": index}
        identity = id(value)
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Safe dataset mappings require string keys.")
            if identity in active:
                raise ValueError("Cyclic dataset samples cannot be serialized.")
            active.add(identity)
            try:
                return {
                    "type": "mapping",
                    "items": [
                        [
                            key,
                            self._encode_node(item, arrays, depth=depth + 1, active=active),
                        ]
                        for key, item in sorted(value.items())
                    ],
                }
            finally:
                active.remove(identity)
        if isinstance(value, (list, tuple)):
            if identity in active:
                raise ValueError("Cyclic dataset samples cannot be serialized.")
            active.add(identity)
            try:
                return {
                    "type": "tuple" if isinstance(value, tuple) else "list",
                    "items": [
                        self._encode_node(item, arrays, depth=depth + 1, active=active)
                        for item in value
                    ],
                }
            finally:
                active.remove(identity)
        raise TypeError(
            f"NumpyDatasetSerializer does not support {type(value).__module__}."
            f"{type(value).__qualname__}; use explicit trusted pickle only when necessary."
        )

    def _decode_node(
        self,
        node: Any,
        arrays: list[np.ndarray[Any, Any]],
        *,
        depth: int,
    ) -> tuple[Any, set[int]]:
        if depth > self.max_depth or not isinstance(node, dict):
            raise ValueError("Dataset archive contains an invalid value tree.")
        kind = node.get("type")
        if kind == "none" and set(node) == {"type"}:
            return None, set()
        if kind == "bool" and set(node) == {"type", "value"}:
            if type(node["value"]) is not bool:
                raise ValueError("Dataset archive boolean value is invalid.")
            return node["value"], set()
        if kind == "int" and set(node) == {"type", "value"}:
            if not isinstance(node["value"], str):
                raise ValueError("Dataset archive integer value is invalid.")
            return int(node["value"]), set()
        if kind == "float" and set(node) == {"type", "value"}:
            if not isinstance(node["value"], str):
                raise ValueError("Dataset archive float value is invalid.")
            return float(node["value"]), set()
        if kind == "str" and set(node) == {"type", "value"}:
            if not isinstance(node["value"], str):
                raise ValueError("Dataset archive string value is invalid.")
            return node["value"], set()
        if kind == "bytes" and set(node) == {"type", "value"}:
            if not isinstance(node["value"], str):
                raise ValueError("Dataset archive bytes value is invalid.")
            try:
                return base64.b64decode(node["value"], validate=True), set()
            except (TypeError, ValueError) as error:
                raise ValueError("Dataset archive contains invalid base64 bytes.") from error
        if kind in {"ndarray", "numpy_scalar", "tensor"} and set(node) == {
            "type",
            "array",
        }:
            index = node["array"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(arrays)
            ):
                raise ValueError("Dataset archive references an invalid array.")
            array = np.array(arrays[index], copy=True)
            if kind == "tensor":
                try:
                    return torch.from_numpy(array), {index}
                except (RuntimeError, TypeError) as error:
                    raise ValueError("Cached array dtype cannot reconstruct a tensor.") from error
            if kind == "numpy_scalar":
                if array.ndim != 0:
                    raise ValueError("Cached NumPy scalar is not zero-dimensional.")
                return array[()], {index}
            return array, {index}
        if kind == "mapping" and set(node) == {"type", "items"}:
            items = node["items"]
            if not isinstance(items, list):
                raise ValueError("Dataset archive mapping items must be a list.")
            mapping_result: dict[str, Any] = {}
            mapping_used: set[int] = set()
            for pair in items:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                    or pair[0] in mapping_result
                ):
                    raise ValueError("Dataset archive mapping entry is invalid.")
                value, child_used = self._decode_node(
                    pair[1],
                    arrays,
                    depth=depth + 1,
                )
                mapping_result[pair[0]] = value
                mapping_used.update(child_used)
            return mapping_result, mapping_used
        if kind in {"list", "tuple"} and set(node) == {"type", "items"}:
            items = node["items"]
            if not isinstance(items, list):
                raise ValueError("Dataset archive sequence items must be a list.")
            sequence_result: list[Any] = []
            sequence_used: set[int] = set()
            for child in items:
                value, child_used = self._decode_node(child, arrays, depth=depth + 1)
                sequence_result.append(value)
                sequence_used.update(child_used)
            value = tuple(sequence_result) if kind == "tuple" else sequence_result
            return value, sequence_used
        raise ValueError("Dataset archive contains an unsupported value node.")

    def _append_array(
        self,
        value: np.ndarray[Any, Any],
        arrays: list[np.ndarray[Any, Any]],
    ) -> int:
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise TypeError("Object or structured NumPy dtypes are not safe cache values.")
        if len(arrays) >= self.max_arrays:
            raise ValueError("Dataset sample exceeds max_arrays.")
        total = sum(int(item.nbytes) for item in arrays) + int(array.nbytes)
        if total > self.max_decoded_bytes:
            raise ValueError("Dataset sample arrays exceed max_decoded_bytes.")
        arrays.append(np.array(array, copy=True))
        return len(arrays) - 1

    def _validate_array_member(self, payload: bytes) -> int:
        stream = io.BytesIO(payload)
        try:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("Dataset archive uses an unsupported NPY version.")
        except (EOFError, ValueError) as error:
            raise ValueError("Dataset archive contains an invalid NPY member.") from error
        if dtype.hasobject or dtype.fields is not None:
            raise ValueError("Dataset archive contains an unsafe array dtype.")
        expected = math.prod(shape) * int(dtype.itemsize)
        if expected > self.max_decoded_bytes:
            raise ValueError("Dataset array exceeds max_decoded_bytes.")
        if len(payload) - stream.tell() != expected:
            raise ValueError("Dataset NPY member length does not match its declared shape.")
        return expected

    @staticmethod
    def _array_name(index: int) -> str:
        return f"arrays/{index:08d}.npy"

    @staticmethod
    def _write_member(
        archive: zipfile.ZipFile,
        name: str,
        payload: bytes,
        compression: int,
    ) -> None:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = compression
        info.create_system = 0
        info.external_attr = 0
        archive.writestr(info, payload)
