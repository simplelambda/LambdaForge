"""Write one owned sample's numeric fields; publish with Work.outputs.dataset afterwards."""

from pathlib import Path
from typing import Any

from lambdaforge.catalog.errors import ContractError


class WriteSampleAssets:
    """Explicit immutable-dataset member construction, not an alternate publisher.

    `directory` must be an owned Work checkpoint/output staging directory. A checkpointed
    consumer recipe can skip completed samples with resume_map; final publication stays atomic.
    """

    def __call__(
        self,
        directory: Path,
        *,
        member_id: str,
        arrays: dict[str, Any],
        sidecar: dict[str, Any],
        targets: dict[str, Any],
        partitions: dict[str, Any],
    ) -> dict[str, Any]:
        import hashlib
        import json

        import numpy as np

        from lambdaforge.work.snapshot import validate_path

        validate_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = hashlib.sha256(member_id.encode()).hexdigest()
        numeric: dict[str, Any] = {k: np.asarray(v) for k, v in arrays.items()}
        if any(v.dtype.hasobject for v in numeric.values()):
            raise ContractError("UNSAFE_DTYPE", "Sample writer refuses pickle/object arrays.")
        base, metadata = directory / (stem + ".npz"), directory / (stem + ".json")
        if base.exists() or metadata.exists():
            raise ContractError(
                "EXISTING_OUTPUT", "Choose fresh staging or restore a completed checkpoint."
            )
        with base.open("xb") as file:
            np.savez_compressed(file, **numeric)
        with metadata.open("x", encoding="utf-8") as file:
            json.dump(sidecar, file, allow_nan=False)
        return {
            "id": member_id,
            "assets": {"sample": base, "sidecar": metadata},
            "targets": targets,
            "partitions": partitions,
        }
