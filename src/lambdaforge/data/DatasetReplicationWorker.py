"""Controller-relay command owned by the native detached Job supervisor."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def run(path: Path) -> None:
    from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
    from lambdaforge.controlplane.ClusterProfile import ClusterProfile
    from lambdaforge.data.DatasetRegistry import DatasetRegistry
    from lambdaforge.data.DatasetService import DatasetService
    from lambdaforge.work.atomic import atomic_write_json
    from lambdaforge.work.snapshot import validate_path

    validate_path(path)
    request = json.loads(path.read_text(encoding="utf-8"))
    catalog = ClusterCatalog(
        {
            name: ClusterProfile.from_mapping(name, value)
            for name, value in request["profiles"].items()
        }
    )
    service = DatasetService(DatasetRegistry(request["registry"]), catalog)

    def progress(value: Any) -> None:
        from datetime import datetime, timezone

        path = Path(os.environ["LAMBDAFORGE_PROGRESS_PATH"])
        previous = json.loads(path.read_text()) if path.is_file() else {}
        now = datetime.now(timezone.utc).isoformat()
        advanced = value.get("compressed_bytes", 0) > previous.get("compressed_bytes", 0)
        atomic_write_json(
            path,
            {
                **previous,
                **dict(value),
                "updated_at_utc": now,
                "last_byte_progress_at_utc": now
                if advanced
                else previous.get("last_byte_progress_at_utc"),
            },
        )

    service.replicate(
        request["dataset"],
        source=request["source"],
        destination=request["destination"],
        expected_content_id=request["content_id"],
        route="relay",
        apply=True,
        progress=progress,
    )
    placement = next(
        p
        for p in service.registry.get(request["dataset"]).placements
        if p.cluster == request["destination"] and p.verified
    )
    atomic_write_json(
        Path(os.environ["LAMBDAFORGE_JOB_RESULT_PATH"]),
        {
            "dataset": request["dataset"],
            "content_id": request["content_id"],
            "destination": request["destination"],
            "verified": True,
            "placement": placement.to_dict(),
        },
    )
    progress({"phase": "completed", "message": "Verified replica registered; transfer complete."})


if __name__ == "__main__":
    run(Path(sys.argv[1]))
