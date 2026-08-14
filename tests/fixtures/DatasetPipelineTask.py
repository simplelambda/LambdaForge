"""Generic multi-stage DatasetRecipe fixture."""

from __future__ import annotations

import shutil
from pathlib import Path

from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.tasks import ArtifactDeclaration, TaskArtifact, TaskContext, TaskOutput


class DatasetPipelineTask:
    """Create/copy/enrich one generic logical member collection."""

    def __init__(
        self,
        stage: str,
        upstream: str | None = None,
        marker: str = "v1",
        fail: bool = False,
    ) -> None:
        self.stage = stage
        self.upstream = upstream
        self.marker = marker
        self.fail = fail

    def run(self, context: TaskContext) -> TaskOutput:
        if self.fail:
            raise RuntimeError(f"requested fixture failure in {self.stage}")
        root = context.output_path("dataset")
        if self.upstream:
            shutil.copytree(Path(self.upstream), root)
        else:
            root.mkdir(parents=True)
        index_path = root / "members.jsonl"
        if self.stage == "roster":
            (root / "samples").mkdir()
            members = []
            for number, split in ((1, "train"), (2, "test")):
                sample = root / "samples" / f"{number}.txt"
                sample.write_text(f"sample-{number}\n", encoding="utf-8")
                members.append(
                    DatasetMember(
                        f"sample-{number}",
                        partitions={"split": split, "fold": number},
                        targets={"score": float(number)},
                        metadata={"source": "fixture"},
                        display={"description": f"Sample {number}"},
                        assets={"raw": self._asset(root, sample)},
                    )
                )
        else:
            members = list(DatasetIndex(index_path))
        if self.stage == "features":
            enriched = []
            for member in members:
                feature = root / "features" / f"{member.member_id}.bin"
                feature.parent.mkdir(exist_ok=True)
                feature.write_bytes(member.member_id.encode("utf-8"))
                enriched.append(
                    DatasetMember(
                        member.member_id,
                        member.partitions,
                        member.targets,
                        member.metadata,
                        member.display,
                        {**member.assets, "features": self._asset(root, feature)},
                    )
                )
            members = enriched
        if self.stage == "annotation":
            enriched = []
            for member in members:
                directory = root / "annotations" / member.member_id
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "annotation.json").write_text(
                    '{"quality":"ok"}\n', encoding="utf-8"
                )
                enriched.append(
                    DatasetMember(
                        member.member_id,
                        member.partitions,
                        member.targets,
                        {**member.metadata, "annotation_version": self.marker},
                        member.display,
                        {
                            **member.assets,
                            "annotation": self._asset(root, directory, kind="directory"),
                        },
                    )
                )
            members = enriched
            (root / "vocabulary.json").write_text('{"quality":["ok"]}\n', encoding="utf-8")
        DatasetIndex.write(index_path, members)
        return TaskOutput(
            outputs={"stage": self.stage, "member_count": len(members)},
            artifacts=[ArtifactDeclaration("dataset", name="dataset")],
        )

    @staticmethod
    def _asset(root: Path, path: Path, *, kind: str = "file") -> DatasetAsset:
        digest, size = TaskArtifact.fingerprint_path(path)
        return DatasetAsset(
            path.relative_to(root).as_posix(), kind, f"sha256:{digest}", size
        )

