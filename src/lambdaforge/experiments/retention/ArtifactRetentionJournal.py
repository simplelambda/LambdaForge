"""Atomic recovery journal for artifact-retention transactions."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.experiments.retention.ArtifactRetentionArchiveRecord import (
    ArtifactRetentionArchiveRecord,
)
from lambdaforge.experiments.retention.ArtifactRetentionPhase import ArtifactRetentionPhase
from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan


class ArtifactRetentionJournal(JsonResult):
    """Persist enough transaction state to roll back or finish after a crash."""

    VERSION = 1
    FIELD_NAMES = frozenset(
        {
            "retention_journal_version",
            "plan",
            "phase",
            "archives",
        }
    )

    def __init__(
        self,
        *,
        plan: ArtifactRetentionPlan,
        phase: ArtifactRetentionPhase | str,
        archives: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if not isinstance(plan, ArtifactRetentionPlan):
            raise TypeError("Retention journal plan must be an ArtifactRetentionPlan.")
        if not plan.is_ready:
            raise ValueError("Retention journal requires a ready preview plan.")
        self.plan = plan
        if not isinstance(phase, (ArtifactRetentionPhase, str)):
            raise TypeError("Retention journal phase must be a string or typed phase.")
        self.phase = (
            phase if isinstance(phase, ArtifactRetentionPhase) else ArtifactRetentionPhase(phase)
        )
        self.archives = ArtifactRetentionArchiveRecord.validated_many(
            archives,
            plan_id=plan.plan_id,
            operations=plan.operations,
            archive_name=plan.archive_name,
        )
        payload = FrozenJsonMapping(
            {
                "retention_journal_version": self.VERSION,
                "plan": plan.to_dict(),
                "phase": self.phase.value,
                "archives": list(self.archives),
            }
        )
        self._freeze_mapping(dict(payload))

    def with_phase(
        self,
        phase: ArtifactRetentionPhase,
        *,
        archives: Sequence[Mapping[str, Any]] | None = None,
    ) -> ArtifactRetentionJournal:
        """Return the next immutable journal snapshot."""
        return ArtifactRetentionJournal(
            plan=self.plan,
            phase=phase,
            archives=self.archives if archives is None else archives,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> ArtifactRetentionJournal:
        """Load a journal and reconstruct its typed plan."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError("Retention journal JSON must contain an object.")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactRetentionJournal:
        """Parse an exact durable transaction journal without filtering entries."""
        if not isinstance(value, Mapping):
            raise TypeError("Retention journal must be an object.")
        cls._validate_fields(value)
        version = value["retention_journal_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != cls.VERSION:
            raise ValueError(
                f"Unsupported retention journal version {version!r}; expected {cls.VERSION}."
            )
        plan = value["plan"]
        if not isinstance(plan, Mapping):
            raise TypeError("Retention journal has no valid plan.")
        archives = value["archives"]
        if isinstance(archives, (str, bytes, bytearray)) or not isinstance(
            archives,
            Sequence,
        ):
            raise TypeError("Retention journal archives must be a sequence.")
        for index, archive in enumerate(archives):
            if not isinstance(archive, Mapping):
                raise TypeError(f"Retention journal archive {index} must be an object.")
        return cls(
            plan=ArtifactRetentionPlan.from_mapping(plan),
            phase=value["phase"],
            archives=archives,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an independent ordinary JSON mapping."""
        return copy.deepcopy(dict(self))

    @classmethod
    def _validate_fields(cls, value: Mapping[str, Any]) -> None:
        keys = set(value)
        if keys == cls.FIELD_NAMES:
            return
        missing = sorted(cls.FIELD_NAMES - keys)
        unknown = sorted(str(key) for key in keys - cls.FIELD_NAMES)
        raise ValueError(
            f"Retention journal fields are malformed: missing={missing}, unknown={unknown}."
        )
