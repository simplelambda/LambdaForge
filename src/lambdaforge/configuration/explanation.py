"""Research-oriented explanation of materialized LambdaForge documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.configuration.ConfigurationDescriptor import ConfigurationDescriptor
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.execution.ConfigurationResourceResolver import ConfigurationResourceResolver


def explain_configuration(path: str | Path) -> dict[str, Any]:
    """Explain scientific intent without importing or constructing consumer targets."""
    descriptor = ConfigurationDescriptor.from_path(path)
    values = descriptor.materialized
    payload: dict[str, Any] = {
        "kind": descriptor.kind.value,
        "name": descriptor.name,
        "source": str(descriptor.source),
        "scientific_identity": descriptor.scientific_identity,
        "scientific_revision": descriptor.revision,
        "datasets": list(descriptor.datasets),
        "planned_work": {"total": descriptor.planned_units, "unit": descriptor.unit},
        "resources": ConfigurationResourceResolver.resolve(descriptor.source).to_dict(),
    }
    if descriptor.kind is ConfigurationKind.EXPERIMENT:
        experiment = values.get("experiment", {})
        experiment = experiment if isinstance(experiment, Mapping) else {}
        trainer = values.get("trainer", {})
        trainer = trainer if isinstance(trainer, Mapping) else {}
        payload.update(
            {
                "model": _object(values.get("model")),
                "losses": [_object(item) for item in _sequence(values.get("losses"))],
                "optimizer": _object(values.get("optimizer")),
                "training": {
                    "max_epochs": trainer.get("max_epochs"),
                    "accelerator": trainer.get("accelerator"),
                    "devices": trainer.get("devices"),
                },
                "seeds": list(_sequence(experiment.get("seeds", (experiment.get("seed"),)))),
                "objective": _objective(values),
            }
        )
    elif descriptor.kind is ConfigurationKind.DATASET:
        stages = values.get("stages", {})
        stages = stages if isinstance(stages, Mapping) else {}
        payload["stages"] = [
            {
                "name": str(name),
                "depends_on": list(_sequence(stage.get("depends_on")))
                if isinstance(stage, Mapping)
                else [],
                "required": bool(stage.get("required", True))
                if isinstance(stage, Mapping)
                else True,
            }
            for name, stage in stages.items()
        ]
        payload["publication"] = values.get("publish", {})
    elif descriptor.kind is ConfigurationKind.WORKFLOW:
        nodes = values.get("nodes", {})
        nodes = nodes if isinstance(nodes, Mapping) else {}
        payload["nodes"] = [
            {
                "name": str(name),
                "depends_on": list(_sequence(node.get("depends_on")))
                if isinstance(node, Mapping)
                else [],
            }
            for name, node in nodes.items()
        ]
    else:
        payload["task"] = _object(values.get("task"))
        payload["inputs"] = values.get("inputs", [])
        payload["required_artifacts"] = values.get("required_artifacts", [])
    return payload


def _object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "target": value.get("target", value.get("ref", value.get("plugin"))),
        "params": dict(value.get("params", {}))
        if isinstance(value.get("params", {}), Mapping)
        else {},
    }


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _objective(values: Mapping[str, Any]) -> dict[str, Any] | None:
    hpo = values.get("hpo", {})
    if isinstance(hpo, Mapping) and isinstance(hpo.get("objective"), Mapping):
        return dict(hpo["objective"])
    checkpoint = values.get("checkpoint", {})
    if isinstance(checkpoint, Mapping) and checkpoint.get("monitor"):
        return {
            "metric": checkpoint.get("monitor"),
            "direction": checkpoint.get("mode"),
        }
    return None
