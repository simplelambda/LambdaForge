"""Normalize concise authoring YAML into existing strict runner schemas."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.configuration.MaterializedConfig import MaterializedConfig


class AuthoringConfigNormalizer:
    """Compile friendly authoring forms to one strict, backward-compatible IR."""

    VERSION = "1.0"
    _TARGET_KEYS = frozenset({"model", "task", "train", "val", "test", "predict", "datamodule"})
    _TARGET_LIST_KEYS = frozenset({"losses", "train_metrics", "val_metrics", "test_metrics"})

    def normalize(
        self, values: Mapping[str, Any], *, source: str | Path | None = None
    ) -> MaterializedConfig:
        """Detect a document family and return its strict materialization."""
        if not isinstance(values, Mapping):
            raise TypeError("LambdaForge authoring configuration must be a mapping.")
        data = copy.deepcopy(dict(values))
        declared_authoring = data.pop("authoring_version", self.VERSION)
        if str(declared_authoring) != self.VERSION:
            raise ValueError(
                f"Unsupported authoring_version {declared_authoring!r}; "
                f"current is {self.VERSION!r}."
            )
        kind = self.detect(data)
        if kind is ConfigurationKind.DATASET:
            data.setdefault("kind", "dataset")
            data.setdefault("schema_version", "1.0")
        elif kind is ConfigurationKind.TASK:
            data = self._task(data)
        elif kind is ConfigurationKind.WORKFLOW:
            data.setdefault("kind", "workflow")
            data.setdefault("schema_version", "1.0")
        else:
            if data.get("kind") == "experiment":
                data.pop("kind")
            data.setdefault("schema_version", "1.1")
            data = self._experiment(data)
            data = self._object_shorthand(data)
        return MaterializedConfig(
            kind=kind,
            values=data,
            source=Path(source).resolve() if source is not None else None,
            authoring_version=self.VERSION,
        )

    def detect(self, values: Mapping[str, Any]) -> ConfigurationKind:
        """Infer the family only from unambiguous structural keys."""
        explicit = values.get("kind")
        if explicit is not None:
            try:
                return ConfigurationKind(str(explicit))
            except ValueError as error:
                raise ValueError(
                    f"Unknown LambdaForge configuration kind: {explicit!r}."
                ) from error
        if "nodes" in values:
            return ConfigurationKind.WORKFLOW
        # A training experiment also owns a top-level ``task`` object.  Its
        # explicit experiment block therefore has priority over the concise
        # generic-task shorthand.
        if "experiment" in values:
            return ConfigurationKind.EXPERIMENT
        if "task" in values or "preprocess" in values:
            return ConfigurationKind.TASK
        return ConfigurationKind.EXPERIMENT

    def _task(self, values: dict[str, Any]) -> dict[str, Any]:
        values.setdefault("kind", "task")
        values.setdefault("schema_version", "1.0")
        values["inputs"] = self._inputs(values.get("inputs", ()))
        authoring = dict(self._authoring_extensions(values))
        outputs = values.pop("outputs", None)
        if outputs is not None:
            if not isinstance(outputs, Mapping) or not outputs:
                raise TypeError("Authoring outputs must map logical names to relative paths.")
            authoring["outputs"] = {str(key): str(value) for key, value in outputs.items()}
        resources = values.pop("resources", None)
        if resources is not None:
            if not isinstance(resources, Mapping):
                raise TypeError("Authoring resources must be a mapping.")
            authoring["resources"] = copy.deepcopy(dict(resources))
        for key in ("data_catalog", "code_version"):
            item = values.pop(key, None)
            if item is not None:
                if not isinstance(item, str) or not item.strip():
                    raise TypeError(f"Authoring {key} must be a non-empty string.")
                authoring[key] = item
        preprocess = values.pop("preprocess", None)
        if preprocess is not None:
            values["task"] = self._preprocessing(preprocess, values, authoring)
            configured_outputs = authoring.get("outputs", {"processed": "processed"})
            preprocessing_params = values["task"].get("params", {})
            publishes_dataset = bool(
                preprocessing_params.get("publish_dataset", False)
                or preprocessing_params.get("dataset_name")
            )
            values.setdefault(
                "required_artifacts",
                [
                    *dict.fromkeys(str(path) for path in configured_outputs.values()),
                    "preprocessing-manifest.json",
                    *(["dataset-artifact.json"] if publishes_dataset else []),
                ],
            )
        elif "task" in values:
            values["task"] = self._spec(values["task"])
        extensions = dict(values.get("extensions", {}))
        if authoring:
            extensions["authoring"] = authoring
        if extensions:
            values["extensions"] = extensions
        return self._object_shorthand(values)

    def _experiment(self, values: dict[str, Any]) -> dict[str, Any]:
        """Expand beginner training aliases without changing the strict runner."""
        name = values.pop("name", None)
        if name is not None:
            experiment = dict(values.get("experiment", {}))
            experiment.setdefault("name", str(name))
            values["experiment"] = experiment
        values.setdefault("data", {})
        singular_loss = values.pop("loss", None)
        if singular_loss is not None:
            if "losses" in values:
                raise ValueError("Use either loss or losses, not both.")
            aliases = {
                "torch.nn.CrossEntropyLoss": "lambdaforge.nn.losses.CrossEntropyLoss",
                "torch.nn.BCEWithLogitsLoss": (
                    "lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss"
                ),
                "torch.nn.MSELoss": "lambdaforge.nn.losses.MeanSquaredErrorLoss",
                "torch.nn.L1Loss": "lambdaforge.nn.losses.MeanAbsoluteErrorLoss",
                "torch.nn.HuberLoss": "lambdaforge.nn.losses.HuberLoss",
                "torch.nn.SmoothL1Loss": "lambdaforge.nn.losses.SmoothL1Loss",
            }
            values["losses"] = [aliases.get(singular_loss, singular_loss)]
        trainer = values.get("trainer")
        if isinstance(trainer, Mapping) and "epochs" in trainer:
            trainer = copy.deepcopy(dict(trainer))
            if "max_epochs" in trainer:
                raise ValueError("Use either trainer.epochs or trainer.max_epochs, not both.")
            trainer["max_epochs"] = trainer.pop("epochs")
            values["trainer"] = trainer
        extensions = dict(values.get("extensions", {}))
        authoring = dict(self._authoring_extensions(values))
        for key in ("resources", "data_catalog", "environment"):
            item = values.pop(key, None)
            if item is not None:
                authoring[key] = copy.deepcopy(item)
        code_version = values.pop("code_version", None)
        if code_version is not None:
            extensions["code_version"] = str(code_version)
        if authoring:
            extensions["authoring"] = authoring
        if extensions:
            values["extensions"] = extensions
        return values

    @staticmethod
    def _inputs(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            output: list[dict[str, Any]] = []
            for name, descriptor in value.items():
                if isinstance(descriptor, str):
                    key = "dataset" if descriptor.startswith("dataset:") else "path"
                    output.append({"name": str(name), key: descriptor})
                elif isinstance(descriptor, Mapping):
                    output.append({"name": str(name), **copy.deepcopy(dict(descriptor))})
                else:
                    raise TypeError(
                        "Each authoring input must be a path, dataset reference or mapping."
                    )
            return output
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [copy.deepcopy(dict(item)) for item in value]
        raise TypeError("Task inputs must be a mapping or a list of descriptors.")

    def _preprocessing(
        self, value: Any, root: Mapping[str, Any], authoring: dict[str, Any]
    ) -> dict[str, Any]:
        if isinstance(value, str):
            data: dict[str, Any] = {"function": value}
        elif isinstance(value, Mapping):
            data = copy.deepcopy(dict(value))
        else:
            raise TypeError("preprocess must be an import path or mapping.")
        function = data.pop("function", None)
        if not isinstance(function, str) or not function.strip():
            raise ValueError("preprocess.function must be an importable callable path.")
        inputs = root.get("inputs", ())
        if not isinstance(inputs, Sequence) or not inputs:
            raise ValueError("Concise preprocessing requires at least one named input.")
        input_name = str(data.pop("input", inputs[0]["name"]))
        output_name = str(data.pop("output", "processed"))
        output_paths = authoring.get("outputs", {"processed": "processed"})
        if not isinstance(output_paths, Mapping) or output_name not in output_paths:
            raise ValueError(f"Unknown preprocessing output name: {output_name!r}.")
        authoring.setdefault("outputs", dict(output_paths))
        source_format = str(data.pop("format", "jsonl"))
        if source_format == "jsonl":
            source_target = "lambdaforge.preprocessing.JsonLinesSource"
            source_params: dict[str, Any] = {"input_name": input_name}
            key_field = data.pop("key_field", None)
            if key_field is not None:
                source_params["key_field"] = key_field
        elif source_format == "files":
            source_target = "lambdaforge.preprocessing.FileTreeSource"
            source_params = {"input_name": input_name, "pattern": data.pop("pattern", "**/*")}
        else:
            raise ValueError("preprocess.format must be 'jsonl' or 'files'.")
        params: dict[str, Any] = {
            "source": {"target": source_target, "params": source_params},
            "transforms": [
                {
                    "target": "lambdaforge.preprocessing.CallableTransform",
                    "params": {"function": {"ref": function}},
                }
            ],
            "sink": {
                "target": "lambdaforge.preprocessing.JsonDirectorySink",
                "params": {"output_name": output_name},
            },
        }
        allowed = (
            "workers",
            "workload",
            "on_error",
            "checkpoint_interval",
            "publish_dataset",
            "dataset_name",
            "dataset_version",
            "dataset_splits",
            "dataset_source",
            "dataset_metadata",
        )
        for key in allowed:
            if key in data:
                params[key] = data.pop(key)
        if data:
            raise ValueError(f"Unknown concise preprocessing keys: {sorted(data)}.")
        return {"target": "lambdaforge.preprocessing.PreprocessingTask", "params": params}

    @staticmethod
    def _authoring_extensions(values: Mapping[str, Any]) -> Mapping[str, Any]:
        extensions = values.get("extensions", {})
        if not isinstance(extensions, Mapping):
            return {}
        authoring = extensions.get("authoring", {})
        return authoring if isinstance(authoring, Mapping) else {}

    def _object_shorthand(self, value: Any, *, key: str | None = None) -> Any:
        if (
            isinstance(value, str)
            and key in {"train", "val", "test"}
            and value.startswith("dataset:")
        ):
            return value
        if isinstance(value, str) and key in self._TARGET_KEYS:
            return {"target": value}
        if isinstance(value, Mapping):
            return {
                str(item_key): self._object_shorthand(item, key=str(item_key))
                for item_key, item in value.items()
            }
        if isinstance(value, list):
            if key in self._TARGET_LIST_KEYS:
                return [
                    self._spec(item) if isinstance(item, str) else self._object_shorthand(item)
                    for item in value
                ]
            return [self._object_shorthand(item) for item in value]
        return value

    @staticmethod
    def _spec(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            return {"target": value}
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
        raise TypeError("An object specification must be an import path or mapping.")
