"""Normalize concise authoring YAML into existing strict runner schemas."""

from __future__ import annotations

import copy
import itertools
import re
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
        if "run" in data or "steps" in data or "parallel" in data:
            data = self._simple_work(data)
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
        if "run" in values:
            return (
                ConfigurationKind.WORKFLOW
                if "seeds" in values or "search" in values
                else ConfigurationKind.TASK
            )
        if "steps" in values or "parallel" in values:
            return ConfigurationKind.WORKFLOW
        # A training experiment also owns a top-level ``task`` object.  Its
        # explicit experiment block therefore has priority over the concise
        # generic-task shorthand.
        if "experiment" in values:
            return ConfigurationKind.EXPERIMENT
        if "task" in values or "preprocess" in values:
            return ConfigurationKind.TASK
        return ConfigurationKind.EXPERIMENT

    def _simple_work(self, values: dict[str, Any]) -> dict[str, Any]:
        """Compile function-first YAML to strict Task/Workflow IR."""
        if "steps" in values or ("parallel" in values and "run" not in values):
            return self._simple_steps(values)
        if "run" not in values:
            raise ValueError("Simple work requires run, steps or parallel.")
        seeds = values.pop("seeds", None)
        search = values.pop("search", None)
        maximum = values.pop("max_parallel", None)
        objective = values.pop("objective", None)
        trials = values.pop("trials", None)
        if seeds is None and search is None:
            if objective is not None:
                values["objective"] = objective
            return self._simple_task(values)
        seed_values = self._seed_values(seeds)
        variants = self._search_variants(search, trials=trials)
        base = copy.deepcopy(values)
        nodes: dict[str, Any] = {}
        index = 0
        for variant in variants:
            for seed in seed_values:
                index += 1
                node = copy.deepcopy(base)
                parameters = dict(node.get("with", {}))
                parameters.update(variant)
                node["with"] = parameters
                if seed is not None:
                    node["seed"] = seed
                nodes[f"run-{index:03d}"] = {"config": self._simple_task(node)}
        return {
            "kind": "workflow",
            "schema_version": "1.0",
            "name": str(base.get("name", "work")),
            "nodes": nodes,
            "max_parallel": int(maximum if maximum is not None else min(len(nodes), 4)),
            "metadata": {
                "authoring": "simple-work",
                "seeds": [value for value in seed_values if value is not None],
                "search_variants": len(variants),
                **({"objective": self._objective(objective)} if objective is not None else {}),
            },
            **({"output_root": str(base["output_root"])} if "output_root" in base else {}),
        }

    def _simple_steps(self, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "name",
            "steps",
            "parallel",
            "resources",
            "max_parallel",
            "output_root",
            "metadata",
        }
        unexpected = set(values) - allowed
        if unexpected:
            raise ValueError(f"Unknown simple workflow keys: {sorted(unexpected)}.")
        raw_steps: Any = values.get("steps")
        if raw_steps is None:
            raw_steps = [{"parallel": values.get("parallel")}]
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes, bytearray)):
            raise TypeError("steps must be a list.")
        nodes: dict[str, Any] = {}
        default_resources = values.get("resources", {})
        if not isinstance(default_resources, Mapping):
            raise TypeError("Workflow resources must be a mapping.")
        previous: tuple[str, ...] = ()
        largest_level = 1
        for position, entry in enumerate(raw_steps, 1):
            if not isinstance(entry, Mapping):
                raise TypeError(f"steps[{position - 1}] must be a mapping.")
            group = entry.get("parallel")
            entries: Sequence[Any]
            if group is not None:
                if not isinstance(group, Sequence) or isinstance(group, (str, bytes, bytearray)):
                    raise TypeError(f"steps[{position - 1}].parallel must be a list.")
                entries = group
            else:
                entries = (entry,)
            current: list[str] = []
            largest_level = max(largest_level, len(entries))
            for offset, step in enumerate(entries, 1):
                if not isinstance(step, Mapping):
                    raise TypeError("Each workflow step must be a mapping.")
                raw_name = step.get("name") or str(step.get("run", "step")).rsplit(".", 1)[-1]
                name = self._unique_node_name(str(raw_name), position, offset, nodes)
                node = dict(step)
                node.pop("name", None)
                step_resources = node.get("resources", {})
                if not isinstance(step_resources, Mapping):
                    raise TypeError(f"Resources for step {name!r} must be a mapping.")
                inherited_resources = {
                    **copy.deepcopy(dict(default_resources)),
                    **copy.deepcopy(dict(step_resources)),
                }
                if inherited_resources:
                    node["resources"] = inherited_resources
                nodes[name] = {
                    "config": self._simple_task(node),
                    **({"needs": list(previous)} if previous else {}),
                }
                current.append(name)
            previous = tuple(current)
        maximum = values.get("max_parallel", largest_level)
        return {
            "kind": "workflow",
            "schema_version": "1.0",
            "name": str(values.get("name", "workflow")),
            "nodes": nodes,
            "max_parallel": int(maximum),
            **({"output_root": str(values["output_root"])} if "output_root" in values else {}),
            **({"metadata": copy.deepcopy(values["metadata"])} if "metadata" in values else {}),
        }

    def _simple_task(self, values: Mapping[str, Any]) -> dict[str, Any]:
        data = copy.deepcopy(dict(values))
        allowed = {
            "name",
            "run",
            "with",
            "resources",
            "output_root",
            "metadata",
            "resume",
            "rerun_completed",
            "code_version",
            "data_catalog",
            "seed",
            "objective",
        }
        unexpected = set(data) - allowed
        if unexpected:
            raise ValueError(f"Unknown simple run keys: {sorted(unexpected)}.")
        target = data.get("run")
        class_spec: dict[str, Any] | None = None
        if isinstance(target, Mapping):
            class_spec = copy.deepcopy(dict(target))
            unexpected_run = set(class_spec) - {"class", "target", "init", "method", "with"}
            if unexpected_run:
                raise ValueError(f"Unknown run class keys: {sorted(unexpected_run)}.")
            if "class" in class_spec and "target" in class_spec:
                raise ValueError("Use run.class or run.target, not both.")
            class_path = class_spec.get("class", class_spec.get("target"))
            if not isinstance(class_path, str) or not class_path.strip():
                raise ValueError("run.class must be a non-empty dotted class path.")
            if not isinstance(class_spec.get("init", {}), Mapping):
                raise TypeError("run.init must be a mapping.")
        elif not isinstance(target, str) or not target.strip():
            raise ValueError("run must be a dotted Python callable path or class mapping.")
        if class_spec is not None and "with" in data and "with" in class_spec:
            raise ValueError("Declare with either beside run or inside the advanced class form.")
        parameters = data.get("with", class_spec.get("with", {}) if class_spec else {})
        if not isinstance(parameters, Mapping):
            raise TypeError("with must map Python parameter names to values.")
        inputs: list[dict[str, Any]] = []
        resolved_parameters = self._parameter_inputs(parameters, inputs, marker=False)
        seed = data.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer.")
        task_params: dict[str, Any] = {
            "parameters": resolved_parameters,
        }
        if class_spec is None:
            task_params["callable_path"] = target
        else:
            task_params.update(
                {
                    "class_path": class_spec.get("class", class_spec.get("target")),
                    "init_parameters": self._parameter_inputs(
                        class_spec.get("init", {}), inputs, marker=False
                    ),
                    "method": str(class_spec.get("method", "run")),
                }
            )
        if seed is not None:
            task_params["seed"] = seed
        extensions: dict[str, Any] = {
            "authoring": {
                "simple_work": True,
                **({"resources": copy.deepcopy(data["resources"])} if "resources" in data else {}),
                **{key: data[key] for key in ("code_version", "data_catalog") if key in data},
                **(
                    {"objective": self._objective(data["objective"])} if "objective" in data else {}
                ),
            }
        }
        return {
            "kind": "task",
            "schema_version": "1.0",
            "name": str(
                data.get(
                    "name",
                    str(
                        class_spec.get("class", class_spec.get("target"))
                        if class_spec is not None
                        else target
                    ).rsplit(".", 1)[-1],
                )
            ),
            "inputs": inputs,
            "task": {
                "target": "lambdaforge.runtime.CallableTask",
                "params": task_params,
            },
            "extensions": extensions,
            **({"output_root": str(data["output_root"])} if "output_root" in data else {}),
            **({"metadata": copy.deepcopy(data["metadata"])} if "metadata" in data else {}),
            **({"resume": bool(data["resume"])} if "resume" in data else {}),
            **(
                {"rerun_completed": bool(data["rerun_completed"])}
                if "rerun_completed" in data
                else {}
            ),
        }

    def _parameter_inputs(
        self, value: Any, inputs: list[dict[str, Any]], *, marker: bool = True
    ) -> Any:
        if isinstance(value, Mapping):
            if marker and set(value) == {"file"}:
                name = f"argument_{len(inputs)}"
                inputs.append({"name": name, "path": str(value["file"])})
                return {"__lambdaforge_input__": name}
            if marker and set(value) == {"dataset"}:
                name = f"argument_{len(inputs)}"
                selector = str(value["dataset"])
                inputs.append(
                    {
                        "name": name,
                        "dataset": selector
                        if selector.startswith("dataset:")
                        else f"dataset:{selector}",
                    }
                )
                return {"__lambdaforge_input__": name}
            return {str(key): self._parameter_inputs(item, inputs) for key, item in value.items()}
        if isinstance(value, list):
            return [self._parameter_inputs(item, inputs) for item in value]
        return value

    @staticmethod
    def _seed_values(value: Any) -> tuple[int | None, ...]:
        if value is None:
            return (None,)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError("seeds must be a list of integers.")
        seeds = tuple(value)
        if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise TypeError("seeds must be a non-empty list of integers.")
        if len(seeds) != len(set(seeds)):
            raise ValueError("seeds cannot contain duplicates.")
        return seeds

    @staticmethod
    def _search_variants(value: Any, *, trials: Any = None) -> tuple[dict[str, Any], ...]:
        if value is None:
            if trials is not None:
                raise ValueError("trials requires a search space.")
            return ({},)
        if not isinstance(value, Mapping) or not value:
            raise TypeError("search must map parameter names to finite values.")
        names: list[str] = []
        dimensions: list[Sequence[Any]] = []
        random_space: dict[str, dict[str, Any]] = {}
        has_range = False
        for name, descriptor in value.items():
            if "." in str(name):
                raise ValueError("Simple callable search parameter names cannot contain dots.")
            if isinstance(descriptor, Mapping) and "range" in descriptor:
                bounds = descriptor["range"]
                if (
                    not isinstance(bounds, Sequence)
                    or isinstance(bounds, (str, bytes, bytearray))
                    or len(bounds) != 2
                ):
                    raise TypeError(f"search.{name}.range must contain [low, high].")
                kind = str(descriptor.get("type", "uniform"))
                if kind == "float":
                    kind = "uniform"
                if descriptor.get("scale") == "log":
                    kind = "loguniform"
                random_space[str(name)] = {"type": kind, "low": bounds[0], "high": bounds[1]}
                has_range = True
                continue
            values = descriptor.get("values") if isinstance(descriptor, Mapping) else descriptor
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes, bytearray))
                or not values
            ):
                raise TypeError(f"search.{name} must define values or range.")
            names.append(str(name))
            dimensions.append(tuple(values))
            random_space[str(name)] = {"type": "choice", "values": list(values)}
        if has_range or trials is not None:
            count = 20 if trials is None else int(trials)
            from lambdaforge.hpo.RandomSearch import RandomSearch

            return tuple(
                dict(trial.parameters) for trial in RandomSearch(random_space).trials(count)
            )
        return tuple(
            dict(zip(names, combination, strict=True))
            for combination in itertools.product(*dimensions)
        )

    @staticmethod
    def _objective(value: Any) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise TypeError("objective must contain metric and mode.")
        metric = str(value.get("metric", "")).strip()
        mode = str(value.get("mode", "")).strip().lower()
        if not metric or mode not in {"min", "max"}:
            raise ValueError("objective requires a metric and mode: min or max.")
        return {"metric": metric, "mode": mode}

    @staticmethod
    def _unique_node_name(raw: str, position: int, offset: int, existing: Mapping[str, Any]) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.") or "step"
        candidate = slug
        if candidate in existing:
            candidate = f"{slug}-{position}-{offset}"
        return candidate

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
        optimizer = values.get("optimizer")
        if isinstance(optimizer, Mapping) and "type" in optimizer:
            concise = copy.deepcopy(dict(optimizer))
            selected = str(concise.pop("type")).strip()
            if not selected:
                raise ValueError("optimizer.type cannot be empty.")
            if "ref" in concise or "target" in concise:
                raise ValueError("Use optimizer.type or optimizer.ref, not both.")
            aliases = {
                "adam": "torch.optim.Adam",
                "adamw": "torch.optim.AdamW",
                "sgd": "torch.optim.SGD",
            }
            params = concise.pop("params", {})
            if not isinstance(params, Mapping):
                raise TypeError("optimizer.params must be a mapping.")
            overlap = set(params) & set(concise)
            if overlap:
                raise ValueError(
                    f"Optimizer parameters are declared twice: {tuple(sorted(overlap))}."
                )
            values["optimizer"] = {
                "ref": aliases.get(selected.lower(), selected),
                "params": {**copy.deepcopy(dict(params)), **concise},
            }
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
