"""Reviewed effect/structured-port metadata layered over native AST signatures."""


def enrich(operation):
    name, kind = operation["label"], operation["kind"]
    for parameter in operation["parameters"]:
        if parameter["name"] == "credential":
            parameter.update(secret_reference=True, formats=["env:NAME", "keyring:IDENTIFIER"])
        if parameter["name"] == "csv" and name in {"ReadRecords", "WriteRecords"}:
            parameter["schema_resource"] = "csv-options-v1.json"
        if parameter["name"] == "schema" and name in {
            "ReadRecords",
            "ReadSQL",
            "WriteSQL",
            "FromRows",
            "CastTable",
        }:
            parameter["schema_resource"] = "table-schema-v1.json"
    effects = []
    preview = {"supported": False, "reason": "No native bounded preview entrypoint."}
    if kind == "etl":
        operation["state_policy"] = (
            "Fit once on identified train rows per fold/trial; transform never refits."
            if name == "TrainStandardizer"
            else "Configuration belongs to the instance; cursors/files/budgets belong to the call. "
            "No model weights or implicit training state."
        )
        effects = {
            "ReadRecords": ["read-data"],
            "ListFiles": ["read-metadata"],
            "PreviewArray": ["read-data"],
            "ReadSQL": ["read-data", "network-conditional"],
            "WriteSQL": ["modify-external-source", "network-conditional", "ddl-conditional"],
            "WriteRecords": ["write-staging"],
            "WriteSampleAssets": ["write-staging"],
            "Download": ["network", "write-cache"],
            "TrainStandardizer": ["learn-state"],
        }.get(name, [])
        if name in {"ReadRecords", "ReadSQL", "PreviewArray"}:
            preview = {
                "supported": True,
                "entrypoint": "__call__" if name == "PreviewArray" else "preview",
                "scope": "prefix",
                "budget": "preview-budget-v1",
                "side_effects": "Read-only native entrypoint; not an arbitrary-code sandbox.",
            }
    else:
        effects = ["instance-state"]
        if any(m["name"] == "fit" for m in operation["methods"]):
            effects.append("learn-state")
    operation.update(
        effects={
            "schema_version": "operation-effects-v1",
            "capabilities": effects,
            "authorization": "operator-required"
            if any(
                v in effects for v in ("network", "network-conditional", "modify-external-source")
            )
            else "scoped-inputs",
        },
        preview=preview,
    )
    for method in operation["methods"]:
        method["effects"] = (
            ["learn-state"]
            if method["name"] == "fit"
            else [effect for effect in effects if effect != "learn-state"]
        )
        if name in {
            "ReadRecords",
            "ReadSQL",
            "FromRows",
            "CastTable",
            "FilterTable",
            "SortTable",
            "AggregateTable",
            "ConcatTables",
            "DeduplicateTable",
            "FillNulls",
            "SelectTable",
        }:
            method["outputs"] = [
                {
                    "name": "result",
                    "kind": "table-preview"
                    if method["name"] == "preview"
                    else "row-iterator"
                    if method["name"] == "iter_records"
                    else "value",
                    "schema_parameter": "schema",
                    "nullable": "column-declared",
                }
            ]
    modules = {
        "EnsembleModel": {"models": "module-sequence"},
        "MultiTaskModel": {"backbone": "module", "heads": "named-module-mapping"},
        "MixtureOfExperts": {"experts": "module-sequence", "gate": "module"},
        "NeuralODE": {"dynamics": "module"},
    }
    for parameter, port_kind in modules.get(name, {}).items():
        operation["configuration_ports"].append(
            {
                "name": parameter,
                "kind": port_kind,
                "construction": "ordinary Python modules, not recursive YAML objects",
                "sharing": "Supplied instances are retained; reuse shares parameters/buffers.",
            }
        )
    for method in operation["methods"]:
        if name == "EnsembleModel":
            method["variadic_inputs"] = (
                "All members receive the same ordinary Python *args/**kwargs."
            )
            method["constraints"] = (
                "Member tensor shapes must match for reductions; stack/concat obey explicit axes."
            )
            method["outputs"] = [
                {
                    "name": "members" if method["name"] == "forward_members" else "output",
                    "kind": "tensor-sequence" if method["name"] == "forward_members" else "tensor",
                }
            ]
        elif name == "MultiTaskModel" and method["name"] == "forward":
            method["outputs"] = [
                {
                    "name": "tasks",
                    "kind": "named-tensor-mapping",
                    "keys_from": "heads",
                    "additional_key_if": "return_features: features_key",
                }
            ]
            method["constraints"] = (
                "Heads may have unrelated shapes; detached tasks do not update shared backbone."
            )
        elif name == "MixtureOfExperts":
            method["constraints"] = (
                "Gate logits [...,num_experts]; experts share leading shape/output tails. "
                "Top-k still evaluates ALL experts."
            )
            if method["name"] == "forward_experts":
                method["outputs"] = [{"name": "experts", "kind": "tensor-sequence"}]
        elif name == "NeuralODE" and method["name"] == "forward":
            method["inputs"] = [
                {"name": "initial_state", "kind": "tensor", "shape": ["..."], "dtype": "floating"},
                {
                    "name": "times",
                    "kind": "tensor",
                    "shape": ["T"],
                    "constraints": "Finite, strictly increasing; T>=2.",
                },
            ]
            method["outputs"] = [
                {
                    "name": "states",
                    "kind": "tensor",
                    "shape_rule": "return_trajectory: insert T at time_dimension; "
                    "otherwise initial_state.shape",
                }
            ]
            method["constraints"] = (
                "dynamics(t,y) returns y.shape; fixed-step Euler/midpoint/RK4, no adaptive solver."
            )
        elif name == "ConformalPredictionInterval":
            if method["name"] == "forward":
                method["outputs"] = [
                    {
                        "name": "interval",
                        "kind": "named-tensor-mapping",
                        "keys": ["prediction", "lower", "upper"],
                        "shape_rule": "predictions.shape",
                    }
                ]
            operation["state_policy"] = (
                "fit replaces residual quantile buffer using held-out calibration data; "
                "disjoint from training/test and per-fold. Requires calibrated state."
            )
        elif name == "TemperatureScaler":
            operation["state_policy"] = (
                "fit optimizes scalar temperature on held-out logits, disjoint from "
                "training/test and per-fold; shared instance shares learned state."
            )
            if method["name"] == "forward":
                method["outputs"] = [
                    {"name": "logits", "kind": "tensor", "shape_rule": "logits.shape"}
                ]
    callable_methods = [m for m in operation["methods"] if m["name"] in {"forward", "__call__"}]
    operation["composition"] = {
        "status": "abstract"
        if operation.get("abstract")
        else "declared-structure"
        if callable_methods and all(m.get("outputs") for m in callable_methods)
        else "signature-only",
        "scope": "Declared ports are not a general symbolic shape/symmetry proof.",
        "unresolved_inputs": sorted(
            {
                p["name"]
                for m in callable_methods
                for p in m["inputs"]
                if not p.get("shape") and p.get("kind") == "tensor"
            }
        ),
    }
