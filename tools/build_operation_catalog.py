"""Build packaged metadata from this working tree without importing scientific modules.

Run after changing public signatures: python tools/build_operation_catalog.py.
The checked-in JSON ships in wheels; discovery never requires a source checkout or AST scan.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from operation_contracts import enrich

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/lambdaforge"


def expression(value):
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError):
        return {"python": ast.unparse(value)}


def parameters(function):
    if function is None:
        return []
    args = [*function.args.posonlyargs, *function.args.args]
    defaults = [None] * (len(args) - len(function.args.defaults)) + function.args.defaults
    args += function.args.kwonlyargs
    defaults += function.args.kw_defaults
    rows = []
    for arg, default in zip(args, defaults, strict=True):
        if arg.arg in {"self", "cls"}:
            continue
        annotation = ast.unparse(arg.annotation) if arg.annotation else "Any"
        row = {
            "name": arg.arg,
            "annotation": annotation,
            "required": default is None,
            "default": expression(default),
            "type": "integer"
            if annotation == "int"
            else "number"
            if annotation == "float"
            else "boolean"
            if annotation == "bool"
            else "string"
            if annotation == "str"
            else "structured",
            "inferable": False,
        }
        if annotation.startswith("Literal[") and isinstance(arg.annotation, ast.Subscript):
            values = arg.annotation.slice
            row["options"] = (
                [expression(v) for v in values.elts]
                if isinstance(values, ast.Tuple)
                else [expression(values)]
            )
        rows.append(row)
    return rows


def port(arg):
    name = arg.arg
    annotation = ast.unparse(arg.annotation) if arg.annotation else "Any"
    semantic = (
        "index"
        if name
        in {"edge_index", "indices", "batch", "ptr", "spatial_indices", "in_degree", "out_degree"}
        else (
            "mask"
            if "mask" in name or name == "valid_nodes"
            else "coordinates"
            if name in {"pos", "positions", "coordinates"}
            else "vector"
            if name in {"vectors", "directions", "edge_vectors"}
            else "tensor"
            if "Tensor" in annotation
            else "value"
        )
    )
    shape = {
        "edge_index": [2, "E"],
        "directions": ["E", 3],
        "edge_vectors": ["E", "V", 3],
        "vectors": ["N", "V", 3],
        "distances": ["E"],
        "batch": ["N"],
    }.get(name)
    return {
        "name": name,
        "kind": semantic,
        "shape": shape,
        "dtype": "int64" if semantic == "index" else "bool" if semantic == "mask" else None,
        "annotation": annotation,
        "shape_status": "declared" if shape else "see-native-docstring",
    }


def build():
    operations = []
    version = next(
        ast.literal_eval(n.value)
        for n in ast.parse((SOURCE / "_version.py").read_text()).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "VERSION" for t in n.targets)
    )
    for directory, kind in (("nn", None), ("etl", "etl"), ("estimators", "estimator")):
        for file in sorted((SOURCE / directory).rglob("*.py")):
            if file.name == "__init__.py":
                continue
            tree = ast.parse(file.read_text())
            module = "lambdaforge." + ".".join(file.relative_to(SOURCE).with_suffix("").parts)
            family = file.relative_to(SOURCE).parts[1] if directory == "nn" else directory
            category = kind or {
                "models": "model",
                "activations": "activation",
                "normalizations": "normalization",
                "losses": "loss",
                "distances": "distance",
                "encodings": "encoding",
                "kernels": "kernel",
                "similarities": "similarity",
            }.get(family, family)
            for cls in (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
            ):
                methods = [f for f in cls.body if isinstance(f, ast.FunctionDef)]
                constructor = next((f for f in methods if f.name == "__init__"), None)
                params = parameters(constructor)
                # These bounds are derived from the actual positive-channel validator calls,
                # not guessed from names such as "width" across unrelated architectures.
                if constructor is not None:
                    positive = {
                        keyword.value.id
                        for node in ast.walk(constructor)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "channels"
                        for keyword in node.keywords
                        if isinstance(keyword.value, ast.Name)
                    }
                    for parameter in params:
                        if parameter["name"] in positive:
                            parameter["minimum"] = 1
                        if category == "normalization" and parameter["name"] == "num_features":
                            parameter.update(inferable=True, dimension_source="receiver")
                if cls.name == "SklearnEstimator":
                    choices = next(
                        n.value
                        for n in cls.body
                        if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "ALGORITHMS" for t in n.targets)
                    )
                    for parameter in params:
                        if parameter["name"] == "algorithm":
                            parameter["options"] = [ast.literal_eval(k) for k in choices.keys]
                calls = []
                for method in methods:
                    if method.name not in {
                        "forward",
                        "forward_with_vectors",
                        "__call__",
                        "fit",
                        "predict",
                        "predict_proba",
                        "transform",
                        "preview",
                        "iter_records",
                        "forward_members",
                        "forward_task",
                        "forward_experts",
                        "shared_features",
                        "load_balance_loss",
                    }:
                        continue
                    inputs = [
                        port(a)
                        for a in [*method.args.args, *method.args.kwonlyargs]
                        if a.arg not in {"self", "cls"}
                    ]
                    method_parameters = {p["name"]: p for p in parameters(method)}
                    for item in inputs:
                        item["required"] = method_parameters[item["name"]]["required"]
                    output = ast.unparse(method.returns) if method.returns else "Any"
                    description = ast.get_docstring(method) or ""
                    calls.append(
                        {
                            "name": method.name,
                            "inputs": inputs,
                            "output_annotation": output,
                            "documentation": description,
                            "output_shape_status": "see-native-docstring",
                        }
                    )
                if not calls:
                    continue
                # Configuration flows are separate from tensor/data ports.
                factories = []
                names = {p["name"] for p in params}
                for factory in ("activation", "normalization", "pooling", "distance"):
                    if factory in names:
                        factories.append(
                            {
                                "name": factory,
                                "kind": "factory:" + factory,
                                "kwargs_parameter": factory + "_kwargs"
                                if factory + "_kwargs" in names
                                else None,
                                "dimension_owner": "receiver"
                                if factory == "normalization"
                                else None,
                            }
                        )
                text = ast.get_docstring(cls) or ""
                operations.append(
                    {
                        "id": module + ":" + cls.name,
                        "kind": category,
                        "label": cls.name,
                        "aliases": [cls.name],
                        "implementation": module + ":" + cls.name,
                        "provider": "lambdaforge",
                        "version": version,
                        "description": {"en": text, "translation_key": "operations." + cls.name},
                        "parameters": params,
                        "constructor_status": "declared"
                        if constructor is not None
                        else "inherited",
                        "methods": calls,
                        "configuration_ports": factories,
                        "instance_policy": "new-per-construction; explicit-reuse-shares-state",
                        "abstract": any(
                            ast.unparse(d).endswith("abstractmethod")
                            for f in methods
                            for d in f.decorator_list
                        ),
                        "state_policy": "Parameters/buffers belong to the constructed instance; "
                        "reuse shares weights and mutable state; train/eval is explicit.",
                        "contract_status": "native-signature",
                        "constraints": "Native constructor/method validation is authoritative.",
                        "symmetry": "Not inferred; consult native class/method documentation.",
                    }
                )
    # Exact equations/symmetries stay owned by the actual native models.
    registry = ast.parse((SOURCE / "nn/ComponentRegistry.py").read_text())
    aliases = {}
    for node in ast.walk(registry):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in {"_activations", "_normalizations"}
            and isinstance(node.value, ast.Dict)
        ):
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                if isinstance(value, ast.Name):
                    aliases.setdefault(value.id, []).append(ast.literal_eval(key))
    for op in operations:
        op["aliases"] = sorted(set([*op["aliases"], *aliases.get(op["label"], [])]))
        for method in op["methods"]:
            if method["name"] == "forward_with_vectors":
                method["outputs"] = [
                    {"name": "scalars", "kind": "tensor", "shape": ["N", "latent_scalar_channels"]},
                    {
                        "name": "vectors",
                        "kind": "vector",
                        "shape": ["N", "latent_vector_channels", 3],
                    },
                ]
            for param in method["inputs"]:
                if op["label"] in {"PaiNN", "GVPGNN"} and param["name"] == "x":
                    param.update(kind="tensor", shape=["N", "in_channels"])
        if op["label"] in {"MLP", "ResidualMLP"}:
            for method in op["methods"]:
                if method["name"] == "forward":
                    method["inputs"] = [
                        {
                            "name": "x",
                            "kind": "tensor",
                            "shape": ["...", "in_features"],
                            "dtype": None,
                        }
                    ]
                    method["outputs"] = [
                        {
                            "name": "output",
                            "kind": "tensor",
                            "shape": ["...", "out_features"],
                            "dtype": None,
                        }
                    ]
        if op["label"] in {"PaiNN", "GVPGNN"}:
            op["symmetry"] = (
                "O(3): invariant scalar output, equivariant polar-vector state; "
                "supplied geometry, no implicit graph-level pooling."
            )
        # Curated native contracts, not shapes guessed from a generic Tensor annotation.
        sparse = {"GraphSAGE", "GCN", "NNConv", "GINE", "PaiNN", "GVPGNN", "SchNet", "DimeNet"}
        for method in op["methods"]:
            if op["label"] in sparse:
                shapes = {
                    "x": ["N", "in_channels"],
                    "edge_index": [2, "E"],
                    "edge_features": ["E", "edge_channels"],
                    "distance_weights": ["E"],
                    "edge_vectors": ["E", "edge_vector_channels", 3],
                    "angles": ["T"],
                }
                for item in method["inputs"]:
                    if item["name"] in shapes:
                        item.update(shape=shapes[item["name"]], shape_status="declared")
                    if op["label"] == "DimeNet" and item["name"] == "triplets":
                        item.update(
                            kind="graph-triplets",
                            constraints="Validated k->j->i native GraphTriplets; k != i.",
                        )
                if method["name"] == "forward":
                    method["outputs"] = [
                        {"name": "output", "kind": "tensor", "shape": ["N", "out_channels"]}
                    ]
                elif method["name"] == "forward_with_vectors":
                    scalar = "hidden_channels" if op["label"] == "PaiNN" else "scalar_channels"
                    vector = "hidden_channels" if op["label"] == "PaiNN" else "vector_channels"
                    method["outputs"][0]["shape"] = ["N", scalar]
                    method["outputs"][1]["shape"] = ["N", vector, 3]
            if op["label"] == "GVP" and method["name"] == "forward":
                for item in method["inputs"]:
                    item.update(
                        shape=["N", "vector_in", 3]
                        if item["name"] == "vectors"
                        else ["N", "scalar_in"],
                        shape_status="declared",
                    )
                method["outputs"] = [
                    {"name": "scalars", "kind": "tensor", "shape": ["N", "scalar_out"]},
                    {"name": "vectors", "kind": "vector", "shape": ["N", "vector_out", 3]},
                ]
            if op["label"] == "StructuralGraphTransformer" and method["name"] == "forward":
                shapes = {
                    "x": ["B", "N", "in_channels"],
                    "spatial_indices": ["B", "N", "N"],
                    "in_degree": ["B", "N"],
                    "out_degree": ["B", "N"],
                    "valid_nodes": ["B", "N"],
                    "edge_bias": ["B", "num_heads", "N", "N"],
                }
                for item in method["inputs"]:
                    item.update(shape=shapes[item["name"]], shape_status="declared")
                method["outputs"] = [
                    {"name": "output", "kind": "tensor", "shape": ["B", "N", "out_channels"]}
                ]
    for operation in operations:
        enrich(operation)
    return {
        "schema_version": "operation-catalog-v1",
        "framework_version": operations[0]["version"],
        "operations": operations,
    }


if __name__ == "__main__":
    target = SOURCE / "catalog/native.json"
    target.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
    print(target)
