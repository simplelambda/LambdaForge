"""Installed public model declarations, without importing Torch or consumer implementations."""

from __future__ import annotations

import ast
from importlib.resources import files
from typing import Any


def model_contract(name: str) -> dict[str, Any]:
    """Resolve an exact public export and its declared signatures/docstrings statically.

    Shapes are the installed component's documentation, not a newly invented universal tensor
    contract. Inherited methods are explicitly absent rather than inferred or imported.
    """
    if not name.isidentifier() or len(name) > 100:
        raise ValueError("Use an exact public model name from list_model_catalog.")
    module = "lambdaforge.nn.models"
    symbol = name
    visited = set()
    for _ in range(8):
        if (module, symbol) in visited or not module.startswith("lambdaforge.nn.models"):
            raise ValueError(
                "Installed model export is cyclic or outside the public model package."
            )
        visited.add((module, symbol))
        relative = module.removeprefix("lambdaforge.").replace(".", "/")
        base = files("lambdaforge").joinpath(relative)
        resource = (
            base.joinpath("__init__.py")
            if base.is_dir()
            else files("lambdaforge").joinpath(relative + ".py")
        )
        source = resource.read_text(encoding="utf-8")
        if len(source.encode("utf-8")) > 256 * 1024:
            raise ValueError("Installed component declaration exceeds its bounded metadata limit.")
        tree = ast.parse(source)
        if module == "lambdaforge.nn.models":
            exports: list[str] = next(
                (
                    ast.literal_eval(node.value)
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
                ),
                [],
            )
            if name not in exports:
                raise ValueError("Model is not an installed public export.")
        declaration = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == symbol),
            None,
        )
        if declaration is not None:
            signatures = []
            for node in declaration.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                    "__init__",
                    "forward",
                }:
                    signatures.append(
                        {
                            "method": node.name,
                            "arguments": ast.unparse(node.args),
                            "returns": ast.unparse(node.returns) if node.returns else None,
                            "documentation": ast.get_docstring(node),
                            "line": node.lineno,
                        }
                    )
            return {
                "name": name,
                "public_api": "lambdaforge.nn.models." + name,
                "implementation_module": module,
                "documentation": ast.get_docstring(declaration),
                "declared_methods": signatures,
                "source": str(resource),
                "line": declaration.lineno,
                "backend_imported": False,
                "inherited_methods_inferred": False,
                "tensor_contract_source": "installed class/forward documentation; "
                "no inferred shapes",
            }
        import_node = next(
            (
                (node, item)
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                for item in node.names
                if (item.asname or item.name) == symbol
            ),
            None,
        )
        if import_node is None or import_node[0].level or not import_node[0].module:
            raise ValueError("Cannot resolve this installed public declaration statically.")
        module, symbol = import_node[0].module, import_node[1].name
    raise ValueError("Installed model export chain exceeds its bounded depth.")
