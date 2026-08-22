"""Adapter from ordinary Python callables to the established Task contract."""

from __future__ import annotations

import importlib
import inspect
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CallableTask:
    """Invoke a trusted project function without making it inherit framework classes."""

    def __init__(
        self,
        callable_path: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
        class_path: str | None = None,
        init_parameters: Mapping[str, Any] | None = None,
        method: str = "run",
    ) -> None:
        if (callable_path is None) == (class_path is None):
            raise ValueError("CallableTask requires exactly one callable_path or class_path.")
        self.callable_path = callable_path
        self.parameters = dict(parameters or {})
        self.seed = seed
        self.class_path = class_path
        self.init_parameters = dict(init_parameters or {})
        self.method = method

    def run(self, context: Any) -> Any:
        if self.class_path is None:
            function = import_callable(str(self.callable_path))
        else:
            target = import_callable(self.class_path)
            instance = target(**_resolve_inputs(self.init_parameters, context))
            function = getattr(instance, self.method)
            if not callable(function):
                raise TypeError(
                    f"Configured method is not callable: {self.class_path}.{self.method}"
                )
        parameters = _resolve_inputs(self.parameters, context)
        if self.seed is not None:
            _seed_everything(self.seed)
            signature = inspect.signature(function)
            if "seed" in signature.parameters and "seed" not in parameters:
                parameters["seed"] = self.seed
        return function(**parameters)

    def runtime_parameters(self, context: Any) -> dict[str, Any]:
        """Return resolved parameters for provenance without invoking the function."""
        return _resolve_inputs(self.parameters, context)


def import_callable(path: str) -> Any:
    """Import and validate one dotted callable reference without constructing it."""
    if not isinstance(path, str) or "." not in path:
        raise ValueError("run must be a dotted import path such as 'my_project.train'.")
    module_name, symbol_name = path.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), symbol_name)
    if not callable(value):
        raise TypeError(f"Configured run target is not callable: {path}")
    return value


def signature_errors(
    path: str | None,
    parameters: Mapping[str, Any],
    seed: int | None,
    *,
    class_path: str | None = None,
    init_parameters: Mapping[str, Any] | None = None,
    method: str = "run",
) -> list[str]:
    """Return author-facing callable signature errors without running user code."""
    if class_path is not None:
        target = import_callable(class_path)
        try:
            inspect.signature(target).bind(**_placeholder_inputs(init_parameters or {}))
        except TypeError as error:
            return [f"init parameters do not match {class_path}: {error}"]
        function = getattr(target, method, None)
        if not callable(function):
            return [f"Configured class method is not callable: {class_path}.{method}"]
        display = f"{class_path}.{method}"
    else:
        function = import_callable(str(path))
        display = str(path)
    arguments = _placeholder_inputs(parameters)
    signature = inspect.signature(function)
    if seed is not None and "seed" in signature.parameters and "seed" not in arguments:
        arguments["seed"] = seed
    try:
        signature.bind(**arguments)
    except TypeError as error:
        # Unbound class methods expose self; remove it for ordinary authored arguments.
        if class_path is not None:
            try:
                signature.bind(object(), **arguments)
            except TypeError as class_error:
                return [f"with parameters do not match {display}: {class_error}"]
            return []
        return [f"with parameters do not match {display}: {error}"]
    return []


def _resolve_inputs(value: Any, context: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"__lambdaforge_input__"}:
            return context.input(str(value["__lambdaforge_input__"]))
        return {str(key): _resolve_inputs(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_inputs(item, context) for item in value]
    return value


def _placeholder_inputs(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"__lambdaforge_input__"}:
            return Path("input")
        return {str(key): _placeholder_inputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_placeholder_inputs(item) for item in value]
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
