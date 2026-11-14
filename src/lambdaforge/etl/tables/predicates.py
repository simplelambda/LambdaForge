"""Non-executable scalar table predicates with SQL-like explicit null handling."""

from collections.abc import Mapping
from typing import Any

from lambdaforge.catalog.errors import ContractError


def validate_predicate(spec: Mapping[str, Any], *, depth: int = 0) -> None:
    if depth > 16 or not isinstance(spec, Mapping):
        raise ContractError("INVALID_PREDICATE", "Predicate nesting exceeds 16 or is malformed.")
    if set(spec) in ({"all"}, {"any"}):
        children = spec.get("all", spec.get("any"))
        if not isinstance(children, list) or len(children) > 100:
            raise ContractError("INVALID_PREDICATE", "Expected bounded predicate list.")
        for child in children:
            validate_predicate(child, depth=depth + 1)
    elif set(spec) == {"not"}:
        validate_predicate(spec["not"], depth=depth + 1)
    elif set(spec) <= {"column", "op", "value"} and isinstance(spec.get("column"), str):
        if spec.get("op") not in {"eq", "ne", "lt", "le", "gt", "ge", "in", "is_null", "not_null"}:
            raise ContractError("INVALID_PREDICATE", "Unsupported table predicate operator.")
        if spec["op"] not in {"is_null", "not_null"} and "value" not in spec:
            raise ContractError("INVALID_PREDICATE", "Comparison requires an explicit value.")
        if spec["op"] == "in" and not isinstance(spec.get("value"), (list, tuple)):
            raise ContractError("INVALID_PREDICATE", "Membership requires an explicit list.")
    else:
        raise ContractError("INVALID_PREDICATE", "Expected declarative column predicate.")


def matches(row: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    if "all" in spec:
        return all(matches(row, child) for child in spec["all"])
    if "any" in spec:
        return any(matches(row, child) for child in spec["any"])
    if "not" in spec:
        return not matches(row, spec["not"])
    if spec["column"] not in row:
        raise ContractError("MISSING_FIELD", "Predicate column is absent.")
    value, peer, op = row[spec["column"]], spec.get("value"), spec["op"]
    if op == "is_null":
        return value is None
    if op == "not_null":
        return value is not None
    if value is None or peer is None:
        return False
    if isinstance(value, (dict, list, tuple)):
        raise ContractError("UNSUPPORTED_VALUE", "Scalar predicates do not flatten structures.")
    try:
        if op == "in":
            return value in peer
        return {
            "eq": lambda: value == peer,
            "ne": lambda: value != peer,
            "lt": lambda: value < peer,
            "le": lambda: value <= peer,
            "gt": lambda: value > peer,
            "ge": lambda: value >= peer,
        }[op]()
    except (TypeError, KeyError):
        raise ContractError("INVALID_PREDICATE", "Incompatible predicate operands.") from None
