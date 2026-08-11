"""Callable fixture for concise preprocessing authoring tests."""

from collections.abc import Mapping
from typing import Any


def normalize_record(value: Any) -> Any:
    """Upper-case the text field of a record mapping."""
    if not isinstance(value, Mapping):
        return value
    return {**value, "text": str(value["text"]).upper()}
