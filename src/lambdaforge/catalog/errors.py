"""Stable machine-readable authoring errors, independent of CLI presentation."""

from typing import Any


class ContractError(ValueError):
    """A public operation/sample contract was not satisfied."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code, self.details = code, details

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": self.details,
            "translation_key": "lambdaforge.errors." + self.code,
        }
