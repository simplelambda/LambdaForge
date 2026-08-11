"""One actionable environment diagnostic."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """Represent one portable diagnostic outcome."""

    name: str
    ok: bool
    message: str
    fix: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        """Return a machine-readable check."""
        return {"name": self.name, "ok": self.ok, "message": self.message, "fix": self.fix}
