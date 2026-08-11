"""Environment diagnostic report."""

from dataclasses import dataclass

from lambdaforge.controlplane.DoctorCheck import DoctorCheck


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Collect checks for one local or remote profile."""

    cluster: str
    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        """Return whether all required checks passed."""
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        """Return the stable report envelope."""
        return {
            "cluster": self.cluster,
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        """Render actionable terminal output."""
        lines = [f"LambdaForge doctor ({self.cluster}): {'OK' if self.ok else 'ISSUES'}"]
        for check in self.checks:
            lines.append(f"[{'ok' if check.ok else '!!'}] {check.name}: {check.message}")
            if not check.ok and check.fix:
                lines.append(f"     fix: {check.fix}")
        return "\n".join(lines)
