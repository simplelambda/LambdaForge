"""Zero-argument duck task used to verify the concise compatibility contract."""


class NoContextTask:
    """Return a plain mapping without inheriting the framework base class."""

    def run(self) -> dict[str, bool]:
        """Return one JSON-compatible output mapping."""
        return {"zero_argument": True}
