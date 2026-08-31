"""Small semantic colour layer for LambdaForge's dependency-light terminal UI."""

from __future__ import annotations

import os
import re
from typing import TextIO


class TerminalTheme:
    """Apply colour after layout so ANSI bytes never corrupt width calculations."""

    RESET = "\x1b[0m"
    BOLD_CYAN = "\x1b[1;36m"
    BOLD_BLUE = "\x1b[1;34m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    BOLD_GREEN = "\x1b[1;32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[1;31m"
    DIM = "\x1b[2m"
    SELECTED = "\x1b[30;46m"

    _STATES = {
        "running": GREEN,
        "succeeded": GREEN,
        "complete": GREEN,
        "completed": GREEN,
        "online": GREEN,
        "preparing": YELLOW,
        "staging": YELLOW,
        "queued": YELLOW,
        "scheduled": YELLOW,
        "unknown": YELLOW,
        "pruned": YELLOW,
        "failed": RED,
        "offline": RED,
        "cancelled": RED,
    }

    @staticmethod
    def enabled(stream: TextIO) -> bool:
        """Respect standard terminal capability and ``NO_COLOR`` conventions."""
        return bool(
            getattr(stream, "isatty", lambda: False)()
            and os.environ.get("TERM", "") != "dumb"
            and "NO_COLOR" not in os.environ
        )

    @classmethod
    def apply(cls, screen: str, *, enabled: bool) -> str:
        if not enabled:
            return screen
        lines = screen.splitlines()
        output: list[str] = []
        for index, line in enumerate(lines):
            if not line:
                output.append(line)
                continue
            if line.startswith("LambdaForge"):
                line = cls._wrap(line, cls.BOLD_CYAN)
            elif cls._is_section(line):
                line = cls._wrap(line, cls.BOLD_BLUE)
            else:
                line = re.sub(
                    r"(?<![\w-])(" + "|".join(cls._STATES) + r")(?![\w-])",
                    lambda match: cls._wrap(match.group(0), cls._STATES[match.group(0)]),
                    line,
                )
                line = re.sub(
                    r"[\u2801-\u28ff]+",
                    lambda match: cls._wrap(match.group(0), cls.CYAN),
                    line,
                )
                line = line.replace("●", cls._wrap("●", cls.BOLD_RED))
                line = line.replace("◆", cls._wrap("◆", cls.BOLD_GREEN))
                line = line.replace("◈", cls._wrap("◈", cls.BOLD_GREEN))
                line = line.replace("★", cls._wrap("★", cls.BOLD_GREEN))
                line = line.replace("▶", cls._wrap("▶", cls.SELECTED))
                line = line.replace("[ERROR]", cls._wrap("[ERROR]", cls.BOLD_RED))
                line = line.replace("[WARNING]", cls._wrap("[WARNING]", cls.YELLOW))
            if index == len(lines) - 1:
                line = cls._wrap(line, cls.DIM)
            output.append(line)
        return "\n".join(output)

    @staticmethod
    def _is_section(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        title = re.split(r"\s+(?:·|\()", stripped, maxsplit=1)[0]
        return any(character.isalpha() for character in title) and title == title.upper()

    @classmethod
    def _wrap(cls, text: str, style: str) -> str:
        return f"{style}{text}{cls.RESET}"


__all__ = ["TerminalTheme"]
