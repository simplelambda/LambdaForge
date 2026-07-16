"""Text stream fan-out used by experiment logging."""

from __future__ import annotations

import io
from typing import Any


class TeeStream(io.TextIOBase):
    """Forward each write and flush to every configured text stream."""

    def __init__(self, *streams: Any) -> None:
        super().__init__()
        self._streams = [stream for stream in streams if stream is not None]

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()
