"""Native Work-cache HTTP operation, with no download during discovery or construction."""

from typing import Any


class Download:
    """Native atomic cache/checksum with bounded reads and cooperative cancellation.

    Optional HTTPS bearer authentication uses env:/keyring: references; never put secrets in
    URLs or constructor values. Authenticated redirects cannot leave their original origin.
    """

    def __init__(
        self,
        timeout: float = 30,
        retries: int = 3,
        max_bytes: int = 256 * 1024**2,
        credential: str | None = None,
    ) -> None:
        self.timeout, self.retries = timeout, retries
        self.max_bytes, self.credential = max_bytes, credential

    def __call__(self, work: Any, url: str, *, key: str, sha256: str) -> Any:
        import hashlib
        import re
        import urllib.parse

        from lambdaforge.catalog.errors import ContractError

        parsed = urllib.parse.urlsplit(url)
        if parsed.username or parsed.password or parsed.query:
            raise ContractError("SECRET_URL", "Use a credential-free URL, never embedded secrets.")
        if not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", sha256):
            raise ContractError("INVALID_CHECKSUM", "An exact SHA-256 checksum is required.")
        if work.stop_requested:
            raise ContractError("CANCELLED", "Download cancelled before request.")

        def validate(path: Any) -> bool:
            digest = hashlib.sha256()
            with open(path, "rb") as file:
                while chunk := file.read(1024**2):
                    if work.stop_requested:
                        raise ContractError(
                            "CANCELLED", "Download cancelled during checksum validation."
                        )
                    digest.update(chunk)
            return digest.hexdigest() == sha256.removeprefix("sha256:")

        return work.cache.fetch(
            url,
            key=key,
            timeout=self.timeout,
            retries=self.retries,
            validate=validate,
            max_bytes=self.max_bytes,
            cancelled=lambda: bool(work.stop_requested),
            credential=self.credential,
        )
