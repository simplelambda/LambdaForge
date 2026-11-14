"""Small DB-API boundary: SQLite stdlib and optional Psycopg 3; no query interpreter."""

from __future__ import annotations

import importlib
import json
import math
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lambdaforge.catalog.errors import ContractError


def identifier(value: str) -> str:
    """Portable unqualified identifiers only; quote after syntax validation."""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise ContractError("UNSAFE_IDENTIFIER", "Use a simple declared SQL identifier.")
    return '"' + value + '"'


def validate_connection(
    dialect: str,
    file: str | None,
    credential: str | None,
    timeout: float,
    statement_timeout: float,
    sslmode: str,
) -> None:
    if dialect not in {"sqlite", "postgresql"}:
        raise ContractError("UNSUPPORTED_BACKEND", "Supported SQL backends: sqlite/postgresql.")
    if any(
        isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in (timeout, statement_timeout)
    ):
        raise ContractError("INVALID_PARAMETERS", "Finite positive SQL timeouts required.")
    if dialect == "sqlite":
        if not file or credential is not None:
            raise ContractError("INVALID_PARAMETERS", "SQLite requires a file, not a credential.")
    elif file is not None or not credential or not credential.startswith(("env:", "keyring:")):
        raise ContractError(
            "INVALID_PARAMETERS", "PostgreSQL requires env:/keyring: JSON connection reference."
        )
    if sslmode not in {"verify-full", "verify-ca", "require"}:
        raise ContractError("INVALID_PARAMETERS", "PostgreSQL SSL cannot be disabled.")


@contextmanager
def connection(config: Any, *, readonly: bool, budget: Any = None) -> Iterator[Any]:
    """Close/rollback even when a generator is closed, cancelled or only partially consumed."""
    conn = None
    try:
        if config.dialect == "sqlite":
            import sqlite3

            from lambdaforge.work.snapshot import validate_path

            path = Path(config.file).absolute()
            validate_path(path)
            if not path.is_file():
                raise ContractError(
                    "MISSING_SOURCE", "SQL file must already exist; no implicit database creation."
                )
            conn = sqlite3.connect(
                path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"),
                uri=True,
                timeout=config.timeout,
            )
            if hasattr(conn, "setlimit") and budget is not None:
                conn.setlimit(
                    sqlite3.SQLITE_LIMIT_LENGTH,  # type: ignore[attr-defined]
                    min(budget.max_memory // 4, budget.max_bytes),
                )
            statement_deadline = time.monotonic() + config.statement_timeout

            def progress():
                try:
                    if budget is not None:
                        budget.check()
                    return int(time.monotonic() >= statement_deadline)
                except ContractError:
                    return 1

            conn.set_progress_handler(progress, 500)
            conn.execute("BEGIN")
            if readonly:
                allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
                functions = {
                    "count",
                    "sum",
                    "avg",
                    "min",
                    "max",
                    "coalesce",
                    "ifnull",
                    "nullif",
                    "abs",
                    "round",
                    "lower",
                    "upper",
                    "length",
                    "substr",
                    "substring",
                    "date",
                    "datetime",
                    "strftime",
                    "cast",
                    "like",
                    "glob",
                    "typeof",
                }

                def authorize(action, first, second, db, trigger):
                    return (
                        sqlite3.SQLITE_OK
                        if action in allowed
                        or (action == sqlite3.SQLITE_FUNCTION and str(second).lower() in functions)
                        else sqlite3.SQLITE_DENY
                    )

                conn.set_authorizer(authorize)
        else:
            from lambdaforge.controlplane.CredentialService import CredentialService

            try:
                backend = importlib.import_module("psycopg")
            except ImportError:
                raise ContractError(
                    "MISSING_DEPENDENCY", "Install lambdaforge[sql-postgresql]."
                ) from None
            try:
                values = json.loads(CredentialService().resolve_reference(config.credential))
                if not isinstance(values, dict) or set(values) - {
                    "host",
                    "port",
                    "dbname",
                    "user",
                    "password",
                    "sslrootcert",
                }:
                    raise ValueError("Invalid connection keys")
                conn = backend.connect(
                    **values,
                    sslmode=config.sslmode,
                    connect_timeout=max(1, math.ceil(config.timeout)),
                )
            except Exception:
                # Driver/credential errors may echo DSNs/passwords; never chain them.
                raise ContractError(
                    "SQL_CONNECTION",
                    "Could not open PostgreSQL connection; check host credential reference/TLS.",
                ) from None
            conn.read_only = readonly
            conn.isolation_level = backend.IsolationLevel.REPEATABLE_READ
            conn.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                (str(math.ceil(config.statement_timeout * 1000)),),
            )
        yield conn
        if not readonly:
            conn.commit()
    except ContractError:
        raise
    except Exception:
        if budget is not None:
            budget.check()
        raise ContractError(
            "SQL_ERROR",
            "SQL operation failed; rolled back. Check schema, keys, parameters and access.",
        ) from None
    finally:
        if conn is not None:
            try:
                if config.dialect == "sqlite":
                    conn.set_authorizer(lambda *_: sqlite3.SQLITE_OK)
                    conn.set_progress_handler(None, 0)
                conn.rollback()
            finally:
                conn.close()
