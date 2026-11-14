"""Transactional SQL tests use local disposable SQLite and mocked optional PG adapters."""

import json
import sqlite3
import sys
import traceback
from types import SimpleNamespace

import pytest

from lambdaforge.catalog import ContractError
from lambdaforge.etl import PreviewBudget, PreviewSession, ReadSQL, SQLWritePolicy, WriteSQL
from lambdaforge.etl.tables import Column, TableSchema


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "source.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE samples (id TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        conn.executemany("INSERT INTO samples VALUES (?,?)", [("a", 1), ("b", 2), ("c", 3)])
    return path


def schema():
    return TableSchema([Column("id"), Column("value", "integer")], keys=["id"]).to_dict()


def test_parameterization_projection_chunks_and_mutable_source_evidence(database):
    reader = ReadSQL(
        "sqlite",
        file=str(database),
        query="SELECT * FROM samples WHERE value>? ORDER BY id",
        parameters=[1],
        chunk_size=1,
    )
    result = reader.preview()
    assert result["rows"] == [{"id": "b", "value": 2}, {"id": "c", "value": 3}]
    assert result["content_identity"] is None and result["provenance"]["mutable_source"]
    with sqlite3.connect(database) as conn:
        conn.execute("UPDATE samples SET value=20 WHERE id='b'")
    changed = reader.preview()
    assert changed["observed_rows_sha256"] != result["observed_rows_sha256"]
    assert changed["provenance"]["query_sha256"] == result["provenance"]["query_sha256"]
    assert ReadSQL("sqlite", file=str(database), table="samples", columns=["id"], order_by=["id"])(
        budget=PreviewBudget(max_rows=1)
    ) == [{"id": "a"}]
    injection = ReadSQL(
        "sqlite",
        file=str(database),
        query="SELECT * FROM samples WHERE id=?",
        parameters=["a' OR 1=1 --"],
    )
    assert injection() == []


def test_snapshot_consistency_and_close_releases_read_transaction(database):
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    iterator = ReadSQL(
        "sqlite", file=str(database), table="samples", order_by=["id"], chunk_size=1
    ).iter_records()
    assert next(iterator)["id"] == "a"
    with sqlite3.connect(database) as conn:
        conn.execute("UPDATE samples SET value=9 WHERE id='b'")
    assert next(iterator)["value"] == 2
    iterator.close()
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM samples",
        "PRAGMA journal_mode",
        "SELECT load_extension('arbitrary')",
        "SELECT 1; DELETE FROM samples",
    ],
)
def test_sql_preview_rejects_mutations_extension_loading_and_multiple_statements(database, query):
    with pytest.raises(ContractError):
        ReadSQL("sqlite", file=str(database), query=query).preview()
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 3


def test_unsafe_identifier_and_network_preview_permissions(database, tmp_path):
    with pytest.raises(ContractError, match="identifier"):
        ReadSQL("sqlite", file=str(database), table="samples; DROP TABLE samples")
    session = PreviewSession([tmp_path])
    postgres = ReadSQL("postgresql", credential="env:TEST_PG", table="samples")
    with pytest.raises(ContractError, match="Network"):
        session.sql(postgres)
    writer = WriteSQL("sqlite", "samples", schema(), file=str(database))
    with pytest.raises(ContractError, match="Only native SQL readers"):
        session.sql(writer)
    with pytest.raises(ContractError, match="policy"):
        writer([{"id": "new", "value": 1}], policy=SQLWritePolicy())


def test_sql_writer_transaction_idempotency_schema_and_authorization(database):
    insert = WriteSQL("sqlite", "samples", schema(), file=str(database))
    with pytest.raises(ContractError):
        insert(
            [{"id": "new", "value": 4}, {"id": "a", "value": 1}],
            policy=SQLWritePolicy(allow_insert=True),
        )
    assert not ReadSQL(
        "sqlite", file=str(database), query="SELECT * FROM samples WHERE id=?", parameters=["new"]
    )()
    upsert = WriteSQL("sqlite", "samples", schema(), file=str(database), mode="upsert")
    policy = SQLWritePolicy(allow_insert=True, allow_update=True)
    for _ in range(2):
        assert upsert([{"id": "a", "value": 9}], policy=policy)["transaction"] == "committed"
    rows = ReadSQL("sqlite", file=str(database), table="samples", order_by=["id"])()
    assert len(rows) == 3 and rows[0]["value"] == 9
    replace = WriteSQL("sqlite", "samples", schema(), file=str(database), mode="replace")
    with pytest.raises(ContractError):
        replace([], policy=policy)
    assert (
        replace(
            [{"id": "only", "value": 1}],
            policy=SQLWritePolicy(allow_insert=True, allow_replace=True),
        )["rows"]
        == 1
    )
    create = WriteSQL("sqlite", "another", schema(), file=str(database), create=True)
    with pytest.raises(ContractError):
        create([], policy=policy)
    assert (
        create([{"id": "a", "value": 1}], policy=SQLWritePolicy(allow_insert=True, allow_ddl=True))[
            "rows"
        ]
        == 1
    )


def test_sql_cancel_and_budget_failure_roll_back_all_rows(database):
    writer = WriteSQL("sqlite", "samples", schema(), file=str(database))

    def rows():
        yield {"id": "new", "value": 1}
        yield {"id": "later", "value": 2}

    with pytest.raises(ContractError):
        writer(rows(), policy=SQLWritePolicy(allow_insert=True), budget=PreviewBudget(max_rows=1))
    assert len(ReadSQL("sqlite", file=str(database), table="samples")()) == 3
    with pytest.raises(ContractError, match="cancelled"):
        ReadSQL("sqlite", file=str(database), table="samples").preview(
            budget=PreviewBudget(cancelled=lambda: True)
        )


def test_sqlite_query_work_deadline_is_enforced(database):
    query = (
        "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x WHERE n<100000000) "
        "SELECT SUM(n) FROM x"
    )
    with pytest.raises(ContractError):
        ReadSQL("sqlite", file=str(database), query=query, statement_timeout=0.001).preview()


def test_postgresql_credentials_are_lazy_and_redacted(monkeypatch):
    secret = "unique-password-not-in-logs"
    monkeypatch.setenv("TEST_PG", json.dumps({"host": "fixture.invalid", "password": secret}))
    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(secret)

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    reader = ReadSQL("postgresql", credential="env:TEST_PG", table="samples", timeout=1)
    assert calls == []
    with pytest.raises(ContractError) as error:
        reader.preview()
    assert secret not in "".join(traceback.format_exception(error.value))
    assert calls[0]["sslmode"] == "verify-full" and calls[0]["connect_timeout"] == 1
    with pytest.raises(ContractError):
        ReadSQL("postgresql", credential="raw-password", table="samples")
    with pytest.raises(ContractError):
        ReadSQL("postgresql", credential="env:TEST_PG", table="samples", sslmode="disable")


def test_optional_postgresql_real_fixture_requires_explicit_disposable_connection():
    import os
    from uuid import uuid4

    # Opt-in ONLY for an operator-provided ephemeral test database, never default/local sites.
    reference = os.environ.get("LAMBDAFORGE_TEST_EPHEMERAL_PG_REF")
    if reference is None:
        pytest.skip(
            "No explicitly provided disposable TLS PostgreSQL fixture; no server contacted."
        )
    pytest.importorskip("psycopg")
    import psycopg

    from lambdaforge.controlplane.CredentialService import CredentialService

    table = "lf_test_" + uuid4().hex
    writer = WriteSQL(
        "postgresql", table, schema(), credential=reference, create=True, mode="upsert"
    )
    try:
        policy = SQLWritePolicy(allow_insert=True, allow_update=True, allow_ddl=True)
        writer([{"id": "a", "value": 1}], policy=policy)
        writer([{"id": "a", "value": 2}], policy=policy)
        assert ReadSQL("postgresql", credential=reference, table=table, order_by=["id"])() == [
            {"id": "a", "value": 2}
        ]
    finally:
        values = json.loads(CredentialService().resolve_reference(reference))
        with psycopg.connect(**values, sslmode="verify-full") as conn:
            from psycopg.sql import SQL, Identifier

            conn.execute(SQL("DROP TABLE IF EXISTS {}").format(Identifier(table)))
