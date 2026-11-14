"""Small synthetic streaming/typing/preview fixtures, no consumer data or remote systems."""

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pytest

from lambdaforge.catalog import ContractError
from lambdaforge.etl import (
    AggregateTable,
    CastTable,
    ConcatTables,
    DeduplicateTable,
    FillNulls,
    FilterTable,
    FromRows,
    PreviewArray,
    PreviewBudget,
    PreviewSession,
    ReadRecords,
    SelectTable,
    SortTable,
    WriteRecords,
)
from lambdaforge.etl.tables import Column, TableSchema


def test_large_csv_prefix_does_not_decode_read_or_hash_the_whole_file(tmp_path, monkeypatch):
    file = tmp_path / "large.csv"
    with file.open("wb") as handle:
        handle.write(b"id,value,unused\nfirst,1,ok\nsecond,2,ok\n")
        handle.seek(16 * 1024**2)
        handle.write(b"\xffinvalid unread tail")
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("No full source read"))
    result = ReadRecords("csv", columns=["id", "value"]).preview(
        file, budget=PreviewBudget(max_rows=2, max_bytes=100)
    )
    assert result["rows"] == [{"id": "first", "value": "1"}, {"id": "second", "value": "2"}]
    assert result["scope"] == "prefix" and not result["complete"]
    assert result["content_identity"] is None and result["budget"]["bytes"] < 100
    assert result["provenance"][1]["physical_end_line"] == 3


def test_csv_dialect_headers_comments_multiline_bom_decimal_and_dates(tmp_path):
    file = tmp_path / "heterogeneous.csv"
    file.write_text(
        "\ufeffdiscard\n# comment\ngroup;group;group;group\nid;value;date;label\n"
        'a;"1,25";2026-10-29;"first\n# not a comment"\nb;NA;2026-10-30;second\n'
    )
    schema = TableSchema(
        [
            Column("value", "number", nullable=True, null_values=("NA",)),
            Column("date", "datetime", date_format="%Y-%m-%d", timezone="UTC"),
        ]
    )
    reader = ReadRecords(
        "csv",
        csv={
            "delimiter": ";",
            "bom": "strip",
            "skip_rows": 1,
            "header_rows": 2,
            "comment": "#",
            "decimal": ",",
        },
        schema=schema.to_dict(),
    )
    rows = reader(file)
    assert rows[0]["value"] == 1.25 and rows[1]["value"] is None
    assert rows[0]["label"] == "first\n# not a comment"
    assert rows[0]["date"] == "2026-10-29T00:00:00+00:00"
    file.write_text("\ufeffid,value\na,1\n")
    with pytest.raises(ContractError, match="BOM"):
        ReadRecords("csv", csv={"bom": "error"})(file)


def test_headerless_tsv_explicit_names_and_parse_error_policy(tmp_path):
    file = tmp_path / "input.tsv"
    file.write_text("a\t1\nb\t2\textra\nc\t3\n")
    options = {"header_rows": 0, "columns": ["id", "value"]}
    with pytest.raises(ContractError, match="width"):
        ReadRecords("tsv", csv=options)(file)
    report = ReadRecords("tsv", csv={**options, "parse_errors": "skip"}).preview(file)
    assert [r["id"] for r in report["rows"]] == ["a", "c"]
    assert report["diagnostics"] == [{"code": "SKIPPED_RECORD", "line": 2}]
    assert len(list(ReadRecords("tsv", csv=options, max_rows=1).iter_records(file))) == 1
    with pytest.raises(ContractError):
        ReadRecords("csv", csv={"header_rows": 0})


def test_csv_writing_declared_encoding_quotes_decimal_and_no_overwrite(tmp_path):
    file = tmp_path / "latin.csv"
    rows = [{"id": "café", "number": 1.25, "text": 'a;"b'}]
    options = {"delimiter": ";", "encoding": "latin-1", "decimal": ","}
    WriteRecords("csv", csv=options)(file, rows)
    assert ReadRecords("csv", csv={k: v for k, v in options.items() if k != "decimal"})(file)[
        0
    ] == {
        "id": "café",
        "number": "1,25",
        "text": 'a;"b',
    }
    with pytest.raises(FileExistsError):
        WriteRecords("csv", csv=options)(file, rows)
    with pytest.raises(ContractError):
        WriteRecords("csv", csv={"skip_rows": 3})


def test_global_limits_cancellation_and_temporary_cleanup(tmp_path):
    file = tmp_path / "rows.csv"
    file.write_text("id,value\na,1\nb,2\n")
    budget = PreviewBudget(max_rows=3, max_files=2)
    session = PreviewSession([tmp_path], budget)
    assert len(session.records(ReadRecords("csv"), file)["rows"]) == 2
    assert len(session.records(ReadRecords("csv"), file)["rows"]) == 1
    with pytest.raises(ContractError, match="file budget"):
        session.records(ReadRecords("csv"), file)
    with pytest.raises(ContractError, match="cancelled"):
        ReadRecords("csv").preview(file, budget=PreviewBudget(cancelled=lambda: True))
    expired = PreviewBudget()
    expired.deadline = time.monotonic() - 1
    with pytest.raises(ContractError, match="deadline"):
        ReadRecords("csv").preview(file, budget=expired)
    session = PreviewSession([tmp_path])
    with pytest.raises(RuntimeError):
        with session.staging_directory() as scratch:
            (scratch / "partial").write_text("scratch only")
            raise RuntimeError("fixture")
    assert not scratch.exists() and file.is_file()
    with pytest.raises(ContractError, match="outside"):
        session.records(ReadRecords("csv"), tmp_path / "../foreign.csv")
    file.write_text("id,value\na," + "x" * 10000)
    with pytest.raises(ContractError) as caught:
        ReadRecords("csv").preview(file, budget=PreviewBudget(max_memory=1000))
    assert caught.value.code == "READ_LIMIT"


def test_compressed_streaming_and_large_array_prefix_bounds(tmp_path):
    file = tmp_path / "table.gz"
    with gzip.open(file, "wb") as handle:
        handle.write(b"id,x\na,1\n" + b"b,2\n" * 100000)
    result = ReadRecords("csv", compression="gzip").preview(
        file, budget=PreviewBudget(max_rows=1, max_bytes=100)
    )
    assert result["rows"] == [{"id": "a", "x": "1"}]
    array = np.arange(100000, dtype=np.float32).reshape(-1, 2)
    np.save(tmp_path / "large.npy", array)
    np.savez_compressed(tmp_path / "large.npz", x=array)
    for format in ("npy", "npz"):
        preview = PreviewArray(format, "x" if format == "npz" else None)(
            tmp_path / ("large." + format), budget=PreviewBudget(max_rows=2, max_bytes=1024)
        )
        assert preview["values"] == array[:2].tolist() and preview["shape"] == [50000, 2]
        assert not preview["complete"] and preview["budget"]["bytes"] < 1024
        with pytest.raises(ContractError):
            PreviewArray(format, "x" if format == "npz" else None)(
                tmp_path / ("large." + format), budget=PreviewBudget(max_rows=1000, max_memory=100)
            )
    np.savez_compressed(tmp_path / "fortran.npz", x=np.asfortranarray(array))
    with pytest.raises(ContractError, match="Fortran"):
        PreviewArray("npz", "x")(tmp_path / "fortran.npz")


def test_column_null_conversion_categories_and_timezone_contracts():
    date_column = Column("date", "datetime", date_format="%d/%m/%Y", timezone="UTC")
    assert date_column.convert(date_column.convert("29/10/2026")) == "2026-10-29T00:00:00+00:00"
    with pytest.raises(ContractError):
        Column("n", "integer").convert("1.2")
    assert Column("n", "number", True, invalid="null").convert("bad") is None
    with pytest.raises(ContractError):
        Column("n", "integer", nullable=False, invalid="null")
    with pytest.raises(ContractError, match="convert"):
        Column("label", categories=("a", "b")).convert("c")
    with pytest.raises(ContractError):
        Column("date", "datetime", timezone="Europe/Madrid").convert("2026-10-25T02:30:00")
    with pytest.raises(ContractError):
        Column("n", "number").convert([1, 2])


def test_table_operations_preserve_identity_partitions_and_explicit_derived_lineage():
    rows = [
        {"id": "a", "x": "1", "partitions": {"split": "train"}},
        {"id": "b", "x": "bad", "partitions": {"split": "validation"}},
    ]
    schema = TableSchema([Column("x", "integer", True, invalid="null")])
    cast = list(CastTable(schema.to_dict())(rows))
    filled = list(FillNulls({"x": 0})(cast))
    assert filled[1]["x"] == 0 and filled[1]["partitions"] == rows[1]["partitions"]
    assert list(FilterTable({"column": "x", "op": "gt", "value": 0})(filled))[0]["id"] == "a"
    assert [r["id"] for r in SortTable(["x"])(filled)] == ["b", "a"]
    assert list(SelectTable(["x"], rename={"x": "n"})(filled))[0]["id"] == "a"
    with pytest.raises(ContractError):
        SelectTable(drop=["id"])
    with pytest.raises(ContractError):
        CastTable(TableSchema([Column("id")]).to_dict())
    groups = AggregateTable([], {"mean": {"column": "x", "op": "mean"}}, identity="derive")(filled)
    assert len(groups) == 2 and groups[0]["provenance"]["member_ids"] == ["a"]
    assert groups[0]["id"].startswith("aggregate-")
    assert len(DeduplicateTable(["id"])([*filled, *filled])) == 2
    with pytest.raises(ContractError):
        DeduplicateTable(["x"])([{"id": "a", "x": 0}, {"id": "b", "x": 0}])
    assert ConcatTables()([[filled[0]], [filled[1]]]) == filled
    with pytest.raises(ContractError):
        ConcatTables()([filled, filled])
    declared = TableSchema([Column("id"), Column("x", "integer", True)], keys=["id"])
    assert FromRows(declared.to_dict())(filled) == filled
    with pytest.raises(ContractError):
        FromRows(declared.to_dict())([*filled, *filled])
    with pytest.raises(ContractError):
        SortTable(["x"], max_rows=1)(filled)
    assert json.loads(json.dumps(declared.to_dict())) == declared.to_dict()
