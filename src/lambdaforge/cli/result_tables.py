"""JSON/CSV rendering over the native bounded ResultStore table read model."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from lambdaforge.work.ResultStore import ResultStore


def _csv_cell(value: Any) -> Any:
    if isinstance(value, dict | list):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value  # Inert spreadsheet text, not executable formula content.
    return value


def render_table(value: dict[str, Any], format: str) -> str:
    if format == "json":
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=value["columns"], lineterminator="\n")
    writer.writeheader()
    for row in value["rows"]:
        writer.writerow({name: _csv_cell(row.get(name)) for name in value["columns"]})
    return stream.getvalue()


def run_result_table(arguments: argparse.Namespace, store: ResultStore) -> int:
    if arguments.json and arguments.format != "json":
        raise ValueError("Choose --format csv or --json, not both.")
    columns = tuple(arguments.columns.split(",")) if arguments.columns else ()
    value = store.table(
        arguments.selector,
        metrics=tuple(arguments.metric),
        runs=tuple(arguments.run_id),
        trials=tuple(arguments.trial),
        columns=columns,
        attempts=arguments.attempts,
        offset=arguments.offset,
        limit=arguments.limit,
        fingerprint=arguments.fingerprint,
    )
    rendered = render_table(value, arguments.format)
    if len(rendered.encode()) > 1024 * 1024:
        raise ValueError("Rendered result table exceeds 1 MiB; reduce --limit or --columns.")
    if arguments.output is None:
        print(rendered, end="")
        if arguments.format == "csv":
            import sys

            print(
                f"Derived table · {len(value['rows'])}/{value['total']} rows · "
                f"next_offset={value['next_offset']} · {value['notice']}",
                file=sys.stderr,
            )
        return 0
    path = Path(arguments.output).expanduser().absolute()
    if path.resolve() != path:
        raise ValueError("Result table destination cannot contain symlinks or traversal.")
    if path.is_relative_to(store.root):
        raise ValueError("Write derived tables outside the native Run evidence root.")
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "path": str(path),
        "format": arguments.format,
        "rows": len(value["rows"]),
        "total": value["total"],
        "next_offset": value["next_offset"],
        "evidence_fingerprint": value["evidence_fingerprint"],
        "notice": value["notice"],
    }
    print(json.dumps(receipt, allow_nan=False))
    return 0
