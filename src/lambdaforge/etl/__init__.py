"""Domain data operations called from ordinary Work Python, never a graph interpreter."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.etl.CsvOptions import CsvOptions
    from lambdaforge.etl.Download import Download
    from lambdaforge.etl.JoinRecords import JoinRecords
    from lambdaforge.etl.ListFiles import ListFiles
    from lambdaforge.etl.PartitionRecords import PartitionRecords
    from lambdaforge.etl.PreviewArray import PreviewArray
    from lambdaforge.etl.PreviewBudget import PreviewBudget
    from lambdaforge.etl.PreviewSession import PreviewSession
    from lambdaforge.etl.ReadRecords import ReadRecords
    from lambdaforge.etl.ReadSQL import ReadSQL
    from lambdaforge.etl.SelectFields import SelectFields
    from lambdaforge.etl.SQLWritePolicy import SQLWritePolicy
    from lambdaforge.etl.TableOperations import (
        AggregateTable,
        CastTable,
        ConcatTables,
        DeduplicateTable,
        FillNulls,
        FilterTable,
        FromRows,
        SelectTable,
        SortTable,
    )
    from lambdaforge.etl.TrainStandardizer import TrainStandardizer
    from lambdaforge.etl.ValidateRecords import ValidateRecords
    from lambdaforge.etl.WriteRecords import WriteRecords
    from lambdaforge.etl.WriteSampleAssets import WriteSampleAssets
    from lambdaforge.etl.WriteSQL import WriteSQL

__all__ = [
    "ListFiles",
    "ReadRecords",
    "WriteRecords",
    "SelectFields",
    "JoinRecords",
    "PartitionRecords",
    "WriteSampleAssets",
    "TrainStandardizer",
    "ValidateRecords",
    "Download",
    "CsvOptions",
    "PreviewBudget",
    "PreviewArray",
    "PreviewSession",
    "ReadSQL",
    "WriteSQL",
    "SQLWritePolicy",
    "AggregateTable",
    "CastTable",
    "ConcatTables",
    "DeduplicateTable",
    "FillNulls",
    "FilterTable",
    "FromRows",
    "SelectTable",
    "SortTable",
]
_TABLES = {
    "AggregateTable",
    "CastTable",
    "ConcatTables",
    "DeduplicateTable",
    "FillNulls",
    "FilterTable",
    "FromRows",
    "SelectTable",
    "SortTable",
}
LazyExports.install(
    __name__,
    {
        n: ("lambdaforge.etl.TableOperations" if n in _TABLES else f"lambdaforge.etl.{n}", n)
        for n in __all__
    },
)
