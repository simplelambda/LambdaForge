"""Explicit stack/ragged/pad and sparse disjoint-graph batching; never flatten vector channels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.catalog.errors import ContractError
from lambdaforge.data.samples.SampleSchema import SampleSchema


def collate_samples(samples: Sequence[Mapping[str, Any]], schema: SampleSchema) -> dict[str, Any]:
    import numpy as np

    if not samples:
        raise ContractError("EMPTY_BATCH", "Cannot collate an empty batch.")
    fields: dict[str, Any] = {}
    masks: dict[str, Any] = {}
    for spec in schema.fields:
        values = [s["fields"][spec.name] for s in samples]
        masks[spec.name] = np.asarray([s["present"][spec.name] for s in samples], dtype=bool)
        if spec.collate in {"list", "ragged"}:
            fields[spec.name] = values
            continue
        if any(v is None for v in values):
            raise ContractError("MISSING_BATCH_FIELD", "Stack/pad needs explicit default values.")
        arrays = [np.asarray(v) for v in values]
        if len({a.dtype.str for a in arrays}) != 1:
            raise ContractError("DTYPE_MISMATCH", "Batch fields cannot silently promote dtypes.")
        if spec.collate == "stack":
            if len({a.shape for a in arrays}) != 1:
                raise ContractError("RAGGED_BATCH", "Declare ragged or pad for unequal shapes.")
            fields[spec.name] = np.stack(arrays)
        else:
            if any(a.ndim < 1 or a.shape[1:] != arrays[0].shape[1:] for a in arrays):
                raise ContractError("RAGGED_BATCH", "Padding supports only the first sample axis.")
            length = max(len(a) for a in arrays)
            output = np.zeros((len(arrays), length, *arrays[0].shape[1:]), dtype=arrays[0].dtype)
            valid = np.zeros((len(arrays), length), dtype=bool)
            for i, a in enumerate(arrays):
                output[i, : len(a)], valid[i, : len(a)] = a, True
            fields[spec.name], masks[spec.name + ":padding"] = output, valid
    return {
        "ids": [s["id"] for s in samples],
        "fields": fields,
        "masks": masks,
        "provenance": [s["provenance"] for s in samples],
    }


def collate_graphs(
    graphs: Sequence[Mapping[str, Any]],
    *,
    node_fields: Sequence[str],
    edge_fields: Sequence[str] = (),
    edge_index: str = "edge_index",
) -> dict[str, Any]:
    import numpy as np

    if not graphs or not node_fields:
        raise ContractError("EMPTY_BATCH", "Graph batching needs graphs and named node fields.")
    all_fields = [*node_fields, *edge_fields]
    if len(set(all_fields)) != len(all_fields) or set(all_fields) & {edge_index, "batch", "ptr"}:
        raise ContractError("INVALID_BINDING", "Graph fields overlap or replace reserved indices.")
    nodes, edges, ptr, owner = [], [], [0], []
    for i, graph in enumerate(graphs):
        n = len(graph[node_fields[0]])
        if any(len(graph[k]) != n for k in node_fields):
            raise ContractError("GRAPH_CARDINALITY", "Node fields have inconsistent lengths.")
        indices = np.asarray(graph[edge_index])
        if indices.ndim != 2 or indices.shape[0] != 2 or indices.dtype.kind not in "iu":
            raise ContractError("GRAPH_INDEX", "edge_index must be integer [2,E].")
        if indices.size and (indices.min() < 0 or indices.max() >= n):
            raise ContractError("GRAPH_INDEX", "Edge index outside its sample's node domain.")
        if any(len(graph[k]) != indices.shape[1] for k in edge_fields):
            raise ContractError("GRAPH_CARDINALITY", "Edge fields have inconsistent lengths.")
        edges.append(indices.astype(np.int64) + ptr[-1])
        ptr.append(ptr[-1] + n)
        owner.extend([i] * n)
        nodes.append(graph)
    result: dict[str, Any] = {
        edge_index: np.concatenate(edges, axis=1),
        "batch": np.asarray(owner, dtype=np.int64),
        "ptr": np.asarray(ptr, dtype=np.int64),
    }
    for key in (*node_fields, *edge_fields):
        values = [np.asarray(g[key]) for g in nodes]
        if len({a.dtype.str for a in values}) != 1 or len({a.shape[1:] for a in values}) != 1:
            raise ContractError("INCOMPATIBLE_SHAPE", f"Cannot concatenate graph field {key}.")
        result[key] = np.concatenate(values, axis=0)
    result["ids"] = [g.get("id") for g in graphs]
    result["provenance"] = [g.get("provenance") for g in graphs]
    return result
