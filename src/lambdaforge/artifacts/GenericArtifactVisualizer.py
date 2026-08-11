"""Explicit generic graph, point-cloud and mesh visualizer."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lambdaforge.artifacts.ArtifactVisualizer import ArtifactVisualizer
from lambdaforge.visualization.PlotSpec import PlotSpec


class GenericArtifactVisualizer(ArtifactVisualizer):
    """Build bounded geometry specs only when roles are explicitly provided."""

    def __init__(self, *, max_points: int = 100_000, max_edges: int = 200_000) -> None:
        self.max_points = int(max_points)
        self.max_edges = int(max_edges)

    def specification(
        self, path: Path, *, visualization_type: str, roles: Mapping[str, Any]
    ) -> PlotSpec:
        """Load explicit arrays/formats and produce a generic geometry PlotSpec."""
        if visualization_type == "mesh":
            return self._mesh(path)
        if visualization_type == "graph" and path.suffix.lower() in {
            ".graphml",
            ".gexf",
            ".edgelist",
        }:
            return self._networkx_graph(path)
        if path.suffix.lower() not in {".npy", ".npz"}:
            raise ValueError("Graph/point-cloud built-ins require an explicit NPY/NPZ artifact.")
        loaded = np.load(path, allow_pickle=False)
        try:
            arrays = {path.stem: loaded} if isinstance(loaded, np.ndarray) else loaded
            if visualization_type == "point-cloud":
                name = str(roles.get("positions", ""))
                if not name:
                    raise ValueError("Point-cloud visualization requires --positions ARRAY.")
                positions = self._positions(np.asarray(arrays[name]), name)
                point_rows = tuple(
                    {
                        "index": index,
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "z": float(point[2]),
                    }
                    for index, point in enumerate(positions[: self.max_points])
                )
                return PlotSpec(
                    "point-cloud",
                    data_references=(str(path.resolve()),),
                    x="x",
                    y="y",
                    data=point_rows,
                    metadata={"positions": name, "truncated": len(positions) > self.max_points},
                )
            if visualization_type == "graph":
                node_name = str(roles.get("nodes", ""))
                edge_name = str(roles.get("edges", ""))
                if not node_name or not edge_name:
                    raise ValueError("Graph visualization requires --nodes and --edges arrays.")
                positions = self._positions(np.asarray(arrays[node_name]), node_name)
                edges = np.asarray(arrays[edge_name])
                if edges.ndim != 2 or 2 not in edges.shape:
                    raise ValueError("Graph edges must have shape (2, E) or (E, 2).")
                pairs = edges.T if edges.shape[0] == 2 else edges
                if pairs.size and (pairs.min() < 0 or pairs.max() >= len(positions)):
                    raise ValueError(
                        "Graph edge index references a node outside the positions array."
                    )
                graph_rows: list[dict[str, Any]] = [
                    {
                        "kind": "node",
                        "index": index,
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "z": float(point[2]),
                    }
                    for index, point in enumerate(positions[: self.max_points])
                ]
                graph_rows.extend(
                    {"kind": "edge", "source": int(pair[0]), "target": int(pair[1])}
                    for pair in pairs[: self.max_edges]
                )
                return PlotSpec(
                    "graph",
                    data_references=(str(path.resolve()),),
                    data=tuple(graph_rows),
                    metadata={
                        "nodes": node_name,
                        "edges": edge_name,
                        "node_count": len(positions),
                        "edge_count": len(pairs),
                    },
                )
            raise ValueError("Visualization type must be graph, point-cloud or mesh.")
        finally:
            close = getattr(loaded, "close", None)
            if callable(close):
                close()

    def _networkx_graph(self, path: Path) -> PlotSpec:
        """Load an explicit graph format and delegate deterministic layout to NetworkX."""
        try:
            networkx = importlib.import_module("networkx")
        except ImportError as error:
            raise ImportError(
                "GraphML/GEXF/edge-list support requires lambdaforge[graph]."
            ) from error
        suffix = path.suffix.lower()
        graph = (
            networkx.read_graphml(path)
            if suffix == ".graphml"
            else networkx.read_gexf(path)
            if suffix == ".gexf"
            else networkx.read_edgelist(path)
        )
        node_ids = tuple(graph.nodes)[: self.max_points]
        node_index = {node: index for index, node in enumerate(node_ids)}
        positions = networkx.spring_layout(graph.subgraph(node_ids), dim=3, seed=0)
        rows: list[dict[str, Any]] = [
            {
                "kind": "node",
                "index": index,
                "label": str(node),
                "x": float(positions[node][0]),
                "y": float(positions[node][1]),
                "z": float(positions[node][2]),
            }
            for node, index in node_index.items()
        ]
        edges = tuple(
            (left, right)
            for left, right in graph.edges
            if left in node_index and right in node_index
        )
        rows.extend(
            {
                "kind": "edge",
                "source": node_index[left],
                "target": node_index[right],
            }
            for left, right in edges[: self.max_edges]
        )
        return PlotSpec(
            "graph",
            data_references=(str(path.resolve()),),
            data=tuple(rows),
            metadata={
                "format": suffix.removeprefix("."),
                "layout": "networkx.spring_layout(dim=3, seed=0)",
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "truncated": len(node_ids) < graph.number_of_nodes() or len(edges) > self.max_edges,
            },
        )

    def _mesh(self, path: Path) -> PlotSpec:
        if path.suffix.lower() not in {".obj", ".ply", ".stl", ".off"}:
            raise ValueError("Mesh visualization supports OBJ, PLY, STL and OFF explicitly.")
        try:
            trimesh = importlib.import_module("trimesh")
        except ImportError as error:
            raise ImportError("Mesh support requires lambdaforge[viz3d].") from error
        mesh = trimesh.load(path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        rows = tuple(
            {"kind": "vertex", "index": index, "x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
            for index, v in enumerate(vertices[: self.max_points])
        ) + tuple(
            {"kind": "face", "indices": [int(value) for value in face]}
            for face in faces[: self.max_edges]
        )
        return PlotSpec(
            "mesh",
            data_references=(str(path.resolve()),),
            data=rows,
            metadata={"vertex_count": len(vertices), "face_count": len(faces)},
        )

    @staticmethod
    def _positions(value: np.ndarray, name: str) -> np.ndarray:
        if value.ndim != 2 or value.shape[1] not in {2, 3}:
            raise ValueError(f"Positions array {name!r} must have shape (N, 2) or (N, 3).")
        if value.shape[1] == 2:
            value = np.column_stack((value, np.zeros(len(value))))
        return value
