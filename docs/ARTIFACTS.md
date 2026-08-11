[English](ARTIFACTS.md) | [Español](ARTIFACTS.es.md)

# Artifact inspection and visualization

Artifacts are scientific outputs, not trusted Python objects. Inspection answers “what is in this
file”; visualization answers “how should explicit semantics be drawn”; validation answers “does it
meet declared constraints”. Those responsibilities are separate public contracts.

## 1. Safe inspection and export

```bash
lambdaforge artifact inspect predictions.npz --json
lambdaforge artifact inspect predictions.npz --array logits --rows 20 --slice 0:100,:
lambdaforge artifact export predictions.npz --array logits --format csv --output logits.csv
lambdaforge artifact validate graph.npz --require-array positions \
  --shape positions=*,3 --finite
```

Built-ins support NPY, NPZ, CSV, TSV, JSON and JSONL. NumPy always uses `allow_pickle=False`.
Object arrays, symlinks and missing/empty files fail. Preview rows are capped at 1000. Statistics
are exact only below the configured element limit; larger arrays use a deterministic bounded sample
and say `sampled:N`. `--slice` accepts integers and `start:stop[:step]` only—never Python `eval`.
Export supports CSV, JSON or safe NPY. Parquet result export is the separate `parquet` extra.

## 2. Explicit geometry

```bash
lambdaforge artifact visualize graph.npz --type graph \
  --nodes positions --edges edge_index --output graph.svg
lambdaforge artifact visualize points.npz --type point-cloud \
  --positions xyz --output points.html
lambdaforge artifact visualize surface.ply --type mesh --output surface.html
```

An `(N,3)` array is not assumed to be coordinates. NPZ graph roles must be named; edges must be
`(2,E)` or `(E,2)` and indices are range-checked. Point clouds require explicit positions. Mesh
OBJ/PLY/STL/OFF loading needs `lambdaforge[viz3d]`; HTML also needs Plotly. Rendering is bounded by
point/edge limits and its `PlotSpec` records truncation/counts.

## 3. Discovery, remote fetch and plugins

```bash
lambdaforge artifact list baseline
lambdaforge artifact list JOB
lambdaforge artifact fetch JOB predictions --output analysis/predictions.npz
lambdaforge artifact plugins --json
```

Local listing reads result envelopes and plot sidecars. Remote listing first synchronizes only small
result metadata; `fetch` then retrieves exactly one logical name and rejects traversal outside the
job workspace. It never guesses by modification time.

Projects can publish `ArtifactInspector`, `ArtifactVisualizer`, `ArtifactSchema`, exporter and
`ArtifactValidator` providers through the `lambdaforge.artifact_*` entry-point groups. Discovery
lists metadata without importing providers. Domain conventions—protein arrays, medical volumes,
special graph fields—belong in the consumer plugin, not LambdaForge core.
