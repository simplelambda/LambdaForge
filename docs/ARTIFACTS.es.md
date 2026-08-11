[English](ARTIFACTS.md) | [Español](ARTIFACTS.es.md)

# Inspección y visualización de artifacts

Los artifacts son resultados científicos, no objetos Python fiables. Inspección responde “qué
contiene”; visualización, “cómo representar una semántica explícita”; validación, “si cumple las
restricciones”. Son contratos separados.

## 1. Inspección y export seguros

```bash
lambdaforge artifact inspect predictions.npz --array logits --rows 20 --slice 0:100,:
lambdaforge artifact export predictions.npz --array logits --format csv --output logits.csv
lambdaforge artifact validate graph.npz --require-array positions \
  --shape positions=*,3 --finite
```

Los built-ins soportan NPY, NPZ, CSV, TSV, JSON y JSONL. NumPy usa siempre `allow_pickle=False`.
Arrays object, symlinks y archivos ausentes/vacíos fallan. Preview se limita a 1000 filas. En arrays
grandes se usa una muestra determinista acotada marcada `sampled:N`. `--slice` sólo acepta enteros y
`start:stop[:step]`, nunca `eval`. Exporta CSV, JSON o NPY seguro.

## 2. Geometría explícita

Un array `(N,3)` no se asume coordenadas. Para NPZ graph hay que nombrar nodes y edges; se comprueba
forma e índices. Point cloud requiere positions. Mesh OBJ/PLY/STL/OFF necesita
`lambdaforge[viz3d]`; HTML necesita Plotly. El render se limita y `PlotSpec` registra conteos y
truncado.

## 3. Descubrimiento y plugins

`artifact list SELECTOR` lee envelopes y plots; con JOB sincroniza sólo metadata pequeña.
`artifact fetch JOB NAME` trae un único nombre lógico y rechaza traversal. Nunca selecciona por
fecha. Los proyectos pueden publicar inspector, visualizer, schema, exporter y validator en grupos
`lambdaforge.artifact_*`; el core no debe incorporar convenciones de un laboratorio.
