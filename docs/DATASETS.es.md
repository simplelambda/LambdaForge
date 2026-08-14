# Ciclo de vida de datasets

Español | [English](DATASETS.md) | [Guía raíz](../README.es.md)

## 1. Modelo mental

LambdaForge 0.7 separa:

```text
DatasetRecipe (cómo) -> DatasetBuild (ejecución) -> DatasetVersion (qué)
                                                     -> DatasetPlacement (dónde)
```

Una Task produce artifacts normales. Un preprocesado no publica salvo `publish_dataset: true` o el
`dataset_name` legacy; la frontera preferida es DatasetRecipe. DatasetVersion es una colección
lógica inmutable, no un directorio ni un `Dataset` de PyTorch. Puede tener varios placements con la
misma identidad.

## 2. Contenido e identidad

`DatasetIndex` procesa JSONL canónico en streaming y entrega `DatasetMember`. Cada miembro posee ID
estable, `partitions` y `targets` arbitrarios, metadata científica, `display` descriptivo fuera de
identidad y assets con nombre. Los assets pueden ser ficheros, varios ficheros, directorios, records
en contenedores o URIs; no se impone un layout por muestra.

```python
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember

members = [DatasetMember(
    "sample-001",
    partitions={"split": "train", "fold": 3, "cohort": "external"},
    targets={"class": 1, "affinity": 4.27},
    metadata={"source": "instrument-a"},
    display={"description": "No forma identidad científica"},
    assets={"features": DatasetAsset("samples/001/features.npz", sha256="sha256:<hash real>")},
)]
DatasetIndex.write("dataset/members.jsonl", members)
```

Los resúmenes se derivan del índice. DatasetArtifact v2 separa `content_id`/`dataset_id` de
`build_id`. El primero depende de IDs, partitions, targets, metadata científica y checksums, no de
ruta, display, clúster o provenance. Mover bytes conserva content ID; otra build del mismo contenido
cambia build ID. V1 sigue siendo legible.

## 3. Receta y build

Parte del [ejemplo genérico](../examples/dataset-recipe.yaml). Cada stage es una Task existente;
`needs`/`bindings` reutilizan Workflow. `required` expresa necesidad científica y, de forma
ortogonal, `reuse: auto|never` controla reutilización content-addressed.

```yaml
kind: dataset
dataset: {name: example-records, version: "1"}
stages:
  discover: {task: tasks/discover.yaml, required: true, reuse: auto}
  normalize:
    task: tasks/normalize.yaml
    needs: [discover]
    bindings:
      task.params.roster: ${nodes.discover.artifacts.roster.jsonl}
publish: {from: normalize, root: dataset, index: members.jsonl}
```

```bash
lf validate dataset.yaml
lf datasets plan example-records --on atlas
lf datasets build example-records --on atlas
lf jobs show latest
```

Plan muestra las acciones y `PUBLISH`/`NOOP`; `--verbose` añade cada motivo. Un plan local puede
demostrar `REUSE` con su caché exacta. Un plan remoto hecho desde el controlador marca como
`MISSING` lo que no ha observado, en vez de confundir caché local/remota; el worker durable vuelve a
comprobar los fingerprints Task exactos antes de ejecutar. `--force` fuerza todo y `--force-stage NAME`
invalida downstream. Build es un job durable. Las stages mantienen fingerprint, result y receipts
de integridad en cache reconstruible; un fallo posterior no las borra. GC puede recoger cache no
referenciada/stale, nunca datasets publicados, resultados o jobs activos.

Publish valida IDs, assets, índice, schema JSON de targets y stages requeridas en staging; después
verifica el manifiesto, renombra atómicamente y registra. Un fallo no publica. `name@version` rechaza
otro content ID y los bytes publicados no se sobrescriben.

## 4. Resolver y consumir

DatasetRegistry es la autoridad operacional de placements gestionados. DatasetResolver fija
versión/content ID y selecciona placement. DataCatalog sigue para aliases, datos externos, loaders,
pins y overrides; los catálogos 0.6 funcionan, pero no hay que duplicar rutas gestionadas.

```yaml
data:
  train: dataset:example-records@1/train
  val:
    target: mi_proyecto.data.MiDataset
    params:
      root: {dataset: example-records, version: "1", subpath: validation}
```

Un split directo necesita `loader` y `path_parameter`; un marker anidado inyecta el path. La
evidencia materializada fija nombre, versión, content/build ID y placement. El fingerprint usa
contenido, no path. Sin versión hay error si varias coinciden.

## 5. Inspección y profiling

```bash
lf datasets ls --all
lf datasets show example-records@1
lf datasets stats example-records@1 --on atlas
lf datasets members example-records@1 --partition split=train --limit 50
lf datasets member example-records@1 sample-001
lf datasets diff example-records@1 example-records@2
lf datasets verify example-records@1 --on atlas
lf datasets lineage example-records@1
```

List consulta registries pequeños, no filesystems. Members devuelve 100 por defecto y admite
offset. Diff separa miembros añadidos/eliminados/cambiados y cambios de partitions, targets/assets.
Stats universales cubren miembros, partitions, tamaño, ficheros, tipos de asset y checksums ausentes.
Sólo un schema explícito activa semántica de targets. Un profiler del proyecto se ejecuta junto al
placement remoto dentro del entorno managed exacto, sin descargar datos. Read-only admite `--json`.

## 6. Materializar, replicar y eliminar

```bash
lf datasets materialize example-records@1 --on atlas              # preview
lf datasets materialize example-records@1 --on atlas --apply
lf datasets replicate example-records@1 --from local --to atlas --apply
lf datasets remove example-records@1 --on atlas
lf datasets delete example-records@1 --on atlas --apply
```

Materialize devuelve `NOOP`, `REPLICATE` o BUILD multi-stage. Apply materializa prerrequisitos
soportados y envía un job `dataset-build`, sin comandos manuales. El relay incluido parte de fuente
local, usa staging verificado/publicación atómica y requiere el controlador online; no finge ser
transferencia durable cluster-a-cluster. Remove sólo desregistra. Delete exige placement exacto,
válido, sin consumidor y `--apply`. GC no selecciona versiones.

## 7. CLI, API y migración

`lf` y `lambdaforge` son el mismo entrypoint. Gramática:
`lf <recurso> <acción> <objeto> [--on CONTEXTO]`; aliases: `ds`, `exp`, `env`, `ls`.
`lf plan CONFIG --on CLUSTER` es dry-run y `lf completion bash|zsh|fish` genera completion.

Importa los modelos lógicos, receta/build, Registry/Resolver/Service, planes y errores desde
`lambdaforge.data`. Artifact/Record v1 y DataCatalog 0.6 siguen legibles. `dataset_name` publica un
v2 legacy; sin él ni `publish_dataset`, preprocessing es sólo Task. Producers legacy se pueden
previsualizar, pero apply necesita receta tipada. Stage artifacts no se listan como DatasetVersion.
