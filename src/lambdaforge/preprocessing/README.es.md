# Preprocesado en LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md) · [Tareas](../tasks/README.es.md)

El preprocesado se implementa componiendo contratos públicos genéricos, no como una excepción para
un proyecto concreto:

```text
PreprocessingSource → PreprocessingTransform(s) → PreprocessingSink
                              │
                              └→ PreprocessingTask → TaskResult + DatasetArtifact
```

El ejemplo ejecutable completo es
[examples/preprocessing.yaml](../../../examples/preprocessing.yaml). Revísalo antes de ejecutarlo:

```bash
lambdaforge validate examples/preprocessing.yaml
lambdaforge inspect examples/preprocessing.yaml
lambdaforge run examples/preprocessing.yaml --dry-run
lambdaforge run examples/preprocessing.yaml
```

## Empieza aquí

Piensa en el preprocesado como una cinta transportadora. La **fuente** decide qué registros existen
y asigna a cada uno una clave estable. Cada **transformación** recibe un registro y devuelve su
valor modificado. El **destino** escribe el resultado y puede demostrar después si esa clave ya está
completa. `PreprocessingTask` coordina la cinta, guarda progreso y produce un manifiesto
`DatasetArtifact` que describe el dataset completo.

Para JSONL normal empieza con la forma concisa y nombra cada ruta una vez:

```yaml
name: normalizar-registros
inputs: {raw: data/raw.jsonl}
outputs: {processed: processed}
preprocess:
  function: mi_proyecto.preprocessing.normalizar
  input: raw
  output: processed
  key_field: id
  workers: 4
  workload: io
```

`workers: 1` procesa secuencialmente. Un valor mayor usa un pool de hilos acotado; `workload`
registra intención `auto`, `io`, `cpu` o `gpu`, pero no elige procesos, dispositivos ni batch size
en secreto. Para cómputo Python CPU aislado o batching GPU usa shards explícitos de workflow o una
tarea de proyecto. Manifiesto, verificación atómica y política de error no cambian.
Las transformaciones y sinks propios usados con varios workers deben ser thread-safe.

El ejemplo del repositorio se ejecuta directamente con su input incluido y una lista de
transformaciones vacía; no necesita un paquete consumidor. La configuración inferior muestra
después dónde añadir una transformación de proyecto opcional. Como en toda configuración,
`validate` la comprueba, `inspect` explica el plan, `run --dry-run` ejercita la planificación y
`run` ejecuta. Empieza con una transformación y un proceso; introduce shards sólo después de
verificar claves estables y reanudación.

## Pipeline incluido

`JsonLinesSource` lee un JSON por línea no vacía y usa un campo o el número estable de línea como
clave. `FileTreeSource` produce ficheros regulares ordenados con claves relativas y rechaza symlinks.
`JsonDirectorySink` escribe un envelope JSON atómico por registro; su nombre SHA-256 evita convertir
claves inseguras en rutas. `CallableTransform` envuelve explícitamente una función YAML `ref` que
transforma sólo el valor.

```yaml
schema_version: "1.0"
kind: task
name: normalize-records
inputs:
  - {name: raw, path: data/raw.jsonl}
task:
  target: lambdaforge.preprocessing.PreprocessingTask
  params:
    source:
      target: lambdaforge.preprocessing.JsonLinesSource
      params: {path: data/raw.jsonl, key_field: id}
    transforms:
      - target: lambdaforge.preprocessing.CallableTransform
        params:
          function: {ref: mi_proyecto.preprocessing.normalize_record}
    sink:
      target: lambdaforge.preprocessing.JsonDirectorySink
      params: {output_dir: processed}
    on_error: fail
    checkpoint_interval: 1
    dataset_name: normalized-records
    dataset_version: "1"
```

`target` construye clases y `ref` importa la función sin llamarla. La ciencia se mantiene en el
paquete instalado del consumidor. Para objetos más ricos implementa los tres contratos pequeños:

```python
from collections.abc import Iterable
from lambdaforge.preprocessing import (
    PreprocessingRecord,
    PreprocessingSource,
    PreprocessingTransform,
    PreprocessingSink,
)


class ProjectSource(PreprocessingSource):
    def records(self, context) -> Iterable[PreprocessingRecord]:
        yield PreprocessingRecord(key="stable-id", value={"raw": 1})


class ProjectTransform(PreprocessingTransform):
    def transform(self, record, context) -> PreprocessingRecord:
        return record.with_value({"feature": record.value["raw"] * 2})


class ProjectSink(PreprocessingSink):
    def write(self, record, context) -> None: ...
```

Un sink debería redefinir `is_complete(key, context)` si puede verificar el output por registro y
devolver `ArtifactDeclaration` agregados desde `finalize(context)`. Así resume omite sólo outputs
que siguen presentes.

## Resume, fallos y shards

El pipeline escribe `preprocessing-manifest.json` atómicamente cada `checkpoint_interval` registros
(default 1). Cada entrada guarda clave, último estado, fecha UTC y error estructurado. Un retry con
el mismo fingerprint reutiliza éxitos sólo si el sink confirma su output; los fallidos se repiten.

`on_error: fail` aborta tras registrar el primer fallo. `skip` registra fallos y permite terminar;
las métricas explicitan que el resultado es parcial. LambdaForge no oculta fallos ni cambia inputs.

`shard_count: N` y `shard_index: i` asignan cada clave mediante SHA-256 módulo N. Los shards son
deterministas, disjuntos y cubren la fuente. Ejecuta cada shard como configuración explícita con su
propio nombre/output; esta versión no los lanza ni fusiona automáticamente. El scheduling local/HPC
general pertenece a la siguiente fase del roadmap.

## DatasetArtifact

Cada `PreprocessingTask` correcto escribe `dataset-artifact.json` con:

- `dataset_id` derivado del contenido;
- nombre y versión humanos;
- número de muestras y splits opcionales;
- fingerprint de preprocesado/task;
- descriptores de fuente/inputs;
- SHA-256 y tamaño de cada artefacto del sink;
- fecha, versión LambdaForge, enlace al environment manifest y metadata.

El ID excluye fecha y ubicación absoluta. La misma ciencia y bytes producen la misma identidad.
Los splits no pueden ser negativos ni superar el total. `environment.json` conserva Git, Python,
paquetes, CUDA y plugins.

## Límites

- Schema 1.0 procesa secuencialmente dentro de un task. Usa shards explícitos para trabajo paralelo;
  los pools CPU y DAG no se ocultan en esta versión.
- Inputs/outputs son locales. Stores compartidos/remotos necesitan futuros contratos de leases y
  atomicidad.
- Los targets/refs Python son código de confianza. Validar JSON no es sandboxing.
- El framework no puede inferir un input externo no declarado. `PreprocessingTask` exige al menos un
  `inputs` content-hashed en el nivel superior, y las rutas de sources JSONL/árbol deben coincidir
  con él o estar dentro de un directorio de entrada declarado.
