[English](PREPROCESSING.md) | [Español](PREPROCESSING.es.md)

# Ejecución y debug de preprocesado

## 1. Camino sencillo

```yaml
name: prepare
inputs: {raw: data/raw.jsonl}
outputs: {processed: processed}
preprocess:
  function: my_project.preprocessing.normalize_record
  key_field: id
  workers: 4
  workload: io
```

Ejecute `validate`, `inspect`, `run --dry-run` y `run`. La forma concisa compila a la misma task
source/transform/sink estricta. Los inputs se hashean por contenido; cambiar bytes cambia identidad.
Claves estables gobiernan shards, manifest y resume. Sólo se reutiliza si el sink verifica bytes.

## 2. Semántica de workload

- un worker es la referencia secuencial;
- `io` usa threads acotados;
- `cpu` usa procesos `spawn` sólo para transforms serializables y el padre escribe sink/manifest;
- `auto` prefiere threads conservadores;
- `gpu` exige un worker; varias GPU requieren shards/jobs/recursos explícitos.

`workers`, `workload` y la cadencia de checkpoint son política de ejecución y se excluyen del
fingerprint de preprocesado built-in. Los modos secuencial/thread/proceso deben producir los mismos
hashes e identidad `DatasetArtifact` para inputs y transforms iguales.

El padre checkpointa futures terminados. FAIL cancela y persiste; SKIP registra y continúa. Claves y
shards son iguales en Linux y Windows.

## 3. Debug de muestra

`lambdaforge debug preprocessing.yaml --records 3 [--intermediates debug/stages]` lee como máximo N
registros y muestra clave/tipo, transform, preview, duración y excepción. No llama al sink, no
finaliza dataset ni crea el output de producción. La identidad `debug:` nunca se reutiliza como
`DatasetArtifact` completo.

## 4. Inspección de dataset

Con el manifest registrado, `lambdaforge data --catalog data-catalog.yaml inspect
dataset:processed-v3` informa de identidad/ubicaciones/tamaño, dataset ID, productor/configuración,
muestras/splits, artifacts y validación.
