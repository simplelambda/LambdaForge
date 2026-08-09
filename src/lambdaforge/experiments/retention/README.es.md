# Retención de artefactos

[Guía de experimentos](../README.es.md) · [Guía del repositorio](../../../../README.es.md) ·
[English](README.md)

Este paquete reduce de forma segura suites de experimentos ya completadas. Permite conservar roles
de checkpoint concretos, archivar archivos intermedios grandes y podar archivos desechables
declarados explícitamente. La retención prioriza la previsualización, es agnóstica de la tarea y
permanece desactivada salvo configuración o petición explícita.

## Contenidos

- [Contrato YAML](#contrato-yaml)
- [Elegibilidad](#elegibilidad)
- [Artefactos protegidos](#artefactos-protegidos)
- [Roles de checkpoint](#roles-de-checkpoint)
- [Reglas y archivos ZIP](#reglas-y-archivos-zip)
- [API de objetos y CLI](#api-de-objetos-y-cli)
- [Transacción y recuperación](#transacción-y-recuperación)
- [Artefactos producidos](#artefactos-producidos)
- [Límites](#límites)

## Contrato YAML

El Schema 1.1 añadió la retención:

```yaml
schema_version: "1.1"

retention:
  mode: preview                 # disabled, preview, apply
  checkpoints:
    keep: last_and_best         # all, best, last, last_and_best
    prune_unselected: true
  protect:
    - reports/**
    - predictions/final.json
  rules:
    - action: compress
      include: [predictions/**/*.json, embeddings/*.npy]
      exclude: [predictions/final.json]
      min_size_bytes: 1048576
      compression:
        level: 9                # override opcional de archive.compression_level
        only_if_smaller: true
    - action: prune
      include: [scratch/**]
      exclude: []
      min_size_bytes: 0
  archive:
    name: artifacts.zip
    compression_level: 6
  lock_timeout_seconds: 60
```

Omitir el bloque equivale a `mode: disabled`, `keep: all`, `prune_unselected: false`, ninguna
regla, `artifacts.zip`, nivel de compresión 6 y timeout de lock de 60 segundos.

- `disabled` conserva el comportamiento histórico. Una llamada explícita a
  `Experiment.apply_retention()` o al CLI con `--apply` sigue siendo una petición manual
  intencionada.
- `preview` habilita la planificación, pero nunca elimina, archiva, crea locks ni escribe informes
  por sí solo.
- `apply` pide a la agregación final correcta que aplique automáticamente la retención.

Todos los mappings son estrictos y rechazan claves desconocidas. La validación rechaza listas
`include` vacías, tamaños negativos, acciones no soportadas, niveles ZIP fuera de 0–9 y opciones
de compresión en reglas de poda. La validación en runtime también rechaza patrones absolutos, con
unidad/UNC, `..`, barras inversas o NUL.

## Elegibilidad

Aplicar exige un `aggregate/aggregation_receipt.json` vigente. El receipt solo se escribe
atómicamente tras una agregación final y solo está completo cuando:

1. la suite expandida contiene al menos un run;
2. existen todas las variantes y semillas esperadas;
3. cada run tiene `status: ok`;
4. cada variante está completa y terminal con los recuentos exactos;
5. cada run conserva `config.yaml`, `environment.json`, `hparams.json` y `result.json` seguros;
   también se compromete `metrics.csv` cuando `trainer.write_epoch_metrics_csv` está habilitado;
6. existe de forma segura cada ruta de `experiment.required_artifacts`;
7. se publicaron los CSV/JSON principales y los demás productos de agregación;
8. siguen coincidiendo los fingerprints de configuración, entradas y salidas comprometidas.

Los resultados fallidos, interrumpidos, ignorados, pendientes o dry-run nunca son elegibles. La
agregación incremental por variante invalida un receipt anterior y usa `final=False`, por lo que no
puede activar retención. Un nuevo entrenamiento invalida el receipt antes de iniciar workers.

`Experiment.preview_retention()` devuelve `not_ready` sin escribir si el receipt falta, está
incompleto o quedó obsoleto. Las fuentes se vuelven a identificar inmediatamente antes de archivar
y poner en cuarentena; cualquier divergencia revierte la transacción.

## Artefactos protegidos

Las reglas genéricas nunca afectan a:

- `config.yaml`, `environment.json`, `hparams.json`, `train.log`, `metrics.csv` o `result.json`;
- `checkpoints/**`, gobernado por su propia política con roles;
- toda ruta exacta de `experiment.required_artifacts` y su subárbol;
- rutas coincidentes con `retention.protect`;
- ZIPs, locks, journals, manifiestos y metadatos de cuarentena de la retención;
- `aggregate/**` y los resúmenes de la suite;
- enlaces simbólicos, junctions/reparse points, archivos especiales o rutas fuera del run;
- archivos que no coincidan con exactamente una regla.

Si dos reglas seleccionan el mismo archivo regular, la planificación falla antes de mutar nada.
Los patrones `include`/`exclude` son globs POSIX relativos al directorio de cada run.

## Roles de checkpoint

`trainer.checkpoint_policy` controla lo que Lightning crea durante el entrenamiento.
`retention.checkpoints` controla de forma independiente lo que sobrevive tras una agregación final
correcta. `prune_unselected: false` conserva todos los checkpoints con independencia de `keep`.

Cuando la poda está activa, `best` y `last` solo se resuelven dentro de `checkpoints/` del run
actual. El resolver admite rutas registradas, rebasing de runs movidos y las convenciones
`best-*`, `last.ckpt` y `epoch-*` de LambdaForge/Lightning. Para una política de entrenamiento
`all`, `last` es el mayor epoch generado no ambiguo. Si falta un rol pedido o es ambiguo, se omite
la poda de checkpoints de todo ese run. Deliberadamente no existe una opción de retención `none`.

Las rutas registradas podadas se eliminan atómicamente de `result.json`. La finalización y el resume
inspeccionan archivos locales seguros, de modo que `checkpoint_policy: all` sigue siendo reutilizable
aunque Lightning no exponga una ruta best/last. `CheckpointChoice.AUTO` carga best, después last y
por último el checkpoint local seguro más reciente; las selecciones exactas `BEST` y `LAST` no
cambian silenciosamente de rol.

## Reglas y archivos ZIP

Una regla `compress` transmite los archivos seleccionados a un ZIP por run y nivel efectivo de
compresión. Los nombres publicados derivan de la base configurada, el nivel y el fingerprint del
plan, por ejemplo
`.lambdaforge/retention/artifacts-l9-0123456789ab.zip`. Mantener los archivos propios de la política
bajo el directorio interno del run impide que un cambio posterior de reglas o nombre seleccione un
ZIP antiguo. Los archivos son inmutables y nunca sobrescriben una ruta existente. Zip64 está
habilitado. Antes de mover originales, LambdaForge vuelve a abrir el ZIP y verifica CRCs, nombres,
tamaños y SHA-256. Un manifiesto interno registra el plan y los hashes de sus miembros.

Con `only_if_smaller: true`, los miembros cuyo contenido comprimido no sea menor permanecen en su
sitio. Si todos los miembros son opcionales y el ZIP completo, incluidos metadatos, no es menor que
los originales, no se publica ningún archivo y todos los originales se conservan.

Una regla `prune` no archiva. Sus archivos también pasan por una cuarentena reversible antes del
marcador de commit. La compresión y poda genéricas nunca seleccionan checkpoints.

## API de objetos y CLI

```python
from lambdaforge import Experiment, LambdaForge

experiment = Experiment.from_yaml("experiment.yaml")
plan = experiment.preview_retention()  # estrictamente solo lectura
print(plan.status, plan.operations)

result = experiment.apply_retention()  # mutación explícita
print(result.status, result.reclaimed_bytes, result.archives)

same_plan = LambdaForge.preview_retention("experiment.yaml")
```

Los objetos tipados `ArtifactRetentionPlan` y `ArtifactRetentionResult` conservan compatibilidad
mapping/JSON, rechazan mutación y exponen estados enum estables.

```powershell
lambdaforge retain experiment.yaml
lambdaforge retain experiment.yaml --json
lambdaforge retain experiment.yaml --apply
lambdaforge retain experiment.yaml --apply --json
```

El comando solo previsualiza salvo que aparezca `--apply`. Preview devuelve 0 únicamente para un
plan preparado. Apply devuelve 0 para `applied` o `already_applied`; resultados not-ready,
conflict y partial devuelven 1. Los errores de sintaxis devuelven 2. Los fallos de carga,
validación, locking o transacción se presentan como error estable; con `--json` la salida estándar
sigue siendo un único objeto JSON.

## Transacción y recuperación

El entrenamiento posee un lock exclusivo de actividad mientras hay workers activos. La agregación
final posee una lease compartida de actividad y un lock exclusivo de agregación. La retención toma,
en ese orden, los locks exclusivos de actividad, agregación y transacción; dos procesos de
LambdaForge no pueden entrenar, publicar agregados o podar a la vez la misma suite. Los locks del
sistema operativo se liberan tras una salida normal o abrupta y nunca se heredan handles vivos a
procesos spawned.

La aplicación utiliza un journal durable:

1. escribir y sincronizar un journal `prepared`;
2. transmitir, cerrar, sincronizar y verificar ZIPs inmutables;
3. revalidar el fingerprint de cada fuente;
4. renombrar atómicamente las fuentes a una cuarentena local a la suite;
5. publicar el marcador `committing`;
6. actualizar metadatos de checkpoints, purgar cuarentena y publicar resultados inmutables/latest;
7. actualizar `summary.json`, refrescar el receipt y retirar el journal.

Un reinicio antes de `committing` restaura las fuentes en cuarentena y elimina los ZIPs de la
transacción. Un reinicio posterior termina hacia delante. Copias conflictivas, journals ilegibles,
archivos sustituidos y rutas inseguras se preservan y se informan; LambdaForge nunca adivina qué
copia eliminar. Un plan revertido se puede reintentar. Reaplicar un plan comprometido devuelve
`already_applied` sin crear un segundo ZIP.

## Artefactos producidos

```text
<suite>/
├── .lambdaforge/
│   ├── activity.lock
│   ├── aggregation.lock
│   └── retention.lock
├── <variant>/seed=<seed>/
│   └── .lambdaforge/retention/
│       └── artifacts-l<level>-<plan-prefix>.zip
└── aggregate/
    ├── aggregation_receipt.json
    ├── retention/
    │   ├── <plan-id>.json
    │   └── latest.json
    └── summary.json
```

`summary.json` empieza con `status: not_applied` y manifiesto nulo. Solo una transacción
comprometida lo actualiza atómicamente con el manifiesto real, estado e identificador de plan. Los
resultados enumeran cada operación, estado, bytes seleccionados/recuperados, hashes de archivos,
avisos y errores. El historial de rollback usa resultados inmutables con sufijo de estado y no
bloquea un commit posterior.

## Límites

- La retención trabaja actualmente con sistemas de archivos locales y archivos regulares. Los
  backends remotos/object-store necesitan contratos propios de atomicidad y leases.
- ZIP/Deflate es el único codec de archivo. Los checkpoints se podan o conservan, nunca se
  comprimen.
- Los archivos abiertos/solo lectura pueden provocar rollback si la plataforma impide
  renombrarlos o eliminarlos.
- Preview no toma locks deliberadamente y puede quedar obsoleto; apply reconstruye y revalida su
  plan bajo locks.
- El YAML sigue siendo configuración de confianza. Los patrones no pueden escapar de un run, pero
  los targets Python y plugins configurados mantienen la frontera más amplia de código de
  confianza.
