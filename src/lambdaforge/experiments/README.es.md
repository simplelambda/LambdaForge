# Sistema de experimentos de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

Este paquete convierte un YAML de confianza en ejecuciones reproducibles por variante y semilla, las
planifica y reduce sus resultados en disco. No contiene lógica específica de modelos o datasets.

## Contenidos

- [Empieza aquí](#empieza-aquí)
- [Objetos principales](#objetos-principales)
- [Ciclo de vida](#ciclo-de-vida)
- [Migraciones de configuración](#migraciones-de-configuración)
- [Reglas de expansión](#reglas-de-expansión)
- [Ejecución](#ejecución)
- [Finalización y reanudación](#finalización-y-reanudación)
- [Resultados tipados](#resultados-tipados)
- [Historial de intentos y selección de resultados](#historial-de-intentos-y-selección-de-resultados)
- [Artefactos y agregación](#artefactos-y-agregación)
- [Retención de artefactos](#retención-de-artefactos)
- [Comparaciones estadísticas](#comparaciones-estadísticas)
- [Carga](#carga)
- [Especificaciones de plugins](#especificaciones-de-plugins)
- [Fronteras de extensión](#fronteras-de-extensión)

## Empieza aquí

Un **experimento** es una pregunta científica descrita por una configuración. LambdaForge la
expande en una **suite** de runs. Una **variante** es una combinación de hiperparámetros/ablación;
una **seed**, una repetición de esa variante; un **run**, su ejecución concreta. Si se reintenta el
mismo run, cada ejecución es un **intento** y se conserva la metadata terminal anterior.

Para el uso normal comienza con `Experiment.from_yaml()` o los comandos `lambdaforge`. La tabla de
objetos explica responsabilidades internas para quien mantenga el framework; el usuario no debe
instanciar todas las clases. El ciclo seguro es validar → inspeccionar → dry-run → ejecutar → auditar
resultados → agregar. Sólo `run` entrena y la retención sólo modifica artefactos al aplicar un plan
de forma explícita.

## Objetos principales

| Objeto | Responsabilidad |
|---|---|
| `Experiment` | Fachada pública para expandir, ejecutar, agregar, retener y cargar. |
| `ExperimentConfig` | Carga YAML, rutas con puntos, validación y expansión. |
| `ExperimentValidator` | Validación de Schema, expansión, recursos e imports sin efectos laterales. |
| `ValidationReport` | Resultado inmutable legible por personas o como JSON. |
| `ExperimentConfigMigrator` | Planificación, aplicación y validación de migraciones exactas hacia delante sin imports del usuario. |
| `ExperimentConfigMigrationResult` | Preview inmutable diff/YAML/JSON y persistencia YAML atómica explícita. |
| `ExperimentConfigMigrationRegistry` | Registro inmutable de cadenas de migración deterministas. |
| `ExperimentSchemaVersion` / `ExperimentSchemaCatalog` | Valor de versión exacto y selección de Schema empaquetado. |
| `ObjectFactory` | Resolución recursiva de especificaciones `target`, `ref` y plugins instalados. |
| `ExecutionConfig` | Validación de recursos y creación de slots GPU lógicos. |
| `ExperimentExecutor` | Selección de ejecución secuencial, paralela o DDP. |
| `ExperimentRunner` | Materialización y ejecución de una configuración y su resultado. |
| `ExperimentAggregator` | Lectura de ejecuciones y creación de estadísticas, CSV y gráficas. |
| `ArtifactRetentionPolicy` | Validación de la política estricta Schema 1.1 de checkpoints/archivos/poda. |
| `ArtifactRetentionManager` | Previsualización o aplicación de retención bajo locks ordenados de la suite. |
| `ArtifactRetentionPlan` / `ArtifactRetentionResult` | Plan y resultado de transacción inmutables y compatibles con mapping/JSON. |
| `RunLoader` | Localización de ejecuciones y reconstrucción de modelos. |
| `RunResult` | Resultado terminal inmutable, tipado y compatible con JSON. |
| `RunFingerprint` | Identidad canónica de la configuración científica expandida. |
| `ResultCatalog` / `ResultRecord` | Descubrimiento, auditoría y selección explícita de intentos actuales o archivados. |
| `AggregateResult` | Mapping inmutable de agregados tipados por variante. |
| `VariantAggregateResult` | Acceso tipado al estado, conteos y métricas de una variante. |
| `StatisticalComparisonConfig` | Validación y materialización del protocolo anidado. |
| `ConfidenceIntervalResult` | Estimación, estado y metadatos de reproducibilidad inmutables. |
| `PairedTestResult` | P-valores, rangos y conteos efectivos inmutables. |

Clases auxiliares como `ExperimentWorker`, `StdIOCapture`, `TeeStream`, `CheckpointChoice` y los
enums de estado viven también cada una en su módulo.

## Ciclo de vida

```text
YAML → ExperimentConfigMigrator → configuración Schema actual + metadata de migración
     → ExperimentValidator → ValidationReport
     → ExperimentConfig → configuraciones por variante/semilla
     → ExecutionConfig → slots de procesos/dispositivos
     → ExperimentRunner → config, entorno, log, métricas, checkpoints, resultado
     → ExperimentAggregator → tablas, gráficas y receipt de finalización
     → ResultCatalog → auditoría actual/histórica y selección explícita de intento
     → ArtifactRetentionManager → plan de solo lectura o transacción habilitada
     → RunLoader → modelo reconstruido
```

Usa el objeto de alto nivel salvo al escribir una integración:

```python
from lambdaforge import Experiment

experiment = Experiment.from_yaml("experiment.yaml")
report = experiment.validate()
for run in experiment.expand():
    print(run["experiment"]["variant"], run["experiment"]["seed"])
results = experiment.run(dry_run=True)
print(results[0].status, results[0]["status"])
```

`lambdaforge validate experiment.yaml` realiza la misma validación. `--json` emite un informe estable
y `--no-imports` permite revisar una plantilla antes de instalar su proyecto externo o sus plugins.
Esta opción omite la carga de `target`, `ref` y entry points. No se instancia ningún objeto ni se
crea un directorio de ejecución; la comprobación normal de imports puede ejecutar código de nivel
superior del módulo, por lo que solo deben revisarse configuraciones de confianza.

## Migraciones de configuración

El Schema 1.1 exige la declaración exacta y entrecomillada `schema_version: "1.1"`. El Schema 1.0
sigue empaquetado para validar exactamente configuraciones históricas. Un YAML sin versión se
reconoce como `unversioned` y sigue la ruta determinista `unversioned -> 1.0 -> 1.1`.
`UnversionedToV1Migration` declara 1.0 sin cambiar la semántica; `ExperimentV1ToV1_1Migration`
avanza ese mapping válido a 1.1, cuyo bloque opcional de retención está desactivado por defecto.
`ExperimentConfig` aplica la cadena completa en memoria en las fronteras normales de carga, por lo
que los orígenes antiguos siguen pudiendo expandirse, ejecutarse, agregarse y recargarse sin ser
editados. Las configuraciones expandidas y materializadas emplean la versión canónica actual.

Usa el comando específico para inspeccionar o persistir el cambio:

```powershell
lambdaforge migrate legacy.yaml                   # diff unificado por defecto
lambdaforge migrate legacy.yaml --format yaml
lambdaforge migrate legacy.yaml --format json
lambdaforge migrate legacy.yaml --check           # 1 si está obsoleto; 0 si está actualizado
lambdaforge migrate legacy.yaml --target-version 1.1
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml
```

`--target-version` usa por defecto el Schema empaquetado actual. `--output` debe ser distinto del
origen y siempre recibe YAML completo; un destino existente necesita `--force`. Incluso con
`--force` se rechaza el origen. `--check` nunca escribe y no se puede combinar con `--output`. Una
previsualización/escritura normal correcta devuelve 0, los errores de migración o salida devuelven
1 y los errores de sintaxis devuelven 2. En modo check, 1 significa específicamente que hace falta
al menos un paso válido.

La ruta de migración conserva presentación y prioriza la previsualización. Rechaza claves YAML
duplicadas, no toca el mapping del llamador, conserva sus tipos Python programáticos y mantiene
comentarios/orden/comillas/anchors/saltos de línea de los archivos cuando es posible. Valida cada
resultado contra el Schema de destino exacto. No llama a `ObjectFactory`, resuelve plugins ni
importa objetos `target`/`ref` configurados. La escritura usa un temporal sincronizado, publicación
atómica sin sobrescritura por defecto y sustitución atómica solo con `--force`; no existe modo
in-place.

```python
from lambdaforge import LambdaForge
from lambdaforge.experiments import ExperimentConfigMigrator, MigrationPreviewFormat

preview = LambdaForge.preview_migration("legacy.yaml")
print(preview.render(MigrationPreviewFormat.DIFF))

mapping_preview = ExperimentConfigMigrator.default().preview_mapping(raw_config)
assert mapping_preview.target_version.value == "1.1"
```

`ValidationReport` expone versiones de origen/destino y descriptores de los pasos aplicados. La
validación del Schema de migración excluye imports deliberadamente; ejecuta después
`lambdaforge validate` para comprobar expansión, recursos e imports opcionales. El registro actual
contiene la ruta exacta `unversioned -> 1.0 -> 1.1` y no admite downgrade, sobrescritura del origen,
versiones inferidas ni saltos no registrados.

Consulta la [guía completa de migraciones](migrations/README.es.md) para los sobres de preview,
códigos de salida, garantías de persistencia, objetos públicos, modos de fallo y el proceso revisado
para añadir un futuro Schema y su objeto de migración consecutivo.

## Reglas de expansión

`experiment.seeds` acepta escalar o lista. `sweep.grid` relaciona rutas con puntos con listas no
vacías y forma su producto cartesiano. `sweep.include_base` decide si se incluye la configuración
original. Cada elemento de `sweep.ablations` añade overrides con puntos y nombre propio.

Para búsqueda dinámica multi-fidelidad usa la ruta excluyente `hpo.enabled: true`. Materializa runs
ordinarios incrementalmente, reanuda presupuestos acumulados desde el último checkpoint y guarda
estado/eventos bajo `.lambdaforge/adaptive/`. El contrato completo de YAML, recursos, semillas,
pruning, recuperación y personalización está en “Optimización adaptativa de experimentos” del README
principal y en `examples/adaptive-hpo.yaml`.

Se usan copias profundas: una ejecución no puede mutar otra. El nombre no puede estar vacío y las
identidades `(variant, seed)` finales han de ser únicas. `lambdaforge inspect` imprime las
configuraciones concretas sin ejecutar los objetos importados.

## Ejecución

`sequential` permanece en el llamador. `parallel` planifica cada ejecución independiente como proceso
`spawn` en slots fijos de una GPU. `ddp` asigna cada ejecución a un grupo de `devices_per_job` GPU y
configura Lightning para DDP. Los overrides CLI tienen precedencia sobre YAML y YAML sobre defaults.

El ejecutor usa objetos worker serializables y el método `spawn`. Los índices GPU son lógicos
respecto a `CUDA_VISIBLE_DEVICES` del padre; los límites de CPU/hilos/workers se aplican a cada
ejecución sin mutar el entorno padre.

Consulta la [guía de procesos](../training/README.es.md) para garantías y límites de apagado.

## Finalización y reanudación

Los estados legibles por máquina son `ok`, `failed`, `dry_run`, `interrupted` y `unknown`.

Una ejecución solo está completa cuando:

1. `result.json` tiene estado `ok`;
2. su fingerprint científico guardado o recuperado del YAML materializado coincide con el config;
3. existe el checkpoint seleccionado si la política lo exige; y
4. existen dentro del directorio todas las rutas de `experiment.required_artifacts`.

Con `rerun_completed: false` se omiten las completas. Con `resume: true`, una incompleta usa el
último checkpoint válido si existe. Los fallos producen un resultado terminal y la suite puede
relanzarse sin descartar semillas correctas. Los artefactos requeridos son rutas relativas definidas
por el proyecto.

Una parada cooperativa persiste `interrupted`, omite `test_after_fit` y sigue siendo reintentable; no
es un estado terminal correcto para agregación o retención. Antes de un reintento real, el resultado
anterior se mueve al historial interno de intentos para que un artefacto `ok` obsoleto no convierta
en correcto un intento que terminó con crash.
Una ejecución incompleta sólo reanuda un checkpoint si coincide esa identidad; cambiar modelo,
datos, loss, métrica, optimizador, trainer o extensiones inicia limpiamente en vez de cargar pesos
incompatibles.

## Resultados tipados

`Experiment.run`, `LambdaForge.run`, `ExperimentRunner` y `ExperimentExecutor` devuelven objetos
`RunResult`. Conservan el contrato histórico de diccionario—incluso `json.dumps(result)` directo—y
añaden atributos tipados como `status: RunStatus`, `seconds`, rutas de checkpoints y mappings de
métricas. `result_version` versiona este sobre JSON independientemente del Schema YAML. La versión 2
registra además `attempt_id`, `config_fingerprint`, `started_at_utc` y `finished_at_utc`.

```python
import json
from lambdaforge import RunResult

result: RunResult = experiment.run()[0]
assert result["status"] == result.status.value
payload = result.to_dict()
same = RunResult.from_mapping(payload)
json_text = json.dumps(same)
```

Los resultados rechazan mutaciones de claves y atributos. `with_updates(...)` crea otro objeto, los
campos JSON desconocidos sobreviven al round trip y se siguen leyendo `result.json` antiguos sin
`result_version`. `write_json` usa un temporal en el mismo directorio, lo sincroniza y sustituye el
destino atómicamente para que la agregación concurrente nunca vea un objeto parcial.

`Experiment.aggregate` y `ExperimentAggregator.write` devuelven `AggregateResult`; el acceso legado
`aggregates["base"]["metrics"]` no cambia. `aggregates.variant("base")` produce un
`VariantAggregateResult` con propiedades tipadas `complete`, `terminal`, `expected_runs`,
`completed_runs` y `metrics`. `AggregateResult.read_json` acepta tanto el mapa histórico de
variantes como el sobre completo de `summary.json`.

## Historial de intentos y selección de resultados

`RunFingerprint` resume una configuración expandida normalizada. Excluye deliberadamente nombres,
rutas de salida, controles de checkpoint/reintento, planificación, agregación, retención y metadata
descriptiva: modifican cómo se opera el intento, no el cálculo científico. Semilla expandida, datos,
modelo, losses, métricas, optimizador, scheduler, task, trainer, runner, callbacks y `extensions` sí
forman parte de la identidad. Cambiarlos impide reutilizar una finalización obsoleta.

Antes de que un reintento sustituya el marcador terminal canónico, LambdaForge lo archiva en
`<run>/.lambdaforge/attempts/result-<attempt-id>.json`. `result.json` representa el intento actual;
el archivo interno conserva historial inmutable. Checkpoints y artefactos mantienen sus rutas
normales, por lo que la metadata terminal archivada no es una copia recargable e independiente de
sus checkpoints.

Usa el catálogo y no un glob sobre `latest`:

```python
experiment = Experiment.from_yaml("experiment.yaml")
records = experiment.results(status="ok")
catalog = experiment.result_catalog()
duplicates = catalog.duplicate_groups()
ambiguous = catalog.ambiguous_successes()
selected = catalog.select(attempt_id="20260722T120000000000Z-a1b2c3d4e5f6-acde1234")
index_path = catalog.write_index()
```

Cada llamada a `records()` escanea el sistema de archivos actual; `result-index.json` sólo es un
snapshot atómico de intercambio. `duplicate_groups()` incluye reintentos con cualquier estado.
`ambiguous_successes()` identifica una identidad científica con varios éxitos, donde un artículo o
proceso posterior debe registrar un `attempt_id` explícito en vez de dejar que decida el orden de
directorios.

```bash
lambdaforge results experiment.yaml
lambdaforge results runs/experiments/study --status ok --no-archived
lambdaforge results experiment.yaml --duplicates --json
lambdaforge results experiment.yaml --write-index --fail-on-ambiguous
```

La última opción devuelve código 2 ante ambigüedad y está pensada para CI/publicación. Discovery
ignora archivos de resultado malformados en vez de tratarlos como correctos. El catálogo es metadata
de sistema de archivos local; un object store remoto requiere una capa explícita de
sincronización/inventario.

## Artefactos y agregación

El directorio contiene `config.yaml` materializado, procedencia tipada en `environment.json`,
`hparams.json`, `train.log`, `metrics.csv`, checkpoints, `result.json` y artefactos propios. El
manifiesto guarda Python/plataforma, versiones principales, CUDA/cuDNN, propiedades de GPU visibles y
commit/rama/estado sucio de Git si están disponibles, además de una lista `plugins` determinista.
Cada entrada contiene `kind`, `name`, `group`, `value` del entry point, distribución y versión. Solo
aparecen entry points resueltos correctamente por esa ejecución: no cuentan validaciones anteriores,
plugins instalados pero no usados ni imports `target`/`ref` normales. Los dry-runs escriben una lista
vacía. El manifiesto se sustituye atómicamente antes del entrenamiento y al salir, incluso con
excepción, para conservar resoluciones alcanzadas antes de un fallo de constructor o entrenamiento.
Las rutas se derivan del nombre, slug de variante y semilla.

`ExperimentAggregator.write` reconstruye informes desde disco: resúmenes terminales y por época,
CSV anchos/largos, estadísticas de semillas, pruebas direccionales por pares, q-valores
Benjamini-Hochberg y gráficas opcionales. Un fallo de Matplotlib queda registrado sin perder las
tablas numéricas. `lambdaforge aggregate --no-plots` sirve para entornos mínimos sin interfaz.

Una agregación final publica `aggregate/aggregation_receipt.json` en último lugar y solo cuando
cada run esperado es `ok`, toda variante está completa y terminal, las entradas requeridas/base son
seguras y coinciden los fingerprints comprometidos de runs y agregados. La agregación incremental
invalida un receipt anterior y nunca puede activar retención.

Las estadísticas son exploratorias. Se informan tamaños muestrales y variantes incompletas para
hacer visibles las semillas ausentes; las decisiones inferenciales corresponden a cada estudio.

## Retención de artefactos

El Schema 1.1 añade una política estricta opcional:

```yaml
retention:
  mode: preview                 # disabled, preview, apply
  checkpoints:
    keep: last_and_best         # all, best, last, last_and_best
    prune_unselected: true
  protect: [reports/**]
  rules:
    - action: compress
      include: [artifacts/intermediate/**]
      exclude: []
      min_size_bytes: 1048576
      compression: {only_if_smaller: true}
    - action: prune
      include: [scratch/**]
      exclude: []
      min_size_bytes: 0
  archive: {name: artifacts.zip, compression_level: 6}
  lock_timeout_seconds: 60
```

Omitirla equivale a `disabled`. `preview` es de solo lectura; `apply` permite la ejecución automática
solo después de una agregación final correcta. Aplicarla manualmente sigue siendo una petición de
mutación explícita:

```python
plan = experiment.preview_retention()
result = experiment.apply_retention()
```

```powershell
lambdaforge retain experiment.yaml
lambdaforge retain experiment.yaml --json
lambdaforge retain experiment.yaml --apply
```

Aplicar exige un receipt de finalización vigente y vuelve a planificar bajo locks. Quedan protegidos
los archivos base del run, `required_artifacts`, globs protegidos, agregados, checkpoints
seleccionados por reglas genéricas, enlaces/reparse points y metadata interna de la transacción. Los
checkpoints usan una política aparte consciente de roles; si falta un rol o es ambiguo, se omite la
poda de ese run en vez de adivinar.

La compresión transmite a ZIPs inmutables por run y verifica nombres, CRC, tamaños y SHA-256 antes
de mover originales a cuarentena reversible. `only_if_smaller` conserva fuentes no compresibles. El
journal durable hace rollback antes de su marcador de commit y termina hacia delante después; un
plan comprometido es idempotente. Entrenamiento, agregación final y retención se coordinan mediante
locks cross-process de actividad, agregación y retención en orden fijo.

Consulta la [guía de retención de artefactos](retention/README.es.md) para el contrato completo de
elegibilidad, reglas de protección, estados tipados, artefactos de transacción, recuperación tras
crash y límites del sistema de archivos local.

## Comparaciones estadísticas

`ExperimentAggregator` empareja cada variante con su baseline usando únicamente semillas con valor
en ambos lados. `delta` es `variante - baseline`; `improvement` coincide con ese delta para
métricas `max` y cambia de signo para métricas `min`. Por tanto, todas las alternativas comparten
una convención estable: positivo significa que la variante es mejor.

Selecciona el protocolo inferencial explícitamente bajo `aggregation.comparisons`:

```yaml
aggregation:
  comparisons:
    alpha: 0.05
    target_power: 0.80
    min_pairs_for_verdict: 3
    confidence_interval:
      method: bootstrap_percentile
      confidence_level: 0.95
      resamples: 10000
      seed: 0
      batch_size: 1024
      max_batch_elements: 1000000
    paired_test:
      method: wilcoxon
      alternative: two_sided
      calculation: auto
      zero_method: wilcox
      continuity_correction: false
      exact_max_pairs: 50
      zero_tolerance: 1.0e-12
      round_decimals: 12
```

Cada mapping anidado rechaza claves desconocidas. Si falta todo el bloque, se conserva el
comportamiento de la versión 3: `normal` al 95 %, `sign` exacta, `observed_direction`,
`alpha = 0.05`, `target_power = 0.80` y `min_pairs_for_verdict = 3`. El p-valor seleccionado
controla tanto la corrección Benjamini-Hochberg como el veredicto de la comparación.

`bootstrap_percentile` estima la media pareada remuestreando con reemplazo. Una semilla efectiva
derivada con SHA-256 combina la semilla base con
`(baseline_variant, variant, metric)`, de modo que reordenar métricas no perturba una comparación
existente. La generación PCG64 se hace por lotes: `batch_size` es el máximo solicitado y
`max_batch_elements` limita los índices transitorios. Solo se conserva una media por remuestreo
(`O(resamples)`); con menos de dos pares se devuelve metadata no disponible explícita y las
muestras constantes se marcan como degeneradas.

`wilcoxon` ofrece `two_sided`, `greater`, `less` y `observed_direction`. `auto` usa enumeración
exacta condicionada de signos mientras el número de pares no nulos no supere
`exact_max_pairs` y después usa aproximación normal; pedir `exact` por encima del límite produce
un resultado no disponible en vez de cambiar de método en silencio. `wilcox` elimina ceros antes
de asignar rangos, `pratt` los incluye al asignarlos pero excluye sus rangos de la suma aleatoria de
signos, y `zsplit` reparte además su contribución entre las estadísticas positiva y negativa
informadas. Los rangos promedio hacen deterministas los empates. El redondeo opcional se aplica
antes de detectar ceros y ordenar rangos.

`baseline_comparisons.csv` contiene campos de intervalo/prueba neutrales respecto al método,
p-valores seleccionados y diagnósticos, tamaños efectivos, datos de rangos/ceros y procedencia de
semillas bootstrap. `reliability.json` incluye el `statistical_protocol` completamente materializado
y todas las comparaciones; `summary.json` registra el protocolo y las rutas de artefactos. La
agregación versión 4 sigue escribiendo columnas históricas del intervalo normal al 95 % y de la
prueba de signos, de forma que los consumidores puedan migrar por separado, mientras
`p_value_directional`, su q-valor BH y el veredicto usan el método elegido.

La configuración, enums y resultados públicos se exportan de forma lazy desde
`lambdaforge.experiments`. Las estrategias concretas y el motor de composición están disponibles en
`lambdaforge.experiments.statistics`:

```python
from lambdaforge.experiments import StatisticalComparisonConfig
from lambdaforge.experiments.statistics import StatisticalComparisonEngine

protocol = StatisticalComparisonConfig.from_mapping(config)
engine = StatisticalComparisonEngine(protocol)
interval = engine.confidence_interval(
    [0.02, 0.01, 0.03],
    identity=("base", "candidate", "val_auroc"),
)
test = engine.paired_test([0.02, 0.01, 0.03])
```

Consulta la [guía del paquete estadístico](statistics/README.es.md) para todos los valores, defaults,
campos de resultado, casos límite y objetos de API.

## Carga

```python
experiment = Experiment.from_yaml("experiment.yaml")
model = experiment.load_model(seed=7, variant="base", which="auto")
```

`CheckpointChoice` ofrece `best`, `last` y `auto`. `AUTO` resuelve best, después last y finalmente el
checkpoint local seguro más reciente; `BEST` y `LAST` exactos nunca cambian silenciosamente de rol.
`RunLoader` valida la ejecución, importa el modelo desde su especificación materializada, carga el
estado directo o elimina el prefijo Lightning `model.`, y devuelve el modelo en evaluación.

`RunLoader.load_result(run_dir)` lee el artefacto terminal como `RunResult`.

## Especificaciones de plugins

Los modelos y métricas instalados se pueden seleccionar mediante una identidad explícita de entry
point en lugar de una ruta de importación:

```yaml
model:
  plugin:
    kind: model
    name: acme_encoder
  params:
    in_features: 32

val_metrics:
  - plugin:
      kind: metric
      name: calibrated_auc
    params:
      output_key: logits

data:
  train:
    plugin: {kind: dataset, name: acme_records}
    params: {split: train}

callbacks:
  - plugin: {kind: callback, name: artifact_marker}

trainer:
  logger:
    plugin: {kind: logger, name: jsonl_logger}
    params: {path: metrics.jsonl}
```

`plugin` contiene únicamente `kind` y `name`; los `params` del constructor son su clave hermana y se
construyen recursivamente. El Schema restringe la posición del modelo a `kind: model` y las listas de
métricas a `kind: metric`, pérdidas a `kind: loss`, splits de datos a `kind: dataset`, callbacks a
`kind: callback` y loggers del trainer a `kind: logger`. Las especificaciones `target` completas
siguen siendo válidas y pueden coexistir con plugins. Las listas de logger pueden combinar objetos,
referencias y plugins.

La validación con comprobación de imports resuelve la clase y verifica su contrato sin instanciarla.
Esa validación no se informa como uso de una ejecución. Durante la ejecución, un contexto por run
captura objetos plugin explícitos y aliases resueltos dentro de constructores de modelos. Los runs
secuenciales y procesos de entrenamiento `spawn` tienen procedencia independiente; un acierto de
caché correcto sí cuenta para el run actual. `--no-imports` valida la forma, pero deliberadamente no
exige que la distribución externa esté instalada. Consulta la
[guía de plugins](../plugins/README.es.md#procedencia-de-plugins-cargados) para el contrato exacto del
artefacto, publicar grupos, descubrir por CLI, entender la caché, los conflictos y la frontera de
código de confianza.

## Fronteras de extensión

- Configura `data.datamodule.target`, `task.target` o `runner.target` propios en vez de bifurcar el
  motor de experimentos.
- Los modelos, pérdidas y `train_metrics`, `val_metrics` y `test_metrics` aceptan plugins instalados
  o especificaciones `target` completas; los valores `ref` y objetos anidados se construyen
  recursivamente. La clave retrocompatible `metrics` completa las etapas no indicadas.
- Los splits de datos, `callbacks` superiores y loggers del trainer admiten plugins instalados
  reutilizables u objetos `target` locales. Se conservan listas de logger y objetos dataset `ref`;
  collators y otros objetos anidados usan la misma sintaxis recursiva.
- Los monitores de checkpoint y parada temprana y sus modos `min`/`max` son ajustes explícitos del
  trainer; si se omiten se usa la primera métrica de validación y su dirección declarada.
- Un runner propio debe conservar métodos `fit` y `test` compatibles con `ExperimentRunner`.
- Trata YAML como código de confianza: los targets importados y plugins resueltos pueden ejecutar
  Python arbitrario.
- La procedencia cubre el proceso del run. Las resoluciones hechas solo en procesos hijo creados por
  el usuario requieren una integración IPC explícita si se deben atribuir al padre.
- Importa clases públicas desde `lambdaforge.experiments`; los archivos son detalles internos.
- Añade los cambios incompatibles futuros como Schemas empaquetados más objetos
  `ExperimentConfigMigration` consecutivos; no infieras versiones a partir de la forma del documento.
- Mantén la limpieza de artefactos tras `ArtifactRetentionPolicy`/`ArtifactRetentionManager` para
  conservar la validación del receipt, guards de rutas, orden de locks y recuperación; un agregador
  personalizado no debería borrar salidas de runs.

Las clases del ciclo de vida permanecen juntas deliberadamente: a diferencia de métricas y
callbacks, sus contratos están muy acoplados y separarlas crearía varios paquetes diminutos sin un
propósito público independiente. Esta frontera se debería revisar cuando aparezca una familia real
de backends o almacenamiento.
