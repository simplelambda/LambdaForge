# Manual de agentes de LambdaForge

[English](AGENTS.md) | [Español](AGENTS.es.md)

Este es el punto de entrada operativo único para un agente. No recorras todo el repositorio ni todos
los README. Usa las rutas documentales del final sólo cuando este fichero no contenga el detalle;
después inspecciona la firma/docstring del símbolo público concreto.

## Qué es y cómo instalarlo

LambdaForge 0.5.2 es una biblioteca instalable PyTorch/Lightning para tasks genéricas,
preprocesado, training, workflows, sweeps/HPO, ejecución CPU/GPU/SLURM, resultados, plots,
artifacts, reproducibilidad y control explícito de clusters. El proyecto consumidor posee modelos,
datasets y código de dominio. Python >=3.10. Importa sólo desde estos namespaces públicos:
`lambdaforge.configuration`, `lambdaforge.controlplane`, `lambdaforge.data`,
`lambdaforge.execution`, `lambdaforge.experiments`, `lambdaforge.hpo`,
`lambdaforge.integrations`, `lambdaforge.results`, `lambdaforge.visualization`,
`lambdaforge.artifacts`, `lambdaforge.metrics`, `lambdaforge.nn`,
`lambdaforge.observability`, `lambdaforge.operations`, `lambdaforge.plugins`,
`lambdaforge.preprocessing`, `lambdaforge.registry`, `lambdaforge.reproducibility`,
`lambdaforge.storage`, `lambdaforge.tasks`, `lambdaforge.tracking`,
`lambdaforge.training` y `lambdaforge.workflows`.

Nunca copies `src/lambdaforge`, compartas `.venv` ni modifiques `PYTHONPATH`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e /ruta/absoluta/LambdaForge
python -m pip install -e .
python -m pip check
python -c "import lambdaforge; print(lambdaforge.__version__)"
```

Para release reproducible construye/instala `lambdaforge-0.5.2-py3-none-any.whl`. El proyecto
consumidor fija la wheel correcta de PyTorch. `nvidia-smi` sólo prueba driver; comprueba
`torch.cuda.is_available()` y `torch.version.cuda`. Extras: `hpo`, `adaptive-hpo`, `s3`, `parquet`,
`onnx`, `viz`, `graph`, `viz3d`, tracking y `dev`.

## Ruta rápida

| Necesidad | Acción |
|---|---|
| Validar/expandir/ejecutar | `validate`, `inspect`, `run --dry-run`, `run` |
| Ver YAML estricto | `inspect CONFIG --resolved` |
| Preprocesar | empieza con `examples/preprocessing-simple.yaml` |
| Debug de N registros | `debug CONFIG --records N` |
| Workflow local | `kind: workflow`; validate/inspect/run |
| Dataset lógico | `data_catalog` + `dataset:NAME/subpath` |
| Cluster | `clusters add`; `doctor --on`; `clusters bootstrap`; `run --on` |
| Reconectar job | `status`; `logs JOB --follow`; `cancel`; `retry` |
| Resultados | `results list/show/compare/export`; auditoría legacy sigue válida |
| Curvas/sweeps | `plot learning/sweep/seeds/hpo/resources` |
| Artifact | `artifact inspect/export/validate/visualize/list/fetch/plugins` |
| Remoto ligero | `results sync JOB`; luego fetch explícito de artifact pesado |
| HPO adaptativo | `hpo.enabled: true`; `examples/adaptive-hpo.yaml` |
| Inferencia/eval/export | tasks de `lambdaforge.operations` |
| Limpiar artifacts | `retain` preview; `retain --apply` sólo tras revisar |

En `resources` se aceptan `cpu`/`cpus`, `ram`/`memory`, `gpu`/`gpus`, `gpu_memory`, `storage`,
`time` y `processes`. En comparaciones declara `--direction` para etiquetar best/worst. En sweeps
multimétrica, `--normalize` hace min-max explícito por métrica y conserva los valores raw.

## Autoría, YAML y objetos

`AuthoringConfig` compila formas sencillas al IR estricto: task 1.0, experiment 1.1 o workflow 1.0.
En training acepta `name`, `model`, una `loss`, `trainer.epochs`, `resources`, `data_catalog` y
`environment`. La sintaxis recursiva es siempre:

- `target: package.module.Class` construye con `params`;
- `ref: package.module.object` importa sin construir salvo params;
- `plugin: {kind: model, name: published}` resuelve entry point;
- specs anidados se construyen recursivamente.

YAML es código confiable porque imports/plugins pueden ejecutar Python. Composición: `extends`,
`include`, hoja y overrides; mappings mezclan, listas reemplazan y `{$delete: true}` elimina.
Interpolación sólo `${config:path}`, `${env:NAME}` y valor completo `${secret:NAME}`. Secrets se
redactan; experiment/workflow durable rechaza secrets. Usa `compose`/`diff` para auditar.

Training mínimo:

```yaml
name: baseline
data_catalog: data-catalog.yaml
environment: local
data: {train: dataset:corpus/train, val: dataset:corpus/val}
model: my_project.models.Model
loss: torch.nn.BCEWithLogitsLoss
optimizer: {ref: torch.optim.AdamW, params: {lr: 0.001}}
trainer: {epochs: 20, accelerator: auto, devices: auto}
resources: {cpus: 8, memory: 32GiB, gpus: 1}
```

Una referencia directa necesita `loader` y `path_parameter` en el catálogo. En params anidados usa
exactamente `{dataset: corpus, subpath: train}`. Nunca adivines strings. El fingerprint sustituye
path físico por referencia+identity lógica; `environment`, path del catálogo y recursos operativos
no cambian ciencia.

## Tasks y preprocesado

Una task de proyecto implementa `run(context)` y devuelve `TaskOutput` con outputs, métricas y
`ArtifactDeclaration`. Inputs se hashean por contenido. Artifacts deben existir bajo el run, se
comprueban contra symlink/traversal y reciben SHA-256. Éxito sólo se omite si los bytes verifican;
un rerun archiva el resultado terminal.

Preprocesado usa `PreprocessingSource.records(context)` -> `PreprocessingRecord` con clave estable;
cada transform conserva la clave; sink escribe/verifica/finaliza. Manifest por registro permite
resume y shards SHA-256. `DatasetArtifact` contiene ID por contenido, muestras/splits, source,
fingerprint y artifacts.

Concurrencia: un worker es secuencial; `io` usa threads; `cpu` usa procesos `spawn` sólo para
transforms importables/picklables y el padre escribe sink/manifest; `auto` usa threads; `gpu` exige
uno. FAIL cancela/persiste; SKIP continúa. `debug --records N` no llama sink ni publica dataset y
usa identidad `debug:`.

## Experimentos, resultados y publicación

Expansión de seeds/grids/ablations es determinista. Modos: sequential, parallel (jobs independientes
por GPU) y DDP (un job sobre grupo de dispositivos). El task Lightning estándar enruta batches
mapping con `model_input_key`, lista posicional o mapping de kwargs. Resultado tensor se envuelve con
`model_output_key`; mapping se conserva.

Cada intento tiene attempt ID, tiempos y fingerprint. Configuración/entorno/métricas/checkpoints se
guardan por run; resultado anterior va a `.lambdaforge/attempts`. Nunca elijas por mtime, glob o
“latest”:

1. `results CONFIG --write-index --fail-on-ambiguous`.
2. Exige fingerprint, variante y seeds.
3. Si hay éxitos ambiguos, elige attempt ID y documenta por qué.
4. Lee paths/métricas del `ResultRecord`.
5. Agrega después de seleccionar.

`ResultService` acepta path, attempt, fingerprint y nombre humano; `show` enumera ambigüedad.
`MetricSeries` normaliza `metrics.csv` a run/seed/variant/split/metric/step/value/timestamp.
`VisualizationService` crea `PlotSpec`, escribe atómicamente PNG/SVG/PDF o HTML opcional y guarda
`.plot.json`. Con `n=1` no hay std/CI. Sweep agrega por seed y no interpola celdas ausentes salvo
opción explícita. HPO/resources sólo visualizan evidencia existente.

## Artifacts

Inspector, visualizer, schema y validator son contratos separados. NPY/NPZ usa
`allow_pickle=False`, preview <=1000 y estadística acotada/determinista; CSV/TSV/JSON/JSONL son
seguros. No cargues pickle genérico. Geometría sólo con roles explícitos nodes/edges/positions o
tipo mesh. Proyectos publican providers `lambdaforge.artifact_*`; convenciones de dominio no van al
core. `artifact fetch JOB NAME` debe seleccionar uno y permanecer bajo el work_dir remoto.

## Clusters y jobs

`ExecutionBundleBuilder` materializa YAML, selecciona ubicación de data del destino y construye
wheels exactas del framework/proyecto. `EnvironmentIdentity` incluye wheels, Python y wheelhouse.
`managed` crea venv idempotente bajo `WORKSPACE/.lambdaforge/environments`; no clona main ni instala
drivers/CUDA. `existing` sólo verifica el intérprete. Offline usa wheelhouse compatible y
`--no-index`; no sintetices wheels de otra plataforma.

`JobRecord` persiste scheduler/scientific/execution/bundle IDs, paths y tiempos; `status` refresca al
reiniciar el PC. `ClusterCatalog` fusiona usuario < proyecto < explícito y `clusters inspect` muestra
la fuente. OpenSSH es el default recomendado: conserva aliases/claves/agente/known_hosts/ProxyJump.
Contraseña opcional sólo puede venir de prompt oculto, referencia `keyring:` o `env:` mediante
`CredentialProvider`; nunca pongas el valor en argv/YAML/job/bundle/fingerprint/log.
`PasswordSshTransport` usa RejectPolicy/SFTP/timeouts. `SlurmProfile` centraliza mapping de recursos,
directivas, comandos argv/placeholders/regex y líneas de script confiables sin secrets; dry-run debe
mostrar directivas, avisos y comando. Lee `docs/CLUSTERS.es.md` y `docs/SECURITY.es.md` si modificas
esta frontera. `results sync` sólo trae evidencia pequeña. No hay placement automático, workflow
entre clusters, daemon, servidor o GUI en 0.5.2.

## HPO adaptativo

No mezclar con sweep. Space usa float/int/ordinal/categorical/bool y `when`; objective es columna
exacta de metrics.csv. START/RESUME/ADD_SEED compiten mediante KG gaussiano/coste/viabilidad;
CONFIRM es separado. Promoción es budget acumulado y necesita checkpoint last. Search seeds se
comparten; confirmation son disjuntos.

Estado/eventos/summary están en `SUITE/.lambdaforge/adaptive/STUDY_ID`; relanzar YAML reconcilia.
BoTorch opcional modela categorías/ordinales/condiciones/fidelidad/pending; fallo registra fallback a
Sobol. Memoria es UNKNOWN/UNBOUNDED/KNOWN, conserva OOM censurado y nunca reduce batch. Allocator cap
no es aislamiento. Para `n` seeds:

$$
\operatorname{Var}(\bar{\mu}) = \frac{\tau^2}{n} +
\frac{v_1 + \cdots + v_n}{n^2}.
$$

Consulta `docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.es.md` sólo al cambiar internals.

## Extensiones y modificación del repositorio

Un `nn.Module` sirve como modelo; `Model` añade predict, count/freeze y grupos. Loss custom hereda
`Loss`, devuelve tensor escalar, conserva grafo y aplica weight. Metric custom implementa
reset/update/compute y estado distribuido mergeable, separando tensors con detach. Dataset hereda
PyTorch Dataset. Callbacks/loggers usan bases Lightning exportadas. Para reutilización publica entry
points de model/loss/metric/dataset/callback/logger/component/task.

Al modificar:

- normalmente una clase/enum pública por módulo homónimo, validación de constructor, tipos,
  docstrings y valores inmutables;
- dependencias especializadas como extras/adaptadores lazy;
- compatibilidad YAML y actualización de Schema, ejemplos, exports, docs EN/ES y ambos AGENTS;
- tests focalizados de valor/shape/gradiente/fallo, import público y construcción YAML;
- no afirmar paridad sin referencia externa fijada;
- no añadir envs, `.env`, caches, `.lambdaforge`, runs, wheels, dashboards ni logs SLURM.

Antes de entregar:

```bash
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
.venv/bin/mypy src/lambdaforge
.venv/bin/pytest -q
python -m build
python -m twine check dist/*
```

Prueba además la wheel instalada desde fuera del source. Revisa `git status --short` e ignorados.

## Rutas documentales focalizadas

- arquitectura global: `docs/ARCHITECTURE.es.md`;
- clusters: `docs/CLUSTERS.es.md`;
- seguridad de credenciales/scheduler: `docs/SECURITY.es.md`;
- resultados/plots: `docs/RESULTS.es.md`;
- artifacts: `docs/ARTIFACTS.es.md`;
- preprocesado: `docs/PREPROCESSING.es.md`;
- HPO internals: `docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.es.md`;
- modelos/métricas/training: README EN/ES del namespace correspondiente.
