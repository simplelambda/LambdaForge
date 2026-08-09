# Arquitectura técnica de LambdaForge

Este documento explica cómo LambdaForge convierte configuración declarativa en trabajo reproducible,
qué clase es responsable de cada decisión y por qué las fronteras están separadas. No es un listado
de ficheros: describe colaboraciones, invariantes y puntos de extensión para mantener el proyecto.

## Objetivo y reglas de diseño

LambdaForge quiere ejecutar trabajo de IA sin apropiarse de la ciencia del consumidor. El framework
posee configuración, identidad, planificación, ejecución, procedencia, resultados y artefactos. El
proyecto consumidor posee datasets, modelos, transformaciones y reglas de dominio. Esta división
permite que una clase de investigación siga siendo Python normal e importable, mientras las partes
repetitivas conservan contratos estables.

Las reglas estructurales son:

- una clase pública por módulo del mismo nombre;
- objetos pequeños con una única razón de cambio;
- valores de configuración/plan/resultado inmutables cuando representan evidencia;
- fachada y reexports públicos; no se importa una ruta física privada desde un consumidor;
- construcción recursiva común mediante `ObjectFactory`, no factories distintas por subsistema;
- resultados de disco como fuente de verdad; registro, dashboard e informes son vistas de lectura;
- mutaciones destructivas siempre preview-first o explícitas;
- proveedores pesados detrás de adaptadores/inyección y dependencias opcionales.

La auditoría AST de 0.4 no encuentra nombres de clase pública duplicados ni módulos con varias clases
públicas. Todos los símbolos nuevos tienen uso interno o reexport público; no hay un segundo runner
de tareas, catálogo de resultados, factory de objetos o lock de ficheros.

## Flujo completo

```text
YAML confiable
  -> ConfigurationComposer (si hay composición)
  -> WorkflowConfig / TaskConfig / ExperimentConfig
  -> Validator + plan inmutable
  -> WorkflowRunner / TaskRunner / ExperimentRunner
  -> ObjectFactory + PluginRegistry
  -> código consumidor / Lightning
  -> TaskResult / RunResult + EnvironmentManifest + TaskArtifact
  -> ResultCatalog
  -> ExperimentRegistry -> Comparator -> ReportBuilder / LocalDashboard
  -> RetentionManager o ArtifactStore (mutación explícita)
```

Cada flecha atraviesa un contrato serializable o tipado. Esto hace posible validar o inspeccionar sin
construir código usuario, reanudar por identidad científica y analizar resultados sin entrenar.

## Configuración

`ConfigurationComposer` sólo resuelve composición. Procesa `extends`, después `include`, después el
documento hoja y por último overrides explícitos. Los mapas se fusionan de forma recursiva, las listas
se sustituyen y `{$delete: true}` elimina un valor heredado. Detecta ciclos por la pila de rutas
resueltas. No valida un experimento ni ejecuta targets porque esas responsabilidades pertenecen al
objeto de configuración y al validator de cada familia.

La interpolación admite `${config:ruta}`, `${env:NOMBRE}` y `${secret:NOMBRE}`. No existe `eval`,
Jinja ni expresión Python. `SecretValue` mantiene el valor separado de su representación;
`ResolvedConfiguration.materialized()` lo sustituye por `***` y revelar exige un argumento explícito.
Un workflow rechaza secretos estructurales porque persiste snapshots de sus nodos; el código
consumidor debe leer credenciales en ejecución. `ResolvedConfiguration.provenance` atribuye cada
ruta a su fichero u override. `ConfigurationDiff` aplana semánticamente mapas y reporta añadido,
eliminado o cambiado ignorando orden de claves.

`TaskConfig` y `ExperimentConfig` siguen separados porque sus Schemas e identidades son distintas.
Un experimento exige modelo/loss/trainer y se expande por semillas/variantes; una tarea arbitraria no
debe fingir esos conceptos. `WorkflowConfig` tampoco hereda de ellos: coordina documentos completos
pero no redefine su Schema.

## Construcción de objetos y plugins

`ObjectFactory` interpreta de forma recursiva `target`, `ref` y `plugin`. Es el único punto de
construcción declarativa, por lo que un componente anidado se comporta igual en preprocessing,
entrenamiento u operaciones. `PluginRegistry` descubre entry points sin importar módulos y valida el
contrato al resolver. `PluginUsageSession` limita la procedencia al run actual y
`EnvironmentManifest` la persiste junto a Python, PyTorch, CUDA, paquetes y Git.

Los targets y plugins son código confiable, no datos. Separar discovery de resolución evita efectos
laterales durante un simple listado; separar `ObjectFactory` de los runners evita que éstos acumulen
reglas de importación.

## Tareas genéricas y preprocessing

`Task` define `run(context)`; `TaskContext` entrega identidad, rutas seguras, inputs hasheados,
cancelación y resume; `TaskOutput` devuelve outputs JSON, métricas escalares y declaraciones de
artefacto. Son tres clases porque entrada de ejecución, salida aún no materializada y evidencia
terminal tienen ciclos de vida diferentes.

`TaskRunner` valida con `TaskValidator`, crea `TaskExecutionPlan`, decide skip/re-run por fingerprint,
captura entorno/plugins/log, invoca código y convierte declaraciones en `TaskArtifact`. Sólo después
de verificar existencia, contención, tamaño y SHA-256 publica `TaskResult` atómicamente. El resultado
anterior pasa al historial de intentos; no se sobrescribe evidencia.

`PreprocessingTask` es una especialización de `Task`. Coordina un `PreprocessingSource`, una secuencia
de `PreprocessingTransform` y un `PreprocessingSink`. La fuente descubre registros, la transformación
posee ciencia por registro y el sink posee serialización/atomicidad. Mezclarlos impediría sustituir
JSONL por ficheros o una transformación local por una del consumidor. `PreprocessingManifest`
checkpointa claves estables; `DatasetArtifact` identifica el dataset resultante y enlaza contenido,
splits, origen y fingerprint.

## Experimentos y entrenamiento

`Experiment` es la API de alto nivel. `ExperimentConfig` normaliza/migra y expande; no entrena.
`ExperimentValidator` acumula errores de Schema, imports, constructores y recursos sin efectos.
`ExperimentRunner` posee el ciclo de suite y `ExperimentExecutor` elige secuencial o procesos.
`TrainingOrchestrator` es la capa de proceso: slots, afinidad, threads, señales, grace y limpieza de
árbol. `LightningRunner` posee únicamente la integración Lightning.

`LightningTask` dirige lotes mapping al modelo, pérdidas y métricas. Loss y Metric son objetos
independientes para permitir múltiples objetivos/etapas y estado distribuido. Los procesos reciben
configuraciones materializadas independientes para no compartir mutación. `ExecutionConfig` crea
slots GPU, grupos DDP o slots CPU vacíos explícitos; valida oversubscription antes de lanzar.

## Workflows

`WorkflowNode` es una descripción inmutable: documento, dependencias, bindings, recursos y política
de rama; su `materialize()` concentra la composición aislada del documento del nodo.
`WorkflowConfig` valida nombres, referencias y ciclos. `WorkflowValidator` delega cada documento
materializado en `TaskValidator` o `ExperimentValidator` y reúne todos los errores en un
`WorkflowValidationReport` ordenado topológicamente, sin ejecutar el DAG. `WorkflowRunner.plan()` calcula niveles
topológicos sin importar targets ni escribir. `WorkflowRunner.run()` ejecuta sólo nodos listos con un
límite local, delega cada documento a `TaskRun` o `Experiment` y conserva resultados de ramas
independientes. Un fallo bloquea descendientes, no hermanos; `continue_on_failure` es explícito.

Los bindings `${nodes.nombre.outputs.ruta}`, `.metrics` y `.artifacts` se resuelven sólo cuando la
dependencia terminó. El DAG no copia la lógica de resume: cada nodo mantiene su fingerprint y runner,
de modo que reejecutar un workflow reutiliza únicamente identidades válidas.

## Recursos, backends y fiabilidad

`ResourceRequest` declara CPU, RAM, GPU, memoria GPU, almacenamiento y duración estimada.
`ResourcePlanner` valida capacidad y produce `ResourcePlan` por first-fit determinista o waves
manuales. No lanza procesos. `ExecutionBackend` recibe comando y recursos ya planificados;
`LocalExecutionBackend` ejecuta un argv y `SlurmExecutionBackend` genera `sbatch`. Así los detalles
SLURM no contaminan Task/Experiment.

SLURM es preview-first: siempre produce el script exacto y sólo llama `sbatch` con `dry_run=False`.
Admite arrays, dependencias, nodos, recursos, entorno, prefijo de contenedor y requeue. Los argumentos
se citan y no se usa `shell=True`. Cancelar/reencolar exige un job id numérico.

`FailureClassifier` convierte evidencia explícita en `FailureCategory`; desconocido no se presume
transitorio. `RetryPolicy` limita intentos y backoff y pasa lineage al callable. Resume reutiliza
estado compatible; restart parte sin él; retry repite un fallo bajo la misma intención; fork crea
una nueva configuración/identidad. No deben usarse como sinónimos.

## Operaciones de modelos e HPO

`ModelOperation` centraliza carga weights-only, stripping de prefijo Lightning, device, DataLoader,
routing y promedio de ensembles. `InferenceTask` concatena predicciones en CPU y publica
`predictions.pt`. `EvaluationTask` mantiene métricas streaming sobre un dataset nuevo.
`ExportTask` usa un ejemplo explícito para TorchScript, `torch.export`, ONNX o un exporter inyectado.
Son `Task`, por lo que reciben inputs hasheados, intentos, artefactos y procedencia sin otro runner.

`RandomSearch` usa un RNG privado, espacios choice/uniform/loguniform/int, condiciones y fingerprints
de trial; no altera grid/ablations. `OptunaSearch` es un adaptador opcional que crea TPE reproducible
y pruners ASHA/Hyperband. No se incluye Optuna en base porque storage/scheduler son decisiones del
usuario; ambos producen configuraciones que usan la planificación existente.

`AdaptiveExperimentOptimizer` es una segunda ruta, opt-in, para estudios que no pueden
materializarse de antemano. `AdaptiveExperimentController` posee la política y el replay state;
searchers sólo proponen configuraciones, fidelity/seed policies sólo proponen continuaciones,
modelos de curva/coste/memoria sólo predicen y `ResourceAdmissionController` sólo decide
viabilidad. `SearchSpace` separa muestreo de representación mixta; `BoTorchSearcher` modela
`f(x,b)` y pending; `LearningCurveModel` mantiene posteriors de curvas/seeds;
`GaussianValueOfInformation` compara START/RESUME/ADD_SEED; `FeatureAwareMemoryModel` aprende
`M(x,z)` con OOM censurado; `MemoryProbePolicy` decide probes concretos y `MemoryCapacity` evita
confundir UNKNOWN, UNBOUNDED y KNOWN(0). Ninguna de esas responsabilidades pertenece al scheduler.
`AdaptiveRunMaterializer` convierte la acción aceptada en un experimento ordinario y
`AdaptiveExperimentWorker` lo ejecuta en el runner existente. `TrainingOrchestrator.run_dynamic`
aporta slots vivos sin conocer HPO. `AdaptiveObservationReader` devuelve la evidencia canónica al
controlador. Esta separación permite probar matemáticas con datos sintéticos y conserva una sola
implementación de training/checkpoint/resultados. La justificación, invariantes de identidad y
migración completa están en [ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md](ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md).

## Artefactos, caché y retención

`ArtifactReference` es identidad lógica (store, key, SHA-256, tamaño, media type).
`ArtifactStore` define publicar, stage y exists. `LocalArtifactStore` funciona en disco local o
compartido con locks, publicación atómica y claves contenidas. `S3ArtifactStore` usa cliente inyectado
o boto3 opcional y verifica metadata/tamaño al stage. No hay método delete en el contrato base: una
referencia publicada queda protegida de borrado accidental.

`DistributedArtifactCache` no es otro store autoritativo. Adquiere un lease por key en un filesystem
compartido, descarga una vez, publica atómicamente y repara corrupción comparando el reference.
Invalidar elimina sólo la copia cacheada. `ArtifactRetentionManager`, en cambio, gestiona artefactos
locales de runs mediante plan, receipt, locks, journal y cuarentena; no pretende borrar providers.

## Resultados, registro e informes

`RunResult`/`TaskResult` son sobres terminales; `ResultRecord` añade la ubicación de un intento;
`ResultCatalog` recorre `result.json` e historial y detecta duplicados/ambigüedad. Es la única fuente
de verdad de resultados.

`ExperimentRegistry` enriquece esas records leyendo snapshots y ofrece filtros/exports; no mantiene
una base mutable. `ExperimentComparator` recibe grupos seleccionados, calcula tabla, dispersión,
efectos y diferencias de config sin elegir “el mejor”. `ReportBuilder` representa esos hechos en
Markdown/HTML sin conclusiones generadas. `LocalDashboard` crea una vista HTML estática de sólo
lectura. Tracking remoto sigue siendo complementario, no autoritativo.

## Observabilidad y reproducibilidad

`EventLogger` añade eventos JSONL bajo lock y limita tamaño. `ResourceMonitor` muestrea bajo demanda,
con frecuencia máxima, CPU/RSS/threads, memoria CUDA e items/s. `ProfilerAdapter` separa runners del
proveedor y `TorchProfilerAdapter` impone una ventana finita para acotar overhead.

`ReproducibilityProfile` ofrece fast/repeatable/strict, aplica RNG y determinismo y separa
fingerprints científico e infraestructura. `SeedDeriver` obtiene semillas jerárquicas estables con
SHA-256, nunca `hash()`. `EnvironmentExporter` produce pip freeze, Conda o snapshot orientado a
contenedor sin modificar el entorno.

## Cómo extender sin romper responsabilidades

- Nueva ciencia por registro: `PreprocessingTransform` del consumidor.
- Nuevo trabajo batch: `Task`; no modificar `TaskRunner`.
- Nuevo modelo/loss/métrica: contrato neural existente y target/plugin.
- Nuevo scheduler: `ExecutionBackend`; no introducir flags del proveedor en `TaskContext`.
- Nuevo almacén: `ArtifactStore`; no hacer que `ResultCatalog` dependa de su SDK.
- Nuevo profiler: `ProfilerAdapter`.
- Nueva vista/consulta: leer `ResultCatalog`/`ExperimentRegistry`; no escribir otra base.
- Nueva versión YAML: Schema, migración previewable, ejemplo, docs y tests de compatibilidad.

Una clase nueva está justificada sólo si posee una política o estado distinto. Una función privada
pequeña pertenece al objeto que posee esa responsabilidad; un nuevo manager genérico no se crea para
ocultar delegación. Los tests deben concentrarse en la frontera cambiada: validación, forma/valor,
gradiente cuando corresponda, fallo y una construcción YAML/import público.
