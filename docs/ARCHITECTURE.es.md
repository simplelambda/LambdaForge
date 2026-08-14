[English](ARCHITECTURE.md) | [Español](ARCHITECTURE.es.md)

# Arquitectura de LambdaForge

Este documento explica cómo se cumplen las promesas públicas y dónde debe modificar el código un
mantenedor. La [guía principal](../README.es.md) enseña a usar el framework; esta explica sus
responsabilidades. LambdaForge es una biblioteca y una CLI, no un servicio alojado. El proyecto
consumidor conserva los datasets, modelos y políticas de dominio.

## 1. Límites arquitectónicos

```text
CLI / fachada LambdaForge
        ↓
servicios de aplicación (tasks, experiments, resultados, artifacts, control plane)
        ↓
planes, registros y contratos públicos inmutables
        ↓
adaptadores de proveedores (Lightning, SLURM, SSH, S3, Plotly, BoTorch...)
```

Los objetos de valor validan al construirse y serializan de forma explícita. Los servicios coordinan
el trabajo; los proveedores traducen una operación ya decidida. Sólo `target`, `ref` y `plugin`
importan código desde YAML. Se importa desde namespaces públicos, no desde ubicaciones privadas.

## 2. Configuración

`AuthoringConfig` compila la sintaxis sencilla (`name`, una `loss`, `trainer.epochs`, datasets
lógicos y recursos portables) a `MaterializedConfig`. Los Schema estrictos de task 1.0, experiment
1.1 y workflow 1.0 siguen siendo IR del runner; `kind: dataset` Schema 1.0 describe una receta.
`ConfigurationComposer` resuelve composición e
interpolación restringida. Los validadores comprueban Schema, imports y firmas sin construir código
del usuario; `ObjectFactory` construye recursivamente sólo al ejecutar.

## 3. Tasks, preprocesado y workflows

`TaskConfig` posee identidad, inputs por contenido, layout y ciclo de vida. `TaskRun` crea un plan
inmutable, ejecuta un `Task`, verifica artifacts y publica atómicamente `TaskResult`. Los intentos
anteriores se archivan.

`PreprocessingTask` compone source, transforms ordenados y sink. Las claves estables gobiernan
sharding y resume. El padre escribe sink y manifest: I/O usa threads; CPU usa procesos `spawn` sólo
para transforms; GPU permanece en un proceso. Así no se comparte estado CUDA ni se corrompen
manifiestos. `Workflow` coordina documentos completos en un DAG, pero 0.7 no distribuye un mismo
DAG entre clusters.

## 4. Experimentos y entrenamiento

`ExperimentConfig` normaliza y expande seeds, grids y ablaciones. Resuelve únicamente marcadores
tipados `DatasetReference` mediante el `DatasetResolver` compartido; jamás adivina paths por el
aspecto de un string. El resolver usa primero `DatasetRegistry` para versiones/placements gestionados
y después `DataCatalog` para aliases, loaders y datos externos. `RunFingerprint`
sustituye ubicaciones físicas por referencia e identidad lógica. `ExperimentRunner` posee un run;
`LightningTask` adapta batch/model/losses/metrics; `TrainingOrchestrator` agenda procesos y rellena
un slot justo después de observar que termina. `EpochMetricsCSV` reescribe atómicamente las curvas.

Las extensiones ligadas al loop siguen siendo callbacks Lightning. Validation devuelve outputs del
modelo desacoplados para hacer diagnóstico incremental sin otro forward. `TrainingCompletionStore`
confirma training antes de que `PostRunService` ejecute cada `PostRunAction` en rank cero. La acción
recibe `PostRunContext`, selecciona checkpoint persistido mediante `CheckpointResolver`, devuelve
declaraciones de artifact compartidas y confirma un recibo verificado por contenido. Las identidades
de training/acción están separadas, así que recuperar o cambiar una acción no repite fit. Otro
allocation sigue siendo responsabilidad de Task/Workflow.

## 5. HPO adaptativo

`AdaptiveExperimentOptimizer` persiste el controlador y materializa cada START/RESUME/ADD_SEED como
un experimento ordinario con checkpoint. `AdaptiveExperimentController` compone búsqueda,
fidelidad, seeds, curvas, coste, memoria, admisión y selección. CONFIRM es una fase científica
separada. Véase [la arquitectura específica](ADAPTIVE_OPTIMIZATION_ARCHITECTURE.es.md).

## 6. Runtime multiclúster explícito

`ClusterCatalog` fusiona `ClusterProfile` de usuario/proyecto/explícito y conserva la fuente ganadora.
`ClusterAuthentication` sólo contiene modo/referencia; `CredentialService` elige un
`CredentialProvider` interactivo, entorno o keyring. `ControlPlaneFactory` mantiene OpenSSH por
defecto y sólo crea `PasswordSshTransport` Paramiko en modo explícito. `SlurmProfile` compone un
único `SlurmResourceMapping` y `SchedulerCommand` seguros antes de que `SlurmScheduler` escriba el
script.

`ExecutionBundleBuilder` materializa YAML, selecciona
ubicaciones lógicas para el destino, construye wheels exactas de LambdaForge/proyecto y genera un
bundle por contenido. `EnvironmentIdentity` incluye bytes instalables, Python y wheelhouse offline.
`ControlPlane` transfiere mediante `Transport`, prepara el intérprete con `EnvironmentProvider` y
envía por `Scheduler`. `ManagedEnvironmentProvider` crea un venv idempotente de usuario;
`ExistingEnvironmentProvider` sólo verifica. `JobStore` permite que `JobService` reconecte después.
`CudaCompatibilityResolver` consulta Python/arquitectura, driver/capability y wheel oficial, e
incluye `TorchInstallationPlan` en `EnvironmentIdentity`. `ManagedEnvironmentProvider` instala y
restringe esa wheel antes del resto y valida CUDA requerida; nunca gestiona drivers/toolkits host.

No hay daemon, placement automático, fichero crypto propio, instalador CUDA ni workflow entre
clusters. Véanse [operaciones](CLUSTERS.es.md) y [seguridad](SECURITY.es.md).

## 7. Resultados, plots y artifacts

`ResultCatalog` descubre intentos; `ResultService` añade selectores humanos, `MetricSeries`,
comparación y export. `VisualizationService` crea `PlotSpec` y renderiza atómicamente fuera del
training; `.plot.json` permite regenerar y cachear. `ArtifactService` mantiene separados inspector,
visualizer, schema y validator. NumPy/tablas tienen límites y pickle deshabilitado; la geometría
requiere roles explícitos. Los servicios remotos sincronizan evidencia pequeña y sólo descargan el
artifact que se pide.

### Responsabilidades del plano de control

`SshConnectionPolicy` posee reutilización/deadlines; los transports sólo ejecutan/transfieren.
`ProcessScheduler` crea el directorio durable y `ProcessSupervisor` posee hijo, leases, heartbeat y
estado remoto atómico. `JobService` es la capa neutral y `JobStore` sólo el índice local.
`ResourceService` elige probe directo/SLURM y conserva la última observación si queda offline.

## 8. Responsabilidades del ciclo de datasets

Las cuatro entidades públicas tienen responsabilidades distintas:

```text
DatasetRecipe (cómo) -> DatasetBuild (ejecución) -> DatasetVersion (qué)
                                                    -> DatasetPlacement (dónde)
```

`DatasetRecipeConfig` valida la receta científica y compila sus etapas al `WorkflowConfig` ya
existente; no crea otro motor DAG. `DatasetBuildService` decide `REUSE`/`EXECUTE` con fingerprints de
Task, propaga `force-stage` downstream y envía un job durable `dataset-build` por `ControlPlane`.
Los resultados Task verificados forman la caché de etapas. `StorageService` puede recoger caché sin
referencias, pero nunca versiones publicadas, resultados ni placements.

`DatasetPublisher` es la única frontera de publicación: comprueba etapas requeridas, valida
`DatasetIndex`, IDs, contención, checksums y schema de targets, copia a staging, renombra
atómicamente y sólo entonces actualiza `DatasetRegistry`. Un build fallido conserva etapas útiles
pero no publica versión.

`DatasetArtifact` v2 es el manifiesto inmutable. `content_id`/`dataset_id` depende de members,
partitions, targets, metadata semántica e identidades de assets, no de roots físicos; `build_id`
depende de recipe/stages/inputs/código. `DatasetRecord` es la proyección pequeña del Registry y
`DatasetPlacement` cada copia física. Mover los mismos bytes añade un placement sin cambiar el
contenido. Los manifiestos/registros v1 siguen siendo legibles.

`DatasetResolver` es la única política de resolución de datos gestionados para tasks, experimentos y
bundles. Fija nombre, versión y contenido en el estado científico y mantiene el placement como dato
operacional. `DatasetService` coordina discovery, plan, materialización, profiling, inspección,
diff, lineage y seguridad; `DatasetOperations` ejecuta operaciones acotadas junto a los bytes.
`DataCatalog` queda para datos externos, aliases, loaders y pins explícitos, no como segundo registry
de placements. Véase [ciclo de datasets](DATASETS.es.md).

## 9. Extensión y límites

Se prefiere una clase del proyecto consumidor mediante `target`; los entry points son para
proveedores reutilizables. Plotly, NetworkX, trimesh, Optuna, BoTorch, stores y trackers siguen
opcionales. Una API nueva necesita validación, re-export público, tests focalizados, documentación
EN/ES y entrada para agentes.

En 0.7 no hay placement automático, workflows multiclúster durables, servidor central/GUI, descargas implícitas
pesadas, instalación CUDA, síntesis de wheels de otra plataforma ni interpretación mágica de
arrays. Tampoco se amplía la matemática de HPO.
