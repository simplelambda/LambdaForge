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
1.1 y workflow 1.0 siguen siendo el IR del runner. `ConfigurationComposer` resuelve composición e
interpolación restringida. Los validadores comprueban Schema, imports y firmas sin construir código
del usuario; `ObjectFactory` construye recursivamente sólo al ejecutar.

## 3. Tasks, preprocesado y workflows

`TaskConfig` posee identidad, inputs por contenido, layout y ciclo de vida. `TaskRun` crea un plan
inmutable, ejecuta un `Task`, verifica artifacts y publica atómicamente `TaskResult`. Los intentos
anteriores se archivan.

`PreprocessingTask` compone source, transforms ordenados y sink. Las claves estables gobiernan
sharding y resume. El padre escribe sink y manifest: I/O usa threads; CPU usa procesos `spawn` sólo
para transforms; GPU permanece en un proceso. Así no se comparte estado CUDA ni se corrompen
manifiestos. `Workflow` coordina documentos completos en un DAG, pero 0.5.1 no distribuye un mismo
DAG entre clusters.

## 4. Experimentos y entrenamiento

`ExperimentConfig` normaliza y expande seeds, grids y ablaciones. Resuelve únicamente marcadores
tipados `DatasetReference`; jamás adivina paths por el aspecto de un string. `RunFingerprint`
sustituye ubicaciones físicas por referencia e identidad lógica. `ExperimentRunner` posee un run;
`LightningTask` adapta batch/model/losses/metrics; `TrainingOrchestrator` agenda procesos y rellena
un slot justo después de observar que termina. `EpochMetricsCSV` reescribe atómicamente las curvas.

## 5. HPO adaptativo

`AdaptiveExperimentOptimizer` persiste el controlador y materializa cada START/RESUME/ADD_SEED como
un experimento ordinario con checkpoint. `AdaptiveExperimentController` compone búsqueda,
fidelidad, seeds, curvas, coste, memoria, admisión y selección. CONFIRM es una fase científica
separada. Véase [la arquitectura específica](ADAPTIVE_OPTIMIZATION_ARCHITECTURE.es.md).

## 6. Runtime multiclúster explícito

`ClusterCatalog` guarda `ClusterProfile`. `ExecutionBundleBuilder` materializa YAML, selecciona
ubicaciones lógicas para el destino, construye wheels exactas de LambdaForge/proyecto y genera un
bundle por contenido. `EnvironmentIdentity` incluye bytes instalables, Python y wheelhouse offline.
`ControlPlane` transfiere mediante `Transport`, prepara el intérprete con `EnvironmentProvider` y
envía por `Scheduler`. `ManagedEnvironmentProvider` crea un venv idempotente de usuario;
`ExistingEnvironmentProvider` sólo verifica. `JobStore` permite que `JobService` reconecte después.
No hay daemon, placement automático, passwords SSH, instalador CUDA ni workflow entre clusters.

## 7. Resultados, plots y artifacts

`ResultCatalog` descubre intentos; `ResultService` añade selectores humanos, `MetricSeries`,
comparación y export. `VisualizationService` crea `PlotSpec` y renderiza atómicamente fuera del
training; `.plot.json` permite regenerar y cachear. `ArtifactService` mantiene separados inspector,
visualizer, schema y validator. NumPy/tablas tienen límites y pickle deshabilitado; la geometría
requiere roles explícitos. Los servicios remotos sincronizan evidencia pequeña y sólo descargan el
artifact que se pide.

## 8. Extensión y límites

Se prefiere una clase del proyecto consumidor mediante `target`; los entry points son para
proveedores reutilizables. Plotly, NetworkX, trimesh, Optuna, BoTorch, stores y trackers siguen
opcionales. Una API nueva necesita validación, re-export público, tests focalizados, documentación
EN/ES y entrada para agentes.

En 0.5.1 no hay placement automático, workflows multiclúster, servidor/GUI, descargas implícitas
pesadas, instalación CUDA, síntesis de wheels de otra plataforma ni interpretación mágica de
arrays. Tampoco se amplía la matemática de HPO.
