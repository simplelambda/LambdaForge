# Capa de entrenamiento y procesos de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

Este paquete conecta objetos genéricos de PyTorch con Lightning y ejecuta entrenamientos
independientes con recursos acotados y un apagado explícito.

## Contenidos

- [Mapa de objetos](#mapa-de-objetos)
- [Contrato de entrenamiento predeterminado](#contrato-de-entrenamiento-predeterminado)
- [Varias entradas y grupos del optimizador](#varias-entradas-y-grupos-del-optimizador)
- [Configuración](#configuración)
- [Ciclo de vida de checkpoints y locks de suite](#ciclo-de-vida-de-checkpoints-y-locks-de-suite)
- [Métricas y registro](#métricas-y-registro)
- [Adaptadores de tracking opcionales](#adaptadores-de-tracking-opcionales)
- [Trabajos simultáneos](#trabajos-simultáneos)
- [Apagado y limpieza](#apagado-y-limpieza)
- [Verificación del ciclo de vida](#verificación-del-ciclo-de-vida)
- [Personalización](#personalización)

## Mapa de objetos

| Objeto | Responsabilidad |
|---|---|
| `LightningTask` | Conectar lote, modelo, pérdidas, métricas, optimizador y scheduler. |
| `LightningDataModule` | Envolver datasets PyTorch ya creados para cada partición. |
| `LightningTrainConfig` | Validar opciones comunes de Trainer y reenviar las avanzadas. |
| `LightningRunner` | Construir callbacks/logger y ejecutar fit/test. |
| `TaskLoggingConfig` | Controlar la publicación de pérdidas totales/individuales y métricas. |
| `EpochMetricsCSV` | Escribir una fila densa de escalares por época. |
| `EpochLogPrinter` | Imprimir resúmenes compactos en los logs capturados. |
| `EpochStats` | Registrar tiempo, pico de memoria GPU y RSS del proceso. |
| `MLflowTrackingLogger` | Logger MLflow local/remoto opcional de `lambdaforge.tracking`. |
| `TensorBoardTrackingLogger` | Logger opcional de eventos TensorBoard de `lambdaforge.tracking`. |
| `WeightsAndBiasesTrackingLogger` | Logger W&B offline/online opcional de `lambdaforge.tracking`. |
| `TrainingJob` | Nombre/callable/dispositivos serializables de un trabajo. |
| `TrainingOrchestrator` | Planificar trabajos, asignar recursos y controlar procesos. |
| `ProcessGuard` | Muerte del padre, pools de hilos, afinidad y limpieza descendiente. |
| `WindowsJobObject` | Contención kill-on-close para descendientes Windows. |
| `LogKeyFilter` | Aplicar patrones include/exclude a claves de CSV y terminal. |

El paquete físico separa `callbacks/`, `data/` y `orchestration/`; el espacio estable
`lambdaforge.training` reexporta los objetos principales. `CheckpointPolicy`, `LoggerMode`,
`MatmulPrecision` y `MonitorMode` son enums de valores cerrados.

## Contrato de entrenamiento predeterminado

`LightningTask` espera un lote con forma de mapa. Con `model_input_key="x"` llama a
`model(batch["x"])`; con clave `None` pasa el mapa completo. Un resultado tensor se envuelve bajo
`model_output_key`; un mapa se conserva.

Cada pérdida recibe `(outputs, batch, context)`, donde `context` es la `LightningTask` activa, y
devuelve un escalar diferenciable. Las pérdidas propias deben conservar `context=None` en la firma
para poder usarse también de forma independiente. Las métricas reciben salidas y lote mediante
`update` y cada etapa posee copias profundas independientes. La tarea registra pérdida total,
pérdidas individuales y métricas. Las claves configurables sirven para clasificación, regresión o
salidas estructuradas sin nombres de dominio incrustados.

El optimizador se representa por una clase y parámetros. El scheduler es opcional y puede aportar
metadatos Lightning (`interval`, `frequency`, `monitor` y campos similares).

## Varias entradas y grupos del optimizador

`model_input_keys` evita crear una tarea propia cuando el modelo consume varios tensores. Una
secuencia dirige argumentos posicionales en orden; un mapa conecta los argumentos con nombre del
modelo con las claves del lote. Es mutuamente excluyente con un `model_input_key` no predeterminado:

```yaml
task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_keys:
      x: node_features
      edge_index: edge_index
```

Los modelos pueden exponer `parameter_groups()` como mapa de nombres estables a iterables de
parámetros. `optimizer_group_kwargs` modifica las opciones por grupo y `optimizer_kwargs` conserva
los valores comunes. Los árboles diferenciables emplean este contrato para separar enrutamiento,
hojas y ensemble sin acoplar `LightningTask` a una arquitectura:

```yaml
task:
  target: lambdaforge.training.LightningTask
  params:
    optimizer_kwargs: {lr: 0.001, weight_decay: 0.0001}
    optimizer_group_kwargs:
      routing: {lr: 0.0005}
      leaves: {weight_decay: 0.0}
```

Los grupos desconocidos fallan al construir el optimizador. Los parámetros entrenables que no
pertenecen a un grupo del modelo se sitúan en `task` y no se pierden silenciosamente.

## Configuración

`LightningTrainConfig` posee campos estables: épocas, acelerador, dispositivos, estrategia,
precisión, precisión matricial, acumulación/clipping, frecuencia de validación, checkpoints, parada
temprana, logger e interfaz. Se validan antes de crear Trainer.

Los parámetros nuevos o poco comunes van en `trainer_kwargs`; allí no se pueden sustituir los campos
explícitos. El logger puede ser `none`, `csv`, `lightning_csv`, un `target`/`ref`/plugin instalado o
una lista no vacía de loggers. `write_epoch_metrics_csv` conserva de forma independiente el artefacto
denso canónico. Los callbacks usan `target` o plugins `kind: callback` en
`runner.params.callbacks` o en la lista superior `callbacks`. No se exige ningún servicio externo.
Los adaptadores y extras opcionales de proveedor se documentan en la
[guía de tracking](../tracking/README.es.md).

Los monitores de checkpoint y parada temprana usan por defecto la primera métrica de **validación**,
no la primera de entreno. `checkpoint_monitor`, `checkpoint_mode`, `early_stopping_monitor` y
`early_stopping_mode` hacen explícita la selección. Una clave personalizada que no sea pérdida exige
un modo `min` o `max`, de modo que LambdaForge nunca adivina su dirección científica.

`LightningDataModule` controla dataset, lote, shuffle, workers, memoria fijada, persistencia,
prefetch, collator, inicializador y `drop_last` de entreno. Los `dataloader_kwargs` comunes o por
partición reenvían opciones extra seguras. Las claves del framework no se pueden repetir.

Los datasets siguen siendo objetos `torch.utils.data.Dataset` normales. El proyecto puede usar los
wrappers públicos de `lambdaforge.data` para carga explícita de archivos/mmap o caché acotada; la capa
de entreno nunca inserta caché implícita. Los splits admiten `target` local o plugin reutilizable
`kind: dataset`. Un `IterableDataset` requiere un datamodule propio que no fuerce shuffle map-style.
Cada worker posee una réplica del dataset, por lo
que conviene leer la [guía de datos](../data/README.es.md) antes de combinar `DatasetCache`,
`num_workers` y `persistent_workers`.

## Ciclo de vida de checkpoints y locks de suite

`trainer.checkpoint_policy` controla lo que Lightning crea durante fit: `none`, `last`, `best`,
`last_and_best` o `all`. La política superior Schema 1.1 `retention.checkpoints` controla de forma
independiente qué puede sobrevivir solo después de agregar una suite completa y correcta. En
particular, `prune_unselected: false` conserva cada checkpoint creado con independencia de `keep`.

Resume y finalización inspeccionan archivos seguros dentro del run. Por eso
`checkpoint_policy: all` sigue siendo reutilizable aunque Lightning no registre una ruta de callback
best/last. La carga con `CheckpointChoice.AUTO` resuelve best, después last y finalmente el
checkpoint local seguro más reciente. `BEST` y `LAST` exactos nunca cruzan roles silenciosamente;
retención también omite un rol ausente o ambiguo en vez de adivinar.

El ejecutor de experimentos posee un lock exclusivo de actividad de suite mientras hay workers de
entrenamiento activos. La agregación final toma una lease compartida de actividad y el lock de
agregación; retención toma, en orden fijo, locks exclusivos de actividad, agregación y retención.
Esto impide que otro proceso LambdaForge publique un receipt de finalización o pode archivos durante
el entrenamiento. Los locks pertenecen al sistema operativo, tienen timeout configurable y se
liberan tras una salida normal o abrupta.

Una parada cooperativa después de fit se registra como `interrupted`, no `ok`, y se omite
`test_after_fit`. El run sigue siendo reintentable y no puede contribuir a un receipt de elegibilidad
para retención. Consulta la
[guía de retención de artefactos](../experiments/retention/README.es.md) para roles de checkpoint,
receipts, recuperación transaccional y protección de rutas.

## Métricas y registro

El modo `csv` desactiva el CSV disperso de Lightning. `EpochMetricsCSV` se controla por separado con
`write_epoch_metrics_csv` y genera una fila densa por época incluso si está activo un logger externo.
`EpochLogPrinter` replica escalares en stdout, capturado después en `train.log`. `EpochStats` añade
`epoch_time_s`, picos de memoria GPU asignada/reservada y RSS de CPU si existe.

`metrics` sigue siendo la lista compartida compatible; `train_metrics`, `val_metrics` y
`test_metrics` seleccionan explícitamente cada etapa. Los nombres deben ser únicos dentro de ella.
`MetricAlias` renombra una métrica delegada cuando, por ejemplo, dos accuracies usan umbrales
distintos.

`TaskLoggingConfig` controla pérdida total/individual, agregación por paso/época, barra de progreso,
logger y sincronización distribuida. `epoch_metrics_include` / `epoch_metrics_exclude` filtran el
CSV denso, y `epoch_console_include` / `epoch_console_exclude` filtran la tabla con patrones de shell.
Sin métrica de validación se usa `val_loss` en modo `min`. La parada temprana se activa al configurar
paciencia.

## Adaptadores de tracking opcionales

`lambdaforge.tracking` expone los targets explícitos `MLflowTrackingLogger`,
`TensorBoardTrackingLogger` y `WeightsAndBiasesTrackingLogger`. Ocupan la misma posición
`trainer.logger` que cualquier logger Lightning propio y se pueden combinar en una lista no vacía:

```yaml
trainer:
  logger:
    - target: lambdaforge.tracking.TensorBoardTrackingLogger
      params:
        save_dir: ./tracking/tensorboard
        name: local-study
    - target: lambdaforge.tracking.WeightsAndBiasesTrackingLogger
      params:
        project: local-study
        offline: true
        save_dir: ./tracking/wandb
        log_model: false
  write_epoch_metrics_csv: true
```

Define `task.params.logging.logger: true` cuando las pérdidas y métricas de la tarea deban llegar
al proveedor; seleccionarlo no sustituye esta política de publicación. Los escalares de ejecución
de `EpochStats` usan la vía logger de Lightning de forma independiente.

Instala el extra mínimo correspondiente —`lambdaforge[mlflow]`, `lambdaforge[tensorboard]` o
`lambdaforge[wandb]`— o `lambdaforge[tracking]` para los tres. Los imports siguen siendo lazy;
construir un adaptador sin su SDK lanza `TrackingDependencyError` con la indicación exacta de
instalación.

`TaskLoggingConfig.logger` gobierna las pérdidas/métricas de la tarea, mientras los callbacks del
framework como `EpochStats` también publican escalares de ejecución y el proveedor puede recopilar
metadata adicional según sus ajustes. Mantén las credenciales fuera del YAML, conserva
`log_model: false` salvo que las copias remotas de checkpoints sean intencionadas y deja
`write_epoch_metrics_csv: true` para disponer de un registro local neutral respecto al proveedor.
La retención de LambdaForge no puede eliminar artefactos subidos o gestionados por el proveedor. La
[guía de tracking](../tracking/README.es.md) contiene los constructores completos, ejemplos MLflow
local/remoto, TensorBoard local y W&B offline/online, advertencias de privacidad y enlaces oficiales.

## Trabajos simultáneos

`TrainingOrchestrator.run` lanza todos los `TrainingJob` suministrados; `run_scheduled` consume en
cambio un pool fijo de slots y proporciona así el límite de concurrencia. Se usa
`torch.multiprocessing` con `spawn`, necesario para inicializar CUDA con seguridad. Las restricciones
GPU se preparan al crear el proceso y se establecen dentro del hijo antes de usar CUDA; el valor del
padre se restaura inmediatamente tras cada spawn.

`TrainingJob.devices` y los slots del scheduler comparten un contrato inmutable:

| Valor | Significado |
|---|---|
| `None` | Heredar sin cambios todo el conjunto CUDA visible para el padre. |
| `[]` o `()` | Modo CPU explícito: el hijo recibe `CUDA_VISIBLE_DEVICES=""`. |
| Una secuencia de enteros no vacía | Posiciones lógicas dentro del conjunto visible del padre, o IDs físicos si no existe la variable. |

`run_scheduled` usa intencionadamente el slot seleccionado en lugar de `TrainingJob.devices`; así el
pool de slots, y no un contador global, es la autoridad de recursos.

Las asignaciones rechazan strings, booleanos, índices negativos/fraccionarios y duplicados.
`ExecutionConfig` rechaza también recursos booleanos, fraccionarios o no finitos, exige
`grace_seconds` finito no negativo, conteos positivos y grupos GPU no vacíos/coherentes en
parallel/DDP. Los límites CPU opcionales usan `null` para conservar el entorno; solo los workers de
DataLoader permiten cero. `TrainingOrchestrator` exige `grace_seconds` finito no negativo y
`poll_seconds` finito positivo.

Cada hijo recibe límites propios de hilos intra/inter-op, variables BLAS/OpenMP y afinidad CPU. Los
workers heredan afinidad e instalan `GuardedWorkerInit`, que primero aplica la protección y después
llama al inicializador del usuario.

Evita closures, lambdas y callables exclusivos de notebooks: lo enviado a `spawn` debe ser
importable y serializable. En Windows los scripts deben usar `if __name__ == "__main__":`.

## Apagado y limpieza

`TrainingOrchestrator.request_stop()` es el punto de entrada público e idempotente de cancelación.
Con `manage_signals=True` (predeterminado), handlers temporales de SIGINT/SIGTERM y SIGBREAK cuando
está disponible registran una petición local sin locks; el bucle ordinario llama después a ese
método y las guardas hijas retransmiten las señales cooperativas al evento compartido desde contexto
normal de hilo. Así no se toman locks de `multiprocessing.Event` dentro de un handler, lo que podría
bloquearse si la señal interrumpió el mismo lock. `StopEventCallback` comprueba el evento en límites
de lote de entrenamiento/validación/test. El orquestador espera `grace_seconds`, termina los árboles
restantes, mata supervivientes y restaura handlers y afinidad del padre. Los joins y la terminación
recursiva usan deadlines monotónicos compartidos, por lo que la espera no se multiplica por la
cantidad de jobs; un superviviente tras la escalada acotada provoca `RuntimeError`.

Python solo permite instalar handlers de señal en su hilo principal. Un host que invoque
`run`/`run_scheduled` desde un hilo secundario debe construir
`TrainingOrchestrator(manage_signals=False)` y llamar a `request_stop()` desde su propio hook de
apagado. Mantener `manage_signals=True` en un hilo secundario falla antes de lanzar trabajo y
restaura cualquier estado parcialmente instalado. `manage_signals=False` transfiere al host la
propiedad de los handlers; no elimina la cancelación mediante el evento compartido.

En Windows cada raíz se asigna a un Job Object con `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Un fallo de
inicialización emite `RuntimeWarning` y queda registrado en
`process_isolation_warnings` mientras sigue activa la limpieza psutil. Si un Job Object activo no
puede aceptar un worker, ese proceso se termina y se lanza una excepción en lugar de debilitar el
aislamiento silenciosamente.

Cada worker recibe además el `expected_parent_pid` exacto del launcher. Linux solicita una señal
parent-death del kernel mediante `prctl`; todos los workers POSIX comparan el padre observado antes y
después de instalar la guarda y mantienen un watchdog ligero ante reparentados posteriores. Así se
cierra la carrera en la que el launcher muere durante el arranque del hijo. Los workers de
`DataLoader` heredan la guarda y límites aplicables. La limpieza se ejecuta desde bloques `finally`.

La garantía cubre árboles propiedad del framework. Un ejecutable externo que se convierta
intencionadamente en daemon independiente necesita su propia integración de parada.
`TrainingOrchestrator` tiene estado y no es reentrante: nunca solapes `run` y `run_scheduled` sobre
la misma instancia y prefiere una instancia por run con propietario independiente.

## Verificación del ciclo de vida

La CI alojada crea un árbol real launcher/worker/descendiente anidado en Ubuntu y Windows para
CPython 3.10-3.14. En POSIX envía un `killpg(SIGINT)` real al grupo aislado. En Windows una petición
externa hace que el launcher provoque un SIGBREAK Python dirigido porque los eventos nativos de
consola afectan a todo el grupo y alcanzarían también procesos ajenos del runtime numérico. La suite
valida por tanto el handler Python de SIGBREAK, no la entrega nativa CTRL_C/CTRL_BREAK al grupo.

Otro escenario termina abruptamente el launcher, espera a que desaparezca cada identidad registrada
y comprueba que no queden artefactos temporales. Las pruebas unitarias cubren además supresión de
lanzamientos mediante `request_stop()`, política de señales en hilo secundario, mismatch del padre
exacto, semántica CUDA `None` frente a vacío, validación estricta de tiempos/recursos y limpieza
acotada. La limpieza de emergencia evita que una aserción fallida deje workers residuales.

El workflow CUDA self-hosted opcional ejecuta además una época GPU real mediante `ExperimentRunner`
y el contrato YAML público, sincroniza el dispositivo, comprueba los artefactos de métricas/entorno
y registra evidencia del hardware CUDA. Su mera existencia no garantiza CUDA: debe completarse
correctamente en un runner compatible. La entrega nativa al grupo de consola de Windows,
multi-GPU/DDP, muerte de máquina/contenedor, reentrancia de una misma instancia y daemons externos
desacoplados quedan fuera de este contrato automatizado. Consulta
[Integración continua](../../../README.es.md#integración-continua) para el contrato CI completo.

## Personalización

La superficie completa de extensión YAML se ilustra aquí:

```yaml
model:
  target: my_project.models.ProjectModel
  params: {width: 128}
losses:
  - target: my_project.losses.ProjectLoss
    params: {name: reconstruction}
train_metrics:
  - target: my_project.metrics.ProjectMetric
    params: {name: train_score}
val_metrics:
  - target: my_project.metrics.ProjectMetric
    params: {name: selection_score}
task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_keys: {left: left_features, right: right_features}
    optimizer_group_kwargs: {backbone: {lr: 0.0001}, head: {lr: 0.001}}
    logging: {loss_prog_bar: false, metric_prog_bar: true, logger: true}
data:
  train:
    plugin: {kind: dataset, name: acme_records}
    params: {split: train}
trainer:
  logger:
    - target: my_project.logging.ProjectLogger
      params: {project: demo}
    - plugin: {kind: logger, name: jsonl_logger}
      params: {path: metrics.jsonl}
  write_epoch_metrics_csv: true
  checkpoint_monitor: val_selection_score
  checkpoint_mode: max
callbacks:
  - plugin: {kind: callback, name: artifact_marker}
    params: {filename: finished.txt}
```

Los modelos pueden ser objetos `torch.nn.Module` normales. Pérdidas y métricas heredan de los
contratos LambdaForge para mantener explícitos precisión mixta y DDP. Las especificaciones anidadas
`target`/`ref` funcionan también en parámetros. La suite automática ejecuta de extremo a extremo
este mismo patrón con modelo, pérdida, métrica, dataset, logger y callback externos.

- Hereda o sustituye `LightningTask` para lotes no mapeados, varios optimizadores u optimización
  manual. Los modelos ordinarios con varias entradas ya funcionan mediante `model_input_keys`.
- Usa un módulo de datos propio si la preparación/división debe ocurrir dentro de la ejecución.
- Pasa datasets, callbacks y loggers reutilizables mediante plugins instalados o `target` YAML
  locales. Los autores deberían heredar las bases de callback/logger desde
  `lambdaforge.integrations.Lightning`; consulta la [guía de plugins](../plugins/README.es.md).
- Selecciona los targets opcionales incluidos en `lambdaforge.tracking` cuando basten MLflow,
  TensorBoard o W&B; usa un target/plugin del proyecto para otro contrato de logger.
- Sustituye `runner.target` para otro backend; el runner de experimentos espera `fit` y `test`
  compatibles.
- Cubre cambios sensibles de procesos con pruebas `spawn` y valida la interrupción en el SO y host
  GPU objetivo.
