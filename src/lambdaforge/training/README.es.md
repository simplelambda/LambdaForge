# Capa de entrenamiento y procesos de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

Este paquete conecta objetos genéricos de PyTorch con Lightning y ejecuta entrenamientos
independientes con recursos acotados y un apagado explícito.

## Contenidos

- [Mapa de objetos](#mapa-de-objetos)
- [Contrato de entrenamiento predeterminado](#contrato-de-entrenamiento-predeterminado)
- [Configuración](#configuración)
- [Métricas y registro](#métricas-y-registro)
- [Trabajos simultáneos](#trabajos-simultáneos)
- [Apagado y limpieza](#apagado-y-limpieza)
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

Cada pérdida recibe `(outputs, batch)` y devuelve un escalar diferenciable. Las métricas reciben los
mismos objetos mediante `update` y cada etapa posee copias profundas independientes. La tarea
registra pérdida total, pérdidas individuales y métricas. Las claves configurables sirven para
clasificación, regresión o salidas estructuradas sin nombres de dominio incrustados.

El optimizador se representa por una clase y parámetros. El scheduler es opcional y puede aportar
metadatos Lightning (`interval`, `frequency`, `monitor` y campos similares).

## Configuración

`LightningTrainConfig` posee campos estables: épocas, acelerador, dispositivos, estrategia,
precisión, precisión matricial, acumulación/clipping, frecuencia de validación, checkpoints, parada
temprana, logger e interfaz. Se validan antes de crear Trainer.

Los parámetros nuevos o poco comunes van en `trainer_kwargs`; allí no se pueden sustituir los campos
explícitos. El logger puede ser `none`, `csv`, `lightning_csv` o un objeto ya construido.
`write_epoch_metrics_csv` conserva de forma independiente el artefacto denso canónico que necesita
la agregación de LambdaForge. Los callbacks adicionales se pasan a `LightningRunner` con
`runner.params.callbacks`. Por comodidad, el YAML también puede declarar esa lista como `callbacks`
en el nivel superior.

Los monitores de checkpoint y parada temprana usan por defecto la primera métrica de **validación**,
no la primera de entreno. `checkpoint_monitor`, `checkpoint_mode`, `early_stopping_monitor` y
`early_stopping_mode` hacen explícita la selección. Una clave personalizada que no sea pérdida exige
un modo `min` o `max`, de modo que LambdaForge nunca adivina su dirección científica.

`LightningDataModule` controla dataset, lote, shuffle, workers, memoria fijada, persistencia,
prefetch, collator, inicializador y `drop_last` de entreno. Los `dataloader_kwargs` comunes o por
partición reenvían opciones extra seguras. Las claves del framework no se pueden repetir.

LambdaForge no cachea los datasets: son objetos `torch.utils.data.Dataset` normales y deciden cómo
almacenar o cargar muestras.

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

## Trabajos simultáneos

`TrainingOrchestrator.run` recibe objetos `TrainingJob` y concurrencia máxima. Se usa
`torch.multiprocessing` con `spawn`, necesario para inicializar CUDA con seguridad. Las restricciones
GPU se establecen dentro del hijo antes de usar CUDA. `CUDA_VISIBLE_DEVICES` del padre solo se lee.

Cada hijo recibe límites propios de hilos intra/inter-op, variables BLAS/OpenMP y afinidad CPU. Los
workers heredan afinidad e instalan `GuardedWorkerInit`, que primero aplica la protección y después
llama al inicializador del usuario.

Evita closures, lambdas y callables exclusivos de notebooks: lo enviado a `spawn` debe ser
importable y serializable. En Windows los scripts deben usar `if __name__ == "__main__":`.

## Apagado y limpieza

SIGINT y SIGTERM solicitan parada mediante un evento compartido. `StopEventCallback` lo comprueba en
los límites de lote. El orquestador espera `grace_seconds`, termina recursivamente los descendientes
restantes y restaura señales y afinidad del padre.

En Windows cada raíz se asigna a un Job Object con `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. En Linux,
los hijos solicitan una señal al morir el padre cuando está disponible. psutil es el fallback
recursivo portable. La limpieza se ejecuta desde bloques `finally`.

La garantía cubre árboles propiedad del framework. Un ejecutable externo que se convierta
intencionadamente en daemon independiente necesita su propia integración de parada.

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
    model_input_key: features
    logging: {loss_prog_bar: false, metric_prog_bar: true, logger: true}
trainer:
  logger:
    target: my_project.logging.ProjectLogger
    params: {project: demo}
  write_epoch_metrics_csv: true
  checkpoint_monitor: val_selection_score
  checkpoint_mode: max
callbacks:
  - target: my_project.callbacks.ProjectCallback
    params: {}
```

Los modelos pueden ser objetos `torch.nn.Module` normales. Pérdidas y métricas heredan de los
contratos LambdaForge para mantener explícitos precisión mixta y DDP. Las especificaciones anidadas
`target`/`ref` funcionan también en parámetros. La suite automática ejecuta de extremo a extremo
este mismo patrón con modelo, pérdida, métrica, logger y callback externos.

- Hereda o sustituye `LightningTask` para lotes no mapeados, varios optimizadores u optimización
  manual.
- Usa un módulo de datos propio si la preparación/división debe ocurrir dentro de la ejecución.
- Pasa callbacks y loggers como especificaciones YAML `target`.
- Sustituye `runner.target` para otro backend; el runner de experimentos espera `fit` y `test`
  compatibles.
- Cubre cambios sensibles de procesos con pruebas `spawn` y valida la interrupción en el SO y host
  GPU objetivo.
