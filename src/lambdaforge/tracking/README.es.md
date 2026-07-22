# Tracking opcional de experimentos en LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md) ·
[Entrenamiento y procesos](../training/README.es.md)

Este paquete ofrece objetos pequeños y opcionales que adaptan MLflow, TensorBoard y Weights &
Biases a la frontera de logger Lightning existente en LambdaForge. La instalación base continúa
siendo local y no exige servicios: ningún SDK de tracking se importa hasta construir su adaptador.

## Contenidos

- [Alcance y objetos públicos](#alcance-y-objetos-públicos)
- [Instalación](#instalación)
- [Selección YAML](#selección-yaml)
- [MLflow](#mlflow)
- [TensorBoard](#tensorboard)
- [Weights & Biases](#weights--biases)
- [Referencia completa de parámetros](#referencia-completa-de-parámetros)
- [Métricas, privacidad, checkpoints y artefactos](#métricas-privacidad-checkpoints-y-artefactos)
- [Loggers propios y plugins](#loggers-propios-y-plugins)
- [Fallos de dependencias y proveedores](#fallos-de-dependencias-y-proveedores)
- [Referencias oficiales](#referencias-oficiales)

## Alcance y objetos públicos

Importa estos objetos desde `lambdaforge.tracking`:

| Objeto | Responsabilidad |
|---|---|
| `MLflowTrackingLogger` | Construir el logger MLflow de Lightning tras comprobar el extra `mlflow`. |
| `TensorBoardTrackingLogger` | Construir el logger TensorBoard de Lightning tras comprobar el extra `tensorboard`. |
| `WeightsAndBiasesTrackingLogger` | Construir el logger W&B de Lightning tras comprobar el extra `wandb`. |
| `TrackingBackend` | Identificadores canónicos de backend `mlflow`, `tensorboard` y `wandb`. |
| `TrackingDependencyGuard` | Comprobar disponibilidad sin importar un SDK de tracking y, después, exigirlo o importarlo explícitamente. |
| `TrackingDependencyError` | `ImportError` accionable con la indicación exacta para instalar el extra opcional. |

Los tres adaptadores son loggers Lightning ordinarios. La guarda de TensorBoard acepta
`tensorboard` (preferido) o un `tensorboardX` ya instalado; el extra `tensorboard` documentado
instala el backend preferido. No crean otro motor de experimentos, no cambian el bucle de
entrenamiento ni vuelven obligatorio a ningún proveedor. `LightningRunner` entrega un adaptador, o
una lista no vacía de ellos, directamente a Lightning. LambdaForge sigue escribiendo de forma
independiente su configuración materializada, manifiestos de resultado/entorno, log capturado y
—salvo que se desactive— el `metrics.csv` denso canónico. `EnvironmentManifest` también registra
las versiones instaladas de `mlflow`, `tensorboard`, `tensorboardX` y `wandb` cuando están
presentes.

## Instalación

Instala únicamente el proveedor que necesite el proyecto:

```powershell
python -m pip install "lambdaforge[mlflow]"
python -m pip install "lambdaforge[tensorboard]"
python -m pip install "lambdaforge[wandb]"
```

Instala los tres adaptadores con el extra combinado:

```powershell
python -m pip install "lambdaforge[tracking]"
```

Desde un clon editable, utiliza `-e ".[mlflow]"`, `-e ".[tensorboard]"`,
`-e ".[wandb]"` o `-e ".[tracking]"`. El conjunto de dependencias base de `lambdaforge` no
contiene deliberadamente ninguno de estos SDK.

## Selección YAML

Selecciona un adaptador con la misma sintaxis recursiva y de confianza `target` usada por el resto
de objetos:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.MLflowTrackingLogger
    params:
      experiment_name: lambdaforge-research
      run_name: baseline-seed-7
      save_dir: ./mlruns
      log_model: false
  write_epoch_metrics_csv: true
```

Para publicar en varios destinos, usa una lista no vacía de loggers. Cada entrada se construye de
forma independiente y puede mezclar adaptadores de tracking incluidos, clases `target` del proyecto
y plugins `kind: logger` instalados:

```yaml
trainer:
  logger:
    - target: lambdaforge.tracking.TensorBoardTrackingLogger
      params:
        save_dir: ./tracking/tensorboard
        name: comparison
    - target: lambdaforge.tracking.WeightsAndBiasesTrackingLogger
      params:
        project: lambdaforge-research
        name: baseline-seed-7
        offline: true
        save_dir: ./tracking/wandb
        log_model: false
  write_epoch_metrics_csv: true
```

Los strings existentes `none`, `csv` y `lightning_csv` conservan su significado. Los nombres de
proveedor no son nuevos strings mágicos: los adaptadores son objetos explícitos cuyos parámetros de
constructor se pueden configurar por completo.

Seleccionar un proveedor no sustituye la política de publicación de la tarea. Define `logger: true`
dentro del mapping `task.params.logging` existente cuando sus pérdidas y métricas deban llegar al
proveedor:

```yaml
task:
  target: lambdaforge.training.LightningTask
  params:
    # ...routing del modelo y ajustes del optimizador del proyecto...
    logging:
      logger: true
```

Los escalares de ejecución de `EpochStats` usan la vía logger de Lightning de forma independiente.
Mantén el flag de tarea en `false` cuando el proveedor deba recibir estadísticas de ejecución pero
no valores científicos de pérdidas/métricas.

## MLflow

### Archivos locales

Cuando `tracking_uri` es `null` y `MLFLOW_TRACKING_URI` no está definida, Lightning utiliza
`save_dir` para los runs MLflow locales:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.MLflowTrackingLogger
    params:
      experiment_name: local-study
      run_name: seed-7
      tracking_uri: null
      save_dir: ./tracking/mlruns
      tags: {stage: exploratory, owner: research-team}
      log_model: false
```

`save_dir` vale `./mlruns` por defecto. Un `tracking_uri` YAML explícito tiene precedencia sobre
`MLFLOW_TRACKING_URI`; en caso contrario, esa variable de entorno tiene precedencia sobre el
`save_dir` local. Indica una ruta explícita si varios lanzadores pueden tener directorios de trabajo
distintos. MLflow asigna el run cuando `run_id` es `null`; proporcionar un ID previo pide al
proveedor reutilizar ese run y constituye una decisión científica/de ciclo de vida, no la
reanudación de LambdaForge.

### Servidor remoto

Define una URI de tracking HTTP(S) para un servidor remoto:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.MLflowTrackingLogger
    params:
      experiment_name: shared-study
      run_name: seed-7
      tracking_uri: https://mlflow.example.org
      artifact_location: null
      log_model: false
      synchronous: true
```

Las credenciales, tokens y secretos del almacenamiento de objetos pertenecen al entorno del
proceso o al mecanismo de credenciales del proveedor, nunca al YAML. `save_dir` no tiene efecto
cuando se establece `tracking_uri`. El servidor remoto decide la ubicación de artefactos
predeterminada salvo que se proporcione `artifact_location`. Autenticación, TLS, autorización, base
de datos, proxy de artefactos y retención del servidor son responsabilidades operativas ajenas a
LambdaForge.

## TensorBoard

TensorBoard escribe archivos de eventos y no necesita una cuenta de tracking alojada:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.TensorBoardTrackingLogger
    params:
      save_dir: ./tracking/tensorboard
      name: local-study
      version: seed-7
      log_graph: false
      default_hp_metric: true
      prefix: ""
      sub_dir: null
      max_queue: 10
      flush_secs: 120
```

`save_dir` es obligatorio. Los logs se guardan bajo `save_dir/name/version`, seguido de `sub_dir`
cuando está presente. Los parámetros adicionales se envían al `SummaryWriter` de TensorBoard;
`max_queue` y `flush_secs` son ejemplos. Inicia el visor por separado:

```powershell
tensorboard --logdir ./tracking/tensorboard
```

Mantén `version: null` para que Lightning genere versiones o garantiza que una versión explícita
sea única. Varios trabajos simultáneos que escriban el mismo directorio/versión pueden corromper o
entremezclar un flujo de eventos. Las URLs de sistemas de archivos remotos dependen del soporte del
proveedor utilizado por Lightning/fsspec y no están cubiertas por las garantías locales de
atomicidad o retención de LambdaForge.

## Weights & Biases

### Offline

El modo offline guarda un directorio de run W&B autocontenido para revisarlo o sincronizarlo más
adelante:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.WeightsAndBiasesTrackingLogger
    params:
      project: lambdaforge-research
      name: seed-7
      offline: true
      save_dir: ./tracking/wandb
      log_model: false
      save_code: false
      tags: [baseline, offline]
```

Conserva el directorio generado si puede sincronizarse después con `wandb sync DIRECTORY`.
Lightning rechaza `offline: true` junto con `log_model: true` o `"all"`.

### Online

El modo online es el predeterminado y publica durante el entrenamiento:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.WeightsAndBiasesTrackingLogger
    params:
      project: lambdaforge-research
      entity: my-team
      name: seed-7
      group: comparison-2026
      job_type: train
      tags: [baseline]
      offline: false
      log_model: false
      save_code: false
```

Autentícate fuera del YAML, por ejemplo con la CLI del proveedor o `WANDB_API_KEY`.
`offline: false` puede realizar peticiones de red y recopilar metadata del proveedor. Para que una
política aislada de la red resulte inequívoca, define `offline: true` en YAML y/o
`WANDB_MODE=offline` en el entorno de lanzamiento. `version` e `id` se refieren a la misma identidad
de run del proveedor; establece como máximo uno. Del mismo modo, `dir` es alias de `save_dir`.

## Referencia completa de parámetros

Los adaptadores reflejan los constructores de logger de la versión de Lightning instalada. Los
valores siguientes son los defaults públicos de LambdaForge.

### `MLflowTrackingLogger`

| Parámetro | Default | Significado |
|---|---|---|
| `experiment_name` | `"lightning_logs"` | Nombre del experimento/contenedor MLflow. |
| `run_name` | `null` | Nombre de run opcional legible por humanos. |
| `tracking_uri` | `null` | URI local/del proveedor; `null` usa `MLFLOW_TRACKING_URI` si existe y, si no, el `save_dir` local. |
| `tags` | `null` | Metadata de etiquetas enviada a MLflow. |
| `save_dir` | `"./mlruns"` | Almacenamiento local usado cuando no se indica URI de tracking. |
| `log_model` | `false` | Política de publicación de checkpoints `false`, `true` o `"all"` admitida por Lightning. |
| `prefix` | `""` | Prefijo añadido a las claves de métricas publicadas. |
| `artifact_location` | `null` | Raíz de artefactos explícita para un experimento recién seleccionado. |
| `run_id` | `null` | Run previo del proveedor que reutilizar; `null` crea/selecciona normalmente. |
| `synchronous` | `null` | Indicación opcional de logging síncrono; un valor no `null` falla en versiones de Lightning compatibles antiguas que carecen del parámetro. |

### `TensorBoardTrackingLogger`

| Parámetro | Default | Significado |
|---|---|---|
| `save_dir` | obligatorio | Directorio raíz o URL de sistema de archivos admitida. |
| `name` | `"lightning_logs"` | Directorio del experimento; un string vacío omite este nivel. |
| `version` | `null` | Directorio entero/string del run o versión autogenerada. |
| `log_graph` | `false` | Si Lightning debe registrar el grafo cuando las entradas del modelo lo permiten. |
| `default_hp_metric` | `true` | Si se añade la métrica de hiperparámetros predeterminada de TensorBoard. |
| `prefix` | `""` | Prefijo añadido a las claves de métricas publicadas. |
| `sub_dir` | `null` | Directorio opcional bajo la versión seleccionada. |
| `**kwargs` | ninguno | Opciones adicionales de `SummaryWriter` como `max_queue`, `flush_secs` o `filename_suffix`. En YAML se escriben directamente bajo `params`. |

### `WeightsAndBiasesTrackingLogger`

| Parámetro | Default | Significado |
|---|---|---|
| `name` | `null` | Nombre de run W&B legible por humanos. |
| `save_dir` | `"."` | Directorio local de metadata/run W&B. |
| `version` | `null` | ID/identidad de reanudación del proveedor; alias de `id`. |
| `offline` | `false` | Guardar en local en vez de sincronizar en vivo. |
| `dir` | `null` | Alias de `save_dir`. |
| `id` | `null` | Alias de `version`. |
| `anonymous` | `null` | Elección de logging anónimo del proveedor. |
| `project` | `null` | Proyecto W&B; se aplica el fallback del entorno/proveedor si falta. |
| `log_model` | `false` | Publicación de artefactos checkpoint `false`, `true` o `"all"`. |
| `experiment` | `null` | Objeto Run W&B previo, principalmente para Python directo/inyección. |
| `prefix` | `""` | Prefijo añadido a las claves de métricas publicadas. |
| `checkpoint_name` | `null` | Nombre opcional del artefacto checkpoint W&B. |
| `add_file_policy` | `"mutable"` | Política para añadir archivos; `"mutable"` se omite en versiones de Lightning compatibles antiguas y solicitar allí `"immutable"` falla. |
| `**kwargs` | ninguno | Argumentos adicionales de `wandb.init` como `entity`, `group`, `tags`, `notes`, `job_type`, `save_code` o `settings`. En YAML se escriben directamente bajo `params`. |

La versión de Lightning/proveedor instalada rechaza valores no compatibles o mutuamente
excluyentes. LambdaForge no descarta silenciosamente opciones arbitrarias del proveedor.

## Métricas, privacidad, checkpoints y artefactos

- `TaskLoggingConfig.logger` controla si las pérdidas/métricas de la tarea llegan al logger
  Lightning seleccionado. Los callbacks del framework como `EpochStats` también publican escalares
  de tiempo y memoria por época; los filtros de la barra de progreso y del CSV denso no redactan un
  logger remoto.
- Se recomienda conservar `write_epoch_metrics_csv: true`. Crea el artefacto denso y neutral
  respecto al proveedor aunque el tracker deje de estar disponible más adelante.
- Cualquier secreto escrito en YAML se copia a artefactos de configuración/procedencia
  materializados. Usa variables de entorno, almacenes de secretos o archivos de credenciales del
  proveedor para tokens y contraseñas.
- Revisa nombres de métricas, etiquetas, nombres de run, notas, hiperparámetros, registro de grafos,
  captura del código fuente, telemetría del sistema y ajustes del proveedor antes de habilitar un
  destino remoto. Estos adaptadores no suben automáticamente muestras del dataset, aunque un
  callback o llamada propia al logger sí podría hacerlo.
- `log_model` vale `false` por defecto en todos los adaptadores con red. `true` o `"all"` puede
  copiar checkpoints al almacenamiento gestionado por el proveedor, elevar ancho de banda/coste y
  conservar una copia remota después de que la retención local de LambdaForge pode el original.
- La retención de artefactos de LambdaForge gobierna rutas dentro del árbol local de la suite. No
  puede eliminar, revertir, verificar ni aplicar políticas de retención a artefactos MLflow/W&B,
  archivos TensorBoard remotos o archivos offline colocados fuera de ese árbol.
- Cada proceso de entrenamiento spawn crea su propio logger del proveedor. Usa IDs de run y
  elecciones de directorio/versión únicos; compartir un ID explícito o flujo TensorBoard entre
  semillas simultáneas mezcla su ciclo de vida y procedencia científica.

## Loggers propios y plugins

Los adaptadores no cierran la superficie de extensión. Todavía se puede seleccionar un logger
específico del proyecto mediante:

```yaml
trainer:
  logger:
    target: my_project.logging.ProjectLogger
    params: {endpoint: local}
```

Las distribuciones reutilizables pueden publicar una clase en el grupo de entry points
`lambdaforge.loggers` y utilizar:

```yaml
trainer:
  logger:
    plugin: {kind: logger, name: project_logger}
    params: {endpoint: local}
```

Ambas formas deben cumplir el contrato público
`lambdaforge.integrations.Lightning.Logger`. La frontera de plugin añade discovery lazy,
comprobación del contrato y procedencia de plugins cargados; un `target` directo es más sencillo
para código local del proyecto. Los tres adaptadores de tracking incluidos utilizan targets
públicos directos y no son plugins de entry point. Consulta la [guía de plugins](../plugins/README.es.md).

Sustituye `runner.target` únicamente cuando cambie el propio backend de entrenamiento. Una
integración de logger pertenece tras `trainer.logger` y no debe ser responsable de planificación,
checkpoints, agregación ni limpieza de procesos.

## Fallos de dependencias y proveedores

Importar `lambdaforge` o `lambdaforge.tracking` no importa `mlflow`, `tensorboard` ni `wandb`.
Construir un adaptador seleccionado ejecuta primero `TrackingDependencyGuard`. Si falta el SDK,
`TrackingDependencyError` —subclase de `ImportError`— identifica el backend e indica el comando de
instalación mínimo, por ejemplo:

```text
Tracking backend 'mlflow' requires the optional dependency 'mlflow'. Install the recommended backend with: pip install 'lambdaforge[mlflow]'
```

`lambdaforge validate` comprueba la ruta de clase sin construirla, por lo que validar correctamente
la estructura/import no demuestra que el SDK opcional, las credenciales, el servidor remoto o el
sistema de archivos funcionen. Ejecuta un run real mínimo en el entorno de despliegue.

Tras comprobar la dependencia, autenticación, red, permisos, parámetros inválidos, servidores no
disponibles y errores de finalización siguen siendo fallos del proveedor/Lightning. Hacen fallar
normalmente el run de entrenamiento propietario; LambdaForge no los oculta ni cambia
silenciosamente de tracking online a local.

## Referencias oficiales

- [Logger MLflow de Lightning](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html)
- [Servidor de tracking MLflow](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [Logger TensorBoard de Lightning](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html)
- [TensorBoard/`SummaryWriter` de PyTorch](https://docs.pytorch.org/docs/stable/tensorboard.html)
- [Logger W&B de Lightning](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html)
- [Inicialización y modos de runs W&B](https://docs.wandb.ai/models/ref/python/functions/init)
- [Variables de entorno W&B](https://docs.wandb.ai/models/track/environment-variables)
