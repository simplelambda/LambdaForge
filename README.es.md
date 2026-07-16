# LambdaForge

Español · [English](README.md)

LambdaForge es el framework orientado a objetos de SimpleLambda para entrenar modelos de aprendizaje
automático de forma reproducible. Combina PyTorch, Lightning y un motor de experimentos YAML en un
único paquete estable, para que un proyecto de investigación se concentre en sus datos y su tarea en
vez de volver a crear bucles de entrenamiento, registro de métricas, barridos de semillas, carga de
checkpoints, gráficas y planificación multiproceso sobre varias GPU.

> **Estado:** `0.1.0`, utilizable pero anterior a 1.0. Los espacios de nombres públicos documentados
> aquí forman la API prevista; todavía no se garantiza compatibilidad entre versiones menores. El
> repositorio aún no contiene una licencia, por lo que SimpleLambda debe decidir sus condiciones de
> redistribución.

## Contenidos

- [Qué proporciona LambdaForge](#qué-proporciona-lambdaforge)
- [Instalación](#instalación)
- [Inicio rápido](#inicio-rápido)
- [API pública](#api-pública)
- [Arquitectura](#arquitectura)
- [Referencia de experimentos YAML](#referencia-de-experimentos-yaml)
- [Ejecución y seguridad de procesos](#ejecución-y-seguridad-de-procesos)
- [Salidas, reanudación y carga](#salidas-reanudación-y-carga)
- [Componentes incluidos](#componentes-incluidos)
- [Contratos de extensión](#contratos-de-extensión)
- [Hallazgos de la revisión](#hallazgos-de-la-revisión)
- [Desarrollo y verificación](#desarrollo-y-verificación)
- [Limitaciones actuales](#limitaciones-actuales)
- [Mapa de documentación](#mapa-de-documentación)
- [Hoja de ruta propuesta](#hoja-de-ruta-propuesta-no-implementada)

## Qué proporciona LambdaForge

- Una tarea genérica de Lightning para lotes con forma de mapa, una o más pérdidas y métricas
  independientes de entrenamiento, validación y prueba.
- Construcción de objetos desde YAML de confianza mediante rutas completas `target` y `ref`.
- Productos cartesianos de parámetros, ablaciones con nombre y expansión reproducible de semillas.
- Ejecución secuencial, varios entrenamientos independientes por GPU, planificación multi-GPU y un
  trabajo DDP por grupo de dispositivos.
- Cancelación cooperativa, limpieza forzada del árbol de procesos, Job Objects de Windows, guardas de
  muerte del padre en Linux y workers protegidos del `DataLoader` de PyTorch.
- Configuraciones materializadas, logs, CSV densos por época, checkpoints, manifiestos de resultado,
  resúmenes entre semillas, comparaciones estadísticas y gráficas.
- MLP, CNN 2D, modelo de grafos ECMP, activaciones, normalizaciones, operadores de pooling,
  distancias y métricas binarias, multiclase y de regresión reutilizables.
- Una fachada pequeña (`LambdaForge`), una API de objetos (`Experiment`) y una CLI (`lambdaforge`).

LambdaForge es agnóstico respecto a la tarea en sus capas de configuración y orquestación. El
proyecto usuario aporta el `Dataset`, el collator opcional y, cuando el contrato de mapas por defecto
no basta, su propio modelo, tarea, módulo de datos o runner.

## Instalación

Se necesita Python 3.10 o posterior. Desde un clon, crea un entorno e instala el proyecto en modo
editable:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

En Linux/macOS la activación es `source .venv/bin/activate`. Las dependencias de ejecución incluyen
PyTorch, Lightning, TorchMetrics, NumPy, Matplotlib, PyYAML, psutil y threadpoolctl. `pywin32` se
instala solo en Windows. La versión de CUDA debe corresponderse con la compilación de PyTorch;
LambdaForge no instala drivers CUDA.

El paquete prefiere `lightning.pytorch` y conserva compatibilidad en ejecución con el nombre antiguo
`pytorch_lightning`.

## Inicio rápido

Copia [el ejemplo completo](examples/experiment.yaml), sustituye las tres rutas `your_project.*` y
revisa la suite expandida antes de iniciar procesos:

```powershell
lambdaforge inspect examples\experiment.yaml
lambdaforge run examples\experiment.yaml --dry-run
lambdaforge run examples\experiment.yaml
```

La API Python equivalente es:

```python
from lambdaforge import LambdaForge

experiment = LambdaForge.experiment("examples/experiment.yaml")
expanded_runs = experiment.expand()
results = experiment.run()
```

Para regenerar agregados sin reentrenar:

```powershell
lambdaforge aggregate examples\experiment.yaml
```

Las opciones de ejecución de la CLI solo sustituyen los campos de recursos YAML correspondientes:

```powershell
lambdaforge run experiment.yaml --mode parallel --gpus 0,1 --jobs-per-gpu 2
lambdaforge run experiment.yaml --mode ddp --gpus 0,1 --devices-per-job 2
```

## API pública

Los puntos de entrada admitidos son deliberadamente reducidos:

| Punto de entrada | Finalidad |
|---|---|
| `from lambdaforge import LambdaForge` | Cargar, ejecutar o construir objetos mediante la fachada. |
| `from lambdaforge import Experiment` | Inspeccionar, ejecutar, agregar y cargar una suite. |
| `lambdaforge.nn` | Modelos y registro de componentes compatibles con YAML. |
| `lambdaforge.metrics` | Contrato base y métricas incluidas. |
| `lambdaforge.training` | Tarea, runner, configuración y orquestación de procesos. |
| `lambdaforge.experiments` | Configuración, planificación, agregación y carga de bajo nivel. |
| `python -m lambdaforge` / `lambdaforge` | Interfaz de terminal para la misma API de objetos. |

`LambdaForge.build(spec)` expone la factoría genérica:

```python
model = LambdaForge.build({
    "target": "lambdaforge.nn.models.MLP",
    "params": {"in_features": 32, "out_features": 1, "hidden": [64, 32]},
})
```

Importa desde estos espacios de nombres y no dependas de ubicaciones de archivos. Los módulos
internos podrán moverse mientras se mantengan los imports públicos.

## Arquitectura

```text
LambdaForge/
├── examples/                     # plantillas de configuración
├── src/lambdaforge/
│   ├── LambdaForge.py            # fachada única y fácil de descubrir
│   ├── cli/                      # objeto de línea de comandos
│   ├── experiments/              # YAML, barridos, ejecución y agregación
│   ├── integrations/             # adaptadores de compatibilidad externos
│   ├── metrics/                  # contratos; familias binaria/multiclase/regresión
│   ├── nn/                       # modelos, pérdidas y componentes neuronales
│   └── training/                 # núcleo Lightning más callbacks/datos/orquestación
├── tests/                        # pruebas unitarias, de procesos y entreno real
└── pyproject.toml                # configuración del paquete y herramientas
```

La implementación sigue la filosofía de objetos con influencia de Java del proyecto:

- cada `.py` de implementación contiene una clase;
- el comportamiento reutilizable vive en objetos, métodos de clase o métodos estáticos, no en
  funciones auxiliares de módulo;
- `__init__.py` y `__main__.py` son puntos de entrada del paquete y las excepciones intencionadas;
- los enums sustituyen conjuntos cerrados de strings mágicos internos;
- las claves YAML y rutas completas de importación continúan siendo strings porque son fronteras de
  protocolo externas;
- las responsabilidades permanecen separadas aunque eso produzca varios archivos pequeños.

PEP 8 suele favorecer módulos cortos en minúsculas; hacer coincidir clase y módulo es por tanto una
convención intencionada de LambdaForge, aplicada por coherencia y no presentada como estilo Python
universal. Los namespaces de paquete en minúsculas y los reexports públicos evitan que esa elección
afecte a la mayoría de imports consumidores.

Los subpaquetes se crean por fronteras conceptuales estables, no al alcanzar un número arbitrario de
archivos. Clasificación se divide en `binary` y `multiclass`; entrenamiento separa `callbacks`,
`data` y `orchestration`. `nn.pooling` permanece plano pese a su tamaño porque todas sus clases
implementan un contrato estrechamente relacionado; dividirlo en carpetas diminutas por técnica
dificultaría comparar y descubrir operadores. Los imports públicos se reexportan desde
`__init__.py`, de modo que la organización física no afecta al código usuario.

Los anteriores árboles raíz `models`, `metrics`, `distances`, `training` y `experiments` forman ahora
un único paquete instalable `src/lambdaforge`. Los imports ya no dependen de que exista en
`sys.path` otro paquete ajeno llamado `core`.

## Referencia de experimentos YAML

El archivo de ejemplo es la plantilla canónica. Sus bloques superiores son:

| Bloque | Obligatorio | Significado |
|---|---:|---|
| `experiment` | sí | Nombre, salida, semillas, reanudación y política de finalización. |
| `data` | sí | Objetos de entreno/validación/prueba y configuración del módulo de datos. |
| `model` | sí | Especificación del objeto modelo. |
| `losses` | sí | Uno o más objetos de pérdida. |
| `metrics` | no | Métricas compatibles compartidas entre etapas salvo sustitución. |
| `train_metrics`, `val_metrics`, `test_metrics` | no | Listas explícitas por partición. |
| `optimizer` | no | Referencia y parámetros; por defecto AdamW. |
| `scheduler` | no | Referencia, parámetros y metadatos Lightning opcionales. |
| `task` | no | Tarea personalizada o parámetros de `LightningTask`. |
| `trainer` | no | Campos de `LightningTrainConfig` y `trainer_kwargs` avanzados. |
| `runner` | no | Runner personalizado, callbacks o parámetros. |
| `callbacks` | no | Objetos callback adicionales construidos desde YAML. |
| `sweep` | no | Inclusión de base, producto cartesiano y ablaciones. |
| `execution` | no | Recursos secuenciales, paralelos o DDP. |

### Especificaciones de objetos

`target` importa un callable, construye recursivamente sus parámetros y lo invoca:

```yaml
model:
  target: lambdaforge.nn.models.MLP
  params:
    in_features: 32
    out_features: 1
    activation: gelu
```

`ref` importa un objeto sin invocarlo, útil para optimizadores y collators:

```yaml
optimizer:
  ref: torch.optim.AdamW
  params: {lr: 0.001}
```

Las especificaciones pueden anidarse en diccionarios y listas. Ejecutan imports de Python, por lo
que los YAML deben ser de confianza; es configuración, no un entorno aislado.

### Experimento y barrido

```yaml
experiment:
  name: study_name
  output_root: runs/experiments
  seeds: [7, 17, 27]
  resume: true
  rerun_completed: false
  test_after_fit: true
  required_artifacts: [predictions.csv]

sweep:
  include_base: true
  grid:
    model.params.hidden: [[128, 64], [256, 128]]
    optimizer.params.lr: [0.001, 0.0003]
  ablations:
    - name: no_dropout
      set: {model.params.dropout: 0.0}
```

Las claves del grid son rutas con puntos y su producto cartesiano se materializa para cada semilla.
Las ablaciones son configuraciones adicionales con nombre. Las listas vacías y las ejecuciones
duplicadas `(variant, seed)` se rechazan antes de entrenar.

Con `resume: true`, una ejecución incompleta puede continuar desde su último checkpoint. Con
`rerun_completed: false`, una ejecución correcta solo se omite si existen `result.json`, el
checkpoint seleccionado y todas las rutas relativas de `required_artifacts`. Esas rutas son
genéricas; el framework no presupone ningún archivo de predicciones propio de un dominio.

### Vías avanzadas de Trainer y DataLoader

Los campos habituales de Trainer son explícitos y se validan. Las opciones futuras o menos comunes
de Lightning se pasan por `trainer_kwargs`:

```yaml
trainer:
  max_epochs: 100
  precision: bf16-mixed
  checkpoint_policy: last_and_best   # none, last, best, last_and_best, all
  checkpoint_monitor: val_auroc
  checkpoint_mode: max
  logger: csv                        # none, csv, lightning_csv u objeto
  write_epoch_metrics_csv: true      # entrada canónica de informes/agregación
  epoch_metrics_include: ["train_*", "val_*", "epoch_time_s"]
  epoch_console_exclude: ["*_loss_binary_cross_entropy_with_logits"]
  trainer_kwargs:
    limit_train_batches: 1.0
    enable_model_summary: true
```

`trainer.logger` puede ser una especificación `target` anidada de cualquier logger Lightning
compatible. Lo mismo se aplica a `callbacks` en el nivel superior. `write_epoch_metrics_csv`
conserva el artefacto denso canónico de LambdaForge con independencia del logger externo; se
desactiva solo cuando no se necesita agregación. El CSV y la tabla de terminal aceptan patrones
include/exclude, mientras `checkpoint_monitor`, `early_stopping_monitor` y sus modos `min`/`max`
eliminan la dependencia del orden de las métricas.

La publicación desde la tarea se configura de forma independiente:

```yaml
task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_key: x
    logging:
      log_total_loss: true
      log_individual_losses: true
      loss_prog_bar: false
      metric_prog_bar: true
      logger: false                    # true con un logger externo/de Lightning
```

Las listas de métricas determinan qué se calcula. `TaskLoggingConfig` decide qué pérdidas se publican
y qué llega a la barra/logger. `MetricAlias` permite dar nombres distintos a dos instancias de la
misma métrica, por ejemplo accuracy con dos umbrales.

De igual modo, `LightningDataModule` acepta `dataloader_kwargs` comunes y específicos por partición.
No se pueden sustituir desde esos diccionarios las claves gestionadas por LambdaForge (`dataset`,
`shuffle`, `worker_init_fn` y los campos explícitos), evitando configuraciones contradictorias.

## Ejecución y seguridad de procesos

| Modo | Planificación |
|---|---|
| `sequential` | Las ejecuciones se realizan en el proceso llamador, una tras otra. Los slots GPU se ignoran. |
| `parallel` | Cada ejecución es un proceso `spawn` en una GPU lógica; `jobs_per_gpu` permite trabajos independientes simultáneos por GPU. |
| `ddp` | Cada ejecución recibe un grupo de `devices_per_job` GPU lógicas y Lightning ejecuta DDP dentro de ese proceso. |

Los números GPU son posiciones lógicas respecto a `CUDA_VISIBLE_DEVICES` del padre. Si el padre
tiene `CUDA_VISIBLE_DEVICES=4,7`, un trabajo que pida `[1]` ve la GPU física `7` como GPU local `0`.
LambdaForge nunca reescribe la variable del padre.

Los límites de hilos CPU, hilos inter-op, afinidad y workers de datos se aplican por trabajo. Usa
valores conservadores al compartir una GPU: la concurrencia multiplica el uso de CPU y RAM aunque
quepa en memoria GPU.

La limpieza tiene varias capas:

1. SIGINT/SIGTERM activa un evento compartido de parada.
2. Los callbacks de Lightning paran en un límite de lote.
3. El orquestador espera el periodo de gracia configurado.
4. Los árboles descendientes restantes se terminan recursivamente con psutil.
5. En Windows se usa un Job Object con finalización al cerrar; en Linux los workers instalan una
   señal de muerte del padre cuando la plataforma lo permite.
6. Los workers de `DataLoader` instalan la misma guarda y límites de hilos.

Esto protege los procesos creados por el framework y sus descendientes en el mismo árbol del sistema
operativo. Ninguna biblioteca puede garantizar la limpieza de un programa externo que se desacople
intencionadamente como servicio independiente; esos launchers requieren su propio contrato de vida.

En DDP, las métricas sincronizan el estado acumulado antes de calcular valores no lineales como
AUROC, F1 o correlaciones. Una métrica personalizada en DDP debe implementar el contrato de estado
distribuido; LambdaForge genera un error en vez de promediar silenciosamente escalares inválidos por
rank.

## Salidas, reanudación y carga

Cada ejecución concreta tiene su directorio bajo `<output_root>/<experiment>/<variant>/<seed>/` y
puede contener:

- `config.yaml`: configuración completamente materializada;
- `hparams.json`: resumen compacto de hiperparámetros;
- `train.log`: stdout/stderr capturado;
- `metrics.csv`: una fila densa por época;
- `checkpoints/`: archivos según `checkpoint_policy`;
- `result.json`: estado terminal, rutas, duración y métricas mejores/finales;
- artefactos requeridos definidos por el proyecto.

El área agregada contiene CSV por época y variante, estadísticas entre semillas, comparaciones por
pares, q-valores direccionales Benjamini-Hochberg y gráficas PNG opcionales. Puede regenerarse desde
disco sin reconstruir el modelo.

Carga un modelo desde la suite:

```python
experiment = LambdaForge.experiment("experiment.yaml")
model = experiment.load_model(seed=17, variant="base", which="best")
```

`which` acepta `best`, `last` o `auto`. La carga valida el checkpoint y entiende estados de modelo
directos y claves Lightning con prefijo `model.`. La clase del modelo debe seguir siendo importable
desde la configuración materializada.

## Componentes incluidos

- Modelos: `Model`, `MLP`, `CNN2D`, `BatchedKNN`, `ECMP`.
- Activaciones: ELU, GELU, Identity, LeakyReLU, ReLU, Sigmoid, SiLU y Tanh.
- Normalizaciones: BatchNorm (1D/2D/3D), IdentityNorm, LayerNorm y RMSNorm.
- Pooling: atención, atención gated/multi-head, auto-pool, fractional/top-k, log-sum-exp, máximo,
  media, mínimo, momentos, noisy-or, GeM de probabilidad, softmax y suma.
- Distancias: euclídea y euclídea al cuadrado.
- Pérdidas: contrato base ponderado y entropía cruzada binaria con logits.
- Métricas binarias: accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, kappa de
  Cohen, AUROC y AUPRC.
- Métricas multiclase: accuracy, balanced accuracy, F1, AUROC y AUPRC.
- Regresión: MAE, MSE, RMSE, R², correlaciones Pearson y Spearman, además de `MeanMetric`.

Los nombres cortos de activaciones y normalizaciones no distinguen mayúsculas en YAML. Se pueden
añadir alias mediante `ComponentRegistry`; los constructores Python también aceptan clases
compatibles. Consulta la [guía de componentes neuronales](src/lambdaforge/nn/README.es.md) y la
[guía de métricas](src/lambdaforge/metrics/README.es.md).

## Contratos de extensión

### Modelo

Hereda de `torch.nn.Module` o `lambdaforge.nn.models.Model` e implementa
`forward(*args, **kwargs)`. `predict` activa evaluación/inferencia y restaura el modo anterior. Para
la tarea por defecto, devuelve un tensor —se envuelve bajo `model_output_key`— o un mapa.

### Pérdida

Hereda de `Loss` e implementa `forward(outputs, batch) -> Tensor`. Da a cada pérdida un `name`
estable y usa claves de mapa en vez de supuestos de dominio. Varias pérdidas se suman tras aplicar
sus pesos.

### Métrica

Implementa `update`, `compute` y `reset`. Para DDP, expón además `distributed_state` y
`merge_distributed_state`, o usa una métrica del framework. Las instancias se copian profundamente
por etapa para que no se mezcle estado entre entreno, validación y prueba.

Cada etapa exige nombres de métrica únicos. Envuelve una métrica con `MetricAlias` si usas la misma
clase varias veces con parámetros distintos. Las listas explícitas por etapa evitan necesitar
`deepcopy` cuando una métrica propia mantiene un recurso externo no copiable.

### Datos y tarea

`LightningTask` espera lotes con forma de mapa. `model_input_key` elige la entrada y cada
pérdida/métrica elige sus claves. Para tuplas, varias entradas, tareas generativas o flujos de
optimización especiales, configura un `task.target` propio; el resto de experimentos, procesos y
artefactos sigue siendo reutilizable.

### Runner

Un runner propio debe ofrecer métodos `fit` y `test` compatibles. Se configura con `runner.target` y
sus parámetros se construyen recursivamente. Los callbacks adicionales también pueden declararse
como objetos YAML.

`ObjectFactory` construye recursivamente todos estos objetos: los modelos pueden ser cualquier
`torch.nn.Module`; las pérdidas heredan de `Loss`; las métricas, de `Metric`; loggers y callbacks
implementan sus contratos Lightning. Para lotes especiales, varios optimizadores u otro backend se
sustituye `task.target` o `runner.target` sin cambiar el motor de experimentos.

## Hallazgos de la revisión

La revisión completa encontró y resolvió los siguientes riesgos estructurales o de corrección:

| Hallazgo en la organización anterior | Resolución |
|---|---|
| Las carpetas raíz no formaban un paquete instalable y los imports apuntaban a `core`/`coreold` externos. | Se añadieron `pyproject.toml`, `src/lambdaforge` e imports absolutos autocontenidos. |
| Modelos, distancias, experimentos, métricas y entreno competían como entradas raíz. | Se crearon la API `LambdaForge`/`Experiment` y cuatro subpaquetes públicos cohesivos. |
| Módulos grandes contenían muchas funciones sueltas o varias clases sin relación. | El comportamiento pasó a objetos colaboradores y cada clase de implementación ocupa un archivo. |
| `training` y `metrics.classification` se habían vuelto visualmente densos. | Se dividieron solo por contratos estables (`callbacks`, `orchestration`, `binary`, `multiclass`) conservando los imports públicos; familias cohesivas como `pooling` siguen planas. |
| Opciones cerradas y componentes repetían strings literales. | Se añadieron enums y `ComponentRegistry`; los strings quedan en fronteras YAML/serialización. |
| Opciones avanzadas de Trainer/DataLoader exigían editar el código. | Se añadieron `trainer_kwargs` y `dataloader_kwargs` comunes/específicos con validación. |
| Una sola lista de métricas y el orden implícito del monitor limitaban el control. | Se añadieron listas por etapa, alias, modos de monitor explícitos, política de publicación de pérdidas y filtros de CSV/terminal. |
| Elegir un logger Lightning propio eliminaba el CSV canónico de época. | Se separó el logger externo de `write_epoch_metrics_csv`, de modo que los informes siguen disponibles por defecto. |
| DDP podía promediar escalares AUROC/F1/correlación ya calculados, algo matemáticamente incorrecto. | Las métricas reúnen y fusionan estado antes de calcular; una métrica no compatible falla explícitamente. |
| `CNN2D` elegía BatchNorm 1D por defecto para tensores NCHW. | El default incluido crea ahora `BatchNorm2d`. |
| `Model.predict` no garantizaba restaurar el modo de entrenamiento previo. | La inferencia usa `try/finally` y recupera el estado original. |
| `test_after_fit` pedía a un Trainer nuevo un checkpoint `best` desconocido y omitía la parada. | Usa el checkpoint real si existe, o los pesos actuales, y conserva la cancelación. |
| La terminación forzada se centraba en workers raíz y era frágil en Windows. | Se añadieron limpieza recursiva, Job Objects, muerte del padre y workers de datos protegidos. |
| La finalización presuponía un artefacto de predicción específico de dominio. | Se sustituyó por rutas relativas genéricas `required_artifacts`. |
| La documentación describía módulos/scripts obsoletos y no había guía raíz. | Se reemplazó por guías enlazadas en inglés/español verificadas frente al código. |

La carencia principal que queda es la gestión de memoria/caché de datasets, no la ejecución de
entrenos. Se declara abajo y se propone como trabajo futuro en vez de presentarla como terminada.

## Desarrollo y verificación

```powershell
ruff format --check src tests
ruff check src tests
mypy src\lambdaforge
pytest -q
```

La suite actual cubre expansión, construcción de objetos, validación de modelos, métricas,
agregación, planificación con procesos `spawn`, reglas estructurales POO, un entrenamiento Lightning
real de una época en CPU y la construcción YAML de modelo, pérdida, métrica, logger y callback
externos. No se afirma cobertura CUDA cuando CUDA no se ha ejercitado. Los cambios de planificación
o limpieza deben validarse además en el host multi-GPU objetivo e interrumpirse manualmente al menos
una vez.

Todos los módulos y clases fuente tienen docstring. La auditoría comprueba también que nombre de
clase y módulo coincidan, una clase por archivo y la ausencia de funciones auxiliares de módulo en
implementaciones.

## Limitaciones actuales

- LambdaForge envuelve datasets y data loaders de PyTorch, pero **todavía no** implementa caché de
  datasets, memory mapping, streaming ni presupuesto de RAM. El almacenamiento y carga pertenecen al
  proyecto usuario.
- Lightning es el único backend de entrenamiento incluido.
- La tarea predeterminada presupone lotes supervisados con forma de mapa; otros formatos necesitan
  una tarea propia.
- La validación YAML es estructural y se apoya en constructores; aún no hay un JSON Schema publicado.
- Las métricas de curva acumulan predicciones en CPU y pueden consumir mucha RAM en validaciones
  enormes.
- Los resúmenes estadísticos son exploratorios, no sustituyen el protocolo de cada estudio. Algunos
  intervalos usan aproximaciones normales.
- No se incluye tracker remoto, planificador de clúster, optimizador de hiperparámetros ni almacén de
  artefactos.
- Los procesos Windows/CPU se han probado localmente; las pruebas multi-GPU y de interrupción abrupta
  dependen del entorno.

## Mapa de documentación

- [Sistema de experimentos](src/lambdaforge/experiments/README.es.md) · [English](src/lambdaforge/experiments/README.md)
- [Entrenamiento y procesos](src/lambdaforge/training/README.es.md) · [English](src/lambdaforge/training/README.md)
- [Componentes neuronales](src/lambdaforge/nn/README.es.md) · [English](src/lambdaforge/nn/README.md)
- [Métricas](src/lambdaforge/metrics/README.es.md) · [English](src/lambdaforge/metrics/README.md)
- [Ejemplo YAML completo](examples/experiment.yaml)

Cada guía enlaza de vuelta aquí y a su traducción. Los docstrings de clase son la referencia más
precisa para los argumentos de cada constructor.

## Hoja de ruta propuesta (no implementada)

Los siguientes refinamientos ofrecen mucho valor frente a su coste probable. Son solo propuestas;
ninguno se presenta como funcionalidad actual.

1. **JSON Schema y `lambdaforge validate` publicados** (pequeño): detectar claves desconocidas,
   imports fallidos, rutas de barrido inválidas y contradicciones sin crear directorios.
2. **Manifiesto de entorno** (pequeño): guardar Python/plataforma, paquetes, CUDA/cuDNN, GPU y datos
   opcionales de commit/diff de Git junto al resultado.
3. **Objeto `DatasetCache`** (medio): LRU acotada en RAM y adaptadores de disco/memory-map, con claves
   explícitas y estadísticas de aciertos. Cerraría la mayor distancia entre la visión y el código.
4. **Descubrimiento de plugins por entry points** (pequeño/medio): registrar modelos, métricas y alias
   desde paquetes externos sin editar LambdaForge.
5. **Métricas de curva en streaming** (medio): aproximaciones por histogramas/cuantiles para limitar
   RAM en AUROC/AUPRC.
6. **Objetos tipados de resultado/manifiesto** (pequeño): sustituir diccionarios internos libres sin
   perder compatibilidad JSON.
7. **Métodos de comparación más sólidos** (pequeño): bootstrap y Wilcoxon pareado seleccionables en
   YAML.
8. **Matriz CI y pruebas de interrupción** (pequeño/medio): Python 3.10–3.13, Windows/Linux, imports
   Lightning moderno/antiguo y finalización de hijos/nietos.
9. **Adaptadores opcionales de tracking** (medio): objetos para MLflow, TensorBoard o Weights & Biases
   tras la frontera logger/runner existente, sin servicios obligatorios.
