<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" type="image/svg+xml" srcset="icons/lambdaforge-light.svg">
    <source media="(prefers-color-scheme: dark)" type="image/png" srcset="icons/lambdaforge-light.png">
    <source media="(prefers-color-scheme: light)" type="image/svg+xml" srcset="icons/lambdaforge-dark.svg">
    <source media="(prefers-color-scheme: light)" type="image/png" srcset="icons/lambdaforge-dark.png">
    <img src="icons/lambdaforge-dark.png" width="140" alt="Logo de LambdaForge">
  </picture>
</p>

# LambdaForge

Español · [English](README.md)

LambdaForge es el framework orientado a objetos de SimpleLambda para entrenar modelos de aprendizaje
automático de forma reproducible. Combina PyTorch, Lightning y un motor de experimentos YAML en un
único paquete estable, para que un proyecto de investigación se concentre en sus datos y su tarea en
vez de volver a crear bucles de entrenamiento, registro de métricas, barridos de semillas, carga de
checkpoints, gráficas y planificación multiproceso sobre varias GPU.

> **Estado:** `0.2.0`, utilizable pero anterior a 1.0. Los espacios de nombres públicos documentados
> aquí forman la API prevista; todavía no se garantiza compatibilidad entre versiones menores. El
> repositorio aún no contiene una licencia, por lo que SimpleLambda debe decidir sus condiciones de
> redistribución.

## Contenidos

- [Qué proporciona LambdaForge](#qué-proporciona-lambdaforge)
- [Instalación](#instalación)
- [Integración en otro proyecto](#integración-en-otro-proyecto)
- [Por qué existe AGENTS.md](#por-qué-existe-agentsmd)
- [Inicio rápido](#inicio-rápido)
- [API pública](#api-pública)
- [Arquitectura](#arquitectura)
- [Referencia de experimentos YAML](#referencia-de-experimentos-yaml)
- [Migraciones de configuración](#migraciones-de-configuración)
- [Ejecución y seguridad de procesos](#ejecución-y-seguridad-de-procesos)
- [Salidas, reanudación y carga](#salidas-reanudación-y-carga)
- [Retención de artefactos](#retención-de-artefactos)
- [Componentes incluidos](#componentes-incluidos)
  - [Modelos de grafo avanzados y equivariantes](#modelos-de-grafo-avanzados-y-equivariantes)
- [Contratos de extensión](#contratos-de-extensión)
- [Hallazgos de la revisión](#hallazgos-de-la-revisión)
- [Desarrollo y verificación](#desarrollo-y-verificación)
  - [Integración continua](#integración-continua)
- [Limitaciones actuales](#limitaciones-actuales)
- [Mapa de documentación](#mapa-de-documentación)
- [Hoja de ruta](#hoja-de-ruta)

## Qué proporciona LambdaForge

- Una tarea genérica de Lightning para lotes con forma de mapa, una o más pérdidas y métricas
  independientes de entrenamiento, validación y prueba.
- Construcción de objetos desde YAML de confianza mediante rutas completas `target`/`ref` o plugins
  instalados por entry points y validados por contrato, incluidos datasets, callbacks y loggers.
- Validación con Schema Draft 2020-12 de estructura, expansión, recursos e imports antes de ejecutar.
- Migraciones versionadas y preview-first con YAML round trip y salida atómica explícita.
- Productos cartesianos de parámetros, ablaciones con nombre y expansión reproducible de semillas.
- Ejecución secuencial, varios entrenamientos independientes por GPU, planificación multi-GPU y un
  trabajo DDP por grupo de dispositivos.
- Cancelación cooperativa, limpieza forzada del árbol de procesos, Job Objects de Windows, guardas de
  muerte del padre en Linux y workers protegidos del `DataLoader` de PyTorch.
- Configuraciones materializadas, procedencia del entorno y de plugins cargados aislada por run,
  logs, CSV densos por época, checkpoints, manifiestos de resultado, resúmenes entre semillas,
  comparaciones estadísticas y gráficas.
- Retención de artefactos que prioriza la previsualización, con selección de checkpoints por rol,
  ZIPs streaming verificados, reglas explícitas de poda, receipts de finalización y transacciones
  recuperables tras un crash.
- MLP, CNN 2D, modelo de grafos ECMP, activaciones, normalizaciones, operadores de pooling,
  distancias y métricas binarias, multiclase y de regresión reutilizables.
- Caché de datasets acotada por proceso, cuotas de disco/mmap coordinadas entre procesos,
  fingerprints explícitos de dataset/transformación, registros checksum/HMAC y codec seguro
  NumPy/Torch, todo seleccionable mediante la misma sintaxis YAML recursiva.
- Alternativas de AUROC/AUPRC binarias y multiclase de memoria fija cuya resolución, promedio y
  política de logits son parámetros explícitos del experimento.
- Adaptadores de logger MLflow, TensorBoard y Weights & Biases opcionales y cargados de forma lazy,
  seleccionables solos o juntos sin añadir un servicio de tracking a la instalación base.
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
PyTorch, Lightning, TorchMetrics, NumPy, Matplotlib, PyYAML, ruamel.yaml, JSON Schema, psutil y
threadpoolctl. `pywin32` se instala solo en Windows. La versión de CUDA debe corresponderse con la
compilación de PyTorch; LambdaForge no instala drivers CUDA.

El paquete prefiere `lightning.pytorch` y conserva compatibilidad en ejecución con el nombre antiguo
`pytorch_lightning`.

Los proveedores de tracking siguen siendo opcionales. Instala `lambdaforge[mlflow]`,
`lambdaforge[tensorboard]`, `lambdaforge[wandb]` o el extra combinado `lambdaforge[tracking]`;
consulta la [guía de tracking](src/lambdaforge/tracking/README.es.md) antes de habilitar publicaciones
remotas.

## Integración en otro proyecto

LambdaForge es una librería instalable, no un árbol de fuentes que haya que copiar dentro de cada
estudio. Crea un entorno virtual propio para el proyecto consumidor e instala ambos proyectos en
ese entorno. Para desarrollar el framework localmente:

```bash
cd /ruta/a/mi-proyecto-de-investigacion
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e /ruta/absoluta/a/LambdaForge
python -m pip install -e .
python -m pip check
python -c "import lambdaforge; print(lambdaforge.__version__)"
```

El último `pip install -e .` es importante: permite importar rutas como
`mi_proyecto.models.ModeloProyecto` cuando LambdaForge resuelve el YAML. Una estructura consumidora
normal es:

```text
mi-proyecto-de-investigacion/
├── pyproject.toml
├── experiments/baseline.yaml
├── src/mi_proyecto/
│   ├── datasets.py
│   ├── losses.py
│   └── models.py
└── tests/
```

Para una instalación estable u offline, construye un wheel en el checkout de LambdaForge e instala
ese artefacto inmutable en lugar de una ruta editable:

```bash
python -m pip wheel /ruta/absoluta/a/LambdaForge --no-deps --wheel-dir dist
python -m pip install dist/lambdaforge-0.2.0-py3-none-any.whl
```

Deja que el lock o constraints del proyecto consumidor seleccione primero una compilación PyTorch
compatible con su driver; LambdaForge acepta un `torch` compatible ya instalado. Comprueba el
entorno final: que `nvidia-smi` funcione no demuestra que el wheel de Python incluya CUDA.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
lambdaforge validate experiments/baseline.yaml
lambdaforge run experiments/baseline.yaml --dry-run
```

No copies `src/lambdaforge`, no compartas el `.venv` de LambdaForge ni modifiques `PYTHONPATH`: esas
vías ocultan errores de dependencias/import y dificultan reproducir un artículo. Usa un wheel o
versión por entorno para proyectos independientes. Para extensiones reutilizables entre paquetes,
publica plugins por entry point; para un solo proyecto, los targets instalados `mi_proyecto.*` son
más simples. Los [contratos de extensión](#contratos-de-extensión) muestran ambas vías.

## Por qué existe AGENTS.md

Un agente de programación no debería leer cientos de módulos de implementación y todos los README
especializados antes de poder configurar un modelo o añadir una loss. Ese enfoque consume contexto
y dinero, aumenta la probabilidad de olvidar restricciones leídas al principio y lleva al agente a
deducir APIs desde ficheros internos que no forman una interfaz estable.

[AGENTS.md](AGENTS.md) es por ello el manual operativo único y eficiente en tokens del framework.
Reúne el mapa completo de capacidades, las fronteras públicas soportadas, instalación y flujo YAML,
recetas de extensión, reglas de publicación de resultados, comandos de verificación y una pequeña
tabla de rutas para los pocos casos que requieren más detalle. El flujo previsto para un agente es:

1. Leer `AGENTS.md` una vez.
2. Elegir del catálogo un objeto público o contrato de extensión existente.
3. Inspeccionar solo la firma/docstring de ese objeto o la única guía propietaria indicada.
4. Validar y probar mediante los comandos públicos documentados.

No sustituye los README bilingües destinados a personas ni los docstrings precisos de cada clase.
Es un índice comprimido y un contrato de seguridad que evita cargar todo el repositorio en
contexto. Los agentes que trabajan en este checkout lo descubren automáticamente; si LambdaForge
se consume desde otro workspace, proporciona el fichero explícitamente o referencia su ruta desde
el `AGENTS.md` del proyecto consumidor. Los wheels también instalan ese mismo fichero fuente bajo
`share/lambdaforge/AGENTS.md`; obtén su ruta exacta dentro del entorno sin importar el framework:

```bash
python -c "from importlib.metadata import distribution; print(distribution('lambdaforge').locate_file('share/lambdaforge/AGENTS.md'))"
```

## Inicio rápido

Copia [el ejemplo completo](examples/experiment.yaml), sustituye sus rutas `your_project.*` y
valida y revisa la suite expandida antes de iniciar procesos:

```powershell
lambdaforge validate examples\experiment.yaml
lambdaforge inspect examples\experiment.yaml
lambdaforge run examples\experiment.yaml --dry-run
lambdaforge run examples\experiment.yaml
```

La API Python equivalente es:

```python
from lambdaforge import LambdaForge

experiment = LambdaForge.experiment("examples/experiment.yaml")
report = experiment.validate()
expanded_runs = experiment.expand()
results = experiment.run()
print(results[0].status, results[0]["status"])
```

Para regenerar agregados sin reentrenar:

```powershell
lambdaforge aggregate examples\experiment.yaml
lambdaforge retain examples\experiment.yaml          # plan de solo lectura
lambdaforge results examples\experiment.yaml --write-index
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
| `from lambdaforge import RunResult, AggregateResult` | Resultados tipados e inmutables compatibles con dict/JSON legado. |
| `from lambdaforge import ResultCatalog, ResultRecord` | Discovery por identidad y selección explícita del historial de intentos. |
| `from lambdaforge import ArtifactRetentionPlan, ArtifactRetentionResult` | Previsualizaciones y resultados tipados e inmutables de retención. |
| `lambdaforge.data` | Adaptadores de dataset y objetos de caché acotada. |
| `lambdaforge.nn` | Modelos y registro de componentes compatibles con YAML. |
| `lambdaforge.metrics` | Contrato base y métricas incluidas. |
| `lambdaforge.plugins` | Discovery lazy, sesiones de uso, descriptores y errores de resolución. |
| `lambdaforge.integrations` | Objeto estable de compatibilidad Lightning para autores de plugins. |
| `lambdaforge.tracking` | Adaptadores lazy y opcionales para loggers MLflow, TensorBoard y Weights & Biases. |
| `lambdaforge.training` | Tarea, runner, configuración y orquestación de procesos. |
| `lambdaforge.experiments` | Configuración, migraciones, planificación, agregación y carga de bajo nivel. |
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

`LambdaForge.validate(path)` y `Experiment.validate(check_imports=True)` devuelven un
`ValidationReport` inmutable. La CLI ofrece `--json` para automatización y `--no-imports` para
plantillas cuyo proyecto externo todavía no está instalado. La comprobación no instancia los
objetos, pero importar un módulo Python puede ejecutar su código de nivel superior, por lo que la
configuración debe seguir siendo de confianza.

`LambdaForge.preview_migration(path)` devuelve un `ExperimentConfigMigrationResult` inmutable y sin
escrituras. Los objetos de migración, versiones exactas del Schema, catálogos y formatos de
previsualización son públicos desde `lambdaforge.experiments`.

`LambdaForge.preview_retention(path)` y `Experiment.preview_retention()` construyen un plan
estrictamente de solo lectura. La mutación siempre es explícita mediante
`LambdaForge.apply_retention(path)`, `Experiment.apply_retention()` o
`lambdaforge retain ... --apply`; `retention.mode: apply` también puede ejecutarse, pero solo
después de que una agregación final correcta publique un receipt de finalización vigente.

La metadata de plugins instalados se puede inspeccionar sin importar sus módulos:

```powershell
lambdaforge plugins
lambdaforge plugins --kind metric --json
```

## Arquitectura

```text
LambdaForge/
├── .github/workflows/             # CI CPU alojada y CUDA self-hosted opcional
├── examples/                     # plantillas de configuración
├── src/lambdaforge/
│   ├── EnvironmentManifest.py     # procedencia tipada de la ejecución
│   ├── LambdaForge.py            # fachada única y fácil de descubrir
│   ├── cli/                      # objeto de línea de comandos
│   ├── data/                     # adaptadores seguros y backends de caché acotados
│   ├── experiments/              # YAML, barridos, ejecución, agregación y retención
│   ├── integrations/             # adaptadores de compatibilidad externos
│   ├── metrics/                  # contratos; familias binaria/multiclase/regresión
│   ├── nn/                       # modelos, pérdidas y componentes neuronales
│   ├── plugins/                  # extensiones lazy desde paquetes instalados
│   ├── runtime/                  # locks cross-process compartidos de archivos
│   ├── schemas/                  # JSON Schema de experimentos empaquetado
│   ├── tracking/                 # adaptadores logger opcionales y guardas de dependencias
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
| `schema_version` | sí | Versión de compatibilidad exacta y entrecomillada; el valor actual es `"1.1"`. Los archivos históricos 1.0 y sin versión migran hacia delante antes de validar. |
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
| `aggregation` | no | Intervalos, pruebas pareadas y umbrales de fiabilidad entre semillas. |
| `retention` | no | Política preview/apply de roles de checkpoint, archivos verificados y reglas explícitas de poda. |

El [JSON Schema](src/lambdaforge/schemas/experiment.schema.json) empaquetado rechaza claves propias
del framework desconocidas; `metadata` y `extensions` son vías explícitas para información de la
tarea. `lambdaforge validate` comprueba además expansión, contradicciones de recursos y todos los
imports `target`/`ref`/plugin sin construir objetos ni crear directorios de salida.

El Schema canónico exige `schema_version: "1.1"`. El Schema 1.0 sigue empaquetado para validar
exactamente configuraciones históricas. Los archivos que omitían el campo siguen la cadena
determinista `unversioned -> 1.0 -> 1.1`; las configuraciones expandidas y materializadas contienen
la versión actual. `UnversionedToV1Migration` declara 1.0 y `ExperimentV1ToV1_1Migration` añade la
superficie opcional de retención sin activarla.

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

`plugin` resuelve una clase publicada por una distribución instalada. El `kind` explícito evita
strings mágicos dependientes del contexto y permite validar el contrato antes de crear una instancia
nueva:

```yaml
model:
  plugin: {kind: model, name: acme_encoder}
  params: {hidden_features: 128}

val_metrics:
  - plugin: {kind: metric, name: calibrated_auc}
    params: {pred_key: probabilities}

data:
  train:
    plugin: {kind: dataset, name: acme_records}
callbacks:
  - plugin: {kind: callback, name: artifact_marker}
trainer:
  logger:
    plugin: {kind: logger, name: jsonl_logger}
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

### Comparaciones estadísticas entre semillas

Las comparaciones emparejan únicamente semillas comunes. LambdaForge define
`delta = variante - baseline` e invierte su signo para métricas cuyo modo declarado es `min`, de
forma que una `improvement` positiva siempre favorece a la variante. El contrato YAML anidado
completo es:

```yaml
aggregation:
  comparisons:
    alpha: 0.05
    target_power: 0.80
    min_pairs_for_verdict: 3
    confidence_interval:
      method: bootstrap_percentile  # normal o bootstrap_percentile
      confidence_level: 0.95
      resamples: 10000
      seed: 0
      batch_size: 1024
      max_batch_elements: 1000000
    paired_test:
      method: wilcoxon              # sign o wilcoxon
      alternative: two_sided  # two_sided, greater, less, observed_direction
      calculation: auto             # auto, exact, asymptotic
      zero_method: wilcox            # wilcox, pratt, zsplit
      continuity_correction: false
      exact_max_pairs: 50
      zero_tolerance: 1.0e-12
      round_decimals: 12             # null desactiva el redondeo previo a los rangos
```

Omitir `aggregation` conserva el protocolo histórico: intervalo normal al 95 %, prueba pareada
exacta de signos con `observed_direction`, `alpha: 0.05`, potencia objetivo `0.80` y al menos tres
pares antes de emitir un veredicto. El bootstrap percentil usa un flujo determinista derivado de su
semilla base y de la identidad de la comparación; los lotes acotan la matriz transitoria de índices
de remuestreo mientras se conservan `O(resamples)` medias para calcular cuantiles. Wilcoxon
`auto` usa enumeración determinista exacta de rangos hasta `exact_max_pairs` y aproximación normal
por encima. Sus convenciones para ceros, alternativas, modos de cálculo, campos de artefactos y
objetos Python se detallan en la
[guía de comparaciones estadísticas](src/lambdaforge/experiments/statistics/README.es.md).

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

## Migraciones de configuración

Previsualiza la normalización legacy antes de validar o ejecutar:

```powershell
lambdaforge migrate legacy.yaml                 # diff unificado, sin escrituras
lambdaforge migrate legacy.yaml --format yaml   # YAML resultante completo
lambdaforge migrate legacy.yaml --format json   # sobre de resultado estable
lambdaforge migrate legacy.yaml --check         # 1 si hace falta migrar; 0 en caso contrario
```

La persistencia siempre es explícita y siempre apunta a una ruta diferente:

```powershell
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml
lambdaforge validate experiment.v1_1.yaml
```

Un destino existente exige `--force`; incluso entonces no se puede sobrescribir el origen.
`--target-version` acepta un string `MAJOR.MINOR` exacto y por defecto usa el Schema empaquetado
actual. `--format` controla la salida estándar, mientras `--output` siempre escribe YAML completo de
forma atómica. La publicación por defecto no sobrescribe ni siquiera con writers concurrentes;
`--force` cambia únicamente un destino distinto a sustitución atómica.

La cadena determinista actual es `unversioned -> 1.0 -> 1.1`. La migración solo de compatibilidad
`UnversionedToV1Migration` inserta la declaración histórica 1.0.
`ExperimentV1ToV1_1Migration` avanza después ese mapping válido a 1.1; como el nuevo bloque
`retention` es opcional y por defecto está desactivado, el paso no activa mutaciones de artefactos
en experimentos antiguos. Ambos Schemas exactos siguen empaquetados. La previsualización conserva
comentarios, orden, comillas, anchors y saltos de línea cuando el cambio estructural lo permite,
nunca importa targets o plugins del usuario y nunca construye objetos del experimento. La carga
normal aplica la cadena completa en memoria sin editar el origen.

```python
from lambdaforge import LambdaForge
from lambdaforge.experiments import MigrationPreviewFormat

preview = LambdaForge.preview_migration("legacy.yaml")
print(preview.changed, preview.source_version, preview.target_version)
print(preview.render(MigrationPreviewFormat.DIFF))
```

Consulta la [guía de migraciones](src/lambdaforge/experiments/migrations/README.es.md) para el
contrato de códigos de salida, las garantías de escritura atómica, la API completa de objetos, los
modos de fallo y el procedimiento para añadir un futuro paso de Schema.

## Ejecución y seguridad de procesos

| Modo | Planificación |
|---|---|
| `sequential` | Las ejecuciones se realizan en el proceso llamador, una tras otra. Los slots GPU se ignoran. |
| `parallel` | Cada ejecución es un proceso `spawn` en una GPU lógica; `jobs_per_gpu` permite trabajos independientes simultáneos por GPU. |
| `ddp` | Cada ejecución recibe un grupo de `devices_per_job` GPU lógicas y Lightning ejecuta DDP dentro de ese proceso. |

Los números GPU son posiciones lógicas respecto a `CUDA_VISIBLE_DEVICES` del padre. Si el padre
tiene `CUDA_VISIBLE_DEVICES=4,7`, un trabajo que pida `[1]` ve la GPU física `7` como GPU local `0`.
La API directa de objetos diferencia sin ambigüedad herencia y ejecución solo CPU:

| `TrainingJob.devices` / slot del scheduler | Visibilidad del hijo |
|---|---|
| `None` | Hereda sin cambios todo el conjunto CUDA visible para el padre. |
| `[]` o `()` | Establece `CUDA_VISIBLE_DEVICES=""` y oculta CUDA explícitamente. |
| `[0]`, `[1]`, ... | Restringe el hijo a esas posiciones lógicas; sin variable del padre se tratan como IDs físicos. |

Las asignaciones quedan congeladas y rechazan strings, booleanos, índices negativos/fraccionarios y
duplicados. LambdaForge puede preparar un valor restringido al crear el hijo, pero restaura
inmediatamente el entorno del padre y no deja `CUDA_VISIBLE_DEVICES` alterada.

`ExecutionConfig` aplica la misma exigencia antes de planificar: parallel/DDP requieren una lista GPU
no vacía, los grupos DDP deben dividirla exactamente, los conteos deben ser enteros finitos y
`grace_seconds` un real finito no negativo. Los booleanos nunca se aceptan como números; los límites
CPU opcionales usan `null` para heredar, deben ser positivos en otro caso y solo
`dataloader_num_workers_per_job` permite cero.

Los límites de hilos CPU, hilos inter-op, afinidad y workers de datos se aplican por trabajo. Usa
valores conservadores al compartir una GPU: la concurrencia multiplica el uso de CPU y RAM aunque
quepa en memoria GPU.

La limpieza tiene varias capas:

1. `TrainingOrchestrator.request_stop()` o las señales gestionadas SIGINT/SIGTERM/SIGBREAK activan el
   evento compartido de forma idempotente; SIGBREAK solo se instala si la plataforma la expone.
2. Los callbacks de Lightning paran en un límite de lote de entrenamiento/validación/test.
3. El orquestador espera `grace_seconds` mediante deadlines monotónicos compartidos, sin conceder el
   timeout completo una vez por proceso.
4. Los árboles restantes se terminan recursivamente con psutil y se matan si es necesario; cualquier
   superviviente tras la escalada acotada se comunica mediante `RuntimeError`.
5. Cada hijo recibe el `expected_parent_pid` exacto del launcher. Linux añade la entrega parent-death
   mediante `prctl` y los workers POSIX verifican/vigilan además ese PID, cerrando la carrera de
   reparentado.
6. En Windows se usa un Job Object kill-on-close; los workers de `DataLoader` instalan la misma
   guarda descendiente y límites de hilos.

La gestión de señales es explícita. El valor predeterminado `manage_signals=True` solo es válido en
el hilo principal de Python y restaura los handlers anteriores al terminar. Una integración que
ejecute el orquestador en un hilo secundario debe usar `manage_signals=False` y llamar a
`request_stop()` desde su propio hook de ciclo de vida. Desactivar la propiedad de los handlers no
desactiva la cancelación cooperativa.

El aislamiento Windows no se degrada en silencio. Si no se puede inicializar un Job Object nativo,
LambdaForge emite `RuntimeWarning`, guarda el detalle en `process_isolation_warnings` y mantiene la
limpieza recursiva portable. Si un Job Object activo no puede aceptar un worker nuevo, ese worker se
termina y el run falla en lugar de continuar con un aislamiento más débil.

Esto protege los procesos creados por el framework y sus descendientes en el mismo árbol del sistema
operativo. Ninguna biblioteca puede garantizar la limpieza de un programa externo que se desacople
intencionadamente como servicio independiente; esos launchers requieren su propio contrato de vida.
`TrainingOrchestrator` tiene estado y no es reentrante: no solapes llamadas
`run`/`run_scheduled` sobre el mismo objeto y prefiere una instancia por run con propietario
independiente.

En DDP, las métricas sincronizan el estado acumulado antes de calcular valores no lineales como
AUROC, F1 o correlaciones. Una métrica personalizada en DDP debe implementar el contrato de estado
distribuido; LambdaForge genera un error en vez de promediar silenciosamente escalares inválidos por
rank.

## Salidas, reanudación y carga

Cada ejecución concreta tiene su directorio bajo `<output_root>/<experiment>/<variant>/<seed>/` y
puede contener:

- `config.yaml`: configuración completamente materializada;
- `environment.json`: instante UTC, Python/plataforma, versiones principales, CUDA/cuDNN, propiedades
  de GPU visibles, variables CUDA, estado Git y distribución/versión/grupo/valor de cada plugin de
  entry point resuelto correctamente por ese run;
- `hparams.json`: resumen compacto de hiperparámetros;
- `train.log`: stdout/stderr capturado;
- `metrics.csv`: una fila densa por época;
- `checkpoints/`: archivos según `checkpoint_policy`;
- `result.json`: estado terminal, ID de intento, fingerprint de configuración científica, límites
  UTC, rutas, duración y métricas mejores/finales;
- `.lambdaforge/attempts/result-*.json`: historial terminal inmutable retirado antes de reintentar;
- artefactos requeridos definidos por el proyecto.

El área agregada contiene CSV por época y variante, estadísticas entre semillas,
`baseline_comparisons.csv`, `reliability.json`, q-valores Benjamini-Hochberg sobre la prueba pareada
seleccionada y gráficas PNG opcionales. El CSV de comparaciones registra intervalo/prueba elegidos,
estado del cálculo, tamaños efectivos y procedencia de semillas bootstrap. Los campos históricos
`ci95_improvement_*` y `p_value_sign_*` conviven con los campos neutrales respecto al método. Puede
regenerarse desde disco sin reconstruir el modelo.

La finalización usa identidad, no sólo ruta. LambdaForge calcula el fingerprint de la configuración
científica expandida excluyendo controles de almacenamiento, reintento, ejecución, agregación y
retención. Por ello una configuración distinta de modelo/datos/loss/trainer nunca se omite como un
éxito antiguo ni reanuda su checkpoint incompatible. Cuando es posible, los resultados legacy se
comparan con su `config.yaml` materializado y reciben la identidad completa al archivarse.

Audita intentos actuales e históricos antes de elegir valores para un informe o artículo:

```bash
lambdaforge results experiment.yaml
lambdaforge results experiment.yaml --duplicates
lambdaforge results experiment.yaml --json --write-index
lambdaforge results experiment.yaml --fail-on-ambiguous
```

`--fail-on-ambiguous` devuelve 2 si un fingerprint tiene varios intentos correctos, de modo que CI
puede impedir la ambigüedad. `--write-index` publica atómicamente
`.lambdaforge/result-index.json`; es un índice, no una segunda fuente de verdad. Python ofrece el
mismo escaneo fresco del sistema de archivos:

```python
records = experiment.results()                 # incluye intentos archivados
duplicates = experiment.result_catalog().duplicate_groups()
chosen = experiment.result_catalog().select(attempt_id="20260722T...")
```

Nunca elijas un directorio “latest” arbitrario para publicar. Registra `attempt_id`,
`config_fingerprint`, semilla, variante, rol de checkpoint y artefacto agregado en el manifiesto de
evidencia del artículo. Varios éxitos se conservan y aparecen como ambiguos hasta que el investigador
hace esa selección; LambdaForge no elige silenciosamente el resultado más favorable.

Carga un modelo desde la suite:

```python
experiment = LambdaForge.experiment("experiment.yaml")
model = experiment.load_model(seed=17, variant="base", which="best")
```

`which` acepta `best`, `last` o `auto`. `auto` resuelve best, después last y finalmente el checkpoint
local seguro más reciente; las peticiones exactas `best` y `last` nunca cambian silenciosamente de
rol. La carga valida el checkpoint y entiende estados de modelo directos y claves Lightning con
prefijo `model.`. La clase del modelo debe seguir siendo importable desde la configuración
materializada.

## Retención de artefactos

El Schema 1.1 añade un bloque `retention` opcional y estricto. El punto de partida seguro es el modo
preview:

```yaml
retention:
  mode: preview                 # disabled, preview, apply
  checkpoints:
    keep: last_and_best         # all, best, last, last_and_best
    prune_unselected: true
  protect: [reports/**, predictions/final.json]
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

Omitir el bloque equivale a `disabled` y conserva todos los artefactos. `preview` imprime/planifica,
pero nunca escribe, archiva ni elimina; `apply` permite la retención automática solo después de una
agregación final correcta. La misma frontera es explícita mediante API y CLI:

```python
plan = experiment.preview_retention()
result = experiment.apply_retention()
```

```powershell
lambdaforge retain experiment.yaml
lambdaforge retain experiment.yaml --json
lambdaforge retain experiment.yaml --apply
```

Aplicar exige un `aggregate/aggregation_receipt.json` vigente que demuestre que cada variante/semilla
esperada terminó correctamente y que siguen coincidiendo los fingerprints comprometidos de runs y
agregados. Las reglas genéricas no pueden seleccionar archivos base del run, artefactos requeridos,
globs protegidos, agregados, enlaces/reparse points ni metadata interna. Los checkpoints usan su
propia política no ambigua de roles best/last. La compresión transmite a ZIPs inmutables por run,
verifica nombres, CRC, tamaños y SHA-256 antes de la cuarentena y puede conservar fuentes no
compresibles. La poda y la compresión usan un journal durable y una cuarentena reversible.

Entrenamiento, agregación final y retención se coordinan mediante locks cross-process de actividad,
agregación y retención en un orden fijo. Un crash anterior al commit hace rollback; uno posterior al
marcador de commit termina hacia delante, y reaplicar un plan comprometido es idempotente. Consulta
la [guía de retención de artefactos](src/lambdaforge/experiments/retention/README.es.md) para el
contrato YAML completo, el receipt de elegibilidad, los estados de transacción, artefactos y
limitaciones.

## Componentes incluidos

- Modelos: `MLP`/`CNN2D` configurables; `ECMP`; stacks GCN, GraphSAGE, GAT, GATv2, GCN relacional,
  PNA, GraphTransformer disperso, EGNN y GIN más `GraphReadout`; `GradTree`, `GRANDE`, árboles
  oblivious y `NODE`; RNN/LSTM/GRU, encoder
  Transformer encoder/decoder/seq2seq, Conformer, adaptador de espacio de estados y convolución
  temporal; `DeepSets`/`SetTransformer`; MLP residual, `FTTransformer`, TabNet, SAINT, AutoInt y
  DeepFM; `ResNet2D`/`ConvNeXt2D`/`MobileNetV2`, `VisionTransformer2D`, `UNet2D` y
  `FeaturePyramidNetwork2D`; autoencoders, ensembles, mixture-of-experts, multitarea y siamese; y
  VQ-VAE y difusión gaussiana; Neural ODE/CDE, DeepONet, operador Fourier, campos tensoriales
  escalares/vectoriales y adaptadores equivariantes de orden superior opcionales; y representaciones
  implícitas `SIREN`.
- Activaciones: CELU, ELU, GELU, hard sigmoid/swish, Identity, LeakyReLU, Mish, PReLU, ReLU/ReLU6,
  SELU, Sigmoid, SiLU, Softplus, Softsign, SquarePlus, Tanh, Entmax15/Entmoid15, Sine/Snake y la
  familia GLU/GEGLU/SwiGLU/ReGLU que cambia la dimensión.
- Normalizaciones: BatchNorm e InstanceNorm (1D/2D/3D), ChannelLayerNorm, GroupNorm, IdentityNorm,
  L2Norm, LayerNorm, RMSNorm y ScaleNorm.
- Pooling: reducciones básicas/suaves, estadísticas/concatenación, atención aprendida, top-k y
  operadores probabilísticos para conjuntos densos con máscara, más suma/media/máximo/atención
  sparse indexados.
- Componentes por pares: distancias euclídea, euclídea al cuadrado, Manhattan, Minkowski,
  Chebyshev, coseno, angular y Mahalanobis; similitudes dot, coseno y bilineal; kernels RBF,
  laplaciano y polinómico.
- Pérdidas: entropía cruzada y focal binaria/multiclase, MSE, MAE, Smooth L1, Huber, Dice/Tversky,
  contrastiva, triplet-margin, InfoNCE y un objetivo beta-VAE reutilizable. Toda pérdida de entreno
  incluida reduce a escalar y expone claves de mapas, peso y nombre estable.
- Codificaciones y regularización: sinusoidal, aprendida, rotatoria y Fourier; DropPath, dropout de
  features y ruido gaussiano.
- Datos e incertidumbre: `CategoricalFeatureEncoder`, `FileDataset`, `NumpyMemmapDataset`,
  `DatasetCache`, serialización NumPy/Torch segura con
  fingerprint, envelopes checksum/HMAC y backends disco/mmap coordinados. Pickle queda como
  compatibilidad explícita para almacenamiento local confiable. `TemperatureScaler` y
  `ConformalPredictionInterval` proporcionan componentes post-hoc sobre datos reservados.
- Conformidad: `ArchitectureConformanceCase` y `ArchitectureConformancePack` capturan procedencia,
  estado de inicialización, número de parámetros, tensor de salida y tolerancias en referencias
  pequeñas y weights-only, y fallan ante deriva numérica o de forma.
- Métricas binarias: accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, kappa de
  Cohen, AUROC/AUPRC exactas y `StreamingBinaryAUROC`/`StreamingBinaryAUPRC` de memoria fija.
- Métricas multiclase: accuracy, balanced accuracy, F1, AUROC/AUPRC exactas y
  `StreamingMulticlassAUROC`/`StreamingMulticlassAUPRC` de memoria fija con reducción macro,
  ponderada o micro.
- Regresión: MAE, MSE, RMSE, R², correlaciones Pearson y Spearman, además de `MeanMetric`.

Los nombres cortos de activaciones que conservan la forma y de normalizaciones no distinguen
mayúsculas en YAML. Se pueden añadir alias mediante `ComponentRegistry`; los constructores Python
también aceptan clases compatibles. La familia GLU sigue como objeto explícito porque divide una
dimensión y no puede sustituir con seguridad una activación ordinaria por capa. Consulta la
[guía de componentes neuronales](src/lambdaforge/nn/README.es.md) y la
[guía de métricas](src/lambdaforge/metrics/README.es.md).

### Modelos de grafo avanzados y equivariantes

La ruta nativa de grafos usa tensores PyTorch ordinarios y aristas dirigidas dispersas; no necesita
PyG/DGL ni adyacencia densa. En cada `edge_index=(2,E)`, la fila cero es `source` y la fila uno
`destination`.

| Entrada | Contrato |
|---|---|
| `x` | Características flotantes de nodos `(N,in_channels)`. |
| `edge_index` | Tensor entero `(2,E)` con índices en `[0,N)`; se rechazan índices flotantes y Booleanos. |
| `edge_features` | Tensor real opcional `(E,edge_channels)`, obligatorio cuando `edge_channels > 0` y normalizado al dispositivo/dtype de `x`. |
| `edge_types` | Identificadores enteros de relación de GCN relacional, `(E,)` y en `[0,num_relations)`. |
| `coordinates` | Coordenadas flotantes de EGNN `(N,D)`, `D >= 1`, exactamente con el mismo dispositivo/dtype que `x`. |

| Familia | Resultado, controles y fuente primaria |
|---|---|
| `GATv2` | Atención dinámica multi-head `(N,out_channels)` con `hidden_channels` y `heads`, `concatenate_heads`, `share_weights`, dropout de features/atención, pendiente negativa, política/relleno de self-loops, residuo y bias por capa; `edge_channels` compartido; activación/normalización y kwargs solo ocultos. `GATv2Layer.forward_with_attention` devuelve las aristas enrutadas alineadas y pesos `(E_routed,heads)` de una capa. Inspirado en [Brody et al.](https://arxiv.org/abs/2105.14491). |
| `RelationalGCN` | `(N,out_channels)` desde transformaciones tipadas para `num_relations`; `num_bases`, `aggregation` (`sum`/`mean`), `message_chunk_size`, dropout, residuo, peso de raíz y bias son escalares/por capa; activación/normalización y kwargs solo ocultos. Los mensajes se agrupan por relación y proyectan en bloques acotados sin materializar matrices por arista. Inspirado en [Schlichtkrull et al.](https://arxiv.org/abs/1703.06103). |
| `PNA` | `(N,out_channels)` mediante `aggregators` mean/min/max/std no vacíos/sin duplicados cruzados con `scalers` identity/amplification/attenuation/linear/inverse-linear. Configura anchuras de arista/mensaje y MLP anterior/posterior, estadísticas de grado, épsilon, dropout, activación/kwargs y bias; normalización/residuo solo ocultos. `layer_kwargs` puede sobrescribir por capa todas las opciones pertenecientes a la capa, mientras se reservan las anchuras de entrada/salida/arista de la pila. Inspirado en [Corso et al.](https://arxiv.org/abs/2004.05718). |
| `GraphTransformer` | `(N,out_channels)` mediante atención dot-product dispersa local. Los controles por capa son cabezas/concatenación, anchura feed-forward, activación/normalización y kwargs, tres dropouts, relleno de self-loop, pre/post norm, residuo, gate beta y bias; las características de arista modifican claves y valores. Relacionado con [Shi et al.](https://arxiv.org/abs/2009.03509). |
| `EGNN` | Características `(N,out_channels)` o un mapping de características y coordenadas actualizadas `(N,D)`. `message_channels`, feature dropout, residuo, bias y `layer_kwargs` por capa configuran anchuras de MLP de mensaje/nodo/coordenadas, agregación, dropout, normalización/escala de desplazamientos, actualizaciones/tanh de coordenadas y atención de mensajes opcional. Activación/normalización ocultas son políticas de pila; las claves de salida se configuran. Inspirado en [Satorras et al.](https://arxiv.org/abs/2102.09844). |

Una pila de grafo tiene `L_graph = len(hidden_channels) + 1` capas. Una opción escalar por capa se
difunde; una lista debe tener exactamente `L_graph` entradas. Las listas solo ocultas tienen
`len(hidden_channels)` entradas. Las anchuras concatenadas de GATv2 deben ser divisibles por sus
cabezas; las bases de R-GCN no pueden superar `num_relations`; `beta=true` de GraphTransformer
requiere ruta residual. Los self-loops de atención reemplazan loops existentes y sintetizan filas de
arista alineadas en vez de duplicar topología. Las listas vacías de aristas y los nodos aislados
permanecen finitos.

El `message_chunk_size: 65536` predeterminado de R-GCN acota el tensor de mensajes proyectados y
conserva exactamente los resultados dispersos `sum`/`mean`. Usa un entero positivo menor en
dispositivos limitados; `None` elimina ese límite y solo conviene si se conocen el número de aristas
y el presupuesto de memoria.

`average_degree = mean(in_degree)` y
`average_log_degree = mean(log(in_degree + 1))` de PNA deben calcularse **solo con el
split/topología de entrenamiento**, registrarse en YAML y reutilizarse sin cambios en validación,
test e inferencia. Calcularlos con grafos reservados filtra topología. Ambas estadísticas y `epsilon`
deben ser positivas y finitas.

`EGNN.forward` devuelve solo características de nodos por defecto. Con `output_mode: mapping`
devuelve `feature_output_key` y `coordinate_output_key` configurados; `LightningTask` conserva ese
mapping para que cada pérdida elija su `output_key` y cada métrica su clave de predicción/salida
documentada (`pred_key` en la mayoría de las incluidas). `forward_with_coordinates` devuelve
siempre el par. Un [ejemplo YAML completo de PNA](src/lambdaforge/nn/README.es.md#pna-con-estadísticas-de-grado-solo-de-entrenamiento)
y otro de [mapping EGNN](src/lambdaforge/nn/README.es.md#salida-mapping-de-egnn) documentan
`model_input_keys` con nombre y las longitudes relevantes de listas por capa.

Estas implementaciones son núcleos nativos con pocas dependencias, no reproducciones de los
pipelines de entrenamiento de sus autores. No afirman paridad de checkpoints ni benchmarks. El
enrutamiento disperso de atención/mensajes crece aproximadamente como `O(EH)` en lugar de asignar
una matriz global `N²`; GraphTransformer atiende por ello solo sobre las aristas suministradas y no
tiene codificación global o posicional implícita. EGNN es E(n)-equivariante para características
escalares de nodos/aristas y actualizaciones de coordenadas, no equivariante a escala ni una
representación vectorial/tensorial de orden superior.

## Contratos de extensión

### Modelo

Hereda de `torch.nn.Module` o `lambdaforge.nn.models.Model` e implementa
`forward(*args, **kwargs)`. `predict` activa evaluación/inferencia y restaura el modo anterior. Para
la tarea por defecto, devuelve un tensor —se envuelve bajo `model_output_key`— o un mapa.

### Pérdida

Hereda de `Loss` e implementa `forward(outputs, batch, context=None) -> Tensor`. `LightningTask` se
pasa a sí misma como `context`, mientras que el valor predeterminado permite invocar la pérdida de
forma independiente. Da a cada pérdida un `name` estable y usa claves de mapa en vez de supuestos de
dominio. Varias pérdidas se suman tras aplicar sus pesos.

### Métrica

Implementa `update`, `compute` y `reset`. Para DDP, expón además `distributed_state` y
`merge_distributed_state`, o usa una métrica del framework. Las instancias se copian profundamente
por etapa para que no se mezcle estado entre entreno, validación y prueba.

Cada etapa exige nombres de métrica únicos. Envuelve una métrica con `MetricAlias` si usas la misma
clase varias veces con parámetros distintos. Las listas explícitas por etapa evitan necesitar
`deepcopy` cuando una métrica propia mantiene un recurso externo no copiable.

### Datos y tarea

`LightningTask` espera lotes con forma de mapa. `model_input_key` elige un tensor y
`model_input_keys` dirige una secuencia de entradas posicionales o un mapa de argumentos del modelo a
claves del lote. Cada pérdida/métrica elige sus claves. Los lotes tupla, varios optimizadores, la
optimización manual o flujos especiales aún requieren un `task.target` propio; el resto de
experimentos, procesos y artefactos sigue siendo reutilizable.

`DatasetCache` es un wrapper map-style opcional, no una caché global implícita. Su presupuesto
`max_memory_bytes_per_process` cuenta bytes de payload serializado retenidos y se combina con
`max_memory_entries`; no limita el RSS total, los lotes vivos, el prefetch ni el dataset envuelto. La
caché RAM en workers está apagada por defecto porque cada worker de DataLoader posee una réplica.
Cachea solo carga/preprocesado determinista, nunca resultados de augmentations aleatorias. Consulta
la [guía de datos](src/lambdaforge/data/README.es.md) antes de activarla en paralelo o DDP.

### Plugins instalados

Las distribuciones externas pueden publicar modelos, métricas, componentes neuronales, datasets,
callbacks Lightning y loggers Lightning en los grupos canónicos de la
[guía de plugins](src/lambdaforge/plugins/README.es.md). Los datasets heredan de `Dataset` y
callbacks/loggers de las bases públicas de `lambdaforge.integrations.Lightning`. El discovery solo
lee metadata; resolver importa código del proveedor y comparte la frontera de confianza de
`target`. Los componentes incluidos y alias registrados en el proceso conservan precedencia sobre
plugins de activación/normalización.

Cada run real usa una `PluginUsageSession` aislada y guarda atómicamente sus descriptores ordenados
en `environment.json`. No aparecen validaciones anteriores, runs secuenciales, proveedores
instalados pero no usados ni resoluciones fallidas; sí cuentan aciertos de caché y aliases externos
usados realmente. Los dry-runs escriben una lista vacía. Consulta el
[contrato de procedencia](src/lambdaforge/plugins/README.es.md#procedencia-de-plugins-cargados).

### Loggers de tracking

`trainer.logger` admite los targets públicos `MLflowTrackingLogger`,
`TensorBoardTrackingLogger` y `WeightsAndBiasesTrackingLogger`, o una lista no vacía que los mezcle
con loggers del proyecto y plugins de logger instalados. Cada adaptador comprueba su propio extra
opcional solo al construirse; importar LambdaForge continúa libre de proveedores. El
`metrics.csv` denso canónico se controla por separado mediante `write_epoch_metrics_csv`. Las
pérdidas/métricas de la tarea solo llegan al proveedor al habilitar
`task.params.logging.logger`.

Las credenciales del proveedor deben quedar fuera del YAML porque las configuraciones
materializadas son artefactos duraderos del run. Subir checkpoints se habilita de forma explícita
mediante `log_model` y es independiente de las transacciones de retención local de LambdaForge.
Consulta en la [guía de tracking](src/lambdaforge/tracking/README.es.md) los parámetros completos,
ejemplos local/remoto y offline/online, fronteras de privacidad y comportamiento ante fallos.

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
| Modelos, distancias, experimentos, métricas y entreno competían como entradas raíz. | Se crearon la API `LambdaForge`/`Experiment` y subpaquetes públicos cohesivos con reexports estables. |
| Módulos grandes contenían muchas funciones sueltas o varias clases sin relación. | El comportamiento pasó a objetos colaboradores y cada clase de implementación ocupa un archivo. |
| `training` y `metrics.classification` se habían vuelto visualmente densos. | Se dividieron solo por contratos estables (`callbacks`, `orchestration`, `binary`, `multiclass`) conservando los imports públicos; familias cohesivas como `pooling` siguen planas. |
| Opciones cerradas y componentes repetían strings literales. | Se añadieron enums y `ComponentRegistry`; los strings quedan en fronteras YAML/serialización. |
| Opciones avanzadas de Trainer/DataLoader exigían editar el código. | Se añadieron `trainer_kwargs` y `dataloader_kwargs` comunes/específicos con validación. |
| Una sola lista de métricas y el orden implícito del monitor limitaban el control. | Se añadieron listas por etapa, alias, modos de monitor explícitos, política de publicación de pérdidas y filtros de CSV/terminal. |
| Elegir un logger Lightning propio eliminaba el CSV canónico de época. | Se separó el logger externo de `write_epoch_metrics_csv`, de modo que los informes siguen disponibles por defecto. |
| Los errores YAML aparecían tarde durante la construcción de objetos. | Se publicó un Schema Draft 2020-12 estricto y se añadieron `ExperimentValidator`, `ValidationReport` y `lambdaforge validate`. |
| Las ejecuciones no guardaban suficiente procedencia de software/hardware/plugins. | Se añadieron artefactos `EnvironmentManifest` tipados y atómicos y sesiones aisladas de plugins resueltos para runs reales; los dry-runs no importan objetos. |
| DDP podía promediar escalares AUROC/F1/correlación ya calculados, algo matemáticamente incorrecto. | Las métricas reúnen y fusionan estado antes de calcular; una métrica no compatible falla explícitamente. |
| La reutilización de datasets no tenía un contrato de memoria acotado y podía multiplicarse entre workers. | Se añadió una LRU serializada por proceso con límites de entradas/bytes, aislamiento spawn/fork y política explícita de workers. |
| Los writers persistentes podían competir, exceder cuota tras un crash, reutilizar transformaciones obsoletas o deserializar bytes sin verificar. | Se añadieron locks OS compartidos/exclusivos, manifiestos inmutables, pre-expulsión, reconciliación tras crash, tokens de generación, fingerprints explícitos, envelopes checksum/HMAC verificados y un codec NumPy/Torch acotado sin pickle. |
| Las clases externas exigían rutas completas aunque se distribuyesen como paquetes reutilizables. | Se añadieron discovery lazy, contratos neuronales y de dataset/callback/logger, integración YAML, conflictos y listado CLI sin imports. |
| Las métricas de curva exactas retenían todos los scores y targets. | Se añadieron alternativas binarias y multiclase AUROC/AP por histogramas con estado fijo y sincronización tensorial `all_reduce` acotada. |
| El catálogo neuronal se limitaba a una red densa, una CNN y un modelo de paso de mensajes. | Se añadieron familias agnósticas de grafos, árboles diferenciables, secuencias, conjuntos, datos tabulares, visión, composición y representaciones implícitas, además de nuevas categorías. |
| La tarea predeterminada dirigía un único tensor y no exponía grupos de optimizador. | Se añadieron `model_input_keys` posicionales/con nombre y opciones por grupo declaradas por el modelo. |
| `CNN2D` elegía BatchNorm 1D por defecto para tensores NCHW. | El default incluido crea ahora `BatchNorm2d`. |
| `Model.predict` no garantizaba restaurar el modo de entrenamiento previo. | La inferencia usa `try/finally` y recupera el estado original. |
| `test_after_fit` pedía a un Trainer nuevo un checkpoint `best` desconocido y omitía la parada. | Usa el checkpoint real si existe, o los pesos actuales, y conserva la cancelación. |
| La terminación forzada se centraba en workers raíz y era frágil en Windows. | Se añadieron limpieza recursiva, Job Objects, muerte del padre y workers de datos protegidos. |
| La finalización presuponía un artefacto de predicción específico de dominio. | Se sustituyó por rutas relativas genéricas `required_artifacts`. |
| La documentación describía módulos/scripts obsoletos y no había guía raíz. | Se reemplazó por guías enlazadas en inglés/español verificadas frente al código. |

Estas facilidades siguen siendo explícitas: el investigador elige en YAML el presupuesto, proveedor
de plugin y semántica exacta o streaming, sin ocultar costes de recursos ni compromisos científicos.

## Desarrollo y verificación

```powershell
ruff format --check src tests
ruff check src tests
mypy src\lambdaforge
pytest -q
```

La suite actual cubre expansión, construcción de objetos/plugins, validación de modelos, métricas,
agregación, planificación con procesos `spawn`, reglas estructurales POO, un entrenamiento Lightning
real de una época en CPU, validación Schema/CLI, captura del entorno y construcción YAML de modelo,
pérdida, métrica, logger y callback externos, entry points dataset/callback/logger instalados,
cuotas/aislamiento de caché, corrupción/sustitución HMAC, carreras spawn, recuperación tras crash,
leases mmap, datasets lazy/mapeados, estado de métricas streaming y escenarios de preview/apply,
receipt, locks, verificación ZIP, rollback e idempotencia concurrente de retención. Las pruebas de
plugins cubren metadata exacta, aciertos de caché,
aislamiento validación/run, fallos de construcción y un manifiesto real en hijo `spawn`.
Las pruebas de integración crean un árbol real launcher/worker/descendiente. POSIX entrega un
`killpg(SIGINT)` real al grupo de procesos; Windows pide al launcher que provoque un SIGBREAK Python
dirigido porque un evento nativo de control afectaría a todo el grupo de pruebas. Otro escenario
termina el launcher abruptamente y verifica que no quede ningún descendiente registrado ni archivo
temporal. La limpieza de emergencia de cada prueba evita además que un fallo de aserción deje
workers residuales.

### Integración continua

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) se ejecuta automáticamente en pull requests,
pushes a `main`, tags de versión y lanzamientos manuales. Solo concede lectura del contenido,
cancela ejecuciones obsoletas
para la misma ref, limita la duración de cada job, cachea descargas pip a partir de `pyproject.toml`
y conserva informes durante 14 días. Sus jobs independientes son:

- formato, lint y tipos en CPython 3.10;
- la suite CPU completa en Ubuntu y Windows para todas las versiones estables soportadas actualmente,
  desde CPython 3.10 hasta 3.14, usando el índice CPU-only de PyTorch y ocultando CUDA explícitamente;
- la suite completa instalando únicamente `pytorch-lightning==2.2.*`, más una aserción de que el
  adaptador central eligió realmente el namespace antiguo y no `lightning` moderno;
- construcción de sdist/wheel, validación de metadata con `twine` y smoke test aislado del contenido
  del wheel en CPython 3.14. Los informes JUnit y distribuciones se publican con nombres únicos.

[`.github/workflows/cuda.yml`](.github/workflows/cuda.yml) es deliberadamente manual y apunta a un
runner self-hosted con labels
`[self-hosted, linux, x64, cuda]`. El runner necesita Actions Runner 2.327.1 o posterior para las
[versiones Node 24 de las actions](https://github.com/actions/setup-python/releases/tag/v6.0.0)
empleadas, un driver NVIDIA compatible y un build de PyTorch con CUDA. Un preflight rechaza
instalaciones CPU-only o inutilizables y registra las versiones exactas de PyTorch, CUDA y cuDNN
junto con el dispositivo. Después se ejecuta la prueba marcada `cuda`, que entrena una época
mediante la API YAML pública, y el resto de la suite. Definir, encolar u omitir el workflow no
demuestra cobertura CUDA; solo una ejecución correcta sobre el runner real sirve como evidencia.

Los cambios de planificación GPU o limpieza deben ejercitarse además en el host multi-GPU objetivo
e interrumpirse manualmente al menos una vez.

Todos los módulos y clases fuente tienen docstring. La auditoría comprueba también que nombre de
clase y módulo coincidan, una clase por archivo y la ausencia de funciones auxiliares de módulo en
implementaciones.

## Limitaciones actuales

- `DatasetCache` limita payloads serializados retenidos por proceso, no el RSS total. Lotes,
  prefetch, memoria pinned, overhead del allocator y dataset origen quedan fuera; activar cachés en
  workers multiplica el presupuesto entre réplicas de proceso.
- Pickle sigue siendo el default de compatibilidad y puede ejecutar código: selecciona el codec
  NumPy/Torch seguro cuando sea compatible o limita pickle a almacenamiento local confiable. Un
  checksum no autentica; HMAC debe configurarse y no cifra.
- Los fingerprints son snapshots explícitos porque no se puede inferir la semántica de cualquier
  transformación. La coordinación cubre procesos locales cooperativos, no una caché NFS/remota.
- Lightning es el único backend de entrenamiento incluido.
- La tarea predeterminada presupone lotes con forma de mapa y dirige una o varias entradas; los lotes
  tupla y flujos manuales/con varios optimizadores necesitan una tarea propia.
- Las métricas de curva exactas binarias y multiclase siguen reteniendo predicciones. Sus alternativas
  streaming introducen aproximación por bins; el estado multiclase crece como
  `O(num_classes * num_bins)`.
- El discovery cubre contratos neuronales reutilizables además de datasets, callbacks y loggers.
  Tareas, data modules y runners siguen soportados mediante `target` y deliberadamente no tienen
  grupo dedicado.
- La procedencia de plugins cubre el proceso/contexto del run; procesos hijo creados por el usuario
  necesitan IPC explícito si sus cargas independientes se deben atribuir al padre.
- Los resúmenes estadísticos son exploratorios, no sustituyen el protocolo de cada estudio. Los
  intervalos normales y Wilcoxon asintótico son aproximaciones cuando se seleccionan explícitamente
  o `auto` los elige para muestras pareadas mayores.
- Los Schemas 1.0 y 1.1 están empaquetados y la migración admite la ruta determinista
  `unversioned -> 1.0 -> 1.1`. No hay downgrade, reescritura in-place, origen remoto ni ocultación
  de secretos.
- La retención trabaja solo sobre sistemas de archivos locales y actualmente usa ZIP/Deflate.
  Preview puede quedar obsoleto deliberadamente; apply vuelve a planificar y validar bajo locks.
  Los almacenes remotos/de objetos necesitan contratos propios de leases y atomicidad.
- Los adaptadores de tracking para MLflow, TensorBoard y Weights & Biases son opcionales.
  Autenticación/red/almacenamiento del proveedor, retención remota y disponibilidad del servicio
  siguen siendo externas; un fallo del tracker hace fallar su run y la retención de LambdaForge no
  puede eliminar artefactos ya subidos. No se incluye planificador de clúster, optimizador de
  hiperparámetros ni almacén remoto de artefactos neutral respecto al proveedor.
- Las familias avanzadas de grafos son núcleos dispersos nativos sin paridad con checkpoints de
  papers ni benchmarks. GraphTransformer es local a `edge_index`; las estadísticas PNA son entradas
  explícitas del split de entrenamiento; EGNN cubre equivarianza E(n) de características escalares,
  no equivarianza a escala ni características tensoriales de orden superior.
- La CI alojada cubre CPU en Ubuntu/Windows y CPython 3.10-3.14, incluidos SIGINT real al grupo POSIX,
  SIGBREAK Python dirigido en Windows y muerte abrupta del launcher. No ejercita la entrega nativa
  CTRL_C/CTRL_BREAK al grupo de consola de Windows. Una instancia compartida de
  `TrainingOrchestrator` no es reentrante y un daemon externo desacoplado queda fuera de su contrato.
  CUDA real y multi-GPU/DDP siguen dependiendo del host; CUDA solo queda cubierta tras completar
  correctamente el workflow manual self-hosted.

## Mapa de documentación

- [Manual de agentes en un único fichero](AGENTS.md)
- [Sistema de experimentos](src/lambdaforge/experiments/README.es.md) · [English](src/lambdaforge/experiments/README.md)
- [Migraciones de configuración](src/lambdaforge/experiments/migrations/README.es.md) · [English](src/lambdaforge/experiments/migrations/README.md)
- [Retención de artefactos](src/lambdaforge/experiments/retention/README.es.md) · [English](src/lambdaforge/experiments/retention/README.md)
- [Comparaciones estadísticas](src/lambdaforge/experiments/statistics/README.es.md) · [English](src/lambdaforge/experiments/statistics/README.md)
- [Datos y caché](src/lambdaforge/data/README.es.md) · [English](src/lambdaforge/data/README.md)
- [Entrenamiento y procesos](src/lambdaforge/training/README.es.md) · [English](src/lambdaforge/training/README.md)
- [Componentes neuronales](src/lambdaforge/nn/README.es.md) · [English](src/lambdaforge/nn/README.md)
- [Métricas](src/lambdaforge/metrics/README.es.md) · [English](src/lambdaforge/metrics/README.md)
- [Plugins instalados](src/lambdaforge/plugins/README.es.md) · [English](src/lambdaforge/plugins/README.md)
- [Tracking opcional de experimentos](src/lambdaforge/tracking/README.es.md) · [English](src/lambdaforge/tracking/README.md)
- [Ejemplo YAML completo](examples/experiment.yaml)

Cada guía enlaza de vuelta aquí y a su traducción. Los docstrings de clase son la referencia más
precisa para los argumentos de cada constructor.

## Hoja de ruta

Completado en esta iteración: validación JSON Schema, resultados/manifiestos tipados, matriz CI
ampliada, `DatasetCache` endurecida con adaptadores de archivo/mmap, discovery lazy, procedencia
aislada por run, contratos no neuronales dataset/callback/logger, métricas de curva
binarias/multiclase streaming, intervalos bootstrap deterministas, pruebas Wilcoxon pareadas y el
catálogo neuronal categorizado descrito arriba. Las migraciones versionadas de configuración añaden
ahora normalización legacy segura, previsualización y persistencia atómica explícita. La retención
de artefactos añade receipts de finalización, selección de checkpoints por rol, archivos streaming
verificados y transacciones recuperables. El tracking opcional añade objetos logger MLflow,
TensorBoard y Weights & Biases cargados de forma lazy sin ampliar las dependencias base. El soporte
nativo de grafos avanzados añade GATv2, GCN relacional, PNA, GraphTransformer disperso local y EGNN
de características escalares con contratos de aristas alineados y configuración YAML por capa. Los
hitos revisados y su estado actual son:

1. **Objetos de resultado tipados — completado**: resultados terminales y agregados conservan
   compatibilidad directa dict/JSON, añaden atributos tipados, versionan su sobre y escriben de forma
   atómica.
2. **Curvas multiclase streaming — completado**: AUROC/AP one-vs-rest exige `num_classes`, ofrece
   resultados macro/ponderados/micro y por clase, trata explícitamente las clases indefinidas y
   mantiene estado `O(num_classes * num_bins)`.
3. **Endurecimiento de caché persistente — completado**: fingerprints canónicos de
   contenido/transformación/configuración, envelopes checksum/HMAC verificados, codec NumPy/Torch
   acotado sin pickle, manifiestos inmutables, uso atómico, eliminación segura por generación y
   cuotas multiproceso recuperables tras crash mediante objetos Python/YAML.
4. **Procedencia de plugins cargados — completado**: cada run registra el descriptor determinista de
   todo plugin resuelto correctamente, incluidos cache hits y aliases, sin contaminación de
   discovery, validación, runs previos o procesos padre; el manifiesto se sustituye atómicamente
   tanto en éxito como en fallo.
5. **Contratos de plugin no neuronales — completado**: los grupos dataset, callback y logger validan
   herencia PyTorch/Lightning, funcionan en sus posiciones exactas del Schema, conservan
   `target`/`ref` y listas de logger, y participan en la procedencia del run.
6. **Métodos de comparación más sólidos — completado**: YAML selecciona bootstrap percentil
   determinista y de memoria acotada o el intervalo normal legado, además de Wilcoxon pareado
   exacto/asintótico o la prueba de signos legada. Los resultados tipados exponen alternativas,
   tratamiento de ceros, procedencia del cálculo, tamaños efectivos y estados no disponibles sin
   eliminar las columnas agregadas históricas.
7. **Migraciones de configuración — completado**: objetos valor de Schema exactos, catálogo
   empaquetado 1.0/1.1, registro inmutable hacia delante y cadena validada
   `unversioned -> 1.0 -> 1.1` respaldan
   previsualizaciones CLI diff/YAML/JSON, `--check` para CI, compatibilidad transparente en memoria
   y salida atómica explícita incapaz de sobrescribir el origen o pisar por carrera un destino sin
   `--force`.
8. **Política de retención de artefactos — completado**: el Schema 1.1 estricto selecciona
   disabled/preview/apply, roles best/last, globs protegidos y compresión streaming verificada o
   poda explícita. Un receipt fingerprinted de agregación final habilita transacciones durables con
   journal/cuarentena; locks cross-process ordenados, rollback/recuperación hacia delante y
   manifiestos inmutables hacen que la aplicación concurrente sea segura e idempotente.
9. **CI y pruebas de interrupción ampliadas — completado**: la suite CPU cubre Ubuntu/Windows y
   CPython 3.10-3.14; las pruebas estrictas de tiempos/dispositivos/recursos, `request_stop()` desde
   el host, el opt-out de señales en hilo secundario, las guardas de padre exacto, el apagado acotado
   y la degradación visible de Job Objects endurecen el contrato de procesos. Un job aislado
   demuestra el fallback a `pytorch_lightning` 2.2, mientras SIGINT real al grupo POSIX, SIGBREAK
   dirigido en Windows y muerte abrupta del launcher comprueban que no queden hijos. Un workflow
   CUDA manual self-hosted ejecuta una época GPU desde YAML público, registra evidencia del hardware
   y después lanza las regresiones restantes sin convertir CUDA en requisito de la CI alojada.
10. **Adaptadores opcionales de tracking — completado**: objetos logger públicos y lazy envuelven
   MLflow, TensorBoard y Weights & Biases tras la frontera `trainer.logger` individual/en lista.
   Extras separados y combinado mantienen la instalación base libre de proveedores; parámetros
   local/remoto y offline/online explícitos, publicación de checkpoints opt-in, errores de
   dependencia accionables y guía bilingüe de privacidad/ciclo de vida evitan servicios
   obligatorios.
11. **Capas de grafo avanzadas — completado**: GATv2, GCN relacional, PNA, GraphTransformer local
   consciente de aristas y EGNN E(n)-equivariante son pilas nativas con pocas dependencias. Los
   contratos de aristas dirigidas, características/relaciones/coordenadas, configuración por capa,
   inspección de atención, estadísticas PNA solo de entrenamiento y salidas tensor/mapping de EGNN
   están documentados y probados sin afirmar paridad de checkpoints de autores ni benchmarks.
12. **Visión más allá de encoders — completado**: predicción densa U-Net configurable, FPN genérico
   sobre el contrato compartido de backbone jerárquico, Vision Transformer de resolución variable y
   etapas de residuos invertidos estilo MobileNetV2 son públicas desde Python y YAML recursivo. La
   política de patches sobrantes, las salidas token/mapa, la alineación del decoder con tamaños
   impares y los canales fine-to-coarse son explícitos y tienen pruebas de gradiente.
13. **Investigación tabular ampliada — completado**: preprocesado categórico determinista y objetos
   nativos TabNet, SAINT, AutoInt y DeepFM son públicos desde Python/YAML. Pruebas de forma, rango,
   máscaras y gradiente cubren los núcleos ligeros; los casos de conformidad dan la vía de paridad.
14. **Secuencias largas — completado**: decoder/seq2seq Transformer y Conformer son modelos nativos
   batch-first con contratos explícitos de máscaras. `StateSpaceAdapter` integra módulos estilo
   S4/Mamba sin imponer sus kernels compilados en la instalación base.
15. **Objetos generativos e incertidumbre — completado**: objetivo beta-VAE reutilizable, VQ-VAE,
   schedules lineal/coseno, muestreo DDPM/DDIM, calibración de temperatura e intervalos
   split-conformal son componibles y están probados.
16. **Arquitecturas científicas — completado en la frontera ligera**: Neural ODE/CDE de paso fijo,
   DeepONet, operador Fourier 1D y mensajes tensoriales E(3) escalares/vectoriales nativos tienen
   pruebas numéricas/de forma/equivariancia. Los proveedores opcionales de orden superior usan un
   adaptador validado; irreps `l>=2` nativas y solvers adaptativos rígidos quedan como trabajo futuro.
17. **Packs de conformidad arquitectónica — completado**: casos enlazados a una fuente capturan
   inicialización, parámetros, salidas y tolerancias, guardan referencias pequeñas weights-only y
   agrupan paridad numérica/de forma/checksum en CI sin redistribuir checkpoints de autores.
