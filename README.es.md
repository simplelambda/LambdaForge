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

LambdaForge es el framework orientado a objetos de SimpleLambda para trabajo de IA reproducible.
Combina tareas genéricas, preprocesado componible, PyTorch, Lightning y un motor YAML en un único
paquete estable, para que un proyecto de investigación se concentre en sus datos y su ciencia en vez
de volver a crear pipelines, bucles de entrenamiento, procedencia, gestión de resultados y
planificación de procesos.

> **Estado:** `0.5.2`, utilizable pero anterior a 1.0. Los espacios de nombres públicos documentados
> aquí forman la API prevista; todavía no se garantiza compatibilidad entre versiones menores. El
> repositorio aún no contiene una licencia, por lo que SimpleLambda debe decidir sus condiciones de
> redistribución.

## 0. Contenidos

- Primeros pasos
  - [1. Qué proporciona](#1-qué-proporciona-lambdaforge)
  - [2. Instalación](#2-instalación)
  - [3. Integración](#3-integración-en-otro-proyecto)
  - [4. Inicio rápido](#4-inicio-rápido)
  - [5. Glosario](#5-glosario-en-lenguaje-directo)
  - [6. Autoría sencilla e IR](#6-autoría-sencilla-y-modelo-interno-estricto)
- Conceptos principales
  - [7. Tareas y preprocesado](#7-tareas-genéricas-y-preprocesado)
  - [8. Identidad y reutilización](#8-identidad-científica-reutilización-y-repeticiones-explícitas)
  - [9. Workflows y composición](#9-workflows-y-composición-de-configuración)
- Ejecución y datos
  - [10. Plano de control multiclúster](#10-plano-de-control-local-y-multiclúster)
  - [11. Jobs y ubicación de datos](#11-jobs-persistentes-y-ubicación-de-datos)
  - [12. Operaciones y HPO](#12-inferencia-evaluación-exportación-y-hpo)
  - [13. Recursos y fiabilidad](#13-recursos-backends-y-fiabilidad)
- Resultados e inspección
  - [14. Resultados, plots y artifacts](#14-almacenes-registro-e-informes)
  - [15. Observabilidad](#15-observabilidad-y-reproducibilidad)
- Referencia y extensión
  - [16. CLI](#16-referencia-de-cli)
  - [17. API pública](#17-api-pública)
  - [18. Modelo conceptual](#18-modelo-conceptual-de-ejecución)
  - [19. Arquitectura](#19-arquitectura)
  - [20–26. YAML, ejecución, outputs, componentes y extensión](#20-referencia-de-experimentos-yaml)
- Información del proyecto
  - [27. Hallazgos](#27-hallazgos-de-la-revisión)
  - [28. Desarrollo](#28-desarrollo-y-verificación)
  - [29. Limitaciones](#29-limitaciones-actuales)
  - [30. AGENTS.md](#30-por-qué-existe-agentsmd)
  - [31. Mapa documental](#31-mapa-de-documentación)
  - [32. Hoja de ruta](#32-hoja-de-ruta)
  - [33. Historial 0.2](#33-historial-de-la-hoja-de-ruta-02)

## 1. Qué proporciona LambdaForge

- Una tarea genérica de Lightning para lotes con forma de mapa, una o más pérdidas y métricas
  independientes de entrenamiento, validación y prueba.
- Una familia YAML estricta e independiente de tareas genéricas para preprocesado y otros trabajos
  reproducibles sin entrenamiento, con planes dry-run, entradas dirigidas por contenido, artefactos
  tipados e historial de intentos.
- Preprocesado componible source/transform/sink, checkpoints atómicos por registro, shards
  deterministas y un manifiesto `DatasetArtifact` versionado.
- DAG task/experiment, composición/interpolación segura, procedencia/diff y planificación CPU o
  heterogénea acotada.
- Tareas de inferencia/evaluación/ensemble/export, HPO finito/adaptativo y backends local/SLURM
  preview-first.
- Stores local/compartido/S3-compatible verificados, caché distribuida, registro, informes/dashboard
  factuales y perfiles de observabilidad/reproducibilidad.
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
- Una fachada pequeña (`LambdaForge`), APIs (`Experiment`, `TaskRun`, `Workflow`) y una sola CLI
  (`lambdaforge`).

LambdaForge es agnóstico respecto a la tarea en sus capas de configuración y orquestación. El
proyecto usuario aporta el `Dataset`, el collator opcional y, cuando el contrato de mapas por defecto
no basta, su propio modelo, tarea, módulo de datos o runner.

## 2. Instalación

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

Las integraciones opcionales nunca amplían las dependencias base:

| Extra | Añade |
|---|---|
| `hpo` | Adaptador de búsqueda finita Optuna. |
| `adaptive-hpo` | Adquisición bayesiana BoTorch/GPyTorch para HPO adaptativo; Sobol/random no lo necesitan. |
| `s3` | Cliente boto3 por defecto para `S3ArtifactStore`; un cliente compatible inyectado no requiere el extra. |
| `parquet` | Export del registro mediante Pandas/PyArrow. |
| `onnx` | Export de modelos ONNX/ONNX Script. |
| `cluster-password` | SSH/SFTP por contraseña con Paramiko y keyring del SO; OpenSSH no lo necesita. |
| `mlflow`, `tensorboard`, `wandb`, `tracking` | Un proveedor de tracking o los tres. |
| `dev` | Tests, tipado y formato para contribuir a LambdaForge. |

Instala sólo lo que use el consumidor, por ejemplo
`python -m pip install "lambdaforge[adaptive-hpo,s3]"`.

## 3. Integración en otro proyecto

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
python -m pip install dist/lambdaforge-0.5.2-py3-none-any.whl
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
más simples. Los [contratos de extensión](#26-contratos-de-extensión) muestran ambas vías.

## 4. Inicio rápido

### Las ideas necesarias antes de ejecutar un comando

LambdaForge lee un **fichero de configuración**: un documento YAML de texto plano que indica qué
trabajo debe hacerse, qué objetos Python lo implementan, qué parámetros reciben, dónde se guardan
los resultados y qué recursos pueden utilizarse. YAML depende de la indentación; usa espacios,
nunca tabuladores.

Existen tres tipos de documento:

| Documento | Para qué sirve | Ejemplo |
|---|---|---|
| **Tarea** | Una operación reproducible que no tiene por qué entrenar. | Preprocesar datos, exportar un modelo o generar un informe. |
| **Experimento** | Entrenar y evaluar modelos con configuraciones y seeds. | Comparar dos anchuras de MLP con tres seeds. |
| **Workflow** | Conectar tareas o experimentos completos mediante dependencias. | Preprocesar primero y entrenar después con el dataset producido. |

Hay dos vistas de una configuración. La **configuración de autoría** es la que escribe el usuario:
puede omitir `kind` y `schema_version`, usar nombres lógicos de entrada/salida y abreviar imports no
ambiguos. LambdaForge la compila a una **configuración materializada** estricta y versionada que
consumen los runners existentes. Un **Schema** enumera los campos y tipos permitidos y detecta
errores antes de iniciar trabajo caro. Los YAML estrictos anteriores siguen siendo válidos.

Las especificaciones de objetos reutilizan siempre tres claves:

```yaml
task:
  target: mi_proyecto.tasks.TareaInforme  # importa y construye esta clase hipotética
  params:                                 # argumentos nombrados del constructor
    output_name: resultado.json
```

- `target` es la ruta Python completa de una clase que se debe construir.
- `params` es el mapa enviado como argumentos nombrados a esa clase.
- `ref` importa una función, clase o valor sin construirlo; por ejemplo,
  `ref: torch.optim.AdamW` entrega la clase del optimizador a la tarea de entrenamiento.

Una configuración es código de confianza porque sus imports pueden ejecutar Python. Utiliza
ficheros propios o revisados y conserva los objetos del dominio en el paquete consumidor instalado.

### Ejecutar el ejemplo generado

Empieza con la tarea generada: funciona sin necesitar un dataset ni una GPU.

```bash
lambdaforge init mi-proyecto-ia
cd mi-proyecto-ia
python -m pip install -e .
```

`init` crea un paquete instalable `my_project`, una `ExampleTask` pequeña,
`experiments/task.yaml`, ajustes de Schema para el editor y un `.gitignore` apropiado. Instalar el
proyecto con `-e .` permite importar `my_project.tasks.ExampleTask` mientras se edita.

El YAML generado significa:

```yaml
kind: task                          # selecciona la familia de tareas genéricas
schema_version: "1.0"              # valida con el Schema de tareas 1.0
name: example                       # nombre humano estable de la tarea
task:
  target: my_project.tasks.ExampleTask
required_artifacts: [output.json]   # el éxito exige que exista este fichero
```

Utiliza los comandos en este orden. `inspect --resolved` permite aprender qué significan los
valores por defecto sin ejecutar código del usuario:

| Comando | Qué responde | ¿Inicia trabajo? | ¿Escribe resultados? |
|---|---|---:|---:|
| `lambdaforge validate CONFIG` | ¿Es válido el YAML y se pueden importar los objetos Python referenciados? | No | No |
| `lambdaforge inspect CONFIG --resolved` | ¿En qué configuración estricta se convirtió mi YAML corto? | No | No |
| `lambdaforge inspect CONFIG` | ¿Qué runs o plan exacto se utilizarían? | No | No |
| `lambdaforge run CONFIG --dry-run` | ¿Puede la capa de ejecución preparar el mismo plan inmutable sin lanzar código usuario? | No | No |
| `lambdaforge run CONFIG` | Ejecuta o reanuda de forma segura el trabajo planificado. | Sí | Sí |
| `lambdaforge results CONFIG` | ¿Qué intentos existen y hay resultados exitosos ambiguos? | No | Sólo con `--write-index` |

Recorre ahora el flujo seguro completo:

```bash
lambdaforge validate experiments/task.yaml
lambdaforge inspect experiments/task.yaml --resolved
lambdaforge inspect experiments/task.yaml
lambdaforge run experiments/task.yaml --dry-run
lambdaforge run experiments/task.yaml
lambdaforge results experiments/task.yaml --write-index --fail-on-ambiguous
```

La ejecución real crea un directorio identificado por fingerprint bajo `runs/tasks/example/` con
la configuración materializada, procedencia del entorno, log, eventos, `result.json` y el
`output.json` declarado. Repetir el comando sólo reutiliza un éxito compatible mientras su
identidad y los hashes de sus artefactos sigan siendo válidos.

### Pasar de la tarea de ejemplo al entrenamiento

Un experimento sigue el mismo flujo, pero necesita datasets, modelo, pérdidas, métricas y opciones
de entrenamiento. Copia [el ejemplo completo de experimento](examples/experiment.yaml) en el
proyecto consumidor y sustituye todas las rutas `your_project.*` por clases de su paquete
instalado. El ejemplo es una plantilla, no un dataset incluido ejecutable: LambdaForge no intenta
adivinar el significado ni la forma de los datos del dominio.

```bash
cp /ruta/a/LambdaForge/examples/experiment.yaml experiments/baseline.yaml
lambdaforge validate experiments/baseline.yaml
lambdaforge inspect experiments/baseline.yaml
lambdaforge run experiments/baseline.yaml --dry-run
lambdaforge run experiments/baseline.yaml
```

Las llamadas Python equivalentes usan la misma configuración y las mismas barreras de seguridad:

```python
from lambdaforge import LambdaForge

experiment = LambdaForge.experiment("experiments/baseline.yaml")
report = experiment.validate()  # ValidationReport; no entrena
planned_runs = experiment.expand()  # un mapa materializado por variante y seed
results = experiment.run(dry_run=True)
results = experiment.run()  # lista de valores RunResult tipados
print(results[0].status, results[0].run_dir)
```

Después de entrenar, `aggregate` reconstruye tablas y gráficas entre seeds sin reentrenar,
`results` audita intentos y `retain` prepara un plan de borrado/compresión de artefactos. La
retención sigue siendo de sólo lectura salvo que se indique `--apply`:

```bash
lambdaforge aggregate experiments/baseline.yaml
lambdaforge results experiments/baseline.yaml --write-index --fail-on-ambiguous
lambdaforge retain experiments/baseline.yaml
```

Las opciones de recursos como `--mode parallel` o `--gpus 0,1` sólo sustituyen sus campos YAML de
ejecución correspondientes. Añádelas después de que funcione la configuración secuencial; la
[sección de ejecución](#22-ejecución-y-seguridad-de-procesos) explica la semántica de procesos y GPU.

## 5. Glosario en lenguaje directo

| Término | Significado en LambdaForge |
|---|---|
| **Configuración** | Los valores YAML que describen el trabajo solicitado. |
| **Schema** | Reglas legibles por máquina para rechazar configuración inválida antes de ejecutar. No es el esquema del dataset. |
| **Tarea** | Una operación reproducible que no es entrenamiento. |
| **Experimento** | Un estudio de entrenamiento que puede expandirse en varios runs. |
| **Workflow** | Un grafo de dependencias cuyos nodos son tareas o experimentos completos. |
| **Suite** | Todos los runs producidos por una configuración de experimento. |
| **Variante** | Una combinación concreta base/grid/ablación de hiperparámetros. |
| **Seed** | Identificador registrado de inicialización/repetición aleatoria. |
| **Run** | Una variante y una seed ejecutadas juntas. |
| **Intento** | Una ejecución de un run o tarea; los reintentos conservan su propia metadata terminal. |
| **Fingerprint** | Hash determinista de la identidad científica que impide reutilizar trabajo incompatible. |
| **Artefacto** | Fichero o directorio de salida verificado, como un checkpoint, dataset o informe. |
| **Checkpoint** | Estado de entrenamiento guardado para inferencia o continuación exacta. |
| **Agregación** | Combinar métricas terminadas en resúmenes, comparaciones y gráficas entre seeds. |
| **Procedencia** | Configuración, código/entorno y plugins registrados que explican cómo se produjo una salida. |
| **HPO** | Optimización de hiperparámetros: elegir configuraciones prometedoras de un espacio declarado. |
| **Fidelidad** | Presupuesto acumulado ya entregado a un candidato, normalmente épocas. |
| **Dry-run** | Comprobación de planificación de sólo lectura; nunca llama a la tarea ni entrena. |

`target`, `ref` y `plugin` son formas de resolver objetos, no tipos de documento. `target` construye
una clase importable, `ref` importa un objeto existente y `plugin` resuelve una extensión con nombre
publicada por otra distribución instalada.

## 6. Autoría sencilla y modelo interno estricto

LambdaForge 0.5 separa facilidad de escritura y ejecución estricta:

```text
YAML corto -> AuthoringConfig -> AuthoringConfigNormalizer -> MaterializedConfig -> validador/runner existente
```

No hay dos motores. La capa de autoría sólo expande abreviaturas y valores seguros; no entrena,
importa objetos configurados ni inventa decisiones científicas. El Schema de autoría 1.0 está en
`schemas/authoring.schema.json`; las tareas, experimentos y workflows materializados conservan sus
Schemas estrictos. Se puede ver y validar el resultado con:

```bash
lambdaforge inspect experiments/prepare.yaml --resolved
lambdaforge validate experiments/prepare.yaml
```

Un preprocesado corto completo es:

```yaml
name: prepare-data
inputs: {raw: ../data/raw.jsonl}
outputs: {processed: processed}
preprocess:
  function: my_project.preprocessing.normalize_record
  input: raw
  output: processed
  key_field: id
  workers: 4
  workload: io
resources: {cpus: 4, memory: 8GiB, time: 30m}
```

`raw` y `processed` son nombres lógicos. La fuente usa `context.input("raw")` y el sink,
`context.output("processed")`; las rutas físicas quedan como procedencia. Las APIs antiguas de
rutas siguen siendo compatibles, pero el código nuevo debe preferir nombres. Un worker es
secuencial; `io` usa threads; `cpu` usa procesos `spawn` sólo para transforms y el padre posee
sink/manifest; `auto` usa threads conservadores; `gpu` exige uno. Transforms CPU deben ser
importables/picklables en Linux y Windows y varias GPUs usan shards/jobs explícitos. `workers`,
`workload` y la cadencia de checkpoint son operacionales: no cambian la identidad científica del
preprocesado/dataset y los modos equivalentes deben producir el mismo contenido. Para probar N
registros sin publicar dataset usa `lambdaforge debug CONFIG --records N`; `--intermediates` guarda
sólo artifacts de debug.

Sólo los campos de objeto inequívocos admiten cadenas; por ejemplo
`model: my_project.models.ProjectModel` se materializa como un `target`. Cuando haya parámetros o
duda entre `target` y `ref`, se usa siempre el mapa completo.

## 7. Tareas genéricas y preprocesado

Una tarea genérica es la unidad reproducible mínima de LambdaForge: un objeto Python recibe un
`TaskContext`, realiza trabajo acotado y devuelve outputs, métricas y declaraciones de artefactos.
Es la opción adecuada cuando la operación no es un bucle de entrenamiento Lightning. Los
experimentos usan el Schema 1.1 y las tareas genéricas el Schema 1.0. El YAML estricto declara
`kind: task`; el corto se detecta por `task` o `preprocess` y materializa esa declaración.

El preprocesado es una tarea especializada formada por tres papeles fáciles de distinguir:

1. Una **fuente** produce registros con identificadores estables.
2. Cero o más **transformaciones** modifican cada registro.
3. Un **destino** escribe los resultados y puede verificar si un registro anterior ya está completo.

El ejemplo incluido lee JSON Lines y escribe un JSON atómico por registro sin cambiar su valor. Se
puede ejecutar directamente desde este checkout y no necesita código del proyecto:

```bash
lambdaforge validate examples/preprocessing.yaml
lambdaforge inspect examples/preprocessing.yaml
lambdaforge run examples/preprocessing.yaml --dry-run
lambdaforge run examples/preprocessing.yaml
lambdaforge results examples/preprocessing.yaml --write-index --fail-on-ambiguous
```

Su núcleo YAML es un pipeline source → transforms → sink construido recursivamente. Una lista de
transformaciones vacía significa «copiar/serializar registros»; el comentario muestra dónde añadir
lógica del proyecto:

```yaml
schema_version: "1.0"
kind: task
name: normalize-records
inputs:
  - {name: raw, path: data/raw.jsonl}
task:
  target: lambdaforge.preprocessing.PreprocessingTask
  params:
    source:
      target: lambdaforge.preprocessing.JsonLinesSource
      params: {path: data/raw.jsonl, key_field: id}
    transforms: []
    # Para transformar valores, instala el paquete consumidor y sustituye [] por:
    # - target: lambdaforge.preprocessing.CallableTransform
    #   params: {function: {ref: my_project.preprocessing.normalize_record}}
    sink:
      target: lambdaforge.preprocessing.JsonDirectorySink
      params: {output_dir: processed}
```

Cada entrada de nivel superior se resuelve respecto al YAML y se hashea por contenido antes de
planificar. Cambiar los bytes originales selecciona un nuevo directorio de run con fingerprint en
vez de reutilizar silenciosamente resultados obsoletos. Cada intento terminal registra
configuración, entorno/plugins, logs, errores estructurados, métricas escalares y artefactos
SHA-256. `PreprocessingTask` guarda además checkpoints por clave estable para reintentos seguros,
admite shards explícitos deterministas y publica un `dataset-artifact.json` dirigido por contenido.

La lógica específica permanece en el paquete instalado del proyecto consumidor. Usa
`CallableTransform` para una función pequeña o implementa los contratos públicos
`PreprocessingSource`, `PreprocessingTransform` y `PreprocessingSink`. Para otro trabajo batch,
implementa `Task.run(TaskContext) -> TaskOutput` y usa el mismo YAML/CLI. Consulta la
[guía de tareas genéricas](src/lambdaforge/tasks/README.es.md), la
[guía de preprocesado](src/lambdaforge/preprocessing/README.es.md) y el
[ejemplo completo](examples/preprocessing.yaml).

## 8. Identidad científica, reutilización y repeticiones explícitas

La versión 0.5 separa tres identidades:

| Identidad | Contiene | No contiene |
|---|---|---|
| `DatasetIdentity` | Hash estricto, hash de manifiesto, ID generado o versión externa explícita. | Punto de montaje o ruta del clúster. |
| `CodeIdentity` | Commit Git limpio; commit y hash del diff sucio; release explícita; o versión y hash de fuentes disponibles de un proyecto instalable. | Directorios de salida y scheduler. |
| `ExecutionIdentity` | Clúster, recursos y política de entorno. | Elecciones de modelo, datos y código. |

El valor seguro por defecto lee todos los bytes. Para datasets grandes e inmutables se elige una
estrategia auditable:

```yaml
inputs:
  raw:
    path: /datasets/corpus
    identity: {strategy: manifest, manifest: ../data/corpus.sha256}
# Alternativas: dataset_id o {strategy: version, namespace: lab/corpus, version: "2026-08-11"}
```

`manifest` hashea un manifiesto revisado; `dataset_id` lee el ID emitido por `PreprocessingTask`;
`version` confía en una versión externa inmutable. Cambiar bytes sin cambiar esa versión es un error
del usuario. La ruta física se conserva para auditoría, pero sólo la identidad lógica entra en el
fingerprint de tarea.

La ejecución es idempotente por defecto. Un éxito con la misma identidad y artefactos válidos se
reutiliza. Las excepciones son explícitas:

| Comando | Éxito previo | Estado parcial |
|---|---|---|
| `run CONFIG` | Reutiliza. | Reanuda si está permitido. |
| `run CONFIG --no-resume` | Reutiliza. | Nuevo intento sin continuación. |
| `run CONFIG --force` | Nuevo intento. | Puede reanudar estado compatible. |
| `run CONFIG --restart` | Nuevo intento. | Empieza desde cero. |

`lambdaforge explain changes actual.yaml --against anterior.yaml` muestra qué rutas científicas
cambiaron. Los hashes siguen protegiendo los directorios internos, pero no se obliga al usuario a
gestionarlos manualmente.

## 9. Workflows y composición de configuración

Un workflow conecta documentos completos de tarea o entrenamiento; no inventa una segunda sintaxis
de tarea:

```yaml
kind: workflow
schema_version: "1.0"
name: preparar-y-entrenar
output_root: runs/workflows
max_parallel: 2
nodes:
  preprocess:
    config: preprocessing.yaml
  train:
    config: experiment.yaml
    needs: [preprocess]
    bindings:
      data.train.params.dataset_manifest: >-
        ${nodes.preprocess.artifacts.dataset-artifact.json}
```

Usa `validate`, `inspect`, `run --dry-run` y finalmente `run`. `needs` crea aristas; se rechazan
ciclos y nodos desconocidos. Un fallo bloquea sólo descendientes, no ramas independientes.
`continue_on_failure: true` debe ser explícito. Los bindings exactos admiten
`${nodes.NOMBRE.outputs.RUTA}`, `.metrics.RUTA` y `.artifacts.RUTA_RELATIVA`. Cada nodo conserva su
fingerprint/resume, y `max_parallel` limita los nodos locales listos.

Tareas y experimentos pueden componerse con `extends` (primero), `include` (después) y el documento
hoja (último). Las rutas son relativas al fichero que las declara; los mapas fusionan
recursivamente, las listas sustituyen y `{$delete: true}` elimina una clave heredada:

```yaml
extends: configs/base.yaml
include: [configs/datos-locales.yaml]
model: {params: {dropout: 0.2}}
data_root: ${env:DATA_ROOT}
run_root: ${config:experiment.output_root}
```

`lambdaforge compose study.yaml` muestra valores resueltos ocultando secretos, fuentes y procedencia
por ruta. `lambdaforge diff left.yaml right.yaml` compara semántica, no texto. Sólo existen
`${config:ruta}`, `${env:NOMBRE}` y el valor completo `${secret:NOMBRE}`; nunca se evalúa Python.
En una tarea, el secreto llega al constructor pero `config.yaml` guarda `***`; no puede incrustarse
en otra cadena. Entrenamiento y estructura del workflow rechazan secretos persistibles: el código
del proveedor debe leer credenciales del entorno en ejecución. Las APIs son
`ConfigurationComposer`, `ConfigurationDiff`, `Workflow` y `LambdaForge.workflow()`.

## 10. Plano de control local y multiclúster

El plano de control 0.5 es un coordinador local. Los catálogos se fusionan por nombre con precedencia
usuario (`~/.config/lambdaforge/clusters.yaml`), proyecto (`lambdaforge.clusters.yaml`) y archivo
explícito (`--clusters-file`/`--clusters`). `clusters add` usa usuario por defecto; `--scope project`
es explícito y `clusters inspect` muestra fuente ganadora y tapadas.

```yaml
# lambdaforge.clusters.yaml
clusters:
  atlas:
    transport: ssh
    host: atlas-login
    user: mi-usuario
    auth: {mode: openssh}
    scheduler: slurm
    workspace: /scratch/mi-usuario/lambdaforge
    python: /shared/envs/research/bin/python
    data_environment: atlas
    resource_mapping:
      gpu: {option: gres, value: "gpu:a100:{gpus}"}
      memory: {option: mem, value: "{memory_gib}G"}
    scheduler_directives: {partition: gpu, account: project123, exclusive: true}
  atlas-container:
    transport: ssh
    host: atlas-login
    scheduler: slurm
    workspace: /scratch/mi-usuario/lambdaforge
    command_prefix: [apptainer, exec, /shared/images/project.sif]
    python: python
profiles:
  una-gpu:
    cluster: atlas
    resources: {cpus: 8, memory: 32GiB, gpus: 1, gpu_memory: 20GiB, time: 4h}
```

`command_prefix` es argv. OpenSSH sigue recomendado y conserva aliases, claves, agente,
`known_hosts` y ProxyJump. El modo opcional de contraseña instala
`lambdaforge[cluster-password]`, verifica host con Paramiko y obtiene el secreto por prompt oculto,
referencia `keyring:` o `env:NAME`. No existe `--password`: el valor nunca se serializa, registra,
empaqueta ni fingerprinta.

```bash
lambdaforge doctor
lambdaforge clusters list
lambdaforge clusters inspect atlas
lambdaforge clusters test atlas
lambdaforge run experiment.yaml --on atlas --dry-run
lambdaforge run experiment.yaml --on atlas --cpus 8 --memory 32GiB --resource-gpus 1 --time 4h
lambdaforge run experiment.yaml --profile una-gpu
```

`ResourceRequest` normaliza CPU, RAM, GPU, VRAM, duración, almacenamiento y procesos. Una sola capa
por clúster traduce a `gpus`, GRES genérico/tipado, CPU, memoria y tiempo; admite flags/repeticiones,
comandos submit/queue/accounting/cancel, regex de ID y script confiable. `scheduler_options` antiguo
sigue compatible. Dry-run muestra script, recursos, directivas, avisos y argv exacto. La
[guía de clústeres](docs/CLUSTERS.es.md) contiene esquemas, credenciales y trade-offs completos.

Un `ExecutionBundle` contiene YAML, manifiesto, wheels exactas de LambdaForge/proyecto y sólo inputs
pequeños; se cachea por contenido. El source dirty se construye tal cual, nunca se sustituye por
`git clone main`. `managed` crea/reutiliza un venv de usuario identificado por bytes bajo
`WORKSPACE/.lambdaforge/environments`; `existing` no instala y exige el Python exacto. Sin Internet
se aporta wheelhouse compatible y se usa `--no-index`. LambdaForge verifica PyTorch/CUDA pero no
instala drivers, CUDA de sistema ni cuDNN. El remoto ejecuta el mismo `python -m lambdaforge run`.

`LocalTransport`, `SshTransport` OpenSSH y `PasswordSshTransport` opcional, junto con los schedulers,
son proveedores independientes. `doctor` comprueba auth, workspace, Python/proyecto/framework,
PyTorch/CUDA, todos los ejecutables, mapping y partición sin enviar job. Los tests usan fakes.

## 11. Jobs persistentes y ubicación de datos

Cada envío devuelve un `job_id` corto y escribe JSON atómico en el directorio XDG de estado
(`~/.local/state/lambdaforge/jobs` por defecto). Registra clúster, ID del scheduler, comando exacto,
recursos, bundle, tiempos y linaje de retry. El job describe cómo se programó la ejecución;
`result.json` sigue siendo la evidencia científica.

```bash
lambdaforge status --on atlas --state running --name experimento
lambdaforge status JOB_ID
lambdaforge logs JOB_ID --follow
lambdaforge cancel JOB_ID
lambdaforge retry JOB_ID --dry-run
```

`JobService`, `DataService` y `Doctor` ofrecen las mismas operaciones y `to_dict()` para notebooks,
automatización o una GUI futura; no hay lógica privada sólo para CLI.

El registro incluye identidades científica/ejecución, entorno exacto y paths remotos. Otro proceso
refresca el scheduler, por lo que SLURM sobrevive al cierre de la CLI. `lambdaforge results sync JOB` trae sólo
metadata/métricas/manifests/resúmenes/plots pequeños; `plot learning JOB --follow` los consume.
`artifact list JOB` y `artifact fetch JOB best-checkpoint` enumeran y descargan un artifact lógico
explícito. Nunca se copian checkpoints/datasets grandes de forma implícita.

Los datos grandes nunca se copian por usar `--on`. Un catálogo separa identidad y ubicación:

```yaml
datasets:
  raw-corpus:
    identity: {strategy: version, namespace: lab/raw-corpus, version: "2026-08-11"}
    locations:
      local: /data/raw-corpus
      atlas: /datasets/project/raw-corpus
```

La tarea usa `data_catalog: ../data-catalog.yaml` e `inputs: {raw: dataset:raw-corpus}`. Si falta la
ubicación del entorno destino, el envío falla antes del scheduler. Una ruta ordinaria de hasta
10 MiB se incluye en el bundle; una mayor se rechaza. La transferencia es separada y primero se
previsualiza:

```bash
lambdaforge data --catalog data-catalog.yaml locations raw-corpus
lambdaforge data --catalog data-catalog.yaml replicate raw-corpus --from local --to atlas
lambdaforge data --catalog data-catalog.yaml replicate raw-corpus --from local --to atlas --apply
```

El proveedor incluido usa rsync entre ubicaciones ya declaradas; no adivina destinos ni reescribe
el catálogo. Un workflow puede anotar `on: atlas` y su dry-run muestra ubicaciones. La versión 0.5
rechaza ejecutar un DAG mixto: transferencia de artefactos descendentes y recuperación durable
necesitan semántica más fuerte que sondear logs. Cada nodo remoto se envía explícitamente con
`run --on`; los workflows locales siguen completos.

Los experimentos usan el mismo catálogo. `data.train: dataset:raw-corpus/train` necesita un
`loader` con `path_parameter`; dentro de params se usa `{dataset: raw-corpus, subpath: train}`.
Sólo se resuelven esos marcadores tipados, nunca strings ordinarios. El mount puede variar, pero el
fingerprint conserva referencia e identidad lógica.

## 12. Inferencia, evaluación, exportación y HPO

Las operaciones de modelo son tareas genéricas, con inputs hasheados, intentos, procedencia y
artefactos. Declara cada checkpoint en `inputs`:

```yaml
kind: task
schema_version: "1.0"
name: inferencia-test
inputs: [{name: checkpoint, path: checkpoints/best.ckpt}]
task:
  target: lambdaforge.operations.InferenceTask
  params:
    model: {target: mi_proyecto.models.Modelo, params: {features: 32}}
    checkpoints: checkpoints/best.ckpt
    data: {target: mi_proyecto.data.TestDataset}
    batch_size: 128
    model_input_key: x
```

`InferenceTask` publica `predictions.pt` en CPU; varios checkpoints forman un ensemble promediando
outputs tensoriales coincidentes. `EvaluationTask` recibe una lista `metrics` y evalúa otro dataset.
`ExportTask` exige `example_inputs` y soporta `torchscript`, `torch_export`, `onnx` o un exporter
inyectado con `export(model, args, path)`. Carga state dict plano o envelope Lightning con
`weights_only=True`.

Grid/ablations siguen siendo la búsqueda exhaustiva. `RandomSearch` ofrece `choice`, `uniform`,
`loguniform`, `int`, condiciones `when`, RNG privado y fingerprint de trial:

```python
from lambdaforge.hpo import RandomSearch

search = RandomSearch(
    {
        "optimizer.params.lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "model.params.width": {"type": "choice", "values": [64, 128, 256]},
    },
    seed=17,
)
trial_configs = search.materialize(base_config, count=20)
```

`OptunaSearch` es un adaptador opcional para TPE reproducible y pruning `asha`/`hyperband`; instala
Optuna en el proyecto consumidor. No sustituye identidad, resultados ni scheduling de LambdaForge.

### Optimización adaptativa de experimentos

Usa un bloque superior `hpo` habilitado cuando un grid finito malgaste presupuesto. Parte de
[`examples/adaptive-hpo.yaml`](examples/adaptive-hpo.yaml): el experimento ordinario no cambia y
`hpo.space` enumera rutas científicas con tipos `float`, `int`, `ordinal`, `categorical` o `bool`, escala
lineal/logarítmica y condiciones `when`. `validate` comprueba el Schema e `inspect`/`run --dry-run`
muestran el plan sin crear estado ni entrenar.

Referencia completa de la política adaptativa (los defaults se aplican sólo con
`hpo.enabled: true`):

| Campo | Default | Significado |
|---|---:|---|
| `controller_seed` / `max_concurrency` | `0` / `1` | Decisiones reproducibles y máximo de acciones independientes vivas. |
| `objective.metric` / `direction` | `val_loss` / `minimize` | Objetivo exacto del CSV de épocas y su orientación. |
| `objective.risk.type` / `lambda` | `mean` / `0` | Objetivo científico; `mean_minus_std` es opt-in y cambia la pregunta optimizada. |
| `space` | obligatorio | Rutas con `float`, `int`, `ordinal` ordenado, `categorical` no ordenado o `bool`; escala numérica `linear` salvo `log`; `when` añade condición padre. |
| `initialization.strategy` / `trials` | `sobol` / `auto` | Diseño inicial; `auto = max(4, 2 * (dimensiones_efectivas + 1))`. Random es el baseline. |
| `search.strategy` | `bayesian` | `bayesian`, `sobol` o `random`. |
| `search.candidate_pool_size` / `refresh_interval` | `128` / `1` | Muestras raw e intervalo de observaciones puntuadas para reajustar/cachear el surrogate. |
| `acquisition.strategy` / `exploration_weight` | `cost_aware_knowledge_gradient` / `1` | KG con coste, KG o expected improvement; el selector global conserva coste y factibilidad. |
| `fidelity.strategy` | `adaptive_learning_curve` | Curvas adaptativas, `fixed` o baseline `successive_halving` determinista. |
| `fidelity.min` / `max` / `step` | `5` / `100` / `min` | Fronteras acumulativas de épocas; actualmente sólo se admite `unit: epochs`. |
| `seeds.strategy` / `values` | `adaptive_racing` / `[0]` | Seeds de búsqueda compartidas y ordenadas; `fixed` ejecuta todas. |
| `seeds.confirmation_values` / `max_search_seeds` | `[]` / número de seeds de búsqueda | Seeds finales disjuntas y techo de repeticiones durante search. |
| `seeds.probability_threshold` | `0.9` | Umbral de competitividad para gastar otra seed adaptativa. |
| `pruning.enabled` / `min_budget_before_drop` | `true` / fidelidad mínima | Si se permite pruning competitivo posterior y cuándo puede empezar. |
| `pruning.probability_threshold` / `equivalence_margin` | `0.01` / `0` | Umbral conservador de descarte y margen de equivalencia práctica. |
| `memory.per_job_budget` / `headroom` | `0` / `0` | Reserva lógica/techo opcional del allocator y margen extra; cero desactiva el límite lógico. |
| `memory.safety_quantile` / `min_observations` | `0.99` / `3` | Cuantil feature-aware conservador y evidencia necesaria para abandonar cold start. |
| `memory.resource_features` | `{}` | Nombre genérico → ruta candidata/base, por ejemplo batch size, tokens o resolución. |
| `memory.allocator_cap` / `preflight` | `true` / `false` | Techo PyTorch defensivo y switch compatible para probes aislados candidate-aware. |
| `memory.probe_policy.mode` | `auto` con preflight, si no `never` | `auto`, `always` o `never`; auto considera cold start, incertidumbre, OOD, cercanía al límite y OOM. |
| `memory.unknown_capacity` | `declared_budget` | Usa el budget positivo declarado si falla discovery, o `fail_closed`; UNKNOWN nunca equivale a ilimitado. |
| `memory.device_capacities` | descubierto o CPU UNBOUNDED | Bytes explícitos por GPU; `KNOWN(0)` continúa siendo un cero real. |
| `memory.structural.*` | estados de parámetros/gradientes/optimizador | Estimación opcional por número de parámetros; activaciones/workspaces quedan fuera y fuerzan cautela. |
| `budget.max_actions` / `max_total_epochs` | `50` / acciones × fidelidad máxima | Límites duros, incluidos compromisos pending. |
| `budget.max_gpu_seconds` | sin definir | Techo opcional de tiempo GPU medido/predicho. |
| `confirmation.top_k` | `1` | Configuraciones posteriores congeladas para confirmación con seeds disjuntas. |
| `components.*` | built-ins | Reemplazos `target`/`params` confiables para las ocho fronteras listadas debajo. |

El controlador compara `START_NEW`, `RESUME` y `ADD_SEED` con la misma aproximación Knowledge
Gradient de momentos gaussianos, dividida por coste y multiplicada por viabilidad. Se registra como
`gaussian_moment_knowledge_gradient`, sin presentarla como KG exacto de BoTorch; la confirmación es
una fase científica separada. Comienza con Sobol. El search bayesiano aprende todos los puntos
`f(x,b)`: usa kernel de fidelidad en espacios numéricos y geometría Hamming más fidelidad explícita
en espacios mixtos. Las categorías son canónicas e invariantes a permutación y las condiciones
tienen estado inactivo/máscara. Los trabajos pendientes entran en `X_pending`. Un ajuste fallido se
reintenta con numérica segura antes de registrar `HPO_SURROGATE_FALLBACK` y usar Sobol. Instálalo con
`pip install "lambdaforge[adaptive-hpo]"`; `sobol` y
`random` no añaden dependencias.
`PAUSE` ocurre cooperativamente en el límite de época elegido y `DROP` impide promociones futuras;
no se mata un proceso a mitad de un optimizer step sólo para reordenar un slot. Los incrementos de
fidelidad acotan cuánto tarda en aplicarse una decisión nueva.

La fidelidad son épocas acumuladas: cada pausa deja el último checkpoint y la curva completa; una
promoción restaura modelo, optimizador, scheduler, scaler y estado del bucle y sólo calcula las
épocas restantes. Exige `checkpoint_policy: last`, `last_and_best` o `all`. El pruning HPO es
distinto del early stopping y espera el presupuesto mínimo y una probabilidad posterior
conservadora. Las curvas parciales usan regresión bayesiana de bases, no extrapolación de la última
pendiente, y conservan incertidumbre ante warm-up, curvatura, plateaus y no monotonía. El racing usa
el mismo orden de semillas para todos, aprovecha diferencias pareadas y repite sólo donde reduce
incertidumbre. La confirmación ejecuta el top K a presupuesto completo con semillas disjuntas no
empleadas para seleccionar candidatos.

`memory.per_job_budget` reserva VRAM en frío y puede aplicar un techo público del allocator PyTorch
dentro del hijo. Picos y `resource_features` genéricas alimentan un predictor conservador; una
configuración fuera de distribución aumenta su incertidumbre. Un OOM bajo `L` bytes se conserva
como observación censurada `M(x,z) > L`. `device_capacities` declara capacidad en
clústeres restringidos. No hacen falta `nvidia-smi`, variables propias, MIG/MPS ni privilegios de
administración. El techo es defensivo, no aislamiento físico, y nunca cambia silenciosamente el
batch size.
Con `preflight: true` y `probe: {target: ...}`, el callable recibe
`(materialized_candidate, resource_context)`, construye esa candidata y ejecuta
forward/backward/step representativos en un hijo aislado sobre la GPU lógica elegida. La política
determinista evita probes innecesarios; los probes antiguos sin argumentos siguen admitidos. El
framework no inventa un batch supuestamente representativo para un dominio arbitrario.
Las acciones adaptativas actuales son trials independientes de un proceso; se rechaza DDP en HPO
porque la reserva de grupo y el techo por rank no serían fiables. Los experimentos estáticos siguen
soportando DDP.

Un estudio real guarda `state.json` atómico, `events.jsonl` explicativo y `summary.json` en
`SUITE/.lambdaforge/adaptive/STUDY_ID/`; cada acción sigue siendo un run ordinario auditable.
Relanzar el mismo YAML reconcilia trabajos terminados y continúa la secuencia. La salida informa de
acciones, épocas, equivalentes de entrenamiento completo, segundos GPU, OOM y fallbacks reales, sin
inventar ahorro contrafactual. También incluye las seeds de búsqueda/confirmación por configuración,
curvas combinadas, picos de memoria y, para confirmación, media, desviación muestral, error estándar,
intervalo normal del 95 % y diferencias pareadas sobre seeds compartidas. Con una sola seed la
dispersión y el intervalo quedan correctamente como null; usa la capa de agregación/estadística para
bootstrap o análisis no paramétricos de publicación. Sweep y HPO habilitado son excluyentes. El ejemplo
contiene todos los ajustes y el diseño interno está en
[`docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md`](docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md).

Los defaults son políticas reemplazables. `hpo.components` admite `target`/`params` del paquete
instalado del consumidor para `searcher`, políticas de fidelidad/semillas, modelos de curva/coste/
memoria, controlador de admisión y selector. El modelo de memoria lee `parameters` y
`resource_features`, conserva lower bounds censurados y devuelve una estimación conservadora.
Implementa el mismo método público de la clase built-in
(consulta su firma con `lambdaforge target`): searcher usa `propose(space, state, count)`, fidelidad
usa `resume_candidates(state)` y opcionalmente `dominated(state, model)`, y seeds usa
`candidates(state, model)`. Son fronteras por duck typing: no hay que heredar una clase built-in ni
el runner, y no se toca Lightning interno.

## 13. Recursos, backends y fiabilidad

Para barridos paralelos sólo CPU omite `gpus`:

```yaml
execution:
  mode: parallel
  cpu_jobs: 4
  cpu_cores_per_job: 2
  cpu_threads_per_job: 2
  cpu_interop_threads_per_job: 1
  dataloader_num_workers_per_job: 0
```

Se rechaza `cpu_jobs * cpu_cores_per_job` superior a la afinidad disponible. Cada slot CPU oculta
CUDA y parchea Lightning a CPU. Los slots GPU conservan `jobs_per_gpu` y DDP usa
`devices_per_job`.

`ResourceRequest` declara CPU, RAM, GPU/memoria GPU, almacenamiento y duración opcional.
`ResourcePlanner` valida waves manuales o empaqueta first-fit determinista y devuelve picos y
estimaciones sólo cuando han sido declaradas. `ExecutionBackend` separa plan y lanzamiento.
`LocalExecutionBackend` ejecuta argv; `SlurmExecutionBackend` siempre genera primero
`submit.sbatch` y sólo llama `sbatch` con `dry_run=False`. Admite partition, nodos, array,
dependencias, prefijo de contenedor, entorno y requeue, cita argumentos y nunca usa `shell=True`.

`FailureClassifier` distingue cancelación, preemption, OOM CPU/GPU, transitorio, error usuario y
desconocido. `RetryPolicy` limita categorías, intentos y backoff. `AttemptMode` separa: resume
reutiliza estado compatible, restart empieza limpio, retry repite un fallo y fork crea identidad.
Errores de usuario/desconocidos no se reintentan por defecto.

## 14. Almacenes, registro e informes

Empieza por servicios estables, no por directorios hash:

```bash
lambdaforge results list --root runs
lambdaforge results show baseline --root runs
lambdaforge results compare baseline ablation --metric val_loss --direction minimize
lambdaforge results export baseline --series --format csv --output analysis/curves.csv
```

Los selectores aceptan config/path, attempt ID, fingerprint, nombre de run/experimento o variante.
`show` devuelve candidatos y ambigüedad, sin elegir por fecha. La sintaxis anterior `results SOURCE
--write-index` sigue compatible como `results audit`. `MetricSeries` normaliza el `metrics.csv`
existente a run/seed/variant/split/metric/step/value/timestamp. JSON/CSV son core y Parquet opcional.

Los plots consumen resultados fuera del training:

```bash
lambdaforge plot learning baseline --metric val_loss --aggregate mean --uncertainty std
lambdaforge plot seeds baseline --metric val_accuracy --kind violin
lambdaforge plot sweep sweep.yaml --x optimizer.params.lr --metric val_loss
lambdaforge plot sweep sweep.yaml --x model.params.width --y optimizer.params.lr \
  --metric val_accuracy --output plots/sweep.html
lambdaforge plot sweep sweep.yaml --x model.params.width \
  --metric val_loss --metric val_accuracy --normalize
lambdaforge plot hpo runs/STUDY/.lambdaforge/adaptive/ID --parameter optimizer.params.lr
```

Curvas/celdas muestran `n`; con una seed no inventan std/CI. Celdas 2-D ausentes siguen ausentes
salvo interpolación explícita. `--normalize` aplica min-max por métrica sólo sobre celdas observadas
y conserva los valores originales en `PlotSpec`. Las comparaciones calculan delta contra el primer
selector y sólo llaman best/worst con `--direction` explícito. HPO/recursos sólo leen evidencia
existente. `VisualizationService`
crea `PlotSpec` (`--json`), renderiza atómicamente y guarda `FIGURE.plot.json` con timestamp para
caché. Dentro de `run/plots/`, figura y spec aparecen en `artifact list`.

NPY/NPZ se inspecciona con `lambdaforge artifact inspect`, siempre `allow_pickle=False`, preview acotado y
muestra estadística determinista en arrays grandes; también CSV/TSV/JSON/JSONL. `artifact export`
y `artifact validate` permiten depurar/validar. Graph/point-cloud/mesh exige roles/tipo explícitos;
`artifact visualize graph.npz --type graph --nodes positions --edges edge_index` nunca adivina por
shape. Inspector, visualizer, schema, exporter y validator son extensiones separadas. Véanse
[resultados](docs/RESULTS.es.md), [artifacts](docs/ARTIFACTS.es.md),
[clusters](docs/CLUSTERS.es.md) y [preprocesado](docs/PREPROCESSING.es.md).

`ArtifactReference` contiene store, key, SHA-256, tamaño y media type. `ArtifactStore` publica
contenido inmutable y stagea una copia verificada. `LocalArtifactStore` cubre filesystem
local/compartido; `S3ArtifactStore` acepta cliente compatible inyectado o `boto3` opcional. Se
rechazan claves absolutas/traversal y cada stage valida tamaño/hash.

`DistributedArtifactCache(root, upstream)` coordina leases en filesystem compartido, publica de
forma atómica, repara corrupción e invalida sólo caché. El contrato base no tiene delete: una
referencia no puede ser eliminada accidentalmente por retención local.

```bash
lambdaforge results experiments/study.yaml --write-index --fail-on-ambiguous
lambdaforge registry runs --output analysis/registry.csv
lambdaforge dashboard runs --output analysis/dashboard.html
```

`ExperimentRegistry` lee `ResultCatalog` y snapshots, nunca una segunda base. `RegistryQuery` filtra
estado, nombre, tags requeridos, metadata y fingerprint. Exporta JSON/CSV o Parquet opcional.
`ExperimentComparator` produce conteos, medias, desviaciones, intervalos normales configurados,
efectos y diferencias semánticas. `ReportBuilder` genera Markdown/HTML y figura factual sin elegir
ganador ni inventar conclusiones. `LocalDashboard` es un HTML estático de sólo lectura.

## 15. Observabilidad y reproducibilidad

Cada tarea guarda `events.jsonl` acotado con inicio/fin y categoría de fallo, además de `task.log`.
`EventLogger` permite eventos consumidores bajo lock. `ResourceMonitor.sample()` entrega CPU, RSS,
threads, memoria CUDA opcional y throughput con frecuencia máxima. `ProfilerAdapter` desacopla el
proveedor y `TorchProfilerAdapter` impone una ventana finita.

```python
from lambdaforge.reproducibility import ReproducibilityProfile, SeedDeriver

profile = ReproducibilityProfile.named("strict", seed=7)
profile.apply()
loader_seed = SeedDeriver(7).derive("dataloader", "train", 0)
fingerprints = profile.fingerprints(config_materializada)
```

Los perfiles son `fast`, `repeatable` y `strict`; strict activa algoritmos deterministas. Las
semillas derivan con SHA-256, no `hash()`. Los fingerprints científico e infraestructura se guardan
por separado. `EnvironmentExporter` produce `pip`, `conda` o snapshot JSON para contenedor sin
modificar el entorno.

## 16. Referencia de CLI

| Comando | Finalidad | Escribe por defecto |
|---|---|---:|
| `init DIRECTORIO [--template minimal|preprocessing|training|full]` | Scaffold enfocado; no pisa sin `--force`. | sí |
| `doctor [--on CLUSTER]` | Comprueba Python, framework, scheduler y PyTorch/CUDA. | no |
| `validate CONFIG` | Schema/imports/recursos/DAG. | no |
| `inspect CONFIG --resolved` | Compila autoría corta al documento estricto. | no |
| `inspect CONFIG` | Runs expandidos o plan task/workflow. | no |
| `run CONFIG --dry-run` | Plan exacto. | no |
| `run CONFIG` | Ejecuta experimento, tarea o workflow. | sí |
| `run CONFIG --force|--restart|--no-resume` | Controla reutilización y continuación parcial. | sí |
| `run CONFIG --on CLUSTER|--profile PROFILE` | Cachea un bundle y envía al plano de control. | metadata; remoto sin dry-run |
| `clusters add|list|show|inspect|export|credentials|test|bootstrap` | Gestiona perfiles por ámbito, credenciales externas, diagnóstico y entorno exacto. | add/credentials/bootstrap |
| `status|logs|cancel|retry` (`jobs ...`) | Filtra, reconecta, sigue y controla jobs. | cancel/retry |
| `data --catalog FILE list|locations|inspect|replicate` | Ubicaciones/manifest y réplica explícita. | sólo `--apply` |
| `compose CONFIG` | Materialización oculta + procedencia. | no |
| `diff LEFT RIGHT` | Diferencias semánticas. | no |
| `explain authoring|experiment|task|workflow RUTA` | Fragmento de JSON Schema. | no |
| `explain changes CONFIG [--against OLD]` | Identidad científica y cambios exactos. | no |
| `target IMPORT.PATH` | Firma y docstring. | no |
| `migrate CONFIG` | Preview; `--output` es explícito. | no |
| `plugins` | Metadata sin importar proveedor. | no |
| `results SOURCE` / `results audit SOURCE` | Auditoría compatible; índice sólo con flag. | no |
| `results list|show|compare|export|sync` | Selectores, estadística, tablas y evidencia remota ligera. | export/sync |
| `plot learning|sweep|seeds|hpo|resources` | `PlotSpec` JSON o figuras atómicas. | salvo `--json` |
| `artifact inspect|export|validate|visualize|list|fetch|plugins` | Inspección acotada y retrieval/geometría explícitos. | según acción |
| `debug CONFIG --records N` | Muestra preprocesado sin sink/finalización de producción. | sólo intermediates |
| `aggregate CONFIG` | Regenera agregados. | sí |
| `retain CONFIG` | Preview; sólo `--apply` muta. | no |
| `registry ROOT [--output FILE]` | Consulta/exporta registro. | sólo con output |
| `dashboard ROOT --output FILE` | Snapshot HTML de sólo lectura. | sí |

`lambdaforge init mi-proyecto --template preprocessing` es la vía rápida para datos; `training`
crea un baseline pequeño ejecutable, `minimal` una tarea y `full` ambas familias. Renombra
`my_project`, implementa el dominio, instala con `pip install -e .` y valida.

## 17. API pública

Los puntos de entrada admitidos son deliberadamente reducidos:

| Punto de entrada | Finalidad |
|---|---|
| `from lambdaforge import LambdaForge` | Cargar, ejecutar o construir objetos mediante la fachada. |
| `from lambdaforge import MaterializedConfig, JobHandle` | Inspeccionar autoría compilada y envíos persistentes. |
| `from lambdaforge import Experiment` | Inspeccionar, ejecutar, agregar y cargar una suite. |
| `from lambdaforge import TaskRun, TaskResult, TaskExecutionPlan` | Validar, inspeccionar, ejecutar y auditar una tarea genérica. |
| `from lambdaforge import Workflow, WorkflowPlan, WorkflowResult, WorkflowValidationReport` | Validar, planificar y ejecutar un DAG task/experiment. |
| `from lambdaforge import RunResult, AggregateResult` | Resultados tipados e inmutables compatibles con dict/JSON legado. |
| `from lambdaforge import ResultCatalog, ResultRecord` | Discovery por identidad y selección explícita del historial de intentos. |
| `from lambdaforge import ResultService, VisualizationService, PlotSpec, ArtifactService` | Servicios estables de consulta, plots e inspección. |
| `from lambdaforge import ArtifactRetentionPlan, ArtifactRetentionResult` | Previsualizaciones y resultados tipados e inmutables de retención. |
| `lambdaforge.data` | Identidad/catálogo/ubicación, transferencias, adaptadores y cachés. |
| `lambdaforge.tasks` | Contratos genéricos de tarea, contexto, plan, resultado y artefacto. |
| `lambdaforge.preprocessing` | Preprocesado componible de registros y manifiestos de dataset. |
| `lambdaforge.configuration` | Autoría a IR, includes, interpolación, secretos, procedencia y diff. |
| `lambdaforge.controlplane` | Clústeres, transportes, schedulers, bundles, doctor y jobs. |
| `lambdaforge.results` | Selectores humanos, MetricSeries, comparación/export y sync remoto. |
| `lambdaforge.visualization` | PlotSpec independiente de renderer y escritura atómica. |
| `lambdaforge.artifacts` | Contratos inspector/visualizer/schema/validator y servicios. |
| `lambdaforge.workflows` | Configuración, nodos, planes y resultados de DAG. |
| `lambdaforge.operations` | Tareas de inferencia, evaluación, ensemble y exportación. |
| `lambdaforge.hpo` | Random/Optuna finito y optimización adaptativa multi-fidelidad persistente. |
| `lambdaforge.execution` | Recursos, backends local/SLURM y retry. |
| `lambdaforge.storage` | Referencias, stores y caché distribuida. |
| `lambdaforge.registry` | Consultas, comparación, informes y dashboard sobre catálogo. |
| `lambdaforge.observability` | Eventos, monitorización y adaptadores de profiler. |
| `lambdaforge.reproducibility` | Identidades científicas/código/ejecución, perfiles, semillas y entorno. |
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
model = LambdaForge.build(
    {
        "target": "lambdaforge.nn.models.MLP",
        "params": {"in_features": 32, "out_features": 1, "hidden": [64, 32]},
    }
)
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

## 18. Modelo conceptual de ejecución

LambdaForge separa **intención científica**, **planificación operativa** y **evidencia terminal**.
Este es el modelo mental compacto que conecta experimentos, workflows y optimización adaptativa.

Sea `c` una configuración científica completamente compuesta y `o` sus controles operativos, como
rutas de salida, concurrencia o retención. Su identidad científica es conceptualmente:

```text
fingerprint científico = SHA-256(configuración científica canónica)
```

Cambiar `o` no crea silenciosamente otra afirmación científica. Cambiar modelo, datos, optimizador,
pérdida, hiperparámetros muestreados o seed sí. Un run sólo se omite o reanuda cuando coinciden esta
identidad y el contrato de artefactos/checkpoint; en otro caso se crea otro intento y se conserva el
resultado terminal anterior.

Un experimento ordinario se expande a un conjunto finito de trabajos:

```text
trabajos = variantes × seeds
```

donde `V` contiene variante base/grid/ablations y `S` las seeds declaradas. `sequential` ejecuta en
el llamador, `parallel` asigna trabajos independientes a slots CPU/GPU explícitos y `ddp` asigna un
trabajo a un grupo de dispositivos. Los slots estáticos respetan `cpu_jobs`, `jobs_per_gpu` y
`devices_per_job`; no inventan una estimación de VRAM. Cuando se solicita planificación portable,
cada trabajo declara un vector de recursos `r_j` y el planner busca una asignación `z_{jd}` que, para
cada recurso/dispositivo `d`, cumpla:

```text
suma de recursos reservados por trabajos en el dispositivo d ≤ capacidad declarada de d
```

con `C_d` como capacidad declarada. El planner usa first-fit/waves deterministas, sin afirmar que
resuelve un packing global óptimo y costoso.

El HPO adaptativo sustituye la lista finita por decisiones repetidas. Una configuración es `x`, una
seed es `s`, la fidelidad acumulada es `b <= B` y el objetivo observado es `Y(x,s,b)`. El objetivo
científico es la esperanza entre seeds a presupuesto completo:

```text
valor científico de x = esperanza de Y(x, seed, presupuesto completo) entre seeds
```

En la decisión `t`, el historial `D_t` contiene curvas completas, seeds, trabajos pendientes, coste,
picos de memoria y fallos. Las acciones candidatas incluyen iniciar `x`, reanudar `(x,s)`, añadir
una seed o confirmar un finalista. Su utilidad aproximada común es:

```text
utilidad(acción | historial)
    = valor de información(acción | historial)
      / coste incremental esperado(acción | historial)
      × probabilidad(la acción cabe en memoria | historial)
```

`I` es Value of Information: BoTorch usa KG multi-fidelity para proponer `x`, mientras las acciones
START/RESUME/ADD_SEED comparten una aproximación KG gaussiana de un paso documentada; ya no es la
heurística `improvement + uncertainty`. `C` es tiempo incremental predicho. La admisión reserva
`R_M = Q_q(M | D_t) + headroom`; sólo empaqueta
acciones cuyas reservas caben. La probabilidad influye en el ranking y la reserva funciona como
restricción dura del scheduler.

El surrogate opcional observa cada `Y(x,s,b)` disponible en lugar de sustituir la curva por un
único target extrapolado. Las categorías no ordenadas usan Hamming, las ordinales conservan orden y
una condición inactiva tiene estado/máscara propios. Con varianzas de estimación internas
`v₁, …, vₙ` y varianza poblacional estimada entre seeds `tau²`, la incertidumbre se propaga así:

```text
varianza de la media estimada = tau² / n + (v₁ + ... + vₙ) / n²
```

Aquí `n` es el número de seeds. El primer término representa la variación real entre seeds; el
segundo, la incertidumbre que aún queda dentro de cada estimación de curva. Por ejemplo, con dos
seeds, `tau² = 4` y `v₁ = v₂ = 1`, el resultado es `4/2 + (1+1)/4 = 2.5`. El término `v` se divide
por `n²` una sola vez porque la media aritmética pondera cada seed con `1/n`.

La capacidad de memoria es `UNKNOWN`, `UNBOUNDED` o `KNOWN(N)`: fallar al descubrir una GPU no
desactiva protecciones y `KNOWN(0)` no puede confundirse con ilimitado.

El bucle online es: incorporar evidencia terminada → actualizar curvas/coste/memoria → generar
acciones nuevas/resume/seed → rechazar acciones inseguras o sin presupuesto → ordenar por utilidad
→ best-fit en recursos libres → lanzar. Un slot libre recibe trabajo sin esperar una barrera global.
La pausa es una frontera de fidelidad con checkpoint, por lo que una promoción continúa de `b` a
`b + delta` en vez de recomputar de `0` a `b + delta`.

La incertidumbre dirige exploración y seeds adicionales, pero no penaliza la media científica salvo
que se elija explícitamente `mean_minus_std`. Al final se congelan hiperparámetros y seeds disjuntas
de confirmación estiman el resultado analizable. Son aproximaciones transparentes y reemplazables,
no una promesa de que una heurística sea óptima para cualquier dominio.

## 19. Arquitectura

```text
LambdaForge/
├── .github/workflows/             # CI CPU alojada y CUDA self-hosted opcional
├── examples/                     # plantillas de configuración
├── src/lambdaforge/
│   ├── EnvironmentManifest.py     # procedencia tipada de la ejecución
│   ├── LambdaForge.py            # fachada única y fácil de descubrir
│   ├── cli/                      # objeto de línea de comandos
│   ├── configuration/            # composición, secretos, procedencia y diff
│   ├── data/                     # adaptadores seguros y backends de caché acotados
│   ├── execution/                # recursos, backends y políticas de retry
│   ├── experiments/              # YAML, barridos, ejecución, agregación y retención
│   ├── integrations/             # adaptadores de compatibilidad externos
│   ├── hpo/                      # optimización finita y adaptativa por acciones
│   ├── metrics/                  # contratos; familias binaria/multiclase/regresión
│   ├── nn/                       # modelos, pérdidas y componentes neuronales
│   ├── observability/            # eventos JSONL, recursos y profilers
│   ├── operations/               # tareas de inferencia, evaluación y export
│   ├── plugins/                  # extensiones lazy desde paquetes instalados
│   ├── preprocessing/            # pipelines source/transform/sink e identidad de datasets
│   ├── registry/                 # consultas, comparaciones, informes y dashboard
│   ├── reproducibility/          # perfiles, semillas y exports de entorno
│   ├── runtime/                  # locks cross-process compartidos de archivos
│   ├── schemas/                  # JSON Schemas de experimentos y tareas genéricas
│   ├── tasks/                    # planes, ejecución, resultados y artefactos no de entrenamiento
│   ├── storage/                  # stores, referencias y caché distribuida
│   ├── tracking/                 # adaptadores logger opcionales y guardas de dependencias
│   ├── training/                 # núcleo Lightning más callbacks/datos/orquestación
│   └── workflows/                # DAG de tareas/experimentos
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

## 20. Referencia de experimentos YAML

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

## 21. Migraciones de configuración

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

## 22. Ejecución y seguridad de procesos

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

## 23. Salidas, reanudación y carga

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
records = experiment.results()  # incluye intentos archivados
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

## 24. Retención de artefactos

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

## 25. Componentes incluidos

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

## 26. Contratos de extensión

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
callbacks/loggers Lightning y tareas genéricas en los grupos canónicos de la
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

## 27. Hallazgos de la revisión

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

## 28. Desarrollo y verificación

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
Las pruebas de HPO cubren permutación categórica/máscaras condicionales, BoTorch mixto
multi-fidelity, pending y fallback; incertidumbre analítica entre seeds, slow starters, racing
pareado, pruning, VoI, memoria feature-aware/censurada, estados de capacidad, dispatch asíncrono,
persistencia y confirmación. Los colaboradores sintéticos evitan entrenar redes para probar lógica.
Las rutas CUDA reales cubren resume acumulativo sin repetir épocas, preflight candidate-aware, OOM
aislado, allocator cap y trials concurrentes en una GPU. El smoke de dos GPU sólo corre con dos
dispositivos visibles.
Las pruebas de integración crean un árbol real launcher/worker/descendiente. POSIX entrega un
`killpg(SIGINT)` real al grupo de procesos; Windows pide al launcher que provoque un SIGBREAK Python
dirigido porque un evento nativo de control afectaría a todo el grupo de pruebas. Otro escenario
termina el launcher abruptamente y verifica que no quede ningún descendiente registrado ni archivo
temporal. La limpieza de emergencia de cada prueba evita además que un fallo de aserción deje
workers residuales.

### Higiene del repositorio y release

Git debe contener fuentes, tests, Schemas, ejemplos, documentación humana/de agentes, workflows,
iconos y metadata de empaquetado. No debe contener entornos virtuales, credenciales, bytecode/cachés
de herramientas, wheels construidos, informes de profiler/tests ni salidas locales de experimentos
o tracking. El `.gitignore` raíz cubre esas categorías, incluidas `.lambdaforge/`, `runs/`, carpetas
de proveedores, dashboards y stdout/stderr de SLURM. El scaffold generado aplica las mismas reglas
esenciales.

No resuelvas la higiene ignorando extensiones científicas amplias como `*.yaml`, `*.json`, `*.csv`,
`*.pt` o todo `data/`: Schemas, protocolos, fixtures pequeñas y assets de referencia revisados pueden
pertenecer a Git. Coloca salidas bajo `runs/` y decide explícitamente si cada dataset/checkpoint grande
es un artefacto externo o un asset revisado. `.env.example` puede documentar nombres, pero los `.env*`
reales se ignoran y nunca deben contener credenciales versionadas.

Antes del commit de release revisa tanto el estado visible como el ignorado:

```bash
git status --short
git status --ignored --short
git diff --check
ruff format --check . && ruff check .
mypy src/lambdaforge && pytest -q
python -m pip wheel . --no-deps --wheel-dir /tmp/lambdaforge-wheel-check
```

El wheel debe contener Schemas, READMEs especializados, `AGENTS.md`, changelog, documentos de
arquitectura y ejemplos ejecutables. Elegir y añadir una licencia sigue siendo la única decisión legal
del owner previa a redistribución; no se infiere una licencia porque el código sea público.

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

## 29. Limitaciones actuales

- `DatasetCache` limita payloads serializados retenidos por proceso, no el RSS total. Lotes,
  prefetch, memoria pinned, overhead del allocator y dataset origen quedan fuera; activar cachés en
  workers multiplica el presupuesto entre réplicas de proceso.
- Pickle sigue siendo el default de compatibilidad y puede ejecutar código: selecciona el codec
  NumPy/Torch seguro cuando sea compatible o limita pickle a almacenamiento local confiable. Un
  checksum no autentica; HMAC debe configurarse y no cifra.
- Los fingerprints de dataset/transformación siguen siendo explícitos. `DatasetCache` coordina
  procesos en un filesystem; el stage cross-machine usa `DistributedArtifactCache`, directorio de
  leases compartido y upstream explícito.
- Lightning es el único backend de entrenamiento incluido.
- La tarea predeterminada presupone lotes con forma de mapa y dirige una o varias entradas; los lotes
  tupla y flujos manuales/con varios optimizadores necesitan una tarea propia.
- Las métricas de curva exactas binarias y multiclase siguen reteniendo predicciones. Sus alternativas
  streaming introducen aproximación por bins; el estado multiclase crece como
  `O(num_classes * num_bins)`.
- El discovery cubre contratos neuronales reutilizables, datasets, callbacks, loggers y tareas
  genéricas. Los data modules y runners de experimento siguen soportados mediante `target` y
  deliberadamente no tienen grupos dedicados.
- La procedencia de plugins cubre el proceso/contexto del run; procesos hijo creados por el usuario
  necesitan IPC explícito si sus cargas independientes se deben atribuir al padre.
- Los resúmenes estadísticos son exploratorios, no sustituyen el protocolo de cada estudio. Los
  intervalos normales y Wilcoxon asintótico son aproximaciones cuando se seleccionan explícitamente
  o `auto` los elige para muestras pareadas mayores.
- Los Schemas de experimento 1.0/1.1 y task/workflow 1.0 están vigentes. No hay downgrade,
  reescritura in-place ni origen remoto. Los secretos de task se ocultan; experimento/workflow los
  rechazan y esperan credenciales del entorno.
- La retención de runs trabaja sólo sobre filesystem local y ZIP/Deflate. Preview puede quedar
  obsoleto y apply replantea bajo locks. Los `ArtifactStore` publican/stagean; borrado remoto y
  lifecycle siguen perteneciendo al proveedor.
- Los adaptadores de tracking para MLflow, TensorBoard y Weights & Biases son opcionales.
  Autenticación/red/almacenamiento del proveedor, retención remota y disponibilidad del servicio
  siguen siendo externas; un fallo del tracker hace fallar su run y la retención de LambdaForge no
  puede eliminar artefactos ya subidos. Tracking no es la fuente de verdad de resultados.
- Los workflows siguen siendo locales y acotados. Los planes registran `on` y el plano de control
  envía configs individuales por local/SSH y local/SLURM, pero 0.5 no finge coordinar transferencia
  de artefactos ni recuperación durable de un DAG mixto.
- Bootstrap managed instala wheels exactas en un venv de usuario, pero no sintetiza wheels de otra
  plataforma/CUDA, drivers, módulos del centro ni contenedores. Offline requiere wheelhouse
  compatible; existing sigue en manos del usuario. Réplica usa rsync sobre ubicaciones declaradas.
- La selección es explícita en 0.5.2: no descubre capacidad/colas/coste ni promete placement.
  `DataCatalog` resuelve splits y marcadores anidados tipados; strings arbitrarios siguen siendo del
  proyecto. Sync remoto es allowlisted/acotado y artifacts pesados requieren fetch lógico.
- Random/Optuna finito permanece. HPO adaptativo agenda trials locales independientes; integrar
  recursos del DAG, acciones DDP y callbacks remotos de pruning no es implícito.
- BoTorch mixto modela fidelidad y geometría Hamming. El KG gaussiano de acciones y las bases de
  curva son aproximaciones documentadas, no un posterior conjunto universal exacto. Las features
  de recursos y probes representativos siguen siendo responsabilidad del consumidor.
- El store S3 depende del cliente/metadata y no implementa multipart-resume, leases provider-side ni
  lifecycle destructivo. La caché distribuida necesita un filesystem coherente para leases.
- El dashboard es un snapshot HTML estático, no servicio multiusuario. Sus intervalos de comparación
  son aproximaciones normales; para publicación usa agregación bootstrap/Wilcoxon y protocolo.
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

## 30. Por qué existe AGENTS.md

Un agente de programación no debería leer cientos de módulos de implementación y todos los README
especializados antes de poder configurar un modelo o añadir una loss. Ese enfoque consume contexto
y dinero, aumenta la probabilidad de olvidar restricciones leídas al principio y lleva al agente a
deducir APIs desde ficheros internos que no forman una interfaz estable.

[AGENTS.es.md](AGENTS.es.md) y su edición [inglesa](AGENTS.md) son el manual operativo único y
eficiente en tokens del framework.
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
el `AGENTS.md` del proyecto consumidor. Las wheels instalan ambas ediciones bajo
`share/lambdaforge`; obtén su ruta exacta sin importar el framework:

```bash
python -c "from importlib.metadata import distribution; print(distribution('lambdaforge').locate_file('share/lambdaforge/AGENTS.md'))"
```

## 31. Mapa de documentación

- [Manual de agentes](AGENTS.es.md) · [English](AGENTS.md)
- [Arquitectura técnica](docs/ARCHITECTURE.es.md) · [English](docs/ARCHITECTURE.md)
- [Clusters y jobs](docs/CLUSTERS.es.md) · [English](docs/CLUSTERS.md)
- [Seguridad de credenciales/scheduler](docs/SECURITY.es.md) · [English](docs/SECURITY.md)
- [Resultados y plots](docs/RESULTS.es.md) · [English](docs/RESULTS.md)
- [Inspección de artifacts](docs/ARTIFACTS.es.md) · [English](docs/ARTIFACTS.md)
- [Ejecución/debug de preprocesado](docs/PREPROCESSING.es.md) · [English](docs/PREPROCESSING.md)
- [Autoría y configuración](src/lambdaforge/configuration/README.es.md) · [English](src/lambdaforge/configuration/README.md)
- [Plano de control](src/lambdaforge/controlplane/README.es.md) · [English](src/lambdaforge/controlplane/README.md)
- [Internos del optimizador](docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.es.md) · [English](docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md)
- [Changelog](CHANGELOG.md) · [Versionado](docs/GOVERNANCE.es.md) · [English](docs/GOVERNANCE.md) · [Seguridad](SECURITY.md)
- [Sistema de experimentos](src/lambdaforge/experiments/README.es.md) · [English](src/lambdaforge/experiments/README.md)
- [Migraciones de configuración](src/lambdaforge/experiments/migrations/README.es.md) · [English](src/lambdaforge/experiments/migrations/README.md)
- [Retención de artefactos](src/lambdaforge/experiments/retention/README.es.md) · [English](src/lambdaforge/experiments/retention/README.md)
- [Comparaciones estadísticas](src/lambdaforge/experiments/statistics/README.es.md) · [English](src/lambdaforge/experiments/statistics/README.md)
- [Datos y caché](src/lambdaforge/data/README.es.md) · [English](src/lambdaforge/data/README.md)
- [Tareas genéricas](src/lambdaforge/tasks/README.es.md) · [English](src/lambdaforge/tasks/README.md)
- [Preprocesado](src/lambdaforge/preprocessing/README.es.md) · [English](src/lambdaforge/preprocessing/README.md)
- [Entrenamiento y procesos](src/lambdaforge/training/README.es.md) · [English](src/lambdaforge/training/README.md)
- [Componentes neuronales](src/lambdaforge/nn/README.es.md) · [English](src/lambdaforge/nn/README.md)
- [Métricas](src/lambdaforge/metrics/README.es.md) · [English](src/lambdaforge/metrics/README.md)
- [Plugins instalados](src/lambdaforge/plugins/README.es.md) · [English](src/lambdaforge/plugins/README.md)
- [Tracking opcional de experimentos](src/lambdaforge/tracking/README.es.md) · [English](src/lambdaforge/tracking/README.md)
- [Ejemplo YAML completo](examples/experiment.yaml)
- [Ejemplo de preprocesado ejecutable](examples/preprocessing.yaml)
- [Ejemplo de preprocesado conciso](examples/preprocessing-simple.yaml)
- [Ejemplo de catálogo de clústeres](examples/lambdaforge.clusters.yaml) · [Ejemplo de catálogo de datos](examples/data-catalog.yaml)
- [Ejemplo de workflow](examples/workflow.yaml)
- [Ejemplo de HPO adaptativo](examples/adaptive-hpo.yaml)

Cada guía enlaza de vuelta aquí y a su traducción. Los docstrings de clase son la referencia más
precisa para los argumentos de cada constructor.

## 32. Hoja de ruta

La hoja de ruta vive aquí para que su estado no diverja en otro fichero. “Completado” significa API
pública, documentación y pruebas focalizadas; no significa incluir cada proveedor o método externo.

| Prioridad | Capacidad | Estado 0.5 |
|---:|---|---|
| 1 | Contrato de tarea genérica | Completado |
| 2 | Schema/configuración task independiente | Completado |
| 3 | Validación task | Completado |
| 4 | Planes inmutables | Completado |
| 5 | Fachada y CLI unificadas | Completado |
| 6 | Resultados y procedencia task | Completado |
| 7 | Artefactos tipados y hasheados | Completado |
| 8 | Preprocesado componible | Completado |
| 9 | Resume y shards deterministas | Completado |
| 10 | Dataset artifacts versionados | Completado |
| 11 | DAG de workflow | Completado: runner local acotado y nodos task/experiment |
| 12 | Composición de configuración | Completado: include/extends/merge/delete/ciclos |
| 13 | Interpolación segura y secretos | Completado con ocultación persistente en tasks |
| 14 | Procedencia y diff semántico | Completado |
| 15 | Scheduling CPU | Completado |
| 16 | Recursos/planes/packing/estimaciones | Completado |
| 17 | Contrato de backend | Completado |
| 18 | Adaptador SLURM/HPC | Completado en frontera explícita plan/envío |
| 19 | Fallos/retry/preemption | Completado: taxonomía, retry, modos y requeue |
| 20 | Inferencia/evaluación/ensemble/export | Completado |
| 21 | HPO | Completado: random/Optuna finito y optimización asíncrona multi-fidelidad por acciones, semillas adaptativas, admisión coste/VRAM, persistencia y BoTorch opcional |
| 22 | Caché distribuida | Completado con leases en filesystem compartido |
| 23 | Stores y referencias | Completado: local/compartido/S3-compatible |
| 24 | Registro y exports | Completado |
| 25 | Comparación e informes | Completado sin conclusiones generadas |
| 26 | Dashboard local | Completado como HTML estático de sólo lectura |
| 27 | Observabilidad estructurada | Completado: eventos/recursos/profiler/taxonomía OOM |
| 28 | Perfiles de reproducibilidad | Completado |
| 29 | Ergonomía CLI/IDE/ejemplos | Completado: init/explain/target/compose/diff y ejemplos probados |
| 30 | Adopción/gobernanza | Código/docs completos; la licencia debe elegirla legalmente el owner |
| 31 | AuthoringConfig corto -> MaterializedConfig estricto | Completado con Schema 1.0 e `inspect --resolved` |
| 32 | Inputs/outputs con nombre y preprocesado corto | Completado; APIs de ruta anteriores compatibles |
| 33 | Identidad lógica de datos y código | Completado con cuatro estrategias de datos y Git/distribución/versión explícita |
| 34 | Idempotencia explícita | Completado con reuse, `--force`, `--restart` y `--no-resume` |
| 35 | Plano de control multiclúster portable | Completado para transporte local/SSH y scheduler local/SLURM |
| 36 | Servicio de jobs persistente | Completado con list/status/logs/cancel/retry y JSON |
| 37 | Ubicación/réplica explícita de datos | Completado con catálogos, rechazo preventivo y rsync |
| 38 | Coordinador de workflow multiclúster | Diferido: planifica ubicación pero rehúsa ejecutar hasta garantizar transferencias y recuperación durable |
| 39 | Estabilización CI/wheel instalada | Completado: Windows fsync, refill dinámico, CSV atómico y smoke aislado |
| 40 | Workloads reales de preprocesado | Completado: secuencial, I/O threads, CPU spawn y GPU segura |
| 41 | Training sencillo y datasets de experimento | Completado: aliases, recursos, refs directas/anidadas e identidad portable |
| 42 | Entorno cluster managed/offline | Completado: wheels exactas, identidad, venv, bootstrap/doctor/wheelhouse |
| 43 | Jobs/resultados remotos | Completado: filtros/follow, sync ligero y fetch explícito |
| 44 | ResultService/MetricSeries | Completado: selectores, ambigüedad, compare y export |
| 45 | Plots reproducibles | Completado: learning/seeds/sweep/HPO/resources, PlotSpec y sidecar |
| 46 | Toolkit seguro de artifacts | Completado: NumPy/tablas, validación, geometría explícita y plugins |
| 47 | Debug/dataset inspection | Completado: N records aislados e informe DatasetArtifact |
| 48 | Workflow distribuido/placement automático | Diferido explícitamente después de 0.5.2 |

Las ampliaciones futuras deben responder a necesidades de investigación demostradas y conservar las
fronteras de la [arquitectura técnica](docs/ARCHITECTURE.md), no reabrir lo ya cerrado.

## 33. Historial de la hoja de ruta 0.2

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
