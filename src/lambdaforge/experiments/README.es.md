# Sistema de experimentos de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

Este paquete convierte un YAML de confianza en ejecuciones reproducibles por variante y semilla, las
planifica y reduce sus resultados en disco. No contiene lógica específica de modelos o datasets.

## Contenidos

- [Objetos principales](#objetos-principales)
- [Ciclo de vida](#ciclo-de-vida)
- [Reglas de expansión](#reglas-de-expansión)
- [Ejecución](#ejecución)
- [Finalización y reanudación](#finalización-y-reanudación)
- [Artefactos y agregación](#artefactos-y-agregación)
- [Carga](#carga)
- [Fronteras de extensión](#fronteras-de-extensión)

## Objetos principales

| Objeto | Responsabilidad |
|---|---|
| `Experiment` | Fachada pública para expandir, ejecutar, agregar y cargar. |
| `ExperimentConfig` | Carga YAML, rutas con puntos, validación y expansión. |
| `ObjectFactory` | Resolución recursiva de especificaciones `target` y `ref`. |
| `ExecutionConfig` | Validación de recursos y creación de slots GPU lógicos. |
| `ExperimentExecutor` | Selección de ejecución secuencial, paralela o DDP. |
| `ExperimentRunner` | Materialización y ejecución de una configuración y su resultado. |
| `ExperimentAggregator` | Lectura de ejecuciones y creación de estadísticas, CSV y gráficas. |
| `RunLoader` | Localización de ejecuciones y reconstrucción de modelos. |

Clases auxiliares como `ExperimentWorker`, `StdIOCapture`, `TeeStream`, `CheckpointChoice` y los
enums de estado viven también cada una en su módulo.

## Ciclo de vida

```text
YAML → ExperimentConfig → configuraciones por variante/semilla
     → ExecutionConfig → slots de procesos/dispositivos
     → ExperimentRunner → config, log, métricas, checkpoints, resultado
     → ExperimentAggregator → tablas y gráficas entre semillas
     → RunLoader → modelo reconstruido
```

Usa el objeto de alto nivel salvo al escribir una integración:

```python
from lambdaforge import Experiment

experiment = Experiment.from_yaml("experiment.yaml")
for run in experiment.expand():
    print(run["experiment"]["variant"], run["experiment"]["seed"])
results = experiment.run(dry_run=True)
```

## Reglas de expansión

`experiment.seeds` acepta escalar o lista. `sweep.grid` relaciona rutas con puntos con listas no
vacías y forma su producto cartesiano. `sweep.include_base` decide si se incluye la configuración
original. Cada elemento de `sweep.ablations` añade overrides con puntos y nombre propio.

Se usan copias profundas: una ejecución no puede mutar otra. El nombre no puede estar vacío y las
identidades `(variant, seed)` finales han de ser únicas. `lambdaforge inspect` imprime las
configuraciones concretas sin ejecutar los objetos importados.

## Ejecución

`sequential` permanece en el llamador. `parallel` planifica cada ejecución independiente como proceso
`spawn` en slots fijos de una GPU. `ddp` asigna cada ejecución a un grupo de `devices_per_job` GPU y
configura Lightning para DDP. Los overrides CLI tienen precedencia sobre YAML y YAML sobre defaults.

El ejecutor usa objetos worker serializables y el método `spawn`. Los índices GPU son lógicos
respecto a `CUDA_VISIBLE_DEVICES` del padre; los límites de CPU/hilos/workers se aplican a cada
ejecución sin mutar el entorno padre.

Consulta la [guía de procesos](../training/README.es.md) para garantías y límites de apagado.

## Finalización y reanudación

Los estados legibles por máquina son `ok`, `failed`, `dry_run`, `interrupted` y `unknown`.

Una ejecución solo está completa cuando:

1. `result.json` tiene estado `ok`;
2. existe el checkpoint seleccionado si la política lo exige; y
3. existen dentro del directorio todas las rutas de `experiment.required_artifacts`.

Con `rerun_completed: false` se omiten las completas. Con `resume: true`, una incompleta usa el
último checkpoint válido si existe. Los fallos producen un resultado terminal y la suite puede
relanzarse sin descartar semillas correctas. Los artefactos requeridos son rutas relativas definidas
por el proyecto.

## Artefactos y agregación

El directorio contiene `config.yaml` materializado, `hparams.json`, `train.log`, `metrics.csv`,
checkpoints, `result.json` y artefactos propios. Las rutas se derivan del nombre, slug de variante y
semilla.

`ExperimentAggregator.write` reconstruye informes desde disco: resúmenes terminales y por época,
CSV anchos/largos, estadísticas de semillas, pruebas direccionales por pares, q-valores
Benjamini-Hochberg y gráficas opcionales. Un fallo de Matplotlib queda registrado sin perder las
tablas numéricas. `lambdaforge aggregate --no-plots` sirve para entornos mínimos sin interfaz.

Las estadísticas son exploratorias. Se informan tamaños muestrales y variantes incompletas para
hacer visibles las semillas ausentes; las decisiones inferenciales corresponden a cada estudio.

## Carga

```python
experiment = Experiment.from_yaml("experiment.yaml")
model = experiment.load_model(seed=7, variant="base", which="auto")
```

`CheckpointChoice` ofrece `best`, `last` y `auto`. `RunLoader` valida la ejecución, importa el modelo
desde su especificación materializada, carga el estado directo o elimina el prefijo Lightning
`model.`, y devuelve el modelo en evaluación.

## Fronteras de extensión

- Configura `data.datamodule.target`, `task.target` o `runner.target` propios en vez de bifurcar el
  motor de experimentos.
- Modelos, pérdidas y `train_metrics`, `val_metrics` y `test_metrics` aceptan especificaciones
  `target`/`ref` construidas recursivamente. La clave retrocompatible `metrics` completa las etapas
  no indicadas.
- `callbacks` de nivel superior y loggers, collators y otros objetos anidados usan la misma sintaxis.
- Los monitores de checkpoint y parada temprana y sus modos `min`/`max` son ajustes explícitos del
  trainer; si se omiten se usa la primera métrica de validación y su dirección declarada.
- Un runner propio debe conservar métodos `fit` y `test` compatibles con `ExperimentRunner`.
- Trata YAML como código de confianza: los targets pueden ejecutar Python arbitrario.
- Importa clases públicas desde `lambdaforge.experiments`; los archivos son detalles internos.

Las clases del ciclo de vida permanecen juntas deliberadamente: a diferencia de métricas y
callbacks, sus contratos están muy acoplados y separarlas crearía varios paquetes diminutos sin un
propósito público independiente. Esta frontera se debería revisar cuando aparezca una familia real
de backends o almacenamiento.
