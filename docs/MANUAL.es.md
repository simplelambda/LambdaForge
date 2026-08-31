# Manual de LambdaForge 0.13

[English](MANUAL.md) · Español

## Índice

1. [Modelo mental](#1-modelo-mental)
2. [Instalación y estructura](#2-instalación-y-estructura)
3. [API de Work](#3-api-de-work)
4. [Ficheros gestionados, cache, map, outputs y herramientas](#4-ficheros-gestionados-cache-map-outputs-y-herramientas)
5. [Referencia YAML](#5-referencia-yaml)
6. [Ficheros y datasets](#6-ficheros-y-datasets)
7. [Secuencia, paralelismo, seeds y búsqueda](#7-secuencia-paralelismo-seeds-y-búsqueda)
8. [Ejecución, identidad y reutilización](#8-ejecución-identidad-y-reutilización)
9. [Resultados y metadata](#9-resultados-y-metadata)
10. [Clústeres y Jobs](#10-clústeres-y-jobs)
11. [Limpieza y seguridad](#11-limpieza-y-seguridad)
12. [Referencia CLI](#12-referencia-cli)
13. [Clustering](#13-clustering)
14. [Componentes neuronales reutilizables](#14-componentes-neuronales-reutilizables)
15. [Arquitectura y extensiones](#15-arquitectura-y-extensiones)

## 1. Modelo mental

LambdaForge ejecuta clases Python y posee la infraestructura que las rodea:

```text
YAML Work -> WorkConfig -> plan -> scheduler/Job -> WorkRunner -> Work.run()
```

El investigador implementa una subclase de `lambdaforge.Work` y su método `run()`. YAML elige la
clase, argumentos normales, recursos y repeticiones. LambdaForge resuelve entradas externas,
calcula identidad, enlaza servicios, captura salidas y conserva un resultado inspeccionable. Una
función o una clase que no herede `Work` no puede ser target de YAML.

La jerarquía durable es:

- Work: operación o estudio con nombre humano;
- Execution: una invocación en un destino;
- Run: un miembro científico de seed/variante;
- Attempt: un intento de completar ese Run;
- Job: el proceso o trabajo del scheduler que ejecuta el Attempt.

## 2. Instalación y estructura

Framework y proyecto consumidor son paquetes independientes instalados en el entorno del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge
python -m pip install -e .
python -m pip check
```

Para desarrollar el framework se sustituye la primera instalación por
`python -m pip install -e /ruta/absoluta/LambdaForge`. No se comparte el `.venv` del framework, no
se modifica `PYTHONPATH` en producción ni se copia el código fuente dentro del consumidor.
`lf init DIRECTORIO` crea `pyproject.toml`, un paquete `src/`, YAML y exclusiones seguras.

```text
proyecto/
  pyproject.toml
  src/mi_proyecto/work.py
  experiments/estudio.yaml
  data/entrada-pequeña.json
  .lambdaforge/                 # estado gestionado; ignorado por Git
```

`lf --help`, `lf help`, `lf COMANDO --help` y `lf help COMANDO SUBCOMANDO` son rutas de ayuda
equivalentes que terminan con éxito. Para el primer destino remoto, `lf clusters setup` ofrece un
asistente interactivo explicado; `lf clusters modify [NOMBRE]` edita un perfil. Ambos llaman solo
a los mismos subcomandos nativos usados por scripts y nunca ponen passwords en argv o YAML. Cada
pregunta acepta `0`, `q`, `quit` o `exit`; una respuesta sin confirmar no se aplica.

## 3. API de Work

Una subclase normalmente no define constructor y tiene una única entrada científica: `run()`.
Todas las propiedades de runtime fallan si se usan fuera de una ejecución gestionada.

| API | Vida | Responsabilidad |
|---|---|---|
| `run(**parameters)` | una vez por Attempt | cálculo científico y resultado JSON primario |
| `name`, `config` | inmutable | nombre, clase, parámetros y recursos normalizados |
| `inputs` | inmutable | procedencia y rutas resueltas de file/dataset tipados |
| `outputs` | Attempt | valores, ficheros/directorios, artefactos y datasets |
| `metrics` | append-only | historia escalar y valores finales |
| `checkpoints` | Run | estado explícito para reanudar |
| `cache` | reconstruible | valores simples y ficheros reutilizables ligados a identidad |
| `tools` | ejecución | programas externos y su procedencia |
| `progress` | snapshot | avance vivo `completed/total/message` |
| `log(...)` | stream | diagnóstico humano con fecha y flush |
| `resources`, `seed`, `trial` | inmutable | reserva y miembro del estudio |
| `run_dir` | Attempt | raíz durable avanzada |
| `temp_dir` | Attempt | temporales eliminados al finalizar |
| `source_dir` | inmutable | contexto del paquete consumidor |
| `resuming` | inmutable | existía checkpoint compatible al comenzar |
| `map(...)` | Job | concurrencia ordenada sin persistencia |
| `resume_map(...)` | Job | reanudación explícita por elemento y clave estable |

`outputs.value` acepta una vez valores JSON. `outputs.file/directory` es la vía recomendada para
artefactos nuevos; `outputs.artifact` importa de forma avanzada una ruta ya creada. `metrics.log`
acepta escalares finitos y `step/split` opcionales. `print()`, stdout, stderr y `logging` se capturan
en los logs del Job y en `work.log`. `self.log()` añade nivel y fecha; `progress.update` representa
avance, mientras `metrics.log` conserva evidencia científica.

## 4. Ficheros gestionados, cache, map, outputs y herramientas

### 4.1 Elegir almacenamiento

| Necesidad | Servicio | Sobrevive retry | Lo elimina `lf clean` | Resultado publicado |
|---|---|---:|---:|---:|
| bytes/texto/JSON pequeño reconstruible | `cache.put/get` | se reutiliza | sí | no |
| fichero descargable/calculable | `cache.file/fetch` | se reutiliza | sí | no |
| estado secuencial para reanudar | `checkpoints.file/save_json` | sí | no | evidencia de resume |
| fichero/árbol científico final | `outputs.file/directory` | pertenece al Attempt | solo duplicado publicado verificado | sí |
| intermedio desechable | `temp_dir` | no | automático | no |

`cache.path`, `checkpoints.path` y escrituras directas bajo `run_dir` son escapes avanzados. El
usuario que los elige también asume validación, atomicidad y registro.

### 4.2 ManagedFile y cache

El caso normal no exige rutas ni callbacks:

```python
self.cache.put("resumen", {"aceptadas": 1842, "schema": 3})
resumen = self.cache.get("resumen")
fallback = self.cache.get("ausente", {"aceptadas": 0})
```

`put(clave, contenido)` admite `bytes`, texto UTF-8 o JSON estricto y reemplaza atómicamente el
valor. `get(clave, default=None)` devuelve el tipo soportado original o el default si no existe una
entrada íntegra. No se admite pickle ni serialización implícita de objetos. El perfil de storage
elige la ubicación física; el Work normal no necesita hacerlo. `cache.path` queda como escape
avanzado explícito.

```bash
lf clusters add gpu --host HOST --user USER --workspace /remote/work \
  --cache-root /scratch/USER/lambdaforge-cache
```

La raíz elegida se propaga a cada proceso Work, también a Jobs directos detached; no depende del
estado del shell. La ejecución local usa por defecto `.lambdaforge/cache` del proyecto.

Cuando una librería necesita una ruta o la construcción es pesada se usa `ManagedFile`:

```python
estructura = self.cache.file(
    f"estructuras/{identificador}.cif",
    build=lambda destino: crear_estructura(identificador, destino),
    validate=lambda fichero: fichero.size_bytes > 0 and estructura_valida(str(fichero)),
)
```

`build(destino: Path)` escribe un temporal real; su retorno se ignora. En un hit se comprueban
registro, SHA-256, tamaño y validador semántico. En un miss o entrada inválida se adquiere un lock
exclusivo entre procesos para esa clave, se vuelve a comprobar, se construye junto al destino, se
valida, se hace `fsync`, se promociona atómicamente y se actualiza el registro. Un fallo no publica
bytes parciales. Las claves son relativas; se rechazan rutas absolutas, `..` y symlinks.

`ManagedFile` es de solo lectura y compatible con `os.PathLike`: ofrece `str`, `path` como escape,
`open/read_text/read_bytes`, `exists` y metadata inmutable `key`, `sha256`, `size_bytes` y `scope`.
Los checkpoints guardan clave y evidencia de contenido, nunca la ruta física.

```python
limite = self.cache.rate_limit("archivo", requests_per_second=4)
fichero = self.cache.fetch(
    url,
    key=f"archivo/{identificador}.json",
    retries=5,
    retry_backoff=0.5,
    timeout=30,
    decompress="gzip",
    validate=registro_valido,
    rate_limit=limite,
)
```

`fetch` hace un GET HTTP(S) cacheable, no sustituye a un cliente HTTP general. Añade timeout,
reintentos después del intento inicial, backoff exponencial y gzip opcional. Los cortes de conexión,
lecturas chunked incompletas —también durante gzip—, estados 408/425/429 y 5xx se reintentan;
errores permanentes como 401 o 404 fallan de inmediato. Cada intento pasa por el limitador y empieza
con un temporal privado vacío. Al agotarlos, el error conserva URL, número de intentos y causa final.
Solo se publica atómicamente un fichero completo y validado, nunca bytes parciales. El limitador es seguro
entre threads de esa instancia de Work; no coordina Jobs distribuidos. `lf clean` muestra cada cache
de Work y `--apply` lo elimina solo cuando obtiene el lock exclusivo; un Work activo mantiene una
lease compartida.

`DatasetCache` sigue siendo el cache especializado de muestras serializadas. Work cache reutiliza
el `CrossProcessFileLock`, convenciones de huella/atomicidad y ownership, pero no usa su sobre opaco
porque un fichero científico debe seguir siendo path-like.

### 4.3 Map simple y reanudable

La operación habitual es directa:

```python
resultados = self.map(filas, procesar, workers=8, executor="thread", retries=2)
```

`map(items, function, *, workers=1, executor="thread", name=None, retries=0,
retry_backoff=0.5)` conserva el orden de entrada, limita concurrencia, informa progreso y no crea
cache ni checkpoints. `thread` es apropiado para I/O; `process` exige callback y argumentos
serializables mediante spawn.

Solo un cálculo largo que deba reanudar elementos elige la variante explícita:

```python
resultados = self.resume_map(
    filas,
    procesar,
    key="record_id",
    workers=16,
    executor="thread",
    resume=True,
    name="features",
    validate=resultado_restaurado_valido,
    retries=2,
    retry_backoff=0.5,
)
```

La firma completa es `resume_map(items, function, *, key, workers=1, executor="thread",
resume=True, name=None, validate=None, retries=0, retry_backoff=0.5)`. La clave es un campo de mapping, atributo
de dataclass/objeto o callable explícito; debe ser única y estable. El resultado conserva el orden
de entrada. `map(..., key=...)` conserva esta semántica por compatibilidad, pero el nombre
`resume_map` hace visible el efecto persistente y es el recomendado para código nuevo.

Cada elemento se guarda como JSON estricto ampliado solo con referencias `ManagedFile` de cache o
checkpoint. Para cada fichero devuelto, anidado en el resultado o tocado mediante `self.cache` dentro de callback
secuencial/thread se conservan scope, key, SHA y tamaño. Al restaurar se verifican primero las
dependencias. Si falta una, está corrupta o `validate` devuelve false, solo ese elemento vuelve a
pending. Una excepción del validador se informa porque puede ser un bug científico. Los reintentos
son por elemento y su agotamiento sigue siendo fail-fast. En executor `process`, el callback debe
ser serializable por spawn y las dependencias deben estar en item/resultado: el contexto de runtime
no cruza procesos.

### 4.4 Outputs y checkpoints

```python
informe = self.outputs.file(
    "informe",
    filename="informe.json",
    role="report",
    media_type="application/json",
    publish_to="resultados/informe.json",
)
informe.write_json(resumen)

figuras = self.outputs.directory("figuras", role="visualization")
renderizar(Path(figuras))
```

Un fichero gestionado tiene `write_text`, `write_bytes`, `write_json` y `build`, todos con promoción
atómica. Un directorio existe desde su declaración, permite hijos seguros con `/` y solo se hashea
al finalizar. Después de un `run()` correcto se validan tipo, containment y symlinks de todo el
conjunto y se registra automáticamente. Si una declaración falta o es insegura, la finalización
falla y ninguna declaración gestionada del conjunto entra en el resultado. No se llama además a
`outputs.artifact`.

`publish_to` es opcional para ficheros y directorios. Una ruta relativa parte del directorio que
contiene el YAML original; una absoluta se usa explícitamente. En remoto la publicación relativa
requiere el `project_root` del clúster y usa el directorio YAML equivalente bajo ese mirror. Nunca
resuelve bajo el hash del bundle/Job. Sin mirror se debe usar una ruta remota absoluta persistente;
LambdaForge rechaza una relativa en vez de publicar silenciosamente en almacenamiento interno
desechable.

LambdaForge valida y hashea primero el artefacto gestionado y después publica una copia atómica por
destino. Reutiliza contenido idéntico y rechaza contenido distinto existente salvo
`overwrite=True`. El resultado registra `published_to`, SHA-256 y tamaño. Tras persistir toda la
Execution, LambdaForge vuelve a hashear el destino y elimina por defecto los bytes redundantes del
Attempt. `retain_internal=True` conserva deliberadamente una segunda copia. Los outputs correctos
sin `publish_to` permanecen en el Attempt porque no existe otra copia. Los Attempts fallidos o
interrumpidos eliminan `artifacts/` parciales, pero conservan logs, métricas, resultado,
procedencia y checkpoints. Las escrituras directas en `run_dir` y rutas importadas con
`outputs.artifact` son escapes avanzados: la limpieza no adivina que bytes arbitrarios son
desechables. No se copian árboles grandes al controlador implícitamente. También se rechazan symlinks, cambios de
tipo y destinos de directorio que contienen su fuente o están contenidos en ella; `overwrite=True`
nunca autoriza sustituir la raíz del proyecto o del Attempt.

```python
indice = self.checkpoints.file(
    "leakage/mmseqs.tsv",
    build=lambda destino: construir_indice(destino),
    validate=indice_valido,
)
```

El checkpoint usa construcción temporal, registro de integridad y validador como el cache, pero
pertenece al Run y nunca es cache reconstruible. Retry reutiliza uno válido y reconstruye uno
ausente/corrupto/inválido. `save_json/load_json/exists` es la vía para estado pequeño. `--restart`
elimina el árbol compatible antes del nuevo Attempt.

### 4.5 Herramientas externas

```python
mmseqs = self.tools.require("mmseqs", version_args=["version"])
completado = self.tools.run(
    [mmseqs, "easy-search", str(consulta), str(base), str(salida)],
    name="MMseqs2",
    threads=self.resources.cpu,
    cwd=self.temp_dir,
    env={"PROJECT_MODE": "strict"},
    timeout=3600,
)
```

`require(executable, version_args=None, version_timeout=10)` resuelve `PATH` y solo ejecuta el probe
pedido. Devuelve un `Tool` path-like. `run(command, *, name=None, threads=None, cwd=None, env=None,
timeout=None, check=True)` exige argv y nunca usa shell. Emite stdout/stderr línea a línea, conserva
una cola acotada en `ToolResult`, registra duración/status y lanza `ToolExecutionError` si procede.
`threads` configura OMP/MKL/OpenBLAS/NumExpr solo en el hijo. La herramienta y su versión explícita
se registran una vez en `environment.json`.

## 5. Referencia YAML

Los campos superiores son `name`, `run`, `with`, `resources`, `seeds`, `search`, `objective` y
`steps`. Un documento define exactamente `run` o `steps`; los campos desconocidos fallan.

```yaml
name: evaluar
run: proyecto.work.Evaluar
with:
  threshold: 0.5
resources:
  cpu: 4
  memory: 8GiB
  gpu: 1
  gpu_memory: 12GiB
  time: 4h
  storage: 20GiB
  processes: 2
```

Los recursos son la reserva absoluta. `processes` no supera CPU. Se admiten unidades decimales y
binarias de bytes y `s/m/h/d` para tiempo. `lf validate` importa cada Work y comprueba herencia,
constructor, firma, tipos evidentes, recursos, entradas, expansión y referencias sin enviar nada.
`lf explain` muestra docstring, tipos, valores y defaults.

## 6. Ficheros y datasets

```yaml
with:
  manifiesto: {file: ../data/manifest.json}
  corpus: {dataset: research-corpus@3}
```

Un `file` o directorio se resuelve respecto al YAML, se comprueba/hashea y se pasa como `Path`. En
remoto, hasta el límite inline por defecto de 10 MiB se copia al bundle inmutable. Una ruta mayor
que pertenezca al proyecto sigue el contrato de mirror de la sección 10: se conserva su nombre
relativo al proyecto, el worker recibe la ruta absoluta remota equivalente y se comprueban tipo,
bytes y SHA-256 antes del scheduler y otra vez en el worker. No se transfiere nada grande
implícitamente. Una ruta grande externa al proyecto se rechaza: debe ser un dataset gestionado o
formar parte de un layout de proyecto/datos explícito y revisable.

El marcador `file` solo describe una entrada que debe existir antes del envío. Un resultado nuevo
se crea en Python con `outputs.file/directory/value/dataset`; no hay otro esquema YAML de outputs.
El artefacto gestionado es el default seguro y `publish_to` solo se añade cuando una herramienta o
persona necesita además una ruta convencional, teniendo presente que local y remoto son sistemas
de ficheros físicos distintos.

`dataset` resuelve una versión exacta e inmutable, content ID y placement válida. Se publica desde
Python con `self.outputs.dataset(name=..., version=..., members=...)`. Los miembros se escriben en
JSONL, se copian y hashean assets, se valida el índice, se calcula identidad independiente de ruta,
se promociona staging atómicamente y se registra placement. Reutilizar nombre/versión con otro
contenido se rechaza; DatasetArtifact v1 sigue siendo legible.

## 7. Secuencia, paralelismo, seeds y búsqueda

```yaml
name: comparacion
steps:
  - name: preparar
    run: proyecto.Preparar
  - parallel:
      - name: a
        run: proyecto.EntrenarA
        with: {data: {from: preparar.dataset}}
      - name: b
        run: proyecto.EntrenarB
        with: {data: {from: preparar.dataset}}
  - name: comparar
    run: proyecto.Comparar
```

Cada nivel espera al anterior. Cada miembro paralelo usa un proceso spawn aislado. Un estudio
exhaustivo es serial dentro de su reserva; uno adaptativo gestiona Runs hijos independientes dentro
de esa reserva. Una referencia requiere un único Run
productor. Ramas, condiciones y bucles complejos pertenecen a Python.

`seeds` crea Runs independientes. Con `objective`, omitir `strategy` activa por defecto todas las
optimizaciones seguras: inicio Sobol scrambled, propuestas dependientes de resultados, carrera
probabilística de seeds, pruning de curvas, convergencia, recuperación acotada y confirmación con
seeds nuevas. El pool acotado por `trials` es planificación interna; solo los Trials realmente
propuestos aparecen en `lf top`. Tras `startup_trials`, `sampler: auto` prefiere GP mixto qLogNEI de
BoTorch si está instalado `lambdaforge[adaptive-hpo]` y hay evidencia suficiente; ante dependencia
ausente o inestabilidad numérica usa el surrogate mixto k-NN determinista. Es un modelo de decisión
realmente conjunto: números, categorías e indicadores de activación condicional comparten un mismo
vector, por lo que el posterior puede depender de interacciones. Si las seeds repetidas producen
error estándar, se pasa como ruido observado; qLogNEI considera ese ruido y los miembros pendientes
en su totalidad. La familia log-EI se usa por su mayor estabilidad numérica
([guía de adquisición de BoTorch](https://botorch.org/docs/optimization)).

El objetivo admite restricciones de resultado explícitas:

```yaml
objective:
  metric: val_auprc
  mode: max
  constraints:
    val_accuracy: {min: 0.55}
    val_kappa: {min: 0.05}
```

Cada restricción acepta `min`, `max` o ambos. LambdaForge toma su valor en la época exacta donde el
objetivo primario fue mejor y después promedia ese valor alineado entre seeds terminadas. Una
métrica ausente o una media que viole el límite hace al candidato no factible: permanece auditable,
pero no entra en carrera de seeds, observaciones del surrogate ni selección final. Así un checkpoint
afortunado no oculta un modelo que incumple un criterio científico declarado. LambdaForge nunca
infiere restricciones de otras métricas, sus nombres o direcciones: hacerlo cambiaría en silencio
la pregunta científica. Varios objetivos requieren un problema Pareto explícito y no se simulan
mediante pesos ocultos.

Por defecto se inicia con hasta tres seeds declaradas por candidato y se generan tres seeds de
confirmación deterministas y disjuntas. Puede cambiarse cada valor y `confirmation_seeds: []`
desactiva expresamente la confirmación. El orden compartido permite diferencias pareadas

$$
d_s=Y(i,s)-Y(i^\star,s).
$$

Se añade evidencia mientras

$$
P(\mu_i\geq\mu_{i^\star}-\epsilon\mid D)\geq\delta,
$$

donde `equivalence_margin` es \(\epsilon\) y `seed_probability_threshold` es \(\delta\). Los
candidatos dominados dejan de recibir seeds y la reducción de incertidumbre se divide por el coste
temporal observado. La selección final usa una cota conservadora o, preferiblemente, la media de
`confirmation_seeds` nuevas sobre un top-K congelado. `max_runs`, `max_time`,
`convergence_patience` y `min_improvement` limitan gasto.

Cada Run de seed conserva dos valores deliberadamente distintos. `current` es la observación del
último step y se usa para proyectar curvas comparables y decidir pruning. `best` es el mínimo/máximo
de toda la curva terminada y es el valor de checkpoint usado por la carrera de seeds, el surrogate
y el ranking final. Así una época tardía sobreajustada no infravalora un Run terminado, pero un
punto temprano prometedor tampoco rescata un Run parcial podado. El valor HPO de un Trial es la
media de los mejores valores por seed, nunca la seed más afortunada. Las seeds frescas de
confirmación y la cota conservadora reducen el sesgo de validar repetidamente. El suelo inicial por
defecto es de hasta tres seeds declaradas por Trial propuesto: permite estimar variabilidad sin
gastar todas las seeds en una configuración claramente dominada.

El diario append-only `hpo-control/decisions.jsonl` explica inicialización, surrogate/fallback,
`START_NEW`, `ADD_SEED`, `RESUME`, convergencia, confirmación y final. `state.json` conserva el
snapshot compacto; `summary.adaptive_controller` enlaza ambos sin copiar modelos ni checkpoints.

El bucle de adquisición es asíncrono y acotado. Una observación terminal puede rellenar un slot
libre mientras otros Runs del lote siguen activos. El controlador propone como máximo
`max(1, min(2, paralelismo // 3))` candidatos de anticipación por oleada, condiciona la adquisición
bayesiana en todos los candidatos pendientes y registra `START_NEW` con
`reason=bounded-async-lookahead`. Esto oculta stragglers largos sin construir una cola grande desde
posteriors obsoletos. No interrumpe un Run sano porque cambie el ranking: el pruning cooperativo y
las decisiones de fidelidad siguen siendo los únicos mecanismos de cancelación científica.

### 7.1 Sweeps exactos sin HPO

`strategy: exhaustive` ejecuta literalmente cada combinación finita y cada seed:

```yaml
seeds: [7, 17]
search:
  strategy: exhaustive
  optimizer: {values: [adamw, sgd]}
  momentum: {values: [0.8, 0.9], when: {optimizer: sgd}}
```

Son exactamente seis Runs: una variante AdamW y dos SGD por seed. No hay surrogate, pruning,
asignación adaptativa de seeds ni confirmación. Un `range` continuo no puede ser exhaustivo y se
rechaza, igual que `trials`; hay que discretizar con `values` o usar HPO adaptativo. Puede declararse
`objective` para resumir el sweep completo sin cambiar qué Runs se ejecutan.

### 7.2 Entrenos independientes por GPU

`resources` es la reserva externa fija. `resources.gpu` indica cuántas GPUs reserva el estudio y
`search.runs_per_gpu` cuántos procesos de entreno independientes pueden compartir cada una:

```yaml
seeds: [4, 7, 32, 54, 65, 94, 109, 124]
search:
  strategy: adaptive
  trials: 40
  runs_per_gpu: 4
  startup_trials: 10
  min_seeds: 1
  reduction_factor: 2
  failure_retries: 1
  early_stopping: {enabled: true, min_step: 5, confirmations: 2}
  learning_rate: {range: [0.00001, 0.003], scale: log}
  hidden_dim: {values: [64, 128, 256]}
objective: {metric: val_auprc, mode: max}
resources:
  gpu: 2
  gpu_memory: 16GiB
  cpu: 16
  memory: 32GiB
  time: 24h
```

El máximo es \(2\times4=8\) entrenos concurrentes. Seis GPUs con `runs_per_gpu: 2` dan 12; dos con
`runs_per_gpu: 1` dan 2. Un packing mayor que uno exige `gpu_memory`, interpretado como umbral de
admisión por Run. Antes de lanzar cada hijo individual se comprueba

$$
\texttt{gpu\_memory}
\leq \text{memoria de dispositivo libre en ese momento}.
$$

`runs_per_gpu` solo es el máximo por dispositivo. LambdaForge sondea todas las GPU asignadas y
ocupa únicamente slots que cumplen la condición; los Runs sin hueco permanecen en cola y se vuelve
a sondear. Los lanzamientos en una misma GPU se separan cinco segundos para que el nuevo proceso
materialice su asignación antes de observar otra vez. El controlador fija además un presupuesto
conservador de memoria libre al iniciar una oleada y contabiliza el umbral completo de cada Run
admitido aunque CUDA aún no lo haya asignado; solo lo recalcula cuando esa GPU no conserva Runs de
LambdaForge activos. Cada Run admitido tiene un proceso spawn nuevo con un único worker, que se
cierra al recibir resultado o error y libera el contexto CUDA completo. Un pool ocioso persistente
retendría VRAM y podría bloquear la cola. Una GPU llena o pequeña se omite mientras las demás continúan. Solo un umbral
mayor que la VRAM total de todas las GPU asignadas es imposible y falla como configuración. Las
consultas de memoria se ejecutan en un hijo efímero que termina tras devolver JSON, por lo que el
controlador HPO no aparece como un proceso CUDA ocioso por GPU ni consume un slot de
`runs_per_gpu`. Las esperas/admisiones se escriben inmediatamente en el log del Work.

Es admisión preventiva, no una promesa frente a procesos externos que reserven después ni un límite
duro dentro del código consumidor: debe declararse un pico conservador. `max_parallel` reduce el
máximo GPU global y limita estudios solo CPU.

El aislamiento sigue el límite de Run. Tanto en CPU como GPU cada Run adaptativo usa un proceso
nuevo de un worker: matar uno no rompe un pool compartido ni cancela candidatos ajenos. Un worker
perdido antes de devolver resultado y una OOM/asignación CUDA se reintentan como Attempt nuevo hasta
`failure_retries` (1 por defecto, rango 0–3), conservando checkpoints compatibles bajo el mismo
Run. Si el impedimento se repite queda terminal; las excepciones normales de datos o código no se
repiten a ciegas. Los demás Runs activos o pendientes continúan. El Work exterior sigue quedando
fallido si un Run agota recuperación, para no ocultar evidencia científica incompleta.

### 7.3 Continuación multi-fidelidad

`search.fidelity` es opcional porque solo el Work consumidor sabe si su presupuesto es acumulativo
y reanudable; LambdaForge nunca lo infiere de un parámetro llamado `epochs`:

```yaml
search:
  trials: 40
  fidelity: {min: 5, max: 100, reduction_factor: 3}
  learning_rate: {range: [0.00001, 0.003], scale: log}
objective: {metric: val_auprc, mode: max}
```

El primer Attempt recibe `self.fidelity.current == 0` y `target == 5`. Si se promociona conserva la
misma identidad científica y raíz de checkpoints, con targets acumulativos 15, 45 y 100. El Work
debe avanzar solo de `current` a `target`, guardar estado con `self.checkpoints` y registrar la
métrica; no debe repetir desde cero todo el target. La confirmación siempre usa `maximum`.
`LightningRunner` limita `max_epochs`, guarda un `last.ckpt` gestionado y lo entrega al Attempt
siguiente; un trainer propio debe implementar expresamente el mismo contrato. Sin `fidelity` se
mantiene el entreno normal a presupuesto completo.

### 7.4 Contrato de early stopping

El stop es cooperativo. Un Work registra `self.metrics.log("val_auprc", valor, step=epoch)` y
consulta `self.stop_requested` en un límite seguro, guardando checkpoint antes de retornar. El
controlador compara Runs a un step común tras `early_stopping.min_step`. Una proyección lineal local
acotada conserva incertidumbre residual y solicita parar solo cuando la probabilidad de seguir a
`equivalence_margin` del incumbent cae bajo `seed_probability_threshold`; no elimina una fracción
fija. Por defecto la condición debe mantenerse en dos steps comunes distintos
(`confirmations: 2`): sondear repetidamente la misma época no suma evidencia. Usa
`confirmations: 1` solo para una política deliberadamente agresiva. `LightningRunner` enlaza
`callback_metrics[objective]` y para en límites de batch. Un trainer
propio usa la API pública anterior. Si solo hay métrica final se adaptan las seeds, pero no puede
detenerse de forma segura el entreno actual.

### 7.5 Observabilidad viva del estudio

El Job externo del scheduler delimita reserva y cancelación, pero no obliga a mezclar el significado
científico. LambdaForge mantiene junto al Job un índice compacto del estudio y un registro pequeño
por Run interno. Los `work.log`, `metrics.jsonl`, `training-metrics.jsonl` y `result.json` ya
existentes siguen siendo la autoridad; el índice los referencia y resume solo estado y últimos
escalares. Nunca copia checkpoints de modelos, directorios de output ni bytes de artefactos.

`lf top` presenta esta jerarquía:

```text
Work
  └─ Trial 17: {dropout: 0.21, hidden_dim: 128, ...}
       ├─ Seed 4: ejecutando → curvas, tiempos y log vivo aislado
       └─ Seed 7: terminada  → curvas, tiempos y log aislado
```

La vista separa candidatos propuestos del presupuesto planificado y muestra Runs
activos/terminados/fallidos/podados y estado de promoción. La tabla separa mejor objective, su
seed/época y objective actual; HPO usa aparte la media de los mejores checkpoints de las seeds
terminadas. Debajo aparece cada parámetro del Trial seleccionado. La tabla de seeds incluye última
época, mejor época, mejor objective, objective actual e índice GPU, y termina con un preview acotado
de métricas; el panel completo queda a un Enter y se reservan al menos tres filas de seed con una
altura de terminal normal. `pruned` indica que el Run aceptó una solicitud probabilística terminal;
no es un fallo, no queda pausado ni se reanuda después, y se muestra su motivo persistido. Un Trial
con solo seeds podadas queda `pruned`, no `failed`. Sus curvas parciales se conservan como evidencia
censurada, pero no entran como valores exactos en medias de seeds completas, ajuste del surrogate
ni estadísticas marginales del objective: tratar su mejor punto parcial como presupuesto completo
sesgaría la búsqueda. LambdaForge sí conserva la información: el modelo de lectura HPO muestra la
tasa terminal de poda por región y un Trial solo podado añade una penalización suave de vecindad al
seleccionar propuestas. Es evidencia negativa censurada, no un objetivo escalar inventado. El
detalle JSON conserva además el token heredado exacto para grants UUID/MIG.

Pulsa `i` desde candidatos o Trial de un estudio adaptativo para abrir la consola de evidencia HPO.
Por cada hiperparámetro detectado calcula un diagnóstico acotado por candidato: dirección de rango,
contraste estandarizado bajo/alto y posible umbral para números, o medias y cobertura para
categorías. Etiqueta evidencia baja/media/alta con suelos de muestra conservadores y un score de
asociación, y sugiere la
comparación que más reduciría ambigüedad. La cabecera separa la última acción real del controlador.
También separa observaciones comparables, provisionales y podas censuradas. Las asociaciones
marginales pueden estar confundidas por parámetros correlacionados, activación
condicional, distinta cantidad de seeds o fidelidad parcial, mientras el sampler opera en el
espacio mixto multivariable. Por eso dice «parece asociado», nunca causal ni garantizado. Métodos
de importancia de interacciones como functional ANOVA pueden resumir un surrogate maduro
([artículo fANOVA](https://proceedings.mlr.press/v32/hutter14.html)), pero aplicarlos a los primeros
Trials produciría falsa precisión inestable. Por eso las explicaciones tempranas siguen siendo
marginales y auditables mientras el controlador GP/k-NN real sí permanece multivariable.
Enter/derecha sobre el parámetro seleccionado abre una curva numérica agrupada o barras de medias
categóricas y una tabla de calor de relaciones por pares. Cada valor es la ganancia predictiva
leave-one-out de k-NN conjunto frente al mejor predictor marginal, normalizada por la desviación del
objective: indica ayuda predictiva conjunta, no interacción causal. Los puntos acotados y una matriz
máxima 12×12 permiten reproducir la vista en wrappers. La respuesta descriptiva y la cobertura
conjunta empiezan con dos observaciones; la ganancia predictiva empieza con tres, manteniendo suelos
de muestra conservadores para la confianza. Un observador nuevo reconstruye localmente un snapshot
acotado antiguo a partir de la telemetría de candidatos, sin reiniciar un estudio remoto activo. La
consola también muestra restricciones y
candidatos no factibles. El mismo JSON está en `overview` →
`work.items[].study.hpo_analysis` y las acciones recientes en
`.controller`, acotadas a 25 eventos.

La vista de Run se divide en dos regiones. Arriba aparecen la lista completa y alineada de
parámetros, el resumen vivo de duración/tiempos y hasta cuatro curvas. `n` y `p` avanzan o retroceden
la página numerada usando teclas simples que funcionan igual con distintas distribuciones. Abajo
hay una tabla por época: arriba/abajo selecciona, cada curva aplicable la marca en rojo, la época
óptima permanece con marcador verde y fila destacada, y Enter/derecha abre todos sus escalares. `o` cambia la región inferior a la salida
bruta aislada; Mayús+izquierda/derecha la desplaza horizontalmente. La duración se calcula desde
`started_at_utc` durante la ejecución. Los `epoch_time_s` y `validation_time_s` medidos sustituyen
la estimación media por época, etiquetada como aproximada, cuando llega la telemetría. El detalle se
refresca fuera del loop de terminal.

Las curvas se leen con límite de bytes y se reducen a 10–500 puntos representativos conservando
extremos y la época óptima exacta; el TUI pide 80. El log también está acotado durante el refresco y observar nunca modifica
la evidencia.

Las gráficas interactivas siguen la gramática visual de nvtop —ejes temporales enmarcados y curvas
vivas— pero usan un pequeño renderer Unicode, sin añadir una pila de plotting ni sustituir el loop
probado de polling y procesos por un framework asíncrono. Una capa ANSI semántica final aplica color
después de calcular anchos, respeta `NO_COLOR` y mantiene planos snapshots y salida máquina. Las
filas de clúster contienen historiales compactos de CPU/RAM/GPU; el detalle grafica porcentajes
globales de CPU, RAM, utilización GPU y memoria GPU durante `lf top --history SEGUNDOS`, e informa
por separado de la cuota LambdaForge actual. Mayús+izquierda/derecha desplaza líneas largas de logs
de Attempt o salida bruta de Run; izquierda sin modificar vuelve y `h`/`l` son alternativas.

Recogida y presentación son independientes. `LightningRunner` conserva todos los escalares finitos
acotados, pero un proyecto selecciona con patrones shell las curvas que dibuja `lf top`:

```python
from lambdaforge.training import LightningTrainConfig

training = LightningTrainConfig(
    epoch_console_include=["train_loss", "val_*", "*_time_s"],
    epoch_chart_include=["val_*", "epoch_time_s", "validation_time_s"],
    epoch_chart_exclude=["*_aux"],
)
```

`epoch_console_include`/`epoch_console_exclude` seleccionan la tabla humana por época; los patrones
de gráfica eligen independientemente solo la presentación TUI. La selección del CSV sigue disponible
mediante `epoch_metrics_include`/`epoch_metrics_exclude`.

El orden automático prioriza objective, `epoch_time_s`, `validation_time_s`, memoria, validación y
entreno. Los nombres CUDA son precisos: `gpu_mem_mb` es el pico de tensores vivos
(`max_memory_allocated`); `gpu_reserved_mb` es el pool actual de caché del allocator PyTorch;
`gpu_peak_reserved_mb` es su pico en la época. La memoria reservada incluye asignaciones vivas y
bloques cacheados reutilizables, por lo que puede ser mucho mayor sin ser otra reserva del scheduler.
El packing adaptativo nunca deduce concurrencia de ese dato: usa `resources.gpu_memory` por Run como
umbral vivo de admisión y la memoria libre actual del driver, llenando slots seguros sin exigir el
máximo de antemano. Tras cada Run empaquetado, termina su proceso dedicado y libera el contexto
CUDA completo antes de readmitir el slot.

No existe un tipo de ejecución específico para entreno. La validación local marca como estudio de
parámetros cualquier Work ordinario con `search` o varias `seeds`. Por ello su pantalla está
disponible durante `preparing` y explica que la telemetría seguirá pendiente hasta que el worker
actual cree el índice acotado. Los clientes máquina reciben la misma declaración en
`work.items[].study_expected`; `study` permanece `null` hasta que exista evidencia viva. Un Run
lanzado con un worker antiguo no puede generar retroactivamente telemetría por Run y necesita una
nueva ejecución con el runtime remoto actualizado.
Varios pasos de workflow, un nivel paralelo y la concurrencia de `self.map()` no forman por sí
mismos un estudio de parámetros: esos Works conservan la navegación Attempt/log y no crean índice
de estudio. Esta inferencia semántica evita un selector de presentación en YAML que podría
contradecir la ejecución real.

`LightningRunner` registra automáticamente los escalares finitos de callbacks tras validar.
`EpochStats` aporta la duración de época y el bridge mide la validación. Otro trainer usa las
métricas genéricas, sin API especial:

```python
self.metrics.log("train_loss", train_loss, step=epoch)
self.metrics.log("val_auprc", val_auprc, step=epoch)
self.progress.update(epoch, epochs, message="entrenando")
self.log(f"época {epoch} terminada")  # narración opcional con flush inmediato
```

`print()`, logging estándar y `self.log()` quedan capturados en el `work.log` del Run seleccionado,
de modo que el drill-down es legible incluso con 24 hijos activos. `metrics.log` es evidencia
numérica estructurada y alimenta curvas/objective; `progress.update` expresa progreso grueso; el log
explica a humanos. No se deben codificar curvas imprimiéndolas.

Los consumidores máquina usan el mismo modelo de lectura sin parsear el TUI:

```bash
lf overview --json
lf show WORK --json
lf show WORK --run trial-00017-seed-4 --json
lf logs WORK --run trial-00017-seed-4 --tail 300
```

`work.items[].study` contiene catálogo compacto y claves exactas. `show --run` devuelve parámetros,
últimos valores, curvas reducidas, log acotado, fallo y rutas de evidencia; `--curve-points N` elige
10–500 puntos. `logs --run` emite solo ese log. Una interfaz headless debe consultar el JSON;
`--follow` queda para logs externos porque `lf top` ya refresca cada Run de forma segura.

## 8. Ejecución, identidad y reutilización

La identidad científica incluye import del Work, identidad del código consumidor, argumentos,
hashes de ficheros, content IDs de datasets, seed y parámetros de variante. Clúster, rutas, tiempo,
Job y Attempt son procedencia operacional y no cambian la definición.

El plan no crea estado. Una ejecución normal reutiliza un éxito verificado. Un Run fallido puede
tener otro Attempt usando el snapshot YAML enviado aunque el original cambie; checkpoints
compatibles activan `resuming`. `--restart` elimina checkpoints del Run. `--rerun` crea otra
Execution deliberada. El control plane rechaza un duplicado activo de misma identidad/destino salvo
`--allow-duplicate`.

## 9. Resultados y metadata

Cada Attempt escribe `result.json`, `environment.json`, `work.log`, JSONL de métricas, progreso,
outputs y artefactos. La Execution escribe `execution.json`, su configuración inmutable y un
`result.json` agregado. Se preservan identidad, paquete/código consumidor, argumentos lógicos,
entradas, recursos pedidos, tiempos, resultado, métricas, checksums, datasets, fallo, resume y Job.

`environment.json` es la única fuente de procedencia del entorno: Python/plataforma, paquetes
críticos, Torch/CUDA, Git y herramientas externas usadas. No se duplica el listado en cada resultado.
Las rutas físicas son evidencia operacional y nunca sustituyen la identidad lógica.
`lf results list/show/compare` lee manifests, no infiere semántica mediante globs. La comparación
calcula count/media/min/max; una clasificación exige `--metric` y dirección explícita.

## 10. Clústeres y Jobs

Un usuario puede configurar el perfil completo mediante un flujo explicado en terminal:

```bash
lf clusters setup
lf clusters modify gpu
```

El asistente cubre conexión/autenticación, rutas de workspace/proyecto/datasets, storage, Python
gestionado o existente, política PyTorch/CUDA, dialecto scheduler y acceso GPU. La automatización
usa `clusters add/set/unset/credentials/test`; el asistente llama esas operaciones, sin mantener una
segunda implementación de configuración. Backend responde cómo se lanza el proceso: Direct para
un host SSH normal incluso si la GPU se obtiene con `gpu exec`, y SLURM solo para `sbatch`. El paso
posterior de acceso GPU configura reserva/claims. Todas las preguntas ofrecen
`0`/`q`/`quit`/`exit` sin aplicar la respuesta actual. En un TTY interactivo, arriba/abajo (o
`j`/`k`) cambia la opción enfocada, un panel contextual explica qué controla y qué riesgo implica,
y Enter selecciona. Una terminal redirigida recibe las mismas descripciones en el fallback
numerado. El acceso GPU no es un tercer backend: `exclusive` controla leases locales,
`shared` permite coexistencia externa si la política del centro lo autoriza, `scheduler` confía en
el grant de SLURM y `command` preserva la visibilidad creada por el launcher del centro.

```bash
lf clusters add gpu --host HOST --user USER --workspace /remote/work \
  --project-root /scratch/USUARIO/mi-proyecto
lf clusters bootstrap gpu --dry-run
lf clusters bootstrap gpu
lf doctor --on gpu
lf run experiments/train.yaml --on gpu
```

Para un perfil existente:

```bash
lf clusters set gpu project_root /scratch/USUARIO/mi-proyecto
lf clusters show gpu
```

El acceso a GPU se configura una vez por clúster:

| `gpu_access.mode` | Significado |
|---|---|
| `auto` | `scheduler` con SLURM; `exclusive` en un host de procesos directo |
| `scheduler` | el batch scheduler posee reserva y aislamiento |
| `exclusive` | espera lease LambdaForge y evita uso externo observado |
| `shared` | coordina Jobs LambdaForge, pero admite una GPU ocupada externamente |
| `command` | no crea lease directo; antepone un argv de claim/launcher del centro |

```bash
lf clusters set host-libre gpu_access.mode shared
lf clusters set citius-gpu gpu_access '{mode: command, command_prefix: [gpu, exec]}'
```

El comando es argv y nunca un string de shell. En CITIUS, `gpu exec` es el modo gpuctl preferido:
su reserva coincide con la vida del comando. Si se necesita claim persistente, claim/release se
declaran juntos:

```bash
lf clusters set citius-gpu gpu_access \
  '{mode: command, command_prefix: [gpu, exec], claim_command: [gpu, claim, --numgpus, "{gpu_count}"], release_command: [gpu, release]}'
```

`{gpu_count}` es la única interpolación y procede de la petición GPU absoluta del YAML. El claim
ocurre en el envío durable en segundo plano; el supervisor directo libera tras éxito, fallo o
cancelación y también se intenta liberar si falla el submit. En SLURM se rechazan claims
persistentes. En acceso command/scheduler, el proceso hereda `CUDA_VISIBLE_DEVICES` del centro:
LambdaForge trata índices/UUID/MIG UUID como grants opacos, solo los estrecha por Run y nunca los
amplía o sustituye. Una asignación ausente, repetida o insuficiente falla de forma segura. El Python
gestionado es una ruta absoluta, sin depender de activación shell/Conda. `shared` sigue siendo una
elección explícita de riesgo en hosts permisivos, no el default.

`workspace` y `project_root` tienen responsabilidades distintas. `workspace` es estado propiedad de
LambdaForge: bundles, entornos, directorios de Job y logs se pueden limpiar según sus reglas.
`project_root` es un mirror parcial persistente propiedad del investigador del directorio local que
contiene `pyproject.toml`; LambdaForge nunca lo sincroniza ni elimina. Debe ser una ruta SSH absoluta
distinta de `/`. `lf doctor` comprueba que exista el directorio configurado.

Se conserva el layout original. Si el YAML es `PROYECTO/experiments/design.yaml`,
`{file: ../data/dna/design}` corresponde a `PROYECTO_REMOTO/data/dna/design`; también
`publish_to="../data/resultados.json"` se mapea así. Un `publish_to` absoluto sigue siendo una ruta
remota explícita y puede quedar fuera del mirror.

Hay tres vías deliberadas:

1. Entrada pequeña (hasta 10 MiB): snapshot y transferencia automática en el bundle.
2. Entrada mayor del mirror: el investigador o servicio de transferencia del centro la coloca en la
   ruta relativa equivalente; LambdaForge lee ambos lados y exige tipo/tamaño/SHA-256 exactos, por
   lo que datos ausentes, viejos, parciales o con symlinks fallan de forma segura.
3. Corpus grande reutilizable: dataset gestionado/materializado, que evita rehashear un mirror de TB
   en cada envío y aporta identidad inmutable y placements por clúster.

El mirror no es un `rsync` oculto: copiar cientos de GB durante `lf run` haría impredecibles la
latencia, cuota y red, y muchos centros exigen un DTN. Se sincroniza mediante el mecanismo aprobado
por el centro y después se ejecutan `lf doctor` y `lf run`.

El bootstrap humano informa cada fase por stderr y emite un latido periódico durante solves o
instalaciones largas. Esto no ensucia `--json`, cuyo stdout sigue siendo un único documento máquina.

`lf run` local/remoto crea un Job durable de preparación y devuelve control. `--wait-for-submit`
espera preparación y aceptación del scheduler, no el cálculo; `--dry-run` es directo y sin efectos.
SSH usa OpenSSH, claves/agent/known_hosts/ProxyJump y un ControlMaster privado durante su persistencia.
Los passwords solo proceden de prompt/keyring/env y no entran en argv/YAML/bundles/estado/logs.

Los entornos gestionados son instalaciones inmutables de usuario identificadas por wheels, Python
y plan Torch. Pueden reutilizar Conda o micromamba verificado, pero nunca modifican Python/CUDA/
drivers del sistema ni hacen fallback silencioso a CPU. Cada Job tiene workspace mutable propio;
no ejecuta sobre bundle cache. En remoto `source_dir` es el código consumidor staged.

### Paquetes nativos declarados por el proyecto

Los ejecutables nativos pertenecen al proyecto instalable, no a cada YAML. La declaración opcional
es:

```toml
[tool.lambdaforge.environment]
manager = "conda"
file = "environment.yml"
required_executables = ["mmseqs", "foldseek"]
```

`manager` solo admite `conda`, pero LambdaForge usa su micromamba fijado y verificado; no exige
Conda global. `file` y `lockfile` son excluyentes y quedan bajo la raíz del proyecto. Los ejecutables
son nombres simples. `package_cache` solo acompaña un lock offline. En `environment.yml` solo se
admiten `name`, `channels` y `dependencies` como strings. Se rechazan prefix, variables, secciones
`pip:` anidadas, hooks y paquetes Torch/CUDA, cuyo plan pertenece al clúster.

```yaml
name: wisdom
channels: [conda-forge, bioconda]
dependencies: [python=3.11, pip, biopython>=1.84, foldseek, mmseqs2]
```

```bash
lf clusters bootstrap gpu --project . --dry-run
lf clusters bootstrap gpu --project .
lf doctor --on gpu
lf run experiments/design.yaml --on gpu
```

El dry-run no descarga, resuelve canales, construye wheels ni modifica el clúster: explica hash,
paquetes, plataforma/subdir, ejecutables, conectividad y un solve exacto ya cacheado si existe. La
preparación real resuelve inventario name/version/build/channel/subdir y lo incorpora a la identidad
junto con bytes de especificación y wheels, plataforma, Python, Torch y política offline.

El resultado es un solo prefijo: micromamba crea un temporal exacto y su propio Python instala Torch
y wheels. Antes del rename atómico se comprueban `pip check`, inventario Conda, TLS, framework,
Torch/CUDA y ejecutables. El recibo registra path/versión y paquete propietario con versión, build,
canal y subdir. Builds concurrentes comparten lock por identidad. `tools.require()` busca primero el
`bin` de ese Python y sigue siendo comprobación, nunca instalador.

La verificación enriquece `micromamba list --json` con los registros regulares `conda-meta` del
prefijo, conservando el subdir real `linux-*` o `noarch` aunque esa versión del gestor lo omita. La
metadata ausente es un error de entorno y una diferencia real se resume por paquete/campo sin volcar
el prefijo completo.

Offline requiere lock Conda `@EXPLICIT` específico de plataforma, SHA-256 en cada URL y sus bytes:

```toml
[tool.lambdaforge.environment]
manager = "conda"
lockfile = "locks/linux-64.explicit"
package_cache = "vendor/conda-linux-64"
required_executables = ["mmseqs", "foldseek"]
```

Cada basename/hash debe corresponder a un fichero regular; links, duplicados, credenciales, queries
y plataforma distinta fallan antes del envío. El cache es content-addressed y reconstruible tras
publicar el prefijo. Para que también pip/Torch sea offline se mantiene además el `wheelhouse` del
perfil. Cada plataforma (`linux-64`, `linux-aarch64`, `linux-ppc64le`) necesita su lock/cache.
`environment: existing` conserva intencionadamente su contrato manual y no instala esta declaración.

`lf top` muestra Works y clústeres semánticos: arriba/abajo recorre. En un estudio Enter/derecha
avanza por Trials, Runs de seed y su panel vivo; `a` abre los Attempts externos numerados. Los demás
Works avanzan directamente por Attempts y logs. Un estudio terminal sin índice publicado cae a sus
Attempts externos, conservando la salida ordinaria y el error aunque el fallo precediera a la
telemetría. Dentro de una seed fallida, `e` amplía tipo/mensaje/fase/ruta/traceback; en el log de un
Attempt la misma tecla alterna el traceback persistido. Izquierda vuelve. El log abierto se actualiza solo y conserva
el scroll manual; situado al final sigue líneas nuevas. Los IDs operativos largos se reservan para
`lf jobs` y la vista máquina. `lf overview --json` expone Works, `attempt_history` y Jobs para
herramientas. `lf jobs ...` conserva el
control avanzado de IDs, scheduler, logs, cancelación y reconciliación. Cada Job local conserva sus
raíces absolutas, por lo que envío detached, observación, logs y borrado no dependen del directorio
actual; los registros relativos antiguos se resuelven desde la configuración fuente guardada. Un
proveedor inaccesible es estado unknown/last-known, no un falso fallo científico; al recuperarse se
restaura el estado real y desaparece el error de conexión obsoleto.

La cancelación respeta la jerarquía. `lf cancel WORK` cancela todos los Jobs no terminales agrupados
en ese Work semántico; aunque falle un proveedor intenta los demás y después informa de operación
incompleta. `lf jobs cancel JOB` y cancelar un Attempt numerado afectan solo a ese Job. Un supervisor
directo señala primero el grupo científico cuya identidad verificó y después localiza descendientes
del mismo usuario con el `LAMBDAFORGE_JOB_ID` exacto heredado, incluidos workers reparentados o con
sesión propia. Termina, escala tras una gracia acotada y comprueba cero supervivientes antes de
confirmar. La misma limpieza se aplica al salir el proceso principal, evitando que un éxito nominal
abandone trainers o dataloaders. Este marcador es procedencia interna; el consumidor no debe
asignarlo ni sustituirlo. En SLURM se usa el comando configurado, cuya política debe cancelar la
asignación completa y sus steps. La cancelación es idempotente para Jobs directos ya registrados
como cancelados: repetir `lf cancel WORK` vuelve a verificar ownership y corrige fugas huérfanas de
versiones anteriores.

Cada Job Work publica además un resultado estructurado acotado en la raíz exacta del Job.
`lf logs WORK` y `lf jobs logs JOB` añaden `Scientific failure` después del stream recortado si los
logs no contenían el diagnóstico. Siempre conserva tipo, mensaje, fase/ubicación y ruta del
`result.json`; `--verbose` o `--debug` añade el traceback salvo que ya esté visible. `--tail N` solo
recorta stdout/stderr, nunca la causa terminal. `--json` devuelve `failure`, `failures` y
`result_path`. `lf show` y `lf jobs show` exponen el mismo fallo. Workspaces 0.12 anteriores se leen
de su ubicación legacy acotada; los nuevos usan la copia exacta de la raíz del Job.

## 11. Limpieza y seguridad

- Attempt/Execution: resultados, logs, artefactos, progreso y workspace exacto;
- Run: checkpoints e intentos;
- durable independiente: datasets publicados y placements;
- compartido inmutable: entornos y bundles referenciados;
- reconstruible: caches y entradas no referenciadas;
- externo: fuentes y ubicaciones mencionadas, nunca poseídas por referencia.

`lf delete WORK` es preview-first y elimina una Execution local exacta o Job terminal exacto. Los
datasets y materializaciones tienen operaciones separadas. `lf clean` presenta cache reconstruible
y la misma compactación de Attempts terminales verificada por hash que se ejecuta automáticamente.
Conserva referencias/leases activas, outputs correctos no publicados y evidencia ligera; rechaza
raíces/symlinks inseguros y es idempotente. Esto permite recuperar con seguridad el bulk de Jobs
anteriores tras actualizar. YAML es código confiable y puede importar Python arbitrario;
LambdaForge no es sandbox.

En `lf top`, `d` confirma el borrado del Work terminal o Attempt seleccionado y `D` confirma la
limpieza de todo el historial terminal. Los Jobs activos nunca se borran. La operación se ejecuta
fuera del loop de teclado, elimina workspace, eventos y registro de envío exactos y conserva
datasets publicados, caches, entornos y Jobs ajenos. `lf jobs clear` presenta la operación y
`lf jobs clear --apply` la aplica; un fallo conserva el registro local afectado.

Al terminar existe una retención automática más estrecha: elimina únicamente artefactos
gestionados parciales de Attempts fallidos/interrumpidos y duplicados internos verificados de
outputs publicados. Conserva toda evidencia ligera requerida por `lf top`, `lf logs`, resultados y
reproducción; `retention.json` registra bytes recuperados. Los entornos inmutables sustituidos
también se podan tras activar un reemplazo verificado, excepto el activo y los referenciados por
Jobs vivos. Bootstrap y preparación automática comparten la regla; la procedencia del Attempt
permanece aunque se recojan bytes reconstruibles del entorno.

## 12. Referencia CLI

| Comando | Propósito único |
|---|---|
| `init` | crear proyecto Work instalable |
| `validate` | validar configuración/clase/entradas localmente |
| `explain` | explicar firma, defaults y recursos |
| `run` | única ejecución científica |
| `top`, `overview` | vista humana viva y vista máquina |
| `show`, `logs`, `cancel`, `retry`, `delete` | operaciones semánticas de Work |
| `jobs ...` | control de bajo nivel; `clear [--apply]` limpia historial terminal |
| `clusters setup/modify` | interfaz interactiva explicada sobre operaciones nativas |
| `clusters ...`, `doctor`, `resources` | configuración scriptable; bootstrap admite `--project` y `--dry-run` sin efectos |
| `datasets ...` | inspección/verificación/placement/borrado de versiones |
| `results list/show/compare` | consultar resultados de Execution |
| `clean` | preview/aplicación de GC reconstruible |

`lf help`, `lf --help`, `lf help clusters add` y los `--help` anidados terminan con código cero.
Los fallos tienen categorías/códigos estables. `--json` es para tooling y `--debug` añade traceback.
No existen comandos antiguos de authoring o dataset build: todo cálculo usa `lf run`.

## 13. Clustering

Se instala con `python -m pip install "lambdaforge[clustering]"`. Importar `lambdaforge` funciona
sin el extra; usar un clusterer sin backend produce el comando de instalación. El backend es
scikit-learn >=1.3,<2 por incluir una implementación madura de HDBSCAN.

```python
import lambdaforge as lf

resultado = lf.clustering.HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
    distance="euclidean",
    threads=8,
).cluster(features)
```

El paquete está fuera de `nn` porque no es una capa neuronal. No hay factory ni registro adicional:
las clases directas son explícitas y una extensión hereda el ABC pequeño `Clusterer`.

Todo clusterer acepta matriz finita `[N, F]` de NumPy o PyTorch; el tensor se separa y normaliza por
CPU. Un `Distance` custom se evalúa en el device/dtype de sus parámetros o buffers y la matriz se
separa después hacia CPU para el backend.
Devuelve `ClusteringResult` inmutable con `labels`, `n_clusters`, `noise_count`, `noise_fraction` y
`diagnostics`. `centers`, `inertia` y `probabilities` solo existen cuando son reales. Ruido es `-1`.

| Clusterer | Distancias | Evidencia específica |
|---|---|---|
| `KMeans`, `MiniBatchKMeans` | solo Euclídea/cuadrada | centros e inercia; `seed` determinista |
| `DBSCAN` | euclidean, manhattan, minkowski, chebyshev, cosine o `Distance` precomputed | ruido |
| `HDBSCAN` | mismas métricas density o `Distance` precomputed | ruido y probabilidades |
| `Agglomerative` | Ward solo euclidean; otros linkages native/precomputed | etiquetas |

Solo existe el contrato `lambdaforge.nn.distances.Distance`: `EuclideanDistance`,
`SquaredEuclideanDistance`, `ManhattanDistance`, `MinkowskiDistance`, `ChebyshevDistance`,
`CosineDistance`, `AngularDistance` y `MahalanobisDistance`. Un objeto no nativo crea de forma
explícita `[N,N]` bajo `no_grad`, valida resultado y usa `metric="precomputed"`. El límite por defecto
es 512 MiB; `max_pairwise_bytes=None` es un escape deliberado con coste O(N²).

`adjusted_rand_index`, `silhouette_score` y `stability` producen evidencia. Escalado, imputación,
PCA, parámetros y umbral de estabilidad siguen siendo política científica del proyecto.

## 14. Componentes neuronales reutilizables

Los componentes PyTorch no dependen de Work/YAML: se instancian en Python del proyecto.
`lambdaforge.nn.models` reexporta el catálogo público:

| Familia | Opciones públicas |
|---|---|
| general/densa | `Model`, `MLP`, `CNN2D`, `ECMP`, `BatchedKNN` |
| grafos | `GCN`, `GAT`, `GATv2`, `GIN`, `GraphSAGE`, `PNA`, `RelationalGCN`, `GraphTransformer`, `EGNN`, `TensorFieldNetwork`, capas públicas y `GraphReadout` |
| secuencia | `RNNModel`, `GRUModel`, `LSTMModel`, `TemporalConvNet`, Transformer encoder/decoder/seq2seq, `ConformerModel`, `StateSpaceAdapter` |
| conjuntos | `DeepSets`, `SetTransformer` |
| tabular | `ResidualMLP`, `FTTransformer`, `TabNet`, `SAINT`, `AutoInt`, `DeepFM` |
| visión | `ResNet2D`, `UNet2D`, `MobileNetV2`, `ConvNeXt2D`, `VisionTransformer2D`, `FeaturePyramidNetwork2D` y bloques/backbone públicos |
| composición | `AutoEncoder`, `VariationalAutoEncoder`, `SiameseModel`, `MultiTaskModel`, `MixtureOfExperts`, `EnsembleModel` |
| generativos | `GaussianDiffusion`, `DiffusionSchedule`, `VectorQuantizedAutoEncoder` |
| científicos/implícitos | `NeuralODE`, `NeuralCDE`, `DeepONet`, `FourierNeuralOperator1D`, `SIREN` |
| árboles diferenciables | `ObliviousDecisionTree`, `NODE`, `GradTree`, `GRANDE` |

Los namespaces contiguos son `lambdaforge.nn.losses`, `activations`, `normalizations`, `pooling`,
`distances`, `similarities`, `kernels`, `encodings`, `regularization` y `uncertainty`; las métricas
clásicas están en `lambdaforge.metrics`. Las firmas y docstrings concretas son la autoridad de
parámetros.

Esta revisión no añade un `GNN` genérico, otro MLP ni una factory. El catálogo ya cubre baselines
densos, grafos, secuencias, imágenes, conjuntos y tabular. Un alias vago ocultaría decisiones
científicas de topología, agregación, readout, equivariancia y forma de salida, además de duplicar
clases probadas. Una arquitectura de dominio pertenece al proyecto; un nuevo primitivo reutilizable
solo se añade con contrato tensorial y pruebas de conformidad precisas.

## 15. Arquitectura y extensiones

`WorkConfig` posee parsing, introspección y expansión. `WorkRunner` enlaza runtime, llama una vez a
`run()` y finaliza resultados. `ControlPlane` selecciona destino, evita duplicados y prepara entorno,
bundle y scheduler. `Transport` y `Scheduler` son fronteras reales. Servicios de Job, dataset y
storage conservan ownership durable independiente.

| Componente | Responsabilidad | Por qué le pertenece |
|---|---|---|
| `WorkConfig` | esquema, firma Python, expansión y valores enviados | la configuración termina antes del scheduler |
| `WorkRunner` | directorios, identidad, entradas, una llamada a `run()` y resultado | es la frontera local del lifecycle científico |
| `AdaptiveSearchPolicy` y controlador WorkRunner | presupuestos de seeds, promoción conservadora, slots GPU y stops | la decisión adaptativa queda sobre cada Work y bajo una reserva externa |
| `WorkRuntime` | servicios/vistas enlazados una vez | evita estado global mutable |
| `ManagedFileStore` | claves, records, lock, validación, huella y promoción | cache/checkpoint comparten invariantes de bytes |
| `WorkCache` | valores tipados, ficheros/fetch/rate limit y lease GC | su vida difiere de checkpoint |
| `CheckpointCollection` | JSON/ficheros del Run | resume no pertenece a outputs ni cache GC |
| `OutputCollection` y retención | outputs, publicación atómica, evidencia exacta y compactación terminal | solo metadata prueba redundancia; el proveedor cubre interrupciones |
| `ToolService` | resolución, proceso argv, logs, env hijo y ledger | el proyecto elige comando; framework ejecuta seguro |
| `EnvironmentManifest` | procedencia software/hardware/Git/plugin/tool | una sola fuente de verdad |
| `SubmissionService`/`ControlPlane` | preparación asíncrona y destino | responsabilidad operacional |
| `NativeEnvironmentSpecification`/`NativeEnvironmentPlanner` | declaración estricta, solve de plataforma e inventario Conda | software nativo pertenece al despliegue, no a parámetros/hooks del Work |
| `ManagedEnvironmentProvider` | venv o prefijo Conda temporal verificado, pip/Torch y publicación atómica | convierte un plan completo en entorno ejecutable |
| `GpuAccessPolicy` | admisión scheduler/exclusive/shared/wrapper de claim | la política del clúster no entra en YAML científico ni Work |
| `StorageOperations` | GC exacto, poda de entornos, borrado de Job y compactación | toda mutación queda acotada, idempotente y auditable |
| `Transport` | conexión y transferencia | no decide semántica del scheduler |
| `Scheduler` | submit/observe/signals de Jobs | proceso directo y SLURM tienen autoridades distintas |
| `DatasetPublisher`/`DatasetRegistry` | bytes/índice inmutable y placements | dataset sobrevive al Attempt |
| adapters `Clusterer` | input, capabilities y resultado normalizado | ciencia Python independiente del runtime/YAML |

Un Attempt resuelve entradas/identidad, crea servicios, captura entorno, llama `run()` redirigiendo
logs, valida el JSON primario, finaliza todo output, añade las herramientas usadas al mismo manifest,
elimina temporales y persiste `WorkResult`. Un fallo no publica outputs gestionados ni staging de
dataset incompleto, pero conserva logs y diagnóstico.

El proyecto extiende Python normal: módulos PyTorch, losses, métricas, lectores y helpers se crean
dentro del Work. La infraestructura permanece en servicios cohesionados
`outputs/metrics/checkpoints/cache/tools/progress`. No se introduce otro ejecutable, DSL, Task,
Source/Transform/Sink ni selector global. Un campo YAML nuevo solo se justifica por una decisión de
planificación del investigador, no por un objeto interno que puede seguir en Python.
