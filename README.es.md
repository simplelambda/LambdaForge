# LambdaForge

Español · [English](README.md)

LambdaForge es un runtime gestionado para investigación científica reproducible. El investigador
escribe Python normal en una clase `lambdaforge.Work`; el framework se ocupa de ejecución local o
remota, recursos, reintentos, métricas, artefactos, datasets, procedencia, resultados y limpieza.

## Índice

1. [Instalación](#instalación)
2. [Primer Work](#primer-work)
3. [Servicios y operación](#servicios-y-operación)
4. [Infraestructura científica gestionada](#infraestructura-científica-gestionada)
5. [Clustering](#clustering)
6. [Modelos neuronales reutilizables](#modelos-neuronales-reutilizables)
7. [Composición y experimentos adaptativos](#composición-y-experimentos-adaptativos)
8. [Observación y operación](#observación-y-operación)
9. [Análisis de estudios](#análisis-de-estudios)
10. [Consola de investigación](#consola-de-investigación)

## Instalación

El proyecto científico y LambdaForge son paquetes independientes instalados en el entorno del
proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge==0.14.0
python -m pip install -e .
python -m pip check
```

`lf init mi-estudio` genera un proyecto instalable completo. Se requiere Python 3.10 o posterior.

La ayuda funciona tanto en forma convencional como natural y termina correctamente incluso cuando
otra aplicación invoca directamente `CommandLineInterface.main()`:

```bash
lf --help
lf help
lf run --help
lf help clusters add
```

Ejecuta `lf` sin argumentos en una terminal interactiva para abrir la Consola de investigación. Su
pantalla Clusters crea y edita perfiles con ayuda contextual, credenciales seguras y pasos
explícitos de test, bootstrap y doctor. Los scripts siguen usando directamente
`lf clusters add/set/unset/credentials/test`. Con entrada/salida redirigida, o con `lf --json` sin
comando, se imprime la ayuda y nunca se intenta abrir una aplicación a pantalla completa. Las
entradas antiguas `lf top`, `lf clusters setup` y `lf clusters modify` se retiraron en 0.14 para
mantener una sola interfaz interactiva.

Un proyecto que además necesite ejecutables nativos los declara una sola vez; los proyectos solo
pip no necesitan Conda ni configuración adicional:

```toml
[tool.lambdaforge.environment]
manager = "conda"
file = "environment.yml"
required_executables = ["mmseqs", "foldseek"]
```

```yml
channels: [conda-forge, bioconda]
dependencies: [python=3.11, pip, mmseqs2, foldseek]
```

`lf clusters bootstrap gpu --project . --dry-run` explica declaración, paquetes, plataforma y
conectividad; al repetir sin `--dry-run`, LambdaForge usa su micromamba verificado, crea un único
prefijo Conda inmutable e instala ahí los wheels exactos de LambdaForge y del consumidor. También
valida los ejecutables y registra paquete, versión, build, canal y plataforma. `lf run ... --on gpu`
detecta después la declaración automáticamente. Dentro del Work,
`self.tools.require("mmseqs", version_args=["version"])` comprueba la herramienta preparada, pero
nunca instala durante el cálculo. El manual explica locks offline exactos.

## Primer Work

```python
from pathlib import Path
import lambdaforge as lf


class Resumir(lf.Work):
    def run(self, fuente: Path, limite: int = 100) -> dict[str, int]:
        lineas = fuente.read_text(encoding="utf-8").splitlines()[:limite]
        informe = self.outputs.file("informe", filename="informe.txt", role="report")
        informe.write_text("\n".join(lineas))
        self.metrics.log("filas", len(lineas))
        self.outputs.value("resumen", {"filas": len(lineas)})
        return {"filas": len(lineas)}
```

```yaml
name: resumen
run: mi_proyecto.work.Resumir
with:
  fuente: {file: data/entrada.txt}
  limite: 100
resources:
  cpu: 2
  memory: 2GiB
```

La clase debe heredar `Work` y `run()` es su única entrada. La firma y el docstring de Python
definen parámetros, tipos y valores por defecto. Solo `{file: ...}`, `{dataset: NOMBRE@VERSION}` y
`{from: paso.salida}` tienen semántica especial; una cadena normal nunca se interpreta como ruta.
`{file: ...}` declara una entrada que ya existe: LambdaForge la resuelve respecto al YAML, calcula
su hash y la copia automáticamente al remoto si está bajo el límite configurado. Las entradas
grandes del proyecto pueden usar un mirror remoto explícito; los corpus compartidos estables
deberían ser datasets gestionados. Los resultados no se declaran en `with`; se crean dentro del
Work mediante `self.outputs`.

```bash
lf validate experiments/resumen.yaml
lf explain experiments/resumen.yaml
lf run experiments/resumen.yaml --dry-run
lf run experiments/resumen.yaml
lf run experiments/resumen.yaml --on cluster-gpu
```

Si el clúster contiene una copia parcial del proyecto, se declara una vez su raíz remota absoluta:

```bash
lf clusters set cluster-gpu project_root /scratch/USUARIO/WISDOM
lf doctor --on cluster-gpu
```

El mismo `{file: ../data/dna/design}` apunta entonces a `PROYECTO/data/dna/design` local y a
`/scratch/USUARIO/WISDOM/data/dna/design` remoto. Hasta 10 MiB sigue siendo un snapshot automático
del bundle. Por encima del límite LambdaForge no copia implícitamente: la ruta debe estar dentro del
proyecto local, su equivalente remoto debe existir y se comparan tipo, bytes y SHA-256 antes del
scheduler y otra vez en el worker. El SHA-256 de un directorio usa una identidad de árbol canónica
y versionada: normaliza Unicode, ordena los componentes por bytes, usa `/` como separador lógico,
delimita cada registro e incluye directorios vacíos, pero excluye permisos y fechas del host. Por
tanto no depende de `find`, del `sort` del shell, del locale ni del orden de creación. Un mirror
ausente, cambiante o desactualizado falla de forma segura.
Sincronizarlo mediante el servicio recomendado por el clúster es responsabilidad explícita del
investigador. Para cientos de GB/TB es preferible un dataset gestionado: verificar el mirror también
lee todos los bytes, mientras el dataset aporta identidad y placements reutilizables.

Los envíos son asíncronos tanto en local como en remoto: la terminal vuelve tras crear un Job
durable que puede seguirse con la Consola o `lf logs`. `--wait-for-submit` espera expresamente a la
preparación y al scheduler, no al cálculo científico; `--dry-run` sigue siendo directo y sin efectos.

## Servicios y operación

Para trabajar en varios proyectos, ejecuta `lf` dentro de la carpeta de cada uno o sus subcarpetas.
`lf project --json` muestra la raíz y el identificador actuales. El entorno virtual selecciona las
dependencias Python; el `pyproject.toml` más cercano selecciona el proyecto de LambdaForge. Jobs,
Works, Studies y YAML recientes se filtran por proyecto; resultados e índice de datasets locales
permanecen en su `.lambdaforge`. Los perfiles de clúster y las credenciales se pueden compartir.
Los nuevos jobs, registros y cachés remotos viven bajo
`<workspace>/.lambdaforge/projects/<project-id>/`. Las reservas de GPU y recursos siguen compartidas
para coordinar ambos proyectos sobre el mismo hardware.

`lf init` escribe un ID estable automáticamente. En un proyecto existente, configúralo **antes del
primer envío** si debe conservar el mismo namespace remoto al mover el repositorio:

```toml
[tool.lambdaforge]
project_id = "mi-proyecto-de-investigacion"
```

Sin ese campo, se deriva un ID legible de la raíz local resuelta; dos carpetas o paquetes con el
mismo nombre en rutas distintas quedan separados. Reutilizar un ID explícito comparte deliberadamente
el espacio remoto. Un `lambdaforge.clusters.yaml` local al proyecto permite sobrescribir
`project_root` para su espejo remoto sin repetir host ni autenticación. Véase
[aislamiento y actualización de proyectos](docs/MANUAL.es.md#18-aislamiento-por-proyecto).

Durante `run()`, `config`, `inputs`, `resources`, `seed`, `trial`, `source_dir` y `resuming` son
inmutables. `outputs`, `metrics`, `checkpoints`, `cache`, `tools` y `progress` son servicios
gestionados;
`run_dir` pertenece al Attempt y `temp_dir` es efímero. `self.map(items, funcion, workers=...)`
ofrece concurrencia ordenada sin persistencia oculta. Solo cuando se necesita reanudar elementos se
usa `self.resume_map(..., key=...)`; la forma antigua `map(..., key=...)` sigue siendo compatible.

`print(..., flush=True)` y el `logging` normal aparecen en los logs del Job. También se puede usar
`self.log("mensaje", level="info")` para emitir inmediatamente una línea con fecha y severidad.
`self.progress.update(...)` es la opción adecuada para progreso cuantificable y
`self.metrics.log(...)` para evidencia numérica científica.

## Infraestructura científica gestionada

Hay tres almacenamientos con responsabilidades diferentes:

- `self.cache.put(clave, contenido)` y `get(clave, default=None)` guardan/recuperan bytes, texto o
  JSON estricto reconstruible sin exponer rutas;
- `self.cache.file/fetch(...)` es la opción avanzada para ficheros grandes o producidos por una
  librería;
- `self.checkpoints.file(...)` y `save_json/load_json` conservan estado necesario para reanudar el
  mismo Run;
- `self.outputs.file/directory(...)` declara resultados científicos durables, que se verifican,
  hashean y registran automáticamente cuando `run()` termina correctamente; `publish_to=...`
  publica en una ruta del investigador y elimina por defecto los bytes internos redundantes después
  de registrar toda la Execution. `retain_internal=True` conserva deliberadamente ambas copias.

El uso normal no necesita crear directorios, nombres `.part`, locks, `fsync` ni llamar después a
`outputs.artifact`:

```python
self.cache.put("resumen", {"aceptadas": 1842})
resumen = self.cache.get("resumen")
normalizados = self.map(registros, normalizar, workers=8)
```

Para descargas y cálculos costosos reanudables se elige explícitamente la API avanzada:

```python
def run(self, identificadores: list[dict[str, str]], workers: int = 8):
    limite = self.cache.rate_limit("archivo", requests_per_second=4)

    def preparar(item):
        return self.cache.fetch(
            f"https://archive.example/{item['id']}.json.gz",
            key=f"archivo/{item['id']}.json",
            retries=4,
            decompress="gzip",
            rate_limit=limite,
            validate=lambda fichero: fichero.size_bytes > 0,
        )

    ficheros = self.resume_map(
        identificadores,
        preparar,
        key="id",
        workers=workers,
        executor="thread",
        name="descargas",
        retries=2,
    )
    indice = self.checkpoints.file(
        "indices/archivo.tsv",
        build=lambda destino: construir_indice(ficheros, destino),
        validate=indice_valido,
    )
    informe = self.outputs.file(
        "informe",
        filename="informe.json",
        role="report",
        media_type="application/json",
        publish_to="resultados/informe.json",
    )
    informe.write_json({"ficheros": len(ficheros), "indice_sha256": indice.sha256})
    return {"ficheros": len(ficheros)}
```

`ManagedFile` es compatible con `os.PathLike` y ofrece `str`, `open`, `read_text`, `read_bytes`,
`key`, `sha256` y `size_bytes`. El cache usa un lock por clave entre procesos, destino temporal,
validación, `fsync`, SHA-256 y promoción atómica. `fetch` añade timeout acotado, reintentos con
backoff, descompresión gzip opcional y limitación de tasa local al Work. Los cortes HTTP/chunked/gzip
y respuestas transitorias se reintentan desde un temporal vacío; los 4xx permanentes no entran en
bucles y nunca se registra contenido parcial. Los checkpoints de
`self.map` guardan referencias lógicas, no rutas de la máquina: si `lf clean --apply` elimina un
fichero referenciado, solo se recalcula ese elemento.

`publish_to` relativo parte del directorio del YAML original. Con el `project_root` anterior,
`publish_to="../data/informe.json"` publica en el directorio equivalente del proyecto remoto, no en
una ruta interna con hash del Job. Sin `project_root`, una publicación relativa remota se rechaza y
debe usarse una ruta remota absoluta. La publicación solo aparece tras un `run()` correcto, es
atómica y rechaza contenido distinto existente salvo `overwrite=True`. El resultado conserva
nombre, hash, tamaño y ruta publicada, pero la copia interna se compacta únicamente tras volver a
verificar que el destino coincide. Un Attempt fallido o interrumpido elimina `artifacts/` parciales
y conserva logs, métricas, procedencia, checkpoints y fallo. Un output correcto sin `publish_to`
se conserva internamente porque es su única copia; las escrituras directas mediante el escape
avanzado `run_dir` no se adivinan ni borran. LambdaForge no confunde una ruta remota con una local
ni transfiere árboles grandes implícitamente al controlador.

Para programas científicos externos:

```python
herramienta = self.tools.require("mmseqs", version_args=["version"])
resultado = self.tools.run(
    [herramienta, "easy-search", str(consulta), str(base), str(salida)],
    name="MMseqs2",
    threads=self.resources.cpu,
    cwd=self.temp_dir,
)
```

El comando es una secuencia argv sin `shell=True`; stdout/stderr se emiten en tiempo real y se
capturan de forma acotada, los límites de threads solo afectan al hijo y una salida distinta de cero
produce un error claro. La ruta y la versión consultada se registran una sola vez en
`environment.json`.

## Clustering

```bash
python -m pip install "lambdaforge[clustering]==0.14.0"
```

```python
import lambdaforge as lf

resultado = lf.clustering.HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
    distance="euclidean",
    threads=8,
).cluster(features)
```

`KMeans`, `MiniBatchKMeans`, `DBSCAN`, `HDBSCAN` y `Agglomerative` comparten
`Clusterer.cluster(X) -> ClusteringResult`. Admiten matrices finitas `[N, F]` de NumPy o tensores
PyTorch que se separan del grafo y pasan a CPU. Las etiquetas, número de clusters, ruido y
diagnóstico tienen la misma forma; centros, inercia y probabilidades aparecen solo si existen.
Scikit-learn >=1.3 es una dependencia opcional y se carga únicamente al ejecutar clustering.

Las restricciones científicas no se ocultan: KMeans es euclídeo, Ward exige distancia euclídea y
los demás algoritmos declaran sus métricas. Un `lambdaforge.nn.distances.Distance` existente se
convierte en matriz precalculada solo cuando el algoritmo puede consumirla y después de comprobar
un límite explícito de memoria O(N²). LambdaForge no normaliza, imputa, aplica PCA ni decide qué
estabilidad es aceptable. `adjusted_rand_index`, `silhouette_score` y `stability` calculan evidencia,
no una política científica.

## Modelos neuronales reutilizables

No falta una familia básica MLP/GNN: `lambdaforge.nn.models` ya ofrece MLP/CNN; GCN, GAT/GATv2,
GIN, GraphSAGE, PNA, RGCN, graph transformer, EGNN y tensor-field networks; RNN/GRU/LSTM, TCN,
Transformer y Conformer; DeepSets/SetTransformer; modelos tabulares; ResNet, U-Net, MobileNet,
ConvNeXt, ViT y FPN; composiciones, generativos y modelos científicos como ODE/CDE, DeepONet, FNO
y SIREN. Losses, métricas, activaciones, normalizaciones, pooling, distancias, kernels e
incertidumbre viven en los namespaces correspondientes. El manual contiene el mapa completo. No se
añade una clase vaga `GNN`: topología, agregación y equivariancia son decisiones científicas.

## Composición y experimentos adaptativos

`steps` expresa una secuencia y `{parallel: [...]}` un grupo paralelo aislado por procesos. `seeds`
crea Runs independientes; `search` expande variantes y `objective` define una métrica registrada o
una utilidad compuesta explícita de rangos fijos. Siempre que `search` tenga `objective`, omitir `strategy` activa la
política adaptativa segura completa: inicio Sobol, propuestas dependientes de resultados, carrera
probabilística de seeds, pruning de curvas, detección de convergencia y confirmación con seeds
nuevas. Un pool Sobol scrambled proporciona puntos reproducibles que cubren el espacio. Si se omite
`startup_trials`, LambdaForge deriva un `InitialDesignPlan` del `ParameterSpace` declarado: conserva
de forma greedy el rank útil de efectos principales/curvatura, cubre niveles categóricos, estados
condicionales activo/inactivo y soporte numérico bajo/interior/alto, y desempata mediante
D-optimal/maximin determinista. Su número de anchors nunca depende de GPUs, `runs_per_gpu` ni
`max_parallel`. Un `startup_trials` explícito sigue siendo el presupuesto autoritativo de anchors,
con la misma selección geométrica. Después, los resultados eligen los candidatos siguientes.
`sampler: auto` usa qLogNEI con GP mixto de BoTorch si está instalado el extra
`lambdaforge[adaptive-hpo]` y existe evidencia suficiente, con fallback k-NN determinista ante una
dependencia ausente o un fallo numérico. El GP ve conjuntamente todas las dimensiones codificadas,
incluidas categorías y activación condicional, y qLogNEI incorpora el error estándar entre seeds.
Cada `(candidato, rung exacto)` es una observación distinta, agregada solo entre seeds que alcanzaron
ese presupuesto acumulativo. Los espacios numéricos usan el GP multi-fidelidad de BoTorch, los
mixtos conservan la fidelidad normalizada en el GP conjunto y el fallback k-NN pondera evidencia de
rung exacto. Nunca se etiqueta como presupuesto completo una media de fidelidades heterogéneas.
Los paneles por parámetro son explicaciones marginales, no el modelo que decide. Solo aparecen en
la Consola los Trials ya propuestos. La arquitectura es un surrogate sensible a fidelidad más un
scheduler externo sensible a coste, no una supuesta adquisición bayesiana multi-fidelidad universal
para todo espacio mixto/condicional.

La autoridad de decisión HPO siempre es una utilidad escalar auditable. La forma compacta heredada
es `objective: {metric: val_score, mode: max}`. Si la calidad científica depende realmente de
varias métricas, declara rangos y pesos fijos:

```text
objective:
  aggregation: geometric
  metrics:
    val_auprc: {mode: max, weight: 3, range: [0.0, 1.0]}
    val_balanced_accuracy: {mode: max, weight: 1, range: [0.0, 1.0]}
```

Existen `weighted_mean`, `geometric` y `chebyshev`. Los pesos se normalizan una vez y los valores
fuera del rango se recortan. Todos los componentes deben registrarse en el mismo `step` entero:
LambdaForge nunca mezcla mejores épocas distintas ni últimos valores independientes. Esa utilidad
gobierna propuestas, seeds, pruning, fidelidad, confirmación y ranking. Los componentes crudos
siguen visibles y la Consola marca el frente Pareto no dominado solo como diagnóstico.

Si una utilidad alta puede resultar engañosa
por sí solo, declara guardas de resultado explícitas en vez de esperar que LambdaForge adivine qué
otras métricas importan. Cada límite se evalúa en el mejor checkpoint del objetivo. La factibilidad
agrega esos valores del mismo checkpoint entre seeds mediante `seed_aggregation: mean` (legacy),
`worst` o `lcb`; `lcb` admite `confidence` y falla cerrado hasta tener repetición suficiente para
estimar incertidumbre. Una métrica puede ser a la vez componente de utilidad y constraint. La
evidencia ausente falla cerrado y los candidatos no factibles siguen visibles, pero no guían el
surrogate ni pueden ganar:

```yaml
name: entreno-con-guardas
run: mi_proyecto.Training
objective:
  metric: val_auprc
  mode: max
  constraints:
    val_accuracy: {min: 0.55, seed_aggregation: worst}
    val_kappa: {min: 0.05, seed_aggregation: lcb, confidence: 0.95}
```

Esto es optimización restringida de un único objetivo, no un compromiso multiobjetivo implícito.
Los umbrales deben expresar validez científica real; no añadas toda métrica registrada solo porque
exista.

El número de seeds es probabilístico, no igual ni fijado por rondas. Un orden compartido permite
diferencias pareadas; se añade una seed solo mientras la probabilidad de estar a
`seed_racing.equivalence_margin` del incumbent alcance
`seed_racing.probability_threshold`. El ganador usa una cota
conservadora de búsqueda o, preferiblemente, la media de `confirmation_seeds` nuevas sobre un top-K
congelado. Por defecto LambdaForge empieza con una seed declarada por candidato y genera
tres seeds de confirmación deterministas y nuevas; cada valor puede sobrescribirse expresamente.
Los Runs de confirmación nunca reciben pruning de rendimiento ni preemption oportunista. Si falla
una seed requerida o no cabe en el presupuesto global, `summary.confirmation.status` queda
`incomplete`, `confirmation_incomplete` es true y nunca se elige el subconjunto afortunado que
logró sobrevivir.

Desactivar HPO es deliberadamente sencillo. `strategy: exhaustive` significa un sweep finito
literal: ejecuta cada combinación de `values` y cada seed, incluidas ramas `when` finitas exactas.
Un `range` continuo no puede agotarse; hay que discretizarlo con `values` o usar modo adaptativo.
`trials` es un presupuesto de candidatos y por eso se rechaza en modo exhaustivo.

```yaml
name: sweep-optimizadores
run: mi_proyecto.Training
seeds: [7, 17]
search:
  strategy: exhaustive
  optimizer: {values: [adamw, sgd]}
  momentum: {values: [0.8, 0.9], when: {optimizer: sgd}}
```

```yaml
name: entreno-adaptativo
run: mi_proyecto.Training
seeds: [4, 7, 32, 54, 65, 94, 109, 124]
search:
  trials: 40
  proposal_pool_size: 640
  min_seeds: 1
  # Presupuesto científico exacto opcional; omitido deriva de la geometría, no del hardware.
  startup_trials: 10
  seed_racing: {probability_threshold: 0.1, equivalence_margin: 0.002}
  confirmation_top_k: 2
  confirmation_seeds: [1001, 1002, 1003]
  sampler: auto
  max_runs: 180
  max_time: 12h
  # Opcional: parar tras 8 resultados completos sin mejorar 0.0005 el récord.
  # Omite ambos campos para consumir los 40 candidatos completos (valor seguro por defecto).
  convergence_patience: 8
  min_improvement: 0.0005
  runs_per_gpu: auto  # o un entero positivo como máximo duro por GPU
  max_parallel: auto  # o un entero positivo como máximo duro global
  failure_retries: 1
  early_stopping: {enabled: true, min_step: 5, confirmations: 2,
                   probability_threshold: 0.05, equivalence_margin: 0.002}
  learning_rate: {range: [0.00001, 0.003], scale: log}
  hidden_dim: {values: [64, 128, 256]}
objective: {metric: val_auprc, mode: max}
resources:
  gpu: 2
  gpu_memory: 16GiB  # VRAM libre mínima exigida antes de iniciar cada Run
  cpu: 16            # reserva total; parte estable por cada posible Run concurrente
  memory: 32GiB      # reserva total; las partes nunca exceden este límite exterior
```

La ejecución separa tres capas. La **planificación científica** decide qué evidencia debería
existir: anchors protegidos, candidatos de optimización, probes emparejados y acciones de
seed/fidelidad. La **planificación de recursos** decide qué cabe de forma segura ahora. El
**dispatch** crea el proceso aislado. Por ello un anchor necesario puede esperar recursos sin estar
cancelado. Si una cota inferior dura demuestra que es imposible en todos los dispositivos
asignados, LambdaForge lo reemplaza por el candidato que mejor conserva sus obligaciones y registra
`REPLACE_STARTUP_ANCHOR`. La capacidad física sobrante no agranda el diseño protegido: mientras hay
poca evidencia ejecuta puntos `OPPORTUNISTIC_COVERAGE` que sí pueden replanificarse.

El `proposal_pool_size` determinista se almacena una sola vez; cada especificación ligera de
candidato/seed referencia solo sus propios valores. La memoria de planificación crece así de forma
lineal y un estudio grande válido no necesita reducir su presupuesto YAML para evitar amplificación
de memoria del framework. La interpretación científica viva tampoco depende del tamaño bruto de
ese pool: LambdaForge reutiliza su geometría y distancias inmutables, remuestrea con precisión viva
acotada y rota una shortlist representativa tras cada evento. El sampler de optimización conserva
el pool determinista completo y puede elegir cualquier miembro; es un límite computacional del
análisis explicativo, no una reducción oculta del espacio de búsqueda. El análisis terminal puede
usar evidencia más densa porque ya no retrasa la recogida ni planificación de Runs.

`trials` es el presupuesto de candidatos y LambdaForge lo consume completo por defecto. Una
confianza baja o cobertura escasa nunca se interpreta silenciosamente como convergencia.
`max_runs` y `max_time` son límites globales explícitos. `convergence_patience` es una política de
racha de récords deliberadamente opt-in: un valor positivo deja de proponer solo cuando esa
cantidad de resultados completos no mejora el incumbent más que `min_improvement`; no es una
prueba de cobertura multidimensional. El evento final indica si se agotó el presupuesto de
candidatos, Runs o tiempo, el pool de propuestas, o esta convergencia explícita. Pruning y carrera
de seeds siguen ahorrando cómputo sin recortar el presupuesto de candidatos declarado. La cobertura
distingue **search coverage** (región intentada o podada científicamente) de **response coverage**
(respuesta completa comparable), junto con diversidad de contextos emparejados. El controlador
puede pedir `COVER_PARAMETER_VALUE` o `COVER_INTERACTION_CELL` si una conclusión está confundida por
soporte estrecho. No hay cuotas mágicas: el valor de otro probe cae cuando evidencia diversa
resuelve la pregunta.

`PERFORMANCE_PRUNE` significa que no compensa continuar una Run para encontrar el óptimo; no dice
que su curva parcial carezca de información. Si después esa Run censurada es la vía más barata para
resolver una pregunta, LambdaForge puede registrar `SCIENTIFIC_CONTINUATION` y reanudar el mismo
Trial y seed desde su checkpoint durable como un nuevo Attempt. No consume otro slot de `trials`,
evita el pruning competitivo y sí consume presupuesto de Runs/tiempo. El prune original permanece
visible y sigue siendo correcto.

Con `auto`, ARI puede aumentar el packing de cada GPU hasta que VRAM física, recursos host,
throughput o política del sitio indiquen esperar. Sustituirlo por `runs_per_gpu: 4` impondría
**como máximo** cuatro Runs por dispositivo; no exigiría crear cuatro slots fijos. Cada candidato
recibe una distribución de memoria futura aprendida de Runs
terminales y activas censuradas compatibles. La memoria física es la autoridad y el headroom futuro se
aproxima mediante

$$
H_g(t)=C_g-E_g(t)-\sum_i M_{i,\mathrm{future}}(t),
$$

donde $C_g$ es la capacidad utilizable, $E_g$ el uso no atribuido a Runs de LambdaForge y
$M_{i,\mathrm{future}}=m_i(t)+R_i(t)$ combina el máximo observado y la distribución de memoria que
todavía puede aparecer. El planner elige la
GPU segura con menor holgura (best fit), por lo que puede juntar una configuración grande y otra
pequeña. Empieza conservador, pero fase, progreso, cambios de trayectoria, allocator y checkpoints
convierten Runs activas en evidencia provisional compartida antes de que terminen.

Los pequeños máximos nuevos del allocator no se consideran automáticamente crecimiento material.
LambdaForge separa jitter, deriva del allocator, asignación monótona y picos abruptos, y actualiza
una prior débil de supervivencia tras progreso o transiciones de fase significativos, no tras cada
sondeo de NVML. Cambiar la frecuencia de monitorización no fabrica confianza. La trayectoria
acotada sirve para mostrar datos; estadísticas incrementales versionadas conservan todos los ciclos
de una Run larga. La distribución futura mantiene colas raras incluso inferiores al uno por ciento
y compone incertidumbre de candidato y residentes con integración exacta de soportes pequeños. Las
fases relevantes se aprenden por familia de Work compatible: se conserva el riesgo de validation o
checkpoint cuando la historia lo justifica, sin exigir a un Work genérico un ciclo de entrenamiento
fijo. `RAMPING` o `PROVISIONALLY_STABLE` explican el estado; el placement consume hazard por fase y
el residual ponderado.

`SAFE_ADMISSION` cabe considerando incertidumbre y cotas OOM. `EXPLORATORY_ADMISSION` es un paso
1→2→3 consciente de checkpoints cuyo progreso e información esperados superan el coste de rollback
e interferencia. Con dos o más GPU intercambiables queda un carril de progreso protegido y solo una
hermana prueba el mismo escalón incierto. Un éxito provisional promueve el packing antes del último
epoch; una OOM posterior lo invalida. Por ello un cold start largo no queda bloqueado con una Run
por GPU solo porque ningún entreno haya terminado.

Esperar también tiene coste. Si existe trabajo útil pendiente y VRAM físicamente utilizable ociosa,
LambdaForge integra fracción ociosa × tasa de valor científico normalizado. Este *wait regret*
cierra cada tramo temporal con su tasa anterior, sobrevive a reinicios y no se descuenta porque
cambien frontera, hazard o checkpoint. Solo se reinicia cuando la oportunidad desaparece o se
satisface; las cotas físicas, políticas del sitio, run cap y throughput negativo siguen mandando.
El controlador científico entrega rango, valor normalizado, incertidumbre y coste esperado, por lo
que su score interno nunca es moneda de recursos. La duración desconocida sigue siendo incierta.
El cálculo de checkpoint trata cada residente por separado, conserva el rollback no
checkpointable, cobra el coste aprendido y espera confirmación durable antes de explorar.

`gpu_memory` es opcional. Si se declara conserva su semántica de suelo de seguridad mínimo para
cada lanzamiento; el compromiso efectivo es el máximo entre ese suelo, la envolvente superior
aprendida y cualquier cota inferior OOM conocida. Nunca se multiplica por Runs activos ni limita el
proceso consumidor. Si se omite se usa predicción automática. `runs_per_gpu` sigue siendo un máximo
duro por dispositivo y `max_parallel` un máximo global. Un candidato que no cabe ahora queda
`RESOURCE_BLOCKED`, no fallido ni pruned, y se reconsidera cuando cambia la memoria. Un candidato
menos prioritario puede hacer backfill seguro si termina antes de la ventana esperada del candidato
pesado. El best fit y la frontera científica acotada evitan reintentos aleatorios y starvation. Si
la primera frontera no puede ejecutarse, el controlador pide una única ampliación acotada a la
misma política científica y lanza su miembro factible más valioso; nunca genera candidatos al azar
hasta que alguno quepa.

CPU, RAM y almacenamiento conservan una parte estable y prudente por Run, derivada del máximo
global de concurrencia. No se prometen temporalmente todos los recursos host a las primeras Runs
del cold start, porque no podrían redimensionarse con seguridad al crecer el packing. La capacidad
no usada sigue disponible para el sistema operativo y `self.resources` comunica siempre la parte
con la que una Run puede contar.

LambdaForge sondea solo las GPU concedidas, reacciona al uso externo, escalona lanzamientos y
actualiza las envolventes con trayectorias acotadas durante toda la Run, incluso cuando todos los
slots están ocupados y la cola temporal está vacía. Persiste picos físicos, diagnósticos del
allocator, duración/tiempo al pico, terminaciones censuradas y cotas inferiores OOM. La admisión
usa el mínimo entre la VRAM física libre y el headroom futuro predicho, de modo que una nueva
presión externa no pueda quedar oculta. Los PID del worker y descendientes CUDA se cruzan con NVML;
los heartbeats del allocator añaden fase, step y picos breves. El fallback se etiqueta como inferido.
El snapshot acotado de evidencia activa se escribe atómicamente: tras reiniciar el controlador
puede informar la incertidumbre como evidencia provisional anterior, pero LambdaForge nunca supone
que sus PID sigan vivos. La evidencia terminal lo sustituye y un cierre correcto vacía el snapshot.
Una OOM sin atribución fiable restringe el packing concreto, no inventa el tamaño del candidato.
No se repite el mismo experimento o uno dominado, pero `heavy+heavy` no prohíbe globalmente
`heavy+small`. Una OOM exploratoria crea un Attempt del mismo Run y reanuda su checkpoint; nunca es
un objective malo ni otro Trial.
La interferencia medida entre co-runners también puede impedir añadir otro proceso aunque quepa en
VRAM: se maximiza trabajo científico útil por tiempo, no memoria ocupada.
Una Run podada por rendimiento sí enseña recursos cuando sus fases y medidas por proceso completan
el perfil; su objective científico continúa censurado. Los diagnósticos persisten cambios
`RESOURCE_WAIT`, `RESOURCE_EXPLORE`, checkpoint, promoción, invalidación y recuperación con P(fit),
hazard, rollback, wait regret y motivo explícito, sin escribir en cada sondeo.

Cada evento terminal replanifica la frontera científica contra el estado físico actual. La
disponibilidad física también es un evento: si la cola ejecutable queda vacía y una GPU concedida
tiene capacidad admisible, se solicita inmediatamente una frontera científica acotada sin esperar
a que termine otra Run. El controlador sigue eligiendo la acción y el planner decide si es segura o
si merece un único escalón exploratorio. Cada Run empaquetado posee un proceso
spawn nuevo que termina al acabar el Run; no se reutiliza un worker CUDA ocioso cuyo contexto
podría retener VRAM y bloquear para siempre la cola.
Una GPU llena no falla todo el estudio; si otra admite dos de cuatro slots, ejecuta
dos. Solo se rechaza como imposible un umbral mayor que la memoria total de todas las GPU asignadas.
En CPU puede limitarse la concurrencia con `max_parallel`. El observador de memoria usa un proceso
hijo efímero, así que no deja un contexto CUDA ocioso en cada dispositivo ni consume una plaza
científica de `runs_per_gpu`.
Si aun así un hijo sufre CUDA OOM, solo ese Run pasa a `retrying`: se conservan el Attempt y la
evidencia censurada. Los Runs sanos y las demás GPU continúan. La recuperación está acotada por
`failure_retries`; una OOM repetida tras una mejora real del placement queda como evidencia terminal
honesta en vez de crear un bucle infinito.
`self.metrics.log("val_auprc", valor, step=epoch)` permite pruning basado en probabilidad. Una
tendencia local proyecta cada curva más allá del step común y solo solicita parada cuando la
competitividad práctica cae bajo el umbral, protegiendo mejor arranques lentos que eliminar una
fracción inferior fija. `LightningRunner` enlaza automáticamente la métrica y la parada. Un loop
propio debe registrar `step=` y retornar en un checkpoint seguro al detectar `self.stop_requested`.
Con una métrica solo final se adaptan seeds, pero no puede pararse el entreno actual.

Para entrenos reanudables y acumulativos, `fidelity` evita conceder a todos los candidatos el
presupuesto completo de epochs. La unidad la define el Work: LambdaForge nunca supone que un entero
arbitrario signifique epochs. El Work recibe `self.fidelity` y debe guardar checkpoint al llegar a
`target` y continuar desde `current`; `LightningRunner` limita `max_epochs` y gestiona `last.ckpt`
automáticamente:

```yaml
name: entreno-multi-fidelidad
run: mi_proyecto.Training
search:
  trials: 40
  fidelity: {min: 5, max: 100, reduction_factor: 3}
  learning_rate: {range: [0.00001, 0.003], scale: log}
objective: {metric: val_auprc, mode: max}
```

Solo se promueven configuraciones estadísticamente competitivas y la confirmación final siempre
usa el máximo. El controlador guarda `hpo-control/state.json` y el diario append-only
`hpo-control/decisions.jsonl` con cada `START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY`, fallback, convergencia y
confirmación; el resumen del resultado enlaza ambos ficheros.

La planificación está dirigida por eventos. Cada Run terminal hace que el slot libre reconsidere
`START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY` y `RESUME_PREEMPTED`; cada alternativa guarda
`controller_value`, coste incremental y prioridad en `hpo-control/decisions.jsonl`. El valor es una
heurística auditable en escala común, no ganancia de información ni probabilidad calibrada: seeds
usan reducción relativa del error estándar, regiones nuevas usan cobertura/escasez y promociones
usan incertidumbre de fidelidad, todo dividido por coste incremental observado. Las acciones que
no son anchors y aún no se despacharon son provisionales: nueva evidencia puede sustituirlas con
coste científico cero y registrar `CANCEL_PLANNED_DISPATCH` o `CANCEL_SCIENTIFIC_ACTION` según se
abandone solo el plan de despacho o también la acción científica. Un anchor inicial protegido
registra `DEFER_STARTUP_ANCHOR` y continúa como deuda científica. Startup es un diseño protegido
de cobertura, no una barrera:
puede empezar trabajo guiado por el modelo mientras quedan Runs iniciales lentos. Los candidatos
pendientes condicionan el surrogate en su fidelidad objetivo real, las identidades en cola evitan
seeds duplicadas y es legítimo esperar si ninguna acción aporta valor científico. La fidelidad es
entrada explícita del modelo.
Un modelo Beta/k-NN conjunto de supervivencia usa candidatos completados frente a podados con
incertidumbre; fallos de recursos y pausas del scheduler son neutrales y nunca se inventa un
objective completo para una curva parcial.
Una acción ya iniciada solo se pausa si posee checkpoint gestionado, ha superado un tiempo mínimo y
una alternativa nueva supera tanto su prioridad original como el valor de continuar por un margen
de histéresis del 50 %. La petición es cooperativa en una frontera segura, nunca mata el proceso;
confirmación y la cobertura startup sin score son inmunes. Los eventos `PREEMPT`, `PAUSE` y
`RESUME_PREEMPTED` conservan prioridades y motivo sin convertir la pausa en evidencia negativa.

El HPO adaptativo tiene dos objetivos simultáneos: encontrar configuraciones fuertes y aprender
cómo se comporta el espacio de búsqueda declarado. LambdaForge mantiene por ello dos cantidades
distintas. La **oportunidad de optimización** (`O`) es la mejora práctica restante predicha por el
surrogate conjunto; la **incertidumbre científica** (`K`) es la entropía aún no resuelta de
preguntas explícitas sobre parámetros e interacciones por pares. Ambas se normalizan con la
evidencia actual, producen pesos automáticos y se dividen por el coste incremental observado al
competir las acciones. No existe un cambio de fase por número de trials ni otro umbral de confianza
en YAML: la optimización vuelve a ganar prioridad si una observación hace plausible otra mejora.

Cada pregunta de parámetro termina en una conclusión estructurada: valor/región preferida,
equivalencia práctica, preferencia débil, plano, dependiente del contexto, sin preferencia clara o
no resuelta. Las preguntas por pares distinguen interacción material/débil, evidencia aditiva y no
resuelta. Aquí **confianza** es la fracción de realizaciones deterministas de evidencia, con
remuestreo por candidato y seeds compartidas, que reproduce exactamente la conclusión mostrada. No
es cobertura, tamaño de efecto, p-valor ni intervalo de confianza frecuentista. Por ello, con
evidencia suficiente tanto «plano» como «dependiente del contexto» pueden tener confianza alta. La
cobertura y el soporte condicional ausente se muestran aparte; los efectos son descriptivos o
predictivos, nunca causales.

La **región óptima práctica** contiene configuraciones observadas cuyo regret con incertidumbre es
compatible con el margen de equivalencia declarado. Indica qué parámetros están restringidos y
cuáles son flexibles dentro de ella. Tras el arranque, el controlador puede emitir un
`DESIGNED_PROBE`: un candidato válido no observado elegido para resolver un parámetro o interacción,
prefiriendo un contrafactual emparejado cuando el espacio condicional lo permite. Su decisión
registra pregunta, valor de rendimiento, reducción esperada de incertidumbre, calidad del match,
coste y alternativas. La carrera de seeds usa solo varianza dentro de cada candidato—nunca
dispersión entre candidatos—, replica el incumbent si su incertidumbre limita comparaciones y
prefiere una seed compartida aún no usada cuando mejora la comparación pareada. Un candidato
claramente inferior puede quedarse así con una seed y una comparación útil recibir más.

Cada Run adaptativo posee un proceso separado tanto en CPU como GPU. Una excepción normal falla
solo ese Run. Un worker perdido/matado o una OOM de asignación CUDA se reintenta hasta
`failure_retries` veces (1 por defecto) como Attempt nuevo y reutiliza checkpoints compatibles; si
se repite queda terminal sin entrar en un bucle. Errores de aplicación como datos o tensores
inválidos no se reintentan a ciegas. El resto de candidatos continúa, aunque el Work final conserva
estado fallido si algún Run agota la recuperación.

El workspace de un Study adaptativo incluye una pestaña **HPO**. Su tabla separa para cada
hiperparámetro el dominio declarado, el realmente observado, la región mejor respaldada, su
importancia y la confianza conservadora. Al abrir una fila aparecen la respuesta predictiva con
límites de incertidumbre del modelo, soporte observado, dispersión empírica y ganancias predictivas
conjuntas medidas frente a los demás parámetros. El área del controlador separa la última decisión
real `START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY`, fallback o
confirmación. Son asociaciones exploratorias marginales, no relaciones causales; el sampler
real aparece aparte como `SURROGATE BELIEF`, con backend, fidelidad objetivo, región predicha e
incertidumbre; una tendencia marginal nunca se presenta como creencia GP/k-NN. El sampler
conjunto multivariable sigue siendo la autoridad. Selecciona un parámetro y pulsa Enter/derecha para
abrir su curva de respuesta agrupada y el panel de relaciones por pares. Este último muestra la
ganancia predictiva leave-one-out conjunta frente al mejor predictor de un parámetro, medida en
desviaciones estándar del objective: es un diagnóstico visual de estructura conjunta, no
importancia causal. La respuesta descriptiva aparece desde dos candidatos comparables; con dos se
muestra también cobertura conjunta y la ganancia predictiva comienza con tres. Estos paneles
tempranos permanecen marcados con confianza baja. El observador recalcula localmente snapshots de
análisis antiguos, por lo que un Work remoto ya activo obtiene la vista nueva sin reiniciarse. Un
Study vivo no necesita un resultado final de Execution: su snapshot HPO persistido aporta filas de
parámetros, respuesta/soporte observados y confianza numérica y textual hasta que lo sustituye el
análisis final. La
consola muestra además las guardas explícitas y candidatos no factibles.
Automatización recibe puntos de respuesta, matriz de relaciones y acciones en
`work.items[].study.hpo_analysis`, `.controller` y `.surrogate_belief`.

Las gráficas de terminal siguen siendo compactas y fieles: las dimensiones categóricas conservan
etiquetas como `true` y `false`, y pulsar un punto o barra muestra sus coordenadas exactas. La curva
de respuesta de terminal omite una pseudobanda visualmente ambigua; la incertidumbre continúa como
dato numérico en la tabla. **Interactive HTML** exporta un informe Plotly offline con hover exacto,
banda real de incertidumbre, mapas de calor por pares y superficies 3D cuando ambos parámetros son
numéricos. El renderer opcional exige `lambdaforge[analysis-report]` y solo escribe tras una acción
explícita del usuario.

Métricas dependientes de threshold como F1, balanced accuracy, kappa de Cohen, accuracy, precision
y recall se usan exactamente como las registra el Work. LambdaForge nunca busca automáticamente un
threshold, elige uno distinto por candidato ni las considera equivalentes a AUROC/AUPRC. La política
de threshold pertenece al protocolo de evaluación del Work y debe ser comparable entre candidatos.

No es necesario leer un único stream mezclado cuando hay entrenos concurrentes. Un estudio no es
un tipo especial de Work: cualquier Work normal con `search` o varias `seeds` queda marcado
durante la validación local, por lo que la Consola abre su vista de estudio mientras continúa
la preparación remota. Permite avanzar como `Work → Trial (combinación de parámetros) → Run de seed → panel
vivo`. La pantalla de Trial distingue combinaciones pendientes, activas, promocionadas, eliminadas
o terminadas. Cada Run muestra la GPU asignada, sus parámetros y log aislado, los actualiza
automáticamente y dibuja
curvas acotadas junto con las últimas métricas, tiempo de época y tiempo de validación.
`LightningRunner` publica automáticamente sus métricas escalares de callback y esos tiempos. Un
loop propio usa la API genérica:

Un Work de preprocesado, un workflow con varios pasos o un Work que usa `self.map()` sigue siendo
ordinario salvo que su YAML declare realmente una búsqueda de parámetros o varias seeds.
Enter/derecha abre por tanto su Attempt numerado y el log combinado normal. No hay un “tipo de
entreno” manual que pueda quedar desincronizado.

```python
for epoch in range(epochs):
    train_loss, val_score = train_one_epoch(epoch)
    self.metrics.log("train_loss", train_loss, step=epoch + 1)
    self.metrics.log("val_auprc", val_score, step=epoch + 1)
    self.progress.update(epoch + 1, epochs, message="entrenando")
    self.log(f"época {epoch + 1} completada")  # narración humana opcional
```

`print`, el logging de Python y `self.log()` siguen siendo evidencia local a cada proceso, por lo
que el panel no mezcla las líneas de las demás seeds. Las mismas observaciones están disponibles
para automatización mediante `lf overview --json`, `lf show WORK --json`,
`lf show WORK --run trial-00001-seed-4 --json` y
`lf logs WORK --run trial-00001-seed-4 --tail 300`. El índice vivo conserva solo estado compacto y
JSONL escalar; referencia los logs/resultados existentes y reduce las curvas al leerlas, sin
duplicar checkpoints, modelos ni outputs pesados. La pestaña **Artifacts** de una seed y
`lf show WORK --run CLAVE` enumeran cada artefacto gestionado ya finalizado con su ruta utilizable
preferida, rol, tipo MIME y tamaño. El JSON añade SHA-256, rutas gestionada/publicada, retención y
metadatos. Un output `published-only` apunta a `publish_to`; los demás permanecen dentro del Run.

El acceso físico a GPU pertenece al perfil del clúster. `gpu_access.mode=auto` usa SLURM en un
clúster SLURM y leases exclusivos conservadores en hosts directos. Para un host deliberadamente
compartido o un centro con comando de claim:

```bash
lf clusters set gpu-libre gpu_access.mode shared
lf clusters set citius-gpu gpu_access '{mode: command, command_prefix: [gpu, exec]}'
```

`shared` admite ocupación externa pero coordina Jobs de LambdaForge entre sí; no equivale a una
reserva dura y la ocupación puede cambiar después del preflight. En CITIUS, `gpu exec` es el modo
preferido para una sola GPU porque su reserva dura exactamente lo que el comando. Para un Work
multi-GPU —o si el centro exige claim persistente— se declaran claim y release juntos. Es también
la alternativa reproducible a reclamar manualmente en otra shell de login:

```bash
lf clusters set citius-gpu gpu_access \
  '{mode: command, command_prefix: [gpu, exec], claim_command: [gpu, claim, --numgpus, "{gpu_count}"], release_command: [gpu, release]}'
```

El claim ocurre en la preparación durable en segundo plano y el supervisor ejecuta release tras
éxito, fallo o cancelación. SLURM rechaza claims persistentes: debe poseer la reserva el scheduler o
un wrapper autocontenido. Todos son argv, nunca shell. En modos `command` y `scheduler`,
`CUDA_VISIBLE_DEVICES` debe proceder del centro. LambdaForge conserva sus tokens opacos (índices,
UUID o MIG UUID), puede estrecharlos por hijo y jamás amplía o sustituye la asignación. Una lista
ausente o duplicada falla de forma segura y scheduler exige el grant exacto. Un Study adaptativo
tras un launcher command reduce su concurrencia si recibe menos GPU que el máximo solicitado: una
concedida se usa como una, dos como dos y nunca se inventa una tercera. Además, pedir GPU fuerza un
entorno CUDA; un fallo del launcher ya no puede degradar silenciosamente a PyTorch CPU.

El grant command puede cambiar durante un Study largo. LambdaForge consulta los tokens opacos
actuales mediante `gpu_access.visibility_command`; con el prefijo habitual `[gpu, exec]` deriva
automáticamente `[gpu, env]`. Solo admite que un token heredado desaparezca o se restaure, nunca
acepta identificadores físicos nuevos. Si el grant se reduce, detiene exclusivamente el worker
verificado de cada token revocado y reencola su Run lógica usando checkpoint; las demás continúan.
Si el probe de propiedad falla temporalmente, conserva las Runs activas pero no admite otras
nuevas. Un comando personalizado debe imprimir en su primera línea no vacía los tokens actuales
separados por comas.

## Observación y operación

```bash
lf                          # Consola de investigación interactiva
lf overview --json
lf show WORK
lf logs WORK --follow
lf retry WORK
lf delete WORK              # vista previa
lf delete WORK --apply
lf jobs clear               # vista previa del historial terminal
lf jobs clear --apply
lf datasets list
lf results list
lf clean                    # vista previa de limpieza segura
```

La Consola es la interfaz humana para Work, Studies, Clusters, Datasets y Results. Overview separa
Clusters, Work ordinarios y Studies en tres paneles: un Study no se mezcla con Work ni desaparece
al fallar. Al seleccionar un clúster aparecen historiales vivos de CPU/RAM/GPU, capacidad total,
recursos solicitados exactos y uso personal observado cuando el proveedor puede medirlo. El
progreso es texto breve, no JSON serializado. Las pantallas contextuales descubren
progresivamente Attempts, Runs, logs, admisión de recursos, evidencia de estudios y resultados.
`Ctrl+P` abre la paleta difusa, Enter abre la selección, Esc vuelve y `?` explica el contexto. Las
operaciones lentas se ejecutan fuera del event loop; una caída transitoria conserva el último dato
correcto marcado como obsoleto, sin convertirla en fallo científico. Solo la primera lectura usa
una pantalla de carga. Los sondeos posteriores conservan la vista y muestran abajo cuánto tiempo
pasó desde la última actualización correcta; tanto la tabla de estados de Work como sus logs
acotados se refrescan en vivo sin solapar peticiones. **Run Work** presenta validación y encolado
durable como una sola operación visible, también al elegir un YAML reciente. Las acciones
destructivas requieren confirmación y mantienen el preview y límites de propiedad de la CLI. La automatización
debe consumir los comandos `--json`, no analizar la interfaz a pantalla completa.

Las vistas de colección usan modelos de lectura deliberadamente ligeros. Overview hace una lectura
del inventario por proveedor directo —o solo del estado de Jobs activos si el scheduler no tiene
inventario—, consulta los contadores del registro local de datasets y
nunca transfiere los índices de candidatos/Runs de todos los Studies. Work y Studies tampoco
sondean recursos ni datasets. Abrir un Study carga su índice acotado; el análisis HPO, el historial
completo del controlador y los logs se solicitan solo al abrir sus pestañas. Curvas por epoch,
artefactos y logs aislados se leen únicamente al entrar en una seed. `lf overview --json` conserva
este contrato compacto; `lf show WORK` y `lf show WORK --run CLAVE --json` añaden los siguientes
niveles de detalle. Si el proveedor no permite confirmar un Job más allá de `unknown`, el borrado
ofrece una operación explícita solo sobre el historial local: no toca el proceso ni el workspace
remotos no verificados. Si el cómputo aún puede estar activo, primero hay que reconectar y cancelarlo.
El worker mantiene `study/interactive.json` como índice compacto separado del resumen rico
autoritativo; los resúmenes legacy demasiado grandes se proyectan en el host de ejecución y el
historial completo se descarga en páginas JSONL acotadas solo al abrir Action history. Los redraws
vivos conservan fila seleccionada, scroll de tablas/logs y viewport manual de cada gráfica.

`LightningRunner` registra curvas escalares, `epoch_time_s`, `validation_time_s`, el pico de
tensores vivos (`gpu_mem_mb`) y la caché del allocator de PyTorch (`gpu_reserved_mb` y
`gpu_peak_reserved_mb`). La memoria reservada es caché reutilizable: no es otra reserva de GPU de
LambdaForge ni significa que los tensores consuman esa cantidad. El empaquetado HPO continúa
gobernado por el umbral de admisión `resources.gpu_memory` por Run y la memoria libre que informa el
driver: ocupa solo slots seguros y deja los demás Runs en cola. Al terminar un Run empaquetado,
sale su proceso dedicado y libera el contexto CUDA completo antes de readmitir ese slot.
Si el probe CUDA efímero falla de forma transitoria, se pausa únicamente la admisión: las Runs ya
activas continúan, el sondeo se reintenta y se conserva su `stderr` acotado. Solo una pérdida
persistente de la asignación falla cuando ya no quedan Runs activas que proteger. El botón rojo
**Exit LambdaForge** siempre está visible y `q` sale desde las vistas raíz.

Se eligen las curvas visibles sin descartar las demás así:

```python
config = lf.training.LightningTrainConfig(
    epoch_console_include=["train_loss", "val_*", "*_time_s"],
    epoch_chart_include=["val_*", "epoch_time_s", "validation_time_s"],
    epoch_metric_display_names={"val_balanced_accuracy": "Exactitud equilibrada"},
    epoch_chart_exclude=["*_aux"],
)
```

`epoch_console_include`/`epoch_console_exclude` controlan la tabla humana por época;
`epoch_chart_include`/`epoch_chart_exclude` controlan independientemente las curvas interactivas.
Al cambiar de página de cuatro curvas, cada gráfica reutilizada recupera límites automáticos para
que una métrica con escala distinta sea visible de inmediato; el zoom/desplazamiento manual sigue
disponible dentro de la página.
`epoch_metric_display_names` cambia opcionalmente solo las etiquetas de la Consola: no altera las
claves reales, el lookup del objective ni la identidad científica persistida. Los nombres comunes
se humanizan automáticamente (`val_auprc` se muestra como `AUPRC`, `train_loss` como `Train loss` y
la utilidad compuesta interna como `Composite selection score`).

`lf cancel WORK` y `x` sobre un Work son cancelación semántica: se contacta cada Job activo agrupado
en ese Work, y cada supervisor directo detiene tanto su grupo de procesos verificado como procesos
reparentados o con sesión propia que conserven la identidad única del Job. Solo termina con éxito
cuando no queda ninguno. Cancelar un Attempt seleccionado o `lf jobs cancel JOB` mantiene el
alcance más estrecho de bajo nivel. En SLURM se delega la asignación completa al comando de
cancelación configurado. Repetir `lf cancel WORK` también reconcilia Jobs directos que ya figuren
como cancelados, limpiando con seguridad huérfanos dejados por versiones anteriores.

`lf clean` también presenta cache reconstruible y artefactos gestionados parciales o duplicados de
Attempts terminales que pueden eliminarse con prueba exacta. Nunca incluye un Job activo, un output
correcto no publicado, resultados, métricas, checkpoints ni datasets.

Los entornos gestionados obsoletos se eliminan automáticamente tras activar un reemplazo verificado,
salvo los referenciados por Jobs vivos. Así se conserva la inmutabilidad sin acumular un prefijo de
varios GB por cada identidad histórica.

La creación de datasets se realiza con `self.outputs.dataset(...)`; `lf datasets` inspecciona,
verifica, materializa o elimina versiones inmutables. La guía completa de YAML, estudios, clústeres,
metadata y seguridad está en [el manual](docs/MANUAL.es.md) ([English](docs/MANUAL.md)). Para agentes,
[AGENTS.es.md](AGENTS.es.md)
es el contrato compacto que evita recorrer todo el repositorio e inventar APIs.

## Análisis de estudios

La búsqueda adaptativa decide qué ejecutar después; el Análisis de estudios determina qué sostiene
la evidencia reunida. Al terminar, un estudio escribe automáticamente un `analysis.json` versionado
y atómico. También puede calcularse o actualizarse expresamente:

```bash
lf results analyze EXECUTION
lf results analyze EXECUTION --recompute
lf results analyze EXECUTION --json
python -m pip install "lambdaforge[analysis-report]==0.14.0"
lf results report EXECUTION --output study-report.html
lf results replay EXECUTION --policy ari-v3.1
lf results replay EXECUTION --policy ari-v2-compat --json
```

El replay de recursos lee la traza versionada del scheduler, nunca texto de terminal. Es factual
hasta que la política elegida toma una decisión distinta; desde ahí cada métrica se etiqueta como
simulación condicionada por la traza, y los resultados terminales propios de la rama real pasan a
`null` en vez de atribuirse a otra política. `recorded`, `ari-v3-compat` y `ari-v2-compat` permiten auditar
sin mantener schedulers antiguos en producción. Los Studies nuevos informan tiempo a cada nivel de
concurrencia, capacidad ociosa, tasa científica/útil, throughput, OOM, rollback, checkpoint,
starvation, calibración de P(fit) y error de predicción cuando existe evidencia. Un Study antiguo
sin traza sigue siendo legible, pero no puede prometer replay contrafactual.

Todo el HPO comparte un único `ParameterSpace` authored. `scale: log` usa coordenadas logarítmicas
en Sobol, sampler bayesiano y fallback, similitud de recursos, diseño científico y análisis final;
integer, categorical e inactividad condicional también significan lo mismo. Los diccionarios de
candidatos y sus identidades científicas no cambian.

El análisis distingue cuatro conceptos. `current_observed_objective` es el último paso con todos los
componentes necesarios; `best_observed_objective`, la mejor observación completa;
`final_objective` solo existe para una Run terminal a fidelidad completa; y
`selection_objective` es la cantidad agregada por seeds usada para seleccionar. Una Run podada
conserva su curva parcial y mejor observación como evidencia censurada, pero nunca recibe un score
final inventado. Los componentes que falten en un objetivo compuesto se indican explícitamente.

`analysis.json` incluye incertidumbre, comparaciones pareadas, screening/confirmación separados,
validación leave-one-candidate-out real, importancia funcional global y en la región superior
observada, respuestas ajustadas, interacciones, resolución marginal/conjunta observada, bordes,
pruning, constraints y dos Pareto distintos. Una seed es evidencia insuficiente para estabilidad
empírica; la incertidumbre modelada se informa aparte cuando fue persistida. La fiabilidad combina
soporte, calidad de validación, cobertura y extrapolación. La confirmación reutiliza el margen de
equivalencia científico y la eficiencia por Run comparable nunca se mezcla con el gasto total del
controlador. Son resúmenes predictivos observacionales, no causales. En vivo son `PROVISIONAL`; con
evidencia terminal son `FINAL`.

El informe Plotly opcional es autocontenido y funciona offline. Plotly no forma parte de las
dependencias base: sin el extra siguen funcionando la ejecución, el JSON y la Consola.

La admisión de recursos también queda estructurada: capacidad GPU/CPU/RAM, Runs activas/en cola,
VRAM libre y requerida por GPU y motivo concreto de espera. `runs_per_gpu` es un máximo, no una
exigencia; dos GPU capaces de admitir cinco Runs cada una pueden ejecutar diez, mientras los slots
sin capacidad esperan y se vuelven a evaluar de forma escalonada sin tumbar el estudio.

## Consola de investigación

`lf` sin argumentos abre en un TTY la Consola de investigación basada en Textual 8.2. La instalación
base incluye `textual-plot`, un widget nativo de Textual: el historial de recursos y las curvas de
aprendizaje tienen marcas legibles, líneas Braille de alta resolución, claves de color y
zoom/desplazamiento por teclado o ratón sin mantener un segundo motor gráfico propio. Sus seis
destinos son Overview, Work, Studies, Clusters, Datasets y Results. Enter/derecha abre la entidad,
Esc/izquierda o el botón visible **Back to …** retrocede exactamente un nivel y Home vuelve a la
raíz; además, cada segmento anterior de las migas de pan es pulsable. Las vistas remotas de Study y
Seed muestran un estado de carga centrado hasta recibir su snapshot persistido y un error explícito
si no se puede recuperar, en vez de hacer pasar una tabla vacía por evidencia. El recorrido
científico es:

```text
Studies → Study → Trials → Trial → Seeds → Seed → Curves / Epochs / Logs
```

La vista de seed separa current, best, final y selection; muestra GPU/recursos; pagina las curvas de
cuatro en cuatro con botones anterior/siguiente y un selector numerado aptos para ratón (`n`/`p`
siguen como atajos); marca en rojo la época seleccionada y en verde la mejor; y coloca todos los
escalares directamente en la tabla de épocas con desplazamiento horizontal. Una Run podada conserva
con `†` su mejor evidencia censurada, pero no
recibe un final inventado. Analysis ofrece Summary, Parameters, Interactions, Coverage, Seeds,
Pareto y Findings sobre el mismo `analysis.json` de CLI e informe HTML. Coverage usa tarjetas
resumen, gráfica marginal exacta y tabla de niveles/rangos observados frente a declarados, bins y
soporte en bordes; describe candidatos muestreados, no atribuye cobertura a un pool no observado.
Pulsar puntos o barras muestra valores exactos. Los controles contextuales `?` explican objetivo,
confianza, fiabilidad, efectos de región superior, ganancia predictiva, cobertura y parada. Los
controles **Interactive HTML** abren curvas, informes de Study o respuesta/interacciones de un
parámetro, además del historial acotado de recursos del clúster, en Plotly autocontenido y de alta
resolución cuando está instalado el extra opcional.
La estabilidad de seeds se resume en tarjetas, los Pareto científico/de recursos en tablas exactas
separadas y los findings en una tabla de severidad/fiabilidad con recomendación legible.
Las superficies HPO de respuesta e interacción indican y usan siempre el objetivo de selección
declarado del Study. La UI no ofrece un cambio arbitrario de métrica que pudiera confundirse con el
modelo del controlador; los componentes de un objetivo compuesto siguen visibles en la tabla de
Pareto científico.

El Overview del Study combina estado, candidatos/Runs, líder actual y tiempo de cómputo con
gráficas de objective por trial y estados, además de los parámetros líderes. Los conteos exactos se
muestran bajo las barras, por lo que los grupos pequeños siguen siendo legibles en escalas grandes.
**HPO** separa tarjetas
resumen, tabla de evidencia por parámetro e historial completo de acciones persistidas. Este último
usa siempre Acción / Trial / Motivo; abrir una acción expone los umbrales, prioridades y evidencia
exactos guardados por el controlador. Las decisiones nuevas usan un historial append-only separado
para mantener pequeño el snapshot vivo; los Studies anteriores sin él muestran su tail persistido
disponible. Las marcas `★` mejor, `◆` Pareto y `†` censurado tienen su
propia columna y nunca tapan el número de trial.

`Ctrl+P` solo anuncia operaciones con handler real. Las restantes se etiquetan como solo-CLI hasta
que exista el flujo completo de formulario/preview/servicio/refresh; un nombre en un inventario no
se considera paridad. Las acciones contextuales llaman servicios Python, nunca un subproceso `lf`,
y todo borrado muestra un preview exacto antes de confirmar. **Run Work…** ofrece un explorador de
proyecto filtrado a YAML y una lista acotada construida con el historial local de Jobs existente y
las nuevas selecciones MRU validadas. Explorar o seleccionar una fila reciente solo rellena el
mismo selector YAML. Una única acción **Submit Work** valida y explica después el plan y solo lo
envía si es válido, al clúster seleccionado visible. Durante este flujo atómico deshabilita los
controles y muestra la fase activa, evitando envíos duplicados por clics repetidos. Usa el mismo
servicio durable/asíncrono que `lf run`. El estado MRU solo guarda rutas locales y
metadata de presentación adicional, nunca contenido YAML, credenciales ni inputs; los ficheros ausentes dejan
de mostrarse. Un YAML inválido permanece en el diálogo y se explica una sola vez con fichero,
línea/columna exactas, fragmento acotado e indicación correctiva. Las contraseñas quedan fuera del
YAML.

La CLI sigue siendo la interfaz autoritativa para scripts. `lf --help` enumera comandos estables; sin
comando y sin TTY imprime esa ayuda y sale. Al migrar desde 0.13, sustituye `lf top`,
`lf clusters setup` y `lf clusters modify` por `lf`.

Overview conserva exactamente el panel y la entidad seleccionados al refrescar datos. La barra
lateral separa **Action**, navegación **Browse** y la salida de **Session**, evitando presentar
acciones puntuales como pestañas. Clusters y Datasets usan tarjetas compactas, tablas semánticas y
secciones acotadas bajo demanda en vez de JSON crudo; tanto la lista de Clusters como el workspace
de cada clúster reutilizan las mismas gráficas CPU/RAM/GPU de Overview. La consola de actividad del
clúster transmite fases y tiempo transcurrido de bootstrap; permite elegir tamaño
Compact/Comfortable/Large o arrastrar el separador. La zona superior se convierte en un viewport
independiente con scroll y altura mínima útil, por lo que ampliar la salida nunca vuelve
inaccesibles sus acciones, pestañas o evidencia. Serializa las operaciones, reutiliza el
transporte y pausa los probes de recursos hasta terminar. Aplicar bootstrap exige además una
confirmación exacta presentada mediante secciones legibles de objetivo, cambios y elementos
preservados, no como JSON crudo. El mismo renderer acotado se usa para todas las confirmaciones de
mutación y previews modales de Work. Plan sigue siendo de solo lectura. Las pantallas raíz ocultas tampoco hacen
lecturas remotas. **Add cluster** y **Edit** comparten un editor completo por pestañas. Permite
cambiar identidad/autenticación, backend, raíces de proyecto/datos, Python gestionado, política
PyTorch/CUDA, retención de caché, reutilización SSH y todos los campos de `gpu_access` (modo,
prefijo, claim y release). Los mapas poco habituales de OpenSSH y del dialecto SLURM permanecen
editables en Advanced YAML validado y nunca se pierden silenciosamente. Al guardar se reconstruye
`ClusterProfile`: una combinación backend/GPU inválida queda explicada y no se escribe. Las
contraseñas no son campos del perfil; solo se editan referencias `keyring:`/`env:` y
**Credentials** gestiona el secreto. Un Study sin telemetría accesible continúa abriendo como Study y conserva sus
logs; la ausencia se muestra como evidencia no disponible, nunca cerrando la consola. **Cancel
Study** está disponible en esa vista degradada desde que existe el Work semántico: detiene todos
sus Attempts activos y Runs descendientes aunque el worker aún no haya publicado el primer
snapshot. Los fallos de proveedor durante el sondeo quedan como estado obsoleto/error inline y
nunca generan notificaciones emergentes repetitivas. **Delete Study History…** previsualiza y
elimina después el estado terminal exacto del Study/Attempt mediante su `work_id`; un Study activo
debe cancelarse primero, y no toca datasets publicados, entornos compartidos ni ejecuciones
homónimas ajenas.

Summary de Dataset carga conteos exactos de splits y del target principal desde el índice lógico,
sin recorrer árboles pesados de assets. Members carga automáticamente una página acotada. Stats
físicas e Integrity siguen siendo acciones explícitas dentro de su pestaña porque pueden ser
costosas. La acción **Delete DatasetVersion…**, siempre visible, informa inmediatamente del
progreso de preview/aplicación, previsualiza todas las ubicaciones y exige confirmación antes de
borrar la DatasetVersion completa. Mientras está activa bloquea nuevas pulsaciones. Si una carpeta fue borrada manualmente, limpia el registro obsoleto sin
fallar ni dejar una versión lógica vacía. Las Runs podadas muestran «not final · pruned» o «not
observed» para ausencias esperadas, no el ambiguo «unavailable».

Las publicaciones remotas nuevas usan la raíz efectiva del proyecto actual, incluido
`<raíz-datasets-configurada>/projects/<project-id>`. Una colocación antigua verificada puede seguir
funcionando en su ruta sin scope registrada explícitamente: es evidencia durable, no una búsqueda
global implícita. La última carpeta hash representa la identidad del contenido. Si el preprocesado
cambia cualquier byte de los assets, debe publicarse una versión nueva en vez de reutilizar
`NOMBRE@VERSION`. `lf datasets reconcile NOMBRE@VERSION --on CLUSTER` previsualiza una reparación
solo de índices; añade `--apply` tras revisarla. Una colocación conflictiva existente o inaccesible
nunca se elimina automáticamente. LambdaForge no retransmite silenciosamente un dataset remoto
grande a través del controlador: la colocación entre dos remotos usa el servicio durable de
transferencia del centro y después `reconcile`, cuando ya existe allí el directorio exacto con su
manifiesto.

El nombre de un Work es una etiqueta, no su identidad. El mismo YAML puede ejecutarse a la vez en
local y en uno o más clústeres; las tablas y acciones destructivas usan el `work_id` exacto, por lo
que nombres iguales siguen separados y cancelar/borrar no puede apuntar silenciosamente a otra ejecución.
