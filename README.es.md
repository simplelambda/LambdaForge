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
scheduler y otra vez en el worker. Un mirror ausente o desactualizado falla de forma segura.
Sincronizarlo mediante el servicio recomendado por el clúster es responsabilidad explícita del
investigador. Para cientos de GB/TB es preferible un dataset gestionado: verificar el mirror también
lee todos los bytes, mientras el dataset aporta identidad y placements reutilizables.

Los envíos son asíncronos tanto en local como en remoto: la terminal vuelve tras crear un Job
durable que puede seguirse con la Consola o `lf logs`. `--wait-for-submit` espera expresamente a la
preparación y al scheduler, no al cálculo científico; `--dry-run` sigue siendo directo y sin efectos.

## Servicios y operación

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
nuevas. Un pool Sobol scrambled proporciona puntos
reproducibles que cubren el espacio; tras `startup_trials`, los resultados eligen los candidatos
siguientes. `sampler: auto` usa qLogNEI con GP mixto de BoTorch si está instalado el extra
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
congelado. Por defecto LambdaForge empieza con hasta tres seeds declaradas por candidato y genera
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
  startup_trials: 10
  seed_racing: {probability_threshold: 0.1, equivalence_margin: 0.002}
  confirmation_top_k: 2
  confirmation_seeds: [1001, 1002, 1003]
  sampler: auto
  max_runs: 180
  max_time: 12h
  convergence_patience: 8
  min_improvement: 0.0005
  runs_per_gpu: 4
  failure_retries: 1
  early_stopping: {enabled: true, min_step: 5, confirmations: 2,
                   probability_threshold: 0.05, equivalence_margin: 0.002}
  learning_rate: {range: [0.00001, 0.003], scale: log}
  hidden_dim: {values: [64, 128, 256]}
objective: {metric: val_auprc, mode: max}
resources:
  gpu: 2
  gpu_memory: 16GiB  # límite por Run independiente
  cpu: 16            # reserva total repartida entre Runs activos
  memory: 32GiB      # reserva total repartida entre Runs activos
```

El ejemplo permite ocho entrenos simultáneos: cuatro en cada una de dos GPUs. Seis GPUs con
`runs_per_gpu: 2` permiten doce, y `runs_per_gpu: 1` da aislamiento uno-a-uno. Empaquetar más de uno
exige `gpu_memory`, que es el umbral mínimo de VRAM libre para admitir cada Run nuevo, no una
obligación de iniciar inmediatamente la concurrencia máxima. LambdaForge sondea todas las GPU
asignadas, lanza donde quepa y mantiene el resto en cola. Si la VRAM está ocupada vuelve a sondear y
separa temporalmente lanzamientos sobre el mismo dispositivo para dejar materializar la asignación
anterior. Un presupuesto conservador por oleada reserva además el umbral declarado para Runs activos
aunque su asignación CUDA todavía no sea visible; solo se recalcula al quedar esa GPU sin Runs de
LambdaForge. Cada Run empaquetado posee un proceso spawn nuevo que termina al acabar el Run; no se
reutiliza un worker CUDA ocioso cuyo contexto podría retener VRAM y bloquear para siempre la cola.
Una GPU llena no falla todo el estudio; si otra admite dos de cuatro slots, ejecuta
dos. Solo se rechaza como imposible un umbral mayor que la memoria total de todas las GPU asignadas.
En CPU puede limitarse la concurrencia con `max_parallel`. El observador de memoria usa un proceso
hijo efímero, así que no deja un contexto CUDA ocioso en cada dispositivo ni consume una plaza
científica de `runs_per_gpu`.
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
usan incertidumbre de fidelidad, todo dividido por coste incremental observado. Las acciones en
cola aún no despachadas son provisionales: nueva evidencia puede sustituirlas con coste científico
cero y registrar `CANCEL_QUEUED_ACTION` con prioridad anterior/nueva y motivo. Startup es una cola
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

Cada Run adaptativo posee un proceso separado tanto en CPU como GPU. Una excepción normal falla
solo ese Run. Un worker perdido/matado o una OOM de asignación CUDA se reintenta hasta
`failure_retries` veces (1 por defecto) como Attempt nuevo y reutiliza checkpoints compatibles; si
se repite queda terminal sin entrar en un bucle. Errores de aplicación como datos o tensores
inválidos no se reintentan a ciegas. El resto de candidatos continúa, aunque el Work final conserva
estado fallido si algún Run agota la recuperación.

Los estudios adaptativos añaden una consola HPO accesible con `i`. Compara cada hiperparámetro con
el objective por candidato y muestra cobertura, dirección o posible umbral numérico, contraste
categórico, efecto estandarizado, confianza conservadora y qué evidencia convendría obtener
después. La cabecera separa la última decisión real `START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY`, fallback o
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
análisis antiguos, por lo que un Work remoto ya activo obtiene la vista nueva sin reiniciarse. La
consola muestra además las guardas explícitas y candidatos no factibles.
Automatización recibe puntos de respuesta, matriz de relaciones y acciones en
`work.items[].study.hpo_analysis`, `.controller` y `.surrogate_belief`.

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
duplicar checkpoints, modelos ni outputs pesados.

El acceso físico a GPU pertenece al perfil del clúster. `gpu_access.mode=auto` usa SLURM en un
clúster SLURM y leases exclusivos conservadores en hosts directos. Para un host deliberadamente
compartido o un centro con comando de claim:

```bash
lf clusters set gpu-libre gpu_access.mode shared
lf clusters set citius-gpu gpu_access '{mode: command, command_prefix: [gpu, exec]}'
```

`shared` admite ocupación externa pero coordina Jobs de LambdaForge entre sí; no equivale a una
reserva dura y la ocupación puede cambiar después del preflight. En CITIUS, `gpu exec` es el modo
preferido porque su reserva dura exactamente lo que el comando. Si el centro exige claim
persistente se declaran claim y release juntos:

```bash
lf clusters set citius-gpu gpu_access \
  '{mode: command, command_prefix: [gpu, exec], claim_command: [gpu, claim, --numgpus, "{gpu_count}"], release_command: [gpu, release]}'
```

El claim ocurre en la preparación durable en segundo plano y el supervisor ejecuta release tras
éxito, fallo o cancelación. SLURM rechaza claims persistentes: debe poseer la reserva el scheduler o
un wrapper autocontenido. Todos son argv, nunca shell. En modos `command` y `scheduler`,
`CUDA_VISIBLE_DEVICES` debe proceder del centro. LambdaForge conserva sus tokens opacos (índices,
UUID o MIG UUID), puede estrecharlos por hijo y jamás amplía o sustituye la asignación; una lista
ausente, duplicada o menor de lo pedido falla de forma segura. Los recursos YAML siguen siendo la
reserva externa absoluta.

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

La Consola es la interfaz humana para Work, Studies, Clusters, Datasets y Results. Overview resume
qué está ejecutándose, esperando o necesita atención; las pantallas contextuales descubren
progresivamente Attempts, Runs, logs, admisión de recursos, evidencia de estudios y resultados.
`Ctrl+P` abre la paleta difusa, Enter abre la selección, Esc vuelve y `?` explica el contexto. Las
operaciones lentas se ejecutan fuera del event loop; una caída transitoria conserva el último dato
correcto marcado como obsoleto, sin convertirla en fallo científico. Las acciones destructivas
requieren confirmación y mantienen el preview y límites de propiedad de la CLI. La automatización
debe consumir los comandos `--json`, no analizar la interfaz a pantalla completa.

`LightningRunner` registra curvas escalares, `epoch_time_s`, `validation_time_s`, el pico de
tensores vivos (`gpu_mem_mb`) y la caché del allocator de PyTorch (`gpu_reserved_mb` y
`gpu_peak_reserved_mb`). La memoria reservada es caché reutilizable: no es otra reserva de GPU de
LambdaForge ni significa que los tensores consuman esa cantidad. El empaquetado HPO continúa
gobernado por el umbral de admisión `resources.gpu_memory` por Run y la memoria libre que informa el
driver: ocupa solo slots seguros y deja los demás Runs en cola. Al terminar un Run empaquetado,
sale su proceso dedicado y libera el contexto CUDA completo antes de readmitir ese slot. Se eligen las
curvas visibles sin descartar las demás así:

```python
config = lf.training.LightningTrainConfig(
    epoch_console_include=["train_loss", "val_*", "*_time_s"],
    epoch_chart_include=["val_*", "epoch_time_s", "validation_time_s"],
    epoch_chart_exclude=["*_aux"],
)
```

`epoch_console_include`/`epoch_console_exclude` controlan la tabla humana por época;
`epoch_chart_include`/`epoch_chart_exclude` controlan independientemente las curvas interactivas.

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
```

El análisis distingue cuatro conceptos. `current_observed_objective` es el último paso con todos los
componentes necesarios; `best_observed_objective`, la mejor observación completa;
`final_objective` solo existe para una Run terminal a fidelidad completa; y
`selection_objective` es la cantidad agregada por seeds usada para seleccionar. Una Run podada
conserva su curva parcial y mejor observación como evidencia censurada, pero nunca recibe un score
final inventado. Los componentes que falten en un objetivo compuesto se indican explícitamente.

`analysis.json` incluye incertidumbre por candidato/seed e intervalos bootstrap deterministas cuando
hay soporte, comparaciones pareadas por seed, screening separado de confirmación, validación cruzada
del surrogate a nivel candidato, importancia funcional global y en la región superior, respuestas
ajustadas, interacciones/superficies por pares, cobertura marginal y conjunta, saturación de bordes,
resolución del pool, estabilidad del ganador, calidad del pruning, constraints, Pareto de componentes
y Pareto de eficiencia de recursos. Con poca evidencia comunica la limitación en vez de producir una
conclusión aparentemente precisa. Son resúmenes predictivos observacionales, no afirmaciones
causales. Durante la ejecución son `PROVISIONAL`; con evidencia terminal son `FINAL`.

El informe Plotly opcional es autocontenido y funciona offline. Plotly no forma parte de las
dependencias base: sin el extra siguen funcionando la ejecución, el JSON y la Consola.

La admisión de recursos también queda estructurada: capacidad GPU/CPU/RAM, Runs activas/en cola,
VRAM libre y requerida por GPU y motivo concreto de espera. `runs_per_gpu` es un máximo, no una
exigencia; dos GPU capaces de admitir cinco Runs cada una pueden ejecutar diez, mientras los slots
sin capacidad esperan y se vuelven a evaluar de forma escalonada sin tumbar el estudio.

## Consola de investigación

`lf` sin argumentos abre en un TTY la Consola de investigación basada en Textual 8.2. Sus seis
destinos son Overview, Work, Studies, Clusters, Datasets y Results. La paleta `Ctrl+P` ofrece la
contrapartida interactiva de cada familia CLI y llama directamente a servicios Python, nunca a un
subproceso `lf`. Studies distingue evidencia incompleta/censurada, muestra admisión y enlaza el
análisis persistido; Results analiza o exporta exactamente la misma evidencia que la CLI. Los
formularios de clúster empiezan por decisiones humanas y no guardan contraseñas en configuración.

La CLI sigue siendo la interfaz autoritativa para scripts. `lf --help` enumera comandos estables; sin
comando y sin TTY imprime esa ayuda y sale. Al migrar desde 0.13, sustituye `lf top`,
`lf clusters setup` y `lf clusters modify` por `lf`.
