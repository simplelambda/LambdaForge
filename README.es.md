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

## Instalación

El proyecto científico y LambdaForge son paquetes independientes instalados en el entorno del
proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge==0.13.0
python -m pip install -e .
python -m pip check
```

`lf init mi-estudio` genera un proyecto instalable completo. Se requiere Python 3.10 o posterior.

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
durable que puede seguirse con `lf top` o `lf logs`. `--wait-for-submit` espera expresamente a la
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
python -m pip install "lambdaforge[clustering]==0.13.0"
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
crea Runs independientes; `search` expande variantes y `objective` selecciona la métrica escalar
exacta registrada por el Work. Con objective y varias seeds, la estrategia por defecto es halving
adaptativo: primero asigna `min_seeds` a cada candidato, promociona solo la fracción prometedora y
ordena mediante media y error estándar conservador, nunca por la seed más afortunada. Usa
`strategy: exhaustive` cuando necesites todas las parejas variante/seed.

```yaml
name: entreno-adaptativo
run: mi_proyecto.Training
seeds: [4, 7, 32, 54, 65, 94, 109, 124]
search:
  strategy: adaptive
  trials: 40
  min_seeds: 1
  reduction_factor: 2
  runs_per_gpu: 4
  early_stopping: {enabled: true, min_step: 5}
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
exige `gpu_memory`; LambdaForge comprueba `runs_per_gpu × gpu_memory` contra memoria actualmente
libre antes de lanzar hijos. En CPU puede limitarse con `max_parallel`.
`self.metrics.log("val_auprc", valor, step=epoch)` permite parar Runs poco prometedores;
`LightningRunner` enlaza automáticamente la métrica de validación y la petición de parada. Un loop
propio debe registrar `step=` y retornar en un checkpoint seguro al detectar `self.stop_requested`.
Con una métrica solo final se adaptan seeds, pero no puede pararse el entreno actual.

No es necesario leer un único stream mezclado cuando hay entrenos concurrentes. Un estudio no es
un tipo especial de Work: cualquier Work normal con `search` o varias `seeds` queda marcado
durante la validación local, por lo que `lf top` abre su vista de estudio incluso mientras continúa
la preparación remota. Permite avanzar como `Work → Trial (combinación de parámetros) → Run de seed → panel
vivo`. La pantalla de Trial distingue combinaciones pendientes, activas, promocionadas, eliminadas
o terminadas. Cada Run muestra solo sus parámetros y log, los actualiza automáticamente y dibuja
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
lf clusters set gpu-claim gpu_access '{mode: command, command_prefix: [gpu, run, --]}'
```

`shared` admite ocupación externa pero coordina Jobs de LambdaForge entre sí; no equivale a una
reserva dura y la ocupación puede cambiar después del preflight. El prefijo `command` es argv, no
un fragmento de shell. Los recursos YAML siguen siendo la reserva externa absoluta.

## Observación y operación

```bash
lf top
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

En `lf top`, arriba/abajo recorren clústeres y Works como una sola lista sin mostrar IDs operativos
largos en la ruta principal. Enter o derecha entra en Trials, Runs de seed, curvas y log vivo
aislado cuando el Work es un estudio. Un Work normal avanza a `Attempt 1`, `Attempt 2`, etc. y a su
log completo; `a` abre esos Attempts externos desde un estudio. El detalle de clúster usa las mismas
etiquetas y la izquierda vuelve. El log abierto se actualiza automáticamente, sigue el final por
defecto y conserva el scroll manual.
Los fallos añaden la excepción científica estructurada aunque `--tail` haya recortado las líneas;
`--verbose`/`--debug` incluye traceback y `--json` devuelve el fallo y ruta exacta. El bootstrap
humano muestra fases y latidos por stderr sin contaminar JSON. Los IDs siguen disponibles en
`lf jobs` y `lf overview --json`. `d` elimina el Work/Attempt terminal
seleccionado tras confirmación y `D` limpia todo el historial terminal, conservando siempre los Jobs
activos. Se eliminan únicamente sus workspaces y registros propios; datasets, caches y entornos
compartidos se preservan.

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
