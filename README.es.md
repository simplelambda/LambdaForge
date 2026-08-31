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

La ayuda funciona tanto en forma convencional como natural y termina correctamente incluso cuando
otra aplicación invoca directamente `CommandLineInterface.main()`:

```bash
lf --help
lf help
lf run --help
lf help clusters add
lf clusters setup --help
```

Para uso interactivo, `lf clusters setup` abre un asistente explicado para conexión,
credenciales, rutas, storage, Python, PyTorch, scheduler y política GPU. `lf clusters modify`
permite seleccionar y editar un perfil existente, incluidos los campos avanzados. Es solo una
capa humana sobre `clusters add/set/unset/credentials/test`; automatización y wrappers deben usar
esos comandos nativos. Las contraseñas nunca entran en argv ni YAML. En una terminal real, las
flechas recorren opciones, la opción enfocada explica sus consecuencias y Enter la selecciona; el
fallback numerado imprime la misma ayuda al redirigir la terminal. `0`, `q`, `quit` o `exit`
permite salir desde cualquier pregunta sin aplicar esa respuesta. El backend de ejecución y el
acceso GPU son conceptos separados: elige Direct si el sitio lanza procesos normales en el host
(también si los envuelve con `gpu exec`) y SLURM solo cuando realmente usa `sbatch`; el wrapper GPU
se configura en la pregunta posterior de acceso GPU.

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
exacta registrada por el Work. Siempre que `search` tenga `objective`, omitir `strategy` activa la
política adaptativa segura completa: inicio Sobol, propuestas dependientes de resultados, carrera
probabilística de seeds, pruning de curvas, detección de convergencia y confirmación con seeds
nuevas. Un pool Sobol scrambled proporciona puntos
reproducibles que cubren el espacio; tras `startup_trials`, los resultados eligen los candidatos
siguientes. `sampler: auto` usa qLogNEI con GP mixto de BoTorch si está instalado el extra
`lambdaforge[adaptive-hpo]` y existe evidencia suficiente, con fallback k-NN determinista ante una
dependencia ausente o un fallo numérico. El GP ve conjuntamente todas las dimensiones codificadas,
incluidas categorías y activación condicional, y qLogNEI incorpora el error estándar entre seeds;
los paneles por parámetro son explicaciones marginales, no el modelo que decide. Solo aparecen en
`lf top` los Trials ya propuestos.

El objetivo primario sigue siendo un escalar auditable. Si un valor alto puede resultar engañoso
por sí solo, declara guardas de resultado explícitas en vez de esperar que LambdaForge adivine qué
otras métricas importan. Cada límite se evalúa en la mejor época del objetivo primario; la
factibilidad del candidato usa después la media entre seeds de esos valores de la misma época. La
evidencia ausente falla cerrado y los candidatos no factibles siguen visibles, pero no guían el
surrogate ni pueden ganar:

```yaml
name: entreno-con-guardas
run: mi_proyecto.Training
objective:
  metric: val_auprc
  mode: max
  constraints:
    val_accuracy: {min: 0.55}
    val_kappa: {min: 0.05}
```

Esto es optimización restringida de un único objetivo, no un compromiso multiobjetivo implícito.
Los umbrales deben expresar validez científica real; no añadas toda métrica registrada solo porque
exista.

El número de seeds es probabilístico, no igual ni fijado por rondas. Un orden compartido permite
diferencias pareadas; se añade una seed solo mientras la probabilidad de estar a
`equivalence_margin` del incumbent alcance `seed_probability_threshold`. El ganador usa una cota
conservadora de búsqueda o, preferiblemente, la media de `confirmation_seeds` nuevas sobre un top-K
congelado. Por defecto LambdaForge empieza con hasta tres seeds declaradas por candidato y genera
tres seeds de confirmación deterministas y nuevas; cada valor puede sobrescribirse expresamente.

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
  min_seeds: 1
  startup_trials: 10
  seed_probability_threshold: 0.1
  equivalence_margin: 0.002
  confirmation_top_k: 2
  confirmation_seeds: [1001, 1002, 1003]
  sampler: auto
  max_runs: 180
  max_time: 12h
  convergence_patience: 8
  min_improvement: 0.0005
  runs_per_gpu: 4
  failure_retries: 1
  early_stopping: {enabled: true, min_step: 5, confirmations: 2}
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
`hpo-control/decisions.jsonl` con cada `START_NEW`, `ADD_SEED`, `RESUME`, fallback, convergencia y
confirmación; el resumen del resultado enlaza ambos ficheros.

La planificación es asíncrona de forma conservadora. Si termina un Run mientras el resto del lote
de adquisición continúa, el controlador puede rellenar el slot libre desde el posterior actualizado
sin esperar al más lento. Solo admite uno o dos candidatos de anticipación según la concurrencia, y
BoTorch condiciona qLogNEI en todos los candidatos pendientes. Así limita decisiones obsoletas y el
crecimiento de cola. No cancela un Run solo porque un posterior actualizado lo ordene peor; únicamente
la política normal probabilística/de fidelidad puede podarlo con seguridad. Cada lanzamiento queda
auditado como `START_NEW` con motivo `bounded-async-lookahead`.

Cada Run adaptativo posee un proceso separado tanto en CPU como GPU. Una excepción normal falla
solo ese Run. Un worker perdido/matado o una OOM de asignación CUDA se reintenta hasta
`failure_retries` veces (1 por defecto) como Attempt nuevo y reutiliza checkpoints compatibles; si
se repite queda terminal sin entrar en un bucle. Errores de aplicación como datos o tensores
inválidos no se reintentan a ciegas. El resto de candidatos continúa, aunque el Work final conserva
estado fallido si algún Run agota la recuperación.

Los estudios adaptativos añaden una consola HPO accesible con `i`. Compara cada hiperparámetro con
el objective por candidato y muestra cobertura, dirección o posible umbral numérico, contraste
categórico, efecto estandarizado, confianza conservadora y qué evidencia convendría obtener
después. La cabecera separa la última decisión real `START_NEW`, `ADD_SEED`, `RESUME`, fallback o
confirmación. Son asociaciones exploratorias marginales, no relaciones causales; el sampler
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
`work.items[].study.hpo_analysis` y `.controller`.

No es necesario leer un único stream mezclado cuando hay entrenos concurrentes. Un estudio no es
un tipo especial de Work: cualquier Work normal con `search` o varias `seeds` queda marcado
durante la validación local, por lo que `lf top` abre su vista de estudio incluso mientras continúa
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
lf top                      # historial de recursos de 60 segundos
lf top --history 180        # conserva tres minutos en las gráficas vivas
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

En `lf top`, arriba/abajo recorren clústeres y Works como una sola lista con color semántico y sin
mostrar IDs operativos largos en la ruta principal. Respeta `NO_COLOR` y salidas no interactivas.
Enter o derecha entra en Trials, Runs de seed y el panel de entreno. El resumen añade un historial
compacto de CPU/RAM/GPU por clúster; su detalle dibuja series temporales enmarcadas de CPU, RAM,
utilización GPU y memoria GPU, y `--history` controla la ventana.

Las pantallas de estudio no vuelcan JSON ilegible. Cada Trial separa mejor objective y su
seed/época de la media actual de las seeds observadas. HPO clasifica un Trial terminado con la
media del mejor checkpoint de cada seed, no con una época final sobreajustada; las seeds nuevas de
confirmación protegen la elección final de un checkpoint afortunado. Cada fila de seed muestra
objective actual y mejor, última y mejor época y GPU. La tabla reserva siempre varias filas de seed
en una terminal normal y el preview acotado de métricas continúa en el panel del Run. `pruned` es
una parada temprana cooperativa terminal, no un fallo ni una pausa, y al seleccionar la seed se
muestra su motivo probabilístico. Las curvas parciales podadas permanecen visibles como evidencia
censurada y `lf top` muestra tasas de poda por región del parámetro. No cuentan como objetivos
completos exactos en la carrera de seeds, el ajuste del surrogate ni las estadísticas marginales
del objective: hacerlo exageraría un presupuesto inacabado. Un Trial solo podado aplica en cambio
una penalización suave de vecindad a propuestas posteriores, conservando la señal negativa sin
inventar un score. La cabecera
separa Trials propuestos del presupuesto total, por lo que una combinación futura no aparece como
si el optimizador ya la hubiese decidido. Debajo de cada
tabla, paneles alineados muestran previews de parámetros del Trial y métricas del Run seleccionado;
Enter abre el detalle completo. Un Run enseña como máximo
cuatro curvas; `n`/`p` avanza o retrocede sus páginas numeradas sin símbolos dependientes de la
distribución del teclado. Debajo, arriba/abajo selecciona una época en una tabla compacta y
un punto rojo la localiza en cada curva aplicable; la mejor época permanece como diamante verde y
fila destacada. Enter/derecha abre todos sus escalares. La
duración avanza mientras corre el Run; los últimos tiempos medidos de época/validación aparecen al
llegar y, mientras tanto, el tiempo medio aproximado se etiqueta como tal. `o` alterna el panel
inferior entre métricas estructuradas y salida bruta aislada con refresco automático.
Si un Run falla, su excepción resumida permanece sobre las curvas y `e` abre un documento
desplazable con tipo, mensaje, fase, `result.json` persistido y traceback.
Pulsa `i` desde un estudio adaptativo o un Trial para abrir su consola viva de evidencia HPO;
izquierda/back vuelve a candidatos.

Un Work normal avanza a `Attempt 1`, `Attempt 2`, etc. y a su log completo; `a` abre esos Attempts
externos desde un estudio. Un estudio terminal que falló antes de publicar su índice abre esos
Attempts automáticamente, en vez de mostrar una pantalla de telemetría vacía. El log abierto se
actualiza automáticamente, sigue el final por defecto
y conserva el scroll manual. Mayús+izquierda/derecha desplaza horizontalmente líneas brutas largas
(`h`/`l` son alternativas); `e` muestra u oculta el traceback persistido e izquierda sin modificar
vuelve.
Los fallos añaden la excepción científica estructurada aunque `--tail` haya recortado las líneas;
`--verbose`/`--debug` incluye traceback y `--json` devuelve el fallo y ruta exacta. El bootstrap
humano muestra fases y latidos por stderr sin contaminar JSON. Los IDs siguen disponibles en
`lf jobs` y `lf overview --json`. `d` elimina el Work/Attempt terminal
seleccionado tras confirmación y `D` limpia todo el historial terminal, conservando siempre los Jobs
activos. Se eliminan únicamente sus workspaces y registros propios; datasets, caches y entornos
compartidos se preservan.

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
