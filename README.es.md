# LambdaForge

Español · [English](README.md)

LambdaForge es un runtime gestionado para investigación científica reproducible. El investigador
escribe Python normal en una clase `lambdaforge.Work`; el framework se ocupa de ejecución local o
remota, recursos, reintentos, métricas, artefactos, datasets, procedencia, resultados y limpieza.

## Instalación

El proyecto científico y LambdaForge son paquetes independientes instalados en el entorno del
proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge==0.12.0
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
  publica además una copia explícita fuera del almacenamiento gestionado.

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
debe usarse una ruta remota absoluta. La copia solo aparece tras un `run()` correcto, es atómica y
rechaza contenido distinto existente salvo `overwrite=True`. LambdaForge no confunde una ruta
remota con una local ni transfiere árboles grandes implícitamente al controlador.

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
python -m pip install "lambdaforge[clustering]==0.12.0"
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

`steps` expresa una secuencia y `{parallel: [...]}` un grupo paralelo aislado por procesos. `seeds`
crea Runs independientes; `search` expande variantes y `objective` selecciona la métrica escalar
exacta registrada por el Work, promediándola entre las seeds de cada variante. Los recursos YAML
son la reserva absoluta.

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
lf clean                    # vista previa de caché reconstruible
```

En `lf top`, arriba/abajo recorren clústeres y Works como una sola lista sin mostrar IDs operativos
largos en la ruta principal. Enter o derecha avanza de Work a `Attempt 1`, `Attempt 2`, etc. y de un
Attempt a sus logs; el detalle de clúster usa las mismas etiquetas y la izquierda vuelve. El log
abierto se actualiza automáticamente, sigue el final por defecto y conserva el scroll manual.
Los fallos añaden la excepción científica estructurada aunque `--tail` haya recortado las líneas;
`--verbose`/`--debug` incluye traceback y `--json` devuelve el fallo y ruta exacta. El bootstrap
humano muestra fases y latidos por stderr sin contaminar JSON. Los IDs siguen disponibles en
`lf jobs` y `lf overview --json`. `d` elimina el Work/Attempt terminal
seleccionado tras confirmación y `D` limpia todo el historial terminal, conservando siempre los Jobs
activos. Se eliminan únicamente sus workspaces y registros propios; datasets, caches y entornos
compartidos se preservan.

La creación de datasets se realiza con `self.outputs.dataset(...)`; `lf datasets` inspecciona,
verifica, materializa o elimina versiones inmutables. La guía completa de YAML, estudios, clústeres,
metadata y seguridad está en [el manual](docs/MANUAL.es.md) ([English](docs/MANUAL.md)). Para agentes,
[AGENTS.es.md](AGENTS.es.md)
es el contrato compacto que evita recorrer todo el repositorio e inventar APIs.
